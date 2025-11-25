"""
foundation model for sleep audio - this is the big one

started with a smaller model (6 layers) but kept scaling up when i saw
improvement. 24 layers with 1024 hidden gives about 312m parameters
which is the most i could fit on my gpu

the architecture is pretty standard transformer but i added:
- rotary position embeddings (better than sinusoidal apparently)
- swiglu activation (from llama)
- pre-norm (easier to train deep models)

pretraining uses masked modeling like mae - mask 75% of patches and
predict them back. this teaches the model about audio structure without
needing labels

training was painful - took like a week on a v100. had to use gradient
accumulation and mixed precision to fit in memory

update: added domain shift metrics for cross-dataset evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class FoundationModelConfig:
    """
    config for the foundation model

    tried a lot of different settings - these are what worked best
    bigger model = better results but slower training

    312m params is around gpt-2 small size
    """
    # architecture
    hidden_dim: int = 1024  # tried 768 first but 1024 was better
    num_layers: int = 24  # 12 was too shallow, 24 seems good
    num_heads: int = 16
    ff_dim: int = 4096  # 4x hidden is standard
    dropout: float = 0.1

    # input handling
    patch_size: int = 16  # smaller = more patches = slower
    n_mels: int = 128
    max_length: int = 2048  # ~60 seconds of audio

    # pretraining
    mask_ratio: float = 0.75  # mae uses 75%, seems to work well
    temperature: float = 0.1

    # fine-tuning
    num_classes: int = 4  # normal, snoring, hypopnea, apnea
    pool_type: str = "cls"


class PatchEmbed(nn.Module):
    """convert spectrogram to patches"""

    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.n_mels = config.n_mels

        # single conv to turn spectrogram into patch embeddings
        self.proj = nn.Conv2d(
            1, config.hidden_dim,
            kernel_size=(config.n_mels, config.patch_size),
            stride=(config.n_mels, config.patch_size)
        )

        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_mels, time)
        x = self.proj(x)
        x = x.squeeze(2).transpose(1, 2)  # (batch, n_patches, hidden)
        x = self.norm(x)
        return x


class RotaryPositionalEmbedding(nn.Module):
    """
    rotary position embeddings - better than learned or sinusoidal

    the idea is to encode position into the angle of complex numbers,
    then multiply queries and keys by rotation. this makes attention
    naturally depend on relative position

    took me a while to understand but it works really well
    """

    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        # frequency for each dimension
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # precompute cos and sin for all positions
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[2]
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )


def rotate_half(x):
    """helper for rope - rotates half the dims"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """apply rotary embeddings to q and k"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """standard multi-head attention with rope"""

    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.rotary = RotaryPositionalEmbedding(self.head_dim, config.max_length)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # project to q, k, v and reshape for multi-head
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # apply rotary embeddings
        cos, sin = self.rotary(x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # weighted sum
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)

        return out


class FeedForward(nn.Module):
    """
    feed-forward with swiglu activation

    swiglu is from the llama paper - its silu(w1*x) * w3*x
    supposedly better than relu or gelu for some reason
    """

    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_dim, config.ff_dim)
        self.w2 = nn.Linear(config.ff_dim, config.hidden_dim)
        self.w3 = nn.Linear(config.hidden_dim, config.ff_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """single transformer layer with pre-norm"""

    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        # pre-norm instead of post-norm (easier to train deep models)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = MultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class SleepAudioFoundation(nn.Module):
    """
    the main foundation model

    this is my biggest model - 312m parameters. architecture is based on
    vit/mae but adapted for audio spectrograms

    i use mae-style pretraining where you mask 75% of patches and train
    to reconstruct them. this works surprisingly well - the model learns
    about audio structure without any labels

    then fine-tune on labeled data for apnea detection

    one thing i learned: bigger model = better results but you need
    enough data and compute. with my ~1000 hours of audio, 312m params
    seems to be around the limit before overfitting
    """

    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        self.config = config

        # turn spectrogram into patches
        self.patch_embed = PatchEmbed(config)

        # cls token for classification (like bert)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # mask token for mae pretraining
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # the main transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_dim)

        # decoder for pretraining - much smaller than encoder
        # only need enough capacity to reconstruct patches
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.n_mels * config.patch_size)
        )

        # classification head for fine-tuning
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )

        # init weights
        self.apply(self._init_weights)

        # count parameters
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"model has {n_params/1e6:.1f}m parameters")

    def _init_weights(self, module):
        """weight initialization - standard stuff"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        random masking for mae pretraining

        the key insight from mae is that you can mask a LOT (75%) and
        still learn useful representations. this makes training faster
        bc you only process 25% of tokens
        """
        batch_size, seq_len, dim = x.shape
        num_keep = int(seq_len * (1 - mask_ratio))

        # random shuffle
        noise = torch.rand(batch_size, seq_len, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep top tokens by random score
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dim))

        # binary mask: 0 is keep, 1 is remove
        mask = torch.ones(batch_size, seq_len, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """encode with optional masking"""
        x = self.patch_embed(x)

        # add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # mask for pretraining
        if mask_ratio > 0:
            x_patches = x[:, 1:]  # dont mask cls
            x_masked, mask, ids_restore = self.random_masking(x_patches, mask_ratio)
            x = torch.cat([x[:, :1], x_masked], dim=1)
        else:
            mask = None
            ids_restore = None

        # transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """decode for reconstruction"""
        # add mask tokens back
        mask_tokens = self.mask_token.expand(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1)
        x_full = torch.cat([x[:, 1:], mask_tokens], dim=1)

        # unshuffle to original order
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        # predict patches
        pred = self.decoder(x_full)

        return pred

    def forward_pretrain(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """pretraining forward pass with masking and reconstruction"""
        # encode with masking
        latent, mask, ids_restore = self.forward_encoder(x, self.config.mask_ratio)

        # decode
        pred = self.forward_decoder(latent, ids_restore)

        # target is original patches
        target = self.patch_embed.proj(x).squeeze(2).transpose(1, 2)

        # mse loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return {
            'loss': loss,
            'pred': pred,
            'mask': mask,
            'latent': latent
        }

    def forward_classify(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """classification forward pass"""
        # encode without masking
        latent, _, _ = self.forward_encoder(x, mask_ratio=0.0)

        # pool - cls token is simplest
        if self.config.pool_type == "cls":
            pooled = latent[:, 0]
        elif self.config.pool_type == "mean":
            pooled = latent[:, 1:].mean(dim=1)
        else:
            # attention pooling - weighted sum
            attn_weights = F.softmax(latent[:, 1:].mean(dim=-1), dim=1)
            pooled = (latent[:, 1:] * attn_weights.unsqueeze(-1)).sum(dim=1)

        if return_features:
            return pooled

        logits = self.classifier(pooled)
        return logits

    def forward(self, x: torch.Tensor, pretrain: bool = False) -> torch.Tensor:
        if pretrain:
            return self.forward_pretrain(x)
        else:
            return self.forward_classify(x)


class MultiTaskHead(nn.Module):
    """
    multi-task head for fine-tuning

    instead of just apnea detection, predict multiple things:
    - sleep stage (5 classes)
    - respiratory event (4 classes)
    - sleep quality (0-100)
    - ahi score (continuous)

    multi-task learning usually helps by regularizing the model
    """

    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        hidden = config.hidden_dim // 2

        self.sleep_stage = nn.Sequential(
            nn.Linear(config.hidden_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 5)
        )

        self.event_detection = nn.Sequential(
            nn.Linear(config.hidden_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4)
        )

        self.quality_regression = nn.Sequential(
            nn.Linear(config.hidden_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )

        self.ahi_prediction = nn.Sequential(
            nn.Linear(config.hidden_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'sleep_stage': self.sleep_stage(features),
            'event': self.event_detection(features),
            'quality': self.quality_regression(features).squeeze(-1),
            'ahi': self.ahi_prediction(features).squeeze(-1)
        }


class DomainShiftMetrics:
    """
    metrics for measuring domain shift between datasets

    this is important for medical ai bc models often fail when tested
    on new hospitals/populations. mmd and classifier accuracy tell you
    how different two datasets are in feature space
    """

    @staticmethod
    def maximum_mean_discrepancy(
        source_features: np.ndarray,
        target_features: np.ndarray,
        kernel: str = 'rbf',
        gamma: float = 1.0
    ) -> float:
        """
        mmd - classic domain adaptation metric

        basically measures difference in feature distributions using
        kernel trick. lower = more similar
        """
        n_source = source_features.shape[0]
        n_target = target_features.shape[0]

        if kernel == 'rbf':
            def kernel_func(x, y):
                diff = x[:, None, :] - y[None, :, :]
                return np.exp(-gamma * np.sum(diff ** 2, axis=-1))
        else:
            def kernel_func(x, y):
                return np.dot(x, y.T)

        K_ss = kernel_func(source_features, source_features)
        K_tt = kernel_func(target_features, target_features)
        K_st = kernel_func(source_features, target_features)

        mmd = (np.sum(K_ss) / (n_source ** 2) +
               np.sum(K_tt) / (n_target ** 2) -
               2 * np.sum(K_st) / (n_source * n_target))

        return float(np.sqrt(max(0, mmd)))

    @staticmethod
    def covariate_shift_ratio(
        source_features: np.ndarray,
        target_features: np.ndarray
    ) -> float:
        """
        domain classifier accuracy

        if you can easily tell which dataset a sample is from,
        thats bad - means theyre very different
        0.5 = same, 1.0 = completely different
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        n_source = source_features.shape[0]
        n_target = target_features.shape[0]

        X = np.vstack([source_features, target_features])
        y = np.array([0] * n_source + [1] * n_target)

        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

        return float(np.mean(scores))


def create_foundation_model(
    pretrained_path: Optional[str] = None,
    config: Optional[FoundationModelConfig] = None
) -> SleepAudioFoundation:
    """helper function to create model"""

    if config is None:
        config = FoundationModelConfig()

    model = SleepAudioFoundation(config)

    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f"loaded weights from {pretrained_path}")

    return model
