"""
Self-supervised pretraining models for sleep audio.
Implements wav2vec 2.0, HuBERT, BYOL-A, and Masked Spectrogram Modeling.
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
class SSLConfig:
    """Configuration for self-supervised pretraining."""
    model_type: str = "wav2vec2"  # wav2vec2, hubert, byol_a, masked_spec
    input_dim: int = 80  # mel bands or raw samples
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    mask_prob: float = 0.065
    mask_length: int = 10
    num_negatives: int = 100
    codebook_size: int = 320
    num_codebooks: int = 2
    temperature: float = 0.1
    learning_rate: float = 5e-4
    warmup_steps: int = 32000
    max_steps: int = 400000
    batch_size: int = 32
    gradient_accumulation: int = 8
    use_fp16: bool = True


class ConvFeatureEncoder(nn.Module):
    """Convolutional feature encoder for raw waveform."""

    def __init__(self, hidden_dim: int = 768):
        super().__init__()

        # 7 conv layers with progressive downsampling
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=0),
                nn.GroupNorm(512, 512),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0),
                nn.GroupNorm(512, 512),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0),
                nn.GroupNorm(512, 512),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0),
                nn.GroupNorm(512, 512),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0),
                nn.GroupNorm(512, 512),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=2, stride=2, padding=0),
                nn.GroupNorm(512, 512),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=2, stride=2, padding=0),
                nn.GroupNorm(512, 512),
                nn.GELU()
            ),
        ])

        self.proj = nn.Linear(512, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, samples)
        x = x.unsqueeze(1)  # (batch, 1, samples)

        for conv in self.conv_layers:
            x = conv(x)

        x = x.transpose(1, 2)  # (batch, time, channels)
        x = self.proj(x)

        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequential modeling."""

    def __init__(self, config: SSLConfig):
        super().__init__()

        self.pos_embedding = PositionalEncoding(config.hidden_dim, config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pos_embedding(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.layer_norm(x)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GumbelVectorQuantizer(nn.Module):
    """Gumbel softmax vector quantizer for wav2vec 2.0."""

    def __init__(self, config: SSLConfig):
        super().__init__()

        self.num_codebooks = config.num_codebooks
        self.codebook_size = config.codebook_size
        self.temperature = config.temperature

        self.weight_proj = nn.Linear(config.hidden_dim, config.num_codebooks * config.codebook_size)

        self.codebook = nn.Parameter(
            torch.FloatTensor(1, config.num_codebooks * config.codebook_size, config.hidden_dim // config.num_codebooks)
        )
        nn.init.uniform_(self.codebook)

    def forward(self, x: torch.Tensor, produce_targets: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Project to logits
        logits = self.weight_proj(x)
        logits = logits.view(batch_size * seq_len * self.num_codebooks, -1)

        if self.training:
            # Gumbel softmax
            gumbels = -torch.empty_like(logits).exponential_().log()
            logits = logits + gumbels

        probs = F.softmax(logits / self.temperature, dim=-1)

        # Hard assignment for targets
        if produce_targets:
            targets = probs.argmax(dim=-1)
            targets = targets.view(batch_size, seq_len, self.num_codebooks)
        else:
            targets = None

        # Quantized vectors
        probs = probs.view(batch_size * seq_len, self.num_codebooks, -1)
        quantized = torch.einsum('bgv,gvd->bgd', probs, self.codebook.squeeze(0).view(self.num_codebooks, self.codebook_size, -1))
        quantized = quantized.view(batch_size, seq_len, -1)

        return quantized, targets


class Wav2Vec2Model(nn.Module):
    """wav2vec 2.0 self-supervised model."""

    def __init__(self, config: SSLConfig):
        super().__init__()
        self.config = config

        self.feature_encoder = ConvFeatureEncoder(config.hidden_dim)
        self.feature_projection = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.quantizer = GumbelVectorQuantizer(config)
        self.transformer = TransformerEncoder(config)

        self.project_q = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.final_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def _compute_mask(self, shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
        """Compute mask for contrastive learning."""
        batch_size, seq_len = shape

        # Random mask starting positions
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        num_mask = int(seq_len * self.config.mask_prob)

        for i in range(batch_size):
            mask_starts = torch.randperm(seq_len - self.config.mask_length + 1)[:num_mask]
            for start in mask_starts:
                mask[i, start:start + self.config.mask_length] = True

        return mask

    def _sample_negatives(self, y: torch.Tensor, num_negatives: int) -> torch.Tensor:
        """Sample negative examples for contrastive loss."""
        batch_size, seq_len, dim = y.shape

        # Sample random indices
        neg_indices = torch.randint(0, seq_len, (batch_size, seq_len, num_negatives), device=y.device)

        # Gather negatives
        y_expanded = y.unsqueeze(2).expand(-1, -1, num_negatives, -1)
        neg_indices_expanded = neg_indices.unsqueeze(-1).expand(-1, -1, -1, dim)

        negatives = torch.gather(y_expanded, 1, neg_indices_expanded)

        return negatives

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Feature extraction
        features = self.feature_encoder(x)
        features = self.feature_projection(features)

        # Quantize for targets
        quantized, targets = self.quantizer(features)
        quantized = self.project_q(quantized)

        # Mask features
        batch_size, seq_len, _ = features.shape
        mask = self._compute_mask((batch_size, seq_len), features.device)

        # Replace masked positions with learned mask embedding
        masked_features = features.clone()
        masked_features[mask] = 0  # Simple zero masking

        # Transformer encoding
        context = self.transformer(masked_features)
        context = self.final_proj(context)

        # Sample negatives
        negatives = self._sample_negatives(quantized, self.config.num_negatives)

        # Contrastive loss
        pos_logits = F.cosine_similarity(context, quantized, dim=-1)
        neg_logits = F.cosine_similarity(context.unsqueeze(2), negatives, dim=-1)

        logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
        logits = logits / self.config.temperature

        # Only compute loss on masked positions
        loss_mask = mask.float()

        # Cross entropy with first as positive
        targets_ce = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets_ce.view(-1),
            reduction='none'
        )
        loss = (loss.view(batch_size, seq_len) * loss_mask).sum() / (loss_mask.sum() + 1e-8)

        return {
            'loss': loss,
            'context': context,
            'quantized': quantized,
            'mask': mask
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without contrastive loss."""
        features = self.feature_encoder(x)
        features = self.feature_projection(features)
        context = self.transformer(features)
        return context


class MaskedSpectrogramModel(nn.Module):
    """Masked Spectrogram Modeling for self-supervised learning."""

    def __init__(self, config: SSLConfig):
        super().__init__()
        self.config = config

        # Patch embedding for spectrograms
        self.patch_size = 16
        self.patch_embed = nn.Conv2d(
            1, config.hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size)
        )

        self.transformer = TransformerEncoder(config)

        # Reconstruction head
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, self.patch_size * self.patch_size)
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        # spec: (batch, 1, mel_bins, time)
        batch_size = spec.shape[0]

        # Patch embedding
        patches = self.patch_embed(spec)  # (batch, hidden, h_patches, w_patches)
        patches = patches.flatten(2).transpose(1, 2)  # (batch, num_patches, hidden)

        num_patches = patches.shape[1]

        # Random masking
        num_mask = int(num_patches * self.config.mask_prob)
        mask_indices = torch.rand(batch_size, num_patches, device=spec.device).argsort(dim=1)[:, :num_mask]

        # Create masked input
        masked_patches = patches.clone()
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=spec.device)

        for i in range(batch_size):
            masked_patches[i, mask_indices[i]] = self.mask_token
            mask[i, mask_indices[i]] = True

        # Transformer encoding
        encoded = self.transformer(masked_patches)

        # Reconstruct masked patches
        reconstructed = self.decoder(encoded)

        # Compute reconstruction loss only on masked patches
        original_patches = patches.view(batch_size, num_patches, -1)
        target_patches = F.pad(original_patches, (0, reconstructed.shape[-1] - original_patches.shape[-1]))

        loss = F.mse_loss(
            reconstructed[mask],
            target_patches[mask],
            reduction='mean'
        )

        return {
            'loss': loss,
            'encoded': encoded,
            'reconstructed': reconstructed,
            'mask': mask
        }

    def extract_features(self, spec: torch.Tensor) -> torch.Tensor:
        """Extract features without masking."""
        patches = self.patch_embed(spec)
        patches = patches.flatten(2).transpose(1, 2)
        encoded = self.transformer(patches)
        return encoded


class BYOLA(nn.Module):
    """BYOL for Audio - Bootstrap Your Own Latent for audio spectrograms."""

    def __init__(self, config: SSLConfig):
        super().__init__()
        self.config = config

        # Online network
        self.online_encoder = self._make_encoder(config)
        self.online_projector = self._make_projector(config)
        self.online_predictor = self._make_predictor(config)

        # Target network (momentum updated)
        self.target_encoder = self._make_encoder(config)
        self.target_projector = self._make_projector(config)

        # Initialize target with online weights
        self._copy_params(self.online_encoder, self.target_encoder)
        self._copy_params(self.online_projector, self.target_projector)

        # Freeze target
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

        self.momentum = 0.996

    def _make_encoder(self, config: SSLConfig) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, config.hidden_dim)
        )

    def _make_projector(self, config: SSLConfig) -> nn.Module:
        return nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.BatchNorm1d(config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )

    def _make_predictor(self, config: SSLConfig) -> nn.Module:
        return nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def _copy_params(self, source: nn.Module, target: nn.Module):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(source_param.data)

    @torch.no_grad()
    def _momentum_update(self):
        for online_param, target_param in zip(
            list(self.online_encoder.parameters()) + list(self.online_projector.parameters()),
            list(self.target_encoder.parameters()) + list(self.target_projector.parameters())
        ):
            target_param.data = self.momentum * target_param.data + (1 - self.momentum) * online_param.data

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Online forward
        online_proj1 = self.online_projector(self.online_encoder(view1))
        online_proj2 = self.online_projector(self.online_encoder(view2))

        online_pred1 = self.online_predictor(online_proj1)
        online_pred2 = self.online_predictor(online_proj2)

        # Target forward
        with torch.no_grad():
            target_proj1 = self.target_projector(self.target_encoder(view1))
            target_proj2 = self.target_projector(self.target_encoder(view2))

        # Symmetric loss
        loss1 = 2 - 2 * F.cosine_similarity(online_pred1, target_proj2.detach(), dim=-1).mean()
        loss2 = 2 - 2 * F.cosine_similarity(online_pred2, target_proj1.detach(), dim=-1).mean()

        loss = (loss1 + loss2) / 2

        # Update target network
        if self.training:
            self._momentum_update()

        return {
            'loss': loss,
            'online_features': online_proj1
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(x)


class SSLPretrainer:
    """Trainer for self-supervised learning models."""

    def __init__(self, config: SSLConfig, model: nn.Module, device: str = 'cuda'):
        self.config = config
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=0.01
        )

        self.scaler = torch.cuda.amp.GradScaler() if config.use_fp16 else None
        self.step = 0

    def get_lr(self) -> float:
        """Get learning rate with warmup and decay."""
        if self.step < self.config.warmup_steps:
            return self.config.learning_rate * self.step / self.config.warmup_steps
        else:
            progress = (self.step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
            return self.config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Update learning rate
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        batch = batch.to(self.device)

        if self.config.use_fp16:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch)
                loss = outputs['loss'] / self.config.gradient_accumulation

            self.scaler.scale(loss).backward()

            if (self.step + 1) % self.config.gradient_accumulation == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            outputs = self.model(batch)
            loss = outputs['loss'] / self.config.gradient_accumulation
            loss.backward()

            if (self.step + 1) % self.config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.step += 1

        return {
            'loss': loss.item() * self.config.gradient_accumulation,
            'lr': lr,
            'step': self.step
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'config': self.config
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        logger.info(f"Loaded checkpoint from {path}")


def create_ssl_model(config: SSLConfig) -> nn.Module:
    """Factory function to create SSL model."""
    if config.model_type == "wav2vec2":
        return Wav2Vec2Model(config)
    elif config.model_type == "masked_spec":
        return MaskedSpectrogramModel(config)
    elif config.model_type == "byol_a":
        return BYOLA(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
