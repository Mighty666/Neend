"""
tests for model architectures

these tests mostly check that shapes are correct and
models can do forward/backward pass without crashing.
actual performance testing needs real data
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.foundation_model import (
    FoundationModelConfig,
    SleepAudioFoundation,
    PatchEmbed,
    MultiTaskHead,
    create_foundation_model
)
from src.research.ssl_pretraining import (
    SSLConfig,
    Wav2Vec2Model,
    MaskedSpectrogramModel,
    BYOLA
)


class TestFoundationModel:
    """tests for the main foundation model"""

    def setup_method(self):
        # smaller config for faster tests
        self.config = FoundationModelConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            ff_dim=512,
            n_mels=64,
            max_length=512,
            num_classes=4
        )
        self.model = SleepAudioFoundation(self.config)
        self.model.eval()  # no dropout for deterministic tests

    def test_model_creation(self):
        """test that model creates without errors"""
        assert self.model is not None

        # check parameter count is reasonable
        n_params = sum(p.numel() for p in self.model.parameters())
        assert n_params > 1e6, "model too small"
        assert n_params < 1e9, "model too big for test config"

    def test_forward_classify(self):
        """test classification forward pass"""
        batch_size = 2
        time_steps = 256  # must be divisible by patch_size

        x = torch.randn(batch_size, 1, self.config.n_mels, time_steps)

        with torch.no_grad():
            logits = self.model.forward_classify(x)

        # output shape should be (batch, num_classes)
        assert logits.shape == (batch_size, self.config.num_classes)

        # check no nans
        assert not torch.isnan(logits).any()

    def test_forward_pretrain(self):
        """test pretraining forward pass"""
        batch_size = 2
        time_steps = 256

        x = torch.randn(batch_size, 1, self.config.n_mels, time_steps)

        output = self.model.forward_pretrain(x)

        # should have loss
        assert 'loss' in output
        assert output['loss'].shape == ()  # scalar

        # should have mask
        assert 'mask' in output

        # loss should be finite
        assert torch.isfinite(output['loss'])

    def test_backward_pass(self):
        """test that gradients flow"""
        self.model.train()

        batch_size = 2
        time_steps = 256

        x = torch.randn(batch_size, 1, self.config.n_mels, time_steps)

        output = self.model.forward_pretrain(x)
        loss = output['loss']

        loss.backward()

        # check gradients exist
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"no gradient for {name}"

    def test_feature_extraction(self):
        """test extracting features without classification"""
        batch_size = 2
        time_steps = 256

        x = torch.randn(batch_size, 1, self.config.n_mels, time_steps)

        with torch.no_grad():
            features = self.model.forward_classify(x, return_features=True)

        # should be (batch, hidden_dim)
        assert features.shape == (batch_size, self.config.hidden_dim)

    def test_different_input_lengths(self):
        """test model handles different input lengths"""
        batch_size = 2

        for time_steps in [128, 256, 512]:
            x = torch.randn(batch_size, 1, self.config.n_mels, time_steps)

            with torch.no_grad():
                logits = self.model(x)

            assert logits.shape == (batch_size, self.config.num_classes)


class TestPatchEmbed:
    """tests for patch embedding layer"""

    def test_output_shape(self):
        config = FoundationModelConfig(
            hidden_dim=256,
            n_mels=64,
            patch_size=16
        )
        embed = PatchEmbed(config)

        batch_size = 2
        time_steps = 128  # should give 128/16 = 8 patches

        x = torch.randn(batch_size, 1, 64, time_steps)

        with torch.no_grad():
            out = embed(x)

        expected_patches = time_steps // config.patch_size
        assert out.shape == (batch_size, expected_patches, config.hidden_dim)


class TestMultiTaskHead:
    """tests for multi-task prediction head"""

    def test_all_outputs(self):
        config = FoundationModelConfig(hidden_dim=256)
        head = MultiTaskHead(config)

        batch_size = 4
        features = torch.randn(batch_size, config.hidden_dim)

        outputs = head(features)

        # check all tasks present
        assert 'sleep_stage' in outputs
        assert 'event' in outputs
        assert 'quality' in outputs
        assert 'ahi' in outputs

        # check shapes
        assert outputs['sleep_stage'].shape == (batch_size, 5)
        assert outputs['event'].shape == (batch_size, 4)
        assert outputs['quality'].shape == (batch_size,)
        assert outputs['ahi'].shape == (batch_size,)


class TestSSLModels:
    """tests for self-supervised learning models"""

    def test_wav2vec2(self):
        """test wav2vec2 model"""
        config = SSLConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            mask_prob=0.065,
            num_negatives=10
        )
        model = Wav2Vec2Model(config)
        model.eval()

        # raw waveform input
        batch_size = 2
        samples = 16000  # 1 second at 16khz

        x = torch.randn(batch_size, samples)

        with torch.no_grad():
            output = model(x)

        assert 'loss' in output
        assert torch.isfinite(output['loss'])

    def test_masked_spectrogram(self):
        """test masked spectrogram model"""
        config = SSLConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            input_dim=64,  # n_mels
            mask_prob=0.3
        )
        model = MaskedSpectrogramModel(config)
        model.eval()

        batch_size = 2
        n_mels = 64
        time_frames = 128

        x = torch.randn(batch_size, 1, n_mels, time_frames)

        with torch.no_grad():
            output = model(x)

        assert 'loss' in output
        assert torch.isfinite(output['loss'])

    def test_byol_a(self):
        """test byol for audio"""
        config = SSLConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4
        )
        model = BYOLA(config)
        model.train()  # byol needs to be in train mode for momentum update

        batch_size = 4

        # two views of same data (simulated augmentation)
        view1 = torch.randn(batch_size, 1, 64, 128)
        view2 = torch.randn(batch_size, 1, 64, 128)

        output = model(view1, view2)

        assert 'loss' in output
        assert torch.isfinite(output['loss'])


class TestCreateModel:
    """tests for model creation helper"""

    def test_default_config(self):
        """test creating model with default config"""
        model = create_foundation_model()

        assert model is not None
        assert isinstance(model, SleepAudioFoundation)

    def test_custom_config(self):
        """test creating model with custom config"""
        config = FoundationModelConfig(
            hidden_dim=512,
            num_layers=8
        )
        model = create_foundation_model(config=config)

        assert model.config.hidden_dim == 512
        assert model.config.num_layers == 8


class TestGPUCompatibility:
    """tests for gpu compatibility (skip if no gpu)"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
    def test_gpu_forward(self):
        """test model runs on gpu"""
        config = FoundationModelConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4
        )
        model = SleepAudioFoundation(config).cuda()

        x = torch.randn(2, 1, 128, 256).cuda()

        with torch.no_grad():
            logits = model(x)

        assert logits.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
    def test_mixed_precision(self):
        """test model with mixed precision"""
        config = FoundationModelConfig(
            hidden_dim=256,
            num_layers=4
        )
        model = SleepAudioFoundation(config).cuda()

        x = torch.randn(2, 1, 128, 256).cuda()

        with torch.cuda.amp.autocast():
            logits = model(x)

        assert logits.dtype in [torch.float16, torch.bfloat16, torch.float32]


# run with: pytest tests/test_models.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
