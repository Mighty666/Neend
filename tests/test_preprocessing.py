"""
tests for preprocessing module

wrote these after i broke preprocessing for the third time by
"optimizing" something. now i run tests before committing

some of these tests use random data which isnt ideal but
getting real test audio is hard
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# add parent dir to path so we can import our code
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.distributed_preprocessing import (
    DistributedPreprocessor,
    PreprocessingConfig,
    LabelFusionModule,
    SyntheticAugmentation
)


class TestDistributedPreprocessor:
    """tests for the main preprocessor class"""

    def setup_method(self):
        """run before each test"""
        self.config = PreprocessingConfig(
            sample_rates=[16000],  # just one for faster tests
            n_mels_variants=[64],
            distributed_backend='ray'
        )
        self.preprocessor = DistributedPreprocessor(self.config)

    def test_snr_calculation(self):
        """test that snr calculation returns reasonable values"""
        # create test signal with known snr
        sr = 16000
        duration = 5
        t = np.linspace(0, duration, sr * duration)

        # clean sine wave
        signal = np.sin(2 * np.pi * 440 * t)

        # add noise
        noise = np.random.randn(len(signal)) * 0.1
        noisy = signal + noise

        snr = self.preprocessor.compute_snr(noisy, sr)

        # snr should be positive and reasonable
        assert snr > 0, "snr should be positive"
        assert snr < 100, "snr shouldnt be crazy high"
        # expected snr around 20db for this noise level
        assert 10 < snr < 40, f"snr {snr} out of expected range"

    def test_spectral_subtraction(self):
        """test denoising doesnt completely break the signal"""
        sr = 16000
        duration = 2

        # create noisy signal
        audio = np.random.randn(sr * duration) * 0.5
        audio[:sr] += np.sin(2 * np.pi * 100 * np.arange(sr) / sr)  # add tone

        denoised = self.preprocessor.spectral_subtraction_denoise(audio, sr)

        # output should be same length
        assert len(denoised) == len(audio), "length changed after denoising"

        # output shouldnt be all zeros
        assert np.std(denoised) > 0.01, "denoised signal is too quiet"

        # output shouldnt be louder than input (usually)
        assert np.max(np.abs(denoised)) < np.max(np.abs(audio)) * 2

    def test_mel_spectrogram_shape(self):
        """test that mel spectrogram has expected shape"""
        sr = 16000
        duration = 3
        audio = np.random.randn(sr * duration)

        mel = self.preprocessor.extract_mel_spectrogram(audio, sr, n_mels=64)

        # should be 2d
        assert len(mel.shape) == 2, "mel should be 2d"

        # first dim should be n_mels
        assert mel.shape[0] == 64, f"expected 64 mels, got {mel.shape[0]}"

        # second dim depends on hop length
        expected_frames = 1 + (len(audio) - self.config.n_fft) // self.config.hop_length
        assert abs(mel.shape[1] - expected_frames) < 5, "unexpected number of frames"

    def test_mfcc_features(self):
        """test mfcc extraction"""
        sr = 16000
        duration = 2
        audio = np.random.randn(sr * duration)

        mfcc_dict = self.preprocessor.extract_mfcc_features(audio, sr)

        # should have all expected keys
        assert 'mfcc' in mfcc_dict
        assert 'delta' in mfcc_dict
        assert 'delta2' in mfcc_dict
        assert 'mfcc_full' in mfcc_dict

        # mfcc should have n_mfcc coefficients
        assert mfcc_dict['mfcc'].shape[0] == self.config.n_mfcc

        # full should be 3x mfcc
        assert mfcc_dict['mfcc_full'].shape[0] == 3 * self.config.n_mfcc

    def test_spectral_features(self):
        """test spectral feature extraction"""
        sr = 16000
        duration = 2
        audio = np.random.randn(sr * duration)

        features = self.preprocessor.extract_spectral_features(audio, sr)

        # check all features present
        expected_features = [
            'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
            'spectral_flatness', 'zero_crossing_rate', 'rms_energy', 'crest_factor'
        ]
        for feat in expected_features:
            assert feat in features, f"missing feature: {feat}"

        # crest factor should be scalar
        assert np.isscalar(features['crest_factor'])

    def test_advanced_features(self):
        """test advanced feature extraction"""
        sr = 16000
        duration = 2
        audio = np.random.randn(sr * duration)

        features = self.preprocessor.extract_advanced_features(audio, sr)

        # check key features
        assert 'teager_energy' in features
        assert 'hilbert_envelope_mean' in features
        assert 'pitch_mean' in features
        assert 'harmonicity' in features

        # teager energy should be positive
        assert features['teager_energy'] >= 0

    def test_jitter_shimmer(self):
        """test voice quality feature extraction"""
        sr = 16000
        duration = 2

        # create periodic signal (like voiced speech)
        t = np.linspace(0, duration, sr * duration)
        audio = np.sin(2 * np.pi * 150 * t)  # 150 hz fundamental
        audio += 0.1 * np.random.randn(len(audio))  # small noise

        features = self.preprocessor.extract_jitter_shimmer(audio, sr)

        assert 'jitter_local' in features
        assert 'shimmer_local' in features

        # jitter should be small for periodic signal
        # (might be larger due to noise but shouldnt be huge)
        assert features['jitter_local'] < 0.5


class TestLabelFusion:
    """tests for dawid-skene label fusion"""

    def test_perfect_agreement(self):
        """test with annotators that perfectly agree"""
        fusion = LabelFusionModule(n_classes=3)

        # 10 items, 3 annotators, all agree
        annotations = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [0, 0, 0],
            [1, 1, 1],
        ])

        true_labels, _ = fusion.dawid_skene(annotations)

        # should recover the true labels
        expected = np.array([0, 1, 2, 0, 1])
        np.testing.assert_array_equal(true_labels, expected)

    def test_missing_annotations(self):
        """test handling of missing annotations (-1)"""
        fusion = LabelFusionModule(n_classes=2)

        # some annotators didnt label everything
        annotations = np.array([
            [0, 0, -1],
            [1, -1, 1],
            [-1, 0, 0],
        ])

        true_labels, annotator_accuracy = fusion.dawid_skene(annotations)

        # should still produce valid labels
        assert len(true_labels) == 3
        assert all(l in [0, 1] for l in true_labels)

    def test_noisy_annotator(self):
        """test that algorithm identifies unreliable annotator"""
        fusion = LabelFusionModule(n_classes=2)

        # annotator 2 is random, others are good
        n_items = 100
        true = np.random.randint(0, 2, n_items)

        annotations = np.column_stack([
            true,  # annotator 0: perfect
            true,  # annotator 1: perfect
            np.random.randint(0, 2, n_items)  # annotator 2: random
        ])

        est_labels, annotator_accuracy = fusion.dawid_skene(annotations, n_iter=50)

        # accuracy of annotator 2 should be lower
        acc_0 = annotator_accuracy[0].diagonal().mean()
        acc_2 = annotator_accuracy[2].diagonal().mean()

        assert acc_0 > acc_2, "should identify unreliable annotator"


class TestSyntheticAugmentation:
    """tests for data augmentation"""

    def setup_method(self):
        self.aug = SyntheticAugmentation(sr=16000)
        self.duration = 3
        self.audio = np.random.randn(self.aug.sr * self.duration)

    def test_time_stretch(self):
        """test time stretching"""
        # stretch by 1.1x (10% longer)
        stretched = self.aug.time_stretch(self.audio, rate=1.1)

        # output should be shorter (faster playback)
        assert len(stretched) < len(self.audio)

    def test_pitch_shift(self):
        """test pitch shifting preserves length"""
        shifted = self.aug.pitch_shift(self.audio, n_steps=2)

        # length should be same
        assert len(shifted) == len(self.audio)

    def test_add_noise(self):
        """test noise addition at different snrs"""
        for snr in [10, 20, 30]:
            noisy = self.aug.add_background_noise(self.audio, 'white', snr)

            assert len(noisy) == len(self.audio)
            # output should be different from input
            assert not np.allclose(noisy, self.audio)

    def test_different_noise_types(self):
        """test different noise colors"""
        for noise_type in ['white', 'pink', 'brown']:
            noisy = self.aug.add_background_noise(self.audio, noise_type, 20)
            assert len(noisy) == len(self.audio)

    def test_reverb(self):
        """test reverb effect"""
        reverbed = self.aug.add_reverb(self.audio, room_size=0.5)

        assert len(reverbed) == len(self.audio)
        # should be different from input
        assert not np.allclose(reverbed, self.audio)

    def test_simulated_apnea(self):
        """test apnea simulation creates quiet region"""
        # create signal with some energy
        audio = np.sin(2 * np.pi * 100 * np.arange(len(self.audio)) / self.aug.sr)

        apnea_start = 1.0  # seconds
        apnea_duration = 1.0

        augmented = self.aug.simulate_apnea_event(audio, apnea_start, apnea_duration)

        # check that apnea region is quiet
        start_sample = int(apnea_start * self.aug.sr)
        end_sample = int((apnea_start + apnea_duration) * self.aug.sr)

        apnea_energy = np.mean(augmented[start_sample:end_sample]**2)
        normal_energy = np.mean(audio[:start_sample]**2)

        assert apnea_energy < normal_energy * 0.01, "apnea region should be quiet"

    def test_batch_augmentation(self):
        """test generating multiple augmentations"""
        n_aug = 5
        augmented = self.aug.augment_batch(self.audio, n_aug)

        assert len(augmented) == n_aug

        # all augmentations should be different
        for i in range(n_aug):
            for j in range(i+1, n_aug):
                assert not np.allclose(augmented[i], augmented[j])


class TestEdgeCases:
    """tests for edge cases and error handling"""

    def test_very_short_audio(self):
        """test handling of very short audio"""
        config = PreprocessingConfig(sample_rates=[16000])
        preprocessor = DistributedPreprocessor(config)

        # 0.1 second audio
        short_audio = np.random.randn(1600)

        # should still work without crashing
        snr = preprocessor.compute_snr(short_audio, 16000)
        assert np.isfinite(snr)

    def test_silent_audio(self):
        """test handling of silent audio"""
        config = PreprocessingConfig(sample_rates=[16000])
        preprocessor = DistributedPreprocessor(config)

        # nearly silent
        silent = np.random.randn(16000) * 1e-10

        # should handle without errors
        snr = preprocessor.compute_snr(silent, 16000)
        assert np.isfinite(snr)

    def test_clipped_audio(self):
        """test handling of clipped/distorted audio"""
        config = PreprocessingConfig(sample_rates=[16000])
        preprocessor = DistributedPreprocessor(config)

        # clipped audio
        audio = np.clip(np.random.randn(32000) * 5, -1, 1)

        features = preprocessor.extract_spectral_features(audio, 16000)

        # should still produce valid features
        assert all(np.isfinite(v).all() if isinstance(v, np.ndarray) else np.isfinite(v)
                   for v in features.values())


# run with: pytest tests/test_preprocessing.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
