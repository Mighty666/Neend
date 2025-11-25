"""
Tests for feature extraction module
"""

import pytest
import numpy as np
from src.audio import AudioPreprocessor, FeatureExtractor, FrequencyAnalyzer


class TestAudioPreprocessor:
    """Tests for AudioPreprocessor class"""

    def setup_method(self):
        self.preprocessor = AudioPreprocessor(sample_rate=16000)

    def test_normalize(self):
        """Test audio normalization"""
        audio = np.array([0.5, -0.3, 0.8, -0.1])
        normalized = self.preprocessor.normalize(audio)
        assert np.max(np.abs(normalized)) <= 1.0

    def test_pre_emphasis(self):
        """Test pre-emphasis filter"""
        audio = np.random.randn(1000)
        emphasized = self.preprocessor.pre_emphasis(audio)
        assert len(emphasized) == len(audio)

    def test_segment_audio(self):
        """Test audio segmentation"""
        # 2 minutes of audio at 16kHz
        audio = np.random.randn(16000 * 120)
        segments = self.preprocessor.segment_audio(audio, segment_duration=30.0)

        # Should get 4 segments of 30 seconds each
        assert len(segments) == 4
        for segment in segments:
            assert len(segment) == 16000 * 30

    def test_process_audio_segment(self):
        """Test complete preprocessing pipeline"""
        audio = np.random.randn(16000 * 30)
        processed = self.preprocessor.process_audio_segment(audio)
        assert len(processed) > 0


class TestFrequencyAnalyzer:
    """Tests for FrequencyAnalyzer class"""

    def setup_method(self):
        self.analyzer = FrequencyAnalyzer(sample_rate=16000)

    def test_compute_fft(self):
        """Test FFT computation"""
        # Create a simple sine wave at 100 Hz
        t = np.linspace(0, 1, 16000)
        audio = np.sin(2 * np.pi * 100 * t)

        freqs, magnitude = self.analyzer.compute_fft(audio)

        assert len(freqs) == len(magnitude)
        # Dominant frequency should be around 100 Hz
        dominant_freq = freqs[np.argmax(magnitude)]
        assert abs(dominant_freq - 100) < 5

    def test_detect_snoring_signature(self):
        """Test snoring detection"""
        # Create low-frequency signal (snoring-like)
        t = np.linspace(0, 1, 16000)
        snoring_signal = np.sin(2 * np.pi * 50 * t)

        freqs, magnitude = self.analyzer.compute_fft(snoring_signal)
        result = self.analyzer.detect_snoring_signature(freqs, magnitude)

        assert 'snore_score' in result
        assert 'is_snoring' in result
        assert 'dominant_freq' in result


class TestFeatureExtractor:
    """Tests for FeatureExtractor class"""

    def setup_method(self):
        self.extractor = FeatureExtractor(sample_rate=16000, n_mfcc=13)

    def test_extract_mfcc(self):
        """Test MFCC extraction"""
        audio = np.random.randn(16000 * 5)  # 5 seconds
        mfcc = self.extractor.extract_mfcc(audio)

        # Should have 39 features (13 MFCC + 13 delta + 13 delta-delta)
        assert mfcc.shape[0] == 39

    def test_extract_energy_features(self):
        """Test energy feature extraction"""
        audio = np.random.randn(16000 * 5)
        features = self.extractor.extract_energy_features(audio)

        assert 'rms_mean' in features
        assert 'rms_std' in features
        assert 'zcr_mean' in features

    def test_extract_all_features(self):
        """Test complete feature extraction"""
        audio = np.random.randn(16000 * 5)
        features = self.extractor.extract_all_features(audio)

        assert 'mfcc' in features
        assert 'spectral' in features
        assert 'snoring' in features
        assert 'energy' in features
        assert 'pitch' in features

    def test_extract_feature_matrix(self):
        """Test feature matrix for model input"""
        audio = np.random.randn(16000 * 30)  # 30 seconds
        matrix = self.extractor.extract_feature_matrix(audio, target_frames=128)

        assert matrix.shape == (128, 13, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
