"""
Dataset loaders and augmentation for sleep apnea detection
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Generator
import logging

logger = logging.getLogger(__name__)

class PhysioNetLoader:
    """Load and process PhysioNet Apnea-ECG and sleep datasets"""

    DATASETS = {
        'apnea-ecg': 'https://physionet.org/content/apnea-ecg/1.0.0/',
        'slpdb': 'https://physionet.org/content/slpdb/1.0.0/',
        'mitdb': 'https://physionet.org/content/mitdb/1.0.0/'
    }

    def __init__(self, data_dir: str = './data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_dataset(self, dataset_name: str = 'apnea-ecg'):
        """Download dataset from PhysioNet"""
        import urllib.request
        url = self.DATASETS.get(dataset_name)
        if not url:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        logger.info(f"Dataset URL: {url}")
        logger.info("Please download manually from PhysioNet due to size")

    def load_annotations(self, annotation_file: str) -> List[Dict]:
        """Load apnea annotations from file"""
        annotations = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    annotations.append({
                        'time': float(parts[0]),
                        'duration': float(parts[1]),
                        'type': parts[2]
                    })
        return annotations

    def generate_synthetic_data(self, n_samples: int = 1000, duration_hours: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic sleep apnea data for testing"""
        n_segments = int(duration_hours * 3600 / 30)
        features_list = []
        labels = []

        for _ in range(n_samples):
            sample_features = []
            sample_labels = []
            severity = np.random.choice(['normal', 'mild', 'moderate', 'severe'], p=[0.3, 0.3, 0.25, 0.15])

            if severity == 'normal':
                event_prob = [0.9, 0.08, 0.015, 0.005]
            elif severity == 'mild':
                event_prob = [0.7, 0.15, 0.1, 0.05]
            elif severity == 'moderate':
                event_prob = [0.5, 0.2, 0.2, 0.1]
            else:
                event_prob = [0.3, 0.2, 0.25, 0.25]

            for _ in range(n_segments):
                label = np.random.choice([0, 1, 2, 3], p=event_prob)
                features = self._generate_features_for_class(label)
                sample_features.append(features)
                sample_labels.append(label)

            features_list.append(sample_features)
            labels.append(sample_labels)

        return np.array(features_list), np.array(labels)

    def _generate_features_for_class(self, label: int) -> np.ndarray:
        """Generate synthetic features for a given class"""
        base_features = np.random.randn(39) * 0.1

        if label == 0:  # Normal
            base_features[0] += 0.5  # Higher energy
            base_features[13:26] += 0.3  # Normal spectral
        elif label == 1:  # Snoring
            base_features[0] += 0.8  # High energy
            base_features[1:5] += 1.0  # Low frequency emphasis
            base_features[26:] += 0.5  # Pitch variation
        elif label == 2:  # Hypopnea
            base_features[0] += 0.2  # Reduced energy
            base_features[13:26] -= 0.3  # Reduced spectral
        else:  # Apnea
            base_features[0] -= 0.5  # Very low energy
            base_features[:] *= 0.3  # Overall reduction

        return base_features

    def stream_data(self, features: np.ndarray, labels: np.ndarray) -> Generator:
        """Stream data sample by sample for online learning"""
        for sample_features, sample_labels in zip(features, labels):
            for feat, label in zip(sample_features, sample_labels):
                yield feat, label

class DataAugmenter:
    """Audio data augmentation for training"""

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

    def augment(self, audio: np.ndarray) -> List[np.ndarray]:
        """Apply multiple augmentations to audio"""
        augmented = [audio]
        augmented.append(self.time_stretch(audio, rate=0.9))
        augmented.append(self.time_stretch(audio, rate=1.1))
        augmented.append(self.pitch_shift(audio, steps=2))
        augmented.append(self.pitch_shift(audio, steps=-2))
        augmented.append(self.add_noise(audio, snr=20))
        augmented.append(self.add_noise(audio, snr=10))
        return augmented

    def time_stretch(self, audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """Time stretch without changing pitch"""
        import librosa
        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(self, audio: np.ndarray, steps: int = 0) -> np.ndarray:
        """Shift pitch by semitones"""
        import librosa
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=steps)

    def add_noise(self, audio: np.ndarray, snr: float = 20) -> np.ndarray:
        """Add Gaussian noise at specified SNR"""
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
        return audio + noise

    def random_crop(self, audio: np.ndarray, crop_length: int) -> np.ndarray:
        """Random crop of audio"""
        if len(audio) <= crop_length:
            return audio
        start = np.random.randint(0, len(audio) - crop_length)
        return audio[start:start + crop_length]

class FeatureNormalizer:
    """Online feature normalization"""

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.mean = np.zeros(n_features)
        self.var = np.ones(n_features)
        self.count = 0

    def partial_fit(self, features: np.ndarray):
        """Update statistics with new sample"""
        self.count += 1
        delta = features - self.mean
        self.mean += delta / self.count
        delta2 = features - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Normalize features"""
        std = np.sqrt(self.var + 1e-8)
        return (features - self.mean) / std

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Update and normalize"""
        self.partial_fit(features)
        return self.transform(features)
