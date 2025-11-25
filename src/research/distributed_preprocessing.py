"""
Distributed preprocessing pipeline using Dask and Ray for heavy-duty audio processing.
Implements multi-resolution feature extraction, denoising variants, and time-frequency representations.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
import json
import hashlib

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
def get_dask_client():
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='4GB')
    return Client(cluster)

def get_ray():
    import ray
    if not ray.is_initialized():
        ray.init(num_cpus=8, num_gpus=1, ignore_reinit_error=True)
    return ray

@dataclass
class PreprocessingConfig:
    """Configuration for distributed preprocessing pipeline."""
    sample_rates: List[int] = field(default_factory=lambda: [16000, 48000])
    n_mels_variants: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_fft: int = 2048
    hop_length: int = 512
    n_mfcc: int = 40
    segment_duration: float = 30.0
    use_gpu: bool = True
    denoising_methods: List[str] = field(default_factory=lambda: ['spectral_subtraction', 'wiener', 'rnnoise'])
    wavelet_scales: int = 128
    distributed_backend: str = 'ray'  # 'ray' or 'dask'


class DistributedPreprocessor:
    """Heavy-duty distributed audio preprocessing with multi-resolution features."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.client = None

    def initialize_cluster(self):
        """Initialize distributed computing cluster."""
        if self.config.distributed_backend == 'dask':
            self.client = get_dask_client()
            logger.info(f"Dask cluster initialized: {self.client.dashboard_link}")
        else:
            self.ray = get_ray()
            logger.info("Ray cluster initialized")

    def compute_snr(self, audio: np.ndarray, sr: int) -> float:
        """Estimate Signal-to-Noise Ratio using energy-based method."""
        frame_length = int(0.025 * sr)
        hop = int(0.010 * sr)

        # Compute frame energies
        n_frames = 1 + (len(audio) - frame_length) // hop
        energies = np.array([
            np.sum(audio[i*hop:i*hop+frame_length]**2)
            for i in range(n_frames)
        ])

        # Estimate noise from lowest 10% energy frames
        sorted_energies = np.sort(energies)
        noise_energy = np.mean(sorted_energies[:max(1, len(sorted_energies)//10)])
        signal_energy = np.mean(sorted_energies)

        if noise_energy > 0:
            snr = 10 * np.log10(signal_energy / noise_energy)
        else:
            snr = 60.0

        return float(snr)

    def spectral_subtraction_denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Spectral subtraction denoising."""
        import librosa

        # Estimate noise from first 0.5s
        noise_samples = int(0.5 * sr)
        noise = audio[:noise_samples] if len(audio) > noise_samples else audio

        # Compute spectrograms
        S_audio = librosa.stft(audio, n_fft=self.config.n_fft, hop_length=self.config.hop_length)
        S_noise = librosa.stft(noise, n_fft=self.config.n_fft, hop_length=self.config.hop_length)

        # Estimate noise spectrum
        noise_mag = np.mean(np.abs(S_noise), axis=1, keepdims=True)

        # Subtract noise
        audio_mag = np.abs(S_audio)
        audio_phase = np.angle(S_audio)

        # Wiener-like subtraction with flooring
        alpha = 2.0
        beta = 0.01
        denoised_mag = np.maximum(audio_mag - alpha * noise_mag, beta * audio_mag)

        # Reconstruct
        S_denoised = denoised_mag * np.exp(1j * audio_phase)
        denoised = librosa.istft(S_denoised, hop_length=self.config.hop_length)

        return denoised

    def wiener_denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Wiener filter denoising."""
        from scipy.signal import wiener
        return wiener(audio, mysize=None, noise=None)

    def extract_mel_spectrogram(self, audio: np.ndarray, sr: int, n_mels: int) -> np.ndarray:
        """Extract mel spectrogram with specified number of mel bands."""
        import librosa

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=self.config.n_fft,
            hop_length=self.config.hop_length, n_mels=n_mels
        )
        return librosa.power_to_db(mel_spec, ref=np.max)

    def extract_cqt(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract Constant-Q Transform."""
        import librosa

        cqt = librosa.cqt(
            audio, sr=sr, hop_length=self.config.hop_length,
            n_bins=84, bins_per_octave=12
        )
        return librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    def extract_cwt_scalogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract Continuous Wavelet Transform scalogram."""
        import pywt

        # Downsample for CWT efficiency
        if sr > 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        scales = np.arange(1, self.config.wavelet_scales + 1)
        coefficients, frequencies = pywt.cwt(audio, scales, 'morl', sampling_period=1/sr)

        return np.abs(coefficients)

    def extract_mfcc_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract MFCCs with deltas."""
        import librosa

        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft, hop_length=self.config.hop_length
        )

        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        return {
            'mfcc': mfcc,
            'delta': delta,
            'delta2': delta2,
            'mfcc_full': np.vstack([mfcc, delta, delta2])
        }

    def extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract classical spectral features."""
        import librosa

        # Compute spectrogram
        S = np.abs(librosa.stft(audio, n_fft=self.config.n_fft, hop_length=self.config.hop_length))

        features = {
            'spectral_centroid': librosa.feature.spectral_centroid(S=S, sr=sr)[0],
            'spectral_bandwidth': librosa.feature.spectral_bandwidth(S=S, sr=sr)[0],
            'spectral_rolloff': librosa.feature.spectral_rolloff(S=S, sr=sr)[0],
            'spectral_flatness': librosa.feature.spectral_flatness(S=S)[0],
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(audio, hop_length=self.config.hop_length)[0],
            'rms_energy': librosa.feature.rms(S=S)[0],
        }

        # Crest factor
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        features['crest_factor'] = peak / (rms + 1e-10)

        return features

    def extract_advanced_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract advanced features: Teager energy, Hilbert envelope, formants, pitch."""
        import librosa
        from scipy.signal import hilbert

        features = {}

        # Teager Energy Operator
        teo = audio[1:-1]**2 - audio[:-2] * audio[2:]
        features['teager_energy'] = np.mean(np.abs(teo))
        features['teager_energy_std'] = np.std(np.abs(teo))

        # Hilbert envelope
        analytic_signal = hilbert(audio)
        envelope = np.abs(analytic_signal)
        features['hilbert_envelope_mean'] = np.mean(envelope)
        features['hilbert_envelope_std'] = np.std(envelope)

        # Pitch using librosa (faster than pYAAPT)
        try:
            pitches, magnitudes = librosa.piptrack(
                y=audio, sr=sr, n_fft=self.config.n_fft, hop_length=self.config.hop_length
            )
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0
        except Exception:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0

        # Harmonicity (harmonic-to-noise ratio approximation)
        try:
            harmonic, percussive = librosa.effects.hpss(audio)
            hnr = np.sum(harmonic**2) / (np.sum(percussive**2) + 1e-10)
            features['harmonicity'] = 10 * np.log10(hnr + 1e-10)
        except Exception:
            features['harmonicity'] = 0

        return features

    def extract_jitter_shimmer(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract jitter and shimmer voice quality features."""
        import librosa

        # Find pitch periods
        pitches, magnitudes = librosa.piptrack(
            y=audio, sr=sr, n_fft=self.config.n_fft, hop_length=self.config.hop_length
        )

        periods = []
        amplitudes = []

        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 50:  # Valid pitch
                period = sr / pitch
                periods.append(period)
                amplitudes.append(magnitudes[index, t])

        features = {}

        if len(periods) > 1:
            periods = np.array(periods)
            amplitudes = np.array(amplitudes)

            # Jitter (period perturbation)
            period_diffs = np.abs(np.diff(periods))
            features['jitter_local'] = np.mean(period_diffs) / np.mean(periods)
            features['jitter_rap'] = np.mean(np.abs(periods[1:-1] - (periods[:-2] + periods[1:-1] + periods[2:]) / 3)) / np.mean(periods)

            # Shimmer (amplitude perturbation)
            amp_diffs = np.abs(np.diff(amplitudes))
            features['shimmer_local'] = np.mean(amp_diffs) / np.mean(amplitudes)
        else:
            features['jitter_local'] = 0
            features['jitter_rap'] = 0
            features['shimmer_local'] = 0

        return features

    def process_single_audio(self, audio_path: str) -> Dict[str, Any]:
        """Process a single audio file with full feature extraction."""
        import librosa

        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)

        results = {
            'path': audio_path,
            'original_sr': sr,
            'duration': len(audio) / sr,
            'snr': self.compute_snr(audio, sr),
        }

        # Resample to target sample rates
        for target_sr in self.config.sample_rates:
            if target_sr != sr:
                audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            else:
                audio_resampled = audio

            sr_key = f'sr_{target_sr}'
            results[sr_key] = {}

            # Denoising variants
            for method in self.config.denoising_methods:
                if method == 'spectral_subtraction':
                    denoised = self.spectral_subtraction_denoise(audio_resampled, target_sr)
                elif method == 'wiener':
                    denoised = self.wiener_denoise(audio_resampled, target_sr)
                else:
                    denoised = audio_resampled  # Placeholder for RNNoise

                results[sr_key][f'denoised_{method}_snr'] = self.compute_snr(denoised, target_sr)

            # Use spectral subtraction denoised for features
            audio_clean = self.spectral_subtraction_denoise(audio_resampled, target_sr)

            # Time-frequency representations
            for n_mels in self.config.n_mels_variants:
                mel_key = f'mel_{n_mels}'
                results[sr_key][mel_key] = self.extract_mel_spectrogram(audio_clean, target_sr, n_mels)

            results[sr_key]['cqt'] = self.extract_cqt(audio_clean, target_sr)

            # CWT only for lower sample rate (compute intensive)
            if target_sr == min(self.config.sample_rates):
                results[sr_key]['cwt'] = self.extract_cwt_scalogram(audio_clean, target_sr)

            # Classical features
            results[sr_key]['mfcc'] = self.extract_mfcc_features(audio_clean, target_sr)
            results[sr_key]['spectral'] = self.extract_spectral_features(audio_clean, target_sr)
            results[sr_key]['advanced'] = self.extract_advanced_features(audio_clean, target_sr)
            results[sr_key]['voice_quality'] = self.extract_jitter_shimmer(audio_clean, target_sr)

        return results

    def process_batch_ray(self, audio_paths: List[str]) -> List[Dict[str, Any]]:
        """Process batch of audio files using Ray."""
        import ray

        @ray.remote(num_cpus=1)
        def process_remote(path, config):
            preprocessor = DistributedPreprocessor(config)
            return preprocessor.process_single_audio(path)

        futures = [process_remote.remote(path, self.config) for path in audio_paths]
        results = ray.get(futures)

        return results

    def process_batch_dask(self, audio_paths: List[str]) -> List[Dict[str, Any]]:
        """Process batch of audio files using Dask."""
        import dask
        from dask import delayed

        @delayed
        def process_delayed(path):
            return self.process_single_audio(path)

        tasks = [process_delayed(path) for path in audio_paths]
        results = dask.compute(*tasks)

        return list(results)

    def process_dataset(self, audio_paths: List[str], output_dir: str) -> str:
        """Process entire dataset with distributed computing."""
        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {len(audio_paths)} audio files")

        if self.config.distributed_backend == 'ray':
            results = self.process_batch_ray(audio_paths)
        else:
            results = self.process_batch_dask(audio_paths)

        # Save results
        manifest = {
            'n_files': len(results),
            'config': self.config.__dict__,
            'files': []
        }

        for i, result in enumerate(results):
            # Save features to numpy files
            file_hash = hashlib.md5(result['path'].encode()).hexdigest()[:8]
            feature_file = output_path / f"{file_hash}_features.npz"

            # Extract numpy arrays for saving
            arrays_to_save = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    arrays_to_save[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            arrays_to_save[f"{key}_{subkey}"] = subvalue

            np.savez_compressed(feature_file, **arrays_to_save)

            manifest['files'].append({
                'path': result['path'],
                'feature_file': str(feature_file),
                'snr': result.get('snr', 0),
                'duration': result.get('duration', 0)
            })

        manifest_file = output_path / 'manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved manifest to {manifest_file}")
        return str(manifest_file)


class LabelFusionModule:
    """Aggregate labels across datasets and annotators using probabilistic fusion."""

    def __init__(self, n_classes: int = 4):
        self.n_classes = n_classes

    def dawid_skene(self, annotations: np.ndarray, n_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dawid-Skene algorithm for label fusion.

        Args:
            annotations: Shape (n_items, n_annotators), -1 for missing
            n_iter: Number of EM iterations

        Returns:
            true_labels: Estimated true labels
            annotator_accuracy: Per-annotator confusion matrices
        """
        n_items, n_annotators = annotations.shape

        # Initialize with majority vote
        true_labels = np.zeros((n_items, self.n_classes))
        for i in range(n_items):
            valid_annotations = annotations[i][annotations[i] >= 0]
            if len(valid_annotations) > 0:
                for ann in valid_annotations:
                    true_labels[i, int(ann)] += 1
                true_labels[i] /= true_labels[i].sum()
            else:
                true_labels[i] = 1.0 / self.n_classes

        # Initialize annotator error rates
        annotator_accuracy = np.zeros((n_annotators, self.n_classes, self.n_classes))
        for j in range(n_annotators):
            annotator_accuracy[j] = np.eye(self.n_classes) * 0.8 + 0.2 / self.n_classes

        # EM iterations
        for _ in range(n_iter):
            # M-step: Update annotator accuracies
            for j in range(n_annotators):
                annotator_accuracy[j] = np.ones((self.n_classes, self.n_classes)) * 1e-6
                for i in range(n_items):
                    if annotations[i, j] >= 0:
                        ann = int(annotations[i, j])
                        for k in range(self.n_classes):
                            annotator_accuracy[j, k, ann] += true_labels[i, k]

                # Normalize rows
                for k in range(self.n_classes):
                    annotator_accuracy[j, k] /= annotator_accuracy[j, k].sum()

            # E-step: Update true label estimates
            for i in range(n_items):
                log_probs = np.zeros(self.n_classes)
                for k in range(self.n_classes):
                    log_probs[k] = np.log(1.0 / self.n_classes)  # Prior
                    for j in range(n_annotators):
                        if annotations[i, j] >= 0:
                            ann = int(annotations[i, j])
                            log_probs[k] += np.log(annotator_accuracy[j, k, ann] + 1e-10)

                # Softmax
                log_probs -= np.max(log_probs)
                true_labels[i] = np.exp(log_probs)
                true_labels[i] /= true_labels[i].sum()

        return true_labels.argmax(axis=1), annotator_accuracy


class SyntheticAugmentation:
    """Generate synthetic sleep audio augmentations for training data expansion."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Time stretch without pitch change."""
        import librosa
        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(self, audio: np.ndarray, n_steps: float) -> np.ndarray:
        """Pitch shift."""
        import librosa
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)

    def add_background_noise(self, audio: np.ndarray, noise_type: str = 'white', snr_db: float = 20) -> np.ndarray:
        """Add realistic background noise."""
        if noise_type == 'white':
            noise = np.random.randn(len(audio))
        elif noise_type == 'pink':
            # Pink noise (1/f)
            freqs = np.fft.rfftfreq(len(audio))
            freqs[0] = 1  # Avoid division by zero
            pink_filter = 1 / np.sqrt(freqs)
            white = np.random.randn(len(audio))
            pink_spectrum = np.fft.rfft(white) * pink_filter
            noise = np.fft.irfft(pink_spectrum, n=len(audio))
        elif noise_type == 'brown':
            # Brownian noise
            noise = np.cumsum(np.random.randn(len(audio)))
            noise = noise - np.mean(noise)
        else:
            noise = np.random.randn(len(audio))

        # Scale noise to desired SNR
        signal_power = np.mean(audio**2)
        noise_power = np.mean(noise**2)
        scale = np.sqrt(signal_power / (noise_power * 10**(snr_db/10)))

        return audio + scale * noise

    def add_reverb(self, audio: np.ndarray, room_size: float = 0.5) -> np.ndarray:
        """Add simple reverb effect."""
        from scipy.signal import fftconvolve

        # Simple exponential decay impulse response
        ir_length = int(room_size * self.sr)
        t = np.arange(ir_length) / self.sr
        ir = np.exp(-3 * t / room_size) * np.random.randn(ir_length)
        ir = ir / np.sum(np.abs(ir))

        reverbed = fftconvolve(audio, ir, mode='same')
        return 0.7 * audio + 0.3 * reverbed

    def simulate_apnea_event(self, audio: np.ndarray, start: float, duration: float = 15.0) -> np.ndarray:
        """Simulate apnea event by creating breathing cessation."""
        start_sample = int(start * self.sr)
        end_sample = int((start + duration) * self.sr)

        # Gradual fade out and in
        fade_samples = int(0.5 * self.sr)

        augmented = audio.copy()

        # Fade out
        if start_sample > fade_samples:
            fade_out = np.linspace(1, 0, fade_samples)
            augmented[start_sample-fade_samples:start_sample] *= fade_out

        # Silence during apnea (with very low noise)
        if end_sample <= len(augmented):
            augmented[start_sample:end_sample] = np.random.randn(end_sample - start_sample) * 0.001

        # Fade in with gasp
        if end_sample + fade_samples <= len(augmented):
            fade_in = np.linspace(0, 1.5, fade_samples)  # Slight amplitude increase for gasp
            augmented[end_sample:end_sample+fade_samples] *= fade_in

        return augmented

    def augment_batch(self, audio: np.ndarray, n_augmentations: int = 5) -> List[np.ndarray]:
        """Generate multiple augmentations of a single audio."""
        augmented = []

        for i in range(n_augmentations):
            aug = audio.copy()

            # Random combination of augmentations
            if np.random.random() < 0.5:
                aug = self.time_stretch(aug, np.random.uniform(0.9, 1.1))

            if np.random.random() < 0.3:
                aug = self.pitch_shift(aug, np.random.uniform(-2, 2))

            if np.random.random() < 0.7:
                noise_type = np.random.choice(['white', 'pink', 'brown'])
                snr = np.random.uniform(15, 30)
                aug = self.add_background_noise(aug, noise_type, snr)

            if np.random.random() < 0.3:
                aug = self.add_reverb(aug, np.random.uniform(0.2, 0.8))

            augmented.append(aug)

        return augmented
