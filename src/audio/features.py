import numpy as np
import librosa
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

    def compute_fft(self, audio_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = len(audio_frame)
        windowed = audio_frame * np.hamming(N)
        fft_values = fft(windowed)
        fft_magnitude = np.abs(fft_values[:N//2])
        freqs = fftfreq(N, 1/self.sr)[:N//2]
        return freqs, fft_magnitude

    def detect_snoring_signature(
        self,
        freqs: np.ndarray,
        magnitude: np.ndarray
    ) -> Dict:
        low_band = (freqs >= 20) & (freqs <= 100)
        mid_band = (freqs > 100) & (freqs <= 300)
        high_band = (freqs > 300) & (freqs <= 1000)
        low_energy = np.sum(magnitude[low_band]) if np.any(low_band) else 0
        mid_energy = np.sum(magnitude[mid_band]) if np.any(mid_band) else 0
        high_energy = np.sum(magnitude[high_band]) if np.any(high_band) else 0
        total_energy = np.sum(magnitude)
        snore_ratio = low_energy / (mid_energy + 1e-6)
        dominant_freq = freqs[np.argmax(magnitude)] if len(magnitude) > 0 else 0
        return {
            'snore_score': float(snore_ratio),
            'is_snoring': snore_ratio > 2.5 and low_energy > 0.1 * total_energy,
            'dominant_freq': float(dominant_freq),
            'low_energy': float(low_energy),
            'mid_energy': float(mid_energy),
            'high_energy': float(high_energy),
            'total_energy': float(total_energy)
        }

    def compute_spectral_features(
        self,
        freqs: np.ndarray,
        magnitude: np.ndarray
    ) -> Dict:
        total_energy = np.sum(magnitude) + 1e-10
        centroid = np.sum(freqs * magnitude) / total_energy
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / total_energy)
        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        rolloff = freqs[rolloff_idx] if rolloff_idx < len(freqs) else freqs[-1]
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
        arithmetic_mean = np.mean(magnitude)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        return {
            'spectral_centroid': float(centroid),
            'spectral_bandwidth': float(bandwidth),
            'spectral_rolloff': float(rolloff),
            'spectral_flatness': float(flatness)
        }


class FeatureExtractor:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        self.sr = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_analyzer = FrequencyAnalyzer(sample_rate)

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        return features

    def extract_formants(self, audio: np.ndarray) -> Dict:
        from scipy.signal import lfilter
        pre_emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        order = 2 + self.sr // 1000
        spectrum = np.abs(fft(pre_emphasized * np.hamming(len(pre_emphasized))))
        freqs = fftfreq(len(pre_emphasized), 1/self.sr)
        pos_spectrum = spectrum[:len(spectrum)//2]
        pos_freqs = freqs[:len(freqs)//2]
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(pos_spectrum, height=np.mean(pos_spectrum))
        formants = {'F1': 0, 'F2': 0, 'F3': 0}
        if len(peaks) >= 1:
            formants['F1'] = float(pos_freqs[peaks[0]])
        if len(peaks) >= 2:
            formants['F2'] = float(pos_freqs[peaks[1]])
        if len(peaks) >= 3:
            formants['F3'] = float(pos_freqs[peaks[2]])
        return formants

    def extract_pitch(self, audio: np.ndarray) -> np.ndarray:
        f0 = librosa.yin(
            audio,
            fmin=50,
            fmax=500,
            sr=self.sr,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        return f0

    def extract_energy_features(self, audio: np.ndarray) -> Dict:
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )[0]
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )[0]
        return {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'rms_max': float(np.max(rms)),
            'rms_min': float(np.min(rms)),
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr))
        }

    def extract_all_features(self, audio: np.ndarray) -> Dict:
        mfcc = self.extract_mfcc(audio)
        mfcc_features = {
            'mfcc_mean': mfcc.mean(axis=1).tolist(),
            'mfcc_std': mfcc.std(axis=1).tolist()
        }
        freqs, magnitude = self.freq_analyzer.compute_fft(audio)
        snoring_analysis = self.freq_analyzer.detect_snoring_signature(freqs, magnitude)
        spectral_features = self.freq_analyzer.compute_spectral_features(freqs, magnitude)
        formant_features = self.extract_formants(audio)
        pitch = self.extract_pitch(audio)
        pitch_features = {
            'pitch_mean': float(np.mean(pitch)),
            'pitch_std': float(np.std(pitch)),
            'pitch_max': float(np.max(pitch)),
            'pitch_min': float(np.min(pitch))
        }
        energy_features = self.extract_energy_features(audio)
        return {
            'mfcc': mfcc_features,
            'spectral': spectral_features,
            'snoring': snoring_analysis,
            'formants': formant_features,
            'pitch': pitch_features,
            'energy': energy_features
        }

    def extract_feature_matrix(
        self,
        audio: np.ndarray,
        target_frames: int = 128
    ) -> np.ndarray:
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        if mfcc.shape[1] < target_frames:
            pad_width = target_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :target_frames]
        feature_matrix = mfcc.T[:, :, np.newaxis]
        return feature_matrix

    def create_feature_summary(self, features: Dict) -> str:
        summary = []
        energy = features['energy']
        summary.append(f"Energy: mean={energy['rms_mean']:.4f}, std={energy['rms_std']:.4f}")
        snoring = features['snoring']
        summary.append(f"Snoring score: {snoring['snore_score']:.2f}, is_snoring={snoring['is_snoring']}")
        summary.append(f"Dominant frequency: {snoring['dominant_freq']:.1f} Hz")
        summary.append(f"Energy distribution: low={snoring['low_energy']:.2f}, mid={snoring['mid_energy']:.2f}, high={snoring['high_energy']:.2f}")
        spectral = features['spectral']
        summary.append(f"Spectral centroid: {spectral['spectral_centroid']:.1f} Hz")
        summary.append(f"Spectral bandwidth: {spectral['spectral_bandwidth']:.1f} Hz")
        summary.append(f"Spectral rolloff: {spectral['spectral_rolloff']:.1f} Hz")
        summary.append(f"Spectral flatness: {spectral['spectral_flatness']:.4f}")
        pitch = features['pitch']
        summary.append(f"Pitch: mean={pitch['pitch_mean']:.1f} Hz, std={pitch['pitch_std']:.1f}")
        formants = features['formants']
        summary.append(f"Formants: F1={formants['F1']:.1f}, F2={formants['F2']:.1f}, F3={formants['F3']:.1f}")
        return "\n".join(summary)
