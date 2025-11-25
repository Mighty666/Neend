import numpy as np
import librosa
from scipy import signal
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 2048,
        hop_length: Optional[int] = None
    ):
        self.sr = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length or frame_length // 2

    def load_audio(self, file_path: str) -> np.ndarray:
        audio, _ = librosa.load(file_path, sr=self.sr, mono=True)
        return audio

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        return librosa.util.normalize(audio)

    def pre_emphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        return librosa.effects.preemphasis(audio, coef=coef)

    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        noise_sample = audio[:int(0.5 * self.sr)]
        noise_stft = librosa.stft(noise_sample, n_fft=self.frame_length)
        noise_mag = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
        audio_stft = librosa.stft(audio, n_fft=self.frame_length)
        audio_mag = np.abs(audio_stft)
        audio_phase = np.angle(audio_stft)
        mask = audio_mag > (noise_mag * 2)
        cleaned_mag = audio_mag * mask
        cleaned_stft = cleaned_mag * np.exp(1j * audio_phase)
        cleaned_audio = librosa.istft(cleaned_stft)
        return cleaned_audio

    def segment_audio(
        self,
        audio: np.ndarray,
        segment_duration: float = 30.0
    ) -> list[np.ndarray]:
        segment_samples = int(segment_duration * self.sr)
        segments = []
        for start in range(0, len(audio), segment_samples):
            end = start + segment_samples
            segment = audio[start:end]
            if len(segment) < segment_samples:
                segment = np.pad(segment, (0, segment_samples - len(segment)))
            segments.append(segment)
        return segments

    def apply_bandpass_filter(
        self,
        audio: np.ndarray,
        low_freq: float = 20.0,
        high_freq: float = 2000.0
    ) -> np.ndarray:
        nyquist = self.sr / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        return filtered

    def process_audio_segment(self, audio_buffer: np.ndarray) -> np.ndarray:
        audio_normalized = self.normalize(audio_buffer)
        audio_denoised = self.reduce_noise(audio_normalized)
        pre_emphasized = self.pre_emphasis(audio_denoised)
        filtered = self.apply_bandpass_filter(pre_emphasized)
        return filtered

    def frame_audio(self, audio: np.ndarray) -> np.ndarray:
        frames = librosa.util.frame(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        return frames.T

    def get_energy_envelope(self, audio: np.ndarray) -> np.ndarray:
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        return rms

    def detect_silence(
        self,
        audio: np.ndarray,
        threshold_db: float = -40.0,
        min_duration: float = 10.0
    ) -> list[Tuple[float, float]]:
        rms = self.get_energy_envelope(audio)
        rms_db = librosa.amplitude_to_db(rms)
        silent_frames = rms_db < threshold_db
        min_frames = int(min_duration * self.sr / self.hop_length)
        silence_periods = []
        start = None
        for i, is_silent in enumerate(silent_frames):
            if is_silent and start is None:
                start = i
            elif not is_silent and start is not None:
                if i - start >= min_frames:
                    start_time = start * self.hop_length / self.sr
                    end_time = i * self.hop_length / self.sr
                    silence_periods.append((start_time, end_time))
                start = None
        if start is not None and len(silent_frames) - start >= min_frames:
            start_time = start * self.hop_length / self.sr
            end_time = len(silent_frames) * self.hop_length / self.sr
            silence_periods.append((start_time, end_time))
        return silence_periods
