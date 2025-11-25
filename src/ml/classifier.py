import json
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import ollama
from enum import IntEnum

logger = logging.getLogger(__name__)


class SleepEvent(IntEnum):
    NORMAL = 0
    SNORING = 1
    HYPOPNEA = 2
    APNEA = 3


EVENT_NAMES = ['normal', 'snoring', 'hypopnea', 'apnea']


class OllamaClassifier:
    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434"
    ):
        self.model = model
        self.host = host
        self.system_prompt = """You are an expert sleep apnea diagnostic system. Analyze audio features from sleep recordings and classify them into one of four categories:

0 - NORMAL: Regular breathing patterns
1 - SNORING: Periodic vibrations (20-300 Hz peaks), high low-frequency energy
2 - HYPOPNEA: Partial airway obstruction, reduced airflow, moderate energy reduction
3 - APNEA: Complete breathing cessation for >10 seconds, very low energy across all bands

Key indicators:
- High snore_score (>2.5) with dominant frequency 20-300 Hz = SNORING
- Very low total energy (<0.01) with low RMS = APNEA
- Moderate energy reduction (50-70% of normal) = HYPOPNEA
- Regular energy patterns with no snoring = NORMAL

Respond with ONLY a JSON object containing:
{"classification": <0-3>, "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}"""

    def classify(self, features: Dict) -> Dict:
        feature_text = self._format_features(features)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyze these audio features and classify the sleep event:\n\n{feature_text}"}
                ],
                options={"temperature": 0.1}
            )
            result = self._parse_response(response['message']['content'])
            return result
        except Exception as e:
            logger.error(f"Ollama classification error: {e}")
            return self._rule_based_classify(features)

    def _format_features(self, features: Dict) -> str:
        lines = []
        if 'energy' in features:
            e = features['energy']
            lines.append(f"RMS Energy: mean={e['rms_mean']:.6f}, std={e['rms_std']:.6f}, max={e['rms_max']:.6f}")
            lines.append(f"Zero-crossing rate: mean={e['zcr_mean']:.6f}")
        if 'snoring' in features:
            s = features['snoring']
            lines.append(f"Snore score: {s['snore_score']:.3f}")
            lines.append(f"Is snoring: {s['is_snoring']}")
            lines.append(f"Dominant frequency: {s['dominant_freq']:.1f} Hz")
            lines.append(f"Low freq energy: {s['low_energy']:.4f}")
            lines.append(f"Mid freq energy: {s['mid_energy']:.4f}")
            lines.append(f"Total energy: {s['total_energy']:.4f}")
        if 'spectral' in features:
            sp = features['spectral']
            lines.append(f"Spectral centroid: {sp['spectral_centroid']:.1f} Hz")
            lines.append(f"Spectral bandwidth: {sp['spectral_bandwidth']:.1f} Hz")
            lines.append(f"Spectral flatness: {sp['spectral_flatness']:.6f}")
        if 'pitch' in features:
            p = features['pitch']
            lines.append(f"Pitch: mean={p['pitch_mean']:.1f} Hz, std={p['pitch_std']:.1f}")
        return "\n".join(lines)

    def _parse_response(self, response_text: str) -> Dict:
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                result = json.loads(json_str)
                return {
                    'event_type': EVENT_NAMES[result['classification']],
                    'event_class': result['classification'],
                    'confidence': result['confidence'],
                    'reasoning': result.get('reasoning', '')
                }
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to parse Ollama response: {e}")
        return {
            'event_type': 'normal',
            'event_class': 0,
            'confidence': 0.5,
            'reasoning': 'Could not parse LLM response'
        }

    def _rule_based_classify(self, features: Dict) -> Dict:
        energy = features.get('energy', {})
        snoring = features.get('snoring', {})
        rms_mean = energy.get('rms_mean', 0.1)
        snore_score = snoring.get('snore_score', 0)
        total_energy = snoring.get('total_energy', 1)
        is_snoring = snoring.get('is_snoring', False)
        if rms_mean < 0.001 or total_energy < 0.01:
            return {
                'event_type': 'apnea',
                'event_class': 3,
                'confidence': 0.8,
                'reasoning': 'Very low energy indicates breathing cessation'
            }
        elif is_snoring or snore_score > 2.5:
            return {
                'event_type': 'snoring',
                'event_class': 1,
                'confidence': 0.75,
                'reasoning': 'High low-frequency energy indicates snoring'
            }
        elif rms_mean < 0.01 or total_energy < 0.1:
            return {
                'event_type': 'hypopnea',
                'event_class': 2,
                'confidence': 0.7,
                'reasoning': 'Reduced energy indicates partial obstruction'
            }
        else:
            return {
                'event_type': 'normal',
                'event_class': 0,
                'confidence': 0.85,
                'reasoning': 'Normal breathing patterns detected'
            }

    def batch_classify(self, features_list: List[Dict]) -> List[Dict]:
        results = []
        for features in features_list:
            result = self.classify(features)
            results.append(result)
        return results


class SleepEventClassifier:
    def __init__(
        self,
        ollama_model: str = "llama3.2",
        sample_rate: int = 16000
    ):
        from src.audio import AudioPreprocessor, FeatureExtractor
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)
        self.classifier = OllamaClassifier(model=ollama_model)

    def classify_audio(self, audio: np.ndarray) -> Dict:
        processed = self.preprocessor.process_audio_segment(audio)
        features = self.feature_extractor.extract_all_features(processed)
        result = self.classifier.classify(features)
        result['features'] = features
        return result

    def classify_file(self, file_path: str) -> List[Dict]:
        audio = self.preprocessor.load_audio(file_path)
        segments = self.preprocessor.segment_audio(audio, segment_duration=30.0)
        results = []
        for i, segment in enumerate(segments):
            result = self.classify_audio(segment)
            result['segment_index'] = i
            result['timestamp'] = i * 30.0
            results.append(result)
        return results
