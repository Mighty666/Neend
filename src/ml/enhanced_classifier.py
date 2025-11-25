import numpy as np
from typing import Dict, List, Tuple
import logging

from .online.algorithms import OnlineClassifier, StreamingAHI, AdaptiveThreshold
from .classifier import OllamaClassifier, EVENT_NAMES

logger = logging.getLogger(__name__)


class EnhancedClassifier:
    def __init__(
        self,
        n_features: int = 39,
        n_classes: int = 4,
        ollama_model: str = "llama3.2",
        enable_online_learning: bool = True
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.ollama = OllamaClassifier(model=ollama_model)
        self.online_clf = OnlineClassifier(n_features, n_classes) if enable_online_learning else None
        self.streaming_ahi = StreamingAHI()
        self.thresholds = {
            'energy': AdaptiveThreshold(percentile=10),
            'snore': AdaptiveThreshold(percentile=90),
        }
        self.predictions_history = []
        self.confidence_history = []
        self.ollama_weight = 0.7
        self.online_weight = 0.3

    def classify(self, features: Dict, true_label: int = None) -> Dict:
        ollama_result = self.ollama.classify(features)
        ollama_probs = self._result_to_probs(ollama_result)
        feature_vector = self._features_to_vector(features)
        online_probs = np.ones(self.n_classes) / self.n_classes
        if self.online_clf:
            _, online_probs = self.online_clf.predict(feature_vector)
            if true_label is not None:
                self.online_clf.partial_fit(feature_vector, true_label)
        ensemble_probs = (
            self.ollama_weight * ollama_probs +
            self.online_weight * online_probs
        )
        pred_class = int(np.argmax(ensemble_probs))
        confidence = float(ensemble_probs[pred_class])
        self._update_thresholds(features)
        ahi_stats = self.streaming_ahi.update(pred_class)
        self.predictions_history.append(pred_class)
        self.confidence_history.append(confidence)
        return {
            'event_type': EVENT_NAMES[pred_class],
            'event_class': pred_class,
            'confidence': confidence,
            'ensemble_probs': ensemble_probs.tolist(),
            'ollama_result': ollama_result,
            'ahi_stats': ahi_stats,
            'reasoning': ollama_result.get('reasoning', '')
        }

    def _result_to_probs(self, result: Dict) -> np.ndarray:
        probs = np.zeros(self.n_classes)
        event_class = result.get('event_class', 0)
        confidence = result.get('confidence', 0.5)
        probs[event_class] = confidence
        remaining = 1 - confidence
        for i in range(self.n_classes):
            if i != event_class:
                probs[i] = remaining / (self.n_classes - 1)
        return probs

    def _features_to_vector(self, features: Dict) -> np.ndarray:
        vector = []
        if 'mfcc' in features:
            mfcc = features['mfcc']
            vector.extend(mfcc.get('mfcc_mean', [0] * 39))
        if 'energy' in features:
            e = features['energy']
            vector.extend([
                e.get('rms_mean', 0),
                e.get('rms_std', 0),
                e.get('zcr_mean', 0),
                e.get('zcr_std', 0)
            ])
        if 'spectral' in features:
            s = features['spectral']
            vector.extend([
                s.get('spectral_centroid', 0),
                s.get('spectral_bandwidth', 0),
                s.get('spectral_rolloff', 0),
                s.get('spectral_flatness', 0)
            ])
        if 'snoring' in features:
            sn = features['snoring']
            vector.extend([
                sn.get('snore_score', 0),
                sn.get('dominant_freq', 0),
                sn.get('total_energy', 0)
            ])
        vector = np.array(vector[:self.n_features])
        if len(vector) < self.n_features:
            vector = np.pad(vector, (0, self.n_features - len(vector)))
        return vector

    def _update_thresholds(self, features: Dict):
        if 'energy' in features:
            self.thresholds['energy'].update(features['energy'].get('rms_mean', 0))
        if 'snoring' in features:
            self.thresholds['snore'].update(features['snoring'].get('snore_score', 0))

    def get_stats(self) -> Dict:
        return {
            'total_predictions': len(self.predictions_history),
            'avg_confidence': np.mean(self.confidence_history) if self.confidence_history else 0,
            'class_distribution': {
                EVENT_NAMES[i]: self.predictions_history.count(i)
                for i in range(self.n_classes)
            },
            'ollama_weight': self.ollama_weight,
            'online_weight': self.online_weight,
            'energy_threshold': self.thresholds['energy'].threshold,
            'snore_threshold': self.thresholds['snore'].threshold
        }

    def adapt_weights(self, performance_delta: float):
        adjustment = 0.05 * performance_delta
        self.online_weight = np.clip(self.online_weight + adjustment, 0.1, 0.5)
        self.ollama_weight = 1 - self.online_weight

    def reset_streaming(self):
        self.streaming_ahi = StreamingAHI()
        self.predictions_history = []
        self.confidence_history = []


class PersonalizedClassifier(EnhancedClassifier):
    def __init__(self, user_id: str, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.user_baseline = {
            'avg_energy': None,
            'avg_snore_score': None,
            'typical_ahi': None
        }
        self.nightly_patterns = []

    def update_baseline(self, sessions: List[Dict]):
        if not sessions:
            return
        ahi_values = [s.get('ahi', 0) for s in sessions]
        self.user_baseline['typical_ahi'] = np.mean(ahi_values)

    def classify_with_personalization(self, features: Dict) -> Dict:
        result = self.classify(features)
        if self.user_baseline['typical_ahi']:
            current_ahi = result['ahi_stats']['overall_ahi']
            baseline_ahi = self.user_baseline['typical_ahi']
            if current_ahi > baseline_ahi * 1.5:
                result['alert'] = 'AHI significantly higher than your typical pattern'
            elif current_ahi < baseline_ahi * 0.5:
                result['improvement'] = 'AHI lower than usual - good night!'
        return result
