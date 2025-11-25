"""
Online Learning Algorithms for Real-time Sleep Apnea Detection
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import deque
from dataclasses import dataclass

@dataclass
class StreamStats:
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')

class WelfordAccumulator:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, value: float):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)

class OnlineClassifier:
    def __init__(self, n_features: int, n_classes: int = 4, learning_rate: float = 0.01, regularization: float = 0.0001):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.reg = regularization
        self.weights = np.random.randn(n_classes, n_features) * 0.01
        self.bias = np.zeros(n_classes)
        self.grad_sq_weights = np.zeros_like(self.weights)
        self.grad_sq_bias = np.zeros_like(self.bias)
        self.feature_stats = [WelfordAccumulator() for _ in range(n_features)]
        self.n_updates = 0

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        normalized = np.zeros_like(features)
        for i, (val, stats) in enumerate(zip(features, self.feature_stats)):
            normalized[i] = (val - stats.mean) / stats.std if stats.std > 0 else val - stats.mean
        return normalized

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def partial_fit(self, features: np.ndarray, label: int):
        for i, val in enumerate(features):
            self.feature_stats[i].update(val)
        x = self._normalize_features(features)
        logits = self.weights @ x + self.bias
        probs = self._softmax(logits)
        y_onehot = np.zeros(self.n_classes)
        y_onehot[label] = 1.0
        grad_logits = probs - y_onehot
        grad_weights = np.outer(grad_logits, x) + self.reg * self.weights
        grad_bias = grad_logits
        self.grad_sq_weights += grad_weights ** 2
        self.grad_sq_bias += grad_bias ** 2
        self.weights -= self.lr * grad_weights / (np.sqrt(self.grad_sq_weights) + 1e-8)
        self.bias -= self.lr * grad_bias / (np.sqrt(self.grad_sq_bias) + 1e-8)
        self.n_updates += 1

    def predict(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        x = self._normalize_features(features)
        logits = self.weights @ x + self.bias
        probs = self._softmax(logits)
        return int(np.argmax(probs)), probs

class StreamingAHI:
    def __init__(self, window_hours: float = 1.0, segment_duration: float = 30.0, decay_factor: float = 0.99):
        self.window_hours = window_hours
        self.segment_duration = segment_duration
        self.decay_factor = decay_factor
        self.window_size = int(window_hours * 3600 / segment_duration)
        self.event_buffer = deque(maxlen=self.window_size)
        self.total_apnea = 0
        self.total_hypopnea = 0
        self.total_segments = 0
        self.ew_apnea = 0.0
        self.ew_hypopnea = 0.0
        self.ew_total = 0.0
        self.hourly_events = []
        self.current_hour_events = []

    def update(self, event_class: int) -> Dict:
        self.total_segments += 1
        if event_class == 3:
            self.total_apnea += 1
        elif event_class == 2:
            self.total_hypopnea += 1
        self.ew_apnea = self.decay_factor * self.ew_apnea + (1 if event_class == 3 else 0)
        self.ew_hypopnea = self.decay_factor * self.ew_hypopnea + (1 if event_class == 2 else 0)
        self.ew_total = self.decay_factor * self.ew_total + 1
        self.event_buffer.append(event_class)
        self.current_hour_events.append(event_class)
        segments_per_hour = int(3600 / self.segment_duration)
        if len(self.current_hour_events) >= segments_per_hour:
            self.hourly_events.append(self.current_hour_events.count(3) + self.current_hour_events.count(2))
            self.current_hour_events = []
        return self.get_current_stats()

    def get_current_stats(self) -> Dict:
        total_hours = max(self.total_segments * self.segment_duration / 3600, 0.001)
        overall_ahi = (self.total_apnea + self.total_hypopnea) / total_hours
        window_apnea = list(self.event_buffer).count(3)
        window_hypopnea = list(self.event_buffer).count(2)
        window_hours = len(self.event_buffer) * self.segment_duration / 3600
        windowed_ahi = (window_apnea + window_hypopnea) / max(window_hours, 0.001)
        ew_hours = self.ew_total * self.segment_duration / 3600
        ew_ahi = (self.ew_apnea + self.ew_hypopnea) / max(ew_hours, 0.001)
        severity = 'Normal' if overall_ahi < 5 else 'Mild' if overall_ahi < 15 else 'Moderate' if overall_ahi < 30 else 'Severe'
        trend = 'stable'
        if len(self.hourly_events) >= 2:
            diff = self.hourly_events[-1] - self.hourly_events[0]
            trend = 'increasing' if diff > 5 else 'decreasing' if diff < -5 else 'stable'
        return {'overall_ahi': round(overall_ahi, 2), 'windowed_ahi': round(windowed_ahi, 2), 'ew_ahi': round(ew_ahi, 2), 'severity': severity, 'total_apnea': self.total_apnea, 'total_hypopnea': self.total_hypopnea, 'total_hours': round(total_hours, 2), 'trend': trend}

class AdaptiveThreshold:
    def __init__(self, percentile: float = 95.0):
        self.p = percentile / 100.0
        self.n = 0
        self.q = [0.0] * 5
        self.n_pos = [1, 2, 3, 4, 5]
        self.n_prime = [1, 1 + 2*self.p, 1 + 4*self.p, 3 + 2*self.p, 5]
        self.dn = [0, self.p/2, self.p, (1+self.p)/2, 1]

    def update(self, value: float):
        self.n += 1
        if self.n <= 5:
            self.q[self.n - 1] = value
            if self.n == 5:
                self.q.sort()
            return
        if value < self.q[0]:
            self.q[0] = value
            k = 1
        elif value >= self.q[4]:
            self.q[4] = value
            k = 4
        else:
            for i in range(1, 5):
                if value < self.q[i]:
                    k = i
                    break
        for i in range(k, 5):
            self.n_pos[i] += 1
        for i in range(5):
            self.n_prime[i] += self.dn[i]

    @property
    def threshold(self) -> float:
        return self.q[2] if self.n >= 5 else max(self.q[:self.n]) if self.n > 0 else 0.0
