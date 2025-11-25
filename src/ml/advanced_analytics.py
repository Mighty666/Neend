"""
Advanced Analytics for Sleep Apnea Detection
Statistical analysis, trend detection, and predictive modeling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Analyze trends in sleep data over multiple sessions.
    """

    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.ahi_history = deque(maxlen=30)  # 30 days
        self.event_history = deque(maxlen=30)

    def add_session(self, ahi: float, events: Dict):
        """Add a session's data to history"""
        self.ahi_history.append(ahi)
        self.event_history.append(events)

    def get_trend(self) -> Dict:
        """
        Calculate trend using linear regression.
        """
        if len(self.ahi_history) < 3:
            return {'direction': 'insufficient_data', 'slope': 0, 'confidence': 0}

        x = np.arange(len(self.ahi_history))
        y = np.array(list(self.ahi_history))

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Determine trend direction
        if p_value > 0.1:
            direction = 'stable'
        elif slope > 0.5:
            direction = 'worsening'
        elif slope < -0.5:
            direction = 'improving'
        else:
            direction = 'stable'

        return {
            'direction': direction,
            'slope': round(slope, 3),
            'r_squared': round(r_value ** 2, 3),
            'p_value': round(p_value, 4),
            'prediction_next': round(slope * len(self.ahi_history) + intercept, 1)
        }

    def get_weekly_comparison(self) -> Dict:
        """Compare current week to previous week"""
        if len(self.ahi_history) < 14:
            return {'available': False}

        current_week = list(self.ahi_history)[-7:]
        previous_week = list(self.ahi_history)[-14:-7]

        current_avg = np.mean(current_week)
        previous_avg = np.mean(previous_week)

        change = current_avg - previous_avg
        change_pct = (change / previous_avg * 100) if previous_avg > 0 else 0

        return {
            'available': True,
            'current_avg': round(current_avg, 1),
            'previous_avg': round(previous_avg, 1),
            'change': round(change, 1),
            'change_percent': round(change_pct, 1),
            'improved': change < 0
        }

    def detect_anomaly(self) -> Optional[Dict]:
        """Detect anomalous sessions using z-score"""
        if len(self.ahi_history) < 7:
            return None

        values = np.array(list(self.ahi_history))
        mean = np.mean(values[:-1])  # Exclude last value
        std = np.std(values[:-1])

        if std == 0:
            return None

        last_value = values[-1]
        z_score = (last_value - mean) / std

        if abs(z_score) > 2:
            return {
                'detected': True,
                'value': last_value,
                'z_score': round(z_score, 2),
                'direction': 'higher' if z_score > 0 else 'lower',
                'message': f'AHI of {last_value:.1f} is {abs(z_score):.1f} standard deviations {"above" if z_score > 0 else "below"} your average'
            }

        return None


class SleepStageEstimator:
    """
    Estimate sleep stages based on audio features.
    Simplified model based on breathing patterns.
    """

    def __init__(self):
        self.stages = ['awake', 'light', 'deep', 'rem']

    def estimate(self, features: Dict) -> Dict:
        """
        Estimate current sleep stage from audio features.

        Uses simplified heuristics based on:
        - Energy variation
        - Breathing regularity
        - Movement artifacts
        """
        energy = features.get('energy', {})
        spectral = features.get('spectral', {})

        rms_mean = energy.get('rms_mean', 0.05)
        rms_std = energy.get('rms_std', 0.01)
        flatness = spectral.get('spectral_flatness', 0.5)

        # Heuristic scoring
        scores = {
            'awake': 0,
            'light': 0,
            'deep': 0,
            'rem': 0
        }

        # High variability = awake or REM
        variability = rms_std / (rms_mean + 0.001)
        if variability > 0.5:
            scores['awake'] += 2
            scores['rem'] += 1
        elif variability < 0.1:
            scores['deep'] += 2

        # Energy level
        if rms_mean > 0.1:
            scores['awake'] += 1
        elif rms_mean < 0.02:
            scores['deep'] += 1
        else:
            scores['light'] += 1

        # Spectral flatness (noisy = awake)
        if flatness > 0.7:
            scores['awake'] += 1
        else:
            scores['light'] += 1
            scores['deep'] += 0.5

        # Determine stage
        stage = max(scores, key=scores.get)
        confidence = scores[stage] / sum(scores.values()) if sum(scores.values()) > 0 else 0.25

        return {
            'stage': stage,
            'confidence': round(confidence, 2),
            'scores': {k: round(v, 2) for k, v in scores.items()}
        }


class CorrelationAnalyzer:
    """
    Analyze correlations between different metrics.
    """

    def __init__(self):
        self.data_history = {
            'ahi': [],
            'duration': [],
            'snoring': [],
            'efficiency': []
        }

    def add_data(self, metrics: Dict):
        """Add metrics from a session"""
        for key in self.data_history:
            if key in metrics:
                self.data_history[key].append(metrics[key])

    def compute_correlations(self) -> Dict:
        """Compute pairwise correlations"""
        correlations = {}

        keys = list(self.data_history.keys())
        for i, key1 in enumerate(keys):
            for key2 in keys[i+1:]:
                if len(self.data_history[key1]) >= 5 and len(self.data_history[key2]) >= 5:
                    data1 = np.array(self.data_history[key1])
                    data2 = np.array(self.data_history[key2][:len(data1)])

                    if len(data1) == len(data2):
                        corr, p_value = stats.pearsonr(data1, data2)
                        correlations[f'{key1}_vs_{key2}'] = {
                            'correlation': round(corr, 3),
                            'p_value': round(p_value, 4),
                            'significant': p_value < 0.05
                        }

        return correlations

    def find_insights(self) -> List[str]:
        """Generate insights from correlations"""
        insights = []
        correlations = self.compute_correlations()

        for pair, data in correlations.items():
            if data['significant'] and abs(data['correlation']) > 0.5:
                keys = pair.split('_vs_')
                direction = 'positively' if data['correlation'] > 0 else 'negatively'
                insights.append(
                    f"{keys[0].replace('_', ' ').title()} is {direction} correlated with "
                    f"{keys[1].replace('_', ' ')} (r={data['correlation']:.2f})"
                )

        return insights


class RiskPredictor:
    """
    Predict risk of severe apnea based on historical data.
    """

    def __init__(self):
        self.history = []
        self.weights = {
            'ahi_trend': 0.3,
            'event_frequency': 0.25,
            'duration_change': 0.15,
            'consistency': 0.3
        }

    def add_session(self, session_data: Dict):
        """Add session data for prediction"""
        self.history.append(session_data)

    def predict_risk(self) -> Dict:
        """
        Predict risk of severe apnea in near future.

        Returns probability and contributing factors.
        """
        if len(self.history) < 5:
            return {'available': False, 'message': 'Need at least 5 sessions for prediction'}

        # Calculate risk factors
        factors = {}

        # AHI trend factor
        ahi_values = [s.get('ahi', 0) for s in self.history[-7:]]
        if len(ahi_values) >= 3:
            slope = np.polyfit(range(len(ahi_values)), ahi_values, 1)[0]
            factors['ahi_trend'] = min(slope / 2, 1)  # Normalize
        else:
            factors['ahi_trend'] = 0

        # Event frequency factor
        recent_events = sum(s.get('total_events', 0) for s in self.history[-7:])
        avg_events = recent_events / min(len(self.history), 7)
        factors['event_frequency'] = min(avg_events / 50, 1)  # Normalize to 50 events

        # Duration change factor
        durations = [s.get('duration', 7) for s in self.history[-7:]]
        if len(durations) >= 2:
            duration_std = np.std(durations)
            factors['duration_change'] = min(duration_std / 2, 1)
        else:
            factors['duration_change'] = 0

        # Consistency factor (inverse of variability)
        if len(ahi_values) >= 3:
            cv = np.std(ahi_values) / (np.mean(ahi_values) + 0.1)
            factors['consistency'] = min(cv, 1)
        else:
            factors['consistency'] = 0

        # Compute weighted risk score
        risk_score = sum(
            self.weights[k] * factors[k]
            for k in factors
        )

        # Determine risk level
        if risk_score > 0.7:
            risk_level = 'high'
        elif risk_score > 0.4:
            risk_level = 'moderate'
        else:
            risk_level = 'low'

        return {
            'available': True,
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'factors': {k: round(v, 2) for k, v in factors.items()},
            'contributing_factors': [
                k for k, v in sorted(factors.items(), key=lambda x: x[1], reverse=True)[:2]
            ]
        }


class SleepQualityCalculator:
    """
    Calculate overall sleep quality score from multiple metrics.
    """

    def __init__(self):
        self.weights = {
            'ahi_score': 0.35,
            'duration_score': 0.25,
            'efficiency_score': 0.2,
            'consistency_score': 0.2
        }

    def calculate(
        self,
        ahi: float,
        duration_hours: float,
        events: Dict,
        history: List[float] = None
    ) -> Dict:
        """
        Calculate comprehensive sleep quality score (0-100).
        """
        scores = {}

        # AHI score (lower is better)
        if ahi < 5:
            scores['ahi_score'] = 100
        elif ahi < 15:
            scores['ahi_score'] = 80 - (ahi - 5) * 2
        elif ahi < 30:
            scores['ahi_score'] = 60 - (ahi - 15)
        else:
            scores['ahi_score'] = max(0, 45 - (ahi - 30))

        # Duration score (7-9 hours optimal)
        if 7 <= duration_hours <= 9:
            scores['duration_score'] = 100
        elif 6 <= duration_hours < 7 or 9 < duration_hours <= 10:
            scores['duration_score'] = 80
        elif 5 <= duration_hours < 6 or 10 < duration_hours <= 11:
            scores['duration_score'] = 60
        else:
            scores['duration_score'] = 40

        # Efficiency score (based on normal vs total events)
        total_events = sum(events.values())
        normal_events = events.get('normal', 0)
        efficiency = normal_events / total_events if total_events > 0 else 1
        scores['efficiency_score'] = efficiency * 100

        # Consistency score (based on history)
        if history and len(history) >= 3:
            std = np.std(history)
            mean = np.mean(history)
            cv = std / (mean + 0.1)
            scores['consistency_score'] = max(0, 100 - cv * 100)
        else:
            scores['consistency_score'] = 70  # Default

        # Weighted total
        total_score = sum(
            self.weights[k] * scores[k]
            for k in scores
        )

        # Determine grade
        if total_score >= 85:
            grade = 'A'
        elif total_score >= 70:
            grade = 'B'
        elif total_score >= 55:
            grade = 'C'
        elif total_score >= 40:
            grade = 'D'
        else:
            grade = 'F'

        return {
            'total_score': round(total_score, 1),
            'grade': grade,
            'component_scores': {k: round(v, 1) for k, v in scores.items()},
            'weights': self.weights
        }
