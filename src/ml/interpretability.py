"""
Model Interpretability and Explainability for Sleep Apnea Classification
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance for classification decisions.
    Provides explanations for model predictions.
    """

    def __init__(self, n_features: int = 39):
        self.n_features = n_features

        # Feature names for interpretability
        self.feature_names = self._generate_feature_names()

        # Accumulated importance scores
        self.importance_scores = np.zeros(n_features)
        self.n_samples = 0

    def _generate_feature_names(self) -> List[str]:
        """Generate human-readable feature names"""
        names = []

        # MFCC features (13 + 13 delta + 13 delta-delta)
        for i in range(13):
            names.append(f'MFCC_{i+1}')
        for i in range(13):
            names.append(f'MFCC_delta_{i+1}')
        for i in range(13):
            names.append(f'MFCC_delta2_{i+1}')

        return names

    def compute_importance(
        self,
        features: np.ndarray,
        prediction: int,
        model_weights: np.ndarray
    ) -> Dict:
        """
        Compute feature importance for a single prediction.

        Uses gradient-based attribution (simplified).
        """
        # Simple importance: weight * feature value
        class_weights = model_weights[prediction]
        importance = np.abs(class_weights * features)

        # Normalize
        importance = importance / (np.sum(importance) + 1e-8)

        # Update accumulated scores
        self.importance_scores += importance
        self.n_samples += 1

        # Get top features
        top_indices = np.argsort(importance)[-5:][::-1]
        top_features = [
            {
                'name': self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}',
                'importance': float(importance[i]),
                'value': float(features[i])
            }
            for i in top_indices
        ]

        return {
            'prediction': prediction,
            'top_features': top_features,
            'importance_vector': importance.tolist()
        }

    def get_global_importance(self) -> List[Dict]:
        """Get global feature importance across all samples"""
        if self.n_samples == 0:
            return []

        avg_importance = self.importance_scores / self.n_samples

        # Sort by importance
        sorted_indices = np.argsort(avg_importance)[::-1]

        return [
            {
                'name': self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}',
                'importance': float(avg_importance[i]),
                'rank': rank + 1
            }
            for rank, i in enumerate(sorted_indices[:10])
        ]


class PredictionExplainer:
    """
    Generate human-readable explanations for predictions.
    """

    def __init__(self):
        self.event_names = ['normal breathing', 'snoring', 'hypopnea', 'apnea']

        # Explanation templates
        self.templates = {
            0: [
                "Regular breathing pattern detected with normal energy levels.",
                "Audio shows consistent rhythm typical of healthy sleep."
            ],
            1: [
                "Strong low-frequency vibrations (20-300 Hz) indicate snoring.",
                "Periodic breathing sounds with elevated low-band energy."
            ],
            2: [
                "Reduced airflow detected - partial airway obstruction.",
                "Energy levels dropped to {energy_pct}% of normal baseline."
            ],
            3: [
                "Breathing cessation detected for extended period.",
                "Very low energy across all frequency bands indicates apnea event."
            ]
        }

    def explain(
        self,
        prediction: int,
        confidence: float,
        features: Dict,
        importance: Dict = None
    ) -> Dict:
        """
        Generate explanation for a prediction.

        Args:
            prediction: Predicted class
            confidence: Prediction confidence
            features: Extracted features
            importance: Optional feature importance

        Returns:
            Explanation dictionary
        """
        event_name = self.event_names[prediction]

        # Get template explanation
        base_explanation = self.templates[prediction][0]

        # Add feature-specific details
        details = []

        # Energy analysis
        if 'energy' in features:
            rms = features['energy'].get('rms_mean', 0)
            if prediction == 3 and rms < 0.01:
                details.append(f"RMS energy extremely low ({rms:.4f})")
            elif prediction == 1:
                details.append(f"Elevated energy from snoring vibrations")

        # Snoring analysis
        if 'snoring' in features:
            snore_score = features['snoring'].get('snore_score', 0)
            if prediction == 1:
                details.append(f"Snore score: {snore_score:.2f} (threshold: 2.5)")

        # Confidence interpretation
        confidence_level = 'high' if confidence > 0.8 else 'moderate' if confidence > 0.6 else 'low'

        explanation = {
            'event': event_name,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'summary': base_explanation,
            'details': details,
            'recommendation': self._get_recommendation(prediction)
        }

        # Add top features if available
        if importance and 'top_features' in importance:
            explanation['key_features'] = importance['top_features'][:3]

        return explanation

    def _get_recommendation(self, prediction: int) -> str:
        """Get recommendation based on prediction"""
        recommendations = {
            0: "Continue monitoring. Normal breathing patterns detected.",
            1: "Snoring detected. Consider sleep position or nasal strips.",
            2: "Partial obstruction detected. May indicate developing apnea.",
            3: "Apnea event detected. Consult specialist if frequent."
        }
        return recommendations.get(prediction, "")


class SessionSummarizer:
    """
    Generate comprehensive summaries of sleep sessions.
    """

    def __init__(self):
        self.explainer = PredictionExplainer()

    def summarize_session(
        self,
        events_timeline: List[int],
        ahi_stats: Dict,
        duration_hours: float
    ) -> Dict:
        """
        Generate session summary with insights.

        Args:
            events_timeline: List of event classifications
            ahi_stats: AHI statistics
            duration_hours: Total duration in hours

        Returns:
            Comprehensive session summary
        """
        # Event counts
        event_counts = {
            'normal': events_timeline.count(0),
            'snoring': events_timeline.count(1),
            'hypopnea': events_timeline.count(2),
            'apnea': events_timeline.count(3)
        }

        # Calculate percentages
        total = len(events_timeline)
        event_percentages = {
            k: round(v / total * 100, 1) if total > 0 else 0
            for k, v in event_counts.items()
        }

        # Find patterns
        patterns = self._detect_patterns(events_timeline)

        # Generate insights
        insights = self._generate_insights(event_counts, ahi_stats, patterns)

        # Risk assessment
        risk_level = self._assess_risk(ahi_stats['overall_ahi'], patterns)

        return {
            'duration_hours': round(duration_hours, 2),
            'ahi': ahi_stats['overall_ahi'],
            'severity': ahi_stats['severity'],
            'event_counts': event_counts,
            'event_percentages': event_percentages,
            'patterns': patterns,
            'insights': insights,
            'risk_level': risk_level,
            'recommendation': self._get_session_recommendation(risk_level, ahi_stats)
        }

    def _detect_patterns(self, events: List[int]) -> Dict:
        """Detect patterns in event timeline"""
        patterns = {
            'longest_apnea_streak': 0,
            'apnea_clusters': 0,
            'snoring_duration': 0
        }

        current_streak = 0
        cluster_count = 0
        in_cluster = False

        for i, event in enumerate(events):
            if event == 3:  # Apnea
                current_streak += 1
                patterns['longest_apnea_streak'] = max(
                    patterns['longest_apnea_streak'],
                    current_streak
                )
                if not in_cluster:
                    cluster_count += 1
                    in_cluster = True
            else:
                current_streak = 0
                if event != 2:  # Not hypopnea either
                    in_cluster = False

            if event == 1:
                patterns['snoring_duration'] += 0.5  # 30-second segments

        patterns['apnea_clusters'] = cluster_count

        return patterns

    def _generate_insights(
        self,
        counts: Dict,
        ahi_stats: Dict,
        patterns: Dict
    ) -> List[str]:
        """Generate actionable insights"""
        insights = []

        # AHI-based insights
        ahi = ahi_stats['overall_ahi']
        if ahi >= 30:
            insights.append("Severe sleep apnea detected. Urgent medical consultation recommended.")
        elif ahi >= 15:
            insights.append("Moderate sleep apnea detected. Schedule appointment with sleep specialist.")
        elif ahi >= 5:
            insights.append("Mild sleep apnea present. Consider lifestyle modifications.")

        # Pattern-based insights
        if patterns['longest_apnea_streak'] >= 3:
            insights.append(f"Detected {patterns['longest_apnea_streak']} consecutive apnea events - consider CPAP evaluation.")

        if patterns['snoring_duration'] > 60:
            insights.append("Extended snoring periods detected. Positional therapy may help.")

        # Trend insights
        if ahi_stats.get('trend') == 'increasing':
            insights.append("AHI trending upward during the night - possible sleep position issue.")

        return insights

    def _assess_risk(self, ahi: float, patterns: Dict) -> str:
        """Assess overall risk level"""
        score = 0

        # AHI contribution
        if ahi >= 30:
            score += 4
        elif ahi >= 15:
            score += 3
        elif ahi >= 5:
            score += 2
        else:
            score += 1

        # Pattern contribution
        if patterns['longest_apnea_streak'] >= 5:
            score += 2
        elif patterns['longest_apnea_streak'] >= 3:
            score += 1

        if score >= 5:
            return 'high'
        elif score >= 3:
            return 'moderate'
        else:
            return 'low'

    def _get_session_recommendation(self, risk: str, stats: Dict) -> str:
        """Get session-specific recommendation"""
        if risk == 'high':
            return "Please consult a sleep specialist promptly. Bring this report to your appointment."
        elif risk == 'moderate':
            return "Schedule a follow-up with your doctor. Continue monitoring and try sleep position changes."
        else:
            return "Continue monitoring your sleep. Practice good sleep hygiene and maintain healthy habits."
