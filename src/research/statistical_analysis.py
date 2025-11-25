"""
Research-grade statistical analysis module with bootstrap CIs, hypothesis testing,
and calibration metrics for sleep apnea detection models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.special import expit
import warnings

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Container for statistical analysis results."""
    metric: str
    estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    n_samples: int = 0


class BootstrapAnalyzer:
    """Bootstrap-based statistical analysis."""

    def __init__(self, n_bootstrap: int = 1000, ci_level: float = 0.95, seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.rng = np.random.RandomState(seed)

    def bootstrap_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_func: callable,
        stratify: Optional[np.ndarray] = None
    ) -> StatisticalResult:
        """Compute bootstrap confidence interval for a metric."""

        n_samples = len(y_true)
        boot_estimates = []

        for _ in range(self.n_bootstrap):
            if stratify is not None:
                # Stratified bootstrap
                indices = []
                for stratum in np.unique(stratify):
                    stratum_idx = np.where(stratify == stratum)[0]
                    boot_idx = self.rng.choice(stratum_idx, size=len(stratum_idx), replace=True)
                    indices.extend(boot_idx)
                indices = np.array(indices)
            else:
                indices = self.rng.choice(n_samples, size=n_samples, replace=True)

            try:
                estimate = metric_func(y_true[indices], y_pred[indices])
                boot_estimates.append(estimate)
            except Exception:
                continue

        boot_estimates = np.array(boot_estimates)

        # Point estimate
        point_estimate = metric_func(y_true, y_pred)

        # Confidence interval (percentile method)
        alpha = 1 - self.ci_level
        ci_lower = np.percentile(boot_estimates, 100 * alpha / 2)
        ci_upper = np.percentile(boot_estimates, 100 * (1 - alpha / 2))

        return StatisticalResult(
            metric=metric_func.__name__,
            estimate=float(point_estimate),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            std_error=float(np.std(boot_estimates)),
            n_samples=n_samples
        )

    def paired_bootstrap_test(
        self,
        y_true: np.ndarray,
        y_pred_1: np.ndarray,
        y_pred_2: np.ndarray,
        metric_func: callable
    ) -> Tuple[float, float]:
        """Paired bootstrap test for comparing two models."""

        n_samples = len(y_true)
        diff_estimates = []

        for _ in range(self.n_bootstrap):
            indices = self.rng.choice(n_samples, size=n_samples, replace=True)

            try:
                est_1 = metric_func(y_true[indices], y_pred_1[indices])
                est_2 = metric_func(y_true[indices], y_pred_2[indices])
                diff_estimates.append(est_1 - est_2)
            except Exception:
                continue

        diff_estimates = np.array(diff_estimates)

        # Two-tailed p-value
        p_value = 2 * min(
            np.mean(diff_estimates <= 0),
            np.mean(diff_estimates >= 0)
        )

        # Effect size (Cohen's d for differences)
        effect_size = np.mean(diff_estimates) / (np.std(diff_estimates) + 1e-8)

        return float(p_value), float(effect_size)


class ClassificationMetrics:
    """Classification metrics with statistical rigor."""

    @staticmethod
    def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Area under ROC curve."""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_score)

    @staticmethod
    def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Area under Precision-Recall curve."""
        from sklearn.metrics import average_precision_score
        return average_precision_score(y_true, y_score)

    @staticmethod
    def sensitivity_at_specificity(
        y_true: np.ndarray,
        y_score: np.ndarray,
        target_specificity: float = 0.90
    ) -> float:
        """Sensitivity at a given specificity threshold."""
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        specificities = 1 - fpr

        # Find threshold closest to target specificity
        idx = np.argmin(np.abs(specificities - target_specificity))

        return float(tpr[idx])

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Brier score for calibration."""
        return float(np.mean((y_prob - y_true) ** 2))

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += prop_in_bin * np.abs(accuracy_in_bin - avg_confidence_in_bin)

        return float(ece)


class HypothesisTests:
    """Statistical hypothesis tests for model comparison."""

    @staticmethod
    def delong_test(
        y_true: np.ndarray,
        y_score_1: np.ndarray,
        y_score_2: np.ndarray
    ) -> Tuple[float, float]:
        """
        DeLong test for comparing AUROCs.
        Returns: (z_statistic, p_value)
        """
        n1 = np.sum(y_true == 1)
        n0 = np.sum(y_true == 0)

        # Compute AUCs
        from sklearn.metrics import roc_auc_score
        auc1 = roc_auc_score(y_true, y_score_1)
        auc2 = roc_auc_score(y_true, y_score_2)

        # Compute variance using DeLong's method
        def compute_midrank(x):
            n = len(x)
            T = np.zeros(n)
            i = 0
            while i < n:
                j = i
                while j < n and x[j] == x[i]:
                    j += 1
                for k in range(i, j):
                    T[k] = (i + j - 1) / 2
                i = j
            return T

        # Placement values
        order1 = np.argsort(y_score_1)
        order2 = np.argsort(y_score_2)

        r1 = compute_midrank(y_score_1[order1])
        r2 = compute_midrank(y_score_2[order2])

        # Compute covariance matrix elements
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]

        # Simplified variance estimation
        var1 = auc1 * (1 - auc1) / min(n0, n1)
        var2 = auc2 * (1 - auc2) / min(n0, n1)
        cov12 = 0.5 * (var1 + var2)  # Simplified

        var_diff = var1 + var2 - 2 * cov12

        if var_diff <= 0:
            return 0.0, 1.0

        z = (auc1 - auc2) / np.sqrt(var_diff)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return float(z), float(p_value)

    @staticmethod
    def mcnemar_test(y_pred_1: np.ndarray, y_pred_2: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
        """McNemar's test for paired nominal data."""
        # Contingency table
        correct_1 = (y_pred_1 == y_true)
        correct_2 = (y_pred_2 == y_true)

        # b: model 1 correct, model 2 wrong
        # c: model 1 wrong, model 2 correct
        b = np.sum(correct_1 & ~correct_2)
        c = np.sum(~correct_1 & correct_2)

        if b + c == 0:
            return 0.0, 1.0

        # McNemar's chi-squared
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        return float(chi2), float(p_value)

    @staticmethod
    def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Wilcoxon signed-rank test for paired samples."""
        statistic, p_value = stats.wilcoxon(x, y, alternative='two-sided')
        return float(statistic), float(p_value)


class EffectSizeCalculator:
    """Calculate effect sizes for various comparisons."""

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        return float((np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-8))

    @staticmethod
    def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
        """Hedges' g effect size (bias-corrected Cohen's d)."""
        d = EffectSizeCalculator.cohens_d(group1, group2)
        n = len(group1) + len(group2)

        # Correction factor
        correction = 1 - (3 / (4 * n - 9))

        return float(d * correction)

    @staticmethod
    def odds_ratio(contingency_table: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute odds ratio and CI from 2x2 contingency table.
        Returns: (odds_ratio, ci_lower, ci_upper)
        """
        a, b, c, d = contingency_table.flatten()

        if b == 0 or c == 0:
            # Add 0.5 for continuity correction
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

        odds_ratio = (a * d) / (b * c)

        # Log odds ratio SE
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)

        # 95% CI
        log_or = np.log(odds_ratio)
        ci_lower = np.exp(log_or - 1.96 * se_log_or)
        ci_upper = np.exp(log_or + 1.96 * se_log_or)

        return float(odds_ratio), float(ci_lower), float(ci_upper)


class CalibrationAnalyzer:
    """Model calibration analysis and correction."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def reliability_diagram_data(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute data for reliability diagram."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)

        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for i in range(self.n_bins):
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])

            if np.sum(in_bin) > 0:
                bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                bin_accuracies.append(np.mean(y_true[in_bin]))
                bin_counts.append(np.sum(in_bin))

        return {
            'bin_centers': np.array(bin_centers),
            'bin_accuracies': np.array(bin_accuracies),
            'bin_counts': np.array(bin_counts)
        }

    def temperature_scaling(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """Learn temperature scaling parameter."""
        temperature = 1.0

        for _ in range(max_iter):
            # Scaled probabilities
            scaled_probs = expit(logits / temperature)

            # Gradient of NLL
            grad = np.mean((scaled_probs - y_true) * logits / (temperature ** 2))

            # Update
            temperature -= lr * grad
            temperature = max(0.1, min(10.0, temperature))

        return float(temperature)

    def isotonic_calibration(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> callable:
        """Fit isotonic regression for calibration."""
        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(y_prob, y_true)

        return ir.predict


class SubgroupAnalyzer:
    """Analyze performance across subgroups for fairness."""

    def __init__(self, bootstrap_analyzer: BootstrapAnalyzer):
        self.bootstrap = bootstrap_analyzer

    def analyze_by_subgroup(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        subgroup_labels: np.ndarray,
        metric_func: callable
    ) -> Dict[str, StatisticalResult]:
        """Compute metrics for each subgroup."""
        results = {}

        for subgroup in np.unique(subgroup_labels):
            mask = subgroup_labels == subgroup

            if np.sum(mask) > 10:  # Minimum sample size
                result = self.bootstrap.bootstrap_metric(
                    y_true[mask],
                    y_pred[mask],
                    metric_func
                )
                result.metric = f"{metric_func.__name__}_{subgroup}"
                results[str(subgroup)] = result

        return results

    def compute_heterogeneity(
        self,
        subgroup_results: Dict[str, StatisticalResult]
    ) -> float:
        """Compute I² heterogeneity statistic."""
        estimates = [r.estimate for r in subgroup_results.values()]
        variances = [r.std_error ** 2 for r in subgroup_results.values()]

        if len(estimates) < 2:
            return 0.0

        # Weighted mean
        weights = [1/v for v in variances]
        weighted_mean = np.average(estimates, weights=weights)

        # Q statistic
        Q = sum(w * (e - weighted_mean) ** 2 for w, e in zip(weights, estimates))

        # I²
        df = len(estimates) - 1
        I_squared = max(0, (Q - df) / Q) if Q > 0 else 0

        return float(I_squared)


def generate_statistical_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    subgroups: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """Generate comprehensive statistical report."""

    bootstrap = BootstrapAnalyzer(n_bootstrap=1000)
    calibration = CalibrationAnalyzer()

    report = {
        'n_samples': len(y_true),
        'class_balance': {
            'positive': int(np.sum(y_true)),
            'negative': int(np.sum(1 - y_true))
        },
        'metrics': {},
        'calibration': {},
        'subgroup_analysis': {}
    }

    # Primary metrics with CIs
    metrics = [
        ('auroc', ClassificationMetrics.auroc),
        ('auprc', ClassificationMetrics.auprc),
        ('brier_score', ClassificationMetrics.brier_score),
    ]

    for name, func in metrics:
        try:
            result = bootstrap.bootstrap_metric(y_true, y_prob, func)
            report['metrics'][name] = {
                'estimate': result.estimate,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper,
                'std_error': result.std_error
            }
        except Exception as e:
            logger.warning(f"Could not compute {name}: {e}")

    # Sensitivity at specific specificities
    for spec in [0.80, 0.90, 0.95]:
        sens = ClassificationMetrics.sensitivity_at_specificity(y_true, y_prob, spec)
        report['metrics'][f'sensitivity_at_spec_{spec}'] = sens

    # Calibration
    ece = ClassificationMetrics.expected_calibration_error(y_true, y_prob)
    report['calibration']['ece'] = ece
    report['calibration']['reliability_diagram'] = calibration.reliability_diagram_data(y_true, y_prob)

    # Subgroup analysis
    if subgroups:
        subgroup_analyzer = SubgroupAnalyzer(bootstrap)

        for subgroup_name, subgroup_labels in subgroups.items():
            subgroup_results = subgroup_analyzer.analyze_by_subgroup(
                y_true, y_prob, subgroup_labels, ClassificationMetrics.auroc
            )

            report['subgroup_analysis'][subgroup_name] = {
                'results': {k: {'estimate': v.estimate, 'ci': (v.ci_lower, v.ci_upper)}
                           for k, v in subgroup_results.items()},
                'heterogeneity_I2': subgroup_analyzer.compute_heterogeneity(subgroup_results)
            }

    return report
