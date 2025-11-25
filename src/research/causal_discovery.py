"""
Causal discovery and ablation study modules.
Implements graphical models, do-calculus, and exhaustive ablations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from itertools import combinations
import json

logger = logging.getLogger(__name__)


@dataclass
class CausalGraph:
    """Represents a causal graph structure."""
    nodes: List[str]
    edges: List[Tuple[str, str]]  # (cause, effect)
    confounders: Dict[str, List[str]] = None  # node -> confounders


class PCAlgorithm:
    """PC algorithm for causal structure learning."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def _partial_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> float:
        """Compute partial correlation between x and y given z."""
        from scipy import stats

        if z.shape[1] == 0:
            return stats.pearsonr(x, y)[0]

        # Residualize
        from sklearn.linear_model import LinearRegression

        reg_x = LinearRegression().fit(z, x)
        reg_y = LinearRegression().fit(z, y)

        resid_x = x - reg_x.predict(z)
        resid_y = y - reg_y.predict(z)

        return stats.pearsonr(resid_x, resid_y)[0]

    def _conditional_independence_test(
        self,
        data: np.ndarray,
        i: int,
        j: int,
        conditioning_set: List[int]
    ) -> float:
        """Test conditional independence between variables i and j given conditioning set."""
        from scipy import stats

        n = data.shape[0]
        x = data[:, i]
        y = data[:, j]

        if conditioning_set:
            z = data[:, conditioning_set]
        else:
            z = np.zeros((n, 0))

        r = self._partial_correlation(x, y, z)

        # Fisher's z-transformation
        z_stat = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
        se = 1 / np.sqrt(n - len(conditioning_set) - 3)

        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat / se)))

        return p_value

    def fit(self, data: np.ndarray, var_names: List[str]) -> CausalGraph:
        """Learn causal structure from data using PC algorithm."""
        n_vars = data.shape[1]

        # Initialize complete graph
        adj = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        separating_sets = {}

        # Phase 1: Edge removal
        for cond_size in range(n_vars):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if adj[i, j] == 0:
                        continue

                    # Get adjacent nodes excluding i and j
                    adjacent = [k for k in range(n_vars)
                               if k != i and k != j and (adj[i, k] == 1 or adj[j, k] == 1)]

                    if len(adjacent) < cond_size:
                        continue

                    # Test all conditioning sets of size cond_size
                    for cond_set in combinations(adjacent, cond_size):
                        p_value = self._conditional_independence_test(data, i, j, list(cond_set))

                        if p_value > self.alpha:
                            adj[i, j] = 0
                            adj[j, i] = 0
                            separating_sets[(i, j)] = set(cond_set)
                            separating_sets[(j, i)] = set(cond_set)
                            break

        # Phase 2: Orient edges (simplified v-structure detection)
        edges = []
        for i in range(n_vars):
            for j in range(n_vars):
                if adj[i, j] == 1:
                    edges.append((var_names[i], var_names[j]))

        return CausalGraph(nodes=var_names, edges=edges)


class DoCalculus:
    """Implement do-calculus for causal effect estimation."""

    def __init__(self, graph: CausalGraph):
        self.graph = graph
        self._build_adjacency()

    def _build_adjacency(self):
        """Build adjacency structures."""
        self.parents = {node: [] for node in self.graph.nodes}
        self.children = {node: [] for node in self.graph.nodes}

        for cause, effect in self.graph.edges:
            self.parents[effect].append(cause)
            self.children[cause].append(effect)

    def get_backdoor_adjustment_set(
        self,
        treatment: str,
        outcome: str
    ) -> Optional[List[str]]:
        """Find adjustment set using backdoor criterion."""
        # Simple implementation: adjust for all parents of treatment
        adjustment_set = self.parents[treatment].copy()

        # Remove descendants of treatment
        descendants = self._get_descendants(treatment)
        adjustment_set = [v for v in adjustment_set if v not in descendants]

        return adjustment_set if adjustment_set else None

    def _get_descendants(self, node: str) -> set:
        """Get all descendants of a node."""
        descendants = set()
        queue = [node]

        while queue:
            current = queue.pop(0)
            for child in self.children.get(current, []):
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)

        return descendants

    def estimate_ate(
        self,
        data: Dict[str, np.ndarray],
        treatment: str,
        outcome: str,
        method: str = 'backdoor'
    ) -> Tuple[float, float]:
        """
        Estimate Average Treatment Effect.
        Returns: (ATE, standard error)
        """
        if method == 'backdoor':
            adjustment_set = self.get_backdoor_adjustment_set(treatment, outcome)

            if adjustment_set is None:
                # No adjustment needed
                treated = data[outcome][data[treatment] == 1]
                control = data[outcome][data[treatment] == 0]
                ate = np.mean(treated) - np.mean(control)
                se = np.sqrt(np.var(treated)/len(treated) + np.var(control)/len(control))
            else:
                # Regression adjustment
                from sklearn.linear_model import LinearRegression

                X = np.column_stack([data[treatment]] + [data[v] for v in adjustment_set])
                y = data[outcome]

                reg = LinearRegression().fit(X, y)
                ate = reg.coef_[0]

                # Bootstrap SE
                n_bootstrap = 1000
                ates = []
                n = len(y)

                for _ in range(n_bootstrap):
                    idx = np.random.choice(n, n, replace=True)
                    reg_boot = LinearRegression().fit(X[idx], y[idx])
                    ates.append(reg_boot.coef_[0])

                se = np.std(ates)

            return float(ate), float(se)

        elif method == 'ipw':
            # Inverse Probability Weighting
            from sklearn.linear_model import LogisticRegression

            adjustment_set = self.get_backdoor_adjustment_set(treatment, outcome)
            if adjustment_set:
                X_ps = np.column_stack([data[v] for v in adjustment_set])
            else:
                X_ps = np.ones((len(data[treatment]), 1))

            # Propensity score
            ps_model = LogisticRegression(max_iter=1000).fit(X_ps, data[treatment])
            ps = ps_model.predict_proba(X_ps)[:, 1]
            ps = np.clip(ps, 0.01, 0.99)

            # IPW estimator
            T = data[treatment]
            Y = data[outcome]

            ate = np.mean(T * Y / ps) - np.mean((1 - T) * Y / (1 - ps))

            # Bootstrap SE
            n_bootstrap = 1000
            ates = []
            n = len(Y)

            for _ in range(n_bootstrap):
                idx = np.random.choice(n, n, replace=True)
                ate_boot = np.mean(T[idx] * Y[idx] / ps[idx]) - np.mean((1 - T[idx]) * Y[idx] / (1 - ps[idx]))
                ates.append(ate_boot)

            se = np.std(ates)

            return float(ate), float(se)

        else:
            raise ValueError(f"Unknown method: {method}")


class AblationStudy:
    """Exhaustive ablation studies across augmentations, representations, and model depths."""

    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.results = []

    def define_ablations(self) -> Dict[str, List[Any]]:
        """Define ablation dimensions."""
        return {
            # Augmentations
            'use_time_stretch': [True, False],
            'use_pitch_shift': [True, False],
            'use_noise': [True, False],
            'use_reverb': [True, False],

            # Input representations
            'input_type': ['mel_64', 'mel_128', 'mel_256', 'cqt', 'mfcc', 'raw'],

            # Model depths
            'num_layers': [4, 6, 8, 12, 16, 24],

            # Other hyperparameters
            'hidden_dim': [256, 512, 768, 1024],
            'dropout': [0.1, 0.2, 0.3],
        }

    def run_single_ablation(
        self,
        config: Dict[str, Any],
        train_func,
        eval_func
    ) -> Dict[str, Any]:
        """Run a single ablation experiment."""
        # Train model
        model = train_func(config)

        # Evaluate
        metrics = eval_func(model)

        return {
            'config': config,
            'metrics': metrics
        }

    def run_full_study(
        self,
        train_func,
        eval_func,
        n_seeds: int = 3
    ) -> List[Dict[str, Any]]:
        """Run full ablation study."""
        ablations = self.define_ablations()
        results = []

        # Single-factor ablations
        for factor, values in ablations.items():
            for value in values:
                for seed in range(n_seeds):
                    config = self.base_config.copy()
                    config[factor] = value
                    config['seed'] = seed

                    result = self.run_single_ablation(config, train_func, eval_func)
                    result['ablation_factor'] = factor
                    result['ablation_value'] = value
                    result['seed'] = seed

                    results.append(result)
                    self.results.append(result)

        return results

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze ablation study results."""
        import pandas as pd

        df = pd.DataFrame([
            {
                **r['config'],
                **r['metrics'],
                'ablation_factor': r.get('ablation_factor'),
                'ablation_value': r.get('ablation_value')
            }
            for r in self.results
        ])

        analysis = {
            'summary': {},
            'factor_importance': {}
        }

        # Summarize each factor's effect
        for factor in df['ablation_factor'].unique():
            if pd.isna(factor):
                continue

            factor_df = df[df['ablation_factor'] == factor]

            # Group by value and compute mean metrics
            summary = factor_df.groupby('ablation_value').agg({
                'accuracy': ['mean', 'std'],
                'auroc': ['mean', 'std']
            }).round(4)

            analysis['summary'][factor] = summary.to_dict()

            # Compute importance (variance explained)
            total_var = df['accuracy'].var()
            between_var = factor_df.groupby('ablation_value')['accuracy'].mean().var()
            importance = between_var / (total_var + 1e-8)

            analysis['factor_importance'][factor] = float(importance)

        return analysis

    def generate_latex_tables(self) -> str:
        """Generate LaTeX tables for ablation results."""
        analysis = self.analyze_results()

        latex = "\\begin{table}[h]\n\\centering\n"
        latex += "\\caption{Ablation Study Results}\n"
        latex += "\\begin{tabular}{llcc}\n\\hline\n"
        latex += "Factor & Value & Accuracy & AUROC \\\\ \\hline\n"

        for factor, summary in analysis['summary'].items():
            for value in summary.get(('accuracy', 'mean'), {}).keys():
                acc_mean = summary[('accuracy', 'mean')].get(value, 0)
                acc_std = summary[('accuracy', 'std')].get(value, 0)
                auroc_mean = summary[('auroc', 'mean')].get(value, 0)
                auroc_std = summary[('auroc', 'std')].get(value, 0)

                latex += f"{factor} & {value} & {acc_mean:.3f} $\\pm$ {acc_std:.3f} & {auroc_mean:.3f} $\\pm$ {auroc_std:.3f} \\\\\n"

        latex += "\\hline\n\\end{tabular}\n"

        # Factor importance table
        latex += "\n\\vspace{1em}\n"
        latex += "\\begin{tabular}{lc}\n\\hline\n"
        latex += "Factor & Importance \\\\ \\hline\n"

        for factor, importance in sorted(
            analysis['factor_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            latex += f"{factor} & {importance:.4f} \\\\\n"

        latex += "\\hline\n\\end{tabular}\n\\end{table}"

        return latex

    def export_results(self, output_dir: str):
        """Export ablation results to files."""
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # JSON results
        with open(output_path / 'ablation_results.json', 'w') as f:
            # Convert numpy types to Python types
            serializable_results = []
            for r in self.results:
                sr = {
                    'config': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                              for k, v in r['config'].items()},
                    'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                               for k, v in r['metrics'].items()},
                    'ablation_factor': r.get('ablation_factor'),
                    'ablation_value': r.get('ablation_value'),
                    'seed': r.get('seed')
                }
                serializable_results.append(sr)

            json.dump(serializable_results, f, indent=2)

        # LaTeX tables
        latex = self.generate_latex_tables()
        with open(output_path / 'ablation_tables.tex', 'w') as f:
            f.write(latex)

        # Analysis summary
        analysis = self.analyze_results()
        with open(output_path / 'ablation_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        logger.info(f"Exported ablation results to {output_path}")


class CausalSleepAnalyzer:
    """Analyze causal relationships between sleep audio features and outcomes."""

    def __init__(self):
        self.variables = [
            'snore_intensity',
            'breathing_pause_duration',
            'spectral_entropy',
            'heart_rate_variability',
            'oxygen_saturation',
            'sleep_quality',
            'ahi_score',
            'cardiovascular_risk'
        ]

    def estimate_effects(
        self,
        data: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Estimate causal effects between variables."""

        # Define expected causal structure based on domain knowledge
        edges = [
            ('snore_intensity', 'ahi_score'),
            ('breathing_pause_duration', 'ahi_score'),
            ('breathing_pause_duration', 'oxygen_saturation'),
            ('ahi_score', 'sleep_quality'),
            ('oxygen_saturation', 'sleep_quality'),
            ('ahi_score', 'cardiovascular_risk'),
            ('heart_rate_variability', 'cardiovascular_risk'),
        ]

        graph = CausalGraph(nodes=self.variables, edges=edges)
        do_calc = DoCalculus(graph)

        effects = {}

        # Estimate key causal effects
        treatment_outcomes = [
            ('snore_intensity', 'ahi_score'),
            ('breathing_pause_duration', 'oxygen_saturation'),
            ('ahi_score', 'cardiovascular_risk'),
        ]

        for treatment, outcome in treatment_outcomes:
            if treatment in data and outcome in data:
                ate, se = do_calc.estimate_ate(data, treatment, outcome)
                p_value = 2 * (1 - abs(ate / (se + 1e-8)))

                effects[f"{treatment}_to_{outcome}"] = {
                    'ate': ate,
                    'se': se,
                    'ci_lower': ate - 1.96 * se,
                    'ci_upper': ate + 1.96 * se,
                    'p_value': min(1.0, max(0.0, p_value))
                }

        return effects
