"""
Hyperparameter optimization infrastructure using Optuna and Ray Tune.
Implements massive distributed hyperparameter search with early stopping.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging
import json
from pathlib import Path
import time

logger = logging.getLogger(__name__)

# Lazy imports
def get_optuna():
    import optuna
    return optuna

def get_ray_tune():
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
    from ray.tune.search.optuna import OptunaSearch
    return tune, ASHAScheduler, HyperBandScheduler, OptunaSearch


@dataclass
class HyperoptConfig:
    """Configuration for hyperparameter optimization."""
    n_trials: int = 1000
    timeout: int = 86400  # 24 hours
    n_jobs: int = 4
    pruning: bool = True
    study_name: str = "sleep_apnea_hyperopt"
    storage: str = "sqlite:///research/hyperopt/optuna.db"
    direction: str = "maximize"  # or "minimize"
    sampler: str = "tpe"  # tpe, cmaes, random
    early_stopping_patience: int = 10


class SearchSpace:
    """Define hyperparameter search spaces for different model families."""

    @staticmethod
    def cnn_spectrogram():
        """Search space for CNN on spectrograms."""
        return {
            'n_filters_1': ('int', 32, 128),
            'n_filters_2': ('int', 64, 256),
            'n_filters_3': ('int', 128, 512),
            'kernel_size': ('categorical', [3, 5, 7]),
            'pool_size': ('categorical', [2, 3]),
            'dropout': ('float', 0.1, 0.5),
            'fc_dim': ('int', 128, 512),
            'learning_rate': ('loguniform', 1e-5, 1e-2),
            'weight_decay': ('loguniform', 1e-6, 1e-3),
            'batch_size': ('categorical', [16, 32, 64, 128]),
            'n_mels': ('categorical', [64, 128, 256]),
            'optimizer': ('categorical', ['adam', 'adamw', 'sgd']),
            'scheduler': ('categorical', ['cosine', 'step', 'plateau']),
        }

    @staticmethod
    def transformer_audio():
        """Search space for Transformer on audio."""
        return {
            'hidden_dim': ('categorical', [256, 512, 768, 1024]),
            'num_layers': ('int', 4, 12),
            'num_heads': ('categorical', [4, 8, 12, 16]),
            'ff_dim_mult': ('categorical', [2, 4]),
            'dropout': ('float', 0.1, 0.3),
            'learning_rate': ('loguniform', 1e-5, 1e-3),
            'warmup_steps': ('int', 1000, 10000),
            'weight_decay': ('loguniform', 1e-6, 1e-2),
            'batch_size': ('categorical', [8, 16, 32]),
            'max_length': ('categorical', [512, 1024, 2048]),
            'positional_encoding': ('categorical', ['sinusoidal', 'learned', 'rotary']),
        }

    @staticmethod
    def ensemble():
        """Search space for ensemble methods."""
        return {
            'n_estimators': ('int', 10, 100),
            'base_model': ('categorical', ['cnn', 'transformer', 'hybrid']),
            'aggregation': ('categorical', ['voting', 'stacking', 'bagging']),
            'diversity_weight': ('float', 0.0, 1.0),
            'temperature': ('float', 0.5, 2.0),
        }

    @staticmethod
    def ssl_pretraining():
        """Search space for self-supervised pretraining."""
        return {
            'hidden_dim': ('categorical', [512, 768, 1024]),
            'num_layers': ('int', 6, 12),
            'mask_prob': ('float', 0.05, 0.15),
            'mask_length': ('int', 5, 20),
            'num_negatives': ('int', 50, 200),
            'temperature': ('float', 0.05, 0.2),
            'learning_rate': ('loguniform', 1e-5, 1e-3),
            'warmup_ratio': ('float', 0.05, 0.15),
        }


class OptunaOptimizer:
    """Optuna-based hyperparameter optimizer."""

    def __init__(self, config: HyperoptConfig):
        self.config = config
        self.optuna = get_optuna()

        # Create sampler
        if config.sampler == "tpe":
            sampler = self.optuna.samplers.TPESampler(seed=42)
        elif config.sampler == "cmaes":
            sampler = self.optuna.samplers.CmaEsSampler(seed=42)
        else:
            sampler = self.optuna.samplers.RandomSampler(seed=42)

        # Create pruner
        if config.pruning:
            pruner = self.optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        else:
            pruner = self.optuna.pruners.NopPruner()

        self.study = self.optuna.create_study(
            study_name=config.study_name,
            storage=config.storage,
            direction=config.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )

    def _sample_params(self, trial, search_space: Dict) -> Dict:
        """Sample hyperparameters from search space."""
        params = {}

        for name, spec in search_space.items():
            param_type = spec[0]

            if param_type == 'int':
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif param_type == 'float':
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif param_type == 'loguniform':
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif param_type == 'categorical':
                params[name] = trial.suggest_categorical(name, spec[1])

        return params

    def optimize(
        self,
        objective: Callable,
        search_space: Dict,
        callbacks: Optional[List] = None
    ) -> Dict:
        """Run hyperparameter optimization."""

        def wrapped_objective(trial):
            params = self._sample_params(trial, search_space)
            return objective(params, trial)

        self.study.optimize(
            wrapped_objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            callbacks=callbacks,
            show_progress_bar=True
        )

        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'best_trial': self.study.best_trial.number,
            'n_trials': len(self.study.trials)
        }

    def get_top_trials(self, n: int = 20) -> List[Dict]:
        """Get top N trials."""
        trials = sorted(
            self.study.trials,
            key=lambda t: t.value if t.value is not None else float('-inf'),
            reverse=(self.config.direction == "maximize")
        )

        return [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'datetime': str(t.datetime_start)
            }
            for t in trials[:n]
        ]

    def export_results(self, output_dir: str) -> str:
        """Export optimization results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export trials to JSON
        trials_data = []
        for trial in self.study.trials:
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state),
                'datetime_start': str(trial.datetime_start),
                'datetime_complete': str(trial.datetime_complete) if trial.datetime_complete else None,
                'duration': trial.duration.total_seconds() if trial.duration else None
            })

        trials_file = output_path / 'trials.json'
        with open(trials_file, 'w') as f:
            json.dump(trials_data, f, indent=2)

        # Export best params
        best_file = output_path / 'best_params.json'
        with open(best_file, 'w') as f:
            json.dump({
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'best_trial': self.study.best_trial.number
            }, f, indent=2)

        # Export importance
        try:
            importance = self.optuna.importance.get_param_importances(self.study)
            importance_file = output_path / 'param_importance.json'
            with open(importance_file, 'w') as f:
                json.dump(importance, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not compute parameter importance: {e}")

        logger.info(f"Exported results to {output_path}")
        return str(output_path)


class RayTuneOptimizer:
    """Ray Tune-based distributed hyperparameter optimizer."""

    def __init__(self, config: HyperoptConfig):
        self.config = config
        self.tune, self.ASHAScheduler, self.HyperBandScheduler, self.OptunaSearch = get_ray_tune()

    def _convert_search_space(self, search_space: Dict) -> Dict:
        """Convert search space to Ray Tune format."""
        tune_space = {}

        for name, spec in search_space.items():
            param_type = spec[0]

            if param_type == 'int':
                tune_space[name] = self.tune.randint(spec[1], spec[2])
            elif param_type == 'float':
                tune_space[name] = self.tune.uniform(spec[1], spec[2])
            elif param_type == 'loguniform':
                tune_space[name] = self.tune.loguniform(spec[1], spec[2])
            elif param_type == 'categorical':
                tune_space[name] = self.tune.choice(spec[1])

        return tune_space

    def optimize(
        self,
        trainable: Callable,
        search_space: Dict,
        metric: str = "val_loss",
        mode: str = "min",
        num_samples: int = 1000,
        max_epochs: int = 100,
        gpus_per_trial: float = 1.0,
        cpus_per_trial: int = 4
    ) -> Dict:
        """Run distributed hyperparameter optimization with Ray Tune."""

        tune_space = self._convert_search_space(search_space)

        # ASHA scheduler for early stopping
        scheduler = self.ASHAScheduler(
            metric=metric,
            mode=mode,
            max_t=max_epochs,
            grace_period=10,
            reduction_factor=3
        )

        # Optuna search algorithm
        search_alg = self.OptunaSearch(metric=metric, mode=mode)

        analysis = self.tune.run(
            trainable,
            config=tune_space,
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial={
                "cpu": cpus_per_trial,
                "gpu": gpus_per_trial
            },
            local_dir="research/hyperopt/ray_results",
            name=self.config.study_name,
            verbose=1,
            stop={"training_iteration": max_epochs}
        )

        return {
            'best_config': analysis.best_config,
            'best_result': analysis.best_result,
            'results_df': analysis.results_df
        }


class ModelObjective:
    """Objective functions for different model types."""

    def __init__(self, train_loader, val_loader, device: str = 'cuda'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def cnn_objective(self, params: Dict, trial=None) -> float:
        """Objective function for CNN hyperparameter search."""

        # Build model with sampled params
        model = self._build_cnn(params)
        model = model.to(self.device)

        # Setup training
        if params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        elif params['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=params['learning_rate'],
                momentum=0.9
            )

        criterion = nn.CrossEntropyLoss()

        # Training loop with early stopping
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(100):
            # Train
            model.train()
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_x, batch_y in self.val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = model(batch_x)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(batch_y).sum().item()
                    total += batch_y.size(0)

            val_acc = correct / total

            # Report intermediate result for pruning
            if trial is not None:
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    raise get_optuna().TrialPruned()

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        return best_val_acc

    def _build_cnn(self, params: Dict) -> nn.Module:
        """Build CNN model from parameters."""

        class CNN(nn.Module):
            def __init__(self, params):
                super().__init__()

                self.features = nn.Sequential(
                    nn.Conv2d(1, params['n_filters_1'], params['kernel_size'], padding=params['kernel_size']//2),
                    nn.BatchNorm2d(params['n_filters_1']),
                    nn.ReLU(),
                    nn.MaxPool2d(params['pool_size']),

                    nn.Conv2d(params['n_filters_1'], params['n_filters_2'], params['kernel_size'], padding=params['kernel_size']//2),
                    nn.BatchNorm2d(params['n_filters_2']),
                    nn.ReLU(),
                    nn.MaxPool2d(params['pool_size']),

                    nn.Conv2d(params['n_filters_2'], params['n_filters_3'], params['kernel_size'], padding=params['kernel_size']//2),
                    nn.BatchNorm2d(params['n_filters_3']),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )

                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(params['n_filters_3'], params['fc_dim']),
                    nn.ReLU(),
                    nn.Dropout(params['dropout']),
                    nn.Linear(params['fc_dim'], 4)  # 4 classes
                )

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x

        return CNN(params)


def run_hyperopt_study(
    model_family: str,
    train_loader,
    val_loader,
    n_trials: int = 1000,
    device: str = 'cuda'
) -> Dict:
    """Run hyperparameter optimization study for a model family."""

    config = HyperoptConfig(
        n_trials=n_trials,
        study_name=f"sleep_apnea_{model_family}",
        storage=f"sqlite:///research/hyperopt/{model_family}_optuna.db"
    )

    optimizer = OptunaOptimizer(config)
    objective = ModelObjective(train_loader, val_loader, device)

    # Get search space for model family
    if model_family == "cnn":
        search_space = SearchSpace.cnn_spectrogram()
        obj_func = objective.cnn_objective
    elif model_family == "transformer":
        search_space = SearchSpace.transformer_audio()
        obj_func = objective.transformer_objective
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    results = optimizer.optimize(obj_func, search_space)

    # Export results
    output_dir = f"research/hyperopt/{model_family}_results"
    optimizer.export_results(output_dir)

    return results
