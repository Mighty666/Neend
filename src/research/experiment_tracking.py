"""
Experiment tracking with MLflow and Weights & Biases integration.
Tracks hyperparameters, code commits, artifacts, and metrics.
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
from datetime import datetime
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    tracking_uri: str = "mlruns"
    experiment_name: str = "sleep_apnea_detection"
    use_mlflow: bool = True
    use_wandb: bool = False
    wandb_project: str = "neendai"
    wandb_entity: Optional[str] = None
    log_system_metrics: bool = True
    log_code: bool = True
    auto_log_models: bool = True


@dataclass
class RunConfig:
    """Configuration for a single experiment run."""
    run_name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    dataset_version: str
    random_seed: int = 42
    tags: Dict[str, str] = field(default_factory=dict)
    notes: str = ""


class ExperimentTracker:
    """Unified experiment tracking interface."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.mlflow = None
        self.wandb = None
        self.run = None

        if config.use_mlflow:
            self._init_mlflow()
        if config.use_wandb:
            self._init_wandb()

    def _init_mlflow(self):
        """Initialize MLflow tracking."""
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(self.config.tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            mlflow.create_experiment(self.config.experiment_name)

        mlflow.set_experiment(self.config.experiment_name)

        if self.config.auto_log_models:
            mlflow.pytorch.autolog(log_models=True)

        logger.info(f"MLflow initialized with tracking URI: {self.config.tracking_uri}")

    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        import wandb

        self.wandb = wandb
        logger.info(f"W&B initialized for project: {self.config.wandb_project}")

    def _get_git_info(self) -> Dict[str, str]:
        """Get current git commit info."""
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=Path(__file__).parent.parent.parent
            ).decode().strip()

            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=Path(__file__).parent.parent.parent
            ).decode().strip()

            return {'git_commit': commit, 'git_branch': branch}
        except Exception:
            return {'git_commit': 'unknown', 'git_branch': 'unknown'}

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform

        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'hostname': platform.node()
        }

        # GPU info
        try:
            import torch
            if torch.cuda.is_available():
                info['gpu_count'] = torch.cuda.device_count()
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['cuda_version'] = torch.version.cuda
        except Exception:
            pass

        return info

    def start_run(self, run_config: RunConfig) -> str:
        """Start a new experiment run."""
        run_id = None

        if self.mlflow:
            self.run = self.mlflow.start_run(run_name=run_config.run_name)
            run_id = self.run.info.run_id

            # Log parameters
            self.mlflow.log_params(run_config.hyperparameters)
            self.mlflow.log_param("model_type", run_config.model_type)
            self.mlflow.log_param("dataset_version", run_config.dataset_version)
            self.mlflow.log_param("random_seed", run_config.random_seed)

            # Log tags
            for key, value in run_config.tags.items():
                self.mlflow.set_tag(key, value)

            # Log git and system info
            if self.config.log_code:
                git_info = self._get_git_info()
                for key, value in git_info.items():
                    self.mlflow.set_tag(key, value)

            if self.config.log_system_metrics:
                system_info = self._get_system_info()
                for key, value in system_info.items():
                    self.mlflow.set_tag(f"system_{key}", str(value))

            if run_config.notes:
                self.mlflow.set_tag("notes", run_config.notes)

        if self.wandb:
            self.wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=run_config.run_name,
                config={
                    **run_config.hyperparameters,
                    'model_type': run_config.model_type,
                    'dataset_version': run_config.dataset_version,
                    'random_seed': run_config.random_seed
                },
                tags=list(run_config.tags.values()),
                notes=run_config.notes
            )
            run_id = self.wandb.run.id

        logger.info(f"Started run: {run_config.run_name} (ID: {run_id})")
        return run_id

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to tracking systems."""
        if self.mlflow:
            self.mlflow.log_metrics(metrics, step=step)

        if self.wandb:
            self.wandb.log(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        self.log_metrics({key: value}, step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact file."""
        if self.mlflow:
            self.mlflow.log_artifact(local_path, artifact_path)

        if self.wandb:
            self.wandb.save(local_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log a directory of artifacts."""
        if self.mlflow:
            self.mlflow.log_artifacts(local_dir, artifact_path)

        if self.wandb:
            self.wandb.save(f"{local_dir}/*")

    def log_model(self, model, model_name: str, signature=None):
        """Log a model artifact."""
        if self.mlflow:
            import mlflow.pytorch
            mlflow.pytorch.log_model(model, model_name, signature=signature)

        if self.wandb:
            # Save model to temp file and log
            import tempfile
            import torch

            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / f"{model_name}.pt"
                torch.save(model.state_dict(), model_path)
                self.wandb.save(str(model_path))

    def log_figure(self, figure, artifact_name: str):
        """Log a matplotlib figure."""
        if self.mlflow:
            self.mlflow.log_figure(figure, artifact_name)

        if self.wandb:
            self.wandb.log({artifact_name: self.wandb.Image(figure)})

    def log_table(self, data: Dict[str, List], table_name: str):
        """Log tabular data."""
        if self.wandb:
            table = self.wandb.Table(columns=list(data.keys()))
            n_rows = len(list(data.values())[0])
            for i in range(n_rows):
                row = [data[col][i] for col in data.keys()]
                table.add_data(*row)
            self.wandb.log({table_name: table})

        # Also save as JSON artifact for MLflow
        if self.mlflow:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(data, f)
                self.mlflow.log_artifact(f.name, f"tables/{table_name}.json")

    def log_confusion_matrix(self, y_true, y_pred, class_names: List[str]):
        """Log confusion matrix."""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

        self.log_figure(fig, 'confusion_matrix.png')
        plt.close(fig)

    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        if self.mlflow:
            self.mlflow.set_tag(key, value)

        if self.wandb:
            self.wandb.config.update({key: value})

    def end_run(self, status: str = "FINISHED"):
        """End the current run."""
        if self.mlflow:
            self.mlflow.end_run(status=status)

        if self.wandb:
            self.wandb.finish()

        logger.info(f"Run ended with status: {status}")


class DatasetVersioner:
    """Track dataset versions and compute hashes."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        return sha256.hexdigest()

    def create_manifest(self, data_dir: str, output_file: str) -> Dict:
        """Create dataset manifest with file hashes."""
        data_path = Path(data_dir)
        manifest = {
            'created': datetime.now().isoformat(),
            'base_dir': str(data_path),
            'files': []
        }

        for file_path in data_path.rglob('*'):
            if file_path.is_file():
                manifest['files'].append({
                    'path': str(file_path.relative_to(data_path)),
                    'size': file_path.stat().st_size,
                    'hash': self.compute_file_hash(file_path)
                })

        # Overall hash
        file_hashes = sorted([f['hash'] for f in manifest['files']])
        manifest['overall_hash'] = hashlib.sha256(
            ''.join(file_hashes).encode()
        ).hexdigest()[:16]

        with open(output_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        return manifest


class CostEstimator:
    """Estimate GPU hours and costs for experiments."""

    # Approximate costs per GPU hour (AWS p3.2xlarge)
    GPU_COST_PER_HOUR = 3.06

    def __init__(self):
        self.estimates = []

    def add_experiment(
        self,
        name: str,
        n_epochs: int,
        samples_per_epoch: int,
        samples_per_second: float,
        n_trials: int = 1
    ):
        """Add experiment to cost estimate."""
        hours_per_run = (n_epochs * samples_per_epoch) / (samples_per_second * 3600)
        total_hours = hours_per_run * n_trials

        self.estimates.append({
            'name': name,
            'hours_per_run': hours_per_run,
            'n_trials': n_trials,
            'total_hours': total_hours,
            'estimated_cost': total_hours * self.GPU_COST_PER_HOUR
        })

    def generate_report(self) -> str:
        """Generate cost estimation report."""
        total_hours = sum(e['total_hours'] for e in self.estimates)
        total_cost = sum(e['estimated_cost'] for e in self.estimates)

        report = "# GPU Cost Estimation Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        report += "## Experiment Breakdown\n\n"
        report += "| Experiment | Hours/Run | Trials | Total Hours | Est. Cost |\n"
        report += "|------------|-----------|--------|-------------|----------|\n"

        for exp in self.estimates:
            report += f"| {exp['name']} | {exp['hours_per_run']:.1f} | {exp['n_trials']} | {exp['total_hours']:.1f} | ${exp['estimated_cost']:.2f} |\n"

        report += f"\n**Total GPU Hours: {total_hours:.1f}**\n"
        report += f"**Total Estimated Cost: ${total_cost:.2f}**\n"

        return report


def setup_experiment(
    experiment_name: str = "sleep_apnea_detection",
    use_mlflow: bool = True,
    use_wandb: bool = False
) -> ExperimentTracker:
    """Convenience function to set up experiment tracking."""

    config = ExperimentConfig(
        experiment_name=experiment_name,
        use_mlflow=use_mlflow,
        use_wandb=use_wandb
    )

    return ExperimentTracker(config)
