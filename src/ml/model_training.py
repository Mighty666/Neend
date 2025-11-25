"""
Model Training Pipeline with Online Learning Support
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Generator
from pathlib import Path

logger = logging.getLogger(__name__)


class OnlineTrainingPipeline:
    """
    Training pipeline supporting both batch and online learning.
    """

    def __init__(
        self,
        model_dir: str = './models',
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        self.lr = learning_rate

        # Training metrics
        self.metrics_history = {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': []
        }

    def prepare_data_stream(
        self,
        data_dir: str,
        augment: bool = True
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Create data stream for online training.

        Args:
            data_dir: Directory containing training data
            augment: Whether to apply augmentation

        Yields:
            Tuple of (features, label)
        """
        from src.data import PhysioNetLoader, DataAugmenter

        loader = PhysioNetLoader(data_dir)
        augmenter = DataAugmenter() if augment else None

        # Generate synthetic data for demo
        X, y = loader.generate_synthetic_data(n_samples=100)

        for sample_X, sample_y in zip(X, y):
            for features, label in zip(sample_X, sample_y):
                yield features, label

                # Yield augmented versions
                if augmenter and np.random.random() > 0.5:
                    # Simple augmentation: add noise
                    noisy = features + np.random.randn(*features.shape) * 0.1
                    yield noisy, label

    def train_online(
        self,
        data_stream: Generator,
        n_iterations: int = 10000,
        eval_every: int = 100
    ):
        """
        Train model using online learning.

        Args:
            data_stream: Generator yielding (features, label) tuples
            n_iterations: Number of training iterations
            eval_every: Evaluate every N iterations
        """
        from src.ml.online import OnlineClassifier

        classifier = OnlineClassifier(n_features=39, n_classes=4)

        correct = 0
        total = 0

        for i, (features, label) in enumerate(data_stream):
            if i >= n_iterations:
                break

            # Predict before update
            pred, _ = classifier.predict(features)
            if pred == label:
                correct += 1
            total += 1

            # Update model
            classifier.partial_fit(features, label)

            # Evaluate
            if (i + 1) % eval_every == 0:
                accuracy = correct / total
                self.metrics_history['accuracy'].append(accuracy)
                logger.info(f"Iteration {i+1}: Accuracy = {accuracy:.4f}")

                # Reset counters
                correct = 0
                total = 0

        return classifier

    def cross_validate_online(
        self,
        data_stream: Generator,
        n_folds: int = 5,
        fold_size: int = 1000
    ) -> Dict:
        """
        Perform prequential (interleaved test-then-train) evaluation.
        """
        from src.ml.online import OnlineClassifier

        fold_results = []

        for fold in range(n_folds):
            classifier = OnlineClassifier(n_features=39, n_classes=4)
            fold_correct = 0
            fold_total = 0

            for i, (features, label) in enumerate(data_stream):
                if i >= fold_size:
                    break

                # Test
                pred, _ = classifier.predict(features)
                if pred == label:
                    fold_correct += 1
                fold_total += 1

                # Train
                classifier.partial_fit(features, label)

            fold_accuracy = fold_correct / max(fold_total, 1)
            fold_results.append(fold_accuracy)
            logger.info(f"Fold {fold+1}: Accuracy = {fold_accuracy:.4f}")

        return {
            'mean_accuracy': np.mean(fold_results),
            'std_accuracy': np.std(fold_results),
            'fold_results': fold_results
        }

    def train_hybrid(
        self,
        batch_data: Tuple[np.ndarray, np.ndarray],
        stream_data: Generator,
        pretrain_epochs: int = 10,
        online_iterations: int = 5000
    ):
        """
        Hybrid training: pretrain on batch, then fine-tune online.
        """
        # Phase 1: Batch pretraining
        logger.info("Phase 1: Batch pretraining")
        # Would use TensorFlow/PyTorch here for batch training

        # Phase 2: Online fine-tuning
        logger.info("Phase 2: Online fine-tuning")
        classifier = self.train_online(stream_data, online_iterations)

        return classifier

    def save_metrics(self, filename: str = 'training_metrics.npz'):
        """Save training metrics to file"""
        path = self.model_dir / filename
        np.savez(path, **self.metrics_history)
        logger.info(f"Metrics saved to {path}")

    def load_metrics(self, filename: str = 'training_metrics.npz') -> Dict:
        """Load training metrics from file"""
        path = self.model_dir / filename
        data = np.load(path)
        return {key: data[key].tolist() for key in data.files}


class IncrementalEvaluator:
    """
    Evaluate model performance incrementally on streaming data.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions = []
        self.labels = []
        self.confidences = []

    def update(self, prediction: int, label: int, confidence: float):
        """Update evaluator with new prediction"""
        self.predictions.append(prediction)
        self.labels.append(label)
        self.confidences.append(confidence)

        # Keep only recent window
        if len(self.predictions) > self.window_size:
            self.predictions = self.predictions[-self.window_size:]
            self.labels = self.labels[-self.window_size:]
            self.confidences = self.confidences[-self.window_size:]

    def get_metrics(self) -> Dict:
        """Calculate current performance metrics"""
        if not self.predictions:
            return {}

        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        confidences = np.array(self.confidences)

        accuracy = np.mean(predictions == labels)

        # Per-class metrics
        n_classes = 4
        precision = []
        recall = []

        for c in range(n_classes):
            tp = np.sum((predictions == c) & (labels == c))
            fp = np.sum((predictions == c) & (labels != c))
            fn = np.sum((predictions != c) & (labels == c))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0

            precision.append(prec)
            recall.append(rec)

        return {
            'accuracy': accuracy,
            'mean_precision': np.mean(precision),
            'mean_recall': np.mean(recall),
            'mean_confidence': np.mean(confidences),
            'class_precision': precision,
            'class_recall': recall
        }


def train_from_cli():
    """Command-line interface for training"""
    import argparse

    parser = argparse.ArgumentParser(description='Train sleep apnea classifier')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--augment', action='store_true')

    args = parser.parse_args()

    pipeline = OnlineTrainingPipeline()
    data_stream = pipeline.prepare_data_stream(args.data_dir, args.augment)

    classifier = pipeline.train_online(
        data_stream,
        n_iterations=args.iterations,
        eval_every=args.eval_every
    )

    pipeline.save_metrics()
    logger.info("Training complete!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train_from_cli()
