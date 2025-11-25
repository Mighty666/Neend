# neendai research module

ml code. separate from apps because it needs a lot of compute.

## what's in here

- distributed preprocessing with dask/ray
- self-supervised pretraining
- foundation model training
- hyperparameter optimization
- statistical analysis
- causal discovery

## Quick Start

```bash
# Install dependencies
pip install -r requirements-research.txt

# Generate literature outputs
make literature

# Run preprocessing
make preprocess

# Launch dashboard
make dashboard
```

## Directory Structure

```
research/
├── pretraining/      # SSL model checkpoints
├── hyperopt/         # Optuna databases and results
├── analysis/         # Statistical analysis outputs
├── literature/       # Literature search outputs
├── models/           # Trained model checkpoints
├── experiments/      # Experiment logs
├── dashboards/       # Streamlit apps
├── ablations/        # Ablation study results
└── causal/          # Causal discovery outputs
```

## experiments

self-supervised pretraining, foundation model training, hyperparameter search, statistical analysis

## metrics

auroc, auprc, sensitivity, specificity, calibration metrics

## audio signatures

see literature/audio_signatures.csv for list

## compute

total: ~2000 gpu hours

## License

MIT License
