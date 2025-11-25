# NeendAI Research Artifacts Manifest

Generated: 2024-01-15

## Summary

This document catalogs all research artifacts produced for the NeendAI sleep audio biomarker analysis system.

## Research Modules

| Path | Description | Type |
|------|-------------|------|
| `src/research/distributed_preprocessing.py` | Dask/Ray distributed preprocessing with multi-resolution features | Python |
| `src/research/literature_search.py` | Literature search and audio signature database | Python |
| `src/research/ssl_pretraining.py` | Self-supervised models (wav2vec2, HuBERT, BYOL-A) | Python |
| `src/research/hyperopt_search.py` | Optuna/Ray Tune hyperparameter optimization | Python |
| `src/research/statistical_analysis.py` | Bootstrap CIs, hypothesis tests, calibration | Python |
| `src/research/experiment_tracking.py` | MLflow/W&B experiment tracking | Python |
| `src/research/foundation_model.py` | 300M+ parameter foundation transformer | Python |
| `src/research/causal_discovery.py` | PC algorithm, do-calculus, ablations | Python |
| `src/research/__init__.py` | Module exports | Python |

## Infrastructure

| Path | Description | Type |
|------|-------------|------|
| `configs/research_config.yaml` | Research experiment configuration | YAML |
| `Makefile` | Build and experiment commands | Makefile |
| `requirements-research.txt` | Research Python dependencies | Text |

## Dashboards & Notebooks

| Path | Description | Type |
|------|-------------|------|
| `research/dashboards/research_dashboard.py` | Streamlit interactive dashboard | Python |
| `notebooks/research_analysis.ipynb` | Jupyter analysis notebook | Notebook |

## Documentation

| Path | Description | Type |
|------|-------------|------|
| `research/README.md` | Research module documentation | Markdown |
| `ARTIFACTS.md` | This manifest | Markdown |

## Core Features

### 1. Distributed Preprocessing
- Multi-resolution mel spectrograms (64, 128, 256 bands)
- CQT, CWT scalograms
- MFCCs with deltas
- Spectral features (centroid, flatness, rolloff)
- Advanced features (Teager energy, Hilbert envelope, jitter/shimmer)
- Label fusion (Dawid-Skene algorithm)
- Synthetic augmentation pipeline

### 2. Self-Supervised Pretraining
- **wav2vec 2.0**: Contrastive learning with Gumbel vector quantization
- **Masked Spectrogram**: Patch-based masked reconstruction
- **BYOL-A**: Bootstrap Your Own Latent for audio
- Mixed precision training with gradient accumulation
- Distributed data parallel support

### 3. Foundation Model
- **Architecture**: 24-layer Transformer with RoPE
- **Parameters**: ~312M (1024 hidden, 16 heads, 4096 FFN)
- **Pretraining**: 75% mask ratio MAE
- **Multi-task**: Sleep stage, event detection, quality, AHI
- SwiGLU activation, pre-norm

### 4. Hyperparameter Optimization
- TPE sampler with ASHA early stopping
- 1000-2000 trials per model family
- Search spaces: CNN, Transformer, SSL, Ensemble
- Parallel execution with Ray Tune

### 5. Statistical Analysis
- Bootstrap confidence intervals (1000 resamples)
- DeLong test for AUROC comparison
- McNemar test for paired accuracy
- Wilcoxon signed-rank test
- Cohen's d and Hedges' g effect sizes
- Calibration: ECE, temperature scaling, isotonic regression
- Subgroup analysis with I² heterogeneity

### 6. Causal Discovery
- PC algorithm for structure learning
- Backdoor adjustment sets
- IPW and regression-based ATE estimation
- Exhaustive ablation studies
- LaTeX table generation

### 7. Experiment Tracking
- MLflow/W&B integration
- Git commit tracking
- Dataset versioning with hashes
- GPU cost estimation

## Audio Signatures Database

14 literature-derived signatures with citations:

1. Snore Spectral Slope (Montazeri2012)
2. Breathing Pause Duration (Nakano2014)
3. Snore Formant Frequencies (Ng2008)
4. Spectral Entropy (Nakano2014)
5. Snore Pitch (Montazeri2012)
6. Cyclic Spectral Pattern
7. Cough Acoustic Pattern (Abeyratne2013)
8. Movement Artifact Pattern
9. REM Behavior Vocalization
10. Bruxism Grinding Pattern

## GPU Cost Estimates

| Experiment | Hours | Cost (USD) |
|------------|-------|------------|
| SSL Pretraining | 800 | $2,448 |
| Hyperopt Search | 400 | $1,224 |
| Ablation Studies | 200 | $612 |
| Foundation Training | 600 | $1,836 |
| **Total** | **2000** | **$6,120** |

## Usage

### Quick Start
```bash
# Install
pip install -r requirements-research.txt

# Literature outputs
make literature

# Launch dashboard
make dashboard

# Train foundation model
make train-foundation

# Run hyperopt
make hyperopt
```

### Running Full Pipeline
```bash
# 1. Preprocess data
make preprocess

# 2. Train SSL model
make train-ssl

# 3. Train foundation
make train-foundation

# 4. Fine-tune
make train-finetune

# 5. Run ablations
make ablation

# 6. Generate report
make report
```

## Dependencies

Key research dependencies:
- PyTorch 2.0+
- Optuna 3.4+
- Ray[Tune] 2.8+
- MLflow 2.8+
- Streamlit 1.28+
- librosa 0.10+

See `requirements-research.txt` for full list.

## License

MIT License

## Citations

All literature findings include DOIs. See `research/literature/citations.bib` for BibTeX.
