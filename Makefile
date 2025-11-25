# NeendAI Makefile
# Commands for research experiments and deployment

.PHONY: all install test lint preprocess train hyperopt dashboard clean

# Variables
PYTHON := python
PIP := pip
DOCKER := docker
STREAMLIT := streamlit

# Default target
all: install test

# Installation
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-research.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-research.txt
	$(PIP) install -e .

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html

test-fast:
	pytest tests/ -v -x --tb=short

# Linting
lint:
	black src/
	isort src/
	mypy src/ --ignore-missing-imports

# Data preprocessing
preprocess:
	$(PYTHON) -m src.research.distributed_preprocessing \
		--data-dir data/raw \
		--output-dir data/processed \
		--backend ray

preprocess-dask:
	$(PYTHON) -m src.research.distributed_preprocessing \
		--data-dir data/raw \
		--output-dir data/processed \
		--backend dask

# Literature search
literature:
	$(PYTHON) -c "from src.research.literature_search import LiteratureSearchModule; \
		lit = LiteratureSearchModule(); lit.compile_all()"

# Training
train-ssl:
	$(PYTHON) -m src.research.ssl_pretraining \
		--config configs/research_config.yaml \
		--model-type wav2vec2 \
		--data-dir data/processed \
		--output-dir research/models/ssl

train-foundation:
	$(PYTHON) -m src.research.foundation_model \
		--config configs/research_config.yaml \
		--pretrain \
		--data-dir data/processed \
		--output-dir research/models/foundation

train-finetune:
	$(PYTHON) -m src.research.foundation_model \
		--config configs/research_config.yaml \
		--finetune \
		--checkpoint research/models/foundation/best.pt \
		--output-dir research/models/finetuned

# Hyperparameter optimization
hyperopt:
	$(PYTHON) -m src.research.hyperopt_search \
		--config configs/research_config.yaml \
		--model-family cnn \
		--n-trials 1000 \
		--output-dir research/hyperopt

hyperopt-transformer:
	$(PYTHON) -m src.research.hyperopt_search \
		--config configs/research_config.yaml \
		--model-family transformer \
		--n-trials 1000 \
		--output-dir research/hyperopt

# Ablation studies
ablation:
	$(PYTHON) -m src.research.causal_discovery \
		--mode ablation \
		--config configs/research_config.yaml \
		--output-dir research/ablations

# Statistical analysis
analyze:
	$(PYTHON) -m src.research.statistical_analysis \
		--results-dir research/experiments \
		--output-dir research/reports

# Dashboard
dashboard:
	$(STREAMLIT) run research/dashboards/research_dashboard.py \
		--server.port 8501 \
		--theme.base dark

# Docker
docker-build:
	$(DOCKER) build -t neendai-api .
	$(DOCKER) build -t neendai-web ./web

docker-up:
	$(DOCKER)-compose up -d

docker-down:
	$(DOCKER)-compose down

# Experiment tracking
mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000

# Export models
export-onnx:
	$(PYTHON) -m src.research.export \
		--checkpoint research/models/foundation/best.pt \
		--format onnx \
		--output research/models/exported

export-torchscript:
	$(PYTHON) -m src.research.export \
		--checkpoint research/models/foundation/best.pt \
		--format torchscript \
		--output research/models/exported

# Benchmarks
benchmark:
	$(PYTHON) -m src.research.benchmark \
		--model research/models/exported/model.onnx \
		--device cpu

benchmark-gpu:
	$(PYTHON) -m src.research.benchmark \
		--model research/models/exported/model.onnx \
		--device cuda

# Reports
report:
	$(PYTHON) -m src.research.generate_report \
		--results-dir research/experiments \
		--output research/reports/final_report.pdf

# Clean
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf *.egg-info build dist
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

clean-all: clean
	rm -rf data/processed
	rm -rf research/models
	rm -rf research/hyperopt
	rm -rf mlruns

# GPU cost estimation
estimate-cost:
	@echo "Estimated GPU Hours:"
	@echo "  SSL Pretraining: 800 hours"
	@echo "  Hyperopt Search: 400 hours"
	@echo "  Ablation Studies: 200 hours"
	@echo "  Foundation Training: 600 hours"
	@echo "  Total: 2000 hours"
	@echo "  Estimated Cost (p3.2xlarge): \$$6,120"

# Help
help:
	@echo "NeendAI Research Commands:"
	@echo ""
	@echo "  install          - Install dependencies"
	@echo "  test             - Run tests with coverage"
	@echo "  lint             - Format and check code"
	@echo "  preprocess       - Run distributed preprocessing"
	@echo "  literature       - Generate literature search outputs"
	@echo "  train-ssl        - Train self-supervised model"
	@echo "  train-foundation - Train foundation model"
	@echo "  hyperopt         - Run hyperparameter optimization"
	@echo "  ablation         - Run ablation studies"
	@echo "  analyze          - Generate statistical analysis"
	@echo "  dashboard        - Launch Streamlit dashboard"
	@echo "  docker-up        - Start Docker services"
	@echo "  report           - Generate final report"
	@echo "  estimate-cost    - Show GPU cost estimates"
