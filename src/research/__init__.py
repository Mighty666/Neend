"""
NeendAI Research Module

Research-grade sleep audio biomarker analysis with statistical rigor.
"""

from .distributed_preprocessing import (
    DistributedPreprocessor,
    PreprocessingConfig,
    LabelFusionModule,
    SyntheticAugmentation
)
from .literature_search import (
    LiteratureSearchModule,
    AudioSignature,
    LiteratureFinding
)
from .ssl_pretraining import (
    SSLConfig,
    Wav2Vec2Model,
    MaskedSpectrogramModel,
    BYOLA,
    SSLPretrainer,
    create_ssl_model
)
from .hyperopt_search import (
    HyperoptConfig,
    OptunaOptimizer,
    SearchSpace,
    run_hyperopt_study
)
from .statistical_analysis import (
    BootstrapAnalyzer,
    ClassificationMetrics,
    HypothesisTests,
    EffectSizeCalculator,
    CalibrationAnalyzer,
    generate_statistical_report
)
from .experiment_tracking import (
    ExperimentConfig,
    ExperimentTracker,
    RunConfig,
    setup_experiment,
    CostEstimator
)
from .foundation_model import (
    FoundationModelConfig,
    SleepAudioFoundation,
    MultiTaskHead,
    DomainShiftMetrics,
    create_foundation_model
)
from .causal_discovery import (
    CausalGraph,
    PCAlgorithm,
    DoCalculus,
    AblationStudy,
    CausalSleepAnalyzer
)

__version__ = "0.1.0"

__all__ = [
    # Preprocessing
    'DistributedPreprocessor',
    'PreprocessingConfig',
    'LabelFusionModule',
    'SyntheticAugmentation',

    # Literature
    'LiteratureSearchModule',
    'AudioSignature',
    'LiteratureFinding',

    # SSL
    'SSLConfig',
    'Wav2Vec2Model',
    'MaskedSpectrogramModel',
    'BYOLA',
    'SSLPretrainer',
    'create_ssl_model',

    # Hyperopt
    'HyperoptConfig',
    'OptunaOptimizer',
    'SearchSpace',
    'run_hyperopt_study',

    # Statistics
    'BootstrapAnalyzer',
    'ClassificationMetrics',
    'HypothesisTests',
    'EffectSizeCalculator',
    'CalibrationAnalyzer',
    'generate_statistical_report',

    # Tracking
    'ExperimentConfig',
    'ExperimentTracker',
    'RunConfig',
    'setup_experiment',
    'CostEstimator',

    # Foundation
    'FoundationModelConfig',
    'SleepAudioFoundation',
    'MultiTaskHead',
    'DomainShiftMetrics',
    'create_foundation_model',

    # Causal
    'CausalGraph',
    'PCAlgorithm',
    'DoCalculus',
    'AblationStudy',
    'CausalSleepAnalyzer',
]
