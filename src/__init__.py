"""
Deep Learning Survival Analysis Package

A comprehensive package for survival analysis using deep learning models
with cancer survival data.
"""

__version__ = "1.0.0"
__author__ = "DataRoamer"

from .data.survival_preprocessor import SurvivalDataPreprocessor, load_survival_data, create_synthetic_survival_data
from .models.deep_survival import DeepSurv, DeepHit, NeuralMTLR, SurvivalDataset, create_data_loaders
from .evaluation.metrics import (
    harrell_c_index, time_dependent_auc, integrated_brier_score,
    compute_all_metrics, SurvivalMetricsTracker
)
from .utils.trainer import SurvivalTrainer, train_multiple_models
from .utils.visualization import SurvivalVisualizer, create_survival_dashboard
from .reports.report_generator import SurvivalAnalysisReporter, generate_survival_analysis_report

__all__ = [
    # Data preprocessing
    'SurvivalDataPreprocessor',
    'load_survival_data',
    'create_synthetic_survival_data',

    # Models
    'DeepSurv',
    'DeepHit',
    'NeuralMTLR',
    'SurvivalDataset',
    'create_data_loaders',

    # Evaluation
    'harrell_c_index',
    'time_dependent_auc',
    'integrated_brier_score',
    'compute_all_metrics',
    'SurvivalMetricsTracker',

    # Training
    'SurvivalTrainer',
    'train_multiple_models',

    # Visualization
    'SurvivalVisualizer',
    'create_survival_dashboard',

    # Reports
    'SurvivalAnalysisReporter',
    'generate_survival_analysis_report'
]