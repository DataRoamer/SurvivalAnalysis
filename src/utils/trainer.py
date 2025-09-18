"""
Training and Validation Pipeline for Deep Survival Analysis Models

This module provides comprehensive training utilities including model training,
validation, hyperparameter tuning, and model selection for survival analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import optuna
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
import os
import json
from pathlib import Path

from ..models.deep_survival import DeepSurv, DeepHit, NeuralMTLR, create_data_loaders, get_model_class
from ..evaluation.metrics import SurvivalMetricsTracker, compute_all_metrics
from ..data.seer_preprocessor import SEERPreprocessor


class SurvivalTrainer:
    """
    Comprehensive trainer for deep survival analysis models.
    """

    def __init__(self,
                 model_name: str = 'deepsurv',
                 max_epochs: int = 100,
                 early_stopping_patience: int = 10,
                 learning_rate: float = 1e-3,
                 batch_size: int = 64,
                 eval_time_points: Optional[List[float]] = None,
                 save_dir: str = 'survival_experiments',
                 experiment_name: str = 'default',
                 logger_type: str = 'tensorboard',
                 random_state: int = 42):
        """
        Initialize survival trainer.

        Args:
            model_name: Type of model ('deepsurv', 'deephit', 'mtlr')
            max_epochs: Maximum training epochs
            early_stopping_patience: Early stopping patience
            learning_rate: Learning rate
            batch_size: Batch size
            eval_time_points: Time points for evaluation
            save_dir: Directory to save experiments
            experiment_name: Name of experiment
            logger_type: Type of logger ('tensorboard', 'wandb', 'none')
            random_state: Random seed
        """
        self.model_name = model_name.lower()
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.eval_time_points = eval_time_points
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.logger_type = logger_type
        self.random_state = random_state

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = self.save_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.trainer = None
        self.metrics_tracker = None

    def prepare_model(self, input_dim: int, model_config: Optional[Dict] = None) -> pl.LightningModule:
        """
        Prepare model based on configuration.

        Args:
            input_dim: Number of input features
            model_config: Model-specific configuration

        Returns:
            Initialized PyTorch Lightning module
        """
        model_class = get_model_class(self.model_name)

        if model_config is None:
            model_config = {}

        # Default configurations for each model
        default_configs = {
            'deepsurv': {
                'hidden_dims': [64, 32],
                'dropout': 0.5,
                'learning_rate': self.learning_rate,
                'weight_decay': 1e-4,
                'batch_norm': True
            },
            'deephit': {
                'num_times': 100,
                'num_causes': 1,
                'hidden_dims': [64, 32],
                'dropout': 0.5,
                'learning_rate': self.learning_rate,
                'weight_decay': 1e-4,
                'alpha': 1.0,
                'sigma': 1.0
            },
            'mtlr': {
                'num_time_intervals': 50,
                'hidden_dims': [64, 32],
                'dropout': 0.5,
                'learning_rate': self.learning_rate,
                'weight_decay': 1e-4
            }
        }

        config = default_configs[self.model_name].copy()
        config.update(model_config)
        config['input_dim'] = input_dim

        self.model = model_class(**config)
        return self.model

    def setup_callbacks(self) -> List[pl.Callback]:
        """
        Setup PyTorch Lightning callbacks.

        Returns:
            List of callbacks
        """
        callbacks = []

        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop)

        # Model checkpoint
        checkpoint = ModelCheckpoint(
            dirpath=self.experiment_dir / 'checkpoints',
            filename='{epoch}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        callbacks.append(checkpoint)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)

        return callbacks

    def setup_logger(self) -> Optional[pl.loggers.Logger]:
        """
        Setup PyTorch Lightning logger.

        Returns:
            Logger instance or None
        """
        if self.logger_type == 'tensorboard':
            logger = TensorBoardLogger(
                save_dir=str(self.save_dir),
                name=self.experiment_name,
                version=None
            )
            return logger
        elif self.logger_type == 'wandb':
            logger = WandbLogger(
                project='survival_analysis',
                name=self.experiment_name,
                save_dir=str(self.save_dir)
            )
            return logger
        else:
            return None

    def train(self,
              X_train: np.ndarray,
              t_train: np.ndarray,
              e_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              t_val: Optional[np.ndarray] = None,
              e_val: Optional[np.ndarray] = None,
              model_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train survival model.

        Args:
            X_train: Training features
            t_train: Training times
            e_train: Training events
            X_val: Validation features (optional)
            t_val: Validation times (optional)
            e_val: Validation events (optional)
            model_config: Model configuration

        Returns:
            Training results and metrics
        """
        # Prepare model
        input_dim = X_train.shape[1]
        model = self.prepare_model(input_dim, model_config)

        # Create data loaders
        data_loaders = create_data_loaders(
            X_train, t_train, e_train,
            X_val, t_val, e_val,
            batch_size=self.batch_size
        )

        # Setup trainer
        callbacks = self.setup_callbacks()
        logger = self.setup_logger()

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator='auto',
            devices='auto',
            deterministic=True,
            enable_progress_bar=True
        )

        # Train model
        self.trainer.fit(
            model,
            train_dataloaders=data_loaders['train'],
            val_dataloaders=data_loaders.get('val')
        )

        # Evaluate on validation set if available
        results = {'model': model, 'trainer': self.trainer}

        if X_val is not None:
            val_metrics = self.evaluate(model, X_val, t_val, e_val)
            results['validation_metrics'] = val_metrics

        # Save model configuration
        config_path = self.experiment_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            config_to_save = model_config or {}
            config_to_save.update({
                'model_name': self.model_name,
                'input_dim': input_dim,
                'max_epochs': self.max_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate
            })
            json.dump(config_to_save, f, indent=2)

        return results

    def evaluate(self,
                model: pl.LightningModule,
                X_test: np.ndarray,
                t_test: np.ndarray,
                e_test: np.ndarray,
                return_predictions: bool = False) -> Dict[str, Any]:
        """
        Evaluate trained model.

        Args:
            model: Trained model
            X_test: Test features
            t_test: Test times
            e_test: Test events
            return_predictions: Whether to return predictions

        Returns:
            Evaluation metrics and optionally predictions
        """
        model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test)

            if self.model_name == 'deepsurv':
                # Get risk scores
                risk_scores = model.predict_risk(X_tensor).numpy().flatten()
                survival_probs = None
                time_points = None

            elif self.model_name in ['deephit', 'mtlr']:
                # Get survival probabilities
                survival_probs = model.predict_survival(X_tensor).numpy()

                if self.model_name == 'deephit':
                    time_points = np.linspace(0, t_test.max(), model.num_times)
                else:  # mtlr
                    time_points = np.linspace(0, t_test.max(), model.num_time_intervals)

                # Convert to risk scores (negative expected survival time)
                expected_times = np.sum(survival_probs * time_points.reshape(1, -1), axis=1)
                risk_scores = -expected_times

        # Compute metrics
        metrics = compute_all_metrics(
            t_test, e_test, risk_scores, survival_probs, time_points, self.eval_time_points
        )

        results = {'metrics': metrics}

        if return_predictions:
            results['predictions'] = {
                'risk_scores': risk_scores,
                'survival_probs': survival_probs,
                'time_points': time_points
            }

        return results

    def cross_validate(self,
                      X: np.ndarray,
                      t: np.ndarray,
                      e: np.ndarray,
                      n_folds: int = 5,
                      model_config: Optional[Dict] = None,
                      stratify: bool = True) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            X: Features
            t: Times
            e: Events
            n_folds: Number of CV folds
            model_config: Model configuration
            stratify: Whether to stratify by events

        Returns:
            Cross-validation results
        """
        if stratify:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            splits = kf.split(X, e)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            splits = kf.split(X)

        cv_results = {
            'fold_metrics': [],
            'models': []
        }

        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"Training fold {fold + 1}/{n_folds}")

            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            t_train, t_val = t[train_idx], t[val_idx]
            e_train, e_val = e[train_idx], e[val_idx]

            # Create fold-specific trainer
            fold_trainer = SurvivalTrainer(
                model_name=self.model_name,
                max_epochs=self.max_epochs,
                early_stopping_patience=self.early_stopping_patience,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                eval_time_points=self.eval_time_points,
                save_dir=str(self.save_dir),
                experiment_name=f"{self.experiment_name}_fold_{fold}",
                logger_type='none',  # Disable logging for CV
                random_state=self.random_state + fold
            )

            # Train fold
            fold_results = fold_trainer.train(
                X_train, t_train, e_train,
                X_val, t_val, e_val,
                model_config
            )

            cv_results['fold_metrics'].append(fold_results['validation_metrics']['metrics'])
            cv_results['models'].append(fold_results['model'])

        # Aggregate results
        metric_names = cv_results['fold_metrics'][0].keys()
        aggregated_metrics = {}

        for metric in metric_names:
            values = [fold_metrics.get(metric, np.nan) for fold_metrics in cv_results['fold_metrics']]
            valid_values = [v for v in values if not np.isnan(v)]

            if valid_values:
                aggregated_metrics[f"{metric}_mean"] = np.mean(valid_values)
                aggregated_metrics[f"{metric}_std"] = np.std(valid_values)

        cv_results['aggregated_metrics'] = aggregated_metrics

        return cv_results

    def hyperparameter_tuning(self,
                            X_train: np.ndarray,
                            t_train: np.ndarray,
                            e_train: np.ndarray,
                            X_val: np.ndarray,
                            t_val: np.ndarray,
                            e_val: np.ndarray,
                            n_trials: int = 50,
                            metric_name: str = 'c_index',
                            direction: str = 'maximize') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            X_train: Training features
            t_train: Training times
            e_train: Training events
            X_val: Validation features
            t_val: Validation times
            e_val: Validation events
            n_trials: Number of optimization trials
            metric_name: Metric to optimize
            direction: Optimization direction ('maximize' or 'minimize')

        Returns:
            Optimization results
        """
        def objective(trial):
            # Suggest hyperparameters
            config = {}

            # Common hyperparameters
            config['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            config['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            config['dropout'] = trial.suggest_float('dropout', 0.1, 0.7)

            # Hidden layer configuration
            n_layers = trial.suggest_int('n_layers', 1, 4)
            hidden_dims = []
            for i in range(n_layers):
                dim = trial.suggest_int(f'hidden_dim_{i}', 16, 256)
                hidden_dims.append(dim)
            config['hidden_dims'] = hidden_dims

            # Model-specific hyperparameters
            if self.model_name == 'deephit':
                config['alpha'] = trial.suggest_float('alpha', 0.1, 2.0)
                config['sigma'] = trial.suggest_float('sigma', 0.1, 2.0)
                config['num_times'] = trial.suggest_int('num_times', 50, 200)
            elif self.model_name == 'mtlr':
                config['num_time_intervals'] = trial.suggest_int('num_time_intervals', 20, 100)

            # Create trial-specific trainer
            trial_trainer = SurvivalTrainer(
                model_name=self.model_name,
                max_epochs=min(self.max_epochs, 50),  # Reduce epochs for tuning
                early_stopping_patience=5,
                batch_size=self.batch_size,
                eval_time_points=self.eval_time_points,
                save_dir=str(self.save_dir),
                experiment_name=f"{self.experiment_name}_trial_{trial.number}",
                logger_type='none',
                random_state=self.random_state + trial.number
            )

            try:
                # Train model
                results = trial_trainer.train(
                    X_train, t_train, e_train,
                    X_val, t_val, e_val,
                    config
                )

                # Get target metric
                metric_value = results['validation_metrics']['metrics'].get(metric_name, np.nan)

                if np.isnan(metric_value):
                    return float('-inf') if direction == 'maximize' else float('inf')

                return metric_value

            except Exception as e:
                print(f"Trial {trial.number} failed: {e}")
                return float('-inf') if direction == 'maximize' else float('inf')

        # Create study
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)

        # Train final model with best parameters
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best {metric_name}: {study.best_value}")
        print(f"Best parameters: {study.best_params}")

        final_trainer = SurvivalTrainer(
            model_name=self.model_name,
            max_epochs=self.max_epochs,
            early_stopping_patience=self.early_stopping_patience,
            batch_size=self.batch_size,
            eval_time_points=self.eval_time_points,
            save_dir=str(self.save_dir),
            experiment_name=f"{self.experiment_name}_best",
            logger_type=self.logger_type,
            random_state=self.random_state
        )

        best_results = final_trainer.train(
            X_train, t_train, e_train,
            X_val, t_val, e_val,
            study.best_params
        )

        return {
            'study': study,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_model': best_results['model'],
            'best_trainer': final_trainer
        }


def train_multiple_models(X_train: np.ndarray,
                         t_train: np.ndarray,
                         e_train: np.ndarray,
                         X_val: np.ndarray,
                         t_val: np.ndarray,
                         e_val: np.ndarray,
                         X_test: np.ndarray,
                         t_test: np.ndarray,
                         e_test: np.ndarray,
                         models: List[str] = ['deepsurv', 'deephit', 'mtlr'],
                         save_dir: str = 'model_comparison',
                         **trainer_kwargs) -> Dict[str, Any]:
    """
    Train and compare multiple survival models.

    Args:
        X_train: Training features
        t_train: Training times
        e_train: Training events
        X_val: Validation features
        t_val: Validation times
        e_val: Validation events
        X_test: Test features
        t_test: Test times
        e_test: Test events
        models: List of model names to train
        save_dir: Directory to save results
        **trainer_kwargs: Additional trainer arguments

    Returns:
        Comparison results
    """
    results = {}

    for model_name in models:
        print(f"\nTraining {model_name.upper()} model...")

        trainer = SurvivalTrainer(
            model_name=model_name,
            save_dir=save_dir,
            experiment_name=model_name,
            **trainer_kwargs
        )

        # Train model
        train_results = trainer.train(
            X_train, t_train, e_train,
            X_val, t_val, e_val
        )

        # Evaluate on test set
        test_results = trainer.evaluate(
            train_results['model'],
            X_test, t_test, e_test
        )

        results[model_name] = {
            'train_results': train_results,
            'test_results': test_results,
            'trainer': trainer
        }

    # Create comparison table
    comparison_metrics = {}
    for model_name, model_results in results.items():
        test_metrics = model_results['test_results']['metrics']
        comparison_metrics[model_name] = test_metrics

    comparison_df = pd.DataFrame(comparison_metrics).T

    results['comparison'] = comparison_df
    print("\nModel Comparison:")
    print(comparison_df.round(4))

    return results