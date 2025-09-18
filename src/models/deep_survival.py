"""
Deep Learning Models for Survival Analysis

This module implements state-of-the-art deep learning models for survival analysis,
including DeepSurv, DeepHit, and Neural Multi-Task Logistic Regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import warnings


class SurvivalDataset(Dataset):
    """Dataset class for survival analysis data."""

    def __init__(self, features: np.ndarray, times: np.ndarray, events: np.ndarray):
        """
        Initialize survival dataset.

        Args:
            features: Feature matrix (n_samples, n_features)
            times: Survival times (n_samples,)
            events: Event indicators (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.times = torch.FloatTensor(times)
        self.events = torch.LongTensor(events)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'times': self.times[idx],
            'events': self.events[idx]
        }


class DeepSurv(pl.LightningModule):
    """
    Deep Survival Analysis model (DeepSurv) using Cox Proportional Hazards.

    Based on: Katzman et al. "DeepSurv: personalized treatment recommender
    system using a Cox proportional hazards deep neural network"
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.5,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_norm: bool = True):
        """
        Initialize DeepSurv model.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_norm = batch_norm

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (single neuron for risk prediction)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass returning log hazard ratios."""
        return self.network(x)

    def cox_partial_likelihood_loss(self, log_h, times, events):
        """
        Compute Cox partial likelihood loss.

        Args:
            log_h: Log hazard ratios (batch_size, 1)
            times: Survival times (batch_size,)
            events: Event indicators (batch_size,)
        """
        # Sort by time (descending)
        sorted_indices = torch.argsort(times, descending=True)
        log_h_sorted = log_h[sorted_indices].squeeze()
        events_sorted = events[sorted_indices]

        # Compute risk sets and partial likelihood
        risk_scores = torch.exp(log_h_sorted)
        risk_sets = torch.cumsum(risk_scores.flip(0), dim=0).flip(0)

        # Partial log-likelihood
        log_likelihood = log_h_sorted - torch.log(risk_sets + 1e-8)

        # Only include events (not censored observations)
        partial_likelihood = torch.sum(log_likelihood * events_sorted.float())

        # Return negative log-likelihood as loss
        return -partial_likelihood / torch.sum(events_sorted.float() + 1e-8)

    def training_step(self, batch, batch_idx):
        """Training step."""
        features = batch['features']
        times = batch['times']
        events = batch['events']

        log_h = self(features)
        loss = self.cox_partial_likelihood_loss(log_h, times, events)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        features = batch['features']
        times = batch['times']
        events = batch['events']

        log_h = self(features)
        loss = self.cox_partial_likelihood_loss(log_h, times, events)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(),
                              lr=self.learning_rate,
                              weight_decay=self.weight_decay)

    def predict_risk(self, features: torch.Tensor) -> torch.Tensor:
        """Predict risk scores for given features."""
        self.eval()
        with torch.no_grad():
            log_h = self(features)
            risk_scores = torch.exp(log_h)
        return risk_scores


class DeepHit(pl.LightningModule):
    """
    Deep Hit model for survival analysis with competing risks.

    Based on: Lee et al. "DeepHit: A Deep Learning Approach to Survival Analysis
    with Competing Risks"
    """

    def __init__(self,
                 input_dim: int,
                 num_times: int,
                 num_causes: int = 1,
                 hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.5,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 alpha: float = 1.0,
                 sigma: float = 1.0):
        """
        Initialize DeepHit model.

        Args:
            input_dim: Number of input features
            num_times: Number of time points for discretization
            num_causes: Number of competing causes (default: 1)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            learning_rate: Learning rate
            weight_decay: L2 regularization
            alpha: Weight for ranking loss
            sigma: Temperature parameter for ranking loss
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.num_times = num_times
        self.num_causes = num_causes
        self.alpha = alpha
        self.sigma = sigma

        # Shared layers
        shared_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.BatchNorm1d(hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*shared_layers)

        # Cause-specific sub-networks
        self.cause_networks = nn.ModuleList([
            nn.Linear(prev_dim, num_times) for _ in range(num_causes)
        ])

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Probability distribution over times and causes
        """
        # Shared representation
        shared_repr = self.shared_network(x)

        # Cause-specific outputs
        cause_outputs = []
        for cause_net in self.cause_networks:
            cause_outputs.append(F.softmax(cause_net(shared_repr), dim=1))

        # Stack cause outputs: (batch_size, num_causes, num_times)
        outputs = torch.stack(cause_outputs, dim=1)

        return outputs

    def likelihood_loss(self, outputs, times_idx, events):
        """
        Compute likelihood loss.

        Args:
            outputs: Model outputs (batch_size, num_causes, num_times)
            times_idx: Time indices (batch_size,)
            events: Event indicators (batch_size,)
        """
        batch_size = outputs.size(0)

        # Convert to survival function
        # S(t) = prod(1 - sum_k(f_k(tau))) for tau <= t
        hazard = outputs.sum(dim=1)  # Sum over causes
        survival = torch.cumprod(1 - hazard + 1e-8, dim=1)

        # Pad survival with 1.0 at the beginning
        survival = F.pad(survival, (1, 0), value=1.0)[:, :-1]

        likelihood = torch.zeros(batch_size, device=outputs.device)

        for i in range(batch_size):
            t_idx = times_idx[i].long()
            event = events[i]

            if event == 1:  # Event occurred
                # f(t) * S(t-1)
                likelihood[i] = outputs[i, 0, t_idx] * survival[i, t_idx]
            else:  # Censored
                # S(t)
                likelihood[i] = survival[i, t_idx]

        # Negative log-likelihood
        return -torch.mean(torch.log(likelihood + 1e-8))

    def ranking_loss(self, outputs, times, events):
        """
        Compute ranking loss for concordance.

        Args:
            outputs: Model outputs (batch_size, num_causes, num_times)
            times: Survival times (batch_size,)
            events: Event indicators (batch_size,)
        """
        batch_size = outputs.size(0)

        # Compute expected survival times
        time_points = torch.arange(self.num_times, dtype=torch.float32, device=outputs.device)
        expected_times = torch.sum(outputs[:, 0] * time_points.unsqueeze(0), dim=1)

        ranking_loss = 0.0
        count = 0

        for i in range(batch_size):
            for j in range(batch_size):
                if events[i] == 1 and times[i] < times[j]:
                    # Patient i should have higher risk (lower expected time)
                    diff = expected_times[j] - expected_times[i]
                    ranking_loss += torch.exp(-diff / self.sigma)
                    count += 1

        if count > 0:
            ranking_loss /= count

        return ranking_loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        features = batch['features']
        times = batch['times']
        events = batch['events']

        # Discretize times
        max_time = self.num_times - 1
        times_idx = torch.clamp((times / times.max() * max_time).long(), 0, max_time)

        outputs = self(features)

        # Combined loss
        likelihood_loss = self.likelihood_loss(outputs, times_idx, events)
        ranking_loss = self.ranking_loss(outputs, times, events)

        total_loss = likelihood_loss + self.alpha * ranking_loss

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_likelihood', likelihood_loss, on_step=True, on_epoch=True)
        self.log('train_ranking', ranking_loss, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        features = batch['features']
        times = batch['times']
        events = batch['events']

        # Discretize times
        max_time = self.num_times - 1
        times_idx = torch.clamp((times / times.max() * max_time).long(), 0, max_time)

        outputs = self(features)

        likelihood_loss = self.likelihood_loss(outputs, times_idx, events)
        ranking_loss = self.ranking_loss(outputs, times, events)

        total_loss = likelihood_loss + self.alpha * ranking_loss

        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_likelihood', likelihood_loss, on_step=False, on_epoch=True)
        self.log('val_ranking', ranking_loss, on_step=False, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(),
                              lr=self.hparams.learning_rate,
                              weight_decay=self.hparams.weight_decay)

    def predict_survival(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict survival probabilities.

        Args:
            features: Input features (batch_size, input_dim)

        Returns:
            Survival probabilities (batch_size, num_times)
        """
        self.eval()
        with torch.no_grad():
            outputs = self(features)
            hazard = outputs.sum(dim=1)
            survival = torch.cumprod(1 - hazard + 1e-8, dim=1)
        return survival


class NeuralMTLR(pl.LightningModule):
    """
    Neural Multi-Task Logistic Regression for survival analysis.

    Based on: Fotso "Deep Neural Networks for Survival Analysis Based on a
    Multi-Task Framework"
    """

    def __init__(self,
                 input_dim: int,
                 num_time_intervals: int,
                 hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.5,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        """
        Initialize Neural MTLR model.

        Args:
            input_dim: Number of input features
            num_time_intervals: Number of time intervals
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.num_time_intervals = num_time_intervals

        # Feature extractor
        feature_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            feature_layers.append(nn.BatchNorm1d(hidden_dim))
            feature_layers.append(nn.ReLU())
            feature_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*feature_layers)

        # Output layer for each time interval
        self.output_layer = nn.Linear(prev_dim, num_time_intervals)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Logits for each time interval (batch_size, num_time_intervals)
        """
        features = self.feature_extractor(x)
        logits = self.output_layer(features)
        return logits

    def mtlr_loss(self, logits, time_intervals, events):
        """
        Compute MTLR loss.

        Args:
            logits: Model logits (batch_size, num_time_intervals)
            time_intervals: Time interval indices (batch_size,)
            events: Event indicators (batch_size,)
        """
        batch_size = logits.size(0)

        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        loss = 0.0

        for i in range(batch_size):
            t_interval = time_intervals[i].long()
            event = events[i]

            if event == 1:  # Event occurred
                # Probability of event in interval t_interval
                # P(T = t) = P(T >= t) * P(T < t+1 | T >= t)
                if t_interval > 0:
                    survival_prob = torch.prod(1 - probs[i, :t_interval])
                else:
                    survival_prob = 1.0

                event_prob = survival_prob * probs[i, t_interval]
                loss -= torch.log(event_prob + 1e-8)

            else:  # Censored
                # Probability of survival beyond t_interval
                if t_interval < self.num_time_intervals:
                    survival_prob = torch.prod(1 - probs[i, :t_interval+1])
                else:
                    survival_prob = torch.prod(1 - probs[i, :])

                loss -= torch.log(survival_prob + 1e-8)

        return loss / batch_size

    def training_step(self, batch, batch_idx):
        """Training step."""
        features = batch['features']
        times = batch['times']
        events = batch['events']

        # Convert times to interval indices
        max_time = times.max()
        time_intervals = (times / max_time * (self.num_time_intervals - 1)).long()
        time_intervals = torch.clamp(time_intervals, 0, self.num_time_intervals - 1)

        logits = self(features)
        loss = self.mtlr_loss(logits, time_intervals, events)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        features = batch['features']
        times = batch['times']
        events = batch['events']

        # Convert times to interval indices
        max_time = times.max()
        time_intervals = (times / max_time * (self.num_time_intervals - 1)).long()
        time_intervals = torch.clamp(time_intervals, 0, self.num_time_intervals - 1)

        logits = self(features)
        loss = self.mtlr_loss(logits, time_intervals, events)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(),
                              lr=self.hparams.learning_rate,
                              weight_decay=self.hparams.weight_decay)

    def predict_survival(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict survival probabilities.

        Args:
            features: Input features (batch_size, input_dim)

        Returns:
            Survival probabilities for each time interval (batch_size, num_time_intervals)
        """
        self.eval()
        with torch.no_grad():
            logits = self(features)
            probs = torch.sigmoid(logits)

            # Convert to survival probabilities
            survival_probs = torch.ones_like(probs)
            for t in range(self.num_time_intervals):
                if t == 0:
                    survival_probs[:, t] = 1 - probs[:, t]
                else:
                    survival_probs[:, t] = survival_probs[:, t-1] * (1 - probs[:, t])

        return survival_probs


def create_data_loaders(X_train: np.ndarray, t_train: np.ndarray, e_train: np.ndarray,
                       X_val: Optional[np.ndarray] = None,
                       t_val: Optional[np.ndarray] = None,
                       e_val: Optional[np.ndarray] = None,
                       batch_size: int = 64,
                       num_workers: int = 0) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for survival analysis.

    Args:
        X_train: Training features
        t_train: Training times
        e_train: Training events
        X_val: Validation features (optional)
        t_val: Validation times (optional)
        e_val: Validation events (optional)
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Dictionary of DataLoaders
    """
    train_dataset = SurvivalDataset(X_train, t_train, e_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)

    loaders = {'train': train_loader}

    if X_val is not None:
        val_dataset = SurvivalDataset(X_val, t_val, e_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
        loaders['val'] = val_loader

    return loaders


def get_model_class(model_name: str):
    """
    Get model class by name.

    Args:
        model_name: Name of the model ('deepsurv', 'deephit', 'mtlr')

    Returns:
        Model class
    """
    model_classes = {
        'deepsurv': DeepSurv,
        'deephit': DeepHit,
        'mtlr': NeuralMTLR
    }

    if model_name.lower() not in model_classes:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_classes.keys())}")

    return model_classes[model_name.lower()]