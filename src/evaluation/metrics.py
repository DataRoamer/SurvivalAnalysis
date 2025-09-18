"""
Evaluation Metrics for Survival Analysis

This module provides comprehensive evaluation metrics for survival analysis models,
including concordance index, time-dependent AUC, Brier score, and calibration metrics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any, Union
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
import warnings


def harrell_c_index(times_true: np.ndarray,
                   events_true: np.ndarray,
                   risk_scores: np.ndarray) -> float:
    """
    Compute Harrell's Concordance Index (C-index).

    Args:
        times_true: True survival times
        events_true: Event indicators (1=event, 0=censored)
        risk_scores: Predicted risk scores (higher = higher risk)

    Returns:
        C-index value between 0.5 (random) and 1.0 (perfect)
    """
    try:
        # Use lifelines implementation
        c_index = concordance_index(times_true, -risk_scores, events_true)
        return c_index
    except Exception as e:
        warnings.warn(f"Error computing C-index with lifelines: {e}. Using manual implementation.")
        return manual_c_index(times_true, events_true, risk_scores)


def manual_c_index(times_true: np.ndarray,
                  events_true: np.ndarray,
                  risk_scores: np.ndarray) -> float:
    """
    Manual implementation of Harrell's C-index.

    Args:
        times_true: True survival times
        events_true: Event indicators (1=event, 0=censored)
        risk_scores: Predicted risk scores (higher = higher risk)

    Returns:
        C-index value
    """
    n = len(times_true)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Only consider pairs where we can make a comparison
            if times_true[i] != times_true[j]:
                # Determine which patient had the event first
                if times_true[i] < times_true[j]:
                    if events_true[i] == 1:  # Patient i had event
                        # Patient i should have higher risk score
                        if risk_scores[i] > risk_scores[j]:
                            concordant += 1
                        elif risk_scores[i] < risk_scores[j]:
                            discordant += 1
                elif times_true[j] < times_true[i]:
                    if events_true[j] == 1:  # Patient j had event
                        # Patient j should have higher risk score
                        if risk_scores[j] > risk_scores[i]:
                            concordant += 1
                        elif risk_scores[j] < risk_scores[i]:
                            discordant += 1

    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 0.5  # Random performance when no comparable pairs

    return concordant / total_pairs


def time_dependent_auc(times_true: np.ndarray,
                      events_true: np.ndarray,
                      risk_scores: np.ndarray,
                      time_point: float) -> float:
    """
    Compute time-dependent AUC at a specific time point.

    Args:
        times_true: True survival times
        events_true: Event indicators
        risk_scores: Predicted risk scores
        time_point: Time point for AUC computation

    Returns:
        Time-dependent AUC value
    """
    # Create binary outcome: event before time_point
    binary_outcome = (times_true <= time_point) & (events_true == 1)

    # Only include patients who either had event before time_point
    # or were observed beyond time_point
    valid_mask = (times_true <= time_point) | (times_true > time_point)

    if not valid_mask.any() or binary_outcome.sum() == 0:
        return np.nan

    try:
        auc = roc_auc_score(binary_outcome[valid_mask], risk_scores[valid_mask])
        return auc
    except ValueError:
        return np.nan


def integrated_brier_score(times_true: np.ndarray,
                          events_true: np.ndarray,
                          survival_probs: np.ndarray,
                          time_points: np.ndarray) -> Dict[str, float]:
    """
    Compute Integrated Brier Score.

    Args:
        times_true: True survival times
        events_true: Event indicators
        survival_probs: Predicted survival probabilities (n_samples, n_time_points)
        time_points: Time points corresponding to survival probabilities

    Returns:
        Dictionary with Brier scores and integrated Brier score
    """
    n_times = len(time_points)
    brier_scores = np.zeros(n_times)

    for i, t in enumerate(time_points):
        brier_scores[i] = brier_score_at_time(
            times_true, events_true, survival_probs[:, i], t
        )

    # Integrate using trapezoidal rule
    if n_times > 1:
        integrated_bs = np.trapz(brier_scores, time_points) / (time_points[-1] - time_points[0])
    else:
        integrated_bs = brier_scores[0]

    return {
        'time_points': time_points,
        'brier_scores': brier_scores,
        'integrated_brier_score': integrated_bs
    }


def brier_score_at_time(times_true: np.ndarray,
                       events_true: np.ndarray,
                       survival_probs: np.ndarray,
                       time_point: float) -> float:
    """
    Compute Brier score at a specific time point.

    Args:
        times_true: True survival times
        events_true: Event indicators
        survival_probs: Predicted survival probabilities at time_point
        time_point: Time point for evaluation

    Returns:
        Brier score at the specified time point
    """
    # True survival status at time_point
    y_true = (times_true > time_point).astype(float)

    # Weights for inverse probability of censoring weighting (IPCW)
    weights = compute_ipcw_weights(times_true, events_true, time_point)

    # Brier score with IPCW
    brier = np.mean(weights * (y_true - survival_probs) ** 2)

    return brier


def compute_ipcw_weights(times_true: np.ndarray,
                        events_true: np.ndarray,
                        time_point: float) -> np.ndarray:
    """
    Compute Inverse Probability of Censoring Weights (IPCW).

    Args:
        times_true: True survival times
        events_true: Event indicators (1=event, 0=censored)
        time_point: Time point for weight computation

    Returns:
        IPCW weights
    """
    n = len(times_true)
    weights = np.zeros(n)

    # Estimate censoring distribution using Kaplan-Meier
    # Censoring times and indicators (reverse of events)
    censor_times = times_true.copy()
    censor_events = 1 - events_true  # 1=censored, 0=event

    # Sort by time
    sorted_indices = np.argsort(censor_times)
    sorted_times = censor_times[sorted_indices]
    sorted_events = censor_events[sorted_indices]

    # Kaplan-Meier estimation for censoring distribution
    unique_times = np.unique(sorted_times[sorted_events == 1])
    km_probs = np.ones(len(unique_times))

    n_at_risk = n
    for i, t in enumerate(unique_times):
        n_events = np.sum((sorted_times == t) & (sorted_events == 1))
        if n_at_risk > 0:
            km_probs[i] = (n_at_risk - n_events) / n_at_risk
            if i > 0:
                km_probs[i] *= km_probs[i-1]
        n_at_risk -= np.sum(sorted_times == t)

    # Compute weights
    for i in range(n):
        t_i = times_true[i]
        event_i = events_true[i]

        if t_i <= time_point:
            if event_i == 1:  # Event occurred
                # Find survival probability at t_i
                censoring_prob = 1.0
                for j, t_cens in enumerate(unique_times):
                    if t_cens <= t_i:
                        censoring_prob = km_probs[j]
                    else:
                        break
                weights[i] = 1.0 / max(censoring_prob, 1e-8)
            else:  # Censored before time_point
                weights[i] = 0.0
        else:  # Observed beyond time_point
            # Find survival probability at time_point
            censoring_prob = 1.0
            for j, t_cens in enumerate(unique_times):
                if t_cens <= time_point:
                    censoring_prob = km_probs[j]
                else:
                    break
            weights[i] = 1.0 / max(censoring_prob, 1e-8)

    return weights


def calibration_slope(times_true: np.ndarray,
                     events_true: np.ndarray,
                     risk_scores: np.ndarray,
                     time_point: float,
                     n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute calibration slope and calibration plot data.

    Args:
        times_true: True survival times
        events_true: Event indicators
        risk_scores: Predicted risk scores
        time_point: Time point for calibration assessment
        n_bins: Number of bins for calibration plot

    Returns:
        Dictionary with calibration metrics and plot data
    """
    # Convert risk scores to predicted probabilities (using sigmoid as example)
    pred_probs = 1 / (1 + np.exp(-risk_scores))

    # True event status at time_point
    y_true = (times_true <= time_point) & (events_true == 1)

    # Only consider patients with sufficient follow-up
    valid_mask = (times_true > time_point) | y_true
    if not valid_mask.any():
        return {'calibration_slope': np.nan, 'calibration_data': None}

    pred_probs_valid = pred_probs[valid_mask]
    y_true_valid = y_true[valid_mask]

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    observed_rates = []
    predicted_rates = []
    bin_sizes = []

    for i in range(n_bins):
        bin_mask = (pred_probs_valid >= bin_edges[i]) & (pred_probs_valid < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            bin_mask = (pred_probs_valid >= bin_edges[i]) & (pred_probs_valid <= bin_edges[i + 1])

        if bin_mask.any():
            observed_rate = y_true_valid[bin_mask].mean()
            predicted_rate = pred_probs_valid[bin_mask].mean()
            bin_size = bin_mask.sum()

            observed_rates.append(observed_rate)
            predicted_rates.append(predicted_rate)
            bin_sizes.append(bin_size)
        else:
            observed_rates.append(np.nan)
            predicted_rates.append(np.nan)
            bin_sizes.append(0)

    # Compute calibration slope (linear regression)
    valid_bins = ~np.isnan(observed_rates) & ~np.isnan(predicted_rates)
    if valid_bins.sum() >= 2:
        obs_valid = np.array(observed_rates)[valid_bins]
        pred_valid = np.array(predicted_rates)[valid_bins]

        # Weighted linear regression
        weights = np.array(bin_sizes)[valid_bins]
        slope = np.cov(pred_valid, obs_valid, aweights=weights)[0, 1] / np.var(pred_valid, aweights=weights)
    else:
        slope = np.nan

    return {
        'calibration_slope': slope,
        'calibration_data': {
            'bin_centers': bin_centers,
            'observed_rates': observed_rates,
            'predicted_rates': predicted_rates,
            'bin_sizes': bin_sizes
        }
    }


def survival_prediction_error(times_true: np.ndarray,
                             events_true: np.ndarray,
                             survival_curves: np.ndarray,
                             time_points: np.ndarray) -> float:
    """
    Compute Survival Prediction Error (SPE).

    Args:
        times_true: True survival times
        events_true: Event indicators
        survival_curves: Predicted survival curves (n_samples, n_time_points)
        time_points: Time points for survival curves

    Returns:
        Survival prediction error
    """
    n_samples = len(times_true)
    errors = []

    for i in range(n_samples):
        t_true = times_true[i]
        event = events_true[i]

        # Find the time point closest to true time
        closest_idx = np.argmin(np.abs(time_points - t_true))
        t_pred = time_points[closest_idx]

        if event == 1:  # Event occurred
            # Error is predicted survival probability at event time
            error = survival_curves[i, closest_idx]
        else:  # Censored
            # Error is 1 - predicted survival probability at censoring time
            error = 1 - survival_curves[i, closest_idx]

        errors.append(error)

    return np.mean(errors)


def compute_all_metrics(times_true: np.ndarray,
                       events_true: np.ndarray,
                       risk_scores: Optional[np.ndarray] = None,
                       survival_probs: Optional[np.ndarray] = None,
                       time_points: Optional[np.ndarray] = None,
                       eval_time_points: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Compute all available survival analysis metrics.

    Args:
        times_true: True survival times
        events_true: Event indicators
        risk_scores: Predicted risk scores (for C-index, time-dependent AUC)
        survival_probs: Predicted survival probabilities (n_samples, n_time_points)
        time_points: Time points corresponding to survival probabilities
        eval_time_points: Specific time points for evaluation

    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}

    # C-index (requires risk scores)
    if risk_scores is not None:
        metrics['c_index'] = harrell_c_index(times_true, events_true, risk_scores)

    # Time-dependent metrics
    if eval_time_points is None and time_points is not None:
        eval_time_points = [time_points[len(time_points)//4],
                          time_points[len(time_points)//2],
                          time_points[3*len(time_points)//4]]

    if eval_time_points is not None:
        td_aucs = {}
        calibration_data = {}

        for t in eval_time_points:
            if risk_scores is not None:
                td_aucs[f'auc_{t}'] = time_dependent_auc(times_true, events_true, risk_scores, t)
                cal_result = calibration_slope(times_true, events_true, risk_scores, t)
                calibration_data[f'calibration_{t}'] = cal_result

        metrics.update(td_aucs)
        metrics['calibration'] = calibration_data

    # Brier score and related metrics (requires survival probabilities)
    if survival_probs is not None and time_points is not None:
        brier_result = integrated_brier_score(times_true, events_true, survival_probs, time_points)
        metrics.update(brier_result)

        # Survival prediction error
        metrics['survival_prediction_error'] = survival_prediction_error(
            times_true, events_true, survival_probs, time_points
        )

    return metrics


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted summary of computed metrics.

    Args:
        metrics: Dictionary of computed metrics
    """
    print("=== Survival Analysis Evaluation Metrics ===")
    print()

    # C-index
    if 'c_index' in metrics:
        print(f"Harrell's C-index: {metrics['c_index']:.4f}")
        print()

    # Time-dependent AUCs
    auc_keys = [k for k in metrics.keys() if k.startswith('auc_')]
    if auc_keys:
        print("Time-dependent AUCs:")
        for key in sorted(auc_keys):
            time_point = key.split('_')[1]
            print(f"  AUC at t={time_point}: {metrics[key]:.4f}")
        print()

    # Brier scores
    if 'integrated_brier_score' in metrics:
        print(f"Integrated Brier Score: {metrics['integrated_brier_score']:.4f}")
        print()

    # Survival prediction error
    if 'survival_prediction_error' in metrics:
        print(f"Survival Prediction Error: {metrics['survival_prediction_error']:.4f}")
        print()

    # Calibration
    if 'calibration' in metrics:
        print("Calibration Slopes:")
        for key, cal_data in metrics['calibration'].items():
            time_point = key.split('_')[1]
            slope = cal_data['calibration_slope']
            if not np.isnan(slope):
                print(f"  Calibration slope at t={time_point}: {slope:.4f}")
        print()


class SurvivalMetricsTracker:
    """Track survival analysis metrics during training."""

    def __init__(self, eval_time_points: List[float]):
        """
        Initialize metrics tracker.

        Args:
            eval_time_points: Time points for evaluation
        """
        self.eval_time_points = eval_time_points
        self.metrics_history = []

    def update(self,
               times_true: np.ndarray,
               events_true: np.ndarray,
               risk_scores: Optional[np.ndarray] = None,
               survival_probs: Optional[np.ndarray] = None,
               time_points: Optional[np.ndarray] = None,
               epoch: int = None) -> Dict[str, Any]:
        """
        Compute and track metrics for current epoch.

        Args:
            times_true: True survival times
            events_true: Event indicators
            risk_scores: Predicted risk scores
            survival_probs: Predicted survival probabilities
            time_points: Time points for survival probabilities
            epoch: Current epoch number

        Returns:
            Computed metrics for current epoch
        """
        metrics = compute_all_metrics(
            times_true, events_true, risk_scores,
            survival_probs, time_points, self.eval_time_points
        )

        if epoch is not None:
            metrics['epoch'] = epoch

        self.metrics_history.append(metrics)
        return metrics

    def get_best_epoch(self, metric_name: str = 'c_index', mode: str = 'max') -> int:
        """
        Get epoch with best performance for specified metric.

        Args:
            metric_name: Name of metric to optimize
            mode: 'max' or 'min'

        Returns:
            Best epoch number
        """
        if not self.metrics_history:
            return 0

        metric_values = [m.get(metric_name, np.nan) for m in self.metrics_history]
        valid_values = [(i, v) for i, v in enumerate(metric_values) if not np.isnan(v)]

        if not valid_values:
            return 0

        if mode == 'max':
            best_epoch = max(valid_values, key=lambda x: x[1])[0]
        else:
            best_epoch = min(valid_values, key=lambda x: x[1])[0]

        return best_epoch

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Get metrics history as pandas DataFrame.

        Returns:
            DataFrame with metrics history
        """
        if not self.metrics_history:
            return pd.DataFrame()

        # Flatten nested dictionaries
        flattened_metrics = []
        for metrics in self.metrics_history:
            flat_dict = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, dict):
                            continue  # Skip nested dicts for now
                        flat_dict[f"{key}_{subkey}"] = subvalue
                else:
                    flat_dict[key] = value
            flattened_metrics.append(flat_dict)

        return pd.DataFrame(flattened_metrics)