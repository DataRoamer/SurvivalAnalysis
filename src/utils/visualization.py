"""
Visualization and Interpretation Tools for Survival Analysis

This module provides comprehensive visualization and interpretation capabilities
for deep learning survival analysis models, including survival curves, risk
stratification plots, SHAP analysis, and model performance visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SurvivalVisualizer:
    """Comprehensive visualization toolkit for survival analysis."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn'):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        if style:
            plt.style.use(style)

    def plot_kaplan_meier(self,
                         times: np.ndarray,
                         events: np.ndarray,
                         groups: Optional[np.ndarray] = None,
                         group_names: Optional[List[str]] = None,
                         title: str = "Kaplan-Meier Survival Curves",
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Kaplan-Meier survival curves.

        Args:
            times: Survival times
            events: Event indicators
            groups: Group labels for stratification (optional)
            group_names: Names for groups (optional)
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if groups is None:
            # Single survival curve
            survival_prob, time_points = self._compute_kaplan_meier(times, events)
            ax.step(time_points, survival_prob, where='post', linewidth=2, label='All patients')
        else:
            # Multiple survival curves by group
            unique_groups = np.unique(groups)
            if group_names is None:
                group_names = [f'Group {g}' for g in unique_groups]

            for i, group in enumerate(unique_groups):
                mask = groups == group
                group_times = times[mask]
                group_events = events[mask]

                survival_prob, time_points = self._compute_kaplan_meier(group_times, group_events)
                ax.step(time_points, survival_prob, where='post', linewidth=2,
                       label=f'{group_names[i]} (n={mask.sum()})')

        ax.set_xlabel('Time')
        ax.set_ylabel('Survival Probability')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_predicted_vs_actual_survival(self,
                                        times_true: np.ndarray,
                                        events_true: np.ndarray,
                                        survival_probs: np.ndarray,
                                        time_points: np.ndarray,
                                        risk_groups: Optional[np.ndarray] = None,
                                        n_risk_groups: int = 3,
                                        title: str = "Predicted vs Actual Survival",
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot predicted vs actual survival curves by risk groups.

        Args:
            times_true: True survival times
            events_true: Event indicators
            survival_probs: Predicted survival probabilities (n_samples, n_time_points)
            time_points: Time points for survival probabilities
            risk_groups: Risk group assignments (optional)
            n_risk_groups: Number of risk groups to create if not provided
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if risk_groups is None:
            # Create risk groups based on predicted survival at median time
            median_time_idx = len(time_points) // 2
            median_survival = survival_probs[:, median_time_idx]
            risk_groups = pd.cut(median_survival, bins=n_risk_groups, labels=False)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left plot: Predicted survival curves
        unique_groups = np.unique(risk_groups)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))

        for i, group in enumerate(unique_groups):
            mask = risk_groups == group
            group_survival = survival_probs[mask].mean(axis=0)

            axes[0].plot(time_points, group_survival, linewidth=2,
                        color=colors[i], label=f'Risk Group {group + 1}')

        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Predicted Survival Probability')
        axes[0].set_title('Predicted Survival Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right plot: Actual survival curves (Kaplan-Meier)
        for i, group in enumerate(unique_groups):
            mask = risk_groups == group
            group_times = times_true[mask]
            group_events = events_true[mask]

            if len(group_times) > 0:
                survival_prob, km_time_points = self._compute_kaplan_meier(group_times, group_events)
                axes[1].step(km_time_points, survival_prob, where='post', linewidth=2,
                           color=colors[i], label=f'Risk Group {group + 1} (n={mask.sum()})')

        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Actual Survival Probability')
        axes[1].set_title('Actual Survival Curves (Kaplan-Meier)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_calibration(self,
                        times_true: np.ndarray,
                        events_true: np.ndarray,
                        risk_scores: np.ndarray,
                        time_points: List[float],
                        n_bins: int = 10,
                        title: str = "Calibration Plot",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration curves for different time points.

        Args:
            times_true: True survival times
            events_true: Event indicators
            risk_scores: Predicted risk scores
            time_points: Time points for calibration assessment
            n_bins: Number of bins for calibration
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        from ..evaluation.metrics import calibration_slope

        n_timepoints = len(time_points)
        fig, axes = plt.subplots(1, n_timepoints, figsize=(5 * n_timepoints, 5))

        if n_timepoints == 1:
            axes = [axes]

        for i, t in enumerate(time_points):
            cal_result = calibration_slope(times_true, events_true, risk_scores, t, n_bins)
            cal_data = cal_result['calibration_data']

            if cal_data is not None:
                observed = cal_data['observed_rates']
                predicted = cal_data['predicted_rates']
                bin_sizes = cal_data['bin_sizes']

                # Plot calibration
                valid_mask = ~np.isnan(observed) & ~np.isnan(predicted) & (np.array(bin_sizes) > 0)
                if valid_mask.any():
                    axes[i].scatter(np.array(predicted)[valid_mask],
                                  np.array(observed)[valid_mask],
                                  s=np.array(bin_sizes)[valid_mask] * 5,
                                  alpha=0.7, color='blue')

                # Perfect calibration line
                axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

                axes[i].set_xlabel('Predicted Probability')
                axes[i].set_ylabel('Observed Proportion')
                axes[i].set_title(f'Calibration at t={t:.1f}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlim(0, 1)
                axes[i].set_ylim(0, 1)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_risk_distribution(self,
                             risk_scores: np.ndarray,
                             events: np.ndarray,
                             title: str = "Risk Score Distribution",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of risk scores by event status.

        Args:
            risk_scores: Predicted risk scores
            events: Event indicators
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        event_mask = events == 1
        censored_mask = events == 0

        axes[0].hist(risk_scores[event_mask], bins=30, alpha=0.7, label='Events', density=True)
        axes[0].hist(risk_scores[censored_mask], bins=30, alpha=0.7, label='Censored', density=True)
        axes[0].set_xlabel('Risk Score')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Risk Score Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        data_to_plot = [risk_scores[event_mask], risk_scores[censored_mask]]
        axes[1].boxplot(data_to_plot, labels=['Events', 'Censored'])
        axes[1].set_ylabel('Risk Score')
        axes[1].set_title('Risk Score by Event Status')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_feature_importance(self,
                              feature_names: List[str],
                              importance_scores: np.ndarray,
                              importance_type: str = "SHAP",
                              top_k: int = 20,
                              title: str = "Feature Importance",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance scores.

        Args:
            feature_names: Names of features
            importance_scores: Importance scores
            importance_type: Type of importance (for labeling)
            top_k: Number of top features to show
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        # Sort by importance
        sorted_indices = np.argsort(np.abs(importance_scores))[-top_k:]
        sorted_scores = importance_scores[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_names) * 0.3)))

        colors = ['red' if score < 0 else 'blue' for score in sorted_scores]
        bars = ax.barh(range(len(sorted_names)), sorted_scores, color=colors, alpha=0.7)

        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel(f'{importance_type} Importance Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(score + 0.01 * np.sign(score) if score != 0 else 0.01,
                   bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', va='center', ha='left' if score >= 0 else 'right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_learning_curves(self,
                           metrics_history: pd.DataFrame,
                           metric_names: List[str] = ['train_loss', 'val_loss'],
                           title: str = "Learning Curves",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot learning curves from training history.

        Args:
            metrics_history: DataFrame with training metrics
            metric_names: List of metric names to plot
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metric_names):
            if metric in metrics_history.columns:
                axes[i].plot(metrics_history.index, metrics_history[metric], linewidth=2)
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_model_comparison(self,
                            comparison_df: pd.DataFrame,
                            metrics: List[str] = ['c_index', 'integrated_brier_score'],
                            title: str = "Model Comparison",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of multiple models.

        Args:
            comparison_df: DataFrame with model comparison results
            metrics: List of metrics to compare
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                values = comparison_df[metric].values
                models = comparison_df.index.tolist()

                bars = axes[i].bar(models, values, alpha=0.7)
                axes[i].set_ylabel(metric.replace('_', ' ').title())
                axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3, axis='y')

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_survival_surface(self,
                            features: np.ndarray,
                            survival_probs: np.ndarray,
                            time_points: np.ndarray,
                            feature_names: List[str],
                            n_samples: int = 100,
                            title: str = "Survival Surface",
                            save_path: Optional[str] = None):
        """
        Create interactive 3D survival surface plot using Plotly.

        Args:
            features: Feature matrix
            survival_probs: Predicted survival probabilities
            time_points: Time points
            feature_names: Feature names
            n_samples: Number of samples to plot
            title: Plot title
            save_path: Path to save HTML file

        Returns:
            Plotly figure
        """
        # Sample data for visualization
        sample_indices = np.random.choice(len(features), min(n_samples, len(features)), replace=False)
        sample_features = features[sample_indices]
        sample_survival = survival_probs[sample_indices]

        # Use PCA for dimensionality reduction if needed
        if sample_features.shape[1] > 2:
            pca = PCA(n_components=2)
            sample_features_2d = pca.fit_transform(sample_features)
            feature_names_2d = [f'PC1 ({pca.explained_variance_ratio_[0]:.2f})',
                              f'PC2 ({pca.explained_variance_ratio_[1]:.2f})']
        else:
            sample_features_2d = sample_features
            feature_names_2d = feature_names[:2]

        # Create surface plot
        fig = go.Figure()

        # Add survival curves for each sample
        for i in range(len(sample_indices)):
            fig.add_trace(go.Scatter3d(
                x=np.full_like(time_points, sample_features_2d[i, 0]),
                y=np.full_like(time_points, sample_features_2d[i, 1]),
                z=sample_survival[i],
                mode='lines',
                line=dict(width=2),
                opacity=0.7,
                name=f'Sample {i+1}',
                showlegend=False
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=feature_names_2d[0],
                yaxis_title=feature_names_2d[1],
                zaxis_title='Survival Probability'
            ),
            width=800,
            height=600
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _compute_kaplan_meier(self, times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Kaplan-Meier survival curve.

        Args:
            times: Survival times
            events: Event indicators

        Returns:
            Tuple of (survival_probabilities, time_points)
        """
        # Sort by time
        sorted_indices = np.argsort(times)
        sorted_times = times[sorted_indices]
        sorted_events = events[sorted_indices]

        # Find unique event times
        unique_times = np.unique(sorted_times[sorted_events == 1])

        # Initialize
        n_at_risk = len(times)
        survival_prob = [1.0]
        time_points = [0.0]

        for t in unique_times:
            # Number of events at time t
            n_events = np.sum((sorted_times == t) & (sorted_events == 1))

            # Update survival probability
            if n_at_risk > 0:
                survival_prob.append(survival_prob[-1] * (n_at_risk - n_events) / n_at_risk)
            else:
                survival_prob.append(survival_prob[-1])

            time_points.append(t)

            # Update number at risk
            n_at_risk -= np.sum(sorted_times == t)

        return np.array(survival_prob), np.array(time_points)


def create_survival_dashboard(times_true: np.ndarray,
                            events_true: np.ndarray,
                            risk_scores: np.ndarray,
                            survival_probs: Optional[np.ndarray] = None,
                            time_points: Optional[np.ndarray] = None,
                            feature_names: Optional[List[str]] = None,
                            save_dir: str = 'survival_dashboard') -> Dict[str, plt.Figure]:
    """
    Create comprehensive survival analysis dashboard.

    Args:
        times_true: True survival times
        events_true: Event indicators
        risk_scores: Predicted risk scores
        survival_probs: Predicted survival probabilities (optional)
        time_points: Time points for survival probabilities (optional)
        feature_names: Feature names (optional)
        save_dir: Directory to save plots

    Returns:
        Dictionary of generated figures
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    visualizer = SurvivalVisualizer()
    figures = {}

    # 1. Kaplan-Meier curves by risk groups
    risk_groups = pd.cut(risk_scores, bins=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])
    fig_km = visualizer.plot_kaplan_meier(
        times_true, events_true, risk_groups,
        title="Kaplan-Meier Curves by Risk Groups",
        save_path=os.path.join(save_dir, 'kaplan_meier.png')
    )
    figures['kaplan_meier'] = fig_km

    # 2. Risk score distribution
    fig_risk = visualizer.plot_risk_distribution(
        risk_scores, events_true,
        title="Risk Score Distribution",
        save_path=os.path.join(save_dir, 'risk_distribution.png')
    )
    figures['risk_distribution'] = fig_risk

    # 3. Calibration plot
    eval_times = [np.percentile(times_true, 25), np.percentile(times_true, 50), np.percentile(times_true, 75)]
    fig_cal = visualizer.plot_calibration(
        times_true, events_true, risk_scores, eval_times,
        title="Model Calibration",
        save_path=os.path.join(save_dir, 'calibration.png')
    )
    figures['calibration'] = fig_cal

    # 4. Predicted vs actual survival (if survival probabilities available)
    if survival_probs is not None and time_points is not None:
        fig_pred = visualizer.plot_predicted_vs_actual_survival(
            times_true, events_true, survival_probs, time_points,
            title="Predicted vs Actual Survival",
            save_path=os.path.join(save_dir, 'predicted_vs_actual.png')
        )
        figures['predicted_vs_actual'] = fig_pred

    return figures