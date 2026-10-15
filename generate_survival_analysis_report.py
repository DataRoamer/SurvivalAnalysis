"""
Generate Comprehensive Survival Analysis Report

This script generates detailed HTML, PDF, and Excel reports from survival analysis results.
Run this after training your survival models to get professional analysis reports.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import survival analysis modules
from src.data.survival_preprocessor import SurvivalDataPreprocessor, create_synthetic_survival_data, load_survival_data
from src.utils.trainer import SurvivalTrainer
from src.reports.report_generator import generate_survival_analysis_report


def main():
    """Generate comprehensive survival analysis report with sample data."""
    print("=== Generating Survival Analysis Report ===")
    print()

    # Step 1: Load real survival data
    print("Step 1: Loading survival data...")

    # Try to load real survival data first, fallback to synthetic if not available
    real_data_paths = ['data/survival_data.csv']

    df = None
    for path in real_data_paths:
        if Path(path).exists():
            try:
                print(f"Loading real survival data from: {path}")
                df = load_survival_data(path)
                print("✓ Successfully loaded real survival data")
                break
            except Exception as e:
                print(f"✗ Error loading {path}: {e}")
                continue

    # Fallback to synthetic data if real data not available
    is_real_data = True
    if df is None:
        print("⚠ Real survival data not found, using synthetic data")
        print("To use your own data, save it as: data/survival_data.csv")
        df = create_synthetic_survival_data(n_samples=1500, random_state=42)
        is_real_data = False

    dataset_info = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 2,  # Exclude time and event columns
        'event_rate': df['vital_status'].mean(),
        'median_followup': df['survival_months'].median(),
        'dataset_name': 'Real Data' if is_real_data else 'Synthetic Survival Data'
    }

    print(f"Dataset created: {dataset_info['n_samples']} patients, {dataset_info['n_features']} features")
    print(f"Event rate: {dataset_info['event_rate']:.1%}")
    print()

    # Step 2: Preprocess data
    print("Step 2: Preprocessing data...")
    preprocessor = SurvivalDataPreprocessor(
        time_col='survival_months',
        event_col='vital_status',
        categorical_strategy='onehot',
        missing_strategy='median',
        scale_features=True
    )

    X, t, e = preprocessor.fit_transform(df)

    # Split data
    data_splits = preprocessor.train_test_split(X, t, e, test_size=0.3, random_state=42)
    X_train, t_train, e_train = data_splits['X_train'], data_splits['t_train'], data_splits['e_train']
    X_test, t_test, e_test = data_splits['X_test'], data_splits['t_test'], data_splits['e_test']

    print(f"Training set: {len(X_train)} patients")
    print(f"Test set: {len(X_test)} patients")
    print()

    # Step 3: Train a survival model
    print("Step 3: Training DeepSurv model...")
    trainer = SurvivalTrainer(
        model_name='deepsurv',
        max_epochs=30,  # Reduced for faster demo
        early_stopping_patience=10,
        learning_rate=1e-3,
        batch_size=64,
        save_dir='report_demo_models',
        experiment_name='deepsurv_for_report',
        logger_type='none'  # Disable logging for demo
    )

    # Train model
    train_results = trainer.train(X_train, t_train, e_train)

    # Evaluate on test set
    test_results = trainer.evaluate(
        train_results['model'],
        X_test, t_test, e_test,
        return_predictions=True
    )

    print("Model training completed!")
    print(f"Test C-index: {test_results['metrics']['c_index']:.4f}")
    print()

    # Step 4: Prepare report data
    print("Step 4: Preparing report data...")

    # Enhanced dataset info
    dataset_info.update({
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_names': preprocessor.feature_names,
        'categorical_features': len(preprocessor.categorical_cols),
        'numerical_features': len(preprocessor.numerical_cols)
    })

    # Model results info
    model_results = {
        'model_name': 'DeepSurv',
        'model_config': {
            'model_type': 'DeepSurv',
            'hidden_layers': [64, 32],
            'dropout': 0.5,
            'learning_rate': 1e-3,
            'training_epochs': 30
        },
        'training_metrics': train_results.get('training_metrics', {}),
        'validation_metrics': train_results.get('validation_metrics', {})
    }

    # Predictions data
    predictions = test_results['predictions']

    print("Data preparation completed!")
    print()

    # Step 5: Generate comprehensive reports
    print("Step 5: Generating comprehensive reports...")

    reports = generate_survival_analysis_report(
        times_true=t_test,
        events_true=e_test,
        predictions=predictions,
        dataset_info=dataset_info,
        model_results=model_results,
        feature_names=preprocessor.feature_names,
        output_dir="survival_analysis_reports",
        report_title="Deep Learning Survival Analysis Report - Synthetic Survival Data"
    )

    print()
    print("✅ Report generation completed successfully!")
    print()
    print("📁 Generated Reports:")
    for report_type, path in reports.items():
        file_size = Path(path).stat().st_size / 1024  # Size in KB
        print(f"   {report_type.upper()}: {path} ({file_size:.1f} KB)")

    print()
    print("📊 Report Contents:")
    print("   • Dataset overview and statistics")
    print("   • Model performance metrics (C-index, Brier score, AUC)")
    print("   • Risk stratification analysis")
    print("   • Kaplan-Meier survival curves by risk groups")
    print("   • Model calibration plots")
    print("   • Feature importance analysis")
    print("   • Clinical insights and recommendations")
    print("   • Technical summary tables")

    print()
    print("🌐 To view the HTML report:")
    html_path = reports.get('html', '')
    if html_path:
        print(f"   Open: file://{Path(html_path).absolute()}")

    print()
    print("📈 Key Findings:")
    c_index = test_results['metrics']['c_index']
    if c_index > 0.75:
        print(f"   🟢 Excellent model performance (C-index: {c_index:.3f})")
    elif c_index > 0.65:
        print(f"   🟡 Good model performance (C-index: {c_index:.3f})")
    else:
        print(f"   🔴 Moderate model performance (C-index: {c_index:.3f})")

    # Risk stratification summary
    risk_scores = predictions['risk_scores']
    risk_groups = pd.cut(risk_scores, bins=3, labels=['Low', 'Medium', 'High'])

    for group in ['Low', 'Medium', 'High']:
        mask = risk_groups == group
        event_rate = e_test[mask].mean() if mask.sum() > 0 else 0
        print(f"   • {group} risk group: {mask.sum()} patients, {event_rate:.1%} event rate")

    print()
    print("🔬 Clinical Recommendations:")
    print("   • Use high-risk scores for treatment intensification")
    print("   • Implement risk-based follow-up schedules")
    print("   • Consider model integration in clinical workflows")
    print("   • Validate with external datasets when available")


def generate_quick_report():
    """Generate a quick report with minimal data for demonstration."""
    print("=== Quick Report Generation Demo ===")
    print()

    # Create minimal synthetic data
    np.random.seed(42)
    n_patients = 200

    # Simulate survival data
    times_true = np.random.exponential(24, n_patients)  # Average 24 months
    events_true = np.random.binomial(1, 0.3, n_patients)  # 30% event rate
    risk_scores = np.random.normal(0, 1, n_patients)

    # Create simple dataset info
    dataset_info = {
        'n_samples': n_patients,
        'n_features': 10,
        'event_rate': events_true.mean(),
        'median_followup': np.median(times_true),
        'dataset_name': 'Demo Data'
    }

    # Create predictions
    predictions = {
        'risk_scores': risk_scores,
        'survival_probs': None,
        'time_points': None
    }

    # Generate report
    reports = generate_survival_analysis_report(
        times_true=times_true,
        events_true=events_true,
        predictions=predictions,
        dataset_info=dataset_info,
        output_dir="quick_demo_reports",
        report_title="Quick Demo Survival Analysis Report"
    )

    print("Quick demo report generated!")
    for report_type, path in reports.items():
        print(f"   {report_type}: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate survival analysis reports')
    parser.add_argument('--quick', action='store_true',
                       help='Generate quick demo report with minimal data')
    parser.add_argument('--full', action='store_true',
                       help='Generate full report with model training (default)')

    args = parser.parse_args()

    if args.quick:
        generate_quick_report()
    else:
        main()  # Default to full report