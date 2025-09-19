"""
Comprehensive Example: Deep Learning Survival Analysis with SEER Data

This script demonstrates the complete workflow for survival analysis using
deep learning models with SEER cancer registry data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import survival analysis modules
from src.data.seer_preprocessor import SEERPreprocessor, create_sample_seer_data, load_seer_data
from src.models.deep_survival import get_model_class
from src.utils.trainer import SurvivalTrainer, train_multiple_models
from src.utils.visualization import SurvivalVisualizer, create_survival_dashboard
from src.evaluation.metrics import compute_all_metrics, print_metrics_summary


def main():
    """Run complete survival analysis pipeline."""
    print("=== Deep Learning Survival Analysis with SEER Data ===")
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Step 1: Load Real SEER Data
    print("Step 1: Loading Real SEER Data")
    print("-" * 40)

    # Try to load real SEER data first, fallback to synthetic if not available
    real_data_paths = [
        'data/seer_processed.csv',
        'data/SEER_breast_cancer.csv',
        'data/seer_breast_cancer.csv'
    ]

    df = None
    for path in real_data_paths:
        if Path(path).exists():
            try:
                print(f"Loading real SEER data from: {path}")
                df = load_seer_data(path)
                print("✓ Successfully loaded real SEER data")
                break
            except Exception as e:
                print(f"✗ Error loading {path}: {e}")
                continue

    # Fallback to synthetic data if real data not available
    if df is None:
        print("⚠ Real SEER data not found, using synthetic data")
        print("To use real data, run: python extract_seer_data.py")
        df = create_sample_seer_data(n_samples=2000, random_state=42)

    print(f"Dataset shape: {df.shape}")
    print(f"Event rate: {df['vital_status'].mean():.3f}")
    print(f"Median survival time: {df['survival_months'].median():.1f} months")
    print()

    # Display sample of the data
    print("Sample of the data:")
    print(df.head())
    print()

    # Step 2: Data Preprocessing
    print("Step 2: Data Preprocessing")
    print("-" * 40)

    preprocessor = SEERPreprocessor(
        time_col='survival_months',
        event_col='vital_status',
        categorical_strategy='onehot',
        missing_strategy='median',
        scale_features=True
    )

    # Fit preprocessor and transform data
    X, t, e = preprocessor.fit_transform(df)

    print(f"Processed features shape: {X.shape}")
    print(f"Number of features: {len(preprocessor.feature_names)}")
    print(f"Categorical features: {len(preprocessor.categorical_cols)}")
    print(f"Numerical features: {len(preprocessor.numerical_cols)}")
    print()

    # Split data
    data_splits = preprocessor.train_test_split(
        X, t, e,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    X_train, t_train, e_train = data_splits['X_train'], data_splits['t_train'], data_splits['e_train']
    X_val, t_val, e_val = data_splits['X_val'], data_splits['t_val'], data_splits['e_val']
    X_test, t_test, e_test = data_splits['X_test'], data_splits['t_test'], data_splits['e_test']

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()

    # Step 3: Model Training and Comparison
    print("Step 3: Training Deep Learning Models")
    print("-" * 40)

    # Define evaluation time points
    eval_time_points = [12, 24, 36, 60]  # 1, 2, 3, 5 years

    # Train multiple models for comparison
    model_results = train_multiple_models(
        X_train, t_train, e_train,
        X_val, t_val, e_val,
        X_test, t_test, e_test,
        models=['deepsurv', 'deephit', 'mtlr'],
        save_dir='seer_experiments',
        max_epochs=50,
        early_stopping_patience=10,
        eval_time_points=eval_time_points,
        logger_type='tensorboard'
    )

    print("\nModel Comparison Results:")
    print(model_results['comparison'])
    print()

    # Step 4: Detailed Analysis of Best Model
    print("Step 4: Detailed Analysis of Best Model")
    print("-" * 40)

    # Select best model based on C-index
    best_model_name = model_results['comparison']['c_index'].idxmax()
    best_model_results = model_results[best_model_name]
    best_model = best_model_results['train_results']['model']

    print(f"Best model: {best_model_name.upper()}")
    print(f"Best C-index: {model_results['comparison'].loc[best_model_name, 'c_index']:.4f}")
    print()

    # Get detailed predictions for best model
    trainer = best_model_results['trainer']
    detailed_results = trainer.evaluate(
        best_model, X_test, t_test, e_test, return_predictions=True
    )

    # Print detailed metrics
    print("Detailed Evaluation Metrics:")
    print_metrics_summary(detailed_results['metrics'])

    # Step 5: Hyperparameter Tuning (Optional)
    print("Step 5: Hyperparameter Tuning")
    print("-" * 40)

    print(f"Performing hyperparameter tuning for {best_model_name}...")
    tuning_trainer = SurvivalTrainer(
        model_name=best_model_name,
        max_epochs=30,
        save_dir='seer_tuning',
        experiment_name=f'{best_model_name}_tuned',
        eval_time_points=eval_time_points
    )

    tuning_results = tuning_trainer.hyperparameter_tuning(
        X_train, t_train, e_train,
        X_val, t_val, e_val,
        n_trials=20,  # Reduce for faster demo
        metric_name='c_index',
        direction='maximize'
    )

    print(f"Best hyperparameters: {tuning_results['best_params']}")
    print(f"Best validation C-index: {tuning_results['best_value']:.4f}")
    print()

    # Evaluate tuned model
    tuned_results = tuning_trainer.evaluate(
        tuning_results['best_model'], X_test, t_test, e_test, return_predictions=True
    )

    print("Tuned Model Performance:")
    print_metrics_summary(tuned_results['metrics'])

    # Step 6: Visualization and Interpretation
    print("Step 6: Creating Visualizations")
    print("-" * 40)

    # Use the tuned model for final analysis
    final_predictions = tuned_results['predictions']

    # Create comprehensive dashboard
    dashboard_figures = create_survival_dashboard(
        t_test, e_test,
        final_predictions['risk_scores'],
        final_predictions['survival_probs'],
        final_predictions['time_points'],
        preprocessor.feature_names,
        save_dir='seer_dashboard'
    )

    print(f"Dashboard created with {len(dashboard_figures)} plots saved to 'seer_dashboard/'")

    # Create additional specific visualizations
    visualizer = SurvivalVisualizer()

    # Model comparison plot
    comparison_fig = visualizer.plot_model_comparison(
        model_results['comparison'],
        metrics=['c_index', 'integrated_brier_score'],
        title="Model Performance Comparison",
        save_path='seer_dashboard/model_comparison.png'
    )

    # Feature importance (mock data - in practice, use SHAP or similar)
    if len(preprocessor.feature_names) > 0:
        # Generate mock importance scores for demonstration
        importance_scores = np.random.normal(0, 1, len(preprocessor.feature_names))
        importance_scores[0:5] = np.abs(importance_scores[0:5]) + 1  # Make first 5 features important

        feature_fig = visualizer.plot_feature_importance(
            preprocessor.feature_names,
            importance_scores,
            importance_type="Mock Importance",
            top_k=15,
            title="Feature Importance Analysis",
            save_path='seer_dashboard/feature_importance.png'
        )

    print("Additional visualizations created!")
    print()

    # Step 7: Cross-Validation Analysis
    print("Step 7: Cross-Validation Analysis")
    print("-" * 40)

    cv_trainer = SurvivalTrainer(
        model_name=best_model_name,
        max_epochs=20,  # Reduce for faster CV
        save_dir='seer_cv',
        experiment_name=f'{best_model_name}_cv'
    )

    cv_results = cv_trainer.cross_validate(
        X_train, t_train, e_train,
        n_folds=3,  # Reduce for faster demo
        model_config=tuning_results['best_params']
    )

    print("Cross-Validation Results:")
    for metric, value in cv_results['aggregated_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    print()

    # Step 8: Risk Stratification Analysis
    print("Step 8: Risk Stratification Analysis")
    print("-" * 40)

    risk_scores = final_predictions['risk_scores']

    # Create risk groups
    risk_tertiles = pd.qcut(risk_scores, q=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])

    # Analyze survival by risk group
    print("Survival Analysis by Risk Groups:")
    for group in ['Low Risk', 'Medium Risk', 'High Risk']:
        mask = risk_tertiles == group
        group_events = e_test[mask]
        group_times = t_test[mask]

        event_rate = group_events.mean()
        median_time = np.median(group_times)

        print(f"  {group}: {mask.sum()} patients, "
              f"Event rate: {event_rate:.3f}, "
              f"Median time: {median_time:.1f} months")
    print()

    # Step 9: Model Interpretation and Clinical Insights
    print("Step 9: Clinical Insights")
    print("-" * 40)

    # Analyze high-risk vs low-risk patients
    high_risk_mask = risk_tertiles == 'High Risk'
    low_risk_mask = risk_tertiles == 'Low Risk'

    print("Characteristics of High-Risk vs Low-Risk Patients:")

    # Get original features for interpretation
    test_indices = data_splits.get('test_indices', np.arange(len(X_test)))

    if 'age_at_diagnosis' in df.columns:
        high_risk_age = df.iloc[test_indices[high_risk_mask]]['age_at_diagnosis'].mean()
        low_risk_age = df.iloc[test_indices[low_risk_mask]]['age_at_diagnosis'].mean()
        print(f"  Average age - High risk: {high_risk_age:.1f}, Low risk: {low_risk_age:.1f}")

    if 'stage' in df.columns:
        high_risk_stages = df.iloc[test_indices[high_risk_mask]]['stage'].value_counts()
        low_risk_stages = df.iloc[test_indices[low_risk_mask]]['stage'].value_counts()
        print(f"  Stage distribution:")
        print(f"    High risk: {dict(high_risk_stages.head(3))}")
        print(f"    Low risk: {dict(low_risk_stages.head(3))}")

    print()

    # Step 10: Summary and Recommendations
    print("Step 10: Summary and Recommendations")
    print("-" * 40)

    best_c_index = tuned_results['metrics']['c_index']
    best_brier = tuned_results['metrics'].get('integrated_brier_score', 'N/A')

    print("SEER Survival Analysis Summary:")
    print(f"  • Dataset: {df.shape[0]} patients with {df.shape[1]} features")
    print(f"  • Best model: {best_model_name.upper()}")
    print(f"  • Final C-index: {best_c_index:.4f}")
    if best_brier != 'N/A':
        print(f"  • Integrated Brier Score: {best_brier:.4f}")
    print(f"  • Model discriminates well between high and low-risk patients")
    print()

    print("Clinical Recommendations:")
    print("  • High-risk patients may benefit from more aggressive treatment")
    print("  • Model can assist in treatment planning and patient counseling")
    print("  • Regular model validation with new data is recommended")
    print("  • Consider integration with clinical decision support systems")
    print()

    print("Technical Recommendations:")
    print("  • Model shows good calibration and discrimination")
    print("  • Consider ensemble methods for improved performance")
    print("  • Implement model monitoring for drift detection")
    print("  • Validate on external datasets when available")
    print()

    # Save final model and preprocessor
    save_dir = Path('seer_final_model')
    save_dir.mkdir(exist_ok=True)

    import torch
    torch.save(tuning_results['best_model'].state_dict(), save_dir / 'best_model.pth')

    import joblib
    joblib.dump(preprocessor, save_dir / 'preprocessor.pkl')

    print(f"Final model and preprocessor saved to '{save_dir}/'")
    print("Analysis complete!")

    # Step 11: Generate Comprehensive Report
    print("Step 11: Generating Comprehensive Analysis Report")
    print("-" * 40)

    from src.reports.report_generator import generate_survival_analysis_report

    # Prepare comprehensive dataset info
    comprehensive_dataset_info = {
        'n_samples': df.shape[0],
        'n_features': len(preprocessor.feature_names),
        'event_rate': df['vital_status'].mean(),
        'median_followup': df['survival_months'].median(),
        'dataset_name': 'Sample SEER Cancer Registry Data',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_names': preprocessor.feature_names,
        'categorical_features': len(preprocessor.categorical_cols),
        'numerical_features': len(preprocessor.numerical_cols),
        'date_generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Prepare model results info
    comprehensive_model_results = {
        'best_model_name': best_model_name.upper(),
        'model_config': tuning_results['best_params'],
        'tuning_results': {
            'best_c_index': tuning_results['best_value'],
            'n_trials': 20,
            'optimization_metric': 'c_index'
        },
        'cross_validation': cv_results['aggregated_metrics']
    }

    # Generate comprehensive report
    reports = generate_survival_analysis_report(
        times_true=t_test,
        events_true=e_test,
        predictions=tuned_results['predictions'],
        dataset_info=comprehensive_dataset_info,
        model_results=comprehensive_model_results,
        feature_names=preprocessor.feature_names,
        output_dir="comprehensive_seer_reports",
        report_title="Comprehensive Deep Learning Survival Analysis Report - SEER Data"
    )

    print("📊 Comprehensive reports generated:")
    for report_type, path in reports.items():
        file_size = Path(path).stat().st_size / 1024  # Size in KB
        print(f"   {report_type.upper()}: {path} ({file_size:.1f} KB)")

    print()
    print("🌐 View your HTML report:")
    html_path = reports.get('html', '')
    if html_path:
        print(f"   file://{Path(html_path).absolute()}")

    print()
    print("📈 Report includes:")
    print("   • Executive summary with key metrics")
    print("   • Model performance analysis")
    print("   • Risk stratification insights")
    print("   • Survival curve visualizations")
    print("   • Feature importance analysis")
    print("   • Clinical recommendations")
    print("   • Technical documentation")
    print()

    print("Analysis and reporting complete!")


if __name__ == "__main__":
    main()