# Deep Learning Survival Analysis with SEER Data

A comprehensive deep learning framework for survival analysis using SEER (Surveillance, Epidemiology, and End Results) cancer registry data. This package implements state-of-the-art survival analysis models including DeepSurv, DeepHit, and Neural Multi-Task Logistic Regression (MTLR).

## Features

### 🔬 **Advanced Deep Learning Models**
- **DeepSurv**: Cox proportional hazards deep neural network
- **DeepHit**: Deep learning with competing risks
- **Neural MTLR**: Multi-task logistic regression for survival analysis

### 📊 **Comprehensive Data Pipeline**
- SEER data preprocessing and feature engineering
- Automated handling of missing values and categorical encoding
- Stratified train/validation/test splitting
- Feature scaling and normalization

### 📈 **Robust Evaluation Metrics**
- Harrell's Concordance Index (C-index)
- Time-dependent AUC
- Integrated Brier Score
- Calibration analysis
- Survival prediction error

### 🎯 **Training & Optimization**
- PyTorch Lightning integration
- Hyperparameter tuning with Optuna
- Cross-validation support
- Early stopping and model checkpointing
- TensorBoard and Weights & Biases logging

### 📊 **Rich Visualizations**
- Kaplan-Meier survival curves
- Risk stratification plots
- Model calibration curves
- Feature importance analysis
- Interactive survival surfaces

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SurvivalAnalysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.data.seer_preprocessor import create_sample_seer_data, SEERPreprocessor
from src.utils.trainer import SurvivalTrainer
from src.utils.visualization import create_survival_dashboard

# 1. Load or create SEER data
df = create_sample_seer_data(n_samples=1000)

# 2. Preprocess data
preprocessor = SEERPreprocessor()
X, t, e = preprocessor.fit_transform(df)

# 3. Split data
splits = preprocessor.train_test_split(X, t, e)

# 4. Train model
trainer = SurvivalTrainer(model_name='deepsurv')
results = trainer.train(
    splits['X_train'], splits['t_train'], splits['e_train'],
    splits['X_val'], splits['t_val'], splits['e_val']
)

# 5. Evaluate and visualize
test_results = trainer.evaluate(
    results['model'],
    splits['X_test'], splits['t_test'], splits['e_test'],
    return_predictions=True
)

# Create comprehensive dashboard
dashboard = create_survival_dashboard(
    splits['t_test'], splits['e_test'],
    test_results['predictions']['risk_scores']
)
```

## Comprehensive Example

Run the complete analysis pipeline:

```bash
python example_seer_survival_analysis.py
```

This example demonstrates:
- Data loading and preprocessing
- Model comparison across all three architectures
- Hyperparameter tuning
- Cross-validation
- Risk stratification analysis
- Comprehensive visualizations
- Clinical insights generation

## Model Architectures

### DeepSurv
Cox proportional hazards model with deep neural networks. Optimizes the partial likelihood loss function.

**Key Features:**
- Non-linear risk modeling
- Cox regression foundation
- Handles censored data naturally

### DeepHit
Survival analysis with competing risks using deep learning.

**Key Features:**
- Multiple competing events
- Ranking loss for concordance
- Discrete time modeling

### Neural MTLR
Multi-Task Logistic Regression implemented with neural networks.

**Key Features:**
- Time interval modeling
- Logistic regression foundation
- Flexible time discretization

## Data Structure

### Input Data Format
Your SEER data should include:

| Column | Description | Type |
|--------|-------------|------|
| `survival_months` | Time to event/censoring | Numeric |
| `vital_status` | Event indicator (1=death, 0=alive) | Binary |
| `age_at_diagnosis` | Age at cancer diagnosis | Numeric |
| `primary_site` | Primary cancer site | Categorical |
| `stage` | Cancer stage | Categorical |
| `grade` | Tumor grade | Categorical |
| `race` | Patient race/ethnicity | Categorical |
| `sex` | Patient sex | Categorical |
| `tumor_size` | Tumor size (cm) | Numeric |
| ... | Additional clinical features | Mixed |

### Sample Data Generation
```python
from src.data.seer_preprocessor import create_sample_seer_data

# Generate realistic SEER-like data for testing
df = create_sample_seer_data(
    n_samples=5000,
    random_state=42
)
```

## Evaluation Metrics

### Concordance Index (C-index)
Measures the model's ability to correctly order patients by risk.
- Range: 0.5 (random) to 1.0 (perfect)
- Values > 0.7 indicate good discrimination

### Time-dependent AUC
Area under ROC curve at specific time points.
- Evaluates binary classification at fixed times
- Accounts for censoring properly

### Integrated Brier Score
Measures prediction accuracy across all time points.
- Lower values indicate better predictions
- Accounts for censoring using IPCW

### Calibration
Assesses agreement between predicted and observed event rates.
- Perfect calibration: slope = 1, intercept = 0
- Important for clinical decision making

## Training Configuration

### Basic Training
```python
trainer = SurvivalTrainer(
    model_name='deepsurv',
    max_epochs=100,
    early_stopping_patience=10,
    learning_rate=1e-3,
    batch_size=64
)
```

### Hyperparameter Tuning
```python
tuning_results = trainer.hyperparameter_tuning(
    X_train, t_train, e_train,
    X_val, t_val, e_val,
    n_trials=100,
    metric_name='c_index',
    direction='maximize'
)
```

### Cross-Validation
```python
cv_results = trainer.cross_validate(
    X, t, e,
    n_folds=5,
    stratify=True
)
```

## Visualization Examples

### Survival Curves by Risk Groups
```python
from src.utils.visualization import SurvivalVisualizer

visualizer = SurvivalVisualizer()

# Risk stratification
risk_groups = pd.cut(risk_scores, bins=3, labels=['Low', 'Medium', 'High'])

# Plot Kaplan-Meier curves
fig = visualizer.plot_kaplan_meier(
    times, events, risk_groups,
    title="Survival by Risk Groups"
)
```

### Model Calibration
```python
# Calibration at multiple time points
fig = visualizer.plot_calibration(
    times, events, risk_scores,
    time_points=[12, 24, 36, 60],  # 1, 2, 3, 5 years
    title="Model Calibration"
)
```

### Model Comparison
```python
# Compare multiple models
comparison_results = train_multiple_models(
    X_train, t_train, e_train,
    X_val, t_val, e_val,
    X_test, t_test, e_test,
    models=['deepsurv', 'deephit', 'mtlr']
)

fig = visualizer.plot_model_comparison(
    comparison_results['comparison'],
    metrics=['c_index', 'integrated_brier_score']
)
```

## Clinical Applications

### Risk Stratification
```python
# Identify high-risk patients
risk_tertiles = pd.qcut(risk_scores, q=3, labels=['Low', 'Medium', 'High'])
high_risk_patients = risk_tertiles == 'High'

print(f"High-risk patients: {high_risk_patients.sum()}")
print(f"Event rate in high-risk group: {events[high_risk_patients].mean():.3f}")
```

### Treatment Planning
The models provide personalized risk assessments that can inform:
- Treatment intensity decisions
- Follow-up scheduling
- Clinical trial eligibility
- Patient counseling

### Survival Predictions
```python
# Get survival probabilities over time
survival_probs = model.predict_survival(patient_features)
median_survival = time_points[survival_probs < 0.5][0]

print(f"Predicted median survival: {median_survival:.1f} months")
```

## Advanced Features

### Feature Importance Analysis
```python
# SHAP values for model interpretation
import shap

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Plot feature importance
shap.plots.beeswarm(shap_values)
```

### Model Ensembling
```python
# Combine multiple models
ensemble_predictions = (
    deepsurv_predictions * 0.4 +
    deephit_predictions * 0.4 +
    mtlr_predictions * 0.2
)
```

### External Validation
```python
# Validate on external dataset
external_results = trainer.evaluate(
    model, X_external, t_external, e_external
)
```

## Configuration Files

### Model Configuration
```json
{
  "model_name": "deepsurv",
  "hidden_dims": [128, 64, 32],
  "dropout": 0.3,
  "learning_rate": 0.001,
  "weight_decay": 1e-4,
  "batch_norm": true
}
```

### Training Configuration
```json
{
  "max_epochs": 100,
  "early_stopping_patience": 15,
  "batch_size": 128,
  "eval_time_points": [12, 24, 36, 60],
  "logger_type": "tensorboard"
}
```

## Best Practices

### Data Preprocessing
1. **Handle Missing Data**: Use appropriate imputation strategies
2. **Feature Engineering**: Create meaningful clinical features
3. **Scaling**: Standardize numerical features
4. **Encoding**: Use one-hot encoding for categorical variables

### Model Training
1. **Validation Strategy**: Use proper train/val/test splits
2. **Early Stopping**: Prevent overfitting
3. **Regularization**: Use dropout and weight decay
4. **Hyperparameter Tuning**: Optimize systematically

### Model Evaluation
1. **Multiple Metrics**: Don't rely on single metric
2. **Time-dependent Analysis**: Evaluate at clinically relevant times
3. **Calibration**: Check prediction reliability
4. **External Validation**: Test on independent datasets

### Clinical Translation
1. **Interpretability**: Provide feature importance
2. **Uncertainty**: Quantify prediction confidence
3. **Validation**: Continuous monitoring in practice
4. **Integration**: Work with clinical workflows

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{deep_survival_seer,
  title={Deep Learning Survival Analysis with SEER Data},
  author={DataRoamer},
  year={2024},
  url={https://github.com/your-repo/survival-analysis}
}
```

## References

1. Katzman, J. L., et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC medical research methodology 18.1 (2018): 1-12.

2. Lee, C., et al. "DeepHit: A deep learning approach to survival analysis with competing risks." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 32. No. 1. 2018.

3. Fotso, S. "Deep neural networks for survival analysis based on a multi-task framework." arXiv preprint arXiv:1801.05512 (2018).

## Support

For questions and issues:
- Create an issue on GitHub
- Check documentation and examples
- Review existing issues and discussions

---

**Note**: This package is for research purposes. Clinical applications require proper validation and regulatory approval.