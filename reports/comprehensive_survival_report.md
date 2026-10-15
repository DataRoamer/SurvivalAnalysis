# Comprehensive Survival Analysis Report

## Executive Summary
This analysis used a synthetic cancer survival cohort (n=1,000) to demonstrate survival analysis methodologies and validate model performance. The cohort's case mix and stage-specific survival are calibrated to published US cancer statistics for six primary sites.

## Dataset Overview
- **Total Patients**: 1,000
- **Events (Deaths)**: 351 (35.1%)
- **Median Follow-up**: 67 months (reverse Kaplan-Meier)
- **Age Range**: 28 - 95 years (median 67)
- **Diagnosis Years**: 2010 - 2020, follow-up through December 2021

## Cancer Distribution
### Primary Sites
- **Breast**: 271 patients (27.1%)
- **Prostate**: 242 patients (24.2%)
- **Lung**: 192 patients (19.2%)
- **Colorectal**: 134 patients (13.4%)
- **Melanoma**: 95 patients (9.5%)
- **Bladder**: 66 patients (6.6%)

### Stage Distribution
- **Stage I**: 415 patients (41.5%)
- **Stage II**: 252 patients (25.2%)
- **Stage III**: 116 patients (11.6%)
- **Stage IV**: 167 patients (16.7%)
- **Stage Unknown**: 50 patients (5.0%)

## Survival Analysis Results

### Kaplan-Meier Analysis
- **Overall**: median 132 months, 5-year survival 67.3%
- **Stage-specific survival**:
  - Stage I: median Not reached, 5-year survival 81.4% (n=415)
  - Stage II: median Not reached, 5-year survival 79.8% (n=252)
  - Stage III: median 88 months, 5-year survival 62.9% (n=116)
  - Stage IV: median 13 months, 5-year survival 16.7% (n=167)
  - Stage Unknown: median 98 months, 5-year survival 63.9% (n=50)
- **Site-specific survival**:
  - Breast: median Not reached, 5-year survival 82.1% (n=271)
  - Prostate: median Not reached, 5-year survival 79.1% (n=242)
  - Lung: median 16 months, 5-year survival 28.4% (n=192)
  - Colorectal: median 87 months, 5-year survival 62.8% (n=134)
  - Melanoma: median Not reached, 5-year survival 84.3% (n=95)
  - Bladder: median 93 months, 5-year survival 58.2% (n=66)

### Cox Proportional Hazards Model
- **C-index**: 0.794 on a 30% held-out test set (good discrimination)
- **Significant predictors** (p < 0.05; references: stage I, breast, well differentiated, female, White, no surgery/radiation/chemotherapy):
  - stage=IV (HR: 4.81, 95% CI 3.33-6.96, p < 0.001)
  - grade=Undifferentiated (HR: 3.00, 95% CI 1.37-6.58, p = 0.006)
  - stage=Unknown (HR: 2.56, 95% CI 1.51-4.35, p < 0.001)
  - grade=Poorly differentiated (HR: 2.18, 95% CI 1.45-3.28, p < 0.001)
  - primary_site=Lung (HR: 2.17, 95% CI 1.48-3.18, p < 0.001)
  - stage=III (HR: 2.11, 95% CI 1.42-3.13, p < 0.001)
  - age_at_diagnosis (HR: 1.03, 95% CI 1.02-1.04, p < 0.001)
  - surgery=Yes (HR: 0.58, 95% CI 0.43-0.80, p < 0.001)
  - primary_site=Prostate (HR: 0.57, 95% CI 0.34-0.94, p = 0.028)

## Key Findings
1. **Stage hierarchy**: 5-year survival falls from 81% (stage I) to 17% (stage IV)
2. **Site differences**: Lung has the poorest prognosis and Melanoma the best
3. **Age effect**: Increased mortality risk with advancing age
4. **Grade and treatment**: Poorly differentiated and undifferentiated tumors carry higher risk; surgery is associated with lower risk
5. **Model performance**: Good predictive capability

## Notes
- Deaths are from any cause, so survival is lower than cancer-specific (relative) survival, especially for older patients
- Treatment effects are observational associations, not causal estimates

## Conclusions
The synthetic survival data demonstrates:
- Site- and stage-specific cancer survival patterns
- Effective risk stratification models
- Clinical relevance of key prognostic factors
- Robust analytical methodology

*Generated on 2026-10-15 15:40:34*
