"""
Kaplan-Meier and Cox proportional hazards summaries for the synthetic cancer survival data.

The report scripts use these results instead of hard-coded numbers, so regenerating the
data and re-running the scripts keeps every report consistent with the dataset.
"""

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

TIME_COL = 'survival_months'
EVENT_COL = 'vital_status'

STAGE_ORDER = ['I', 'II', 'III', 'IV', 'Unknown']
AGE_BINS = [0, 50, 65, 80, 120]
AGE_LABELS = ['<50', '50-64', '65-79', '80+']

# Reference levels for the Cox model's categorical covariates
COX_CATEGORICAL = {
    'stage': 'I',
    'primary_site': 'Breast',
    'grade': 'Well differentiated',
    'sex': 'Female',
    'race': 'White',
    'surgery': 'No/Unknown',
    'radiation': 'No/Unknown',
    'chemotherapy': 'No/Unknown',
}
COX_NUMERIC = ['age_at_diagnosis', 'nodes_positive', 'tumor_size_mm']


def format_months(value: Optional[float]) -> str:
    """Format a median survival time, which is undefined when survival stays above 50%."""
    if value is None or not np.isfinite(value):
        return 'Not reached'
    return f"{value:.0f} months"


def _km_group(times: pd.Series, events: pd.Series) -> Dict[str, Any]:
    kmf = KaplanMeierFitter().fit(times, events)
    return {
        'patients': int(len(times)),
        'events': int(events.sum()),
        'median_survival': float(kmf.median_survival_time_),
        'survival_1yr': float(kmf.predict(12)),
        'survival_5yr': float(kmf.predict(60)),
    }


def _km_by(df: pd.DataFrame, column: str, order=None) -> Dict[str, Dict[str, Any]]:
    groups = order if order is not None else df[column].value_counts().index
    return {g: _km_group(df.loc[df[column] == g, TIME_COL], df.loc[df[column] == g, EVENT_COL])
            for g in groups if (df[column] == g).sum() > 0}


def prepare_cox_data(df: pd.DataFrame) -> pd.DataFrame:
    """Build the Cox design matrix: dummy-coded categoricals and imputed tumor size."""
    data = df[[TIME_COL, EVENT_COL] + COX_NUMERIC].copy()
    data['tumor_size_missing'] = data['tumor_size_mm'].isna().astype(int)
    data['tumor_size_mm'] = data['tumor_size_mm'].fillna(data['tumor_size_mm'].median())
    # Scale tumor size so its hazard ratio is per 10 mm
    data['tumor_size_mm'] = data['tumor_size_mm'] / 10
    data = data.rename(columns={'tumor_size_mm': 'tumor_size_per_10mm'})

    for column, reference in COX_CATEGORICAL.items():
        levels = [lvl for lvl in df[column].dropna().unique() if lvl != reference]
        for level in sorted(levels):
            data[f"{column}={level}"] = (df[column] == level).astype(int)
    return data


def compute_survival_summary(df: pd.DataFrame, random_state: int = 42) -> Dict[str, Any]:
    """
    Compute descriptive statistics, Kaplan-Meier estimates and a Cox model for the dataset.

    The Cox model's C-index is reported on a held-out 30% test split, and hazard ratios come
    from a model refit on all patients.
    """
    df = df.copy()
    df['age_group'] = pd.cut(df['age_at_diagnosis'], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    reverse_km = KaplanMeierFitter().fit(df[TIME_COL], 1 - df[EVENT_COL])

    summary = {
        'n_patients': int(len(df)),
        'n_events': int(df[EVENT_COL].sum()),
        'event_rate': float(df[EVENT_COL].mean()),
        'median_follow_up': float(reverse_km.median_survival_time_),
        'age_min': int(df['age_at_diagnosis'].min()),
        'age_max': int(df['age_at_diagnosis'].max()),
        'age_mean': float(df['age_at_diagnosis'].mean()),
        'age_median': float(df['age_at_diagnosis'].median()),
        'site_counts': df['primary_site'].value_counts(),
        'stage_counts': df['stage'].value_counts().reindex(STAGE_ORDER).dropna().astype(int),
        'overall': _km_group(df[TIME_COL], df[EVENT_COL]),
        'by_stage': _km_by(df, 'stage', STAGE_ORDER),
        'by_site': _km_by(df, 'primary_site'),
        'by_summary_stage': _km_by(df, 'summary_stage', ['Localized', 'Regional', 'Distant', 'Unknown']),
        'by_age_group': _km_by(df, 'age_group', AGE_LABELS),
    }

    cox_data = prepare_cox_data(df)
    train, test = train_test_split(cox_data, test_size=0.3, random_state=random_state,
                                   stratify=cox_data[EVENT_COL])
    cph_train = CoxPHFitter(penalizer=0.01).fit(train, TIME_COL, EVENT_COL)
    summary['c_index_train'] = float(cph_train.concordance_index_)
    summary['c_index_test'] = float(concordance_index(
        test[TIME_COL], -cph_train.predict_partial_hazard(test), test[EVENT_COL]))

    cph = CoxPHFitter(penalizer=0.01).fit(cox_data, TIME_COL, EVENT_COL)
    hazard_ratios = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
    hazard_ratios.columns = ['hr', 'hr_lower', 'hr_upper', 'p']
    summary['cox_model'] = cph
    summary['hazard_ratios'] = hazard_ratios
    summary['risk_scores'] = cph.predict_partial_hazard(cox_data).values
    summary['cox_data'] = cox_data

    return summary


def describe_c_index(c_index: float) -> str:
    """Qualitative label for a concordance index."""
    if c_index >= 0.8:
        return 'excellent'
    if c_index >= 0.7:
        return 'good'
    if c_index >= 0.6:
        return 'moderate'
    return 'weak'
