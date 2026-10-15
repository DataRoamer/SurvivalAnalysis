"""
Survival Data Preprocessing Pipeline for Survival Analysis

This module provides comprehensive preprocessing capabilities for cancer survival data,
preparing it for deep learning survival analysis models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings

class SurvivalDataPreprocessor:
    """
    Preprocessor for cancer survival data for survival analysis.

    Handles missing values, categorical encoding, feature scaling, and data splitting
    while preserving survival analysis structure (time-to-event and censoring).
    """

    def __init__(self,
                 time_col: str = 'survival_months',
                 event_col: str = 'vital_status',
                 categorical_strategy: str = 'onehot',
                 missing_strategy: str = 'median',
                 scale_features: bool = True):
        """
        Initialize survival data preprocessor.

        Args:
            time_col: Column name for survival time
            event_col: Column name for event indicator (1=death, 0=censored)
            categorical_strategy: 'onehot' or 'label' encoding for categorical variables
            missing_strategy: 'median', 'mean', 'mode', or 'drop' for missing values
            scale_features: Whether to standardize numerical features
        """
        self.time_col = time_col
        self.event_col = event_col
        self.categorical_strategy = categorical_strategy
        self.missing_strategy = missing_strategy
        self.scale_features = scale_features

        self.scaler = StandardScaler() if scale_features else None
        self.encoders = {}
        self.feature_names = []
        self.categorical_cols = []
        self.numerical_cols = []

    def identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Automatically identify categorical and numerical columns.

        Args:
            df: Input dataframe

        Returns:
            Tuple of (categorical_columns, numerical_columns)
        """
        exclude_cols = {self.time_col, self.event_col}

        categorical_cols = []
        numerical_cols = []

        for col in df.columns:
            if col in exclude_cols:
                continue

            if df[col].dtype == 'object' or df[col].nunique() < 10:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)

        return categorical_cols, numerical_cols

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values based on specified strategy.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with missing values handled
        """
        df_clean = df.copy()

        if self.missing_strategy == 'drop':
            df_clean = df_clean.dropna()
        else:
            for col in df_clean.columns:
                if col in {self.time_col, self.event_col}:
                    continue

                if df_clean[col].isnull().any():
                    if col in self.categorical_cols:
                        # Use mode for categorical
                        mode_val = df_clean[col].mode()
                        if len(mode_val) > 0:
                            df_clean[col] = df_clean[col].fillna(mode_val[0])
                        else:
                            df_clean[col] = df_clean[col].fillna('Unknown')
                    else:
                        # Use specified strategy for numerical
                        if self.missing_strategy == 'median':
                            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                        elif self.missing_strategy == 'mean':
                            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

        return df_clean

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using specified strategy.

        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for training, False for inference)

        Returns:
            Dataframe with encoded categorical features
        """
        df_encoded = df.copy()

        if self.categorical_strategy == 'onehot':
            for col in self.categorical_cols:
                if fit:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(df_encoded[[col]])
                    self.encoders[col] = encoder

                    # Create column names
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                else:
                    encoder = self.encoders[col]
                    encoded_data = encoder.transform(df_encoded[[col]])
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]

                # Add encoded columns
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df_encoded.index)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded_df], axis=1)

        elif self.categorical_strategy == 'label':
            for col in self.categorical_cols:
                if fit:
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = encoder
                else:
                    encoder = self.encoders[col]
                    # Handle unknown categories
                    known_labels = set(encoder.classes_)
                    df_encoded[col] = df_encoded[col].astype(str)
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: x if x in known_labels else encoder.classes_[0]
                    )
                    df_encoded[col] = encoder.transform(df_encoded[col])

        return df_encoded

    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.

        Args:
            df: Input dataframe
            fit: Whether to fit scaler (True for training, False for inference)

        Returns:
            Dataframe with scaled numerical features
        """
        if not self.scale_features or not self.numerical_cols:
            return df

        df_scaled = df.copy()

        if fit:
            df_scaled[self.numerical_cols] = self.scaler.fit_transform(df_scaled[self.numerical_cols])
        else:
            df_scaled[self.numerical_cols] = self.scaler.transform(df_scaled[self.numerical_cols])

        return df_scaled

    def prepare_survival_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for survival analysis models.

        Args:
            df: Preprocessed dataframe

        Returns:
            Tuple of (features, times, events)
        """
        # Extract survival components
        times = df[self.time_col].values
        events = df[self.event_col].values

        # Extract features (excluding survival columns)
        feature_cols = [col for col in df.columns if col not in {self.time_col, self.event_col}]
        features = df[feature_cols].values

        self.feature_names = feature_cols

        return features, times, events

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform training data.

        Args:
            df: Raw survival dataframe

        Returns:
            Tuple of (features, times, events)
        """
        # Validate required columns
        if self.time_col not in df.columns:
            raise ValueError(f"Time column '{self.time_col}' not found in dataframe")
        if self.event_col not in df.columns:
            raise ValueError(f"Event column '{self.event_col}' not found in dataframe")

        # Identify column types
        self.categorical_cols, self.numerical_cols = self.identify_column_types(df)

        # Preprocessing pipeline
        df_processed = df.copy()
        df_processed = self.handle_missing_values(df_processed)
        df_processed = self.encode_categorical_features(df_processed, fit=True)
        df_processed = self.scale_numerical_features(df_processed, fit=True)

        # Prepare survival data
        features, times, events = self.prepare_survival_data(df_processed)

        return features, times, events

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform new data using fitted preprocessor.

        Args:
            df: Raw survival dataframe

        Returns:
            Tuple of (features, times, events)
        """
        # Preprocessing pipeline (without fitting)
        df_processed = df.copy()
        df_processed = self.handle_missing_values(df_processed)
        df_processed = self.encode_categorical_features(df_processed, fit=False)
        df_processed = self.scale_numerical_features(df_processed, fit=False)

        # Prepare survival data
        features, times, events = self.prepare_survival_data(df_processed)

        return features, times, events

    def train_test_split(self,
                        features: np.ndarray,
                        times: np.ndarray,
                        events: np.ndarray,
                        test_size: float = 0.2,
                        val_size: float = 0.1,
                        random_state: int = 42) -> Dict[str, np.ndarray]:
        """
        Split data into train/validation/test sets.

        Args:
            features: Feature matrix
            times: Survival times
            events: Event indicators
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed

        Returns:
            Dictionary containing split datasets
        """
        # First split: separate test set
        X_temp, X_test, t_temp, t_test, e_temp, e_test = train_test_split(
            features, times, events, test_size=test_size, random_state=random_state, stratify=events
        )

        # Second split: separate train and validation
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, t_train, t_val, e_train, e_val = train_test_split(
                X_temp, t_temp, e_temp, test_size=val_size_adjusted,
                random_state=random_state, stratify=e_temp
            )
        else:
            X_train, X_val, t_train, t_val, e_train, e_val = X_temp, None, t_temp, None, e_temp, None

        result = {
            'X_train': X_train, 't_train': t_train, 'e_train': e_train,
            'X_test': X_test, 't_test': t_test, 'e_test': e_test
        }

        if val_size > 0:
            result.update({
                'X_val': X_val, 't_val': t_val, 'e_val': e_val
            })

        return result

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about processed features.

        Returns:
            Dictionary with feature information
        """
        return {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols,
            'encoders': list(self.encoders.keys()),
            'preprocessing_config': {
                'categorical_strategy': self.categorical_strategy,
                'missing_strategy': self.missing_strategy,
                'scale_features': self.scale_features
            }
        }


def load_survival_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load survival data from various file formats.

    Args:
        file_path: Path to survival data file
        **kwargs: Additional arguments for pandas read functions

    Returns:
        Loaded dataframe
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, **kwargs)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path, **kwargs)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


# Reference parameters for the synthetic cohort. Values are approximations of
# published US population cancer statistics (incidence share, sex ratio, median
# age at diagnosis, summary stage mix and 5-year relative survival by stage).
SITE_PROFILES = {
    'Breast': {
        'share': 0.26, 'male': 0.01, 'median_age': 63,
        'stage_mix': {'Localized': 0.65, 'Regional': 0.27, 'Distant': 0.06, 'Unknown': 0.02},
        'five_year_survival': {'Localized': 0.99, 'Regional': 0.87, 'Distant': 0.32, 'Unknown': 0.60},
        'grade_mix': [0.20, 0.42, 0.33, 0.01, 0.04],
        'tumor_size_mm': 20, 'tumor_size_missing': 0.04,
        'nodes_examined_rate': 0.85, 'nodes_examined_mean': 6,
        'surgery': (0.97, 0.93, 0.30), 'radiation': (0.55, 0.60, 0.35), 'chemotherapy': (0.25, 0.70, 0.60),
    },
    'Prostate': {
        'share': 0.25, 'male': 1.00, 'median_age': 67,
        'stage_mix': {'Localized': 0.70, 'Regional': 0.13, 'Distant': 0.08, 'Unknown': 0.09},
        'five_year_survival': {'Localized': 0.995, 'Regional': 0.99, 'Distant': 0.34, 'Unknown': 0.85},
        'grade_mix': [0.05, 0.45, 0.40, 0.00, 0.10],
        'tumor_size_mm': 15, 'tumor_size_missing': 0.70,
        'nodes_examined_rate': 0.20, 'nodes_examined_mean': 6,
        'surgery': (0.35, 0.45, 0.05), 'radiation': (0.30, 0.45, 0.30), 'chemotherapy': (0.02, 0.10, 0.30),
    },
    'Lung': {
        'share': 0.20, 'male': 0.52, 'median_age': 71,
        'stage_mix': {'Localized': 0.27, 'Regional': 0.22, 'Distant': 0.45, 'Unknown': 0.06},
        'five_year_survival': {'Localized': 0.64, 'Regional': 0.36, 'Distant': 0.09, 'Unknown': 0.12},
        'grade_mix': [0.08, 0.22, 0.30, 0.03, 0.37],
        'tumor_size_mm': 30, 'tumor_size_missing': 0.10,
        'nodes_examined_rate': 0.55, 'nodes_examined_mean': 10,
        'surgery': (0.65, 0.30, 0.05), 'radiation': (0.20, 0.55, 0.45), 'chemotherapy': (0.15, 0.60, 0.60),
    },
    'Colorectal': {
        'share': 0.13, 'male': 0.53, 'median_age': 66,
        'stage_mix': {'Localized': 0.37, 'Regional': 0.36, 'Distant': 0.22, 'Unknown': 0.05},
        'five_year_survival': {'Localized': 0.91, 'Regional': 0.74, 'Distant': 0.16, 'Unknown': 0.35},
        'grade_mix': [0.08, 0.68, 0.14, 0.02, 0.08],
        'tumor_size_mm': 40, 'tumor_size_missing': 0.08,
        'nodes_examined_rate': 0.90, 'nodes_examined_mean': 18,
        'surgery': (0.95, 0.93, 0.50), 'radiation': (0.03, 0.25, 0.15), 'chemotherapy': (0.05, 0.65, 0.70),
    },
    'Melanoma': {
        'share': 0.09, 'male': 0.58, 'median_age': 66,
        'stage_mix': {'Localized': 0.77, 'Regional': 0.09, 'Distant': 0.04, 'Unknown': 0.10},
        'five_year_survival': {'Localized': 0.995, 'Regional': 0.74, 'Distant': 0.35, 'Unknown': 0.85},
        'grade_mix': [0.02, 0.02, 0.02, 0.01, 0.93],
        'tumor_size_mm': 8, 'tumor_size_missing': 0.30,
        'nodes_examined_rate': 0.30, 'nodes_examined_mean': 3,
        'surgery': (0.99, 0.95, 0.50), 'radiation': (0.01, 0.15, 0.30), 'chemotherapy': (0.01, 0.20, 0.45),
    },
    'Bladder': {
        'share': 0.07, 'male': 0.76, 'median_age': 73,
        'stage_mix': {'Localized': 0.85, 'Regional': 0.07, 'Distant': 0.05, 'Unknown': 0.03},
        'five_year_survival': {'Localized': 0.86, 'Regional': 0.40, 'Distant': 0.08, 'Unknown': 0.45},
        'grade_mix': [0.05, 0.10, 0.25, 0.20, 0.40],
        'tumor_size_mm': 30, 'tumor_size_missing': 0.25,
        'nodes_examined_rate': 0.25, 'nodes_examined_mean': 15,
        'surgery': (0.97, 0.85, 0.40), 'radiation': (0.03, 0.20, 0.20), 'chemotherapy': (0.30, 0.55, 0.60),
    },
}

GRADES = ['Well differentiated', 'Moderately differentiated', 'Poorly differentiated',
          'Undifferentiated', 'Unknown']
GRADE_LOG_HR = np.array([-0.3, 0.0, 0.4, 0.5, 0.1])
RACES = ['White', 'Black', 'Asian or Pacific Islander', 'American Indian/Alaska Native', 'Unknown']
MARITAL_STATUSES = ['Married', 'Single', 'Divorced/Separated', 'Widowed', 'Unknown']
SUMMARY_STAGES = ['Localized', 'Regional', 'Distant', 'Unknown']


def create_synthetic_survival_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Create a synthetic cancer survival cohort that mimics population cancer registry data.

    Site, sex, age, stage mix, grade, treatment and stage-specific 5-year survival are
    calibrated per primary site to published US cancer statistics. Deaths combine
    cancer-specific mortality with age- and sex-dependent background mortality, and
    follow-up is censored administratively at the end of 2021 for diagnoses in 2010-2020.

    Args:
        n_samples: Number of samples to generate
        random_state: Random seed

    Returns:
        Dataframe with one row per patient; `survival_months` is in whole months and
        `vital_status` is 1 for death from any cause, 0 if alive at last follow-up
    """
    rng = np.random.default_rng(random_state)
    sites = list(SITE_PROFILES)

    site = rng.choice(sites, n_samples, p=[SITE_PROFILES[s]['share'] for s in sites])
    columns = {name: np.empty(n_samples, dtype=object) for name in
               ['sex', 'summary_stage', 'stage', 'grade', 'race', 'hispanic_origin',
                'marital_status', 'surgery', 'radiation', 'chemotherapy']}
    age = np.empty(n_samples)
    tumor_size = np.empty(n_samples)
    nodes_examined = np.zeros(n_samples, dtype=int)
    nodes_positive = np.zeros(n_samples, dtype=int)
    log_hr = np.zeros(n_samples)
    target_survival = np.empty(n_samples)

    for s in sites:
        idx = np.flatnonzero(site == s)
        n = len(idx)
        if n == 0:
            continue
        p = SITE_PROFILES[s]

        columns['sex'][idx] = np.where(rng.random(n) < p['male'], 'Male', 'Female')
        age[idx] = rng.normal(p['median_age'], 12, n).clip(20, 95).round()

        summary = rng.choice(SUMMARY_STAGES, n, p=[p['stage_mix'][k] for k in SUMMARY_STAGES])
        columns['summary_stage'][idx] = summary
        target_survival[idx] = [p['five_year_survival'][k] for k in summary]

        # AJCC stage nested within summary stage; later stage within a group carries extra risk
        upper = rng.random(n)
        stage = np.select(
            [summary == 'Localized', summary == 'Regional', summary == 'Distant'],
            [np.where(upper < 0.3, 'II', 'I'), np.where(upper < 0.6, 'III', 'II'), 'IV'],
            'Unknown')
        columns['stage'][idx] = stage
        within_stage_hr = np.where(((summary == 'Localized') & (stage == 'II')) |
                                   ((summary == 'Regional') & (stage == 'III')), 0.4, 0.0)

        grade_idx = rng.choice(len(GRADES), n, p=p['grade_mix'])
        columns['grade'][idx] = np.array(GRADES)[grade_idx]

        race_p = [0.97, 0.01, 0.01, 0.005, 0.005] if s == 'Melanoma' else \
                 [0.72, 0.16, 0.09, 0.01, 0.02] if s == 'Prostate' else \
                 [0.77, 0.11, 0.09, 0.01, 0.02]
        race = rng.choice(RACES, n, p=race_p)
        columns['race'][idx] = race
        columns['hispanic_origin'][idx] = np.where(
            rng.random(n) < (0.03 if s == 'Melanoma' else 0.13), 'Hispanic', 'Non-Hispanic')

        widowed = np.clip((age[idx] - 55) / 100, 0.02, 0.30)
        married = 0.55 - widowed / 2
        marital = [rng.choice(MARITAL_STATUSES, p=[m, 0.15, 0.13, w, 1 - m - 0.28 - w])
                   for m, w in zip(married, widowed)]
        columns['marital_status'][idx] = marital

        stage_pos = np.select([summary == 'Localized', summary == 'Regional', summary == 'Distant'],
                              [0, 1, 2], 1)
        treatments = {}
        for t in ['surgery', 'radiation', 'chemotherapy']:
            received = rng.random(n) < np.array(p[t])[stage_pos]
            treatments[t] = received
            columns[t][idx] = np.where(received, 'Yes', 'No/Unknown')

        size_factor = np.select([stage == 'I', stage == 'II', stage == 'III', stage == 'IV'],
                                [0.7, 1.0, 1.4, 1.6], 1.2)
        size = (p['tumor_size_mm'] * size_factor * rng.lognormal(0, 0.5, n)).clip(1, 200).round()
        size[rng.random(n) < p['tumor_size_missing']] = np.nan
        tumor_size[idx] = size

        examined = (rng.random(n) < p['nodes_examined_rate']) & (summary != 'Unknown')
        n_exam = np.where(examined, rng.poisson(p['nodes_examined_mean'], n).clip(1, 60), 0)
        positive_p = np.select([summary == 'Regional', summary == 'Distant'], [0.25, 0.35], 0.0)
        n_pos = rng.binomial(n_exam, positive_p)
        # Regional disease with nodes examined implies at least one positive node
        n_pos = np.where((summary == 'Regional') & (n_exam > 0), np.maximum(n_pos, 1), n_pos)
        nodes_examined[idx] = n_exam
        nodes_positive[idx] = n_pos

        # Patient-level prognostic factors, centred within site and summary stage so the
        # stage-specific 5-year survival targets hold on average
        lp = (0.015 * (age[idx] - p['median_age'])
              + GRADE_LOG_HR[grade_idx]
              + 0.15 * (race == 'Black')
              + within_stage_hr
              + 0.8 * np.divide(n_pos, n_exam, out=np.zeros(n), where=n_exam > 0)
              - 0.3 * (treatments['surgery'] & (s != 'Prostate')))
        for k in SUMMARY_STAGES:
            group = summary == k
            if group.any():
                lp[group] -= lp[group].mean()
        log_hr[idx] = lp

    # Cancer-specific death: Weibull with decreasing hazard, scaled per site and stage
    shape = 0.8
    scale = 60 / (-np.log(target_survival)) ** (1 / shape)
    u = rng.random(n_samples)
    cancer_time = scale * (-np.log(u) / np.exp(log_hr)) ** (1 / shape)

    # Background (other-cause) death: Gompertz mortality by age, higher for men
    gompertz_a = 4e-5 * np.where(columns['sex'] == 'Male', 1.5, 1.0)
    gompertz_b = 0.085
    e = rng.exponential(1, n_samples)
    other_years = np.log1p(e * gompertz_b / (gompertz_a * np.exp(gompertz_b * age))) / gompertz_b
    other_time = other_years * 12

    # Diagnosis 2010-2020 with follow-up through December 2021, plus some loss to follow-up
    year = rng.integers(2010, 2021, n_samples)
    month = rng.integers(1, 13, n_samples)
    admin_censor = (2021 - year) * 12 + (12 - month) + rng.random(n_samples)
    lost = rng.exponential(1200, n_samples)

    death_time = np.minimum(cancer_time, other_time)
    follow_up = np.minimum(admin_censor, lost)

    df = pd.DataFrame({
        'year_of_diagnosis': year,
        'age_at_diagnosis': age.astype(int),
        'sex': columns['sex'],
        'race': columns['race'],
        'hispanic_origin': columns['hispanic_origin'],
        'marital_status': columns['marital_status'],
        'primary_site': site,
        'summary_stage': columns['summary_stage'],
        'stage': columns['stage'],
        'grade': columns['grade'],
        'tumor_size_mm': tumor_size,
        'nodes_examined': nodes_examined,
        'nodes_positive': nodes_positive,
        'surgery': columns['surgery'],
        'radiation': columns['radiation'],
        'chemotherapy': columns['chemotherapy'],
    })
    df['survival_months'] = np.floor(np.minimum(death_time, follow_up)).astype(int)
    df['vital_status'] = (death_time <= follow_up).astype(int)

    return df
