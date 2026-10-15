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
                            df_clean[col].fillna(mode_val[0], inplace=True)
                        else:
                            df_clean[col].fillna('Unknown', inplace=True)
                    else:
                        # Use specified strategy for numerical
                        if self.missing_strategy == 'median':
                            df_clean[col].fillna(df_clean[col].median(), inplace=True)
                        elif self.missing_strategy == 'mean':
                            df_clean[col].fillna(df_clean[col].mean(), inplace=True)

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


def create_synthetic_survival_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Create sample synthetic survival data for testing and development.

    Args:
        n_samples: Number of samples to generate
        random_state: Random seed

    Returns:
        Sample dataframe with a realistic cancer survival data structure
    """
    np.random.seed(random_state)

    # Categorical features typical in cancer registry data
    sites = ['Breast', 'Lung', 'Prostate', 'Colorectal', 'Melanoma', 'Bladder']
    stages = ['I', 'II', 'III', 'IV', 'Unknown']
    grades = ['Well', 'Moderate', 'Poor', 'Undifferentiated', 'Unknown']
    races = ['White', 'Black', 'Asian', 'Hispanic', 'Other']
    sexes = ['Male', 'Female']

    data = {
        'age_at_diagnosis': np.random.normal(65, 15, n_samples).clip(18, 95).astype(int),
        'primary_site': np.random.choice(sites, n_samples),
        'stage': np.random.choice(stages, n_samples, p=[0.25, 0.25, 0.25, 0.2, 0.05]),
        'grade': np.random.choice(grades, n_samples, p=[0.2, 0.3, 0.3, 0.15, 0.05]),
        'race': np.random.choice(races, n_samples, p=[0.7, 0.15, 0.08, 0.05, 0.02]),
        'sex': np.random.choice(sexes, n_samples),
        'tumor_size': np.random.lognormal(2, 1, n_samples).clip(0.1, 20),
        'nodes_positive': np.random.poisson(2, n_samples).clip(0, 30),
        'nodes_examined': np.random.poisson(8, n_samples).clip(1, 50)
    }

    df = pd.DataFrame(data)

    # Generate survival outcomes based on risk factors
    risk_score = (
        (df['age_at_diagnosis'] - 65) * 0.02 +
        (df['stage'].map({'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'Unknown': 1.5})) * 0.5 +
        (df['grade'].map({'Well': 0, 'Moderate': 0.5, 'Poor': 1, 'Undifferentiated': 1.2, 'Unknown': 0.7})) * 0.3 +
        np.log(df['tumor_size']) * 0.2 +
        (df['nodes_positive'] / df['nodes_examined']).fillna(0) * 1.5
    )

    # Generate survival times (months)
    baseline_hazard = 0.01
    survival_times = np.random.exponential(1 / (baseline_hazard * np.exp(risk_score))) * 12

    # Generate censoring (assume 30% censoring rate)
    censoring_times = np.random.exponential(60, n_samples)  # Average 5 years follow-up

    observed_times = np.minimum(survival_times, censoring_times)
    events = (survival_times <= censoring_times).astype(int)

    df['survival_months'] = observed_times.clip(0.1, 240)  # Max 20 years follow-up
    df['vital_status'] = events

    return df