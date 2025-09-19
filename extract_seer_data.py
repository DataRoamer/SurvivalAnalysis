"""
Real SEER Data Extraction Script

This script downloads and preprocesses real SEER breast cancer data from multiple sources
for survival analysis. It replaces the synthetic data generation with actual SEER data.
"""

import os
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def download_seer_kaggle_data(save_path: str = "data/seer_breast_cancer.csv") -> str:
    """
    Download SEER breast cancer data from Kaggle dataset.

    Note: This requires kaggle API setup. Alternative: manual download from:
    https://www.kaggle.com/datasets/sujithmandala/seer-breast-cancer-data

    Args:
        save_path: Path to save the downloaded data

    Returns:
        Path to the downloaded file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        # Try to use kaggle API if available
        import kaggle
        kaggle.api.dataset_download_files(
            'sujithmandala/seer-breast-cancer-data',
            path=os.path.dirname(save_path),
            unzip=True
        )
        print(f"SUCCESS: Downloaded SEER data via Kaggle API to {save_path}")

        # Find the actual CSV file in the downloaded folder
        data_dir = Path(os.path.dirname(save_path))
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            return str(csv_files[0])
        else:
            raise FileNotFoundError("No CSV file found in downloaded data")

    except ImportError:
        print("WARNING: Kaggle API not available. Please download manually from:")
        print("  https://www.kaggle.com/datasets/sujithmandala/seer-breast-cancer-data")
        print(f"  Save as: {save_path}")
        return save_path


def download_ieee_dataport_data(save_path: str = "data/ieee_seer_breast.csv") -> str:
    """
    Instructions for downloading SEER breast cancer data from IEEE DataPort.

    Args:
        save_path: Path where data should be saved

    Returns:
        Path where data should be saved
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("To download IEEE DataPort SEER Breast Cancer Data:")
    print("1. Visit: https://ieee-dataport.org/open-access/seer-breast-cancer-data")
    print("2. Create free IEEE account if needed")
    print("3. Download the dataset")
    print(f"4. Save as: {save_path}")
    print("\nThis dataset contains 4,024 breast cancer patients from SEER 2006-2010")

    return save_path


def standardize_seer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize SEER data columns to match expected format for survival analysis.

    Args:
        df: Raw SEER dataframe

    Returns:
        Standardized dataframe with consistent column names
    """
    # Common SEER column mappings
    column_mappings = {
        # Time and event columns
        'Survival Months': 'survival_months',
        'survival_months': 'survival_months',
        'time': 'survival_months',
        'Time': 'survival_months',

        # Event/status columns
        'Status': 'vital_status',
        'status': 'vital_status',
        'vital_status': 'vital_status',
        'event': 'vital_status',
        'Event': 'vital_status',
        'Vital_status': 'vital_status',

        # Demographic columns
        'Age': 'age',
        'age': 'age',
        'Age_at_diagnosis': 'age',
        'Age at diagnosis': 'age',

        'Race': 'race',
        'race': 'race',
        'Race_recode': 'race',

        'Sex': 'sex',
        'sex': 'sex',
        'Gender': 'sex',

        # Cancer-specific columns
        'Grade': 'grade',
        'grade': 'grade',
        'Grade_recode': 'grade',

        'Stage': 'stage',
        'stage': 'stage',
        'AJCC_Stage': 'stage',
        'Historic_Stage_A': 'stage',

        'Primary_Site': 'primary_site',
        'Primary Site': 'primary_site',
        'primary_site': 'primary_site',

        'Tumor_Size': 'tumor_size',
        'Tumor Size': 'tumor_size',
        'tumor_size': 'tumor_size',
        'Size': 'tumor_size',

        # Treatment columns
        'Surgery': 'surgery',
        'surgery': 'surgery',
        'Surgery_recode': 'surgery',

        'Radiation': 'radiation',
        'radiation': 'radiation',
        'Radiation_recode': 'radiation',

        'Chemotherapy': 'chemotherapy',
        'chemotherapy': 'chemotherapy',
        'Chemotherapy_recode': 'chemotherapy',
    }

    # Rename columns
    df_standardized = df.copy()
    df_standardized.columns = [column_mappings.get(col, col.lower().replace(' ', '_'))
                              for col in df_standardized.columns]

    # Ensure we have required columns
    required_cols = ['survival_months', 'vital_status']
    missing_cols = [col for col in required_cols if col not in df_standardized.columns]

    if missing_cols:
        print(f"WARNING: Missing required columns: {missing_cols}")
        print("Available columns:", list(df_standardized.columns))

    return df_standardized


def preprocess_seer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess SEER data for survival analysis.

    Args:
        df: Raw SEER dataframe

    Returns:
        Preprocessed dataframe ready for survival analysis
    """
    df_processed = df.copy()

    # Convert survival months to numeric
    if 'survival_months' in df_processed.columns:
        df_processed['survival_months'] = pd.to_numeric(
            df_processed['survival_months'], errors='coerce'
        )

        # Remove negative survival times
        df_processed = df_processed[df_processed['survival_months'] >= 0]

    # Standardize vital status (1 = death, 0 = alive/censored)
    if 'vital_status' in df_processed.columns:
        # Handle different encoding schemes
        if df_processed['vital_status'].dtype == 'object':
            status_mapping = {
                'Dead': 1, 'dead': 1, 'Died': 1, 'died': 1,
                'Alive': 0, 'alive': 0, 'Living': 0, 'living': 0,
                'Yes': 1, 'No': 0,
                'Death': 1, 'Censored': 0
            }
            df_processed['vital_status'] = df_processed['vital_status'].map(
                status_mapping
            ).fillna(df_processed['vital_status'])

        # Convert to numeric
        df_processed['vital_status'] = pd.to_numeric(
            df_processed['vital_status'], errors='coerce'
        )

    # Clean age column
    if 'age' in df_processed.columns:
        df_processed['age'] = pd.to_numeric(df_processed['age'], errors='coerce')
        # Remove unrealistic ages
        df_processed = df_processed[
            (df_processed['age'] >= 0) & (df_processed['age'] <= 120)
        ]

    # Clean tumor size
    if 'tumor_size' in df_processed.columns:
        df_processed['tumor_size'] = pd.to_numeric(
            df_processed['tumor_size'], errors='coerce'
        )

    # Remove rows with missing critical information
    critical_cols = ['survival_months', 'vital_status']
    for col in critical_cols:
        if col in df_processed.columns:
            df_processed = df_processed.dropna(subset=[col])

    print(f"SUCCESS: Preprocessed data: {len(df_processed)} patients")
    print(f"  Event rate: {df_processed['vital_status'].mean():.3f}")
    print(f"  Median survival: {df_processed['survival_months'].median():.1f} months")

    return df_processed


def main():
    """Main function to extract and preprocess real SEER data."""
    print("=== SEER Real Data Extraction ===\n")

    # Create data directory
    os.makedirs("data", exist_ok=True)

    # Option 1: Try Kaggle download
    print("Option 1: Attempting Kaggle download...")
    kaggle_path = download_seer_kaggle_data()

    # Option 2: IEEE DataPort instructions
    print("\nOption 2: IEEE DataPort download instructions...")
    ieee_path = download_ieee_dataport_data()

    # Try to load available data
    data_paths = [kaggle_path, ieee_path, "data/SEER_breast_cancer.csv"]

    df_real = None
    for path in data_paths:
        if os.path.exists(path):
            try:
                print(f"\nSUCCESS: Loading data from: {path}")
                df_real = pd.read_csv(path)
                print(f"  Raw data shape: {df_real.shape}")
                print(f"  Columns: {list(df_real.columns)}")
                break
            except Exception as e:
                print(f"  ERROR loading {path}: {e}")
                continue

    if df_real is not None:
        # Standardize and preprocess
        print("\n--- Standardizing columns ---")
        df_standardized = standardize_seer_columns(df_real)

        print("\n--- Preprocessing data ---")
        df_processed = preprocess_seer_data(df_standardized)

        # Save processed data
        output_path = "data/seer_processed.csv"
        df_processed.to_csv(output_path, index=False)
        print(f"SUCCESS: Saved processed data to: {output_path}")

        # Display sample
        print("\n--- Sample of processed data ---")
        print(df_processed.head())

        print("\n--- Data summary ---")
        print(df_processed.describe())

        return output_path
    else:
        print("\nWARNING: No SEER data found. Please download manually:")
        print("1. Kaggle: https://www.kaggle.com/datasets/sujithmandala/seer-breast-cancer-data")
        print("2. IEEE: https://ieee-dataport.org/open-access/seer-breast-cancer-data")
        print("3. Save as: data/SEER_breast_cancer.csv")
        return None


if __name__ == "__main__":
    main()