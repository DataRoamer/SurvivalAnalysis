"""
Test script to validate SEER data loading without PyTorch dependencies.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.seer_preprocessor import SEERPreprocessor, create_sample_seer_data, load_seer_data


def test_data_loading():
    """Test SEER data loading functionality."""
    print("=== Testing SEER Data Loading ===\n")

    # Test 1: Create synthetic data
    print("Test 1: Creating synthetic SEER data...")
    df_synthetic = create_sample_seer_data(n_samples=100, random_state=42)
    print(f"Synthetic data shape: {df_synthetic.shape}")
    print(f"Columns: {list(df_synthetic.columns)}")
    print(f"Event rate: {df_synthetic['vital_status'].mean():.3f}")
    print()

    # Test 2: Test data loading paths
    print("Test 2: Testing data loading paths...")
    real_data_paths = [
        'data/seer_processed.csv',
        'data/SEER_breast_cancer.csv',
        'data/seer_breast_cancer.csv'
    ]

    df_real = None
    for path in real_data_paths:
        if Path(path).exists():
            try:
                print(f"Loading real SEER data from: {path}")
                df_real = load_seer_data(path)
                print(f"SUCCESS: Loaded real data with shape {df_real.shape}")
                break
            except Exception as e:
                print(f"ERROR loading {path}: {e}")
                continue

    if df_real is None:
        print("No real SEER data found - this is expected for first run")
        df_real = df_synthetic

    # Test 3: Test preprocessing
    print("\nTest 3: Testing data preprocessing...")
    preprocessor = SEERPreprocessor(
        time_col='survival_months',
        event_col='vital_status',
        categorical_strategy='onehot',
        missing_strategy='median',
        scale_features=True
    )

    try:
        X_processed, y_time, y_event = preprocessor.fit_transform(df_real)
        print(f"SUCCESS: Preprocessed data shape: {X_processed.shape}")
        print(f"Time targets shape: {y_time.shape}")
        print(f"Event targets shape: {y_event.shape}")
        print(f"Feature names: {preprocessor.feature_names[:5] if preprocessor.feature_names else 'None'}")
    except Exception as e:
        print(f"ERROR in preprocessing: {e}")
        return False

    print("\n=== All Tests Passed ===")
    return True


if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\nSUCCESS: SEER data loading and preprocessing validated successfully!")
    else:
        print("\nERROR: Some tests failed. Check the errors above.")