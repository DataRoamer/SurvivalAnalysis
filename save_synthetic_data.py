"""
Save synthetic SEER data to CSV file for inspection.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.seer_preprocessor import create_sample_seer_data

def main():
    """Generate and save synthetic SEER data."""
    print("Generating synthetic SEER data...")

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Generate synthetic data
    df = create_sample_seer_data(n_samples=1000, random_state=42)

    # Save to CSV
    output_path = "data/synthetic_seer_sample.csv"
    df.to_csv(output_path, index=False)

    print(f"SUCCESS: Synthetic SEER data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Event rate: {df['vital_status'].mean():.3f}")
    print(f"Median survival: {df['survival_months'].median():.1f} months")

    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    main()