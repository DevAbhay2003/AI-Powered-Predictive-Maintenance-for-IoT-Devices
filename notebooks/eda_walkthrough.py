"""
notebooks/eda_walkthrough.py
----------------------------
Exploratory Data Analysis (EDA) walkthrough for the IoT Predictive Maintenance dataset.
Can be run as a standalone script or exported into a Jupyter Notebook.
"""

import pandas as pd
import numpy as np

def run_eda():
    print("=" * 60)
    print("      IoT PREDICTIVE MAINTENANCE: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # 1. Load Data
    df = pd.read_csv("data/raw_sensor_data.csv")
    print(f"\n1. DATASET OVERVIEW:")
    print(f"Total Records: {len(df):,}")
    print(f"Total Telemetry Features: {df.shape[1]}")
    print("\nFirst 5 Records:")
    print(df.head())

    # 2. Summary Statistics
    print("\n2. NUMERICAL DESCRIPTIVE STATISTICS:")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]])

    # 3. Class Imbalance Check
    print("\n3. CLASS DISTRIBUTION (MACHINE FAILURE):")
    counts = df["machine_failure"].value_counts()
    percentages = df["machine_failure"].value_counts(normalize=True) * 100
    for cls in [0, 1]:
        label = "Healthy (0)" if cls == 0 else "Failure (1)"
        print(f"  {label:<15}: {counts.get(cls, 0):>5} samples ({percentages.get(cls, 0):.2f}%)")

    # 4. Failure Mode Breakdown
    failure_types = ["hdf_failure", "pwf_failure", "osf_failure", "twf_failure", "hpf_failure"]
    print("\n4. BREAKDOWN BY SPECIFIC FAILURE MODE:")
    for ft in failure_types:
        if ft in df.columns:
            total_cases = df[ft].sum()
            print(f"  {ft.upper():<16}: {total_cases:>4} incidents")

    # 5. Correlation with Target
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()["machine_failure"].sort_values(ascending=False)
    print("\n5. TOP CORRELATIONS WITH FAILURE:")
    print(correlations.to_string())

if __name__ == "__main__":
    run_eda()
