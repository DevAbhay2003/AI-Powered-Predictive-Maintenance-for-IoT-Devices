"""
preprocess.py
-------------
Data cleaning, domain-specific feature engineering, and preparation
for machine learning model training.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Feature columns used for predictive modeling
FEATURE_COLUMNS = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mm_s",
    "hydraulic_pressure_bar",
    "temp_difference",
    "mechanical_power_w",
    "strain_index",
    "vibration_to_speed_ratio",
    "pressure_deviation",
]

TARGET_COLUMN = "machine_failure"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific industrial IoT physics features.

    Features engineered:
    - temp_difference: Thermal gradient between component and ambient air
    - mechanical_power_w: Power draw = Torque (Nm) * Angular velocity (rad/s)
    - strain_index: Product of tool wear time and operating torque
    - vibration_to_speed_ratio: Bearing harmonic stress indicator
    - pressure_deviation: Absolute delta from standard nominal hydraulic baseline (140 Bar)
    """
    df = df.copy()

    # 1. Thermal delta
    df["temp_difference"] = df["process_temperature_k"] - df["air_temperature_k"]

    # 2. Power in Watts: Torque * (RPM * 2 * pi / 60)
    angular_velocity = df["rotational_speed_rpm"] * (2 * np.pi / 60.0)
    df["mechanical_power_w"] = df["torque_nm"] * angular_velocity

    # 3. Mechanical strain index
    df["strain_index"] = df["tool_wear_min"] * df["torque_nm"]

    # 4. Vibration to speed ratio
    df["vibration_to_speed_ratio"] = df["vibration_mm_s"] / (df["rotational_speed_rpm"] + 1e-5)

    # 5. Hydraulic pressure deviation from baseline (140 bar)
    df["pressure_deviation"] = (df["hydraulic_pressure_bar"] - 140.0).abs()

    return df


def load_and_preprocess(
    raw_path: str = "data/raw_sensor_data.csv",
    save_processed_path: str = "data/processed_sensor_data.csv",
    scaler_path: str = "models/feature_scaler.joblib",
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Full preprocessing pipeline:
    - Loads raw dataset
    - Applies feature engineering
    - Performs stratified train-test split
    - Standardizes numerical features using StandardScaler
    - Persists fitted scaler for production inference
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}. Run data_generator.py first.")

    print(f"[PREPROCESS] Loading raw data from: {raw_path}")
    raw_df = pd.read_csv(raw_path)

    # Handle any potential missing values
    missing_count = raw_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"[PREPROCESS] Imputing {missing_count} missing values with median...")
        raw_df = raw_df.fillna(raw_df.median(numeric_only=True))
    else:
        print("[PREPROCESS] Data integrity check passed: 0 missing values detected.")

    # Apply feature engineering
    print("[PREPROCESS] Calculating industrial feature transformations...")
    engineered_df = engineer_features(raw_df)

    # Save processed dataframe for inspection/EDA
    os.makedirs(os.path.dirname(save_processed_path), exist_ok=True)
    engineered_df.to_csv(save_processed_path, index=False)
    print(f"[PREPROCESS] Processed dataset saved to: {save_processed_path}")

    # Extract X and y
    X = engineered_df[FEATURE_COLUMNS]
    y = engineered_df[TARGET_COLUMN]

    # Stratified split to maintain failure ratio across train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save fitted scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[PREPROCESS] Fitted StandardScaler saved to: {scaler_path}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, FEATURE_COLUMNS


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, sc, feats = load_and_preprocess()
    print(f"[PREPROCESS] Training set shape: {X_tr.shape}, Testing set shape: {X_te.shape}")
    print(f"[PREPROCESS] Feature list ({len(feats)} features): {feats}")
