"""
data_generator.py
-----------------
Generates realistic industrial IoT sensor telemetry simulating real machinery
(e.g., CNC milling machines, rotating turbines, and industrial motors).
Includes authentic failure modes:
1. Tool Wear Failure (TWF)
2. Heat Dissipation Failure (HDF)
3. Power Surge/Drop Failure (PWF)
4. Overstrain Failure (OSF)
5. Hydraulic Pressure Anomaly
"""

import os
import numpy as np
import pandas as pd


def generate_sensor_data(n_samples: int = 10000, random_seed: int = 42) -> pd.DataFrame:
    """
    Simulates high-fidelity time-series IoT sensor data for predictive maintenance.

    Parameters:
        n_samples (int): Number of telemetry readings to simulate.
        random_seed (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Simulated telemetry dataset with ground-truth failure indicators.
    """
    np.random.seed(random_seed)

    # 1. Temporal index and Machine Identifiers
    start_time = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = [start_time + pd.Timedelta(minutes=5 * i) for i in range(n_samples)]
    machine_ids = np.random.choice(["MCH_101", "MCH_102", "MCH_103", "MCH_104", "MCH_105"], size=n_samples)

    # 2. Baseline Sensor Telemetry
    # Ambient Air Temperature: Normal around 298 K (25°C) with diurnal fluctuations
    air_temp = 298.0 + np.random.normal(0, 2.0, n_samples)

    # Process Temperature: Component operational heat (usually air_temp + 10 K + workload noise)
    process_temp = air_temp + 10.0 + np.random.normal(0, 1.2, n_samples)

    # Rotational Speed (RPM): Operating typically between 1300 and 2200 RPM
    rotational_speed = np.random.normal(1540, 180, n_samples)
    rotational_speed = np.clip(rotational_speed, 1100, 2800)

    # Torque (Nm): Inversely related to speed with operational variation
    torque = 40.0 + (1500 - rotational_speed) * 0.05 + np.random.normal(0, 9.5, n_samples)
    torque = np.clip(torque, 4.0, 78.0)

    # Tool Wear (minutes): Monotonically accumulating operational time
    tool_wear = np.random.exponential(scale=70, size=n_samples)
    tool_wear = np.clip(tool_wear, 0, 250)

    # IoT Vibration Sensor (mm/s): Healthy baseline is 1.5 - 3.5 mm/s
    vibration = 2.2 + (tool_wear / 100.0) * 0.8 + np.random.normal(0, 0.45, n_samples)

    # Hydraulic Pressure (Bar): Standard operating range is 120 - 160 Bar
    hydraulic_pressure = 140.0 + np.random.normal(0, 12.0, n_samples)

    # 3. Physics-based Industrial Failure Mode Simulation
    # Failure Mode 1: Heat Dissipation Failure (HDF) - Poor cooling + high ambient or low airflow
    temp_diff = process_temp - air_temp
    hdf_condition = (temp_diff < 8.6) & (rotational_speed < 1380)

    # Failure Mode 2: Power Surge/Drop Failure (PWF) - Mechanical Power P = Torque * w
    # w = (2 * pi / 60) * RPM. Power in Watts = Torque (Nm) * w (rad/s)
    power_watts = torque * (rotational_speed * 2 * np.pi / 60)
    pwf_condition = (power_watts < 3500) | (power_watts > 9200)

    # Failure Mode 3: Overstrain Failure (OSF) - Excessive torque combined with high tool wear
    osf_condition = (tool_wear * torque > 11200)

    # Failure Mode 4: Tool Wear Failure (TWF) - Tool degradation past limit + elevated vibration
    twf_condition = (tool_wear > 210) & (vibration > 4.2)

    # Failure Mode 5: Hydraulic Pressure Failure (HPF)
    hpf_condition = (hydraulic_pressure < 105.0) | (hydraulic_pressure > 175.0)

    # Aggregate Failure Target (1 = Failure occurred, 0 = Healthy)
    raw_failure = (
        hdf_condition | pwf_condition | osf_condition | twf_condition | hpf_condition
    )

    # Inject realistic industrial noise / intermittent false alarms and marginal transitions
    # 2% random intermittent sensor anomalies (near failure threshold without immediate breakdown)
    # and 3% stochastic edge cases
    failure_probability = np.zeros(n_samples)
    failure_probability[raw_failure] = 0.92
    random_stress = (vibration > 3.8) & (torque > 60.0)
    failure_probability[random_stress] = np.maximum(failure_probability[random_stress], 0.75)
    
    # Generate binary label with stochastic decision threshold
    rand_draw = np.random.uniform(0, 1, n_samples)
    failure_target = (rand_draw < failure_probability).astype(int)

    # Subtle degradation signature
    vibration = np.where(failure_target == 1, vibration + np.random.normal(1.2, 0.6, n_samples), vibration)
    process_temp = np.where(failure_target == 1, process_temp + np.random.normal(1.5, 0.8, n_samples), process_temp)

    # Assemble into DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "machine_id": machine_ids,
        "air_temperature_k": np.round(air_temp, 2),
        "process_temperature_k": np.round(process_temp, 2),
        "rotational_speed_rpm": np.round(rotational_speed, 1),
        "torque_nm": np.round(torque, 2),
        "tool_wear_min": np.round(tool_wear, 1),
        "vibration_mm_s": np.round(vibration, 3),
        "hydraulic_pressure_bar": np.round(hydraulic_pressure, 2),
        "hdf_failure": hdf_condition.astype(int),
        "pwf_failure": pwf_condition.astype(int),
        "osf_failure": osf_condition.astype(int),
        "twf_failure": twf_condition.astype(int),
        "hpf_failure": hpf_condition.astype(int),
        "machine_failure": failure_target
    })

    return df


def save_dataset(df: pd.DataFrame, file_path: str = "data/raw_sensor_data.csv") -> None:
    """Saves generated dataset to CSV."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"[DATA GENERATOR] Saved {len(df):,} IoT sensor records to: {file_path}")
    fail_count = df["machine_failure"].sum()
    print(f"[DATA GENERATOR] Healthy Samples: {len(df) - fail_count:,} ({(1 - fail_count/len(df))*100:.2f}%)")
    print(f"[DATA GENERATOR] Failure Events:  {fail_count:,} ({fail_count/len(df)*100:.2f}%)")


if __name__ == "__main__":
    dataset = generate_sensor_data(n_samples=10000, random_seed=42)
    save_dataset(dataset)
