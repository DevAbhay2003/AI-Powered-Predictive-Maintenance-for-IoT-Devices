"""
simulate_stream.py
------------------
Virtual IoT Edge Gateway Simulator for Predictive Maintenance.
Simulates a live streaming sensor feed from an industrial machine, passes telemetry
through the trained ML model in real-time, calculates failure probabilities,
generates alert triggers, and exports live operational dashboards.
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocess import engineer_features, FEATURE_COLUMNS


def run_iot_simulation(
    n_cycles: int = 30,
    delay_seconds: float = 0.05,
    machine_id: str = "MCH_101",
    save_plot: bool = True,
):
    """
    Simulates real-time IoT edge telemetry and dynamic predictive maintenance alerts.

    Parameters:
        n_cycles (int): Number of streaming cycles to simulate.
        delay_seconds (float): Real-time interval delay between readings.
        machine_id (str): Target simulated equipment ID.
        save_plot (bool): Whether to generate an operational dashboard plot.
    """
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    # 1. Load artifacts
    model_path = "models/predictive_maintenance_model.joblib"
    scaler_path = "models/feature_scaler.joblib"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or Scaler not found. Train the model first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("\n" + "=" * 80)
    print(f"      IoT EDGE PREDICTIVE MAINTENANCE SIMULATOR - ASSET: {machine_id}")
    print("=" * 80)
    print(f"{'CYCLE':^6} | {'TEMP(K)':^9} | {'RPM':^8} | {'TORQUE':^8} | {'VIB(mm/s)':^10} | {'RISK %':^8} | {'STATUS':^16}")
    print("-" * 80)

    stream_records = []
    alert_logs = []

    # Simulation state variables (accumulating wear & thermal load)
    air_temp = 298.0
    process_temp = 308.5
    rot_speed = 1550.0
    torque = 39.0
    tool_wear = 80.0
    vibration = 2.4
    pressure = 140.0

    for cycle in range(1, n_cycles + 1):
        # Operational degradation trajectory:
        # After cycle 18, simulate progressive bearing fatigue and cooling failure
        if cycle < 18:
            tool_wear += np.random.uniform(1.0, 2.5)
            air_temp += np.random.normal(0, 0.2)
            process_temp = air_temp + 10.5 + np.random.normal(0, 0.3)
            rot_speed = 1550.0 + np.random.normal(0, 25)
            torque = 39.0 + np.random.normal(0, 1.8)
            vibration = 2.3 + (tool_wear / 150.0) * 0.4 + np.random.normal(0, 0.1)
            pressure = 140.0 + np.random.normal(0, 2.0)
        else:
            # Accelerated failure phase
            tool_wear += np.random.uniform(4.0, 8.0)
            air_temp += 0.3
            process_temp += np.random.uniform(1.2, 2.8)  # Heat dissipation breakdown
            rot_speed -= np.random.uniform(20, 45)       # Spindle drag
            torque += np.random.uniform(2.5, 4.0)        # Motor fighting resistance
            vibration += np.random.uniform(0.4, 0.9)     # Bearing flaking / high vibration
            pressure += np.random.choice([-4.5, 5.0])    # Hydraulic fluctuation

        record = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cycle": cycle,
            "machine_id": machine_id,
            "air_temperature_k": round(float(air_temp), 2),
            "process_temperature_k": round(float(process_temp), 2),
            "rotational_speed_rpm": round(float(rot_speed), 1),
            "torque_nm": round(float(torque), 2),
            "tool_wear_min": round(float(tool_wear), 1),
            "vibration_mm_s": round(float(vibration), 3),
            "hydraulic_pressure_bar": round(float(pressure), 2),
        }

        # Real-time feature calculation
        temp_df = pd.DataFrame([record])
        engineered = engineer_features(temp_df)
        X_live = engineered[FEATURE_COLUMNS]
        X_scaled = scaler.transform(X_live)

        # Inference
        pred_prob = float(model.predict_proba(X_scaled)[0, 1]) if hasattr(model, "predict_proba") else 0.0
        risk_percentage = round(pred_prob * 100, 1)
        record["failure_probability"] = risk_percentage

        # Alert tiering
        if risk_percentage < 30.0:
            status = "NORMAL"
            status_display = "[OK] NORMAL"
        elif risk_percentage < 70.0:
            status = "WARNING"
            status_display = "[!] WARNING"
            alert_msg = f"[{record['timestamp']}] WARNING: Asset {machine_id} Cycle {cycle} elevated risk: {risk_percentage}% (Vib: {vibration:.2f} mm/s, Temp: {process_temp:.1f} K)"
            alert_logs.append(alert_msg)
        else:
            status = "CRITICAL"
            status_display = "[CRITICAL ALERT]"
            alert_msg = f"[{record['timestamp']}] CRITICAL: Asset {machine_id} Cycle {cycle} imminent failure predicted: {risk_percentage}% - Automated Work Order Dispatched!"
            alert_logs.append(alert_msg)

        record["status"] = status
        stream_records.append(record)

        print(
            f"{cycle:^6} | "
            f"{process_temp:^9.1f} | "
            f"{rot_speed:^8.0f} | "
            f"{torque:^8.1f} | "
            f"{vibration:^10.2f} | "
            f"{risk_percentage:^7.1f}% | "
            f"{status_display:<16}"
        )

        time.sleep(delay_seconds)

    print("-" * 80)

    # 3. Save logs and CSV
    sim_df = pd.DataFrame(stream_records)
    sim_df.to_csv("outputs/simulation_stream_log.csv", index=False)
    print(f"\n[SIMULATOR] Telemetry stream saved to: outputs/simulation_stream_log.csv")

    with open("outputs/simulation_alerts.log", "w") as f:
        for log in alert_logs:
            f.write(log + "\n")
    print(f"[SIMULATOR] Alert logs saved to: outputs/simulation_alerts.log ({len(alert_logs)} alert events)")

    # 4. Generate Visual Dashboard
    if save_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

        # Subplot 1: Vibration & Pressure
        ax1.plot(sim_df["cycle"], sim_df["vibration_mm_s"], color="#d62728", lw=2, marker="o", label="Vibration (mm/s)")
        ax1.axhline(4.0, color="orange", linestyle="--", alpha=0.7, label="Vibration Warning Threshold (4.0 mm/s)")
        ax1.set_ylabel("Vibration (mm/s)", weight="bold")
        ax1.set_title(f"Industrial Machine Health Telemetry Stream - Asset: {machine_id}", weight="bold", fontsize=12)
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Subplot 2: Thermal Signature
        ax2.plot(sim_df["cycle"], sim_df["process_temperature_k"], color="#e6550d", lw=2, marker="s", label="Process Temp (K)")
        ax2.plot(sim_df["cycle"], sim_df["air_temperature_k"], color="#3182bd", lw=1.5, linestyle=":", label="Ambient Air Temp (K)")
        ax2.set_ylabel("Temperature (Kelvin)", weight="bold")
        ax2.legend(loc="upper left")
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Subplot 3: AI Predicted Breakdown Risk
        ax3.plot(sim_df["cycle"], sim_df["failure_probability"], color="#756bb1", lw=2.5, marker="^", label="AI Failure Probability (%)")
        ax3.axhline(70.0, color="red", linestyle="--", lw=1.5, label="Critical Alarm Threshold (70%)")
        ax3.axhline(30.0, color="gold", linestyle="--", lw=1.5, label="Advisory Warning Threshold (30%)")
        ax3.fill_between(sim_df["cycle"], 70.0, 100.0, color="red", alpha=0.15, label="Emergency Intervention Zone")
        ax3.set_ylabel("Failure Risk (%)", weight="bold")
        ax3.set_xlabel("Operational Telemetry Cycle (Time Step)", weight="bold")
        ax3.set_ylim(-5, 105)
        ax3.legend(loc="upper left")
        ax3.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        dashboard_path = "images/simulation_dashboard.png"
        plt.savefig(dashboard_path, dpi=300)
        plt.close()
        print(f"[SIMULATOR] Visual dashboard saved to: {dashboard_path}")

    return sim_df


if __name__ == "__main__":
    run_iot_simulation(n_cycles=30, delay_seconds=0.01)
