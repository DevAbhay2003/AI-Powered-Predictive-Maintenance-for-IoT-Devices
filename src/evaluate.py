"""
evaluate.py
-----------
Comprehensive model evaluation, metrics export, and publication-ready
visualizations (Confusion Matrix, ROC Curve, Feature Importance, Sensor Trends).
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

from src.preprocess import load_and_preprocess, FEATURE_COLUMNS


def run_evaluation():
    """
    Evaluates the champion model, generates visualization artifacts,
    and saves metric reports.
    """
    os.makedirs("images", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Load data & trained model
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess()

    model_path = "models/predictive_maintenance_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    model = joblib.load(model_path)
    print(f"[EVALUATE] Loaded champion model: {type(model).__name__}")

    # 2. Compute predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    # 3. Textual classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(
        y_test, y_pred, target_names=["Healthy (0)", "Failure (1)"]
    )
    print("\n" + "=" * 60)
    print("                DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    print(report_text)
    print("=" * 60)

    # Save metrics JSON
    with open("outputs/evaluation_metrics.json", "w") as f:
        json.dump(report_dict, f, indent=4)
    print("[EVALUATE] Metrics JSON saved to: outputs/evaluation_metrics.json")

    # Set visualization styling
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = sns.color_palette("mako")

    # -------------------------------------------------------------
    # PLOT 1: Confusion Matrix Heatmap
    # -------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Healthy", "Predicted Failure"],
        yticklabels=["Actual Healthy", "Actual Failure"],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title("Industrial Machine Failure Confusion Matrix", fontsize=12, pad=12, weight="bold")
    plt.tight_layout()
    cm_path = "images/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[EVALUATE] Saved Confusion Matrix to: {cm_path}")

    # -------------------------------------------------------------
    # PLOT 2: ROC Curve
    # -------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC Curve (AUC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="#d62728", lw=1.5, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("ROC-AUC Characteristic Curve", fontsize=12, weight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_path = "images/roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[EVALUATE] Saved ROC Curve to: {roc_path}")

    # -------------------------------------------------------------
    # PLOT 3: Feature Importance
    # -------------------------------------------------------------
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        plt.figure(figsize=(9, 5))
        sns.barplot(
            x=sorted_importances,
            y=sorted_features,
            palette="viridis",
            hue=sorted_features,
            legend=False,
        )
        plt.title("IoT Telemetry Feature Importance for Predictive Maintenance", fontsize=12, weight="bold")
        plt.xlabel("Relative Importance Score", fontsize=11)
        plt.ylabel("Telemetry Feature", fontsize=11)
        plt.tight_layout()
        fi_path = "images/feature_importance.png"
        plt.savefig(fi_path, dpi=300)
        plt.close()
        print(f"[EVALUATE] Saved Feature Importance to: {fi_path}")

    # -------------------------------------------------------------
    # PLOT 4: Sensor Telemetry Distribution & Anomaly Signatures
    # -------------------------------------------------------------
    raw_df = pd.read_csv("data/raw_sensor_data.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Vibration distribution
    sns.kdeplot(
        data=raw_df,
        x="vibration_mm_s",
        hue="machine_failure",
        common_norm=False,
        fill=True,
        ax=axes[0, 0],
        palette={0: "#2ca02c", 1: "#d62728"},
    )
    axes[0, 0].set_title("Vibration Sensor: Healthy vs. Failing", weight="bold")
    axes[0, 0].set_xlabel("Vibration (mm/s)")

    # Tool Wear distribution
    sns.kdeplot(
        data=raw_df,
        x="tool_wear_min",
        hue="machine_failure",
        common_norm=False,
        fill=True,
        ax=axes[0, 1],
        palette={0: "#2ca02c", 1: "#d62728"},
    )
    axes[0, 1].set_title("Tool Wear Degradation: Healthy vs. Failing", weight="bold")
    axes[0, 1].set_xlabel("Tool Wear (minutes)")

    # Torque vs Rotational Speed Scatter
    sns.scatterplot(
        data=raw_df.sample(min(1500, len(raw_df)), random_state=42),
        x="rotational_speed_rpm",
        y="torque_nm",
        hue="machine_failure",
        alpha=0.6,
        ax=axes[1, 0],
        palette={0: "#1f77b4", 1: "#e377c2"},
    )
    axes[1, 0].set_title("Operating Envelope (RPM vs Torque)", weight="bold")
    axes[1, 0].set_xlabel("Spindle Speed (RPM)")
    axes[1, 0].set_ylabel("Motor Torque (Nm)")

    # Process Temperature distribution
    sns.kdeplot(
        data=raw_df,
        x="process_temperature_k",
        hue="machine_failure",
        common_norm=False,
        fill=True,
        ax=axes[1, 1],
        palette={0: "#2ca02c", 1: "#d62728"},
    )
    axes[1, 1].set_title("Thermal Dissipation (Process Temp K)", weight="bold")
    axes[1, 1].set_xlabel("Process Temp (Kelvin)")

    plt.tight_layout()
    trends_path = "images/sensor_telemetry_trends.png"
    plt.savefig(trends_path, dpi=300)
    plt.close()
    print(f"[EVALUATE] Saved Sensor Telemetry Trends to: {trends_path}")

    print("\n[EVALUATE] All evaluation visual assets generated successfully!")


if __name__ == "__main__":
    run_evaluation()
