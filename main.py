"""
main.py
-------
Master CLI Orchestrator for AI-Powered Predictive Maintenance System for IoT Devices.

Usage Examples:
    python main.py --all
    python main.py --generate-data
    python main.py --train
    python main.py --evaluate
    python main.py --simulate --cycles 25
"""

import argparse
import sys
import os

from src.data_generator import generate_sensor_data, save_dataset
from src.preprocess import load_and_preprocess
from src.train import train_and_evaluate_models
from src.evaluate import run_evaluation
from src.simulate_stream import run_iot_simulation


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Predictive Maintenance System for IoT Devices"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run end-to-end pipeline: data generation, preprocessing, model training, evaluation, and live simulation.",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate synthetic industrial IoT sensor telemetry dataset.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run data cleaning and industrial feature engineering.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train and benchmark candidate ML models (Logistic Regression, Random Forest, Gradient Boosting).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate champion model and generate publication-ready plots (Confusion Matrix, ROC, Feature Importance).",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Execute virtual IoT edge streaming simulation and real-time alert trigger engine.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of telemetry samples to generate (default: 10000).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=30,
        help="Number of streaming cycles for simulation (default: 30).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.03,
        help="Delay in seconds between simulated streaming cycles (default: 0.03).",
    )

    args = parser.parse_args()

    # If no flags are provided, show banner and run help or default to --all
    if not (args.all or args.generate_data or args.preprocess or args.train or args.evaluate or args.simulate):
        print("=" * 75)
        print("    AI-POWERED PREDICTIVE MAINTENANCE SYSTEM FOR IoT DEVICES")
        print("=" * 75)
        print("No specific action selected. Running full end-to-end pipeline (--all)...\n")
        args.all = True

    if args.all or args.generate_data:
        print("\n[PHASE 1] Generating high-fidelity IoT sensor dataset...")
        df = generate_sensor_data(n_samples=args.samples)
        save_dataset(df)

    if args.all or args.preprocess:
        print("\n[PHASE 2] Executing data cleaning and industrial feature engineering...")
        load_and_preprocess()

    if args.all or args.train:
        print("\n[PHASE 3] Training and benchmarking machine learning models...")
        train_and_evaluate_models()

    if args.all or args.evaluate:
        print("\n[PHASE 4] Running comprehensive model evaluation & generating visual assets...")
        run_evaluation()

    if args.all or args.simulate:
        print("\n[PHASE 5] Launching virtual IoT edge telemetry streaming simulation...")
        from src.simulate_stream import run_iot_simulation
        run_iot_simulation(n_cycles=args.cycles, delay_seconds=args.delay)

    print("\n" + "=" * 75)
    print(" [COMPLETED] Predictive Maintenance Pipeline Executed Successfully!")
    print(" Visual plots saved to:    ./images/")
    print(" Trained models saved to:   ./models/")
    print(" Logs and metrics saved to: ./outputs/")
    print("=" * 75)


if __name__ == "__main__":
    main()
