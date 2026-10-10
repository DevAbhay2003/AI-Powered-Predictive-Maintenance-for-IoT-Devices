"""
train.py
--------
Trains and benchmarks multiple machine learning algorithms for predictive maintenance:
1. Logistic Regression (Baseline)
2. Random Forest Classifier (Ensemble Bagging)
3. Gradient Boosting Classifier (Ensemble Boosting)

Selects and serializes the highest performing model based on F1-score and ROC-AUC.
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.preprocess import load_and_preprocess


def train_and_evaluate_models():
    """
    Trains candidate models, records performance metrics,
    and exports the best performing model.
    """
    # 1. Load prepared data
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess()

    # 2. Candidate Model Dictionary
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        ),
    }

    results = []
    trained_instances = {}

    print("\n" + "=" * 65)
    print("      BENCHMARKING MACHINE LEARNING MODELS FOR IoT PREDICTIVE MAINTENANCE")
    print("=" * 65)

    for name, model in models.items():
        print(f"\n[TRAIN] Training {name}...")
        model.fit(X_train, y_train)
        trained_instances[name] = model

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc,
        })

    # Summary DataFrame
    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    print("\n" + "=" * 65)
    print("                       MODEL BENCHMARK RESULTS")
    print("=" * 65)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 65)

    # Save benchmark table
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/model_benchmark_results.csv", index=False)
    print("[TRAIN] Benchmark results saved to: outputs/model_benchmark_results.csv")

    # Pick the champion model (highest F1-score)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_instances[best_model_name]
    best_f1 = results_df.iloc[0]["F1-Score"]
    print(f"\n[CHAMPION] Selected best model: {best_model_name} (F1-Score: {best_f1:.4f})")

    # Persist the champion model
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/predictive_maintenance_model.joblib"
    joblib.dump(best_model, model_save_path)
    print(f"[TRAIN] Champion model exported successfully to: {model_save_path}")

    return best_model, best_model_name, results_df


if __name__ == "__main__":
    train_and_evaluate_models()
