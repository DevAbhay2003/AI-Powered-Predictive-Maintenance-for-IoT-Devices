import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Run from the root of the existing P1 project, beside main.py.
ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
DATA = ROOT / "data"

st.set_page_config(
    page_title="IoT Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ IoT Predictive Maintenance System")
st.caption("Local dashboard for sensor data, machine-learning results, and simulated equipment alerts.")

def read_json(path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return {}

def read_csv(path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        pass
    return pd.DataFrame()

def show_image(filename, title):
    path = IMAGES / filename
    if path.exists():
        st.subheader(title)
        st.image(str(path), use_container_width=True)
    else:
        st.info(f"{filename} has not been generated yet. Run the pipeline first.")

def count_alerts():
    path = OUTPUTS / "simulation_alerts.log"
    if not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
    except OSError:
        return 0

def run_pipeline():
    main_file = ROOT / "main.py"
    if not main_file.exists():
        st.error("main.py was not found. Save app.py in the same folder as main.py.")
        return

    command = [sys.executable, str(main_file), "--all"]
    st.info("Running the full pipeline. This may take a little while.")
    log_box = st.empty()
    log_lines = []

    try:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                log_lines.append(line.rstrip())
                # Keep the latest 250 lines visible to avoid an oversized page.
                log_box.code("\n".join(log_lines[-250:]), language="text")
        return_code = process.wait()

        if return_code == 0:
            st.success("Pipeline completed successfully. The dashboard data has been refreshed.")
        else:
            st.error(f"The pipeline stopped with exit code {return_code}. Check the log above.")
    except OSError as exc:
        st.error(f"Could not start the pipeline: {exc}")

with st.sidebar:
    st.header("Controls")
    st.write("Project folder:")
    st.code(str(ROOT), language="text")
    if st.button("▶ Run full pipeline", type="primary", use_container_width=True):
        run_pipeline()
    if st.button("↻ Refresh dashboard", use_container_width=True):
        st.rerun()
    st.divider()
    st.subheader("Project status")
    st.write("Model:", "✅ Available" if (MODELS / "predictive_maintenance_model.joblib").exists() else "— Not found")
    st.write("Raw dataset:", "✅ Available" if (DATA / "raw_sensor_data.csv").exists() else "— Not found")
    st.write("Evaluation:", "✅ Available" if (OUTPUTS / "evaluation_metrics.json").exists() else "— Not found")

benchmark = read_csv(OUTPUTS / "model_benchmark_results.csv")
telemetry = read_csv(OUTPUTS / "simulation_stream_log.csv")
metrics = read_json(OUTPUTS / "evaluation_metrics.json")
alerts_path = OUTPUTS / "simulation_alerts.log"
alert_text = alerts_path.read_text(encoding="utf-8", errors="replace") if alerts_path.exists() else ""

# Prefer the benchmark CSV for headline scores because its columns are known.
best_model = "Not available"
accuracy = None
recall = None
f1_score = None
roc_auc = None
if not benchmark.empty and "Model" in benchmark.columns:
    score_col = "F1-Score" if "F1-Score" in benchmark.columns else None
    best_row = benchmark.sort_values(score_col, ascending=False).iloc[0] if score_col else benchmark.iloc[0]
    best_model = str(best_row.get("Model", best_model))
    accuracy = best_row.get("Accuracy")
    recall = best_row.get("Recall")
    f1_score = best_row.get("F1-Score")
    roc_auc = best_row.get("ROC-AUC")

# The console's benchmark values are stored as fractions in the CSV.
def metric_text(value, digits=2):
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"

total_records = "—"
raw = read_csv(DATA / "raw_sensor_data.csv")
if not raw.empty:
    total_records = f"{len(raw):,}"

alert_count = 0
if not telemetry.empty:
    # Count rows whose status indicates a warning or critical alert, if available.
    status_col = next((c for c in telemetry.columns if "status" in c.lower()), None)
    if status_col:
        alert_count = int(telemetry[status_col].astype(str).str.contains("WARNING|CRITICAL|ALERT", case=False, regex=True).sum())
    else:
        alert_count = count_alerts()
else:
    alert_count = count_alerts()

st.subheader("System overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sensor records", total_records)
c2.metric("Best model", best_model)
c3.metric("Benchmark accuracy", metric_text(accuracy))
c4.metric("Simulation alerts", alert_count)

st.divider()
tab_overview, tab_models, tab_sensors, tab_alerts, tab_logs = st.tabs(
    ["Overview", "Model evaluation", "Sensor telemetry", "Alerts", "Files & logs"]
)

with tab_overview:
    left, right = st.columns(2)
    with left:
        show_image("simulation_dashboard.png", "Simulation dashboard")
        show_image("sensor_telemetry_trends.png", "Sensor telemetry trends")
    with right:
        show_image("feature_importance.png", "Feature importance")
        show_image("roc_curve.png", "ROC curve")

with tab_models:
    st.subheader("Model benchmark results")
    if benchmark.empty:
        st.info("No benchmark results found. Run the pipeline to generate them.")
    else:
        st.dataframe(benchmark, use_container_width=True, hide_index=True)
        st.bar_chart(benchmark.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]]
                     if {"Model", "Accuracy", "Precision", "Recall", "F1-Score"}.issubset(benchmark.columns)
                     else benchmark.set_index("Model"))
    show_image("confusion_matrix.png", "Confusion matrix")
    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", metric_text(
        benchmark.sort_values("F1-Score", ascending=False).iloc[0].get("Precision")
        if not benchmark.empty and "F1-Score" in benchmark.columns and "Precision" in benchmark.columns else None
    ))
    m2.metric("Recall", metric_text(recall))
    m3.metric("F1-score", metric_text(f1_score))
    if metrics:
        with st.expander("Raw evaluation metrics (JSON)"):
            st.json(metrics)

with tab_sensors:
    st.subheader("Simulated sensor telemetry")
    if telemetry.empty:
        st.info("No telemetry CSV found. Run the pipeline to generate the simulation stream.")
    else:
        st.dataframe(telemetry, use_container_width=True, hide_index=True)
        numeric_cols = telemetry.select_dtypes(include="number").columns.tolist()
        preferred = [c for c in numeric_cols if any(k in c.lower() for k in ("temp", "rpm", "torque", "vib", "risk"))]
        chosen = st.multiselect("Plot numeric sensor fields", numeric_cols, default=preferred[:4])
        if chosen:
            st.line_chart(telemetry[chosen])
        csv_bytes = telemetry.to_csv(index=False).encode("utf-8")
        st.download_button("Download telemetry CSV", csv_bytes, "simulation_stream_log.csv", "text/csv")

with tab_alerts:
    st.subheader("Simulation alerts")
    if alert_text.strip():
        st.code(alert_text, language="text")
        st.download_button(
            "Download alert log",
            alert_text.encode("utf-8"),
            "simulation_alerts.log",
            "text/plain",
        )
    else:
        st.info("No alert log found yet.")
    st.caption("These alerts are from the project's simulated sensor stream, not a connected physical IoT device.")

with tab_logs:
    st.subheader("Generated project files")
    file_rows = []
    for label, path in [
        ("Raw sensor dataset", DATA / "raw_sensor_data.csv"),
        ("Processed sensor dataset", DATA / "processed_sensor_data.csv"),
        ("Trained model", MODELS / "predictive_maintenance_model.joblib"),
        ("Feature scaler", MODELS / "feature_scaler.joblib"),
        ("Benchmark results", OUTPUTS / "model_benchmark_results.csv"),
        ("Evaluation metrics", OUTPUTS / "evaluation_metrics.json"),
        ("Telemetry log", OUTPUTS / "simulation_stream_log.csv"),
        ("Alert log", OUTPUTS / "simulation_alerts.log"),
    ]:
        file_rows.append({"File": label, "Path": str(path.relative_to(ROOT)), "Status": "Available" if path.exists() else "Not generated"})
    st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)
    st.caption("Use the Run full pipeline button in the sidebar to regenerate results.")
