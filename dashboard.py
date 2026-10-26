from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "activity_model.joblib"
RESULT_PATH = ROOT / "outputs" / "model_comparison.csv"
SUMMARY_PATH = ROOT / "outputs" / "experiment_summary.json"

st.set_page_config(
    page_title="ActivitySense | HAR Research",
    page_icon="📱",
    layout="wide",
)

st.title("ActivitySense")
st.caption("Human Activity Recognition · Machine Learning Research Dashboard")

st.info(
    "This project uses the UCI Human Activity Recognition dataset. "
    "It recognizes six everyday activities. The standard dataset does not contain "
    "a FALL class, so this dashboard must not be presented as a validated fall detector."
)

if not MODEL_PATH.exists() or not RESULT_PATH.exists():
    st.warning("The model has not been trained yet.")
    st.markdown("Open a terminal in the project folder and run:")
    st.code("python app.py", language="powershell")
    st.write(
        "The script downloads the public dataset if needed, trains baseline models, "
        "and saves evaluation results. Then refresh this dashboard."
    )
    st.stop()

bundle = joblib.load(MODEL_PATH)
results = pd.read_csv(RESULT_PATH)
with open(SUMMARY_PATH, "r", encoding="utf-8") as file:
    summary = json.load(file)

top = st.columns(4)
top[0].metric("Training examples", f"{summary['train_samples']:,}")
top[1].metric("Test examples", f"{summary['test_samples']:,}")
top[2].metric("Input features", f"{summary['features']}")
top[3].metric("Selected model", summary["selected_model"])

st.subheader("Model comparison")
st.dataframe(
    results.style.format({
        "accuracy": "{:.3%}",
        "balanced_accuracy": "{:.3%}",
        "macro_f1": "{:.3%}",
        "weighted_f1": "{:.3%}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Try an example prediction")
st.write(
    "Upload a CSV containing the same 561 feature columns as the UCI HAR feature "
    "matrix. Each row should represent one already-preprocessed feature vector."
)
uploaded = st.file_uploader("Upload feature rows (.csv)", type=["csv"])
if uploaded is not None:
    try:
        frame = pd.read_csv(uploaded)
        expected = bundle["feature_names"]
        if list(frame.columns) != expected:
            st.error(
                "The CSV columns must match the exact UCI HAR feature names and order. "
                "This upload option is for preprocessed feature vectors, not raw phone sensor readings."
            )
        elif frame.isna().any().any():
            st.error("The CSV contains missing values. Please clean it before prediction.")
        else:
            predictions = bundle["model"].predict(frame)
            output = frame.copy()
            output["predicted_activity"] = predictions
            st.success(f"Predicted {len(predictions)} row(s).")
            st.dataframe(output[["predicted_activity"]], use_container_width=True)
            st.download_button(
                "Download predictions",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="activity_predictions.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"Could not process the uploaded CSV: {exc}")

with st.expander("Research limitations"):
    for item in summary["limitations"]:
        st.write(f"- {item}")
    st.write(
        "The model predicts activities from engineered dataset features. "
        "It does not read live smartphone sensors and is not a medical device."
    )

st.caption("Author: Muhammad Hasnain Maqsood · Thal University of Bhakkar, Pakistan")
