import os
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_PATH = os.path.join(BASE_DIR, "outputs", "predictions", "engine_rul_predictions.csv")
SCHEDULE_PATH = os.path.join(BASE_DIR, "outputs", "reports", "maintenance_schedule.csv")

st.set_page_config(
    page_title="Aircraft Predictive Maintenance",
    layout="wide"
)

st.title("✈️ Aircraft Predictive Maintenance & Reliability Analytics")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Fleet Overview",
        "Engine Risk",
        "Maintenance Planner",
        "Upload Engine Data"
    ]
)

pred_df = pd.read_csv(PRED_PATH)
schedule_df = pd.read_csv(SCHEDULE_PATH)

if page == "Fleet Overview":
    st.header("Fleet Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Engines", pred_df["engine_id"].nunique())
    col2.metric("Average RUL", round(pred_df["predicted_rul"].mean(), 2))
    col3.metric(
        "High Risk Engines",
        pred_df[pred_df["risk_level"].str.contains("Red|Orange")].shape[0]
    )
    col4.metric(
        "Red Engines",
        pred_df[pred_df["risk_level"].str.contains("Red")].shape[0]
    )

    st.subheader("RUL Predictions")
    st.dataframe(pred_df)

elif page == "Engine Risk":
    st.header("Engine Risk Details")

    engine_id = st.selectbox(
        "Select Engine",
        pred_df["engine_id"].unique()
    )

    engine_data = pred_df[pred_df["engine_id"] == engine_id].iloc[0]

    st.metric("Predicted RUL", engine_data["predicted_rul"])
    st.metric("Predicted Failure Cycle", engine_data["predicted_failure_cycle"])
    st.metric("Risk Level", engine_data["risk_level"])

elif page == "Maintenance Planner":
    st.header("Maintenance Planner")

    st.dataframe(schedule_df)

    st.subheader("Cost Saving Summary")

    if "estimated_cost_saving" in schedule_df.columns:
        st.metric(
            "Total Estimated Savings",
            f"${schedule_df['estimated_cost_saving'].sum():,.0f}"
        )

elif page == "Upload Engine Data":
    st.header("Upload New Engine Data")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")
        st.dataframe(uploaded_df.head())

        st.info(
            "Next step: connect this uploaded data to the trained XGBoost model for live prediction."
        )