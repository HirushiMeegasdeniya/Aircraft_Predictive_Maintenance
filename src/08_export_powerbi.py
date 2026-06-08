import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PRED_DIR = os.path.join(BASE_DIR, "outputs", "predictions")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
DASHBOARD_DIR = os.path.join(BASE_DIR, "data", "dashboard")

os.makedirs(DASHBOARD_DIR, exist_ok=True)

def main():
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train_FD001_processed.csv"))
    predictions_df = pd.read_csv(os.path.join(PRED_DIR, "engine_rul_predictions.csv"))
    schedule_df = pd.read_csv(os.path.join(REPORT_DIR, "maintenance_schedule.csv"))
    reliability_df = pd.read_csv(os.path.join(REPORT_DIR, "reliability_metrics.csv"))

    train_export = train_df.copy()
    train_export.to_csv(os.path.join(DASHBOARD_DIR, "sensor_history.csv"), index=False)

    predictions_df.to_csv(os.path.join(DASHBOARD_DIR, "rul_predictions.csv"), index=False)
    schedule_df.to_csv(os.path.join(DASHBOARD_DIR, "maintenance_schedule.csv"), index=False)
    reliability_df.to_csv(os.path.join(DASHBOARD_DIR, "reliability_metrics.csv"), index=False)

    fleet_summary = pd.DataFrame({
        "total_engines": [predictions_df["engine_id"].nunique()],
        "average_predicted_rul": [predictions_df["predicted_rul"].mean()],
        "red_engines": [predictions_df["risk_level"].str.contains("Red").sum()],
        "orange_engines": [predictions_df["risk_level"].str.contains("Orange").sum()],
        "yellow_engines": [predictions_df["risk_level"].str.contains("Yellow").sum()],
        "green_engines": [predictions_df["risk_level"].str.contains("Green").sum()]
    })

    fleet_summary.to_csv(os.path.join(DASHBOARD_DIR, "fleet_summary.csv"), index=False)

    print("Power BI files exported to data/dashboard/")

if __name__ == "__main__":
    main()