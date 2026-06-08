import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(BASE_DIR, "outputs", "predictions")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(REPORT_DIR, exist_ok=True)

def recommend_action(risk):
    if "Red" in risk:
        return "Immediate inspection within 24 hours"
    elif "Orange" in risk:
        return "Schedule maintenance within 1 week"
    elif "Yellow" in risk:
        return "Monitor closely and inspect within 2-4 weeks"
    else:
        return "Normal operation"

def estimate_downtime(risk):
    if "Red" in risk:
        return 12
    elif "Orange" in risk:
        return 8
    elif "Yellow" in risk:
        return 4
    else:
        return 0

def main():
    path = os.path.join(PRED_DIR, "engine_rul_predictions.csv")

    if not os.path.exists(path):
        raise FileNotFoundError("Run 06_predict_failure_risk.py first.")

    df = pd.read_csv(path)

    df["recommended_action"] = df["risk_level"].apply(recommend_action)
    df["estimated_downtime_hours"] = df["risk_level"].apply(estimate_downtime)

    df["priority"] = df["risk_level"].map({
        "Red - Immediate Inspection": 1,
        "Orange - Schedule Maintenance": 2,
        "Yellow - Monitor": 3,
        "Green - Healthy": 4
    })

    df = df.sort_values(["priority", "predicted_rul"])

    unscheduled_cost = 50000
    planned_cost = 15000
    df["estimated_cost_saving"] = df["risk_level"].apply(
        lambda x: unscheduled_cost - planned_cost if "Red" in x or "Orange" in x else 0
    )

    output_path = os.path.join(REPORT_DIR, "maintenance_schedule.csv")
    df.to_csv(output_path, index=False)

    print("Maintenance schedule saved:", output_path)
    print(df.head(20))

if __name__ == "__main__":
    main()