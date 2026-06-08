import os
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PRED_DIR = os.path.join(BASE_DIR, "outputs", "predictions")

os.makedirs(PRED_DIR, exist_ok=True)

FEATURE_COLUMNS = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

WINDOW_SIZE = 30

def assign_risk_level(rul):
    if rul > 100:
        return "Green - Healthy"
    elif rul > 30:
        return "Yellow - Monitor"
    elif rul > 10:
        return "Orange - Schedule Maintenance"
    else:
        return "Red - Immediate Inspection"

def main():
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test_FD001_processed.csv"))

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "xgboost_rul_model.pkl"))

    test_df[FEATURE_COLUMNS] = test_df[FEATURE_COLUMNS].fillna(test_df[FEATURE_COLUMNS].median())
    test_df[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    predictions = []

    for engine_id in test_df["engine_id"].unique():
        engine_df = test_df[test_df["engine_id"] == engine_id].sort_values("cycle")

        if len(engine_df) < WINDOW_SIZE:
            continue

        latest_window = engine_df[FEATURE_COLUMNS].values[-WINDOW_SIZE:]
        latest_window_flat = latest_window.reshape(1, -1)

        predicted_rul = model.predict(latest_window_flat)[0]
        predicted_rul = max(0, min(125, predicted_rul))

        latest_cycle = engine_df["cycle"].max()
        predicted_failure_cycle = latest_cycle + predicted_rul

        predictions.append({
            "engine_id": engine_id,
            "latest_cycle": latest_cycle,
            "predicted_rul": round(predicted_rul, 2),
            "predicted_failure_cycle": round(predicted_failure_cycle, 2),
            "risk_level": assign_risk_level(predicted_rul)
        })

    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values("predicted_rul")

    output_path = os.path.join(PRED_DIR, "engine_rul_predictions.csv")
    pred_df.to_csv(output_path, index=False)

    print("Predictions saved:", output_path)
    print(pred_df.head(10))

if __name__ == "__main__":
    main()