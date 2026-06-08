import os
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "train_FD001_processed.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "predictions")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLUMNS = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(DATA_PATH)

X = df[FEATURE_COLUMNS]

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

df["anomaly_score"] = model.fit_predict(X)

df["anomaly_status"] = df["anomaly_score"].map({
    1: "Normal",
    -1: "Anomaly"
})

output_path = os.path.join(OUTPUT_DIR, "anomaly_detection_results.csv")
model_path = os.path.join(MODEL_DIR, "isolation_forest_model.pkl")

df.to_csv(output_path, index=False)
joblib.dump(model, model_path)

print("Anomaly detection completed.")
print("Saved:", output_path)