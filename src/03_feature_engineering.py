import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SEQUENCE_DIR = os.path.join(BASE_DIR, "data", "sequences")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(SEQUENCE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLUMNS = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def create_sequences(df, window_size=30):
    X, y, engine_ids, cycles = [], [], [], []

    for engine_id in df["engine_id"].unique():
        engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")

        feature_values = engine_df[FEATURE_COLUMNS].values
        rul_values = engine_df["RUL"].values
        cycle_values = engine_df["cycle"].values

        for i in range(window_size, len(engine_df)):
            X.append(feature_values[i-window_size:i])
            y.append(rul_values[i])
            engine_ids.append(engine_id)
            cycles.append(cycle_values[i])

    return np.array(X), np.array(y), np.array(engine_ids), np.array(cycles)

def main():
    path = os.path.join(PROCESSED_DIR, "train_FD001_processed.csv")

    if not os.path.exists(path):
        raise FileNotFoundError("Run 01_load_data.py first.")

    df = pd.read_csv(path)

    print("Checking missing values...")
    print(df.isnull().sum().sum())

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())

    scaler = StandardScaler()
    df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    print("Scaler saved:", scaler_path)

    X, y, engine_ids, cycles = create_sequences(df, window_size=30)

    np.save(os.path.join(SEQUENCE_DIR, "X_train.npy"), X)
    np.save(os.path.join(SEQUENCE_DIR, "y_train.npy"), y)
    np.save(os.path.join(SEQUENCE_DIR, "engine_ids.npy"), engine_ids)
    np.save(os.path.join(SEQUENCE_DIR, "cycles.npy"), cycles)

    print("Sequences created.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

if __name__ == "__main__":
    main()