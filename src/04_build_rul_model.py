import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQUENCE_DIR = os.path.join(BASE_DIR, "data", "sequences")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
CHART_DIR = os.path.join(BASE_DIR, "outputs", "charts")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)


def flatten_sequences(X):
    samples, timesteps, features = X.shape
    return X.reshape(samples, timesteps * features)


def evaluate_model(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 125)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n{model_name} Results")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 3))

    return preds, mae, rmse, r2


def create_actual_vs_predicted_chart(y_test, preds):
    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        preds,
        alpha=0.5
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--"
    )

    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Actual vs Predicted RUL")

    chart_path = os.path.join(
        CHART_DIR,
        "actual_vs_predicted_rul.png"
    )

    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Actual vs Predicted chart saved:", chart_path)


def main():
    X = np.load(os.path.join(SEQUENCE_DIR, "X_train.npy"))
    y = np.load(os.path.join(SEQUENCE_DIR, "y_train.npy"))

    print("Loaded sequences:", X.shape)

    X_flat = flatten_sequences(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_flat,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training Random Forest...")

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_depth=15
    )

    rf.fit(X_train, y_train)

    rf_preds, rf_mae, rf_rmse, rf_r2 = evaluate_model(
        rf,
        X_test,
        y_test,
        "Random Forest"
    )

    rf_path = os.path.join(MODEL_DIR, "random_forest_rul_model.pkl")
    joblib.dump(rf, rf_path)

    print("Training XGBoost...")

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    xgb.fit(X_train, y_train)

    xgb_preds, xgb_mae, xgb_rmse, xgb_r2 = evaluate_model(
        xgb,
        X_test,
        y_test,
        "XGBoost"
    )

    create_actual_vs_predicted_chart(y_test, xgb_preds)

    xgb_path = os.path.join(MODEL_DIR, "xgboost_rul_model.pkl")
    joblib.dump(xgb, xgb_path)

    report_path = os.path.join(REPORT_DIR, "model_metrics.txt")

    with open(report_path, "w") as f:
        f.write("RUL Model Metrics\n")
        f.write("=================\n")
        f.write(f"Random Forest MAE: {rf_mae:.2f}\n")
        f.write(f"Random Forest RMSE: {rf_rmse:.2f}\n")
        f.write(f"Random Forest R2: {rf_r2:.3f}\n\n")
        f.write(f"XGBoost MAE: {xgb_mae:.2f}\n")
        f.write(f"XGBoost RMSE: {xgb_rmse:.2f}\n")
        f.write(f"XGBoost R2: {xgb_r2:.3f}\n")

    print("Models saved in models/")
    print("Metrics saved:", report_path)


if __name__ == "__main__":
    main()