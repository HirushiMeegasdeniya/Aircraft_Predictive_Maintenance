import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQUENCE_DIR = os.path.join(BASE_DIR, "data", "sequences")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHART_DIR = os.path.join(BASE_DIR, "outputs", "charts")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

X = np.load(os.path.join(SEQUENCE_DIR, "X_train.npy"))
y = np.load(os.path.join(SEQUENCE_DIR, "y_train.npy"))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Sequential()

model.add(
    LSTM(
        64,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        return_sequences=False
    )
)

model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=64,
    callbacks=[early_stop]
)

preds = model.predict(X_test).flatten()
preds = np.clip(preds, 0, 125)

mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)

print("LSTM Results")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2:", round(r2, 3))

model.save(os.path.join(MODEL_DIR, "lstm_rul_model.h5"))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, preds, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("LSTM Actual vs Predicted RUL")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "lstm_actual_vs_predicted.png"), dpi=300)
plt.close()

with open(os.path.join(REPORT_DIR, "lstm_model_metrics.txt"), "w") as f:
    f.write("LSTM Model Metrics\n")
    f.write("==================\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2: {r2:.3f}\n")

print("LSTM model saved.")