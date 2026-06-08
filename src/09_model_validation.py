import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEQUENCE_DIR = os.path.join(BASE_DIR, "data", "sequences")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHART_DIR = os.path.join(BASE_DIR, "outputs", "charts")

os.makedirs(CHART_DIR, exist_ok=True)

# Load data
X = np.load(os.path.join(SEQUENCE_DIR, "X_train.npy"))
y = np.load(os.path.join(SEQUENCE_DIR, "y_train.npy"))

# Flatten sequences
samples, timesteps, features = X.shape
X = X.reshape(samples, timesteps * features)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load model
model = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_rul_model.pkl")
)

# Predict
predictions = model.predict(X_test)

# Plot
plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    predictions,
    alpha=0.5
)

plt.plot(
    [0,125],
    [0,125],
    color="red"
)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")

plt.title("Actual vs Predicted RUL")

plt.savefig(
    os.path.join(
        CHART_DIR,
        "actual_vs_predicted.png"
    )
)

print("Chart saved successfully.")