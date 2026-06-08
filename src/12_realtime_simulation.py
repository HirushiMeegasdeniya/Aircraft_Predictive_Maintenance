import os
import time
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "outputs", "predictions", "engine_rul_predictions.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "predictions", "engine_stream.csv")

df = pd.read_csv(INPUT_PATH)

stream_rows = []

print("Starting real-time engine monitoring simulation...")

for index, row in df.iterrows():
    stream_rows.append(row)

    stream_df = pd.DataFrame(stream_rows)
    stream_df.to_csv(OUTPUT_PATH, index=False)

    print(
        f"Engine {row['engine_id']} | "
        f"RUL: {row['predicted_rul']} | "
        f"Risk: {row['risk_level']}"
    )

    time.sleep(1)

print("Simulation completed.")
print("Saved:", OUTPUT_PATH)