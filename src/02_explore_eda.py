import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CHART_DIR = os.path.join(BASE_DIR, "outputs", "charts")

os.makedirs(CHART_DIR, exist_ok=True)

def main():
    path = os.path.join(PROCESSED_DIR, "train_FD001_processed.csv")

    if not os.path.exists(path):
        raise FileNotFoundError("Run 01_load_data.py first.")

    df = pd.read_csv(path)

    print("Dataset shape:", df.shape)
    print(df.info())
    print(df.describe())

    selected_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15"]

    for sensor in selected_sensors:
        plt.figure(figsize=(10, 5))
        for engine in df["engine_id"].unique()[:5]:
            engine_df = df[df["engine_id"] == engine]
            plt.plot(engine_df["cycle"], engine_df[sensor], label=f"Engine {engine}")

        plt.title(f"{sensor} trend over cycles")
        plt.xlabel("Cycle")
        plt.ylabel(sensor)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, f"{sensor}_trend.png"))
        plt.close()

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "correlation_heatmap.png"))
    plt.close()

    rul_corr = corr["RUL"].sort_values()
    rul_corr.to_csv(os.path.join(CHART_DIR, "sensor_rul_correlation.csv"))

    print("EDA charts saved in outputs/charts/")

if __name__ == "__main__":
    main()