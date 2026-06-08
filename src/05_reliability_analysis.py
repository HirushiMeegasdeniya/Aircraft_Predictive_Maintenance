import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from lifelines import KaplanMeierFitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CHART_DIR = os.path.join(BASE_DIR, "outputs", "charts")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def main():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "train_FD001_processed.csv"))

    failure_cycles = df.groupby("engine_id")["cycle"].max().values

    mtbf = np.mean(failure_cycles)

    shape, loc, scale = weibull_min.fit(failure_cycles, floc=0)

    print("MTBF:", round(mtbf, 2))
    print("Weibull shape:", round(shape, 3))
    print("Weibull scale:", round(scale, 3))

    t = np.linspace(1, max(failure_cycles), 200)
    reliability = np.exp(-(t / scale) ** shape)

    plt.figure(figsize=(10, 5))
    plt.plot(t, reliability)
    plt.title("Reliability Curve R(t)")
    plt.xlabel("Cycles")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "reliability_curve.png"))
    plt.close()

    kmf = KaplanMeierFitter()
    kmf.fit(failure_cycles, event_observed=np.ones(len(failure_cycles)))

    plt.figure(figsize=(10, 5))
    kmf.plot_survival_function()
    plt.title("Kaplan-Meier Survival Curve")
    plt.xlabel("Cycles")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "kaplan_meier_curve.png"))
    plt.close()

    report = pd.DataFrame({
        "metric": ["MTBF", "Weibull Shape", "Weibull Scale"],
        "value": [mtbf, shape, scale]
    })

    report.to_csv(os.path.join(REPORT_DIR, "reliability_metrics.csv"), index=False)

    print("Reliability analytics completed.")

if __name__ == "__main__":
    main()