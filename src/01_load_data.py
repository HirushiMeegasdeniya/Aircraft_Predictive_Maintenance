import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def load_cmapss_file(filename):
    path = os.path.join(RAW_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Loading {filename}...")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    df = df.iloc[:, :26]
    df.columns = COLUMNS

    print(f"Loaded shape: {df.shape}")
    return df

def add_rul(df, max_rul=125):
    print("Creating RUL target...")

    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycles, on="engine_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=max_rul)
    df.drop(columns=["max_cycle"], inplace=True)

    return df

def main():
    train_df = load_cmapss_file("train_FD001.txt")
    test_df = load_cmapss_file("test_FD001.txt")

    train_df = train_df[train_df["engine_id"] <= 50].copy()
    test_df = test_df[test_df["engine_id"] <= 50].copy()

    train_df = add_rul(train_df)

    train_output = os.path.join(PROCESSED_DIR, "train_FD001_processed.csv")
    test_output = os.path.join(PROCESSED_DIR, "test_FD001_processed.csv")

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    print("Saved processed files:")
    print(train_output)
    print(test_output)
    print(train_df.head())

if __name__ == "__main__":
    main()