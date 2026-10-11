# =============================================================================
# Scalar Nu surrogate, candidate 2 of 3: Kolmogorov-Arnold Network (KAN).
# Same train/test split (70/30, random_state=42) as the XGBoost candidate for
# a fair comparison. Re and Ri are both log-transformed before normalization
# (each spans 2+ orders of magnitude); all inputs/target min-max normalized
# to [-1,1] for KAN's spline basis (grid_range default). Metrics reported in
# original (un-normalized) Nu units for direct comparability with XGBoost.
# =============================================================================

import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from kan import KAN

DATA = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\scalar_dataset_filtered_v2_re_ri.csv"
OUT_METRICS = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\results\kan_scalar_metrics_v2.json"
OUT_MODEL_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\models\kan_scalar_v2"
FEATURES = ["r_star", "epsilon", "re", "ri"]
TARGET = "nu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def normalize(arr, lo, hi):
    return 2 * (arr - lo) / (hi - lo) - 1


def denormalize(arr, lo, hi):
    return (arr + 1) / 2 * (hi - lo) + lo


def main():
    df = pd.read_csv(DATA)
    X_raw = df[FEATURES].copy()
    X_raw["re"] = np.log10(X_raw["re"])  # log-transform the wide-range features
    X_raw["ri"] = np.log10(X_raw["ri"])
    X = X_raw.values
    y = df[TARGET].values

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.30, random_state=SEED)

    x_lo, x_hi = X_train_raw.min(axis=0), X_train_raw.max(axis=0)
    y_lo, y_hi = y_train_raw.min(), y_train_raw.max()

    X_train = normalize(X_train_raw, x_lo, x_hi)
    X_test = normalize(X_test_raw, x_lo, x_hi)
    y_train = normalize(y_train_raw, y_lo, y_hi)
    y_test = normalize(y_test_raw, y_lo, y_hi)

    dataset = {
        "train_input": torch.tensor(X_train, dtype=torch.float32),
        "train_label": torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        "test_input": torch.tensor(X_test, dtype=torch.float32),
        "test_label": torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    }

    model = KAN(width=[4, 8, 1], grid=12, k=3, seed=SEED, device="cpu", ckpt_path=OUT_MODEL_DIR)
    model.fit(dataset, opt="LBFGS", steps=200, lamb=0.0005)

    with torch.no_grad():
        pred_test_norm = model(dataset["test_input"]).numpy().flatten()
        pred_train_norm = model(dataset["train_input"]).numpy().flatten()

    pred_test = denormalize(pred_test_norm, y_lo, y_hi)
    pred_train = denormalize(pred_train_norm, y_lo, y_hi)

    metrics = {
        "model": "KAN",
        "architecture": "width=[4,8,1], grid=12, k=3",
        "train_r2": float(r2_score(y_train_raw, pred_train)),
        "test_r2": float(r2_score(y_test_raw, pred_test)),
        "test_mape_pct": float(mean_absolute_percentage_error(y_test_raw, pred_test) * 100),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test_raw, pred_test))),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "random_state": SEED,
    }

    with open(OUT_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
