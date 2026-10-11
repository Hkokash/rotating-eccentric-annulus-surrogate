# =============================================================================
# Spatial field surrogate, baseline (1 of 2): per-node MLP.
# Per Research Brief (.forge/decisions.log): per-node MLP is a baseline only,
# not a bake-off candidate on equal footing with the Neural Operator - it
# ignores the operator structure of the problem (treats each query point as
# an independent regression sample) but is a necessary sanity floor.
#
# Input: [r_star, epsilon, log10(Re), log10(Ri), x, y] -> output: [temperature, u, v]
# Split is case-level (70/30 on the 35 geometries' converged cases), NOT
# node-level, per decisions.log ("Field-surrogate splits are group-aware by
# case, not by mesh node") - a node-level split would leak the same case's
# geometry/solution into both train and test.
#
# Updated 2026-09-15 for the Re/Ri redesign: DATA/OUT paths point at the new
# field_dataset_filtered_v2_re_ri.parquet; features ra/omega -> log10(re)/
# log10(ri) (both log-transformed, matching the scalar surrogate's
# treatment, since both span 2+ orders of magnitude).
# =============================================================================

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

DATA = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\field_dataset_filtered_v2_re_ri.parquet"
OUT_METRICS = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\results\field_mlp_baseline_metrics_v2.json"
OUT_MODEL = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\models\field_mlp_baseline_v2.pt"
SEED = 42
EPOCHS = 60
BATCH_SIZE = 8192
LR = 1e-3
TARGETS = ["temperature", "u", "v"]

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, n_in=6, n_out=3, n_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def main():
    df = pd.read_parquet(DATA)
    case_ids = df["case_num"].unique()
    train_cases, test_cases = train_test_split(case_ids, test_size=0.30, random_state=SEED)
    print(f"Cases: {len(case_ids)} total, {len(train_cases)} train, {len(test_cases)} test")

    train_df = df[df["case_num"].isin(train_cases)].copy()
    test_df = df[df["case_num"].isin(test_cases)].copy()
    print(f"Rows: {len(train_df)} train, {len(test_df)} test")

    train_df["log_re"] = np.log10(train_df["re"])
    test_df["log_re"] = np.log10(test_df["re"])
    train_df["log_ri"] = np.log10(train_df["ri"])
    test_df["log_ri"] = np.log10(test_df["ri"])
    feat_cols = ["r_star", "epsilon", "log_re", "log_ri", "x", "y"]

    x_lo = train_df[feat_cols].min().values
    x_hi = train_df[feat_cols].max().values
    y_lo = train_df[TARGETS].min().values
    y_hi = train_df[TARGETS].max().values

    def norm(arr, lo, hi):
        return 2 * (arr - lo) / (hi - lo) - 1

    X_train = norm(train_df[feat_cols].values, x_lo, x_hi)
    X_test = norm(test_df[feat_cols].values, x_lo, x_hi)
    Y_train = norm(train_df[TARGETS].values, y_lo, y_hi)
    Y_test = norm(test_df[TARGETS].values, y_lo, y_hi)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n = X_train_t.shape[0]

    for epoch in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            pred = model(X_train_t[idx])
            loss = nn.functional.mse_loss(pred, Y_train_t[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch}: train_mse={total_loss/n:.6f}")

    model.eval()
    with torch.no_grad():
        pred_test_norm = model(X_test_t).cpu().numpy()
        pred_train_norm = model(X_train_t).cpu().numpy()

    def denorm(arr, lo, hi):
        return (arr + 1) / 2 * (hi - lo) + lo

    pred_test = denorm(pred_test_norm, y_lo, y_hi)
    pred_train = denorm(pred_train_norm, y_lo, y_hi)

    metrics = {"model": "per-node MLP baseline", "n_train_cases": len(train_cases),
               "n_test_cases": len(test_cases), "n_train_rows": len(train_df),
               "n_test_rows": len(test_df), "random_state": SEED, "per_target": {}}
    for i, tgt in enumerate(TARGETS):
        y_true = test_df[tgt].values
        y_pred = pred_test[:, i]
        metrics["per_target"][tgt] = {
            "test_r2": float(r2_score(y_true, y_pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }
        y_true_tr = train_df[tgt].values
        y_pred_tr = pred_train[:, i]
        metrics["per_target"][tgt]["train_r2"] = float(r2_score(y_true_tr, y_pred_tr))

    torch.save(model.state_dict(), OUT_MODEL)
    with open(OUT_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
