# =============================================================================
# Uncertainty quantification for the winning spatial field surrogate
# (per-node MLP). True deep ensemble: N_MEMBERS independently-seeded MLPs
# (same architecture as train_field_mlp_baseline.py), each with a different
# random init and training-batch shuffling order. Mean = prediction,
# std across members = epistemic uncertainty estimate. Same case-level 70/30
# split (random_state=42) as every other field-surrogate script.
#
# Updated 2026-09-15 for the Re/Ri redesign: DATA/OUT paths point at the
# new field parquet; features ra/omega -> log10(re)/log10(ri).
# =============================================================================

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

DATA = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\field_dataset_filtered_v2_re_ri.parquet"
OUT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\results\uq_field_mlp_ensemble_v2.json"
SEED = 42
N_MEMBERS = 5
EPOCHS = 40
BATCH_SIZE = 8192
LR = 1e-3
TARGETS = ["temperature", "u", "v"]
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


def train_one(seed, X_train_t, Y_train_t):
    torch.manual_seed(seed)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n = X_train_t.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n, device=device, generator=gen)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            pred = model(X_train_t[idx])
            loss = nn.functional.mse_loss(pred, Y_train_t[idx])
            loss.backward()
            optimizer.step()
    return model


def main():
    df = pd.read_parquet(DATA)
    case_ids = df["case_num"].unique()
    train_cases, test_cases = train_test_split(case_ids, test_size=0.30, random_state=SEED)

    train_df = df[df["case_num"].isin(train_cases)].copy()
    test_df = df[df["case_num"].isin(test_cases)].copy()
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

    def denorm(arr, lo, hi):
        return (arr + 1) / 2 * (hi - lo) + lo

    X_train_t = torch.tensor(norm(train_df[feat_cols].values, x_lo, x_hi), dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(norm(train_df[TARGETS].values, y_lo, y_hi), dtype=torch.float32, device=device)
    X_test_t = torch.tensor(norm(test_df[feat_cols].values, x_lo, x_hi), dtype=torch.float32, device=device)

    member_preds = []
    for m in range(N_MEMBERS):
        print(f"Training ensemble member {m+1}/{N_MEMBERS} (seed={SEED+m})...")
        model = train_one(SEED + m, X_train_t, Y_train_t)
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, X_test_t.shape[0], BATCH_SIZE):
                preds.append(model(X_test_t[i:i + BATCH_SIZE]).cpu().numpy())
        member_preds.append(denorm(np.concatenate(preds, axis=0), y_lo, y_hi))
        del model
        torch.cuda.empty_cache()

    member_preds = np.stack(member_preds, axis=0)  # (N_MEMBERS, n_test_rows, 3)
    pred_mean = member_preds.mean(axis=0)
    pred_std = member_preds.std(axis=0)
    y_true = test_df[TARGETS].values

    z95 = 1.96
    result = {"method": f"Deep ensemble of {N_MEMBERS} independently-seeded per-node MLPs",
              "n_test_rows": len(test_df), "per_target": {}}
    for i, tgt in enumerate(TARGETS):
        abs_err = np.abs(y_true[:, i] - pred_mean[:, i])
        within_95 = np.mean(abs_err <= z95 * pred_std[:, i])
        result["per_target"][tgt] = {
            "ensemble_mean_r2": float(r2_score(y_true[:, i], pred_mean[:, i])),
            "mean_predictive_std": float(pred_std[:, i].mean()),
            "fraction_within_95pct_interval": float(within_95),
            "target_fraction": 0.95,
        }

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
