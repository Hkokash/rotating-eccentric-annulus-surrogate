# =============================================================================
# Spatial field surrogate, primary Neural Operator candidate: DeepONet.
# Chosen over FNO per decisions.log ("Neural Operator (FNO/DeepONet) primary")
# because FNO requires a regular grid (FFT-based) - this dataset is scattered,
# unstructured mesh nodes on a domain whose shape itself changes per case
# (varying r_star/epsilon), which DeepONet handles natively: the branch net
# encodes the case's parameters (the "input function"), the trunk net encodes
# an arbitrary query coordinate (x,y), independent of mesh connectivity.
#
# Architecture: branch(r_star, epsilon, log10(Re), log10(Ri)) -> R^(3*p) (3
# output channels: temperature, u, v, each gets its own p-dim coefficient
# block). trunk(x, y) -> R^p, shared spatial basis across channels.
# Prediction for channel c: dot(branch_c, trunk) + bias_c.
#
# Same case-level 70/30 split (random_state=42) as the MLP baseline, for a
# fair, direct comparison on the same held-out cases.
#
# Updated 2026-09-15 for the Re/Ri redesign: DATA/OUT paths point at the new
# field parquet; branch inputs ra/omega -> log10(re)/log10(ri).
# =============================================================================

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

DATA = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\field_dataset_filtered_v2_re_ri.parquet"
OUT_METRICS = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\results\deeponet_metrics_v2.json"
OUT_MODEL = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\models\deeponet_v2.pt"
SEED = 42
EPOCHS = 60
BATCH_SIZE = 8192
LR = 1e-3
P = 64
TARGETS = ["temperature", "u", "v"]
N_TARGETS = len(TARGETS)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Branch(nn.Module):
    def __init__(self, n_in=4, p=P, n_out=N_TARGETS, n_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(n_hidden, p * n_out),
        )
        self.p = p
        self.n_out = n_out

    def forward(self, x):
        return self.net(x).view(-1, self.n_out, self.p)


class Trunk(nn.Module):
    def __init__(self, n_in=2, p=P, n_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, p), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    def __init__(self, n_branch_in=4, n_trunk_in=2, p=P, n_out=N_TARGETS):
        super().__init__()
        self.branch = Branch(n_branch_in, p, n_out)
        self.trunk = Trunk(n_trunk_in, p)
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, params, coords):
        b = self.branch(params)          # (B, n_out, p)
        t = self.trunk(coords)           # (B, p)
        out = (b * t.unsqueeze(1)).sum(-1) + self.bias  # (B, n_out)
        return out


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
    branch_cols = ["r_star", "epsilon", "log_re", "log_ri"]
    trunk_cols = ["x", "y"]

    b_lo = train_df[branch_cols].min().values
    b_hi = train_df[branch_cols].max().values
    t_lo = train_df[trunk_cols].min().values
    t_hi = train_df[trunk_cols].max().values
    y_lo = train_df[TARGETS].min().values
    y_hi = train_df[TARGETS].max().values

    def norm(arr, lo, hi):
        return 2 * (arr - lo) / (hi - lo) - 1

    B_train = norm(train_df[branch_cols].values, b_lo, b_hi)
    B_test = norm(test_df[branch_cols].values, b_lo, b_hi)
    T_train = norm(train_df[trunk_cols].values, t_lo, t_hi)
    T_test = norm(test_df[trunk_cols].values, t_lo, t_hi)
    Y_train = norm(train_df[TARGETS].values, y_lo, y_hi)

    B_train_t = torch.tensor(B_train, dtype=torch.float32, device=device)
    T_train_t = torch.tensor(T_train, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    B_test_t = torch.tensor(B_test, dtype=torch.float32, device=device)
    T_test_t = torch.tensor(T_test, dtype=torch.float32, device=device)

    model = DeepONet(n_branch_in=len(branch_cols), n_trunk_in=len(trunk_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    n = B_train_t.shape[0]

    for epoch in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            pred = model(B_train_t[idx], T_train_t[idx])
            loss = nn.functional.mse_loss(pred, Y_train_t[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch}: train_mse={total_loss/n:.6f}")

    def batched_predict(model, B_t, T_t, batch_size=BATCH_SIZE):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, B_t.shape[0], batch_size):
                out = model(B_t[i:i + batch_size], T_t[i:i + batch_size])
                preds.append(out.cpu().numpy())
        return np.concatenate(preds, axis=0)

    pred_test_norm = batched_predict(model, B_test_t, T_test_t)
    pred_train_norm = batched_predict(model, B_train_t, T_train_t)
    torch.cuda.empty_cache()

    def denorm(arr, lo, hi):
        return (arr + 1) / 2 * (hi - lo) + lo

    pred_test = denorm(pred_test_norm, y_lo, y_hi)
    pred_train = denorm(pred_train_norm, y_lo, y_hi)

    metrics = {"model": "DeepONet", "p": P, "n_train_cases": len(train_cases),
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
