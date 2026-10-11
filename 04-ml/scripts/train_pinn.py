# =============================================================================
# Scalar Nu surrogate, candidate 3 of 3: Physics-Informed Neural Network (PINN).
# Same train/test split (70/30, random_state=42), same log10 transform and
# [-1,1] normalization as the KAN candidate, for a fair three-way comparison.
#
# Physics constraint, updated 2026-09-09 for the Re/Ri redesign: with Ra now a
# derived quantity (Ra = Gr*Pr = 0.71*Ri*Re^2) rather than a direct feature,
# the old single "dNu/dRa>=0" constraint doesn't translate simply to either
# Re or Ri alone -- moving Re at fixed Ri changes Ra too (Ra ~ Re^2 at fixed
# Ri), so a naive carry-over would conflate two different physical effects
# (buoyancy strength vs. rotation). Checked directly against the dataset
# before assuming anything: at fixed Ri, Nu increases monotonically with Re
# (12 points, case 1, Ri=0.01/1/10, all strictly increasing); at fixed Re, Nu
# increases monotonically with Ri (13 points, Re=10/1000, all strictly
# increasing). Both are enforced as soft penalties on d(Nu)/d(Re)>=0 and
# d(Nu)/d(Ri)>=0 computed via autograd at training points (log10-transformed,
# monotonic transform, so the derivative sign is unaffected). No other
# physics relation is assumed or fabricated.
# =============================================================================

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

DATA = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\scalar_dataset_filtered_v2_re_ri.csv"
OUT_METRICS = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\results\pinn_scalar_metrics_v2.json"
OUT_MODEL = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\models\pinn_scalar_v2.pt"
FEATURES = ["r_star", "epsilon", "re", "ri"]
RE_IDX = FEATURES.index("re")
RI_IDX = FEATURES.index("ri")
TARGET = "nu"
SEED = 42
LAMBDA_PHYS = 0.1
EPOCHS = 3000
LR = 1e-3

torch.manual_seed(SEED)
np.random.seed(SEED)


def normalize(arr, lo, hi):
    return 2 * (arr - lo) / (hi - lo) - 1


def denormalize(arr, lo, hi):
    return (arr + 1) / 2 * (hi - lo) + lo


class PINN(nn.Module):
    def __init__(self, n_in=4, n_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    df = pd.read_csv(DATA)
    X_raw = df[FEATURES].copy()
    X_raw["re"] = np.log10(X_raw["re"])
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

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        pred = model(X_train_t)
        data_loss = nn.functional.mse_loss(pred, y_train_t)

        x_phys = X_train_t.clone().detach().requires_grad_(True)
        nu_phys = model(x_phys)
        grad_nu = torch.autograd.grad(
            nu_phys, x_phys, grad_outputs=torch.ones_like(nu_phys), create_graph=True
        )[0]
        dnu_dre = grad_nu[:, RE_IDX]
        dnu_dri = grad_nu[:, RI_IDX]
        phys_loss = torch.relu(-dnu_dre).mean() + torch.relu(-dnu_dri).mean()

        loss = data_loss + LAMBDA_PHYS * phys_loss
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch}: data_loss={data_loss.item():.5f} phys_loss={phys_loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        pred_test_norm = model(X_test_t).numpy().flatten()
        pred_train_norm = model(X_train_t).numpy().flatten()

    pred_test = denormalize(pred_test_norm, y_lo, y_hi)
    pred_train = denormalize(pred_train_norm, y_lo, y_hi)

    metrics = {
        "model": "PINN",
        "architecture": "MLP [4,64,64,64,1], tanh, physics penalty on d(Nu)/d(Re)>=0 and d(Nu)/d(Ri)>=0, lambda=0.1",
        "epochs": EPOCHS,
        "train_r2": float(r2_score(y_train_raw, pred_train)),
        "test_r2": float(r2_score(y_test_raw, pred_test)),
        "test_mape_pct": float(mean_absolute_percentage_error(y_test_raw, pred_test) * 100),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test_raw, pred_test))),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "random_state": SEED,
    }

    torch.save(model.state_dict(), OUT_MODEL)
    with open(OUT_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
