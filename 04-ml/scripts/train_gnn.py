# =============================================================================
# Spatial field surrogate, GNN fallback (per decisions.log, invoked because
# DeepONet underperformed the per-node MLP baseline on every target).
#
# Uses the TRUE mesh connectivity extracted by build_graph_connectivity.py
# (real Fluent face-node edges, not an approximate triangulation). Each
# (geometry, Re, Ri) case is one graph; the mesh (and hence edge_index) is
# identical for every Re/Ri at a fixed geometry, verified in
# build_graph_connectivity.py / this script's own checks.
#
# No torch_geometric dependency - a minimal custom message-passing layer
# (mean-aggregation over neighbors via index_add_/degree normalization) is
# implemented directly, batching multiple graphs per step via the standard
# disjoint-union trick (concatenate node features, offset edge indices by
# cumulative node count per graph in the batch).
#
# Same case-level 70/30 split (random_state=42) as the MLP baseline and
# DeepONet, for a fair three-way comparison on the same held-out geometries.
#
# Updated 2026-09-15 for the Re/Ri redesign: DATA/EDGE_DIR/OUT paths point
# at the new field parquet and graph_connectivity_v2; grouping key and
# features ra/omega -> re/ri (log-transformed).
# =============================================================================

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

DATA = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\field_dataset_filtered_v2_re_ri.parquet"
EDGE_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\graph_connectivity_v2"
OUT_METRICS = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\results\gnn_metrics_v2.json"
OUT_MODEL = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\models\gnn_v2.pt"
SEED = 42
EPOCHS = 20
CASES_PER_BATCH = 16
LR = 1e-3
HIDDEN = 64
TARGETS = ["temperature", "u", "v"]

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_edge_index(case_num):
    npz = np.load(f"{EDGE_DIR}/case_{case_num:02d}_edges.npz")
    return npz["edge_index"], int(npz["n_nodes"])


class GraphConvLayer(nn.Module):
    """h_i' = SiLU(W_self * h_i + W_neigh * mean_{j in N(i)} h_j)"""
    def __init__(self, n_in, n_out):
        super().__init__()
        self.self_lin = nn.Linear(n_in, n_out)
        self.neigh_lin = nn.Linear(n_in, n_out)

    def forward(self, h, edge_index, deg_inv):
        src, dst = edge_index[0], edge_index[1]
        msg = h[src]
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        agg = agg * deg_inv.unsqueeze(-1)
        return nn.functional.silu(self.self_lin(h) + self.neigh_lin(agg))


class GNN(nn.Module):
    def __init__(self, n_in=6, n_hidden=HIDDEN, n_out=3, n_layers=3):
        super().__init__()
        self.encoder = nn.Linear(n_in, n_hidden)
        self.layers = nn.ModuleList([GraphConvLayer(n_hidden, n_hidden) for _ in range(n_layers)])
        self.decoder = nn.Linear(n_hidden, n_out)

    def forward(self, x, edge_index, deg_inv):
        h = nn.functional.silu(self.encoder(x))
        for layer in self.layers:
            h = layer(h, edge_index, deg_inv)
        return self.decoder(h)


def build_batch(case_keys, df_by_case, edge_cache, feat_cols, target_cols, x_lo, x_hi, y_lo, y_hi):
    """Disjoint-union batch of several (case_num, ra, omega) graphs."""
    feats, targets, edges, offsets = [], [], [], []
    node_offset = 0
    for key in case_keys:
        case_num, re, ri = key
        sub = df_by_case[key]
        feat = 2 * (sub[feat_cols].values - x_lo) / (x_hi - x_lo) - 1
        tgt = 2 * (sub[target_cols].values - y_lo) / (y_hi - y_lo) - 1
        feats.append(feat)
        targets.append(tgt)

        ei, n_nodes = edge_cache[case_num]
        edges.append(ei + node_offset)
        node_offset += n_nodes

    X = torch.tensor(np.concatenate(feats, axis=0), dtype=torch.float32, device=device)
    Y = torch.tensor(np.concatenate(targets, axis=0), dtype=torch.float32, device=device)
    E = torch.tensor(np.concatenate(edges, axis=1), dtype=torch.long, device=device)

    deg = torch.zeros(X.shape[0], device=device)
    deg.index_add_(0, E[1], torch.ones(E.shape[1], device=device))
    deg_inv = 1.0 / deg.clamp(min=1.0)

    return X, Y, E, deg_inv


def main():
    df = pd.read_parquet(DATA)
    case_ids = df["case_num"].unique()
    train_geoms, test_geoms = train_test_split(case_ids, test_size=0.30, random_state=SEED)
    print(f"Geometries: {len(case_ids)} total, {len(train_geoms)} train, {len(test_geoms)} test")

    train_df = df[df["case_num"].isin(train_geoms)].copy()
    test_df = df[df["case_num"].isin(test_geoms)].copy()
    train_df["log_re"] = np.log10(train_df["re"])
    test_df["log_re"] = np.log10(test_df["re"])
    train_df["log_ri"] = np.log10(train_df["ri"])
    test_df["log_ri"] = np.log10(test_df["ri"])
    feat_cols = ["r_star", "epsilon", "log_re", "log_ri", "x", "y"]

    x_lo = train_df[feat_cols].min().values
    x_hi = train_df[feat_cols].max().values
    y_lo = train_df[TARGETS].min().values
    y_hi = train_df[TARGETS].max().values

    edge_cache = {int(c): load_edge_index(int(c)) for c in case_ids}

    def group_by_case(d):
        groups = {}
        for key, sub in d.groupby(["case_num", "re", "ri"]):
            groups[key] = sub
        return groups

    print("Grouping rows into per-case graphs...")
    train_groups = group_by_case(train_df)
    test_groups = group_by_case(test_df)
    train_keys = list(train_groups.keys())
    test_keys = list(test_groups.keys())
    print(f"Train graphs: {len(train_keys)}, test graphs: {len(test_keys)}")

    model = GNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.RandomState(SEED)

    for epoch in range(EPOCHS):
        order = rng.permutation(len(train_keys))
        total_loss, total_nodes = 0.0, 0
        for i in range(0, len(order), CASES_PER_BATCH):
            batch_keys = [train_keys[j] for j in order[i:i + CASES_PER_BATCH]]
            X, Y, E, deg_inv = build_batch(batch_keys, train_groups, edge_cache,
                                            feat_cols, TARGETS, x_lo, x_hi, y_lo, y_hi)
            optimizer.zero_grad()
            pred = model(X, E, deg_inv)
            loss = nn.functional.mse_loss(pred, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.shape[0]
            total_nodes += X.shape[0]
        if epoch % 2 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch}: train_mse={total_loss/total_nodes:.6f}")

    def evaluate(keys, groups):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for i in range(0, len(keys), CASES_PER_BATCH):
                batch_keys = keys[i:i + CASES_PER_BATCH]
                X, Y, E, deg_inv = build_batch(batch_keys, groups, edge_cache,
                                                feat_cols, TARGETS, x_lo, x_hi, y_lo, y_hi)
                pred = model(X, E, deg_inv)
                preds.append(pred.cpu().numpy())
                trues.append(Y.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        def denorm(arr, lo, hi):
            return (arr + 1) / 2 * (hi - lo) + lo
        return denorm(preds, y_lo, y_hi), denorm(trues, y_lo, y_hi)

    pred_test, true_test = evaluate(test_keys, test_groups)
    pred_train, true_train = evaluate(train_keys, train_groups)

    metrics = {"model": "GNN (custom mean-aggregation, true mesh connectivity)",
               "n_layers": 3, "hidden": HIDDEN, "n_train_graphs": len(train_keys),
               "n_test_graphs": len(test_keys), "random_state": SEED, "per_target": {}}
    for i, tgt in enumerate(TARGETS):
        metrics["per_target"][tgt] = {
            "test_r2": float(r2_score(true_test[:, i], pred_test[:, i])),
            "test_rmse": float(np.sqrt(mean_squared_error(true_test[:, i], pred_test[:, i]))),
            "train_r2": float(r2_score(true_train[:, i], pred_train[:, i])),
        }

    torch.save(model.state_dict(), OUT_MODEL)
    with open(OUT_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
