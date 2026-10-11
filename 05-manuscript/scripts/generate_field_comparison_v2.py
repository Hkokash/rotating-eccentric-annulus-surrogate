# =============================================================================
# Rebuilt CFD-vs-surrogate field comparison, per explicit user feedback:
#   - multiple held-out GEOMETRIES (not one), to show real cross-geometry
#     generalization, which is the actual point of the field-surrogate claim
#   - three separate figures: temperature, velocity magnitude, and signed
#     difference maps (not folded into one dense 5-column figure)
#   - professional colormaps: "coolwarm" (hot/cold) for temperature, "jet"
#     for velocity magnitude, a diverging map with overlaid contour lines
#     for the signed difference
#   - one shared colorbar per figure (temperature is physically bounded
#     [0,1] by the boundary conditions, so a single shared scale is exact;
#     velocity uses a common scale sized to the max across the shown cases)
#
# Updated 2026-09-15 for the Re/Ri redesign: DATA/MODEL_PATH point at the
# new field parquet/model; features ra/omega -> log10(re)/log10(ri).
# GEOMS_SHOW unchanged (9, 14, 22, 30) -- verified all four are still in the
# field-surrogate's held-out test split under the new dataset
# (test_cases = {9,10,13,14,16,17,20,22,25,27,30} at random_state=42).
# =============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from sklearn.model_selection import train_test_split

DATA = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\field_dataset_filtered_v2_re_ri.parquet"
MODEL_PATH = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\models\field_mlp_baseline_v2.pt"
OUT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\05-manuscript\figures"
SEED = 42
TARGETS = ["temperature", "u", "v"]
FEAT_COLS = ["r_star", "epsilon", "log_re", "log_ri", "x", "y"]

RE_SHOW = 125
RI_SHOW = 1.0
GEOMS_SHOW = [9, 14, 22, 30]  # (R*,eps) = (0.2,0.0),(0.2,0.2),(0.4,0.0),(0.5,0.06) - all held out


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


def norm(arr, lo, hi):
    return 2 * (arr - lo) / (hi - lo) - 1


def denorm(arr, lo, hi):
    return (arr + 1) / 2 * (hi - lo) + lo


def masked_triangulation(x, y, r_i, e):
    tri = mtri.Triangulation(x, y)
    cx = x[tri.triangles].mean(axis=1)
    cy = y[tri.triangles].mean(axis=1)
    inside_hole = (cx**2 + (cy - (-e))**2) < (0.97 * r_i) ** 2
    tri.set_mask(inside_hole)
    return tri


def load_model_and_norm():
    df = pd.read_parquet(DATA)
    case_ids = df["case_num"].unique()
    train_cases, test_cases = train_test_split(case_ids, test_size=0.30, random_state=SEED)
    train_df = df[df["case_num"].isin(train_cases)].copy()
    train_df["log_re"] = np.log10(train_df["re"])
    train_df["log_ri"] = np.log10(train_df["ri"])
    x_lo = train_df[FEAT_COLS].min().values
    x_hi = train_df[FEAT_COLS].max().values
    y_lo = train_df[TARGETS].min().values
    y_hi = train_df[TARGETS].max().values

    model = MLP()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return df, model, x_lo, x_hi, y_lo, y_hi


def get_case_fields(df, model, x_lo, x_hi, y_lo, y_hi, case_num, re, ri):
    sub = df[(df["case_num"] == case_num) & (abs(df["re"] - re) < 1e-6) & (abs(df["ri"] - ri) < 1e-4)].copy()
    sub["log_re"] = np.log10(sub["re"])
    sub["log_ri"] = np.log10(sub["ri"])
    X = norm(sub[FEAT_COLS].values, x_lo, x_hi)
    with torch.no_grad():
        pred_norm = model(torch.tensor(X, dtype=torch.float32)).numpy()
    pred = denorm(pred_norm, y_lo, y_hi)
    x, y = sub["x"].values, sub["y"].values
    r_i = sub["r_star"].iloc[0] * 0.5
    e_val = sub["epsilon"].iloc[0]
    tri = masked_triangulation(x, y, r_i, e_val)
    return dict(
        x=x, y=y, tri=tri, r_star=sub["r_star"].iloc[0], epsilon=e_val,
        T_true=sub["temperature"].values, T_pred=pred[:, 0],
        u_true=sub["u"].values, v_true=sub["v"].values,
        u_pred=pred[:, 1], v_pred=pred[:, 2],
    )


def fig_temperature(cases):
    fig, axes = plt.subplots(len(cases), 2, figsize=(7, 3.1 * len(cases)))
    for row, c in enumerate(cases):
        for col, (field, label) in enumerate([("T_true", "CFD"), ("T_pred", "Surrogate")]):
            ax = axes[row, col]
            cf = ax.tricontourf(c["tri"], c[field], levels=20, cmap="coolwarm", vmin=0, vmax=1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"$R^*$={c['r_star']:.1f}\n$\\varepsilon$={c['epsilon']:.2f}", fontsize=14)
    fig.subplots_adjust(right=0.85, wspace=0.05, hspace=0.15)
    cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, cax=cax, label="Dimensionless temperature $\\theta$")
    fig.savefig(f"{OUT}/fig_field_temperature.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_velocity(cases):
    speeds_true = [np.sqrt(c["u_true"]**2 + c["v_true"]**2) for c in cases]
    speeds_pred = [np.sqrt(c["u_pred"]**2 + c["v_pred"]**2) for c in cases]
    vmax = max(max(s.max() for s in speeds_true), max(s.max() for s in speeds_pred))

    fig, axes = plt.subplots(len(cases), 2, figsize=(7, 3.1 * len(cases)))
    for row, c in enumerate(cases):
        for col, (speed, label) in enumerate([(speeds_true[row], "CFD"), (speeds_pred[row], "Surrogate")]):
            ax = axes[row, col]
            cf = ax.tricontourf(c["tri"], speed, levels=20, cmap="jet", vmin=0, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"$R^*$={c['r_star']:.1f}\n$\\varepsilon$={c['epsilon']:.2f}", fontsize=14)
    fig.subplots_adjust(right=0.85, wspace=0.05, hspace=0.15)
    cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, cax=cax, label="Velocity magnitude $|\\mathbf{u}|$")
    fig.savefig(f"{OUT}/fig_field_velocity.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_difference(cases):
    diffs_T = [c["T_pred"] - c["T_true"] for c in cases]
    speeds_true = [np.sqrt(c["u_true"]**2 + c["v_true"]**2) for c in cases]
    speeds_pred = [np.sqrt(c["u_pred"]**2 + c["v_pred"]**2) for c in cases]
    diffs_S = [sp - st for sp, st in zip(speeds_pred, speeds_true)]

    # Color range clipped to the 98th percentile of |difference| (pooled across
    # all shown geometries) rather than the raw max: a handful of near-boundary
    # outlier nodes otherwise stretch the diverging colormap so far that the
    # bulk of the field, where the surrogate is actually close, all renders as
    # a single pale color. Values beyond this range simply saturate at the
    # colorbar's endpoint color rather than being lost.
    vmax_T = np.percentile(np.abs(np.concatenate(diffs_T)), 98)
    vmax_S = np.percentile(np.abs(np.concatenate(diffs_S)), 98)

    levels_T = np.linspace(-vmax_T, vmax_T, 21)
    levels_S = np.linspace(-vmax_S, vmax_S, 21)

    # Each diff column gets its own colorbar immediately beside it (not both
    # colorbars clustered together off to the right of both columns), so Delta-
    # theta's scale sits next to the temperature-diff column and Delta-|u|'s
    # scale sits next to the velocity-diff column, with no risk of the two
    # overlapping or being mismatched to the wrong column. Attaching via
    # ax=<list of axes> (rather than a separately-sized GridSpec column) lets
    # matplotlib size the colorbar to the panels' actual rendered height,
    # which set_aspect("equal") shrinks below the nominal row height.
    fig, axes = plt.subplots(len(cases), 2, figsize=(7.6, 3.1 * len(cases)))
    for row, c in enumerate(cases):
        ax = axes[row, 0]
        cf = ax.tricontourf(c["tri"], diffs_T[row], levels=levels_T, cmap="RdBu_r", extend="both")
        ax.tricontour(c["tri"], diffs_T[row], levels=6, colors="black", linewidths=0.3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f"$R^*$={c['r_star']:.1f}\n$\\varepsilon$={c['epsilon']:.2f}", fontsize=14)
        if row == 0:
            ax.set_title("$\\Delta\\theta$ = surrogate $-$ CFD")

        ax = axes[row, 1]
        cf2 = ax.tricontourf(c["tri"], diffs_S[row], levels=levels_S, cmap="RdBu_r", extend="both")
        ax.tricontour(c["tri"], diffs_S[row], levels=6, colors="black", linewidths=0.3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title("$\\Delta|\\mathbf{u}|$ = surrogate $-$ CFD")

    # Wide wspace: a clear visual gap between the two picture-columns (each
    # column plus its own colorbar reads as one group), not a dense block of
    # panels that could be mistaken for eight unrelated subfigures.
    fig.subplots_adjust(wspace=0.85, hspace=0.15)
    fig.canvas.draw()  # finalize axes positions (set_aspect("equal") shrinks
                        # each panel below its nominal grid slot; the colorbar
                        # must be sized from the real, post-shrink positions)

    # Each colorbar is placed from the exact top edge of the first row's panel
    # to the exact bottom edge of the last row's panel, so it spans the full
    # column height edge-to-edge rather than a centered, shrunk approximation
    # that looks disconnected from the first/last row.
    # Width matched (in physical inches, not figure-fraction) to the single
    # colorbar in fig_velocity()/fig_temperature() (0.03 fraction at figsize
    # width 7in = 0.21in), so Figure 12's bars read the same thickness as
    # Figure 11's.
    pos_top0, pos_bot0 = axes[0, 0].get_position(), axes[-1, 0].get_position()
    cax1 = fig.add_axes([pos_top0.x1 + 0.02, pos_bot0.y0, 0.028, pos_top0.y1 - pos_bot0.y0])
    cb1 = fig.colorbar(cf, cax=cax1, extendrect=True)
    cb1.ax.set_ylabel("$\\Delta\\theta$", fontsize=12)

    pos_top1, pos_bot1 = axes[0, 1].get_position(), axes[-1, 1].get_position()
    cax2 = fig.add_axes([pos_top1.x1 + 0.02, pos_bot1.y0, 0.028, pos_top1.y1 - pos_bot1.y0])
    cb2 = fig.colorbar(cf2, cax=cax2, extendrect=True)
    cb2.ax.set_ylabel("$\\Delta|\\mathbf{u}|$", fontsize=12)
    fig.savefig(f"{OUT}/fig_field_difference.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df, model, x_lo, x_hi, y_lo, y_hi = load_model_and_norm()
    cases = [get_case_fields(df, model, x_lo, x_hi, y_lo, y_hi, g, RE_SHOW, RI_SHOW) for g in GEOMS_SHOW]
    fig_temperature(cases)
    fig_velocity(cases)
    fig_difference(cases)
    print("Wrote fig_field_temperature.png, fig_field_velocity.png, fig_field_difference.png")
