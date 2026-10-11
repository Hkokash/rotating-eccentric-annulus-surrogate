# =============================================================================
# Out-of-range field comparison: real CFD (never part of the training grid,
# Re=500, Ri=0.2 at case_10, which is both a scalar blind-test point -- the
# one with the LARGEST scalar blind-test error, 23.33% -- and held out from
# the field surrogate's own training split) vs. the trained per-node MLP
# surrogate's prediction at the same points. Same style as
# generate_field_comparison_v2.py for visual consistency.
#
# Updated 2026-09-15 for the Re/Ri redesign; point changed from
# case_09/Ra=3.5e4/omega=9 to case_10/Re=500/Ri=0.2 (see
# run_oor_field_case_v2.py for the point-selection rationale).
# =============================================================================

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_field_comparison_v2 import MLP, norm, denorm, masked_triangulation, load_model_and_norm, FEAT_COLS

OOR_CSV = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\05-manuscript\data\oor_case10_re500_ri0.2.csv"
OUT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\05-manuscript\figures"
R_STAR, EPSILON, RE, RI = 0.2, 0.04, 500.0, 0.2


def main():
    _, model, x_lo, x_hi, y_lo, y_hi = load_model_and_norm()

    field = pd.read_csv(OOR_CSV)
    field.columns = [c.strip() for c in field.columns]
    x = field["x-coordinate"].values
    y = field["y-coordinate"].values
    T_true = field["temperature"].values
    u_true = field["x-velocity"].values
    v_true = field["y-velocity"].values
    speed_true = np.sqrt(u_true**2 + v_true**2)

    n = len(x)
    feat = np.column_stack([
        np.full(n, R_STAR), np.full(n, EPSILON), np.full(n, np.log10(RE)), np.full(n, np.log10(RI)), x, y,
    ])
    X = norm(feat, x_lo, x_hi)
    with torch.no_grad():
        pred_norm = model(torch.tensor(X, dtype=torch.float32)).numpy()
    pred = denorm(pred_norm, y_lo, y_hi)
    T_pred, u_pred, v_pred = pred[:, 0], pred[:, 1], pred[:, 2]
    speed_pred = np.sqrt(u_pred**2 + v_pred**2)

    tri = masked_triangulation(x, y, R_STAR * 0.5, EPSILON)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.3))

    vmax_s = max(speed_true.max(), speed_pred.max())
    diff_T = T_pred - T_true
    diff_S = speed_pred - speed_true
    # Color range clipped to the 98th percentile of |difference|, not the raw
    # max: a handful of near-boundary outlier nodes otherwise stretch the
    # diverging colormap so far that the bulk of the field, where the
    # surrogate is actually close, all renders as a single pale color.
    vmax_dT = np.percentile(np.abs(diff_T), 98)
    vmax_dS = np.percentile(np.abs(diff_S), 98)
    levels_dT = np.linspace(-vmax_dT, vmax_dT, 21)
    levels_dS = np.linspace(-vmax_dS, vmax_dS, 21)

    panels = [
        (axes[0, 0], T_true, "coolwarm", 20, False, "CFD temperature"),
        (axes[0, 1], T_pred, "coolwarm", 20, False, "Surrogate temperature"),
        (axes[0, 2], diff_T, "RdBu_r", levels_dT, True, "$\\Delta\\theta$ = surrogate $-$ CFD"),
        (axes[1, 0], speed_true, "jet", 20, False, "CFD speed $|\\mathbf{u}|$"),
        (axes[1, 1], speed_pred, "jet", 20, False, "Surrogate speed $|\\mathbf{u}|$"),
        (axes[1, 2], diff_S, "RdBu_r", levels_dS, True, "$\\Delta|\\mathbf{u}|$ = surrogate $-$ CFD"),
    ]
    for ax, field_vals, cmap, levels, is_diff, title in panels:
        kwargs = dict(extend="both") if is_diff else dict(vmin=0, vmax=(1 if cmap == "coolwarm" else vmax_s))
        cf = ax.tricontourf(tri, field_vals, levels=levels, cmap=cmap, **kwargs)
        if is_diff:
            ax.tricontour(tri, field_vals, levels=6, colors="black", linewidths=0.3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10.5)
        # extendrect=True: flat rectangular end segments for the diff panels'
        # out-of-range values instead of pointed triangular arrows.
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, extendrect=is_diff)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_oor_field.png", dpi=170, bbox_inches="tight")
    print(f"Wrote {OUT}/fig_oor_field.png")
    print(f"Mean |dT|={np.abs(diff_T).mean():.4f}  Mean |dSpeed|={np.abs(diff_S).mean():.4f}")


if __name__ == "__main__":
    main()
