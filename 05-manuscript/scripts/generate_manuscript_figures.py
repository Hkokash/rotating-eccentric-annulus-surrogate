# =============================================================================
# Generate the manuscript's summary figures from already-computed result
# JSONs (no new training/computation - pure plotting of real logged numbers).
# =============================================================================

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ML = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml"
OUT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\05-manuscript\figures"

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# Okabe-Ito colorblind-safe qualitative palette (Okabe & Ito, 2008), the
# standard recommendation for scientific figures in Nature/Science-family
# journals - distinguishable in both full color and grayscale/colorblind
# simulation.
OKABE_ITO = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "sky": "#56B4E9", "purple": "#CC79A7",
}


def load(rel):
    with open(f"{ML}/results/{rel}") as f:
        return json.load(f)


def fig_scalar_bakeoff():
    xgb = load("xgboost_scalar_metrics_v2.json")
    pinn = load("pinn_scalar_metrics_v2.json")
    kan = load("kan_scalar_metrics_v2.json")
    names = ["XGBoost", "PINN", "KAN"]
    r2 = [xgb["test_r2"], pinn["test_r2"], kan["test_r2"]]
    mape = [xgb["test_mape_pct"], pinn["test_mape_pct"], kan["test_mape_pct"]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    colors = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"]]
    axes[0].bar(names, r2, color=colors, edgecolor="black")
    axes[0].set_ylabel("Test $R^2$ (70/30 split)")
    axes[0].set_ylim(0.99, 1.0005)
    for i, v in enumerate(r2):
        axes[0].text(i, v + 0.0002, f"{v:.4f}", ha="center", fontsize=9)

    axes[1].bar(names, mape, color=colors, edgecolor="black")
    axes[1].set_ylabel("Test MAPE (%)")
    for i, v in enumerate(mape):
        axes[1].text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_scalar_bakeoff.png", bbox_inches="tight")
    plt.close(fig)


def fig_field_bakeoff():
    mlp = load("field_mlp_baseline_metrics_v2.json")
    deep = load("deeponet_metrics_v2.json")
    gnn = load("gnn_metrics_v2.json")
    targets = ["temperature", "u", "v"]
    models = ["Per-node MLP", "GNN (true mesh)", "DeepONet"]
    data = {
        "Per-node MLP": [mlp["per_target"][t]["test_r2"] for t in targets],
        "GNN (true mesh)": [gnn["per_target"][t]["test_r2"] for t in targets],
        "DeepONet": [deep["per_target"][t]["test_r2"] for t in targets],
    }
    x = np.arange(len(targets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 6))
    style = {
        "Per-node MLP": dict(color=OKABE_ITO["blue"], edgecolor="black"),
        "GNN (true mesh)": dict(color=OKABE_ITO["green"], edgecolor="black"),
        "DeepONet": dict(color=OKABE_ITO["orange"], edgecolor="black"),
    }
    for i, m in enumerate(models):
        ax.bar(x + (i - 1) * width, data[m], width, label=m, **style[m])
    ax.set_xticks(x)
    ax.set_xticklabels(["Temperature", "$u$", "$v$"], fontsize=14)
    ax.set_ylabel("Test $R^2$ (held-out geometries)", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(-0.2, 1.25)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, fontsize=13, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_field_bakeoff.png", bbox_inches="tight")
    plt.close(fig)


def fig_blind_test_parity():
    with open(r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\validation\blind_test_comparison_v2.json") as f:
        blind = json.load(f)
    pts = sorted(blind["per_point"], key=lambda p: p["nu_true_fluent"])
    true = np.array([p["nu_true_fluent"] for p in pts])
    pred = np.array([p["nu_pred"] for p in pts])
    pct_err = np.abs(pred - true) / true * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    lims = [min(true.min(), pred.min()) * 0.9, max(true.max(), pred.max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1, label="One-to-one line", zorder=1)
    for t, p in zip(true, pred):
        ax.plot([t, t], [t, p], color="gray", linewidth=0.9, zorder=2)
    ax.scatter(true, pred, s=70, color=OKABE_ITO["blue"], edgecolor="black", linewidth=0.6,
               zorder=3, label=f"Held-out points ($n$={len(pts)})")
    ax.set_xlabel("True $Nu$ (numerical simulation)")
    ax.set_ylabel("Predicted $Nu$ (XGBoost)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)

    ax2 = axes[1]
    idx = np.arange(len(pts))
    ax2.bar(idx, pct_err, color=OKABE_ITO["orange"], edgecolor="black", width=0.6)
    ax2.axhline(blind["mean_abs_pct_error"], color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1.5,
                label=f"Mean = {blind['mean_abs_pct_error']:.2f}%")
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f"{t:.3g}" for t in true], rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("True $Nu$ (points sorted by value)")
    ax2.set_ylabel("Absolute percentage error (%)")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_blind_test_parity.png", bbox_inches="tight")
    plt.close(fig)


def fig_uq_calibration():
    uq = load("uq_xgboost_ensemble_v2.json")
    blind_pts = uq["true_blind_fluent_test"]["per_point"]
    blind_pts = sorted(blind_pts, key=lambda p: p["nu_true"])
    n = len(blind_pts)
    idx = np.arange(n)
    true = np.array([p["nu_true"] for p in blind_pts])
    mean = np.array([p["nu_pred_mean"] for p in blind_pts])
    std = np.array([p["nu_pred_std"] for p in blind_pts])
    in95 = [p["within_95pct_interval"] for p in blind_pts]
    labels = [f"$Re$={p['re']:g}, $Ri$={p['ri']:g}" for p in blind_pts]

    fig, ax = plt.subplots(figsize=(15, 6))
    width = 0.35
    ax.bar(idx - width / 2, true, width, color=OKABE_ITO["blue"], edgecolor="black", label="True (numerical simulation)")
    ax.bar(idx + width / 2, mean, width, color=OKABE_ITO["orange"], edgecolor="black", label="XGBoost ensemble mean")
    ax.errorbar(idx + width / 2, mean, yerr=1.96 * std, fmt="none", ecolor="black",
                elinewidth=1.6, capsize=6, capthick=1.6, label=r"$\pm\,2\sigma$ interval")

    for i, ok in enumerate(in95):
        mark = "✓" if ok else "✗"
        ax.text(i, max(true[i], mean[i] + 1.96 * std[i]) + 0.55, mark,
                ha="center", fontsize=15, color="black", fontweight="bold")

    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha="right")
    ax.set_ylabel("Nusselt number, $Nu$", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_uq_calibration.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig_scalar_bakeoff()
    fig_field_bakeoff()
    fig_blind_test_parity()
    fig_uq_calibration()
    print("Figures written to", OUT)
