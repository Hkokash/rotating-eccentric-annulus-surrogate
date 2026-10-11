# =============================================================================
# Three more alternative designs for the Figure 8 relative-importance chart,
# deliberately NOT bar-chart variants (user rejected the bar/lollipop/
# gradient-bar/stacked-bar/donut batch): a treemap, a radar/spider chart,
# and a 10x10 icon-array (waffle) chart.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

DATA = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\scalar_dataset_filtered_v2_re_ri.csv"
MODEL = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\models\xgboost_scalar_v2.json"
OUT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\05-manuscript\figures"
FEATURES = ["r_star", "epsilon", "re", "ri"]
SYMBOLS = {"r_star": r"$R^\ast$", "epsilon": r"$\varepsilon$", "re": r"$Re$", "ri": r"$Ri$"}
TARGET = "nu"
SEED = 42

plt.rcParams.update({"font.size": 12})

df = pd.read_csv(DATA)
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=SEED)

model = xgb.XGBRegressor()
model.load_model(MODEL)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
vals = shap_values.values

mean_abs = np.abs(vals).mean(axis=0)
pct = 100 * mean_abs / mean_abs.sum()
order = np.argsort(pct)[::-1]
labels = [SYMBOLS[FEATURES[i]] for i in order]
pcts = pct[order]
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]


# ---- Option F: treemap (simple slice-and-dice for 4 items) ----------------
def option_f():
    fig, ax = plt.subplots(figsize=(6, 4.2))
    W, H = 100.0, 100.0
    fracs = pcts / pcts.sum()

    # Squarified-ish layout for 4 rectangles: place the largest along the
    # full left edge, then stack the remaining three down the right side.
    w0 = fracs[0] * W
    rects = [(0, 0, w0, H)]
    remaining = fracs[1:] / fracs[1:].sum()
    y = 0.0
    for f in remaining:
        h = f * H
        rects.append((w0, y, W - w0, h))
        y += h

    for (x0, y0, w, h), lab, p, c in zip(rects, labels, pcts, OKABE_ITO):
        ax.add_patch(mpatches.Rectangle((x0, y0), w, h, facecolor=c, edgecolor="white", linewidth=2))
        fontsize = 15 if w * h > 600 else 11
        ax.text(x0 + w / 2, y0 + h / 2, f"{lab}\n{p:.0f}%", ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold")

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_shap_option_F_treemap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Option G: radar / spider chart ----------------------------------------
def option_g():
    n = len(order)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values = pcts.tolist()
    angles_closed = angles + angles[:1]
    values_closed = values + values[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, values_closed, color="#0072B2", linewidth=2)
    ax.fill(angles_closed, values_closed, color="#0072B2", alpha=0.25)
    ax.scatter(angles, values, color="#0072B2", s=60, zorder=3, edgecolor="black", linewidth=1)
    for ang, val, lab in zip(angles, values, labels):
        ax.text(ang, val + 9, f"{lab}\n{val:.0f}%", ha="center", va="center", fontsize=12)
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.set_ylim(0, 55)
    ax.set_yticks([10, 20, 30, 40])
    ax.set_yticklabels(["10", "20", "30", "40"], fontsize=9)
    ax.spines["polar"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_shap_option_G_radar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Option H: 10x10 icon array / waffle chart -----------------------------
def option_h():
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    grid = np.zeros((10, 10), dtype=int)
    counts = np.round(pcts).astype(int)
    counts[-1] = 100 - counts[:-1].sum()  # ensure exactly 100 squares
    flat_idx = 0
    cat_of_square = np.zeros(100, dtype=int)
    for cat, c in enumerate(counts):
        cat_of_square[flat_idx:flat_idx + c] = cat
        flat_idx += c
    grid = cat_of_square.reshape(10, 10)

    for row in range(10):
        for col in range(10):
            cat = grid[row, col]
            ax.add_patch(mpatches.Rectangle((col, 9 - row), 0.88, 0.88,
                                             facecolor=OKABE_ITO[cat], edgecolor="white"))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [mpatches.Patch(color=OKABE_ITO[i], label=f"{labels[i]}  {pcts[i]:.0f}%") for i in range(len(order))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=4, frameon=False, fontsize=12, handlelength=1.2)
    ax.set_title("Each square = 1% of total mean $|$SHAP value$|$", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_shap_option_H_waffle.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


option_f()
option_g()
option_h()
print("Wrote options F (treemap), G (radar), H (waffle)")
