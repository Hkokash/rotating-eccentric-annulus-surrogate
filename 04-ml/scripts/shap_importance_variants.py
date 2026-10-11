# =============================================================================
# Five professional alternative designs for the Figure 8 relative-importance
# chart, generated from the same SHAP values as shap_analysis_xgboost_v2.py,
# for the user to compare and choose from.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

BLUE = "#0072B2"


# ---- Option A: horizontal bar, flat color (current) -----------------------
def option_a():
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.barh(range(len(order)), pcts, color=BLUE, edgecolor="black", height=0.6)
    for b, p in zip(bars, pcts):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{p:.0f}%", va="center", fontsize=12)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(0, pcts.max() + 10)
    ax.set_xlabel("Relative importance (% of total mean $|$SHAP value$|$)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_shap_option_A_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Option B: horizontal lollipop (stem + dot) ----------------------------
def option_b():
    fig, ax = plt.subplots(figsize=(6, 3.2))
    y_pos = range(len(order))
    ax.hlines(y_pos, 0, pcts, color=BLUE, linewidth=2.5)
    ax.scatter(pcts, y_pos, color=BLUE, s=160, zorder=3, edgecolor="black", linewidth=1)
    for yp, p in zip(y_pos, pcts):
        ax.text(p + 2.5, yp, f"{p:.0f}%", va="center", fontsize=12)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(0, pcts.max() + 10)
    ax.set_xlabel("Relative importance (% of total mean $|$SHAP value$|$)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_shap_option_B_lollipop.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Option C: horizontal bar, sequential color gradient by magnitude -----
def option_c():
    fig, ax = plt.subplots(figsize=(6, 3.2))
    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=pcts.min() - 15, vmax=pcts.max() + 5)
    colors = [cmap(norm(p)) for p in pcts]
    bars = ax.barh(range(len(order)), pcts, color=colors, edgecolor="black", height=0.6)
    for b, p in zip(bars, pcts):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{p:.0f}%", va="center", fontsize=12)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(0, pcts.max() + 10)
    ax.set_xlabel("Relative importance (% of total mean $|$SHAP value$|$)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_shap_option_C_gradient.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Option D: single 100%-stacked horizontal bar --------------------------
def option_d():
    fig, ax = plt.subplots(figsize=(7, 1.8))
    okabe_ito = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
    left = 0
    for lab, p, c in zip(labels, pcts, okabe_ito):
        ax.barh(0, p, left=left, color=c, edgecolor="white", height=0.6)
        ax.text(left + p / 2, 0, f"{lab}\n{p:.0f}%", ha="center", va="center",
                fontsize=11, color="white" if p > 8 else "black",
                fontweight="bold")
        left += p
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Share of total mean $|$SHAP value$|$ (%)")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_shap_option_D_stacked.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Option E: donut chart --------------------------------------------------
def option_e():
    fig, ax = plt.subplots(figsize=(5, 5))
    okabe_ito = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
    wedges, _ = ax.pie(pcts, colors=okabe_ito, startangle=90, counterclock=False,
                        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5))
    for w, lab, p in zip(wedges, labels, pcts):
        ang = (w.theta2 + w.theta1) / 2
        x = 0.78 * np.cos(np.radians(ang))
        y = 0.78 * np.sin(np.radians(ang))
        ax.text(x, y, f"{lab}\n{p:.0f}%", ha="center", va="center", fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_shap_option_E_donut.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


option_a()
option_b()
option_c()
option_d()
option_e()
print("Wrote 5 SHAP importance figure options (A-E)")
for lab, p in zip(labels, pcts):
    print(f"{lab}: {p:.1f}%")
