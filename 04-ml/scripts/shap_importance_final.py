# =============================================================================
# Final Figure 8 design: vertical bar chart, relative SHAP importance (%),
# muted dark slate-navy, a standard academic-journal figure color.
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
OUT_FIG = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\05-manuscript\figures\fig_shap.png"
FEATURES = ["r_star", "epsilon", "re", "ri"]
SYMBOLS = {"r_star": r"$R^\ast$", "epsilon": r"$\varepsilon$", "re": r"$Re$", "ri": r"$Ri$"}
TARGET = "nu"
SEED = 42
PLUM = "#2C3E50"

plt.rcParams.update({"font.size": 12})


def main():
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

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    bars = ax.bar(range(len(order)), pcts, color=PLUM, edgecolor="black", width=0.6)
    for b, p in zip(bars, pcts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{p:.0f}%",
                ha="center", fontsize=12)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylim(0, pcts.max() + 10)
    ax.set_ylabel("Relative importance\n(% of total mean $|$SHAP value$|$)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")
    for lab, p in zip(labels, pcts):
        print(f"{lab}: {p:.1f}%")


if __name__ == "__main__":
    main()
