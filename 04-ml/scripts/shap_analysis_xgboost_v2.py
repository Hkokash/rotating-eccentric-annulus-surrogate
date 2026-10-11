# =============================================================================
# Single-panel SHAP feature-importance bar chart (percentage of total mean
# |SHAP| across the four inputs) for the winning scalar Nu surrogate
# (XGBoost). Deliberately a single compact bar chart rather than the earlier
# two-panel bar+beeswarm figure: the beeswarm distribution added detail
# (per-sample sign/spread, feature-value coloring) disproportionate to what
# the paper's interpretability claim actually needs -- the ranked, relative
# importance of the four inputs. Same underlying SHAP values as
# shap_analysis_xgboost.py (not recomputed differently).
# =============================================================================

import json
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

    fig, ax = plt.subplots(figsize=(6, 3.2))
    labels = [SYMBOLS[FEATURES[i]] for i in order]
    bars = ax.barh(range(len(order)), pct[order], color="#0072B2", edgecolor="black", height=0.6)
    for b, p in zip(bars, pct[order]):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{p:.0f}%",
                va="center", fontsize=12)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(0, pct.max() + 10)
    ax.set_xlabel("Relative importance (% of total mean $|$SHAP value$|$)")

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")
    for i in order:
        print(f"{FEATURES[i]}: {pct[i]:.1f}%")


if __name__ == "__main__":
    main()
