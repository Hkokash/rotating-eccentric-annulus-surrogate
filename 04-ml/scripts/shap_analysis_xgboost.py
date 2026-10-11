# =============================================================================
# SHAP interpretability analysis for the winning scalar Nu surrogate (XGBoost).
# Supports the "new physical insight" contribution claim in the Research
# Brief: goes beyond the native gain-based feature importance (already in
# xgboost_scalar_metrics.json) to show per-feature effect direction and
# interaction structure via SHAP values on the held-out test set.
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
OUT_JSON = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\results\shap_summary_v2.json"
OUT_FIG_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\figures"
FEATURES = ["r_star", "epsilon", "re", "ri"]
TARGET = "nu"
SEED = 42


def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=SEED)

    model = xgb.XGBRegressor()
    model.load_model(MODEL)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranking = sorted(zip(FEATURES, mean_abs_shap.tolist()), key=lambda kv: -kv[1])

    corr_with_shap = {}
    for i, feat in enumerate(FEATURES):
        corr_with_shap[feat] = float(np.corrcoef(X_test[feat].values, shap_values.values[:, i])[0, 1])

    summary = {
        "method": "SHAP TreeExplainer on winning XGBoost model, evaluated on the same 30% held-out test set",
        "n_test": len(X_test),
        "mean_abs_shap_ranking": [{"feature": f, "mean_abs_shap": v} for f, v in ranking],
        "feature_vs_shap_correlation": corr_with_shap,
        "interpretation_note": "positive correlation means the feature's SHAP contribution increases with the feature's raw value (monotonic positive effect on Nu on average); does not rule out non-monotonic or interaction effects, see the dependence plots.",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT_FIG_DIR}/shap_summary_beeswarm.png", dpi=150)
    plt.close()

    for feat in FEATURES:
        plt.figure()
        shap.dependence_plot(feat, shap_values.values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(f"{OUT_FIG_DIR}/shap_dependence_{feat}.png", dpi=150)
        plt.close()

    print(f"\nFigures written to {OUT_FIG_DIR}")


if __name__ == "__main__":
    main()
