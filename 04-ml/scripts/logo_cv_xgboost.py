# =============================================================================
# Leave-One-Geometry-Out (LOGO) cross-validation for the winning scalar Nu
# surrogate (XGBoost, chosen over PINN/KAN per the 2026-08-29 bake-off decision
# in .forge/decisions.log). Secondary validation protocol per the Research
# Brief: standard 70/30+5-fold CV (already run) mixes rows from the same
# (r_star, epsilon) geometry into both train and test, so it cannot detect
# whether the model is actually interpolating in (Ra, omega) at geometries it
# has already seen rather than truly generalizing across geometry. LOGO holds
# out all rows for one of the 35 distinct (r_star, epsilon) combinations at a
# time and predicts on it using a model trained on the other 34 geometries -
# same hyperparameters as the winning grid-search model, not re-tuned per fold.
# =============================================================================

import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb

DATA = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\scalar_dataset_filtered_v2_re_ri.csv"
BEST_PARAMS_FILE = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\results\xgboost_scalar_metrics_v2.json"
OUT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\results\logo_cv_xgboost_v2.json"
FEATURES = ["r_star", "epsilon", "re", "ri"]
TARGET = "nu"
SEED = 42


def main():
    df = pd.read_csv(DATA)
    with open(BEST_PARAMS_FILE) as f:
        best_params = json.load(f)["best_params"]

    df["geom_id"] = df["r_star"].astype(str) + "_" + df["epsilon"].astype(str)
    geoms = df["geom_id"].unique()

    fold_results = []
    for geom in geoms:
        train_df = df[df["geom_id"] != geom]
        test_df = df[df["geom_id"] == geom]

        model = xgb.XGBRegressor(random_state=SEED, objective="reg:squarederror",
                                  n_jobs=-1, **best_params)
        model.fit(train_df[FEATURES], train_df[TARGET])
        y_pred = model.predict(test_df[FEATURES])
        y_true = test_df[TARGET].values

        fold_results.append({
            "geom_id": geom,
            "r_star": float(test_df["r_star"].iloc[0]),
            "epsilon": float(test_df["epsilon"].iloc[0]),
            "n_test": len(test_df),
            "r2": float(r2_score(y_true, y_pred)),
            "mape_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        })

    r2_vals = [f["r2"] for f in fold_results]
    mape_vals = [f["mape_pct"] for f in fold_results]

    summary = {
        "method": "Leave-One-Geometry-Out (LOGO) CV, XGBoost with fixed best_params from 70/30 grid search",
        "best_params": best_params,
        "n_folds": len(fold_results),
        "r2_mean": float(np.mean(r2_vals)),
        "r2_std": float(np.std(r2_vals)),
        "r2_min": float(np.min(r2_vals)),
        "mape_pct_mean": float(np.mean(mape_vals)),
        "mape_pct_std": float(np.std(mape_vals)),
        "mape_pct_max": float(np.max(mape_vals)),
        "worst_geometries": sorted(fold_results, key=lambda f: f["r2"])[:5],
        "per_geometry_results": fold_results,
    }

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"LOGO CV over {len(fold_results)} geometries:")
    print(f"  R2:   mean={summary['r2_mean']:.4f}  std={summary['r2_std']:.4f}  min={summary['r2_min']:.4f}")
    print(f"  MAPE: mean={summary['mape_pct_mean']:.3f}%  std={summary['mape_pct_std']:.3f}%  max={summary['mape_pct_max']:.3f}%")
    print("\nWorst 5 geometries (lowest R2):")
    for f in summary["worst_geometries"]:
        print(f"  r_star={f['r_star']}, epsilon={f['epsilon']}: r2={f['r2']:.4f}, mape={f['mape_pct']:.2f}%")


if __name__ == "__main__":
    main()
