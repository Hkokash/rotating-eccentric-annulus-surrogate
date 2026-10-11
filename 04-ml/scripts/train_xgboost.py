# =============================================================================
# Scalar Nu surrogate, candidate 1 of 3: modern gradient boosting (XGBoost).
# Validation protocol per .forge/decisions.log (2026-08-05): 70/30 holdout +
# 5-fold CV on the training set (primary), random_state=42 throughout for
# reproducibility. Leave-One-Geometry-Out CV is a separate script.
# =============================================================================

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb

DATA = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\scalar_dataset_filtered_v2_re_ri.csv"
OUT_MODEL = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\models\xgboost_scalar_v2.json"
OUT_METRICS = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\results\xgboost_scalar_metrics_v2.json"
FEATURES = ["r_star", "epsilon", "re", "ri"]
TARGET = "nu"
SEED = 42


def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=SEED)

    param_grid = {
        "n_estimators": [200, 400, 800],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1],
    }
    base = xgb.XGBRegressor(random_state=SEED, objective="reg:squarederror", n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(base, param_grid, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred_test = best.predict(X_test)
    y_pred_train = best.predict(X_train)

    metrics = {
        "model": "XGBoost",
        "best_params": grid.best_params_,
        "cv_best_r2_mean": float(grid.best_score_),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "test_mape_pct": float(mean_absolute_percentage_error(y_test, y_pred_test) * 100),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "random_state": SEED,
    }

    best.save_model(OUT_MODEL)
    with open(OUT_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

    # native feature importance (gain-based)
    importances = dict(zip(FEATURES, best.feature_importances_.tolist()))
    print("\nFeature importances (gain-based):")
    for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
