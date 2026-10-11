# =============================================================================
# Uncertainty quantification for the winning scalar Nu surrogate (XGBoost).
# "Deep ensembles" (per decisions.log/Research Brief) is a neural-network
# technique; adapted here to a bootstrap ensemble of XGBoost models (same
# spirit - multiple independently-perturbed models, spread = uncertainty -
# applied to the actual winning architecture rather than forcing a neural net
# where one didn't win). N_MEMBERS models, same best_params as the winning
# grid-search model, each trained on an independent bootstrap resample of the
# training set with a different random seed.
#
# Calibration is checked TWO ways: (1) the same 70/30 test split used
# throughout, and (2) the 11-point true out-of-sample blind Fluent test set
# (blind_test_comparison.json) - the more meaningful check, since that is
# where the point estimate itself was already shown to be less accurate
# (2.89% mean error) than the same-grid test split suggested.
# =============================================================================

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb

# NOTE (2026-09-09, Re/Ri redesign): BLIND_CSV below does not exist yet for
# the new dataset -- it requires a NEW small batch of held-out Fluent runs
# at Re/Ri combinations outside the training grid (the same role
# run_blind_test_fluent.py played for the old Ra/omega design). That batch
# has not been generated. The same_grid_test_split portion of this script
# only needs the completed scalar dataset and can run as soon as that's
# ready; the true_blind_fluent_test portion will raise FileNotFoundError
# until the new blind-test CSV is built.
DATA = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\scalar_dataset_filtered_v2_re_ri.csv"
BEST_PARAMS_FILE = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\results\xgboost_scalar_metrics_v2.json"
BLIND_CSV = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\validation\blind_test_fluent_results_v2.csv"
OUT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\results\uq_xgboost_ensemble_v2.json"
FEATURES = ["r_star", "epsilon", "re", "ri"]
TARGET = "nu"
SEED = 42
N_MEMBERS = 20
EXCLUDED_BLIND = []  # fill in if any new blind points fail to converge


def main():
    df = pd.read_csv(DATA)
    with open(BEST_PARAMS_FILE) as f:
        best_params = json.load(f)["best_params"]

    X = df[FEATURES].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=SEED)

    rng = np.random.RandomState(SEED)
    n_train = len(X_train)
    test_preds = np.zeros((N_MEMBERS, len(X_test)))

    for m in range(N_MEMBERS):
        boot_idx = rng.randint(0, n_train, size=n_train)
        model = xgb.XGBRegressor(random_state=SEED + m, objective="reg:squarederror",
                                  n_jobs=-1, **best_params)
        model.fit(X_train[boot_idx], y_train[boot_idx])
        test_preds[m] = model.predict(X_test)

    test_mean = test_preds.mean(axis=0)
    test_std = test_preds.std(axis=0)
    z95 = 1.96
    within_95 = np.mean(np.abs(y_test - test_mean) <= z95 * test_std)

    result = {
        "method": f"Bootstrap ensemble of {N_MEMBERS} XGBoost models (same best_params, different bootstrap resample + seed each)",
        "same_grid_test_split": {
            "n_points": len(X_test),
            "ensemble_mean_r2": float(r2_score(y_test, test_mean)),
            "mean_predictive_std": float(test_std.mean()),
            "fraction_within_95pct_interval": float(within_95),
            "target_fraction": 0.95,
        },
    }

    if True:
        blind = pd.read_csv(BLIND_CSV)
        mask = ~blind.apply(lambda r: (int(r.case_num), int(r.re), int(r.ri)) in EXCLUDED_BLIND, axis=1)
        blind_valid = blind[mask].copy()
        X_blind = blind_valid[FEATURES].values
        y_blind = blind_valid[TARGET].values

        blind_preds = np.zeros((N_MEMBERS, len(X_blind)))
        rng2 = np.random.RandomState(SEED)
        for m in range(N_MEMBERS):
            boot_idx = rng2.randint(0, n_train, size=n_train)
            model = xgb.XGBRegressor(random_state=SEED + m, objective="reg:squarederror",
                                      n_jobs=-1, **best_params)
            model.fit(X_train[boot_idx], y_train[boot_idx])
            blind_preds[m] = model.predict(X_blind)

        blind_mean = blind_preds.mean(axis=0)
        blind_std = blind_preds.std(axis=0)
        blind_within_95 = np.mean(np.abs(y_blind - blind_mean) <= z95 * blind_std)
        abs_err = np.abs(y_blind - blind_mean)

        result["true_blind_fluent_test"] = {
            "n_points": len(X_blind),
            "ensemble_mean_r2": float(r2_score(y_blind, blind_mean)),
            "mean_predictive_std": float(blind_std.mean()),
            "fraction_within_95pct_interval": float(blind_within_95),
            "target_fraction": 0.95,
            "per_point": [
                {"case_num": int(blind_valid.iloc[i].case_num), "re": float(blind_valid.iloc[i].re),
                 "ri": float(blind_valid.iloc[i].ri), "nu_true": float(y_blind[i]),
                 "nu_pred_mean": float(blind_mean[i]), "nu_pred_std": float(blind_std[i]),
                 "abs_error": float(abs_err[i]), "within_95pct_interval": bool(abs_err[i] <= z95 * blind_std[i])}
                for i in range(len(X_blind))
            ],
        }

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "true_blind_fluent_test"}, indent=2))
    print("\nTrue blind Fluent test calibration:")
    print(json.dumps({k: v for k, v in result["true_blind_fluent_test"].items() if k != "per_point"}, indent=2))
    for p in result["true_blind_fluent_test"]["per_point"]:
        print(f"  case_{p['case_num']:02d} Re={p['re']:g} Ri={p['ri']:g}: "
              f"true={p['nu_true']:.4f} pred={p['nu_pred_mean']:.4f}+/-{p['nu_pred_std']:.4f} "
              f"in95%={p['within_95pct_interval']}")


if __name__ == "__main__":
    main()
