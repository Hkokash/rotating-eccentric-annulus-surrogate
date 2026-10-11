# =============================================================================
# TRUE out-of-sample comparison: real Fluent Nu (blind_test_fluent_results_v2.csv,
# points genuinely never in the 5,460-run training grid) vs. the winning
# XGBoost model's prediction at the same (r_star, epsilon, re, ri).
#
# NOTE (2026-09-09, Re/Ri redesign): BLIND_CSV does not exist yet -- needs a
# new held-out batch of Fluent runs at Re/Ri combinations outside the new
# training grid (see run_blind_test_fluent.py, which will need the same
# Re/Ri update as this script before it can generate that data). EXCLUDED
# is left empty; populate only if a new blind point fails to converge, per
# the same physical-validity policy applied to the training set.
# =============================================================================

import json
import numpy as np
import pandas as pd
import xgboost as xgb

BLIND_CSV = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\validation\blind_test_fluent_results_v2.csv"
MODEL = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\models\xgboost_scalar_v2.json"
OUT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\validation\blind_test_comparison_v2.json"
FEATURES = ["r_star", "epsilon", "re", "ri"]

EXCLUDED = []


def main():
    df = pd.read_csv(BLIND_CSV)
    mask = ~df.apply(lambda r: (int(r.case_num), int(r.re), int(r.ri)) in EXCLUDED, axis=1)
    df_valid = df[mask].copy()
    df_excluded = df[~mask].copy()

    model = xgb.XGBRegressor()
    model.load_model(MODEL)

    df_valid["nu_pred"] = model.predict(df_valid[FEATURES])
    df_valid["abs_pct_error"] = (df_valid["nu_pred"] - df_valid["nu"]).abs() / df_valid["nu"] * 100

    results = df_valid[["case_num", "r_star", "epsilon", "re", "ri", "nu", "nu_pred", "abs_pct_error"]]
    results = results.rename(columns={"nu": "nu_true_fluent"})

    summary = {
        "description": "True out-of-sample validation: fresh Fluent CFD runs at (Re, Ri) combinations never present in the 5,460-run training grid, compared against the winning XGBoost scalar surrogate's prediction at the same point.",
        "n_points": len(results),
        "n_excluded_nonconverged": len(df_excluded),
        "excluded_points": df_excluded[["case_num", "r_star", "epsilon", "re", "ri"]].to_dict("records"),
        "mean_abs_pct_error": float(results["abs_pct_error"].mean()),
        "median_abs_pct_error": float(results["abs_pct_error"].median()),
        "max_abs_pct_error": float(results["abs_pct_error"].max()),
        "min_abs_pct_error": float(results["abs_pct_error"].min()),
        "per_point": results.to_dict("records"),
    }

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"True out-of-sample comparison ({summary['n_points']} points, "
          f"{summary['n_excluded_nonconverged']} excluded as non-converged):")
    print(f"  Mean abs %% error:   {summary['mean_abs_pct_error']:.2f}%")
    print(f"  Median abs %% error: {summary['median_abs_pct_error']:.2f}%")
    print(f"  Max abs %% error:    {summary['max_abs_pct_error']:.2f}%")
    print()
    for _, row in results.sort_values("abs_pct_error", ascending=False).iterrows():
        print(f"  case_{int(row.case_num):02d} r*={row.r_star} eps={row.epsilon} Re={row.re:g} "
              f"Ri={row.ri:g}: Fluent={row.nu_true_fluent:.4f} XGB={row.nu_pred:.4f} "
              f"err={row.abs_pct_error:.2f}%")


if __name__ == "__main__":
    main()
