# =============================================================================
# Import and filter the production Nu dataset for ML training.
# Physical-validity filter applied BEFORE any training (per .forge/decisions.log,
# 2026-08-05): rows with converged=False are dropped entirely, not weighted
# or flagged as a feature. This mirrors the precedent project's costliest
# corrective fix (lessons-learned item 1.1).
#
# Updated 2026-09-09 for the Re/Ri redesign: SRC/OUT/FEATURES point at the new
# nu_dataset_v2_re_ri.csv (35 x 12 Re x 13 Ri = 5,460 runs) instead of the
# superseded Ra/omega dataset. Output goes to a NEW filename so the original
# scalar_dataset_filtered.csv (trained models still reference it) is left
# untouched.
#
# 2026-09-10: production batch complete (5,460/5,460, 0 failed). The raw CSV's
# "ri" column carries ~1e-15 relative floating-point noise (it's computed as
# gr/re**2 via chained division in run_production_batch_v2_re_ri.py rather
# than read back as the input b directly), so naive grouping sees ~31 unique
# values instead of the intended 13 discrete levels. "re" is unaffected (single
# division, no visible noise). Rounded to 8 decimals below -- far finer than
# the coarsest level (0.01) needs, just enough to collapse the noise.
# =============================================================================

import pandas as pd

SRC = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\results\nu_dataset_v2_re_ri.csv"
OUT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\scalar_dataset_filtered_v2_re_ri.csv"

FEATURES = ["r_star", "epsilon", "re", "ri"]
TARGET = "nu"


def main():
    df = pd.read_csv(SRC)
    n_total = len(df)

    df["converged"] = df["converged"].astype(str).str.strip().str.lower() == "true"
    df_filtered = df[df["converged"]].copy()
    n_kept = len(df_filtered)

    df_filtered["re"] = df_filtered["re"].round(8)
    df_filtered["ri"] = df_filtered["ri"].round(8)

    df_filtered = df_filtered[FEATURES + [TARGET, "case_num"]]

    print(f"Total rows: {n_total}")
    print(f"Converged (kept): {n_kept} ({100*n_kept/n_total:.1f}%)")
    print(f"Dropped (non-converged): {n_total - n_kept} ({100*(n_total-n_kept)/n_total:.1f}%)")
    print()
    print("Feature ranges (filtered set):")
    for f in FEATURES:
        print(f"  {f}: [{df_filtered[f].min():.4g}, {df_filtered[f].max():.4g}]")
    print(f"  {TARGET}: [{df_filtered[TARGET].min():.4g}, {df_filtered[TARGET].max():.4g}]")

    df_filtered.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
