# =============================================================================
# Consolidate the per-case field_data_v2_re_ri/ CSVs (35 geometries x 12 Re x
# 13 Ri = 5,460 runs, ~1.7k-7.3k scattered mesh nodes each) into a single
# parquet file for the spatial field surrogate. Non-converged cases (if any;
# the scalar dataset had zero out of 5,460) are dropped entirely, same
# physical-validity filter as the scalar dataset.
# Columns: case_num, r_star, epsilon, re, ri, x, y, temperature, u, v.
#
# Updated 2026-09-15 for the Re/Ri redesign -- paths, filename pattern
# (re<RE>_ri<RI>.csv, matching run_field_export_batch_v2_re_ri.py), and
# columns (ra/omega -> re/ri) all updated; logic otherwise unchanged.
# =============================================================================

import os
import pandas as pd

MANIFEST = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\field_data_v2_re_ri\manifest.csv"
FIELD_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\field_data_v2_re_ri"
OUT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\field_dataset_filtered_v2_re_ri.parquet"


def main():
    manifest = pd.read_csv(MANIFEST)
    manifest["converged"] = manifest["converged"].astype(str).str.strip().str.lower() == "true"
    manifest_conv = manifest[manifest["converged"]].copy()
    print(f"Manifest: {len(manifest)} total, {len(manifest_conv)} converged ({100*len(manifest_conv)/len(manifest):.1f}%)")

    dfs = []
    n_rows = 0
    for i, row in enumerate(manifest_conv.itertuples()):
        case_dir = os.path.join(FIELD_DIR, f"case_{int(row.case_num):02d}")
        fname = f"re{row.re:.4g}_ri{row.ri:.4g}.csv"
        path = os.path.join(case_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        field = pd.read_csv(path)
        field.columns = [c.strip() for c in field.columns]
        field = field.rename(columns={
            "x-coordinate": "x", "y-coordinate": "y",
            "temperature": "temperature", "x-velocity": "u", "y-velocity": "v",
        })
        field["case_num"] = int(row.case_num)
        field["r_star"] = row.r_star
        field["epsilon"] = row.epsilon
        field["re"] = row.re
        field["ri"] = row.ri
        dfs.append(field[["case_num", "r_star", "epsilon", "re", "ri", "x", "y", "temperature", "u", "v"]])
        n_rows += len(field)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(manifest_conv)} files loaded, {n_rows} rows so far")

    full = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(full)}")
    print(full.describe())

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    full.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
