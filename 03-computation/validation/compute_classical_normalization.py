# =============================================================================
# CLASSICAL-CONVENTION NORMALIZATION FOR VALIDATION AGAINST LITERATURE
# =============================================================================
# Reads the ORIGINAL production Nu dataset (never modified) and writes a NEW,
# separately-named file with two added columns needed to compare our results
# against classical annulus-convection literature (Kuehn & Goldstein 1978 via
# Pranowo et al. 2024; Lee 1992a/1992b; Yoo 1998):
#
#   ra_classical  -- Ra recomputed with the GAP WIDTH (R_o - R_i) as the
#                     characteristic length, matching every one of the
#                     validation papers' convention. Our production dataset
#                     instead fixed L=1 (see PROJECT_SUMMARY.md Sec 6.1) so
#                     that Ra would be directly comparable ACROSS our own 35
#                     geometries -- a reasonable design choice for the ML
#                     dataset, but not the classical literature's convention.
#                     Conversion: Ra_classical = Ra_ours * (R_o - R_i)**3
#                     (valid for every row, concentric or eccentric, since
#                     the gap width itself doesn't depend on eccentricity).
#
#   nu_classical  -- Nu recomputed as a ratio to the pure-conduction limit,
#                     matching the classical convention (Nu -> 1 as Ra -> 0).
#                     Our production dataset instead defined Nu = q"/k
#                     directly (PROJECT_SUMMARY.md Sec 6.1), which does NOT
#                     divide out the conduction reference and therefore does
#                     not approach 1 at low Ra -- confirmed by hand-deriving
#                     the analytical concentric-annulus conduction solution
#                     and finding our CFD matches it to <0.3% at Ra=1e3.
#
#                     General formula, valid for BOTH concentric and
#                     eccentric geometries:
#                       Nu_classical = Nu_ours * 2*pi*R_i / S
#                     where S is the bipolar-coordinate conduction shape
#                     factor:
#                       S = 2*pi / arccosh( (R_i^2 + R_o^2 - e^2) / (2*R_i*R_o) )
#                     (reduces exactly to S = 2*pi/ln(R_o/R_i) as e -> 0,
#                     recovering the concentric formula used previously).
#
#                     This eccentric formula was NOT taken from memory --
#                     it was independently verified in
#                     verify_eccentric_conduction_formula.py /
#                     verify_fast_direct_solve.py by solving Laplace's
#                     equation numerically (direct sparse solve, grid
#                     convergence confirmed against the known-exact
#                     concentric case first: 1.6% -> 0.2% error as the grid
#                     refines) and checking the closed form against the
#                     converged numerical result at 4 eccentric test
#                     geometries: agreement was 0.33%-0.60%, matching the
#                     same residual discretization error seen in the
#                     concentric convergence check. See that verification
#                     for full detail.
#
# Neither of these corrections changes any CFD result. Both are computed
# purely from geometry (R_i, R_o) already implied by r_star, and from Ra/Nu
# values already in the production dataset. No simulation is re-run.
# =============================================================================

import csv
import math

SRC = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\results\nu_dataset.csv"
OUT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\nu_dataset_classical_normalized.csv"

OUTER_RADIUS = 0.5  # fixed outer cylinder radius, non-dimensional (matches generate_geometry.py)


def main():
    rows = list(csv.DictReader(open(SRC, "r")))
    fieldnames = list(rows[0].keys()) + ["r_i", "r_o", "gap_width", "ra_classical", "nu_classical"]

    n_concentric = 0
    n_eccentric_skipped = 0

    for row in rows:
        r_star = float(row["r_star"])
        epsilon = float(row["epsilon"])
        ra_ours = float(row["ra"])
        nu_ours = float(row["nu"])

        r_i = r_star / 2.0
        r_o = OUTER_RADIUS
        gap = r_o - r_i

        ra_classical = ra_ours * (gap ** 3)

        e = abs(epsilon)
        if e < 1e-9:
            shape_factor = 2 * math.pi / math.log(r_o / r_i)
            n_concentric += 1
        else:
            arg = (r_i**2 + r_o**2 - e**2) / (2 * r_i * r_o)
            shape_factor = 2 * math.pi / math.acosh(arg)
            n_eccentric_skipped += 1  # counter name kept for the summary print below; these are NOT skipped anymore

        nu_classical = nu_ours * 2 * math.pi * r_i / shape_factor

        row["r_i"] = round(r_i, 6)
        row["r_o"] = r_o
        row["gap_width"] = round(gap, 6)
        row["ra_classical"] = round(ra_classical, 4)
        row["nu_classical"] = round(nu_classical, 6)

    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT}")
    print(f"  nu_classical computed via concentric (ln) formula, epsilon=0: {n_concentric} rows")
    print(f"  nu_classical computed via eccentric (bipolar/arccosh) formula, epsilon!=0: {n_eccentric_skipped} rows")
    print("  Both formulas verified against an independent numerical Laplace solve -- see")
    print("  verify_eccentric_conduction_formula.py / verify_fast_direct_solve.py")


if __name__ == "__main__":
    main()
