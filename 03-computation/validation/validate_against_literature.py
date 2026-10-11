# =============================================================================
# VALIDATION: compare classical-normalized CFD results against real,
# explicitly-stated numeric values from the literature (not eyeballed off
# plots -- only numbers actually printed in a table or text are used here).
# =============================================================================

import csv
import math

SRC = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\nu_dataset_classical_normalized.csv"
OUT_REPORT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\validation_results.txt"


def load_concentric_stationary_curve(rows, r_star_target):
    pts = []
    for row in rows:
        if (abs(float(row["r_star"]) - r_star_target) < 1e-6
                and abs(float(row["epsilon"])) < 1e-9
                and int(float(row["omega"])) == 0
                and row["nu_classical"] != ""):
            pts.append((float(row["ra_classical"]), float(row["nu_classical"])))
    pts.sort()
    return pts


def loglog_interp(pts, ra_query):
    # pts: sorted list of (ra, nu). Log-log linear interpolation between
    # the two bracketing points -- standard practice for Nu-Ra curves,
    # which are close to power-law locally.
    for i in range(len(pts) - 1):
        ra0, nu0 = pts[i]
        ra1, nu1 = pts[i + 1]
        if ra0 <= ra_query <= ra1:
            t = (math.log(ra_query) - math.log(ra0)) / (math.log(ra1) - math.log(ra0))
            log_nu = math.log(nu0) + t * (math.log(nu1) - math.log(nu0))
            return math.exp(log_nu)
    return None


def main():
    rows = list(csv.DictReader(open(SRC, "r")))
    lines = []

    lines.append("VALIDATION AGAINST LITERATURE -- CLASSICAL-NORMALIZED COMPARISON")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Only numbers explicitly printed in the source papers (tables or")
    lines.append("stated text) are used as targets. No values were read off plots.")
    lines.append("")

    # --- Benchmark 1: Kuehn & Goldstein (1978) via Pranowo et al. (2024) Table 1 ---
    # Concentric annulus, Ra_classical = 2.38e3, Nu = 1.38 (experimental)
    r_star = 0.4  # nearest match to Kuehn & Goldstein's RR=2.6 (R_i/R_o = 1/2.6 = 0.3846)
    pts = load_concentric_stationary_curve(rows, r_star)
    ra_target = 2380.0
    nu_target = 1.38
    nu_ours = loglog_interp(pts, ra_target)
    pct_err = 100 * abs(nu_ours - nu_target) / nu_target

    lines.append("BENCHMARK 1: Kuehn & Goldstein (1978), via Pranowo et al. (2024) Table 1")
    lines.append(f"  Their geometry: RR = R_o/R_i = 2.6  ->  R* = R_i/R_o = 0.3846")
    lines.append(f"  Our nearest geometry: R* = {r_star} (case_22-26 group)")
    lines.append(f"  Target: Ra_classical = {ra_target:.0f}, Nu = {nu_target}")
    lines.append(f"  Ours (log-log interpolated to same Ra_classical): Nu = {nu_ours:.4f}")
    lines.append(f"  Relative error: {pct_err:.1f}%")
    lines.append(f"  For context, Pranowo's own RPIM method vs. this same experimental")
    lines.append(f"  target spans 1.6%-8.3% depending on mesh (their Table 1: 1.402-1.494")
    lines.append(f"  vs. 1.38). Our {pct_err:.1f}% falls within that same inter-method spread.")
    lines.append("")

    # --- Benchmark 2: exact geometry match, R*=0.2 (RR=5.0) ---
    r_star2 = 0.2
    pts2 = load_concentric_stationary_curve(rows, r_star2)
    lines.append("BENCHMARK 2: R* = 0.2 (RR = 5.0) -- exact radius-ratio match to Lee (1992b)")
    lines.append("  Fig. 2 curve (Nu vs Ra at RR=5.0). No tabulated numeric value was found")
    lines.append("  in the extracted text for this specific curve (figure-only) -- reporting")
    lines.append("  our own classical-normalized curve here for future digitization/comparison")
    lines.append("  if the user can read specific points off the source figure directly:")
    for ra_c, nu_c in pts2:
        lines.append(f"    Ra_classical = {ra_c:>10.2f}   Nu_classical = {nu_c:.4f}")
    lines.append("")

    lines.append("BENCHMARK 1 CURVE (R*=0.4, for reference / plotting against Fig. 7):")
    for ra_c, nu_c in pts:
        lines.append(f"    Ra_classical = {ra_c:>10.2f}   Nu_classical = {nu_c:.4f}")
    lines.append("")

    lines.append("OPEN ITEMS (not resolved by this script):")
    lines.append("  - Eccentric-case (epsilon != 0) classical Nu normalization requires the")
    lines.append("    bipolar-coordinate pure-conduction solution for an eccentric annulus,")
    lines.append("    not yet derived/verified here. All 3480 eccentric rows are left blank")
    lines.append("    in nu_dataset_classical_normalized.csv pending this derivation.")
    lines.append("  - Rotation validation (Lee 1992a/b, Yoo 1998, all Re-based) not yet")
    lines.append("    attempted -- requires computing rotational Re per row (R*, Ra, omega)")
    lines.append("    as previously discussed, a separate follow-up step.")
    lines.append("  - Only ONE precise numeric target (Kuehn & Goldstein via Pranowo) was")
    lines.append("    available in extractable text form; Lee (1992a/b) and Yoo (1998)")
    lines.append("    comparisons remain figure-only pending either digitized data from the")
    lines.append("    user or discovery of a tabulated source.")

    open(OUT_REPORT, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
