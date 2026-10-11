# =============================================================================
# FULL-TABLE validation against Kuehn & Goldstein (1978), ASME J. Heat Transfer
# 100(4), 635-640, Table 1 -- 39 experimental data points, D_o/D_i = 2.6,
# concentric, stationary. Their k_eq (Eq. 1g: k_eq = Nu/Nu_cond) is exactly
# the same "ratio to pure conduction" convention independently derived and
# verified for this project's own nu_classical column -- directly comparable,
# no further conversion needed on either side.
# =============================================================================

import csv
import math

KG_TABLE = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\kuehn_goldstein_1978_table1.csv"
OUR_DATA = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\nu_dataset_classical_normalized.csv"
OUT_TXT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\kuehn_goldstein_full_table_comparison.txt"


def load_our_curve(rows, r_star_target):
    pts = []
    for row in rows:
        if (abs(float(row["r_star"]) - r_star_target) < 1e-6
                and abs(float(row["epsilon"])) < 1e-9
                and int(float(row["omega"])) == 0):
            pts.append((float(row["ra_classical"]), float(row["nu_classical"])))
    pts.sort()
    return pts


def loglog_interp(pts, ra_query):
    for i in range(len(pts) - 1):
        ra0, nu0 = pts[i]
        ra1, nu1 = pts[i + 1]
        if ra0 <= ra_query <= ra1:
            t = (math.log(ra_query) - math.log(ra0)) / (math.log(ra1) - math.log(ra0))
            return math.exp(math.log(nu0) + t * (math.log(nu1) - math.log(nu0)))
    return None


def main():
    kg_rows = list(csv.DictReader(open(KG_TABLE, "r")))
    our_rows = list(csv.DictReader(open(OUR_DATA, "r")))
    curve = load_our_curve(our_rows, 0.4)
    ra_min, ra_max = curve[0][0], curve[-1][0]

    lines = []
    lines.append("Kuehn & Goldstein (1978) Table 1 -- FULL comparison, D_o/D_i=2.6")
    lines.append("Our nearest geometry: R*=0.4 (RR=2.5)")
    lines.append(f"Our data's Ra_classical range: {ra_min:.1f} to {ra_max:.1f}")
    lines.append(f"(Kuehn & Goldstein extends to Ra_L={float(kg_rows[-1]['Ra_L']):.2e}, into the turbulent")
    lines.append(" regime -- we only compare over the overlapping laminar range)")
    lines.append("")
    lines.append(f"{'Ra_L (K&G)':>12} {'k_eq (K&G)':>11} {'Nu_ours (interp)':>17} {'%diff':>8}")

    diffs = []
    for row in kg_rows:
        ra_l = float(row["Ra_L"])
        k_eq = float(row["k_eq"])
        if ra_min <= ra_l <= ra_max:
            nu_ours = loglog_interp(curve, ra_l)
            pct = 100 * abs(nu_ours - k_eq) / k_eq
            diffs.append(pct)
            lines.append(f"{ra_l:12.2e} {k_eq:11.3f} {nu_ours:17.4f} {pct:7.1f}%")

    lines.append("")
    lines.append(f"N points compared (within overlapping range): {len(diffs)}")
    if diffs:
        lines.append(f"Mean absolute %% difference: {sum(diffs)/len(diffs):.1f}%")
        lines.append(f"Min / Max %% difference: {min(diffs):.1f}% / {max(diffs):.1f}%")

    open(OUT_TXT, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
