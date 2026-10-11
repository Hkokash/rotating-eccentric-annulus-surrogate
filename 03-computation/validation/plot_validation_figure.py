import csv
import matplotlib.pyplot as plt

SRC = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\nu_dataset_classical_normalized.csv"
OUT_PNG = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\validation_kuehn_goldstein.png"


def load_curve(rows, r_star_target):
    pts = []
    for row in rows:
        if (abs(float(row["r_star"]) - r_star_target) < 1e-6
                and abs(float(row["epsilon"])) < 1e-9
                and int(float(row["omega"])) == 0
                and row["nu_classical"] != ""):
            pts.append((float(row["ra_classical"]), float(row["nu_classical"])))
    pts.sort()
    return pts


def main():
    rows = list(csv.DictReader(open(SRC, "r")))
    curve_04 = load_curve(rows, 0.4)
    curve_02 = load_curve(rows, 0.2)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(left=0.14)

    ax.plot(*zip(*curve_04), "o-", color="#1f77b4", markersize=7, alpha=0.85, linewidth=2,
            label="Present CFD, R*=0.4 (RR=2.5, nearest match to Kuehn & Goldstein RR=2.6)")
    ax.plot(*zip(*curve_02), "s--", color="#ff7f0e", markersize=7, alpha=0.6, linewidth=1.5,
            label="Present CFD, R*=0.2 (RR=5.0, matches Lee 1992b geometry -- shown for context only)")

    # The one real literature number
    target_ra, target_nu = 2380, 1.38
    ax.scatter([target_ra], [target_nu], marker="*", s=380, color="red",
               edgecolor="black", linewidth=1, zorder=6,
               label="Kuehn & Goldstein (1978) experimental value\n(via Pranowo et al. 2024, Table 1)")

    # Our interpolated value at the exact same Ra, for a direct visual A-vs-B
    our_nu_at_target = loglog_interp(curve_04, target_ra)
    ax.scatter([target_ra], [our_nu_at_target], marker="D", s=140, color="#1f77b4",
               edgecolor="black", linewidth=1, zorder=6,
               label=f"Present CFD, same Ra (interpolated): Nu={our_nu_at_target:.3f}")

    # Connect the two points at matching Ra with a visible bracket + %-difference label
    ax.plot([target_ra, target_ra], [target_nu, our_nu_at_target],
            color="black", linestyle="-", linewidth=1.2, zorder=5)
    pct = 100 * abs(our_nu_at_target - target_nu) / target_nu
    mid_nu = (target_nu * our_nu_at_target) ** 0.5  # geometric mean, since axes are log
    ax.annotate(f"{pct:.1f}% difference", xy=(target_ra, mid_nu),
                xytext=(target_ra * 1.35, mid_nu),
                fontsize=10, va="center", fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ra (classical convention, gap-width length scale) [-]", fontsize=12)
    ax.set_ylabel("Nu (classical convention, normalized to conduction limit) [-]", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=8.5, loc="upper left", frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Both series plotted on the same axes: our CFD curve (line) vs.\n"
                  "the one real published data point (star), compared directly at matching Ra",
                  fontsize=10)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")


def loglog_interp(pts, ra_query):
    import math
    for i in range(len(pts) - 1):
        ra0, nu0 = pts[i]
        ra1, nu1 = pts[i + 1]
        if ra0 <= ra_query <= ra1:
            t = (math.log(ra_query) - math.log(ra0)) / (math.log(ra1) - math.log(ra0))
            return math.exp(math.log(nu0) + t * (math.log(nu1) - math.log(nu0)))
    return None


if __name__ == "__main__":
    main()
