import csv
import matplotlib.pyplot as plt

SRC = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\nu_dataset_classical_normalized.csv"
OUT_PNG = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\rotation_behavior_vs_lee1992b.png"


def load_curve(rows, eps):
    pts = []
    for row in rows:
        if (abs(float(row["r_star"]) - 0.1) < 1e-6
                and abs(float(row["epsilon"]) - eps) < 1e-6
                and abs(float(row["ra"]) - 1e5) < 1):
            r_i = float(row["r_i"])
            gap = float(row["gap_width"])
            mu = float(row["mu"])
            omega = float(row["omega"])
            re = r_i * omega * gap / mu
            pts.append((re, float(row["nu_classical"])))
    pts.sort()
    return pts


def main():
    rows = list(csv.DictReader(open(SRC, "r")))
    curve_conc = load_curve(rows, 0.0)
    curve_ecc = load_curve(rows, 0.315)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.subplots_adjust(left=0.14)

    ax.plot(*zip(*curve_conc), "o-", color="#1f77b4", markersize=6, label="Present CFD, $\\epsilon$=0 (concentric)")
    ax.plot(*zip(*curve_ecc), "s-", color="#d62728", markersize=6, label="Present CFD, $\\epsilon$=0.315")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, label="Pure conduction limit (Nu=1)")

    ax.set_xlabel("Re (rotational Reynolds number) [-]", fontsize=12)
    ax.set_ylabel("Nu (classical convention) [-]", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=9, loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_title("Ra=$10^5$, R*=0.1 -- behavioral comparison against Lee (1992b)\n"
                  "Text claim: Nu decreases monotonically with Re, rate depends on eccentricity\n"
                  "(quantitative curve values not available -- Lee's Fig. 6 is not tabulated)",
                  fontsize=9.5)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
