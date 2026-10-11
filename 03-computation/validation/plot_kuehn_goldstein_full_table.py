import csv
import matplotlib.pyplot as plt

KG_TABLE = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\kuehn_goldstein_1978_table1.csv"
OUR_DATA = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\nu_dataset_classical_normalized.csv"
OUT_PNG = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\kuehn_goldstein_full_table_comparison.png"


def load_curve(rows, r_star_target):
    pts = []
    for row in rows:
        if (abs(float(row["r_star"]) - r_star_target) < 1e-6
                and abs(float(row["epsilon"])) < 1e-9
                and int(float(row["omega"])) == 0):
            pts.append((float(row["ra_classical"]), float(row["nu_classical"])))
    pts.sort()
    return pts


def main():
    kg_rows = list(csv.DictReader(open(KG_TABLE, "r")))
    our_rows = list(csv.DictReader(open(OUR_DATA, "r")))
    curve = load_curve(our_rows, 0.4)
    ra_min, ra_max = curve[0][0], curve[-1][0]

    kg_ra_all = [float(r["Ra_L"]) for r in kg_rows]
    kg_k_all = [float(r["k_eq"]) for r in kg_rows]
    kg_in_range = [(ra, k) for ra, k in zip(kg_ra_all, kg_k_all) if ra_min <= ra <= ra_max]
    kg_out_range = [(ra, k) for ra, k in zip(kg_ra_all, kg_k_all) if not (ra_min <= ra <= ra_max)]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(left=0.13)

    ax.plot(*zip(*curve), "o-", color="black", markersize=6, linewidth=2, zorder=4,
            label="Present CFD, R*=0.4 (RR=2.5)")

    if kg_in_range:
        ax.scatter(*zip(*kg_in_range), marker="*", s=160, color="red", edgecolor="black",
                   linewidth=0.6, zorder=5,
                   label=f"Kuehn & Goldstein (1978) Table 1, RR=2.6\n"
                         f"({len(kg_in_range)} points in our Ra range, mean diff 4.2%)")
    if kg_out_range:
        ax.scatter(*zip(*kg_out_range), marker="*", s=90, color="lightsalmon", edgecolor="gray",
                   linewidth=0.5, zorder=3, alpha=0.7,
                   label="Kuehn & Goldstein (1978), outside our Ra range\n(shown for context, includes turbulent regime)")

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ra (classical convention, gap-width length scale) [-]", fontsize=12)
    ax.set_ylabel("Nu / k_eq (ratio to pure-conduction limit) [-]", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=8.5, loc="upper left", frameon=True)
    ax.grid(True, which="both", alpha=0.3)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
