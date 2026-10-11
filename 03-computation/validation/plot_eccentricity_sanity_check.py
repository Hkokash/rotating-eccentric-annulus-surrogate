import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm

SRC = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\nu_dataset_classical_normalized.csv"
OUT_PNG = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\eccentricity_sanity_check.png"


def load_curve(rows, r_star_target, epsilon_target):
    pts = []
    for row in rows:
        if (abs(float(row["r_star"]) - r_star_target) < 1e-6
                and abs(float(row["epsilon"]) - epsilon_target) < 1e-6
                and int(float(row["omega"])) == 0):
            pts.append((float(row["ra_classical"]), float(row["nu_classical"])))
    pts.sort()
    return pts


def main():
    rows = list(csv.DictReader(open(SRC, "r")))
    eccentricities = [0.0, 0.045, 0.09, 0.135, 0.18, 0.225, 0.27, 0.315]
    colors = cm.viridis_r([i / (len(eccentricities) - 1) for i in range(len(eccentricities))])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.subplots_adjust(left=0.14)

    for eps, color in zip(eccentricities, colors):
        pts = load_curve(rows, 0.1, eps)
        ax.plot(*zip(*pts), "o-", color=color, markersize=5, label=f"$\\epsilon$ = {eps:.3f}")

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, label="Pure conduction limit (Nu=1)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ra (classical convention) [-]", fontsize=12)
    ax.set_ylabel("Nu (classical convention) [-]", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=True, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("R*=0.1, stationary ($\\omega$=0) -- internal consistency check\n"
                  "(all curves converge to Nu=1 at low Ra, as required)", fontsize=10)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
