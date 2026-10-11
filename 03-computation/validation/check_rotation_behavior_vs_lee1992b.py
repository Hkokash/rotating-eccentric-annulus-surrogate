# =============================================================================
# QUALITATIVE/BEHAVIORAL comparison against Lee (1992b) Fig. 6 and its text
# description, since precise numeric values cannot be read off that figure.
#
# Lee (1992b) states explicitly (p.367-368, Conclusions):
#   "Nu decreases as Re of the rotating inner cylinder is increased for a
#    given Ra. An initial phase of slow reduction of Nu is followed by a
#    phase of rapid reduction and, finally, a phase where Nu asymptotes
#    towards the conduction limit... Nu is observed to decrease at
#    different rates depending on the eccentricity."
#
# This script checks whether OUR dataset shows the same behavior:
#   1. Nu_classical monotonically decreasing with rotational Re at fixed Ra
#   2. Approaching (not necessarily reaching, within our omega<=20 range)
#      the conduction limit (Nu_classical -> 1) as Re increases
#   3. Different eccentricities showing different decrease rates
#
# Rotational Reynolds number, matching Lee/Yoo's definition:
#   Re = R_i * omega * L / nu,   L = gap width = R_o - R_i,   nu = mu (rho=1)
# =============================================================================

import csv
import math

SRC = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\nu_dataset_classical_normalized.csv"


def main():
    rows = list(csv.DictReader(open(SRC, "r")))

    print("R*=0.1 group, Ra_ours=1e5 (matches Lee Fig 6's focus Ra), varying omega:")
    print(f"{'epsilon':>8} {'omega':>6} {'Re (rotational)':>17} {'Nu_classical':>13}")
    for eps in [0.0, 0.315]:
        print(f"  -- epsilon={eps} --")
        subset = []
        for row in rows:
            if (abs(float(row["r_star"]) - 0.1) < 1e-6
                    and abs(float(row["epsilon"]) - eps) < 1e-6
                    and abs(float(row["ra"]) - 1e5) < 1):
                r_i = float(row["r_i"])
                gap = float(row["gap_width"])
                mu = float(row["mu"])
                omega = float(row["omega"])
                re = r_i * omega * gap / mu
                subset.append((omega, re, float(row["nu_classical"])))
        subset.sort()
        for omega, re, nu_c in subset:
            print(f"{eps:8.3f} {omega:6.1f} {re:17.2f} {nu_c:13.4f}")

        nus = [x[2] for x in subset]
        monotonic = all(nus[i] >= nus[i+1] - 1e-9 for i in range(len(nus)-1))
        print(f"  Monotonically non-increasing with Re? {monotonic}")
        print(f"  Nu at omega=0: {nus[0]:.4f}  ->  Nu at omega=20: {nus[-1]:.4f}  "
              f"(fraction of drop toward conduction limit Nu=1: "
              f"{100*(nus[0]-nus[-1])/(nus[0]-1):.1f}%)")
        print()


if __name__ == "__main__":
    main()
