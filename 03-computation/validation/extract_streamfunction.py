# =============================================================================
# Extract the 2D streamfunction from our exported nodal velocity field, for
# comparison against Fusegi, Farouk & Ball (1986), Figs. 2b/3b.
#
# Method (matches how Fusegi/Yoo/Lee themselves handle this multiply-connected
# domain -- transform to polar coordinates (r, theta) about the cylinder
# center, which turns the annulus into a simply-connected rectangle):
#
#   u_r = (1/r) d(psi)/d(theta)      u_theta = -d(psi)/dr
#
# The inner (solid, no-penetration) wall has u_r=0 everywhere on it, so it is
# automatically a streamline: psi(r_i, theta) = psi_i = constant, independent
# of theta. Setting psi_i = 0 (arbitrary reference, only differences matter),
# psi at any (r, theta) is recovered by radially integrating the tangential
# velocity outward from the inner wall at that same theta:
#
#   psi(r, theta) = - integral_{r_i}^{r} u_theta(r', theta) dr'
#
# A useful internal consistency check: if the flow is genuinely steady and our
# interpolation/integration is accurate, psi(r_o, theta) should come out
# nearly constant across theta (the outer wall is also a streamline) -- this
# is checked and reported, not assumed.
# =============================================================================

import csv
import numpy as np
from scipy.interpolate import griddata

FIELD_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\field_data"
R_OUTER = 0.5


def load_case(case_num, ra_str, omega_str):
    path = f"{FIELD_DIR}\\case_{case_num:02d}\\ra{ra_str}_omega{omega_str}.csv"
    rows = list(csv.DictReader(open(path, "r"), skipinitialspace=True))
    x = np.array([float(r["x-coordinate"]) for r in rows])
    y = np.array([float(r["y-coordinate"]) for r in rows])
    u = np.array([float(r["x-velocity"]) for r in rows])
    v = np.array([float(r["y-velocity"]) for r in rows])
    return x, y, u, v


def streamfunction_from_field(x, y, u, v, r_i, r_o, ex=0.0, ey=0.0, n_r=120, n_theta=240):
    """ex, ey: inner-cylinder center offset from the coordinate origin (0 for
    concentric cases). Returns (psi grid, r values, theta values, outer-wall
    psi consistency spread)."""
    # Convert to polar coordinates about the INNER cylinder's own center,
    # since that's the natural frame for the (r,theta) integration.
    xc, yc = x - ex, y - ey
    r = np.sqrt(xc**2 + yc**2)
    theta = np.arctan2(yc, xc)
    u_r = (u * xc + v * yc) / np.maximum(r, 1e-12)
    u_theta = (-u * yc + v * xc) / np.maximum(r, 1e-12)

    r_grid = np.linspace(r_i, r_o, n_r)
    theta_grid = np.linspace(-np.pi, np.pi, n_theta)
    RR, TT = np.meshgrid(r_grid, theta_grid, indexing="ij")

    pts = np.column_stack([r, theta])
    u_theta_grid = griddata(pts, u_theta, (RR, TT), method="linear")
    # fill any NaN (extrapolation gaps) with nearest-neighbor
    if np.isnan(u_theta_grid).any():
        u_theta_grid_nn = griddata(pts, u_theta, (RR, TT), method="nearest")
        u_theta_grid = np.where(np.isnan(u_theta_grid), u_theta_grid_nn, u_theta_grid)

    psi = np.zeros_like(u_theta_grid)
    dr = r_grid[1] - r_grid[0]
    # psi(r,theta) = -integral_{r_i}^r u_theta dr', cumulative trapezoidal along r
    psi[1:, :] = -np.cumsum((u_theta_grid[1:, :] + u_theta_grid[:-1, :]) / 2 * dr, axis=0)

    outer_wall_psi = psi[-1, :]
    consistency_spread = outer_wall_psi.max() - outer_wall_psi.min()

    return psi, r_grid, theta_grid, consistency_spread


def main():
    targets = [
        dict(case=22, ra_str="5e+03", omega_str="02", r_star=0.4, label="Gr=139 target (Fusegi Fig 2b, Delta_psi=0.457)"),
        dict(case=22, ra_str="5e+05", omega_str="02", r_star=0.4, label="Gr=13900 target (Fusegi Fig 3b, Delta_psi=4.57)"),
    ]

    for t in targets:
        x, y, u, v = load_case(t["case"], t["ra_str"], t["omega_str"])
        r_i = t["r_star"] / 2
        psi, r_grid, theta_grid, spread = streamfunction_from_field(x, y, u, v, r_i, R_OUTER)

        psi_max, psi_min = psi.max(), psi.min()
        print(f"--- {t['label']} ---")
        print(f"  Case {t['case']:02d}, Ra={t['ra_str']}, omega={t['omega_str']}")
        print(f"  Outer-wall psi consistency spread (should be small if integration is accurate): {spread:.4f}")
        print(f"  psi range in domain: min={psi_min:.4f}, max={psi_max:.4f}, |max-min|={psi_max - psi_min:.4f}")
        print()


if __name__ == "__main__":
    main()
