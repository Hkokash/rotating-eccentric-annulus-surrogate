# =============================================================================
# Compare the CFD result from run_couette_validation_case.py against the
# exact analytical circular-Couette solution: v_theta(r) = A*r + B/r,
# A = Omega*Ri^2/(Ri^2-Ro^2), B = -A*Ro^2, for case_27 (R*=0.5, concentric,
# Ri=0.25, Ro=0.5), Omega=10. Buoyancy is off in that run, so this is a pure
# momentum/rotating-wall-BC check, independent of the Re/Ri convention used
# elsewhere in this study.
# =============================================================================

import numpy as np
import pandas as pd

CSV = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\couette_check\couette_case27_omega10.csv"
RI, RO, OMEGA = 0.25, 0.5, 10.0


def main():
    df = pd.read_csv(CSV)
    df.columns = [c.strip() for c in df.columns]
    x = df["x-coordinate"].values
    y = df["y-coordinate"].values
    u = df["x-velocity"].values
    v = df["y-velocity"].values
    r = np.sqrt(x**2 + y**2)

    # exclude the known inner-disk mesh artifact (r < Ri) -- not physical fluid
    fluid = r >= RI - 1e-6
    x, y, u, v, r = x[fluid], y[fluid], u[fluid], v[fluid], r[fluid]

    v_theta_cfd = (x * v - y * u) / r

    A = OMEGA * RI**2 / (RI**2 - RO**2)
    B = -A * RO**2
    v_theta_exact = A * r + B / r

    err = v_theta_cfd - v_theta_exact
    rel_err = np.abs(err) / OMEGA  # normalize by the wall speed scale, not local (near-zero at outer wall)

    print(f"n_fluid_nodes = {len(r)}")
    print(f"v_theta_exact range: [{v_theta_exact.min():.4f}, {v_theta_exact.max():.4f}]")
    print(f"v_theta_cfd   range: [{v_theta_cfd.min():.4f}, {v_theta_cfd.max():.4f}]")
    print(f"max |error| = {np.abs(err).max():.6f}")
    print(f"RMS error = {np.sqrt(np.mean(err**2)):.6f}")
    print(f"max |error| / (Omega*Ri) [wall speed] = {np.abs(err).max() / (OMEGA*RI) * 100:.4f}%")
    print(f"RMS error / (Omega*Ri) [wall speed] = {np.sqrt(np.mean(err**2)) / (OMEGA*RI) * 100:.4f}%")


if __name__ == "__main__":
    main()
