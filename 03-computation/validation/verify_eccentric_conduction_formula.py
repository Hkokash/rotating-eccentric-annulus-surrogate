# =============================================================================
# INDEPENDENT VERIFICATION of the eccentric-annulus pure-conduction shape
# factor formula, BEFORE trusting it to convert 3,480 rows of production data.
#
# Candidate closed-form formula (bipolar-coordinate solution to Laplace's
# equation between two non-concentric circles, radii r_i/r_o, center-to-center
# distance d, held at T_i/T_o):
#
#   S = 2*pi / arccosh( (r_i^2 + r_o^2 - d^2) / (2*r_i*r_o) )
#   Q (heat flow per unit length) = k * (T_i - T_o) * S
#
# Rationale for the closed form: in bipolar coordinates (eta, theta), circles
# of constant eta map to circles in physical space, and Laplace's equation for
# a temperature field depending only on eta (not theta -- true here since both
# boundaries are isotherms) reduces exactly to d^2T/d(eta)^2 = 0, i.e. T is
# LINEAR in eta -- the exact bipolar-coordinate analogue of "T linear in ln(r)"
# for the concentric case. Integrating the resulting linear solution around
# either boundary gives the closed form above.
#
# INDEPENDENT CHECK: rather than trust this from memory, solve the same
# problem numerically (finite-difference relaxation of Laplace's equation on
# a Cartesian grid, masked to the true eccentric annular domain, immersed
# boundary conditions on the two circles) for several test geometries, and
# compare the numerically-obtained shape factor to the closed form above.
#
# WARNING (2026-09-16, caught during a rigorous re-audit): this script's
# fixed-point relaxation (40,000 iterations, numerical_shape_factor below)
# does NOT fully converge at these settings -- re-running it gives 2.6-4.8%
# disagreement with the closed form, which on its own would look like a real
# problem. It isn't: verify_fast_direct_solve_eccentric.py solves the exact
# same geometries with a direct (non-iterative) sparse solve and a proper
# grid-refinement study, and shows clean, monotonic convergence toward the
# closed form (1.5-3.4% at the coarsest grid down to 0.24-0.44% at the
# finest) -- i.e. the closed form is correct and THIS script's number is an
# under-convergence artifact, not evidence against it. Use
# verify_fast_direct_solve_eccentric.py for the real answer; this script is
# kept for its historical role (first cross-check) but its printed %diff
# should not be quoted as the numerical-agreement figure.
# =============================================================================

import numpy as np


def closed_form_shape_factor(r_i, r_o, d):
    x = (r_i**2 + r_o**2 - d**2) / (2 * r_i * r_o)
    return 2 * np.pi / np.arccosh(x)


def numerical_shape_factor(r_i, r_o, d, n=401, iters=40000):
    """Finite-difference solve of Laplace's equation between an inner circle
    (radius r_i, center at (0,-d)) and outer circle (radius r_o, center at
    origin), T=1 on inner, T=0 on outer. Returns the numerically-integrated
    shape factor S = Q / (k*(T_i-T_o)) via the inner-wall heat flux integral,
    non-dimensionalized to match the closed-form definition above."""
    L = r_o
    x = np.linspace(-L, L, n)
    y = np.linspace(-L, L, n)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    r_outer = np.sqrt(X**2 + Y**2)
    r_inner = np.sqrt(X**2 + (Y + d)**2)

    inside_outer = r_outer <= r_o
    inside_inner = r_inner <= r_i
    fluid = inside_outer & (~inside_inner)

    T = np.zeros((n, n))
    T[inside_inner] = 1.0
    T[~inside_outer] = 0.0
    # initialize fluid region with a linear guess between the two walls
    T[fluid] = 1.0 - (r_inner[fluid] - r_i) / (r_o - r_i + 1e-12)
    T[fluid] = np.clip(T[fluid], 0, 1)

    is_bc = inside_inner | (~inside_outer)

    for _ in range(iters):
        T_new = T.copy()
        T_new[1:-1, 1:-1] = 0.25 * (
            T[2:, 1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2]
        )
        T_new[is_bc] = T[is_bc]  # re-impose exact Dirichlet BCs every step
        T = T_new

    # Heat flux integral: total heat entering the domain across the inner
    # wall, via a control-volume flux balance immediately outside r_i.
    # Use a thin annular band just outside the inner cylinder and integrate
    # -k dT/dn using the grid's natural discrete Laplacian flux instead of a
    # a fragile boundary-fitted gradient (robust to the immersed-boundary grid).
    band = fluid & (r_inner < r_i + 3 * dx)
    # Approximate total heat flow via flux through a circle of radius r_i+2dx
    # around the inner cylinder's true center, using bilinear-sampled T.
    n_theta = 720
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    r_probe = r_i + 2 * dx
    xs = r_probe * np.cos(theta)
    ys = -d + r_probe * np.sin(theta)

    def sample(Tgrid, xq, yq):
        ix = (xq - x[0]) / dx
        iy = (yq - y[0]) / dx
        ix0 = np.clip(ix.astype(int), 0, n - 2)
        iy0 = np.clip(iy.astype(int), 0, n - 2)
        fx = ix - ix0
        fy = iy - iy0
        v00 = Tgrid[ix0, iy0]
        v10 = Tgrid[ix0 + 1, iy0]
        v01 = Tgrid[ix0, iy0 + 1]
        v11 = Tgrid[ix0 + 1, iy0 + 1]
        return (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
                + v01 * (1 - fx) * fy + v11 * fx * fy)

    r_probe2 = r_i + 4 * dx
    xs2 = r_probe2 * np.cos(theta)
    ys2 = -d + r_probe2 * np.sin(theta)
    T1 = sample(T, xs, ys)
    T2 = sample(T, xs2, ys2)
    dTdn = (T2 - T1) / (2 * dx)  # outward radial gradient from inner cylinder
    # Q = -k * integral(dT/dn * r_probe dtheta), k=1, T_i-T_o=1 already
    Q = -np.sum(dTdn) * r_probe * (2 * np.pi / n_theta)
    return Q  # this IS the shape factor S since k=1, (T_i-T_o)=1


def main():
    test_cases = [
        (0.10, 0.5, 0.00),   # concentric sanity check (d=0) -- must match ln formula too
        (0.10, 0.5, 0.10),
        (0.20, 0.5, 0.05),
        (0.20, 0.5, 0.15),
        (0.30, 0.5, 0.10),
    ]
    print(f"{'r_i':>6} {'r_o':>6} {'d':>6} {'S_closed_form':>15} {'S_numerical':>14} {'%diff':>8}")
    for r_i, r_o, d in test_cases:
        s_cf = closed_form_shape_factor(r_i, r_o, d) if d > 0 else 2 * np.pi / np.log(r_o / r_i)
        s_num = numerical_shape_factor(r_i, r_o, d, n=301, iters=25000)
        pct = 100 * abs(s_num - s_cf) / s_cf
        print(f"{r_i:6.2f} {r_o:6.2f} {d:6.2f} {s_cf:15.5f} {s_num:14.5f} {pct:7.2f}%")


if __name__ == "__main__":
    main()
