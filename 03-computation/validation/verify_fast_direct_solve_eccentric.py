# =============================================================================
# Extends verify_fast_direct_solve.py's accurate direct sparse-solve method
# (not the cruder iterative relaxation in verify_eccentric_conduction_formula.py)
# to the ECCENTRIC shape-factor formula specifically -- the actually novel part
# of methodology.tex's Equation (shape-factor); the concentric formula is a
# standard textbook result needing no validation of its own.
#
# The original verify_fast_direct_solve.py only tested the concentric case
# (d=0) in its main(). Re-run here at the same eccentric (r_i, r_o, d)
# triples used by verify_eccentric_conduction_formula.py, WITH a grid
# refinement study (not a single resolution), so the reported agreement
# reflects a genuinely grid-converged answer, not an arbitrary snapshot.
# =============================================================================

import numpy as np
from verify_fast_direct_solve import solve_laplace_direct


def closed_form_shape_factor(r_i, r_o, d):
    x = (r_i**2 + r_o**2 - d**2) / (2 * r_i * r_o)
    return 2 * np.pi / np.arccosh(x)


def main():
    test_cases = [(0.10, 0.50, 0.10), (0.20, 0.50, 0.05), (0.20, 0.50, 0.15), (0.30, 0.50, 0.10)]
    grid_ns = [151, 301, 601, 901, 1201]

    for r_i, r_o, d in test_cases:
        exact = closed_form_shape_factor(r_i, r_o, d)
        print(f"\n--- r_i={r_i}, r_o={r_o}, d={d}  (closed-form S={exact:.5f}) ---")
        for n in grid_ns:
            s = solve_laplace_direct(r_i, r_o, d, n)
            print(f"  n={n:5d}  S_numerical={s:.5f}  %diff={100*abs(s-exact)/exact:.3f}%")


if __name__ == "__main__":
    main()
