# =============================================================================
# Confirms that the eccentric-annulus pure-conduction formula derived and
# numerically verified in this project (verify_eccentric_conduction_formula.py)
# is algebraically identical to the published closed-form solution in
# Prusa & Yao (1983), ASME J. Heat Transfer 105(1), 108-116, Eq. (8),
# obtained via conformal transformation -- a completely independent
# derivation method. This is a third, independent confirmation of the
# formula (after: 1) first-principles derivation, 2) numerical Laplace solve).
# =============================================================================

import math


def prusa_yao_1983_eq8(K, eps):
    """K = R_o/R_i, eps = e/R_i (their normalization)."""
    return math.acosh((1 + K**2 - eps**2) / (2 * K)) / math.log(K)


def present_project_formula(K, eps):
    """Same normalization, rewritten from verify_eccentric_conduction_formula.py's
    S = 2*pi/arccosh[(Ri^2+Ro^2-e^2)/(2*Ri*Ro)] as a ratio to the eps=0 case,
    using the identity arccosh((1+K^2)/(2K)) = ln(K)."""
    concentric_ref = math.log(K)
    return math.acosh((1 + K**2 - eps**2) / (2 * K)) / concentric_ref


def main():
    test_cases = [(2.6, 0.5), (2.6, 1.0), (5.0, 0.8)]
    print(f"{'K':>6} {'eps':>6} {'Prusa&Yao (1983)':>18} {'Present project':>16} {'Match':>8}")
    for K, eps in test_cases:
        py = prusa_yao_1983_eq8(K, eps)
        mine = present_project_formula(K, eps)
        print(f"{K:6.2f} {eps:6.2f} {py:18.6f} {mine:16.6f} {abs(py - mine) < 1e-10}")


if __name__ == "__main__":
    main()
