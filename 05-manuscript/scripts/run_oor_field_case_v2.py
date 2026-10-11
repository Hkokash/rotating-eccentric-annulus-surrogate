# =============================================================================
# ONE new CFD run with FULL FIELD export at a genuinely off-grid (Re, Ri)
# combination, for a true out-of-range field comparison figure (CFD ground
# truth vs. per-node MLP surrogate prediction). Re/Ri redesign counterpart to
# run_oor_field_case.py.
#
# Point chosen: case_10 (R*=0.2, epsilon=0.04), Re=500, Ri=0.2. This is one
# of the 12 scalar blind-test points (run_blind_test_fluent_v2.py) -- picked
# specifically because it had the LARGEST scalar blind-test error (23.33%),
# so this field check honestly probes the same known-difficult, sparse-grid
# region rather than cherry-picking an easy point. Case 10 is also in the
# field-surrogate's own held-out test split (verified: train_test_split of
# the 35 geometries at random_state=42 puts case 10 in the 11-geometry test
# set), so this is doubly out-of-sample: never in field training rows, and
# at a (Re,Ri) combination never in the training grid.
#
# Physics setup, mesh, and export method imported directly from
# run_production_batch_v2_re_ri.py, same as run_field_export_batch_v2_re_ri.py,
# for exact parity with the production field dataset.
# =============================================================================

import os
import sys

import ansys.fluent.core as pyfluent

sys.path.insert(0, r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\scripts")
from run_production_batch_v2_re_ri import (
    MESH_DIR, PR_FIXED, setup_fixed_physics, iterate_until_stable,
)

OUT_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\05-manuscript\data"
OUTER_RADIUS = 0.5

CASE_NUM = 10
INNER_DIAM = 0.20
E = 0.04
RE = 500
RI = 0.2
V = 1.0 / RE  # matches derived_nondim: Re = 1/v exactly
B = RI        # matches derived_nondim: Ri = b exactly


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    case_file = os.path.join(MESH_DIR, f"case_{CASE_NUM:02d}.cas.h5")
    r_star = INNER_DIAM / (2 * OUTER_RADIUS)
    k = V / PR_FIXED

    out_file = os.path.join(OUT_DIR, f"oor_case{CASE_NUM:02d}_re{RE}_ri{RI}.csv")

    print(f"Running case_{CASE_NUM:02d} (R*={r_star}, eps={E}) at Re={RE} Ri={RI} (off-grid)...")
    solver = pyfluent.launch_fluent(
        mode=pyfluent.FluentMode.SOLVER, precision=pyfluent.Precision.DOUBLE,
        processor_count=2, dimension=2,
    )
    try:
        solver.file.read_case(file_name=case_file)
        air, inner_mom, rd = setup_fixed_physics(solver, E)
        air.viscosity.value = V
        air.thermal_conductivity.value = k
        air.therm_exp_coeff.value = B

        solver.solution.initialization.hybrid_initialize()
        nu, total_iters, converged = iterate_until_stable(solver, k)
        print(f"Nu={nu:.4f}, iters={total_iters}, converged={converged}")

        solver.file.export.ascii(
            file_name=out_file,
            surface_name_list=["interior:fluid"],
            delimiter="comma",
            quantities=["temperature", "x-velocity", "y-velocity"],
            location="node",
        )
        print(f"Wrote {out_file}")
    finally:
        solver.exit()


if __name__ == "__main__":
    main()
