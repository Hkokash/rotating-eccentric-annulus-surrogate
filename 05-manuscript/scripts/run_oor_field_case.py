# =============================================================================
# One new CFD run with FULL FIELD export (not just scalar Nu) at a genuinely
# off-grid (Ra, omega) combination, to build a true out-of-range field
# comparison figure (CFD ground truth vs. per-node MLP surrogate prediction)
# for the manuscript, per explicit user request: "you calculate it in the
# CFD, and then you do it with the prediction... put them against each other."
# Physics setup, mesh, and export method copied exactly from
# 03-computation/scripts/run_field_export_batch.py for consistency with the
# original field_data/ generation.
# =============================================================================

import math
import os

import ansys.fluent.core as pyfluent

MESH_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\meshes"
OUT_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\05-manuscript\data"
OUTER_RADIUS = 0.5
PR = 0.71

# Held-out test geometry (case 9, r_star=0.2, epsilon=0), off-grid (Ra, omega).
# Moderate condition (unlike the earlier Ra=3.5e5/omega=15 attempt, which failed
# to converge even at 3000 iterations, consistent with the known high-Ra +
# rotation difficulty) chosen specifically to converge cleanly, since the goal
# here is a genuine, trustworthy ground-truth field for comparison.
# Ra=3.5e4 is between the original grid's 2e4 and 5e4; omega=9 is between 8 and 10.
CASE_NUM = 9
INNER_DIAM = 0.20
E = 0.0
RA = 35000
OMEGA = 9


def setup_fixed_physics(solver, e):
    solver.setup.models.energy.enabled = True
    solver.setup.models.viscous.model = "laminar"
    solver.solution.controls.limits.min_temperature = 0
    solver.solution.controls.limits.max_temperature = 100
    solver.setup.general.operating_conditions.gravity.enable = True
    solver.setup.general.operating_conditions.gravity.components = [0, -1]
    solver.setup.general.operating_conditions.operating_pressure = 0
    solver.setup.general.operating_conditions.operating_temperature = 0

    air = solver.setup.materials.fluid["air"]
    air.density.option = "boussinesq"
    air.density.value = 1
    air.specific_heat.value = 1
    air.therm_exp_coeff.value = 1

    ref = solver.setup.reference_values
    ref.area = 1
    ref.density = 1
    ref.length = 1
    ref.temperature = 1
    ref.velocity = 1

    outer = solver.setup.boundary_conditions.wall["outer_wall"].thermal
    outer.thermal_condition = "Temperature"
    outer.temperature = 0
    inner_thermal = solver.setup.boundary_conditions.wall["inner_wall"].thermal
    inner_thermal.thermal_condition = "Temperature"
    inner_thermal.temperature = 1

    inner_mom = solver.setup.boundary_conditions.wall["inner_wall"].momentum
    inner_mom.wall_motion = "Moving Wall"
    inner_mom.rotating = True
    inner_mom.rotation_axis_origin = [0, -e]

    solver.solution.methods.p_v_coupling.flow_scheme = "SIMPLE"
    solver.solution.methods.spatial_discretization.discretization_scheme = {"pressure": "presto!"}
    return air, inner_mom


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    case_file = os.path.join(MESH_DIR, f"case_{CASE_NUM:02d}.cas.h5")
    r_star = INNER_DIAM / (2 * OUTER_RADIUS)
    k = 1.0 / math.sqrt(RA)
    mu = PR * k

    out_file = os.path.join(OUT_DIR, f"oor_case{CASE_NUM:02d}_ra{RA:.0e}_omega{OMEGA:02d}.csv")

    print(f"Running case_{CASE_NUM:02d} (R*={r_star}, eps={E}) at Ra={RA} omega={OMEGA} (off-grid)...")
    solver = pyfluent.launch_fluent(
        mode=pyfluent.FluentMode.SOLVER, precision=pyfluent.Precision.DOUBLE,
        processor_count=2, dimension=2,
    )
    try:
        solver.file.read_case(file_name=case_file)
        air, inner_mom = setup_fixed_physics(solver, E)
        air.viscosity.value = mu
        air.thermal_conductivity.value = k
        inner_mom.rotation_speed = OMEGA

        solver.solution.initialization.hybrid_initialize()
        solver.solution.run_calculation.iterate(iter_count=3000)

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
