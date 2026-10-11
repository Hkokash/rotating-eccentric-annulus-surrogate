# =============================================================================
# TRUE out-of-sample validation of the winning scalar Nu surrogate (XGBoost).
#
# Everything reported so far (test-split R^2/MAPE, 5-fold CV, LOGO CV) draws
# from the same 4200-run design-of-experiments grid (Ra in {1e3,2e3,5e3,1e4,
# 2e4,5e4,1e5,2e5,5e5,1e6}, omega in {0,1,2,4,6,8,10,12,14,16,18,20}). None of
# those (Ra, omega) combinations were ever "new" to the experiment design, even
# when held out of training. Per the user's explicit request: run fresh Fluent
# simulations at (Ra, omega) combinations that were NEVER part of the original
# 4200-run grid at all, then compare the ML prediction to the real converged
# CFD result. This is the only way to know the model interpolates the true
# underlying physics rather than the specific sampling grid.
#
# Reuses the existing 35 pre-meshed geometries (no new geometry/meshing - this
# tests interpolation in (Ra, omega) at known geometries, not new geometries).
# Physics setup, convergence protocol, and Nu definition are copied EXACTLY
# from run_production_batch.py (03-computation/scripts/) for a fair, apples-
# to-apples comparison with the training data's own generation process.
#
# mu/k relationship recovered from the original RA_TABLE (verified exact to
# 3 sig figs across all 10 rows): k = 1/sqrt(Ra), mu = Pr/sqrt(Ra), Pr=0.71
# (standard air Prandtl number under this study's non-dimensionalization,
# where density=specific_heat=thermal_exp_coeff=reference values=1).
# =============================================================================

import csv
import math
import os

import ansys.fluent.core as pyfluent

MESH_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\meshes"
OUT_CSV = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\validation\blind_test_fluent_results.csv"
OUTER_RADIUS = 0.5
PR = 0.71

# (case_num, inner_diam, e) for the 6 test geometries - copied from
# 03-computation/scripts/generate_geometry.py CASES list.
GEOMETRIES = {
    2:  (0.10, 0.045),
    10: (0.20, 0.040),
    18: (0.30, 0.070),
    24: (0.40, 0.048),
    29: (0.50, 0.040),
    34: (0.60, 0.032),
}

# (case_num, ra, omega) - all 12 points below are OFF the original 10-value Ra
# grid and/or OFF the original 12-value omega grid; none appear anywhere in
# 03-computation/data/results/nu_dataset.csv.
BLIND_POINTS = [
    (2,  3500,   5),
    (2,  150000, 15),
    (10, 7000,   9),
    (10, 350000, 3),
    (18, 15000,  13),
    (18, 700000, 7),
    (24, 35000,  17),
    (24, 1500,   11),
    (29, 70000,  19),
    (29, 3500,   3),
    (34, 150000, 7),
    (34, 700000, 13),
]

CSV_FIELDS = ["case_num", "r_star", "epsilon", "e", "ra", "mu", "k", "omega", "q_avg", "nu"]


def load_done_keys():
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, "r", newline="") as f:
            for row in csv.DictReader(f):
                done.add((int(row["case_num"]), float(row["ra"]), float(row["omega"])))
    return done


def setup_fixed_physics(solver, e):
    """Copied exactly from 03-computation/scripts/run_production_batch.py."""
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

    solver.solution.report_definitions.surface.create("inner_wall_hf")
    rd = solver.solution.report_definitions.surface["inner_wall_hf"]
    rd.report_type = "surface-areaavg"
    rd.field = "heat-flux"
    rd.surface_names = ["inner_wall"]

    return air, inner_mom, rd


def run_geometry(case_num, inner_diam, e, points, done_keys, csv_writer, csv_file):
    case_file = os.path.join(MESH_DIR, f"case_{case_num:02d}.cas.h5")
    r_star = inner_diam / (2 * OUTER_RADIUS)

    remaining = [(ra, omega) for ra, omega in points if (case_num, ra, omega) not in done_keys]
    if not remaining:
        print(f"Case {case_num:02d}: all blind points already done, skipping.")
        return

    print(f"\n=== Case {case_num:02d} (r_star={r_star}, e={e}): {len(remaining)} blind points ===")

    solver = pyfluent.launch_fluent(
        mode=pyfluent.FluentMode.SOLVER,
        precision=pyfluent.Precision.DOUBLE,
        processor_count=2,
        dimension=2,
    )
    try:
        solver.file.read_case(file_name=case_file)
        air, inner_mom, rd = setup_fixed_physics(solver, e)

        for ra, omega in remaining:
            k = 1.0 / math.sqrt(ra)
            mu = PR * k
            air.viscosity.value = mu
            air.thermal_conductivity.value = k
            inner_mom.rotation_speed = omega

            solver.solution.initialization.hybrid_initialize()
            solver.solution.run_calculation.iterate(iter_count=1000)

            result = solver.solution.report_definitions.compute(report_defs=["inner_wall_hf"])
            q_avg = result[0]["inner_wall_hf"][0]
            nu = q_avg / k

            csv_writer.writerow(dict(
                case_num=case_num, r_star=r_star, epsilon=e, e=e,
                ra=ra, mu=mu, k=k, omega=omega, q_avg=q_avg, nu=nu,
            ))
            csv_file.flush()
            print(f"  Ra={ra:g} omega={omega}: Nu={nu:.4f}")
    finally:
        solver.exit()


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    file_exists = os.path.exists(OUT_CSV)
    done_keys = load_done_keys()

    points_by_case = {}
    for case_num, ra, omega in BLIND_POINTS:
        points_by_case.setdefault(case_num, []).append((ra, omega))

    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
            f.flush()

        for case_num, (inner_diam, e) in GEOMETRIES.items():
            run_geometry(case_num, inner_diam, e, points_by_case[case_num], done_keys, writer, f)

    print(f"\nDone. Results: {OUT_CSV}")


if __name__ == "__main__":
    main()
