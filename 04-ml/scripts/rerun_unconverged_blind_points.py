# =============================================================================
# 3 of the 12 blind-test points did not cleanly converge at iter_count=1000
# (continuity residual: case18/Ra=7e5/om=7 -> 1.58e-3, case29/Ra=7e4/om=19 ->
# 1.07e-3, case34/Ra=7e5/om=13 -> 0.33, badly unconverged). Same protocol as
# run_blind_test_fluent.py, generous iteration ceiling (Fluent's iterate()
# auto-stops once its own convergence criteria are met, per this project's
# established behavior - see .forge/decisions.log). Overwrites the affected
# rows in blind_test_fluent_results.csv.
# =============================================================================

import csv
import math
import os

import ansys.fluent.core as pyfluent

MESH_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\meshes"
OUT_CSV = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\validation\blind_test_fluent_results.csv"
OUTER_RADIUS = 0.5
PR = 0.71
MAX_ITER = 5000

RERUN_POINTS = [
    (18, 0.30, 0.070, 700000, 7),
    (29, 0.50, 0.040, 70000, 19),
    (34, 0.60, 0.032, 700000, 13),
]

CSV_FIELDS = ["case_num", "r_star", "epsilon", "e", "ra", "mu", "k", "omega", "q_avg", "nu"]


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

    solver.solution.report_definitions.surface.create("inner_wall_hf")
    rd = solver.solution.report_definitions.surface["inner_wall_hf"]
    rd.report_type = "surface-areaavg"
    rd.field = "heat-flux"
    rd.surface_names = ["inner_wall"]
    return air, inner_mom, rd


def main():
    results = {}
    for case_num, inner_diam, e, ra, omega in RERUN_POINTS:
        case_file = os.path.join(MESH_DIR, f"case_{case_num:02d}.cas.h5")
        r_star = inner_diam / (2 * OUTER_RADIUS)
        k = 1.0 / math.sqrt(ra)
        mu = PR * k

        print(f"\n=== Rerun case_{case_num:02d} Ra={ra} omega={omega} (max_iter={MAX_ITER}) ===")
        solver = pyfluent.launch_fluent(
            mode=pyfluent.FluentMode.SOLVER, precision=pyfluent.Precision.DOUBLE,
            processor_count=2, dimension=2,
        )
        try:
            solver.file.read_case(file_name=case_file)
            air, inner_mom, rd = setup_fixed_physics(solver, e)
            air.viscosity.value = mu
            air.thermal_conductivity.value = k
            inner_mom.rotation_speed = omega

            solver.solution.initialization.hybrid_initialize()
            solver.solution.run_calculation.iterate(iter_count=MAX_ITER)

            result = solver.solution.report_definitions.compute(report_defs=["inner_wall_hf"])
            q_avg = result[0]["inner_wall_hf"][0]
            nu = q_avg / k
            print(f"  -> Nu={nu:.4f}")
            results[(case_num, ra, omega)] = dict(
                case_num=case_num, r_star=r_star, epsilon=e, e=e,
                ra=ra, mu=mu, k=k, omega=omega, q_avg=q_avg, nu=nu,
            )
        finally:
            solver.exit()

    rows = []
    with open(OUT_CSV, "r", newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["case_num"]), float(row["ra"]), float(row["omega"]))
            key_int_ra = (int(row["case_num"]), int(float(row["ra"])), float(row["omega"]))
            matched = None
            for rk in results:
                if rk[0] == key[0] and abs(rk[1] - key[1]) < 1e-6 and abs(rk[2] - key[2]) < 1e-6:
                    matched = rk
                    break
            rows.append(results[matched] if matched else row)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nUpdated {OUT_CSV} with {len(results)} rerun rows.")


if __name__ == "__main__":
    main()
