# =============================================================================
# TRUE out-of-sample validation of the scalar Nu surrogate, Re/Ri redesign.
#
# Same rationale and structure as run_blind_test_fluent.py (the Ra/omega-era
# version, now superseded): run fresh Fluent simulations at (Re, Ri)
# combinations that were NEVER part of the 5,460-run production grid
# (12 Re values x 13 Ri values) at all, then compare the ML prediction to
# the real converged CFD result -- the only way to know a surrogate
# interpolates the true underlying physics rather than the specific
# sampling grid.
#
# Reuses the same 6 held-out geometries as the original blind test (one per
# R* group, at an eccentricity value already used there) -- no new
# geometry/meshing. Physics setup, thermal BCs, and Nu definition are
# copied exactly from 03-computation/scripts/run_production_batch_v2_re_ri.py
# (fixed Pr=0.71 via k=v/0.71, omega fixed at 1.0, adaptive
# iterate-until-stable convergence) for a fair, apples-to-apples comparison
# with the training data's own generation process.
# =============================================================================

import csv
import os
import time

import ansys.fluent.core as pyfluent

MESH_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\meshes"
OUT_CSV = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\validation\blind_test_fluent_results_v2.csv"
OUTER_RADIUS = 0.5
PR_FIXED = 0.71
OMEGA_FIXED = 1.0

# (case_num, inner_diam, e) - same 6 held-out geometries as the original
# blind test, copied from 03-computation/scripts/generate_geometry.py.
GEOMETRIES = {
    2:  (0.10, 0.045),
    10: (0.20, 0.040),
    18: (0.30, 0.070),
    24: (0.40, 0.048),
    29: (0.50, 0.040),
    34: (0.60, 0.032),
}

# (case_num, Re, Ri) - all points below fall strictly between grid values on
# both axes (RE_VALUES=[10,15,25,35,50,80,125,190,285,435,660,1000],
# RI_VALUES=[0.01,0.03,0.1,0.3,0.5,0.75,1,1.5,2,3,4,6,10] in
# run_production_batch_v2_re_ri.py); none appear in nu_dataset_v2_re_ri.csv.
BLIND_POINTS = [
    (2,  20,  0.4),
    (2,  800, 5),
    (10, 60,  0.85),
    (10, 500, 0.2),
    (18, 150, 1.25),
    (18, 950, 7),
    (24, 300, 2.5),
    (24, 12,  0.6),
    (29, 100, 1.75),
    (29, 65,  8),
    (34, 42,  0.05),
    (34, 330, 3.5),
    # 2026-09-16: gap-fill points. The original 12 points left a large,
    # unintended gap in ACHIEVED Ra between 270,617 (case34, Re=330,Ri=3.5)
    # and 2,272,000 (case2, Re=800,Ri=5) -- nothing samples Ra in between,
    # which is why the resulting Nu jumped straight from ~7.6 to ~21-31 with
    # no points in the Nu~10-20 range (flagged during a manuscript figure
    # review; verified by independent recomputation of Ra=0.71*Ri*Re^2 for
    # every existing point -- not a bug in the existing 12, just an
    # unintentional sampling gap). These 6 points fill that gap
    # log-uniformly (Ra approx 360k/600k/818k/1.08M/1.46M/1.92M), one per
    # held-out geometry, all strictly off-grid on both axes.
    (2,  480, 2.2),
    (10, 550, 2.8),
    (18, 600, 3.2),
    (24, 650, 3.6),
    (29, 700, 4.2),
    (34, 750, 4.8),
]

CSV_FIELDS = ["case_num", "r_star", "epsilon", "e", "v", "b", "omega", "q_avg", "k",
              "nu", "re", "ri", "converged", "total_iters"]


def launch_with_retry(**kwargs):
    last_err = None
    for attempt in range(4):
        try:
            return pyfluent.launch_fluent(**kwargs)
        except Exception as ex:
            last_err = ex
            print(f"  launch attempt {attempt + 1}/4 failed: {ex}")
            time.sleep(10)
    raise RuntimeError(f"Gave up launching Fluent: {last_err}")


def load_done_keys():
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, "r", newline="") as f:
            for row in csv.DictReader(f):
                done.add((int(row["case_num"]), float(row["re"]), float(row["ri"])))
    return done


def setup_fixed_physics(solver, e):
    """Copied exactly from run_production_batch_v2_re_ri.py."""
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
    inner_mom.rotation_speed = OMEGA_FIXED

    solver.solution.methods.p_v_coupling.flow_scheme = "SIMPLE"
    solver.solution.methods.spatial_discretization.discretization_scheme = {"pressure": "presto!"}

    solver.solution.report_definitions.surface.create("inner_wall_hf")
    rd = solver.solution.report_definitions.surface["inner_wall_hf"]
    rd.report_type = "surface-areaavg"
    rd.field = "heat-flux"
    rd.surface_names = ["inner_wall"]

    return air, inner_mom, rd


def iterate_until_stable(solver, k, base_iter=1000, chunk=500, max_iter=6000, tol=0.003):
    """Copied exactly from run_production_batch_v2_re_ri.py."""
    solver.solution.run_calculation.iterate(iter_count=base_iter)
    total = base_iter

    def current_nu():
        result = solver.solution.report_definitions.compute(report_defs=["inner_wall_hf"])
        return result[0]["inner_wall_hf"][0] / k

    prev_nu = current_nu()
    while total < max_iter:
        solver.solution.run_calculation.iterate(iter_count=chunk)
        total += chunk
        nu = current_nu()
        if abs(nu - prev_nu) / max(abs(nu), 1e-12) < tol:
            return nu, total, True
        prev_nu = nu
    return prev_nu, total, False


def run_geometry(case_num, inner_diam, e, points, done_keys, csv_writer, csv_file):
    case_file = os.path.join(MESH_DIR, f"case_{case_num:02d}.cas.h5")
    r_star = inner_diam / (2 * OUTER_RADIUS)

    remaining = [(re, ri) for re, ri in points if (case_num, re, ri) not in done_keys]
    if not remaining:
        print(f"Case {case_num:02d}: all blind points already done, skipping.")
        return

    print(f"\n=== Case {case_num:02d} (r_star={r_star}, e={e}): {len(remaining)} blind points ===")

    solver = launch_with_retry(
        mode=pyfluent.FluentMode.SOLVER, precision=pyfluent.Precision.DOUBLE,
        processor_count=2, dimension=2,
    )
    try:
        solver.file.read_case(file_name=case_file)
        air, inner_mom, rd = setup_fixed_physics(solver, e)

        for re, ri in remaining:
            v = 1.0 / re
            b = ri
            k = v / PR_FIXED
            air.viscosity.value = v
            air.thermal_conductivity.value = k
            air.therm_exp_coeff.value = b

            solver.solution.initialization.hybrid_initialize()
            nu, total_iters, converged = iterate_until_stable(solver, k)
            result = solver.solution.report_definitions.compute(report_defs=["inner_wall_hf"])
            q_avg = result[0]["inner_wall_hf"][0]

            csv_writer.writerow(dict(
                case_num=case_num, r_star=r_star, epsilon=e, e=e,
                v=v, b=b, omega=OMEGA_FIXED, q_avg=q_avg, k=k, nu=nu,
                re=re, ri=ri, converged=converged, total_iters=total_iters,
            ))
            csv_file.flush()
            print(f"  Re={re:g} Ri={ri:g}: Nu={nu:.4f}  iters={total_iters}  converged={converged}")
    finally:
        solver.exit()


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    file_exists = os.path.exists(OUT_CSV)
    done_keys = load_done_keys()

    points_by_case = {}
    for case_num, re, ri in BLIND_POINTS:
        points_by_case.setdefault(case_num, []).append((re, ri))

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
