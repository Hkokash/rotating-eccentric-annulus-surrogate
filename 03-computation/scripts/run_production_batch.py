# =============================================================================
# ECCENTRIC ANNULUS ML SURROGATE DATASET GENERATOR
# Runs the full physics parameter sweep across all 35 pre-meshed geometries
# (case_01.cas.h5 ... case_35.cas.h5) and records average Nusselt number on
# the inner wall for every (geometry, Ra, omega) combination.
#
# Parameter space (finalized after a 100-case convergence survey + literature
# cross-check -- see project discussion):
#   Ra:    10^3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6   (10 values)
#   omega: 0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 rad/s      (12 values)
#   -> 35 geometries x 10 Ra x 12 omega = 4200 runs
#
# Design notes (learned the hard way during development -- see dev_scratch/):
#   - No zone renaming/retyping during MESHING (unrelated to this script, but
#     the same "don't mutate the solver mid-flight" caution applies here too).
#   - Fluent's default temperature limiter (min=1 K) clamps a non-dimensional
#     [0,1] field almost everywhere -- must widen it (min=0, max=100).
#   - Viscous model defaults to a turbulence model; must force "laminar".
#   - Rotation axis origin must be set to (0, -e) per geometry (eccentric
#     inner cylinder), not (0,0).
#   - Report definition uses raw area-weighted-average 'heat-flux' on
#     inner_wall; Nu = q_avg / k is computed here in Python (k varies per Ra),
#     not via Fluent's built-in 'nusselt-number' field (whose internal
#     reference-value assumptions weren't independently verified).
#   - One Fluent solver session per GEOMETRY (35 launches total), looping
#     over all 120 (Ra, omega) combinations within that session -- avoids an
#     impractical 4200 separate launches.
#   - Incremental CSV writes + resume-skip: safe to interrupt and rerun.
# =============================================================================

import csv
import os

import ansys.fluent.core as pyfluent

from generate_geometry import CASES, OUTER_RADIUS

MESH_DIR = r"C:\Users\ADMIN\Documents\AnnulusProject\meshes"
OUT_CSV = r"C:\Users\ADMIN\Documents\AnnulusProject\results\nu_dataset.csv"

RA_TABLE = [
    (1e3, 0.02245, 0.03162),
    (2e3, 0.01588, 0.02236),
    (5e3, 0.01005, 0.01414),
    (1e4, 0.00710, 0.01000),
    (2e4, 0.00502, 0.00707),
    (5e4, 0.00318, 0.00447),
    (1e5, 0.00225, 0.00316),
    (2e5, 0.00159, 0.00224),
    (5e5, 0.00100, 0.00141),
    (1e6, 0.00071, 0.00100),
]
OMEGAS = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

CSV_FIELDS = ["case_num", "r_star", "epsilon", "e", "ra", "mu", "k", "omega", "q_avg", "nu"]

# Set to a small int while smoke-testing to limit how many geometries run;
# None for the real full batch.
DEBUG_LIMIT_GEOMETRIES = None


def load_done_keys():
    """Read any existing results file and return the set of (case_num, ra, omega)
    combinations already completed, so a rerun can skip them."""
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, "r", newline="") as f:
            for row in csv.DictReader(f):
                done.add((int(row["case_num"]), float(row["ra"]), float(row["omega"])))
    return done


def setup_fixed_physics(solver, e):
    """Settings that only need to be applied once per geometry session."""
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


def run_geometry(case_num, inner_diam, e, done_keys, csv_writer, csv_file):
    case_file = os.path.join(MESH_DIR, f"case_{case_num:02d}.cas.h5")
    r_star = inner_diam / (2 * OUTER_RADIUS)

    combos = [(ra, mu, k, omega) for ra, mu, k in RA_TABLE for omega in OMEGAS]
    remaining = [c for c in combos if (case_num, c[0], c[3]) not in done_keys]
    if not remaining:
        print(f"Case {case_num:02d}: all {len(combos)} combos already done, skipping.")
        return 0, 0

    print(f"\n=== Case {case_num:02d} (inner_diam={inner_diam}, e={e}): "
          f"{len(remaining)}/{len(combos)} combos remaining ===")

    solver = pyfluent.launch_fluent(
        mode=pyfluent.FluentMode.SOLVER,
        precision=pyfluent.Precision.DOUBLE,
        processor_count=2,
        dimension=2,
    )
    ok_count = 0
    fail_count = 0
    try:
        solver.file.read_case(file_name=case_file)
        air, inner_mom, rd = setup_fixed_physics(solver, e)

        last_ra = None
        for ra, mu, k, omega in remaining:
            try:
                if ra != last_ra:
                    air.viscosity.value = mu
                    air.thermal_conductivity.value = k
                    last_ra = ra
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
                ok_count += 1
                print(f"  Ra={ra:.0e} omega={omega:>2}: Nu={nu:.4f}")
            except Exception as ex:
                fail_count += 1
                print(f"  FAILED Ra={ra}, omega={omega}: {ex}")
    finally:
        solver.exit()

    print(f"Case {case_num:02d} done: {ok_count} ok, {fail_count} failed.")
    return ok_count, fail_count


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    file_exists = os.path.exists(OUT_CSV)
    done_keys = load_done_keys()
    if done_keys:
        print(f"Resuming: {len(done_keys)} (case, Ra, omega) combos already completed.")

    total_ok = 0
    total_fail = 0
    cases = CASES if DEBUG_LIMIT_GEOMETRIES is None else CASES[:DEBUG_LIMIT_GEOMETRIES]

    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
            f.flush()

        for case_num, inner_diam, e in cases:
            ok, fail = run_geometry(case_num, inner_diam, e, done_keys, writer, f)
            total_ok += ok
            total_fail += fail

    print("\n" + "=" * 60)
    print("  PRODUCTION BATCH COMPLETE")
    print(f"  Successful runs : {total_ok}")
    print(f"  Failed runs     : {total_fail}")
    print(f"  Output          : {OUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
