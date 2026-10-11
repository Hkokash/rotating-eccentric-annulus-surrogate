# =============================================================================
# ECCENTRIC ANNULUS ML SURROGATE -- SPATIAL FIELD DATA EXPORT (Script 2 equiv.)
# Same 35 geometries x 10 Ra x 12 omega = 4200 combinations as the Nu dataset
# (run_production_batch.py), but exports the full spatial field (x, y,
# temperature, x-velocity, y-velocity per mesh node) instead of a single
# scalar Nu. Reuses the existing production meshes directly (no separate
# coarser mesh -- ours are already modest in size, ~1.7k-7k cells).
#
# Output layout (kept separate from the Nu dataset for tidiness):
#   field_data/case_NN/ra<RA>_omega<OMEGA>.csv   -- one file per run
#   field_data/manifest.csv                       -- index + per-run wall-clock
#                                                     timing (seconds) + node count
#
# Design notes carried over from run_production_batch.py:
#   - No zone renaming during meshing; meshes already have inner_wall/outer_wall/
#     fluid baked in (production .cas.h5 files).
#   - Fluent's ASCII export API: file.export.ascii(file_name, surface_name_list,
#     delimiter, quantities, location). Two gotchas found by direct testing:
#       * delimiter must be the literal word "comma" or "space", not the ","
#         character itself.
#       * "fluid" (the cell zone) is NOT a valid surface for this export --
#         must use "interior:fluid" (the internal face zone covering the whole
#         2D domain). x/y-coordinate are auto-included for location="node", so
#         they must NOT also be listed in `quantities` (produces duplicate
#         columns otherwise).
#   - Incremental manifest writes + resume-skip (by output file existence),
#     same robustness pattern that survived the Windows-update interruption
#     during the Nu batch.
#   - Per-run wall-clock timing recorded directly this time (not reconstructed
#     after the fact).
# =============================================================================

import csv
import os
import time

import ansys.fluent.core as pyfluent

from generate_geometry import CASES, OUTER_RADIUS
from run_production_batch import RA_TABLE, OMEGAS, setup_fixed_physics

MESH_DIR = r"C:\Users\ADMIN\Documents\AnnulusProject\meshes"
FIELD_DIR = r"C:\Users\ADMIN\Documents\AnnulusProject\field_data"
MANIFEST_CSV = os.path.join(FIELD_DIR, "manifest.csv")

EXPORT_SURFACE = "interior:fluid"
EXPORT_QUANTITIES = ["temperature", "x-velocity", "y-velocity"]

MANIFEST_FIELDS = [
    "case_num", "r_star", "epsilon", "e", "ra", "mu", "k", "omega",
    "output_file", "wall_clock_seconds", "node_count",
]

# Set to a small int while smoke-testing to limit how many geometries run;
# None for the real full batch.
DEBUG_LIMIT_GEOMETRIES = None


def load_done_keys():
    done = set()
    if os.path.exists(MANIFEST_CSV):
        with open(MANIFEST_CSV, "r", newline="") as f:
            for row in csv.DictReader(f):
                done.add((int(row["case_num"]), float(row["ra"]), float(row["omega"])))
    return done


def clean_file(path):
    if os.path.exists(path):
        os.remove(path)


def run_geometry(case_num, inner_diam, e, done_keys, manifest_writer, manifest_file):
    case_file = os.path.join(MESH_DIR, f"case_{case_num:02d}.cas.h5")
    case_field_dir = os.path.join(FIELD_DIR, f"case_{case_num:02d}")
    os.makedirs(case_field_dir, exist_ok=True)
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
            out_file = os.path.join(case_field_dir, f"ra{ra:.0e}_omega{omega:02d}.csv")
            clean_file(out_file)
            t0 = time.time()
            try:
                if ra != last_ra:
                    air.viscosity.value = mu
                    air.thermal_conductivity.value = k
                    last_ra = ra
                inner_mom.rotation_speed = omega

                solver.solution.initialization.hybrid_initialize()
                solver.solution.run_calculation.iterate(iter_count=1000)

                solver.file.export.ascii(
                    file_name=out_file,
                    surface_name_list=[EXPORT_SURFACE],
                    delimiter="comma",
                    quantities=EXPORT_QUANTITIES,
                    location="node",
                )
                elapsed = time.time() - t0

                with open(out_file, "r") as f:
                    node_count = sum(1 for _ in f) - 1  # minus header

                manifest_writer.writerow(dict(
                    case_num=case_num, r_star=r_star, epsilon=e, e=e,
                    ra=ra, mu=mu, k=k, omega=omega,
                    output_file=out_file, wall_clock_seconds=round(elapsed, 2),
                    node_count=node_count,
                ))
                manifest_file.flush()
                ok_count += 1
                print(f"  Ra={ra:.0e} omega={omega:>2}: {node_count} nodes, {elapsed:.1f}s -> {os.path.basename(out_file)}")
            except Exception as ex:
                fail_count += 1
                print(f"  FAILED Ra={ra}, omega={omega}: {ex}")
    finally:
        solver.exit()

    print(f"Case {case_num:02d} done: {ok_count} ok, {fail_count} failed.")
    return ok_count, fail_count


def main():
    os.makedirs(FIELD_DIR, exist_ok=True)
    file_exists = os.path.exists(MANIFEST_CSV)
    done_keys = load_done_keys()
    if done_keys:
        print(f"Resuming: {len(done_keys)} (case, Ra, omega) combos already exported.")

    total_ok = 0
    total_fail = 0
    cases = CASES if DEBUG_LIMIT_GEOMETRIES is None else CASES[:DEBUG_LIMIT_GEOMETRIES]

    with open(MANIFEST_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not file_exists:
            writer.writeheader()
            f.flush()

        for case_num, inner_diam, e in cases:
            ok, fail = run_geometry(case_num, inner_diam, e, done_keys, writer, f)
            total_ok += ok
            total_fail += fail

    print("\n" + "=" * 60)
    print("  FIELD EXPORT BATCH COMPLETE")
    print(f"  Successful runs : {total_ok}")
    print(f"  Failed runs     : {total_fail}")
    print(f"  Output          : {FIELD_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
