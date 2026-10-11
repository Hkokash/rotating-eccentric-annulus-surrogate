# =============================================================================
# SPATIAL FIELD DATA EXPORT -- v2 (Re/Ri sweep), field-level companion to
# run_production_batch_v2_re_ri.py. Same 35 geometries x 12 Re x 13 Ri =
# 5,460 combinations as the scalar Nu dataset, exporting the full spatial
# field (x, y, temperature, x-velocity, y-velocity per mesh node) instead of
# a single scalar Nu.
#
# Physics setup, convergence criterion, and derived non-dimensional numbers
# are imported DIRECTLY from run_production_batch_v2_re_ri.py rather than
# copy-pasted, to guarantee exact parity with the already-completed scalar
# dataset (same Re=1/v, Ri=b, Pr=0.71 via k=v/Pr, adaptive
# iterate-until-stable convergence). The report-definition-based Nu is still
# computed and written to the manifest alongside the field export, both as a
# free consistency cross-check against nu_dataset_v2_re_ri.csv and because
# the adaptive convergence check itself needs it regardless.
#
# Output layout:
#   field_data_v2_re_ri/case_NN/re<RE>_ri<RI>.csv   -- one file per run
#   field_data_v2_re_ri/manifest.csv                 -- index + nu (cross-check)
#                                                        + per-run timing + node count
#
# Export gotchas (carried over from the original run_field_export_batch.py,
# re-verified against these same mesh files, see that script's header):
#   - delimiter must be the literal word "comma", not the "," character.
#   - "fluid" (the cell zone) is not a valid export surface; use
#     "interior:fluid" (the internal face zone covering the whole 2D domain).
#   - x/y are auto-included for location="node" and must not also appear in
#     `quantities` (duplicate columns otherwise).
# =============================================================================

import csv
import os
import sys
import time

import ansys.fluent.core as pyfluent

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_geometry import CASES, OUTER_RADIUS
from run_production_batch_v2_re_ri import (
    MESH_DIR, RE_VALUES, V_VALUES, B_VALUES, OMEGA_FIXED, PR_FIXED,
    setup_fixed_physics, iterate_until_stable, derived_nondim,
)

FIELD_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\field_data_v2_re_ri"
MANIFEST_CSV = os.path.join(FIELD_DIR, "manifest.csv")

EXPORT_SURFACE = "interior:fluid"
EXPORT_QUANTITIES = ["temperature", "x-velocity", "y-velocity"]

MANIFEST_FIELDS = [
    "case_num", "r_star", "epsilon", "e", "v", "b", "re", "ri",
    "nu", "converged", "total_iters", "output_file",
    "wall_clock_seconds", "node_count",
]

# Set to a small int while smoke-testing; None for the real full batch.
DEBUG_LIMIT_GEOMETRIES = None
DEBUG_LIMIT_COMBOS = None


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
    if os.path.exists(MANIFEST_CSV):
        with open(MANIFEST_CSV, "r", newline="") as f:
            for row in csv.DictReader(f):
                done.add((int(row["case_num"]), float(row["v"]), float(row["b"])))
    return done


def clean_file(path):
    if os.path.exists(path):
        os.remove(path)


def run_geometry(case_num, inner_diam, e, done_keys, manifest_writer, manifest_file):
    case_file = os.path.join(MESH_DIR, f"case_{case_num:02d}.cas.h5")
    case_field_dir = os.path.join(FIELD_DIR, f"case_{case_num:02d}")
    os.makedirs(case_field_dir, exist_ok=True)
    r_star = inner_diam / (2 * OUTER_RADIUS)

    combos = [(v, b) for v in V_VALUES for b in B_VALUES]
    if DEBUG_LIMIT_COMBOS is not None:
        combos = combos[:DEBUG_LIMIT_COMBOS]
    remaining = [c for c in combos if (case_num, c[0], c[1]) not in done_keys]
    if not remaining:
        print(f"Case {case_num:02d}: all {len(combos)} combos already done, skipping.")
        return 0, 0

    print(f"\n=== Case {case_num:02d} (inner_diam={inner_diam}, e={e}): "
          f"{len(remaining)}/{len(combos)} combos remaining ===")

    solver = launch_with_retry(
        mode=pyfluent.FluentMode.SOLVER, precision=pyfluent.Precision.DOUBLE,
        processor_count=2, dimension=2,
    )

    ok_count = 0
    fail_count = 0
    try:
        solver.file.read_case(file_name=case_file)
        air, inner_mom, rd = setup_fixed_physics(solver, e)

        for v, b in remaining:
            pr, re, gr, ra, ri = derived_nondim(v, b)
            out_file = os.path.join(case_field_dir, f"re{re:.4g}_ri{ri:.4g}.csv")
            clean_file(out_file)
            t0 = time.time()
            try:
                k = v / PR_FIXED
                air.viscosity.value = v
                air.thermal_conductivity.value = k
                air.therm_exp_coeff.value = b

                solver.solution.initialization.hybrid_initialize()
                nu, total_iters, converged = iterate_until_stable(solver, k)

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
                    v=v, b=b, re=re, ri=ri, nu=nu, converged=converged,
                    total_iters=total_iters, output_file=out_file,
                    wall_clock_seconds=round(elapsed, 2), node_count=node_count,
                ))
                manifest_file.flush()
                ok_count += 1
                tag = "" if converged else "  [NOT STABLE at MAX_ITER]"
                print(f"  Re={re:.3g} Ri={ri:.3g}: Nu={nu:.4f}, {node_count} nodes, "
                      f"{elapsed:.1f}s{tag} -> {os.path.basename(out_file)}")
            except Exception as ex:
                fail_count += 1
                print(f"  FAILED Re={re:.3g}, Ri={ri:.3g}: {ex}")
    finally:
        solver.exit()

    print(f"Case {case_num:02d} done: {ok_count} ok, {fail_count} failed.")
    return ok_count, fail_count


def main():
    os.makedirs(FIELD_DIR, exist_ok=True)
    file_exists = os.path.exists(MANIFEST_CSV)
    done_keys = load_done_keys()
    if done_keys:
        print(f"Resuming: {len(done_keys)} (case, v, b) combos already exported.")

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
    print("  FIELD EXPORT BATCH v2 (Re/Ri) COMPLETE")
    print(f"  Successful runs : {total_ok}")
    print(f"  Failed runs     : {total_fail}")
    print(f"  Output          : {FIELD_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
