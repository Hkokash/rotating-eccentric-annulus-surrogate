# =============================================================================
# ECCENTRIC ANNULUS ML SURROGATE DATASET GENERATOR -- v2 (Re/Ri sweep)
# Reuses the same 35 pre-meshed geometries as run_production_batch.py but
# replaces the old fixed-Pr Ra sweep with an independent viscosity /
# thermal-expansion sweep, so the dataset spans both natural- and
# mixed-convection regimes (Ri = 0.01..10, Re = 10..1000) instead of
# assuming pure natural convection.
#
# FOUR real inputs: viscosity (v, controls Re), thermal expansion
# coefficient (b, controls Ri), radius ratio (R*), eccentricity (epsilon).
# Omega is NOT a fifth independent input. For a single rotating cylinder
# (inner rotating, outer stationary -- not two independently-rotating
# cylinders referenced against each other), the rotational effect is fully
# captured by Re; sweeping omega and viscosity independently would be
# redundant (many (omega,v) pairs collapse onto the same Re) and omega
# only earns independent physical meaning in a two-rotating-cylinder
# problem. Omega is instead fixed at a single reference value OMEGA_FIXED=1
# rad/s, consistent with this project's existing "everything not being
# swept is normalized to 1" convention (rho=1, L=1, dT=1, g=1). Verified
# by deriving the non-dimensional momentum equation
# (du*/dt* = -grad(p*) + (1/Re)lap(u*) + (Gr/Re^2) T* g_hat) with reference
# velocity U_ref = omega0*L: this makes Re = U_ref*L/v = 1/v and
# Ri = Gr/Re^2 = b EXACTLY (the v^2 cancels), independent of geometry --
# confirmed numerically. RE_VALUES/V_VALUES span Re=10..1000 exactly;
# B_VALUES span Ri=0.01..10 exactly, densified around Ri~O(1) (see below).
# See .forge/decisions.log for the full derivation and grid-density history.
#
# What changes per run:
#   v (viscosity)          -> Re = 1/v
#   b (thermal expansion)  -> Ri = b  (Gr = b/v^2, Ra = Gr*Pr = 0.71*b/v^2)
# R* and epsilon vary across the 35 fixed geometries (mesh files).
#
# Carried over unchanged from run_production_batch.py (the same fixes are
# needed here -- verified empirically against the raw mesh files before
# writing this script; see .forge/decisions.log):
#   - Energy equation must be force-enabled explicitly (not left to the
#     Boussinesq-density side effect).
#   - Viscous model defaults to a turbulence model; must force "laminar".
#   - Gravity is off by default; must enable with components (0, -1).
#   - Default temperature limiter (min=1) clamps a non-dimensional [0,1]
#     field almost everywhere; must widen to [0, 100].
#   - inner_wall's mesh-inherited default thermal condition is "Coupled"
#     (it has a coincident zero-thickness shadow face, part-internal-shadow,
#     a meshing artifact -- same surface area as inner_wall). Explicitly
#     setting thermal_condition="Temperature" on inner_wall overrides this
#     default and converts it to a normal external boundary; the shadow
#     zone itself needs no separate handling (the original dataset was
#     built successfully on these same mesh files without touching it).
#   - outer_wall's mesh-inherited default is Heat Flux = 0 (adiabatic);
#     must also be set explicitly to Temperature = 0.
#   - Report area-weighted-avg HEAT FLUX on inner_wall, not temperature
#     (inner_wall is a fixed-temperature Dirichlet boundary once set, so
#     its own average temperature is a constant and carries no signal).
#     Nu = q_avg / k.
#   - One Fluent session per GEOMETRY (35 launches), looping over all 156
#     (v, b) combinations within that session.
#   - Incremental CSV writes + resume-skip: safe to interrupt and rerun.
#
# Pr is held fixed at 0.71 (air) for every run, not left to float with v.
# With Cp=1 and rho=1 fixed, Pr = v/k, so simply fixing k=1 (as an earlier
# version of this script did) would have made Pr = v -- ranging 0.0005 to
# 0.1 across the sweep, never reaching air's real Pr and breaking direct
# comparison to air-based literature (e.g. Kuehn & Goldstein). Instead k is
# computed per run as k = v / 0.71, which keeps Pr = v/k = 0.71 exactly
# regardless of v, mirroring how the old Ra-based dataset kept Pr fixed by
# scaling mu and k together. Nu = q_avg / k uses this real per-run k.
#
# Output is written to a NEW file (nu_dataset_v2_re_ri.csv) -- the original
# nu_dataset.csv from the Ra-based sweep is left untouched.
# =============================================================================

import csv
import os
import time

import ansys.fluent.core as pyfluent

from generate_geometry import CASES, OUTER_RADIUS

MESH_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\meshes"
OUT_CSV = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\results\nu_dataset_v2_re_ri.csv"

# Re = 1/v, 12 log-spaced points across 10..1000. Densified from an earlier
# 9-point version after review: the extra points give smoother global
# coverage of the Re axis for the eventual ML surrogate.
RE_VALUES = [10, 15, 25, 35, 50, 80, 125, 190, 285, 435, 660, 1000]
V_VALUES = [1.0 / r for r in RE_VALUES]
# Ri = b, 13 points across 0.01..10, deliberately densified around Ri~O(1)
# (0.5, 0.75, 1, 1.5, 2, 3, 4) -- that is the mixed-convection transition
# region (buoyancy and rotation comparably important), which is the whole
# physical motivation for this dataset; the original 6-point version left a
# 4x gap (0.5 -> 2) with nothing sampling the transition itself.
B_VALUES = [0.01, 0.03, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 10]

OMEGA_FIXED = 1.0  # rad/s -- fixed reference rotation rate (see header)
PR_FIXED = 0.71    # air; k is computed per run as v / PR_FIXED so Pr stays constant
L_REF = 1.0        # outer diameter, matches ref.length=1

CSV_FIELDS = [
    "case_num", "r_star", "epsilon", "e", "v", "b", "omega", "q_avg", "k", "nu",
    "pr", "re", "gr", "ra", "ri", "converged", "total_iters",
]


def derived_nondim(v, b):
    """Achieved non-dimensional numbers for one run. With omega fixed at
    OMEGA_FIXED=1 and reference velocity U_ref=OMEGA_FIXED*L_REF=1, these
    closed forms are exact (verified numerically): Re=1/v, Ri=b, independent
    of geometry -- see header derivation."""
    pr = PR_FIXED
    re = (OMEGA_FIXED * L_REF) * L_REF / v
    gr = b * L_REF ** 3 / v ** 2
    ra = gr * pr
    ri = gr / re ** 2
    return pr, re, gr, ra, ri

# Set to small ints while smoke-testing to limit scope; None for the real
# full batch (35 x 12 x 13 = 5,460 runs).
DEBUG_LIMIT_GEOMETRIES = None
DEBUG_LIMIT_COMBOS = None

# Adaptive convergence: run BASE_ITER, then compare Nu against Nu after each
# further CHUNK_ITER block; stop once consecutive Nu estimates agree within
# STABILITY_TOL, or MAX_ITER is hit (whichever first). A fixed iteration
# count that was fine at the old dataset's Ra range is not something to
# trust blindly at the new range (Ra now reaches ~7e6 at the worst corner,
# Re up to 1000) -- this checks actual stability per case instead of
# assuming a single iteration count covers the whole parameter space.
BASE_ITER = 1000
CHUNK_ITER = 500
MAX_ITER = 6000
STABILITY_TOL = 0.003  # 0.3% relative change between successive Nu checks


def load_done_keys():
    """Read any existing results file and return the set of (case_num, v, b)
    combinations already completed, so a rerun can skip them."""
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, "r", newline="") as f:
            for row in csv.DictReader(f):
                done.add((int(row["case_num"]), float(row["v"]), float(row["b"])))
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
    # thermal_conductivity is set per run (k = v / PR_FIXED) inside the combo
    # loop, not here, since it must track viscosity to keep Pr fixed.

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


def iterate_until_stable(solver, k):
    """Run BASE_ITER, then keep adding CHUNK_ITER blocks and comparing
    successive Nu estimates until they agree within STABILITY_TOL or
    MAX_ITER is reached. Returns (nu, total_iters, converged)."""
    solver.solution.run_calculation.iterate(iter_count=BASE_ITER)
    total = BASE_ITER

    def current_nu():
        result = solver.solution.report_definitions.compute(report_defs=["inner_wall_hf"])
        q_avg = result[0]["inner_wall_hf"][0]
        return q_avg / k

    prev_nu = current_nu()
    while total < MAX_ITER:
        solver.solution.run_calculation.iterate(iter_count=CHUNK_ITER)
        total += CHUNK_ITER
        nu = current_nu()
        if abs(nu - prev_nu) / max(abs(nu), 1e-12) < STABILITY_TOL:
            return nu, total, True
        prev_nu = nu

    return prev_nu, total, False


def run_geometry(case_num, inner_diam, e, done_keys, csv_writer, csv_file):
    case_file = os.path.join(MESH_DIR, f"case_{case_num:02d}.cas.h5")
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

    solver = None
    last_launch_err = None
    for attempt in range(3):
        try:
            solver = pyfluent.launch_fluent(
                mode=pyfluent.FluentMode.SOLVER,
                precision=pyfluent.Precision.DOUBLE,
                processor_count=2,
                dimension=2,
            )
            break
        except Exception as ex:
            last_launch_err = ex
            print(f"  Fluent launch attempt {attempt + 1}/3 failed: {ex}")
            time.sleep(10)
    if solver is None:
        print(f"Case {case_num:02d}: giving up after 3 launch attempts ({last_launch_err}).")
        return 0, len(remaining)

    ok_count = 0
    fail_count = 0
    try:
        solver.file.read_case(file_name=case_file)
        air, inner_mom, rd = setup_fixed_physics(solver, e)

        for v, b in remaining:
            try:
                k = v / PR_FIXED
                air.viscosity.value = v
                air.thermal_conductivity.value = k
                air.therm_exp_coeff.value = b

                solver.solution.initialization.hybrid_initialize()
                nu, total_iters, converged = iterate_until_stable(solver, k)
                q_avg = nu * k
                pr, re, gr, ra, ri = derived_nondim(v, b)

                csv_writer.writerow(dict(
                    case_num=case_num, r_star=r_star, epsilon=e, e=e,
                    v=v, b=b, omega=OMEGA_FIXED, q_avg=q_avg, k=k, nu=nu,
                    pr=pr, re=re, gr=gr, ra=ra, ri=ri,
                    converged=converged, total_iters=total_iters,
                ))
                csv_file.flush()
                ok_count += 1
                tag = "" if converged else "  [NOT STABLE at MAX_ITER]"
                print(f"  v={v} b={b}: Nu={nu:.4f}  (Re={re:.3g} Ra={ra:.3g} Ri={ri:.3g}) "
                      f"iters={total_iters}{tag}")
            except Exception as ex:
                fail_count += 1
                print(f"  FAILED v={v}, b={b}: {ex}")
    finally:
        solver.exit()

    print(f"Case {case_num:02d} done: {ok_count} ok, {fail_count} failed.")
    return ok_count, fail_count


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    file_exists = os.path.exists(OUT_CSV)
    done_keys = load_done_keys()
    if done_keys:
        print(f"Resuming: {len(done_keys)} (case, v, b, omega) combos already completed.")

    total_ok = 0
    total_fail = 0
    cases = CASES if DEBUG_LIMIT_GEOMETRIES is None else CASES[:DEBUG_LIMIT_GEOMETRIES]

    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
            f.flush()

        for case_num, inner_diam, e in cases:
            try:
                ok, fail = run_geometry(case_num, inner_diam, e, done_keys, writer, f)
            except Exception as ex:
                print(f"Case {case_num:02d}: uncaught error, skipping to next geometry: {ex}")
                ok, fail = 0, 0
            total_ok += ok
            total_fail += fail

    print("\n" + "=" * 60)
    print("  PRODUCTION BATCH v2 (Re/Ri/omega) COMPLETE")
    print(f"  Successful runs : {total_ok}")
    print(f"  Failed runs     : {total_fail}")
    print(f"  Output          : {OUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
