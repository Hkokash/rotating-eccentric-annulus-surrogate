# =============================================================================
# Verification pass BEFORE running the new 10,500-case journal at scale.
# Checks two things empirically, not by assumption:
#   1. What thermal boundary conditions (if any) are inherited from the raw
#      mesh file case_01.cas.h5, before any of the new journal's commands run.
#   2. What the exact `define/materials/change-create` TUI command from the
#      new journal actually assigns to each material property, by executing
#      it verbatim (with test values v=0.1, b=0.01) and reading back the
#      resulting material property values via the settings API (which
#      reflects true solver state regardless of how it was set).
# =============================================================================

import ansys.fluent.core as pyfluent

MESH = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\meshes\case_01.cas.h5"

TEST_V = 0.1
TEST_B = 0.01

CHANGE_CREATE_CMD = (
    f"define/materials/change-create air air yes boussinesq 1 yes constant 1 "
    f"yes constant 1 yes constant {TEST_V} no yes {TEST_B}"
)


def main():
    solver = pyfluent.launch_fluent(
        mode=pyfluent.FluentMode.SOLVER, precision=pyfluent.Precision.DOUBLE,
        processor_count=2, dimension=2,
    )
    try:
        solver.file.read_case(file_name=MESH)

        def safe_state(obj):
            try:
                return obj.get_state()
            except Exception as e:
                return f"<error: {e}>"

        print("=== STEP 1: thermal BCs inherited from the raw mesh file (before any new commands) ===")
        for wall in ["inner_wall", "outer_wall"]:
            th = solver.setup.boundary_conditions.wall[wall].thermal
            print(f"  {wall}: full thermal state = {safe_state(th)}")

        print("\n=== STEP 1b: motion/rotation BC inherited from the raw mesh file ===")
        for wall in ["inner_wall", "outer_wall"]:
            mom = solver.setup.boundary_conditions.wall[wall].momentum
            try:
                motion = mom.wall_motion
            except Exception as e:
                motion = f"<error: {e}>"
            print(f"  {wall}: wall_motion={motion!r}")

        print(f"\n=== STEP 2: material properties BEFORE the change-create command ===")
        air_before = solver.setup.materials.fluid["air"]
        for prop in ["density", "specific_heat", "thermal_conductivity", "viscosity", "therm_exp_coeff"]:
            try:
                obj = getattr(air_before, prop)
                print(f"  {prop}: {safe_state(obj)}")
            except Exception as e:
                print(f"  {prop}: <error: {e}>")

        print(f"\n=== STEP 3: executing the exact change-create TUI command (v={TEST_V}, b={TEST_B}) ===")
        print(f"Command string: {CHANGE_CREATE_CMD}")
        try:
            solver.tui.define.materials.change_create(
                "air", "air", "yes", "boussinesq", "1", "yes", "constant", "1",
                "yes", "constant", "1", "yes", "constant", str(TEST_V), "no", "yes", str(TEST_B),
            )
            print("  executed via solver.tui.define.materials.change_create(...)")
        except Exception as e:
            print(f"  solver.tui approach failed: {e}")
            print("  trying raw scheme string execution instead...")
            solver.scheme_eval.exec(('(ti-menu-load-string "' + CHANGE_CREATE_CMD + '")',))

        print(f"\n=== STEP 4: material properties AFTER the change-create command ===")
        air_after = solver.setup.materials.fluid["air"]
        for prop in ["density", "specific_heat", "thermal_conductivity", "viscosity", "therm_exp_coeff"]:
            try:
                obj = getattr(air_after, prop)
                print(f"  {prop}: {safe_state(obj)}")
            except Exception as e:
                print(f"  {prop}: <error: {e}>")
        print(f"\n  Expecting to find TEST_V={TEST_V} on exactly one property and TEST_B={TEST_B} on exactly one property.")

        print("\n=== STEP 5: thermal BCs NOW that the energy equation is active (post change-create) ===")
        print(f"  Energy equation enabled: {safe_state(solver.setup.models.energy.enabled)}")
        for wall in ["inner_wall", "outer_wall"]:
            th = solver.setup.boundary_conditions.wall[wall].thermal
            print(f"  {wall}: full thermal state = {safe_state(th)}")

        print("\n=== STEP 6: full momentum/motion state for both walls (post change-create) ===")
        for wall in ["inner_wall", "outer_wall"]:
            mom = solver.setup.boundary_conditions.wall[wall].momentum
            print(f"  {wall}: full momentum state = {safe_state(mom)}")

    finally:
        solver.exit()


if __name__ == "__main__":
    main()
