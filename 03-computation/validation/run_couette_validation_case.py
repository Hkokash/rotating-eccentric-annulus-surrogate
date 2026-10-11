# =============================================================================
# ONE new, small CFD case: pure circular Couette flow (isothermal, zero
# buoyancy, inner cylinder rotating, outer stationary, concentric). This is
# the only new simulation added during the entire validation effort -- every
# other check used data already in the production dataset.
#
# Purpose: compare against the EXACT analytical solution derived and verified
# in this same session (see decisions.log 2026-08-29 Couette entry):
#   v_theta(r) = A*r + B/r,  A = Omega*Ri^2/(Ri^2-Ro^2),  B = -A*Ro^2
#   psi(r) = -integral_{Ri}^{r} v_theta dr'
# For R*=0.5 (case_27, Ri=0.25, Ro=0.5), Omega=10: psi(Ro) = -0.26512 (exact).
#
# Buoyancy is switched off via gravity.enable=False -- this decouples momentum
# from temperature entirely (Boussinesq buoyancy term vanishes with g=0), so
# the resulting velocity field is pure viscous Couette flow, independent of
# which Ra/mu/k values happen to be set (a further robustness point: this
# problem has no viscosity dependence at all in its steady-state solution).
# =============================================================================

import os
import ansys.fluent.core as pyfluent

MESH_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\meshes"
OUT_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\validation\couette_check"
CASE_NUM = 27  # R*=0.5, e=0 (concentric)
OMEGA = 10.0
MU, K = 0.01, 0.01  # arbitrary -- Couette flow steady solution is viscosity-independent


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    case_file = os.path.join(MESH_DIR, f"case_{CASE_NUM:02d}.cas.h5")

    solver = pyfluent.launch_fluent(
        mode=pyfluent.FluentMode.SOLVER,
        precision=pyfluent.Precision.DOUBLE,
        processor_count=2,
        dimension=2,
    )
    try:
        solver.file.read_case(file_name=case_file)

        solver.setup.models.energy.enabled = True
        solver.setup.models.viscous.model = "laminar"
        solver.solution.controls.limits.min_temperature = 0
        solver.solution.controls.limits.max_temperature = 100

        # Buoyancy OFF -- the one deliberate change from the production setup
        solver.setup.general.operating_conditions.gravity.enable = False

        solver.setup.general.operating_conditions.operating_pressure = 0
        # operating_temperature is only a settable field when buoyancy/gravity
        # is enabled (it feeds the Boussinesq body-force term); irrelevant and
        # inactive here since gravity is off.

        air = solver.setup.materials.fluid["air"]
        air.density.option = "boussinesq"
        air.density.value = 1
        air.specific_heat.value = 1
        air.therm_exp_coeff.value = 1
        air.viscosity.value = MU
        air.thermal_conductivity.value = K

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
        inner_mom.rotation_axis_origin = [0, 0]  # concentric, e=0
        inner_mom.rotation_speed = OMEGA

        solver.solution.methods.p_v_coupling.flow_scheme = "SIMPLE"
        solver.solution.methods.spatial_discretization.discretization_scheme = {"pressure": "presto!"}

        solver.solution.initialization.hybrid_initialize()
        solver.solution.run_calculation.iterate(iter_count=1000)

        out_file = os.path.join(OUT_DIR, f"couette_case{CASE_NUM:02d}_omega{int(OMEGA)}.csv")
        solver.file.export.ascii(
            file_name=out_file,
            surface_name_list=["interior:fluid"],
            delimiter="comma",
            quantities=["temperature", "x-velocity", "y-velocity"],
            location="node",
        )
        print(f"Exported: {out_file}")
    finally:
        solver.exit()


if __name__ == "__main__":
    main()
