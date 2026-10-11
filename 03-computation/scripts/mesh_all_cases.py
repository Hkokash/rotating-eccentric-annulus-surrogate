# =============================================================================
# ECCENTRIC ANNULUS 2D MESH GENERATOR
# Meshes all 35 geometry cases (case_NN.scdocx) via Fluent Meshing's legacy
# 2D Meshing workflow, and writes solver-ready case files (case_NN.cas.h5)
# with named zones: outer_wall, inner_wall, fluid.
#
# Pipeline (validated on case_01 before this batch run):
#   Meshing session (legacy 2D workflow):
#     load_cad_geometry_2d -> update_boundaries_2d -> define_global_sizing_2d
#     -> add_2d_boundary_layers -> generate_initial_surface_mesh -> write_2d_mesh
#   IMPORTANT: no zone renaming/retyping during the meshing phase -- doing so
#   before write_2d_mesh crashes the Fluent Meshing backend (SIGSEGV), because
#   the 2D cell region is internally represented as a face zone and mutating
#   it corrupts the mesh database.
#
#   Solver session (fresh session, dimension=2):
#     read_mesh -> identify outer/inner wall by boundary perimeter (outer
#     diameter is fixed at 1.0 and always larger than any inner diameter used,
#     so this is unambiguous) -> rename to outer_wall/inner_wall (a wall on a
#     hole boundary can come through as a shadowed pair, e.g. "part-internal"
#     / "part-internal-shadow" -- the "-shadow" side is ignored, it mirrors
#     its non-shadow counterpart) -> ensure fluid cell zone is named "fluid"
#     -> write_case
# =============================================================================

import os
import traceback

import ansys.fluent.core as pyfluent

from generate_geometry import CASES, OUTPUT_PATH

MESH_DIR = OUTPUT_PATH  # C:\Users\ADMIN\Documents\AnnulusProject\meshes
INTERMEDIATE_DIR = os.path.join(MESH_DIR, "_intermediate_msh")
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

# Moderately refined mesh settings: finer than a quick sanity-check mesh, but
# not pushed to the extreme fine end (that mesh generates just as fast per
# Fluent's own internal timing -- the earlier slowness came from debugging
# API/workflow bugs across many Fluent launches, not from mesh fineness).
GLOBAL_SIZING = dict(
    curvature_normal_angle=15,
    max_size=0.03,
    min_size=0.005,
    size_functions="Curvature",
)
BOUNDARY_LAYERS = dict(
    control_name="inflation_1",
    number_of_layers=8,
    offset_method_type="uniform",
    first_layer_height=0.0015,
)


def clean_file(path):
    if os.path.exists(path):
        os.remove(path)


def mesh_case(case_num):
    case_file = os.path.join(MESH_DIR, f"case_{case_num:02d}.scdocx")
    mesh_file = os.path.join(INTERMEDIATE_DIR, f"case_{case_num:02d}.msh.h5")
    out_file = os.path.join(MESH_DIR, f"case_{case_num:02d}.cas.h5")
    clean_file(mesh_file)
    clean_file(out_file)

    # --- Meshing session ---
    meshing_session = pyfluent.launch_fluent(
        mode=pyfluent.FluentMode.MESHING,
        precision=pyfluent.Precision.DOUBLE,
        processor_count=2,
    )
    try:
        two_dim = meshing_session.two_dimensional_meshing(legacy=True)

        load_cad = two_dim.load_cad_geometry_2d
        load_cad.file_name = case_file
        load_cad.length_unit = "m"
        load_cad.refaceting.refacet = False
        load_cad()

        update_boundaries = two_dim.update_boundaries_2d
        update_boundaries.selection_type = "zone"
        update_boundaries()

        global_sizing = two_dim.define_global_sizing_2d
        for attr, value in GLOBAL_SIZING.items():
            setattr(global_sizing, attr, value)
        global_sizing()

        add_boundary_layers = two_dim.add_2d_boundary_layers
        add_boundary_layers.add_child = "yes"
        for attr, value in BOUNDARY_LAYERS.items():
            setattr(add_boundary_layers, attr, value)
        add_boundary_layers.add_child_and_update(defer_update=False)

        two_dim.generate_initial_surface_mesh()

        write_task = two_dim.write_2d_mesh
        write_task.file_name = mesh_file
        write_task()
    finally:
        meshing_session.exit()

    # --- Solver session: identify/rename zones, write final case file ---
    solver = pyfluent.launch_fluent(
        mode=pyfluent.FluentMode.SOLVER,
        precision=pyfluent.Precision.DOUBLE,
        processor_count=2,
        dimension=2,
    )
    try:
        solver.file.read_mesh(file_name=mesh_file)

        wall_zones = list(solver.setup.boundary_conditions.wall.keys())
        non_shadow_walls = [w for w in wall_zones if not w.endswith("-shadow")]
        if len(non_shadow_walls) != 2:
            raise RuntimeError(f"expected 2 non-shadow wall zones, got {non_shadow_walls}")

        perimeters = {wz: solver.fields.reduction.area(locations=[wz]) for wz in non_shadow_walls}
        outer_wall_name = max(perimeters, key=perimeters.get)
        inner_wall_name = min(perimeters, key=perimeters.get)

        solver.setup.boundary_conditions.wall.rename(outer_wall_name, "outer_wall")
        solver.setup.boundary_conditions.wall.rename(inner_wall_name, "inner_wall")

        fluid_zones = list(solver.setup.cell_zone_conditions.fluid.keys())
        if fluid_zones and fluid_zones[0] != "fluid":
            solver.setup.cell_zone_conditions.fluid.rename(fluid_zones[0], "fluid")

        solver.file.write_case(file_name=out_file)
    finally:
        solver.exit()

    return out_file


def main():
    print("=" * 60)
    print("  ECCENTRIC ANNULUS 2D MESH GENERATOR")
    print(f"  Total cases: {len(CASES)}")
    print("=" * 60)

    successful = []
    failed = []
    for case_num, inner_diam, e in CASES:
        print(f"\n--- Case {case_num:02d} (inner_diam={inner_diam}, e={e}) ---")
        try:
            out_file = mesh_case(case_num)
            print(f"    OK -> {out_file}")
            successful.append(case_num)
        except Exception:
            print(f"    ERROR in case {case_num:02d}:")
            traceback.print_exc()
            failed.append(case_num)

    print("\n" + "=" * 60)
    print("  MESH GENERATION COMPLETE")
    print(f"  Successful : {len(successful)} cases")
    print(f"  Failed     : {len(failed)} cases")
    if failed:
        print(f"  Failed cases: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
