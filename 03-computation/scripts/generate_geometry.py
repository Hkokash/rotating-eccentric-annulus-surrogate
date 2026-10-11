# =============================================================================
# ECCENTRIC ANNULUS GEOMETRY GENERATOR
# Generates all 35 geometry cases for the ML surrogate study
# Uses ANSYS SpaceClaim via ansys-geometry-core
# =============================================================================

import os
import time
from ansys.geometry.core import launch_modeler
from ansys.geometry.core.designer import DesignFileFormat
from ansys.geometry.core.math import Point2D, ZERO_POINT2D
from ansys.geometry.core.sketch import Sketch

OUTPUT_PATH = r"C:\Users\ADMIN\Documents\AnnulusProject\meshes"

OUTER_DIAMETER = 1.0
OUTER_RADIUS   = 0.5

CASES = [
    ( 1,  0.10, 0.000),
    ( 2,  0.10, 0.045),
    ( 3,  0.10, 0.090),
    ( 4,  0.10, 0.135),
    ( 5,  0.10, 0.180),
    ( 6,  0.10, 0.225),
    ( 7,  0.10, 0.270),
    ( 8,  0.10, 0.315),
    ( 9,  0.20, 0.000),
    (10,  0.20, 0.040),
    (11,  0.20, 0.080),
    (12,  0.20, 0.120),
    (13,  0.20, 0.160),
    (14,  0.20, 0.200),
    (15,  0.20, 0.240),
    (16,  0.30, 0.000),
    (17,  0.30, 0.035),
    (18,  0.30, 0.070),
    (19,  0.30, 0.105),
    (20,  0.30, 0.140),
    (21,  0.30, 0.175),
    (22,  0.40, 0.000),
    (23,  0.40, 0.024),
    (24,  0.40, 0.048),
    (25,  0.40, 0.072),
    (26,  0.40, 0.096),
    (27,  0.50, 0.000),
    (28,  0.50, 0.020),
    (29,  0.50, 0.040),
    (30,  0.50, 0.060),
    (31,  0.50, 0.080),
    (32,  0.60, 0.000),
    (33,  0.60, 0.016),
    (34,  0.60, 0.032),
    (35,  0.60, 0.048),
]

def create_output_folder():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(f"Created output folder: {OUTPUT_PATH}")
    else:
        print(f"Output folder exists: {OUTPUT_PATH}")

def generate_case(design, case_num, inner_diam, e):
    inner_radius = inner_diam / 2.0
    print(f"\n--- Case {case_num:02d} ---")
    print(f"    Inner diameter : {inner_diam}")
    print(f"    Eccentricity e : {e}")
    print(f"    Inner center   : (0, {-e})")

    for ns in list(design.named_selections):
        design.delete_named_selection(ns)
    for body in design.get_all_bodies():
        design.delete_body(body)

    sketch = Sketch()
    sketch.circle(center=ZERO_POINT2D, radius=OUTER_RADIUS)
    sketch.circle(center=Point2D([0, -e]), radius=inner_radius)
    body = design.create_surface(
        name=f"annulus_case_{case_num:02d}",
        sketch=sketch,
    )

    # Tag the domain face as "fluid" -- Fluent's 2D meshing workflow picks this
    # up directly as the resulting cell zone's name/type. Per-edge named
    # selections (outer_wall/inner_wall) are NOT used here: Fluent's 2D CAD
    # import derives boundary-edge zone names from the adjacent face's label,
    # so edge-level tags get silently overridden anyway. Outer/inner wall
    # identification is instead done at meshing time (see meshing script).
    design.create_named_selection("fluid", faces=body.faces)

    print(f"    Geometry created successfully")
    return body

def export_case(design, case_num):
    filename = f"case_{case_num:02d}.scdocx"
    filepath = os.path.join(OUTPUT_PATH, filename)
    design.download(filepath, format=DesignFileFormat.SCDOCX)
    print(f"    Exported: {filename}")
    return filepath

def main():
    print("=" * 60)
    print("  ECCENTRIC ANNULUS GEOMETRY GENERATOR")
    print(f"  Total cases: {len(CASES)}")
    print("=" * 60)
    create_output_folder()
    print("\nLaunching ANSYS SpaceClaim...")
    modeler = launch_modeler()
    print("SpaceClaim connected successfully.")
    design = modeler.create_design("EccentricAnnulus")
    successful = []
    failed = []
    for case_num, inner_diam, e in CASES:
        try:
            generate_case(design, case_num, inner_diam, e)
            export_case(design, case_num)
            successful.append(case_num)
            time.sleep(1)
        except Exception as ex:
            print(f"    ERROR in case {case_num}: {ex}")
            failed.append(case_num)
    modeler.close()
    print("\n" + "=" * 60)
    print("  GEOMETRY GENERATION COMPLETE")
    print(f"  Successful : {len(successful)} cases")
    print(f"  Failed     : {len(failed)} cases")
    if failed:
        print(f"  Failed cases: {failed}")
    print("=" * 60)

if __name__ == "__main__":
    main()
