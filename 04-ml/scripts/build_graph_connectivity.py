# =============================================================================
# Extract TRUE mesh connectivity for the GNN field surrogate (the Research
# Brief's stated fallback after DeepONet underperformed the per-node MLP
# baseline - see .forge/decisions.log, 2026-08-30).
#
# The mesh (case_NN.cas.h5) does not change across (Re, Ri) - only the
# solved field does - so connectivity is extracted ONCE PER GEOMETRY (35
# times), not once per case (5,460 times).
#
# Fluent's 2D mesh stores faces as exactly 2-node entries (a face IS an edge
# in 2D), at meshes/1/faces/nodes/1/nodes, using the mesh file's own raw node
# numbering (meshes/1/nodes/coords/<zone>, 1-indexed: coords[id-1]). This raw
# numbering is NOT the same as the field-export CSV's "nodenumber" column
# (verified: coordinates match exactly between the two but at different
# indices - Fluent renumbers nodes for the CSV field export). Edges are
# therefore re-expressed in terms of CSV row order via an exact-coordinate
# lookup (floats are bit-identical between the mesh file and the field CSV,
# both being read from the same underlying solve).
#
# Updated 2026-09-15 for the Re/Ri redesign: paths point at
# field_data_v2_re_ri (same 35 .cas.h5 mesh files as before, unchanged --
# only the physics/sweep design changed -- so this would in principle
# reproduce the same connectivity as the original run; regenerated from the
# new field data anyway for self-consistent provenance rather than reusing
# the old output). Output directory also renamed to _v2 to avoid ambiguity.
# =============================================================================

import os
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MESH_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\meshes"
FIELD_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\field_data_v2_re_ri"
MANIFEST = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\field_data_v2_re_ri\manifest.csv"
OUT_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\04-ml\data\graph_connectivity_v2"


def extract_case_edges(case_num):
    mesh_path = os.path.join(MESH_DIR, f"case_{case_num:02d}.cas.h5")
    with h5py.File(mesh_path, "r") as f:
        node_zone_key = list(f["meshes/1/nodes/coords"].keys())[0]
        coords = f[f"meshes/1/nodes/coords/{node_zone_key}"][:]
        nnodes_per_face = f["meshes/1/faces/nodes/1/nnodes"][:]
        assert np.all(nnodes_per_face == 2), "expected a 2D mesh (all faces = 2-node edges)"
        face_nodes = f["meshes/1/faces/nodes/1/nodes"][:].reshape(-1, 2)  # raw 1-indexed node ids

    manifest = pd.read_csv(MANIFEST)
    row0 = manifest[manifest["case_num"] == case_num].iloc[0]
    sample_csv = os.path.join(FIELD_DIR, f"case_{case_num:02d}", f"re{row0.re:.4g}_ri{row0.ri:.4g}.csv")
    field = pd.read_csv(sample_csv)
    field.columns = [c.strip() for c in field.columns]
    csv_xy = field[["x-coordinate", "y-coordinate"]].values

    tree = cKDTree(csv_xy)
    dist, row_idx = tree.query(coords, k=1)
    TOL = 1e-8
    raw_id_to_row = np.where(dist < TOL, row_idx, -1).astype(np.int64)
    n_missing = int((raw_id_to_row == -1).sum())

    edges = []
    n_dropped = 0
    for n1, n2 in face_nodes:
        r1 = raw_id_to_row[n1 - 1]
        r2 = raw_id_to_row[n2 - 1]
        if r1 == -1 or r2 == -1:
            n_dropped += 1
            continue
        edges.append((r1, r2))

    edges = np.array(edges, dtype=np.int64).T  # shape (2, n_edges)
    edges_undirected = np.concatenate([edges, edges[[1, 0]]], axis=1)

    return edges_undirected, len(csv_xy), n_missing, n_dropped


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for case_num in range(1, 36):
        edges, n_nodes, n_missing, n_dropped = extract_case_edges(case_num)
        out_path = os.path.join(OUT_DIR, f"case_{case_num:02d}_edges.npz")
        np.savez(out_path, edge_index=edges, n_nodes=n_nodes)
        print(f"case_{case_num:02d}: n_nodes={n_nodes} n_edges(undirected)={edges.shape[1]} "
              f"raw_nodes_unmatched={n_missing} face_edges_dropped={n_dropped}")


if __name__ == "__main__":
    main()
