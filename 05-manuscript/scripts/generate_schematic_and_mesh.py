# =============================================================================
# Domain schematic (drawn) and real mesh visualization (extracted from the
# actual Fluent mesh file, case_16 = R*=0.3, epsilon=0, the GCI validation
# case) - both needed for the manuscript per reviewer-style feedback that a
# geometry/mesh figure is expected in a CFD-based paper.
# =============================================================================

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.lines as mlines

OUT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\05-manuscript\figures"
MESH_DIR = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\meshes"

plt.rcParams.update({"font.size": 12, "figure.dpi": 200, "font.family": "serif"})


def fig_schematic(variant="A", color=False, out_name="fig_schematic"):
    """variant 'A': radius lines from center, label just past the boundary.
    variant 'B': short dimension tick from the boundary itself, label at the tip
    (label position independent of center, reads more like an engineering drawing)."""
    Ro, Ri, e = 0.5, 0.22, 0.14
    outer_color = "#0072B2" if color else "black"   # Okabe-Ito blue -- cold
    inner_color = "#D55E00" if color else "black"   # Okabe-Ito vermillion -- hot
    fig, ax = plt.subplots(figsize=(7.8, 8.4))

    outer = Circle((0, 0), Ro, fill=False, edgecolor=outer_color, linewidth=2.4)
    inner = Circle((0, -e), Ri, fill=False, edgecolor=inner_color, linewidth=2.4, hatch="/////", zorder=3)
    ax.add_patch(outer)
    ax.add_patch(inner)

    # Distinguishable center markers: outer center = small open circle, inner center = cross.
    ax.plot(0, 0, marker="o", markersize=7, markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=1.6, zorder=5)
    ax.plot(0, -e, marker="x", markersize=10, markeredgecolor="black", markeredgewidth=2.0, zorder=5)
    key_x, key_y = -Ro - 0.30, Ro - 0.05
    ax.plot(key_x, key_y, marker="o", markersize=7, markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=1.6, zorder=5)
    ax.text(key_x + 0.055, key_y - 0.012, "$O_o$ (outer center)", fontsize=9, va="center")
    ax.plot(key_x, key_y - 0.09, marker="x", markersize=9, markeredgecolor="black", markeredgewidth=1.8, zorder=5)
    ax.text(key_x + 0.055, key_y - 0.09 - 0.012, "$O_i$ (inner center)", fontsize=9, va="center")

    ang_o = np.radians(35)
    ang_i = np.radians(-30)

    if variant == "A":
        # Radius line from center to the boundary, label placed just OUTSIDE the
        # circle (not at a fraction of the radius, which for R_i previously landed
        # the label on top of the hatching and made it unreadable).
        ax.plot([0, Ro * np.cos(ang_o)], [0, Ro * np.sin(ang_o)], "k--", linewidth=1)
        ax.text(1.14 * Ro * np.cos(ang_o), 1.14 * Ro * np.sin(ang_o), r"$R_o$",
                fontsize=14, ha="center", va="center", color=outer_color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

        ax.plot([0, Ri * np.cos(ang_i)], [-e, -e + Ri * np.sin(ang_i)], "k--", linewidth=1)
        ax.text(1.20 * Ri * np.cos(ang_i), -e + 1.20 * Ri * np.sin(ang_i), r"$R_i$",
                fontsize=13, ha="center", va="center", color=inner_color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    else:
        # Short dimension tick starting AT the boundary and pointing outward, label
        # at the tip -- standard engineering-drawing radius-callout style.
        ax.annotate("", xy=(1.16 * Ro * np.cos(ang_o), 1.16 * Ro * np.sin(ang_o)),
                    xytext=(Ro * np.cos(ang_o), Ro * np.sin(ang_o)),
                    arrowprops=dict(arrowstyle="-", lw=1.1, color=outer_color))
        ax.text(1.22 * Ro * np.cos(ang_o), 1.22 * Ro * np.sin(ang_o), r"$R_o$",
                fontsize=14, ha="center", va="center", color=outer_color)

        ax.annotate("", xy=(1.6 * Ri * np.cos(ang_i), -e + 1.6 * Ri * np.sin(ang_i)),
                    xytext=(Ri * np.cos(ang_i), -e + Ri * np.sin(ang_i)),
                    arrowprops=dict(arrowstyle="-", lw=1.1, color=inner_color))
        ax.text(1.7 * Ri * np.cos(ang_i), -e + 1.7 * Ri * np.sin(ang_i), r"$R_i$",
                fontsize=13, ha="center", va="center", color=inner_color)

    # eccentricity e: vertical double-arrow to the LEFT of both centers, clear of the circle
    xe = -Ri - 0.08
    ax.annotate("", xy=(xe, -e), xytext=(xe, 0), arrowprops=dict(arrowstyle="<->", lw=1.3, color="black"))
    ax.plot([xe, 0], [0, 0], color="black", linewidth=0.6, linestyle=":")
    ax.plot([xe, 0], [-e, -e], color="black", linewidth=0.6, linestyle=":")
    ax.text(xe - 0.075, -e / 2, r"$e$", fontsize=13, color="black", va="center")

    # omega_i: label placed above the inner cylinder, with the curved rotation-direction
    # arrow drawn below the label (between the label and the inner wall).
    theta = np.linspace(0.5, 2.3, 30)
    r_arrow = Ri * 1.15
    arc_y = -e + r_arrow * np.sin(theta) * 0.9 + 0.02
    ax.plot(r_arrow * np.cos(theta), arc_y, color=inner_color, linewidth=1.6)
    ax.annotate("", xy=(r_arrow * np.cos(theta[0]), arc_y[0]),
                xytext=(r_arrow * np.cos(theta[1]), arc_y[1]),
                arrowprops=dict(arrowstyle="-|>", color=inner_color, lw=1.6, mutation_scale=18))
    ax.text(0.02, arc_y.max() + 0.04, r"$\omega_i$", fontsize=15, color=inner_color, ha="center", va="bottom")

    # Inner/outer wall thermal condition labels, placed OUTSIDE the geometry with leader lines
    ax.text(0, Ro + 0.13, r"$\theta_o = 0$ (stationary, cooled)", ha="center", fontsize=11, color=outer_color)
    ax.annotate("", xy=(0, Ro), xytext=(0, Ro + 0.10), arrowprops=dict(arrowstyle="-", lw=0.8, color=outer_color))

    # Inner-wall label: single straight leader line from the inner (hatched) circle's own
    # boundary, well clear of R_i's own leader, out to the label text below the domain.
    # Anchor angle chosen so the line points directly at the label with no bend.
    ang_wall = np.radians(-56)
    inner_wall_pt = (Ri * np.cos(ang_wall), -e + Ri * np.sin(ang_wall))
    leader_end = (Ro * 0.62, -Ro - 0.045)
    ax.text(Ro * 0.62, -Ro - 0.09, r"$\theta_i = 1$ (rotating, heated)", ha="center", fontsize=11, color=inner_color)
    ax.plot([inner_wall_pt[0], leader_end[0]], [inner_wall_pt[1], leader_end[1]],
            color=inner_color, linewidth=0.8, zorder=2)

    # Local x-y axes, drawn in the lower-left corner, clear of all other annotations
    ax_origin = (-Ro - 0.02, -Ro - 0.02)
    ax.annotate("", xy=(ax_origin[0] + 0.11, ax_origin[1]), xytext=ax_origin, arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(ax_origin[0], ax_origin[1] + 0.11), xytext=ax_origin, arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(ax_origin[0] + 0.13, ax_origin[1] - 0.015, "$x$", fontsize=12)
    ax.text(ax_origin[0] - 0.03, ax_origin[1] + 0.13, "$y$", fontsize=12)

    # gravity vector, to the right of the outer circle, vertically centered, pointing
    # down (-y), consistent with ĝ=(0,-1); solid black, larger
    xg = Ro + 0.30
    ax.annotate("", xy=(xg, -Ro * 0.55), xytext=(xg, Ro * 0.55),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2, mutation_scale=22))
    ax.text(xg + 0.08, 0, "$g$", fontsize=15, color="black", va="center")

    ax.set_xlim(-Ro - 0.45, Ro + 0.65)
    ax.set_ylim(-Ro - 0.60, Ro + 0.35)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(f"{OUT}/{out_name}.png", bbox_inches="tight")
    plt.close(fig)


def fig_mesh():
    case_num = 16
    R_I = 0.15  # case_16 inner-wall radius (R*=0.3, concentric)
    with h5py.File(f"{MESH_DIR}/case_{case_num:02d}.cas.h5", "r") as f:
        node_zone_key = list(f["meshes/1/nodes/coords"].keys())[0]
        coords = f[f"meshes/1/nodes/coords/{node_zone_key}"][:]
        face_nodes = f["meshes/1/faces/nodes/1/nodes"][:].reshape(-1, 2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    segs = np.stack([coords[face_nodes[:, 0] - 1], coords[face_nodes[:, 1] - 1]], axis=1)
    from matplotlib.collections import LineCollection
    from matplotlib.patches import ConnectionPatch

    # The raw mesh file contains an extra, unintended disk of cells filling
    # the entire inner cylinder (a legacy geometry-construction artifact,
    # present since the original mesh generation: Fluent's own mesh/check
    # reports total cell area 0.7849 m^2 vs the true annulus area
    # pi*(0.5^2-0.15^2)=0.7147 m^2, the 0.070 m^2 difference matching a full
    # disk of radius 0.15 to within 1%). It is walled off from the real
    # annulus by a genuine wall+shadow pair (inner_wall/part-internal-shadow)
    # and does not affect CFD results -- decoupled by the explicit one-sided
    # Temperature BC on inner_wall, corroborated by the Kuehn & Goldstein
    # validation elsewhere in this manuscript -- but it has no business appearing in a
    # figure of the fluid domain, so it is excluded here by radius.
    seg_r_min = np.minimum(
        np.hypot(segs[:, 0, 0], segs[:, 0, 1]), np.hypot(segs[:, 1, 0], segs[:, 1, 1])
    )
    keep = seg_r_min >= R_I - 1e-6
    segs = segs[keep]

    lc = LineCollection(segs, linewidths=0.3, colors="black")
    axes[0].add_collection(lc)
    axes[0].set_xlim(-0.55, 0.55)
    axes[0].set_ylim(-0.55, 0.55)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, fontsize=13,
                 fontweight="bold", va="top", bbox=dict(boxstyle="round", fc="white", ec="black", lw=0.8))

    # Zoom spans from the inner wall out to the outer wall (this box's top-right
    # corner sits exactly at r=0.5, the outer radius), showing the inflation
    # layers at the inner wall, the bulk triangulation, and the outer boundary
    # together in one view.
    rect_x, rect_y, rect_w, rect_h = 0.15, -0.05, 0.25, 0.35
    axes[0].add_patch(plt.Rectangle((rect_x, rect_y), rect_w, rect_h, fill=False,
                                     edgecolor="black", linewidth=1.3))

    lc2 = LineCollection(segs, linewidths=0.5, colors="black")
    axes[1].add_collection(lc2)
    axes[1].set_xlim(rect_x, rect_x + rect_w)
    axes[1].set_ylim(rect_y, rect_y + rect_h)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$y$")
    axes[1].text(0.04, 0.96, "(b)", transform=axes[1].transAxes, fontsize=13,
                 fontweight="bold", va="top", bbox=dict(boxstyle="round", fc="white", ec="black", lw=0.8))
    for spine in axes[1].spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.3)

    fig.tight_layout()

    # Connector lines linking the zoom rectangle in (a) to the detail panel (b),
    # drawn last so they sit on top in figure coordinates.
    for corner_a, corner_b in [((rect_x + rect_w, rect_y + rect_h), (0, 1)),
                                ((rect_x + rect_w, rect_y), (0, 0))]:
        con = ConnectionPatch(xyA=corner_a, coordsA=axes[0].transData,
                               xyB=corner_b, coordsB=axes[1].transAxes,
                               color="black", linewidth=0.9)
        fig.add_artist(con)

    n_nodes = len(coords)
    n_edges = len(face_nodes)
    fig.savefig(f"{OUT}/fig_mesh.png", bbox_inches="tight")
    print(f"Mesh: {n_nodes} nodes, {n_edges} faces (edges in 2D)")
    plt.close(fig)


if __name__ == "__main__":
    fig_schematic(variant="A", color=False, out_name="fig_schematic_A_bw")
    fig_schematic(variant="A", color=True, out_name="fig_schematic_A_color")
    fig_schematic(variant="B", color=False, out_name="fig_schematic_B_bw")
    fig_schematic(variant="B", color=True, out_name="fig_schematic_B_color")
    fig_schematic(variant="A", color=False, out_name="fig_schematic")  # default used by the manuscript
    fig_mesh()
    print("Wrote fig_schematic.png and fig_mesh.png")
