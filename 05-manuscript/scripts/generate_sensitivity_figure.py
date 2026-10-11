# =============================================================================
# Physical parametric-effects figure, computed directly from the raw CFD
# dataset (nu_dataset_v2_re_ri.csv) -- NOT from the ML surrogate. Distinct in
# kind from the SHAP analysis in ml_models.tex/results.tex: SHAP measures
# what the trained XGBoost model learned; this figure measures what the CFD
# solver actually produced, at one-at-a-time parameter slices through the
# dataset. Four panels: Nu vs Re (fixed geometry, three Ri curves), Nu vs Ri
# (fixed geometry, three Re curves), Nu vs R* (six concentric geometries,
# fixed Re/Ri), Nu vs epsilon (one R* group's full eccentricity sweep, fixed
# Re/Ri).
# =============================================================================

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\03-computation\data\results\nu_dataset_v2_re_ri.csv"
OUT = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\05-manuscript\figures\fig_sensitivity.png"

OKABE_ITO = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "sky": "#56B4E9", "purple": "#CC79A7",
}

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def main():
    df = pd.read_csv(DATA)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) Nu vs Re at case 16 (R*=0.3, eps=0), three Ri curves
    ax = axes[0, 0]
    case16 = df[df["case_num"] == 16]
    for ri, color in zip([0.01, 1.0, 10.0], [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["vermillion"]]):
        sub = case16[abs(case16["ri"] - ri) < 1e-6].sort_values("re")
        ax.plot(sub["re"], sub["nu"], "o-", color=color, label=f"$Ri={ri:g}$")
    ax.set_xscale("log")
    ax.set_xlabel("$Re$")
    ax.set_ylabel("$Nu$")
    ax.legend(fontsize=9)
    ax.text(0.03, 0.95, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    # (b) Nu vs Ri at case 16, three Re curves
    ax = axes[0, 1]
    for re, color in zip([10, 125, 1000], [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["vermillion"]]):
        sub = case16[abs(case16["re"] - re) < 1e-6].sort_values("ri")
        ax.plot(sub["ri"], sub["nu"], "o-", color=color, label=f"$Re={re:g}$")
    ax.set_xscale("log")
    ax.set_xlabel("$Ri$")
    ax.set_ylabel("$Nu$")
    ax.legend(fontsize=9)
    ax.text(0.03, 0.95, "(b)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    # (c) Nu vs R* across the six concentric geometries, fixed Re=125, Ri=1
    ax = axes[1, 0]
    concentric = df[(abs(df["epsilon"]) < 1e-9) & (abs(df["re"] - 125) < 1e-6) & (abs(df["ri"] - 1.0) < 1e-6)]
    concentric = concentric.sort_values("r_star")
    ax.plot(concentric["r_star"], concentric["nu"], "o-", color=OKABE_ITO["green"])
    ax.set_xlabel("$R^\\ast$")
    ax.set_ylabel("$Nu$")
    ax.text(0.03, 0.95, "(c)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(0.97, 0.95, "$Re=125$, $Ri=1$, $\\varepsilon=0$", transform=ax.transAxes,
            fontsize=9, ha="right", va="top")

    # (d) Nu vs epsilon within the R*=0.3 group, fixed Re=125, Ri=1
    ax = axes[1, 1]
    group = df[(df["r_star"] == 0.3) & (abs(df["re"] - 125) < 1e-6) & (abs(df["ri"] - 1.0) < 1e-6)]
    group = group.sort_values("epsilon")
    ax.plot(group["epsilon"], group["nu"], "o-", color=OKABE_ITO["purple"])
    ax.set_xlabel("$\\varepsilon$")
    ax.set_ylabel("$Nu$")
    ax.text(0.03, 0.95, "(d)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(0.97, 0.05, "$Re=125$, $Ri=1$, $R^\\ast=0.3$", transform=ax.transAxes,
            fontsize=9, ha="right", va="bottom")

    fig.tight_layout()
    fig.savefig(OUT)
    plt.close(fig)

    # Printed summary for writing up the text
    print("--- (a)/(b) case 16 (R*=0.3, eps=0) ---")
    for ri in [0.01, 1.0, 10.0]:
        sub = case16[abs(case16["ri"] - ri) < 1e-6].sort_values("re")
        print(f"Ri={ri}: Nu(Re=10)={sub.iloc[0]['nu']:.3f}  Nu(Re=1000)={sub.iloc[-1]['nu']:.3f}  "
              f"ratio={sub.iloc[-1]['nu']/sub.iloc[0]['nu']:.3f}")
    for re in [10, 125, 1000]:
        sub = case16[abs(case16["re"] - re) < 1e-6].sort_values("ri")
        print(f"Re={re}: Nu(Ri=0.01)={sub.iloc[0]['nu']:.3f}  Nu(Ri=10)={sub.iloc[-1]['nu']:.3f}  "
              f"ratio={sub.iloc[-1]['nu']/sub.iloc[0]['nu']:.3f}")
    print("--- (c) concentric, Re=125 Ri=1 ---")
    print(concentric[["r_star", "nu"]].to_string())
    print(f"ratio R*=0.1 to R*=0.6: {concentric.iloc[0]['nu']/concentric.iloc[-1]['nu']:.3f}")
    print("--- (d) R*=0.3 group, Re=125 Ri=1 ---")
    print(group[["epsilon", "nu"]].to_string())
    print(f"ratio eps_max to eps=0: {group.iloc[-1]['nu']/group.iloc[0]['nu']:.3f}")


if __name__ == "__main__":
    main()
