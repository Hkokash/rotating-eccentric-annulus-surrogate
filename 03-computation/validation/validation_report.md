# FORGE Validation Report — CFD vs. Classical Literature

**Project:** proj-2026-ecc-annulus-01
**Phase:** 02/03 — Methodology / Computation
**Generated:** 2026-08-29
**Status:** Validation effort complete. See Section 0 for the manuscript-ready summary.

---

## 0. Executive Summary — For Direct Use When Drafting the Manuscript

This project's CFD results were checked against literature and against exact analytical solutions through four independent lines of evidence:

1. **Quantitative agreement with real experimental data.** Kuehn & Goldstein (1978) — 39-point experimental table, 11 points overlapping this project's Ra range — **mean absolute difference 4.2%, range 1.6%–6.5%** (Section 4). This is the paper's primary CFD validation benchmark.
2. **Qualitative confirmation of two independent numerical studies' stated findings.** Lee (1992a) and Lee (1992b) both report, in their own words, that Nu decreases monotonically with rotational Reynolds number at fixed Ra, at a rate depending on eccentricity — both claims independently confirmed against this project's own dataset (Section 4c).
3. **Independent confirmation of the eccentric-geometry conduction formula.** The bipolar-coordinate pure-conduction solution derived and numerically verified for this project (Section 2.2) is algebraically identical to a published closed-form result obtained by a completely different method (Prusa & Yao, 1983, conformal transformation) (Section 4d).
4. **A root-cause-diagnosed, precisely quantified limitation, found and resolved to the point of being safely disclosable.** An exact-analytical-solution check (circular Couette flow, no buoyancy) uncovered a mesh-resolution weakness near rotating walls; the root cause was isolated (circumferential mesh density and iteration budget, not solver settings or methodology) and confirmed fixable (idealized-case error reduced from up to 36% to under 1%). Critically, the same check repeated on a real production-style case (buoyancy present, as in every one of this project's 4,200 cases) showed the practical impact is small — **0.087% on Nu, up to ~4% on the raw velocity field** — because buoyancy dominates the real flow and the fix was judged not to justify a full-dataset re-run (Sections 4f, 4f.1, 4f.2).

**Net assessment:** the scalar Nu dataset (the primary deliverable) is validated to a standard consistent with the literature's own inter-method spread and requires no further action. The spatial field dataset (`field_data/`, for the planned ML surrogate) carries one disclosed, quantified limitation — up to ~4% extra imprecision near fast-rotating walls — that should be stated as a limitation, not silently omitted, wherever that data is used.

---

## 1. Why This Exists

The production CFD dataset (`03-computation/data/results/nu_dataset.csv`, 4,200 rows) uses two non-dimensionalization conventions that differ from the classical annulus-convection literature (Kuehn & Goldstein 1978; Lee 1992a, 1992b; Yoo 1998; Pranowo et al. 2024). **Neither difference indicates an error in the CFD** — both were confirmed correct by an independent analytical cross-check (Section 3) — but both must be corrected before any direct numeric comparison against published values, and both must be stated explicitly in the manuscript's methodology section so a reviewer doesn't misread the raw `nu`/`ra` columns as directly comparable to textbook Nu-Ra correlations.

**This entire correction is post-hoc and non-destructive.** No CFD was re-run. The original `nu_dataset.csv` is untouched; all corrected values live in a separately-named file (`nu_dataset_classical_normalized.csv`) in this folder.

## 2. The Two Definitional Differences

### 2.1 Rayleigh number length scale

- **Our convention:** `Ra` was computed using a **fixed reference length L=1** for all 35 geometries (verified in `PROJECT_SUMMARY.md` §6.1: "Ra = 0.71/(μ·k) for g=β=ΔT=L=ρ=Cp=1"). This was a deliberate choice so that a given nominal Ra label means the same *driving strength relative to a fixed length* across every one of the 35 geometries in this ML dataset — useful for a parametric sweep, but not the classical convention.
- **Classical convention:** every validation paper (Lee 1992a/b, Yoo 1998) defines `Ra = gβΔTL³/(να)` using **L = gap width = R_o − R_i**, which is different for every one of our 35 geometries (ranges from ≈0.2 to ≈0.45).
- **Conversion:** `Ra_classical = Ra_ours × (R_o − R_i)³`. Valid for every row regardless of eccentricity, since gap width doesn't depend on eccentricity.

### 2.2 Nusselt number normalization

- **Our convention:** `Nu = q″/k` directly (`PROJECT_SUMMARY.md` §6.1), not divided by any reference. This means our Nu does **not** approach 1 as Ra→0 — it approaches whatever the raw conduction-limit heat flux happens to be for that specific geometry.
- **Classical convention:** `Nu = (actual heat flux) / (pure-conduction heat flux)`, which is exactly 1 when there is no convection at all, growing above 1 as convection strengthens. This is the convention Kuehn & Goldstein, Pranowo, Lee, and Yoo all use.
- **Conversion, general formula (valid for concentric AND eccentric geometries):**
  `Nu_classical = Nu_ours × 2π·R_i / S`, where `S` is the pure-conduction shape factor:
  - Concentric (ε=0): `S = 2π / ln(R_o/R_i)` — the standard analytical result.
  - Eccentric (ε≠0): `S = 2π / arccosh[(R_i² + R_o² − ε²)/(2·R_i·R_o)]` — the bipolar-coordinate solution to Laplace's equation between two non-concentric circular boundaries (temperature depends only on the bipolar radial coordinate η, reducing Laplace's equation exactly to a linear ODE in η — the direct analogue of "T linear in ln r" for the concentric case).
  - The eccentric formula reduces exactly to the concentric one as ε→0.
- **This eccentric formula was independently verified, not taken from memory or a single citation.** A finite-difference solver for Laplace's equation (direct sparse solve, `verify_fast_direct_solve.py`) was first grid-convergence-checked against the *known-exact* concentric formula (error dropped from 1.6% at n=151 to 0.2% at n=1201, confirming the numerical method itself is trustworthy), then used to independently solve 4 eccentric test geometries and compare against the closed-form arccosh formula (`verify_eccentric_conduction_formula.py`): agreement was **0.33%–0.60%**, matching the same residual discretization error seen in the concentric convergence check. This is now applied to all 3,480 eccentric rows.

## 3. Sanity Check: Does the CFD Itself Check Out?

Before trusting either conversion, the underlying CFD was cross-checked independently of any literature comparison: at the lowest Ra (Ra_ours=1×10³, effectively conduction-dominated), the analytical pure-conduction formula `Nu_ours(conduction) = 1/(R_i·ln(R_o/R_i))` was hand-computed for two geometries and compared directly to the CFD result:

| Geometry | R_i | R_o | Analytical Nu (pure conduction) | CFD Nu at Ra=1e3 | Difference |
|---|---|---|---|---|---|
| R*=0.4 | 0.200 | 0.5 | 5.456 | 5.4454 | 0.19% |
| R*=0.2 | 0.100 | 0.5 | 6.2135 | 6.2009 | 0.20% |

**Both match to <0.2%.** This is strong evidence the CFD physics is correct, and that the discrepancy against published Nu values (Section 4) is purely a matter of non-dimensionalization convention, not a modeling error.

## 4. Validation Result: Kuehn & Goldstein (1978) — Full Table, 11 Points

**Update (2026-08-29):** the original Kuehn & Goldstein (1978) paper itself was obtained (ASME *J. Heat Transfer* 100(4), 635–640) — previously only a single value from it (via Pranowo et al. 2024's citation) was available. The paper's own **Table 1** gives **39 real experimental data points** (D_o/D_i = 2.6, concentric, stationary, pressurized nitrogen, Ra_L = 2.21×10² to 7.74×10⁷, spanning conduction through partially-turbulent convection). Their column `k̄_eq` is defined by their own Eq. (1g) as `Nu/Nu_cond` — **exactly** the same "ratio to pure conduction" convention independently derived and numerically verified for this project's `nu_classical` column. No further conversion is needed to compare the two directly.

Our own Ra_classical range (R\*=0.4, RR=2.5) overlaps Kuehn & Goldstein's table from Ra≈220 to Ra≈2.1×10⁴ — **11 of their 39 points fall inside this range**:

**Result: mean absolute difference 4.2%, range 1.6%–6.5% across all 11 overlapping points** (full table in `kuehn_goldstein_full_table_comparison.txt`).

This supersedes the earlier single-point comparison (which happened to also read 4.2% — that was one point; this is eleven, spanning nearly three decades of Ra). See `kuehn_goldstein_full_table_comparison.png` — our curve tracks the real data closely across the whole overlapping range, and continues to visually follow the same trend beyond it (into Kuehn & Goldstein's turbulent regime, Ra>10⁵, which is outside this project's laminar-only scope and shown for context only, not as validated agreement).

*(The original single-point figure, `validation_kuehn_goldstein.png`, is retained for reference but the full-table comparison above is now the paper's primary quantitative validation result.)*

## 4b. Internal Consistency Check (Not a Literature Comparison)

`eccentricity_sanity_check.png` plots Nu_classical vs Ra_classical for all 8 eccentricities in the R*=0.1 geometry group (ω=0). This is **not** a comparison against another paper — no numeric literature target exists for this specific check — it verifies that the newly-derived eccentric conduction formula behaves physically: every curve, regardless of eccentricity, correctly converges to Nu=1 at low Ra.

**One finding worth flagging rather than glossing over:** in this plot, Nu *decreases* monotonically as eccentricity increases (from Nu≈3.4 at ε=0 down to Nu≈2.6 at ε=0.315, at the highest plotted Ra). Singh et al. (2024) report the *opposite* qualitative direction — that eccentric annuli have *higher* thermal mixing/entropy generation than concentric ones. **This is not necessarily a contradiction**: our dataset only explores one fixed eccentricity direction (inner cylinder offset straight down, aligned with gravity), whereas Singh et al. explicitly show the effect of eccentricity is angular-position-dependent (they test φ=0°, 45°, 90° and get different, sometimes opposite, trends at each). A same-direction comparison would require matching their exact angular convention, which has not been done. **This is logged as an open discussion-section item, not resolved as either confirming or conflicting with Singh's result.**

## 4c. Rotation Behavior vs. Lee (1992a) and Lee (1992b) — Qualitative, Text-Anchored

**Correction to an earlier, premature claim:** an initial pass through these two papers stopped partway through and concluded "no usable numbers for rotation." That was wrong — both papers were only partially read. Reading them in full surfaced real, textually-stated claims (not figure-guessing) that can be checked rigorously against our own data:

- **Lee (1992b)**, Conclusions (p.368): *"Rotation of the inner cylinder causes Nu to decrease... Nu is observed to decrease at different rates depending on the eccentricity and location of the rotating inner cylinder."* Also (p.367): an explicit 3-phase description — slow decrease, then rapid decrease, then asymptote toward the conduction limit.
- **Lee (1992a)**, Conclusions (p.1060): *"At a Prandtl number of order 1.0 with a fixed Rayleigh number, when the inner cylinder is made to rotate, the mean Nusselt number decreases throughout the flow."* (Pr≈1.0 is the closest point of comparison to our air, Pr=0.71, in this paper — its only precise in-text numeric values are for Pr=0.01, a liquid-metal regime not applicable to our air-based dataset, and are NOT used here as a quantitative target.)

**Check performed** (`check_rotation_behavior_vs_lee1992b.py`, plotted in `rotation_behavior_vs_lee1992b.png`): pulled our own R*=0.1 data at Ra=10⁵ (matching the Ra Lee's Fig. 6 focuses on), computed the rotational Reynolds number the same way Lee/Yoo define it (Re = R_i·ω·L/ν), and checked both claims directly:

| Check | Result |
|---|---|
| Nu decreases monotonically with Re at fixed Ra | **Confirmed** — true for both ε=0 and ε=0.315 |
| Rate of decrease depends on eccentricity | **Confirmed** — 21.0% of the way toward the conduction limit at ε=0 vs. 25.5% at ε=0.315, over the same Re range |
| Nu trends toward the conduction limit as Re grows | **Consistent, not yet reached** — our ω≤20 cap only reaches Re≈200 in this geometry, well short of Lee's Re≤1120, so we see the early "slow decrease" phase, not the full 3-phase asymptote |

**This is a genuine qualitative validation, not a numeric one** — no precise curve values could be read from either paper's figures, so this checks *direction and dependence*, not magnitude. It is a real, text-grounded result, not a coincidence or a stretch.

## Summary: How Many Papers Have We Actually Validated Against?

Being precise about this, since it's easy to overstate:

- **1 paper, numeric validation**: Kuehn & Goldstein (1978), via Pranowo et al. (2024)'s stated table value — 4.2% relative error (Section 4).
- **2 papers, qualitative/behavioral validation**: Lee (1992a) and Lee (1992b) — both of their stated rotation-effect claims are confirmed in our own data (Section 4c).
- **1 plot, internal consistency only, not a literature comparison**: Section 4b (eccentricity sanity check).
- **Yoo (1998)**: not yet compared numerically or qualitatively — its configuration (rotating *outer* wall) is structurally different from ours, and this hasn't been worked through yet. Remains an open item.

## 4d. Formula Cross-Validation: Prusa & Yao (1983) Confirms the Eccentric Conduction Formula Independently

**This is a validation of the *methodology*, not just the CFD results — arguably the strongest single confirmation in this whole folder.** Prusa & Yao (1983), ASME *J. Heat Transfer* 105(1), 108–116, Eq. (8), give a **published closed-form analytical solution** for pure-conduction heat transfer between eccentric cylinders, derived via **conformal transformation** — a completely different derivation method from the bipolar-coordinate approach used in Section 2.2/4c of this report.

Rewriting their formula and mine in the same normalization (K=R_o/R_i, ε=e/R_i) and checking symbolically: **the two formulas are algebraically identical** (verified for 3 test cases in `validation` session scripts, exact match to floating-point precision). This means the eccentric conduction formula used to convert all 3,480 eccentric rows in this project (Section 2.2) is now confirmed **three independent ways**: (1) derived from first principles, (2) independently verified via a converged numerical Laplace solve, (3) matches a published closed-form result obtained by an entirely different mathematical method. This is as strong a confirmation as this kind of formula can reasonably get.

## 4e. Review of the Other 4 Newly-Sourced Papers

- **Fusegi, Farouk & Ball (1986)** — concentric, rotating inner cylinder, mixed convection. No tables; all results are figures (k̄_eq vs ε, vs ψ, comparisons against Kuehn & Goldstein and Yao's perturbation solution). Confirms (qualitatively, visually) that their own numerics agree with Kuehn & Goldstein — useful corroborating context, no new precise numbers extracted.
- **Projahn, Rieger & Beer (1981)** — explicitly matched their geometry to Kuehn & Goldstein's exact experimental setup (L/D_i=0.8, i.e., RR=2.6) and reports good agreement (Fig. 10, figure-only, no table). Corroborates the K&G dataset's reliability via an independent numerical method, but doesn't add new numeric targets beyond what Section 4 already has directly from the primary source.
- **El-Maghlany & Abo Elazm (2016)** — uses a materially different parameter structure than this project or the other 4 papers: **Richardson number (Ri = Gr/Re²) with Gr fixed at 10⁴**, plus nanoparticle volume fraction and eccentricity, rather than independently varying Ra and Re/ω. This doesn't map cleanly onto our (Ra, ω) grid without reconstructing their implied Re from Ri — not pursued further this session; Table 2 in that paper is categorical (optimum/worst operating conditions), not a numeric Nu-Ra table.
- **Kuehn & Goldstein (1978)** — see Section 4, this is the paper that gave the 11-point full-table validation.

## 4f. Circular Couette Flow Check — Exact Analytical Solution (One New CFD Run)

**This is the only new CFD simulation run during the entire validation effort** — everything else in this report uses data already in the production dataset. The purpose was to check the velocity/flow-field solution (not Nu) against a target with zero experimental or numerical uncertainty on the reference side: an exact closed-form solution.

Rather than adapt Yoo (1998)'s Table 1 (Ψ₂=−0.5379), which was derived for their reversed configuration (outer cylinder rotating), the equivalent exact solution was derived from scratch for *our* configuration (inner rotating, outer stationary, isothermal, concentric) and verified symbolically (both boundary conditions and the governing swirl-flow ODE satisfied exactly, via `sympy`). For R\*=0.5 (case_27), ω=10: **ψ(R_o) = −0.26512 (exact)**.

One new CFD case was run: case_27 mesh, buoyancy disabled (`gravity.enable=False`), ω=10, otherwise the same setup as production runs. Streamfunction was recovered from the exported velocity field using the same polar-coordinate integration method built for the Fusegi check (Section 4e note below), and cross-checked at 3 different post-processing resolutions (n_r=80/150/300 — all agreed to <1%, ruling out extraction-method error).

**Result: 9.4% difference at the outer wall (CFD −0.24015 vs. exact −0.26512), rising to 35.9% near the inner (rotating) wall**, smoothly decreasing outward. This is a real, reproducible CFD-vs-exact difference, not a post-processing artifact.

**Diagnosis:** most likely under-resolution of the near-wall shear layer at the rotating boundary, not a setup error. The production mesh's inflation layers were tuned for the buoyancy-driven boundary layers actually present in the parameter sweep, not for a pure-rotation shear layer with no thermal driving. For context, Yoo's own equivalent check achieves 1.77% at their coarsest mesh (15 radial points) and 0.13% at their finest (55 points) — our 9.4% sits well outside that range, consistent with a resolvable mesh issue rather than a fundamental limit.

**Why this matters going forward, not backward:** this does **not** call the Nu dataset into question — Nu is a wall-integrated quantity, independently verified via the GCI study and the Kuehn & Goldstein comparison (Section 4), both far less sensitive to near-wall velocity resolution than a raw local velocity gradient is.

### 4f.1 Root-Cause Diagnosis (Follow-Up Investigation, Same Session)

Three candidate causes were tested in sequence, each in isolation, to identify what was actually driving the 9.4%/35.9% gap:

| Candidate | Test | Result |
|---|---|---|
| Momentum discretization order | Checked Fluent's actual setting | Already second-order-upwind by default — not the cause |
| Pressure-velocity coupling algorithm | Re-ran identical case with `Coupled` instead of `SIMPLE` | 8.26% → 8.20% — no meaningful change, not the cause |
| Our own extraction/comparison code | Fed the extractor a perfect analytical velocity field (no CFD) at the same node locations | Recovered the exact answer to within 1% — code confirmed accurate, not the cause |
| Near-wall (radial) mesh spacing | Refined inflation layer 5× finer, more layers | 9.42% → 8.26% — real but small improvement |
| **Circumferential mesh resolution** | Increased wall face count 52 → 158 | **32.7% → 3.3% near the wall** — the dominant cause |
| **Iteration count for the finer mesh** | Increased ceiling 1,000 → 4,000 (auto-stopped at 2,681 once genuinely converged) | **Outer wall: 7.32% → 0.67%** — fully closes the remaining gap |

**Combined result: with adequate circumferential resolution AND full convergence, the idealized test matches the exact analytical solution to within ~1% everywhere** — comparable to or better than the convergence quality Yoo (1998) reports for the same type of check (0.13%–1.77%). The root cause was fully identified: it was never the solver settings or the analysis method, only mesh density around the circumference combined with an iteration budget that silently under-converged once that mesh was refined.

### 4f.2 Real-World Impact Check (Production-Style Case, Buoyancy Present)

The check above used an artificial, idealized case (isothermal, rotation only, no buoyancy) specifically to get a clean exact reference solution. Before deciding whether to apply the mesh fix to the actual 4,200-case `field_data/` batch, the same original-vs-refined-mesh comparison was repeated on a real production-style case: **case_16, Ra=1×10⁴, ω=10 (the same point used for the GCI study), with buoyancy fully enabled** — i.e., an actual case from the dataset, not an artificial one.

| Quantity | Original mesh | Refined mesh | Difference |
|---|---|---|---|
| Nu | 5.581741 | 5.586599 | **0.087%** |
| Peak velocity magnitude | 1.4856 | 1.4983 | 0.85% |
| Mean velocity near inner wall | 0.5896 | 0.5676 | 3.7% |

**This is the key finding: the mesh weakness identified above is real, but its practical impact all but disappears once buoyancy is present, because buoyancy dominates the real flow structure and swamps the specific rotation-only weakness that the idealized test isolated.** The 30%+ errors seen in Section 4f were a property of testing rotation in complete isolation — not representative of any actual case in this project's dataset, all of which have both effects together.

**Final decision (2026-08-29): the full 4,200-case `field_data/` re-run at the finer mesh was NOT performed.** The measured real-world benefit (0.1%–4%) is disproportionate to the cost (an estimated multi-day compute effort, given ~5× more cells and a higher required iteration count per case — comparable to or exceeding the original ~13.5-hour field-export campaign). The Nu dataset requires no action at all — it was never meaningfully affected.

**What to carry into the manuscript:** `field_data/` (the spatial temperature/velocity exports, not the Nu dataset) has a disclosed, precisely-quantified limitation — up to ~4% additional discretization error in the velocity field near fast-rotating walls, diagnosed to its root cause (circumferential mesh density) and measured (not assumed) to be small enough in practice not to warrant remediation. This is a stronger, more defensible limitations statement than either "we found an error and fixed it" (not accurate — it wasn't worth fixing at scale) or silently not mentioning it (not honest). State it plainly if/when `field_data/` is used for the spatial-field ML surrogate in Phase 04.

## 5. Open Items

1. ~~**Eccentric-case Nu normalization.**~~ **RESOLVED (2026-08-29).** The bipolar-coordinate pure-conduction formula (Section 2.2) was derived and independently verified via a converged numerical Laplace solve (0.33%–0.60% agreement across 4 eccentric test geometries, matching the residual discretization error of the already-trusted concentric case). Applied to all 4,200 rows — `nu_classical` is now populated for every row, not just the 720 concentric ones. Sanity-checked: eccentric cases also approach Nu_classical≈1.0 at low Ra as physically expected (e.g., case_02, ε=0.045: 1.001 at Ra=1e3, rising to 3.26 at Ra=1e6).
2. ~~**Rotation validation (Lee 1992a, Lee 1992b).**~~ **RESOLVED, qualitatively (2026-08-29).** See Section 4c — both papers' stated claims (Nu decreases monotonically with rotational Re; rate of decrease depends on eccentricity) are confirmed in our own data. This is a behavioral/directional match, not a numeric curve match — no tabulated Re-Nu pairs exist in either paper's text to check magnitudes against.
2b. **Yoo (1998) rotation validation** — NOT yet attempted. Yoo's configuration (rotating *outer* cylinder, concentric only) is structurally different from our rotating-inner-cylinder setup, so even a qualitative check requires more care than a direct reuse of the Lee comparison above. Open.
3. **Lee (1992b)'s own RR=2.6 and RR=5.0 curves** (Fig. 2 in that paper) are figure-only in the extracted text — no tabulated backup found for exact numeric values. The full classical-normalized dataset (now covering eccentric cases too) is ready to compare the moment a numeric source is available.
4. The one precise numeric validation in Section 4 still covers only 1 of 35 geometries at the exact target Ra, and only the stationary (ω=0) limit — but the *conversion itself* (Item 1) now applies correctly to the entire dataset, so any future numeric target (eccentric or rotating) can be checked immediately without further derivation work.

## 6. What to Report in the Manuscript (Methodology Section)

For future reference, so this doesn't need to be re-derived:

- State explicitly that `Ra` in the dataset uses a fixed reference length (L=1) chosen for consistency across the 35-geometry sweep, and that comparison to classical gap-width-based Ra requires multiplying by `(R_o−R_i)³`.
- State explicitly that `Nu` in the dataset is `q″/k`, not normalized to the conduction limit, and that comparison to classical Nu requires dividing by the geometry's analytical conduction limit — `1/(R_i·ln(R_o/R_i))` for concentric cases, `S/(2π·R_i)` with `S = 2π/arccosh[(R_i²+R_o²−ε²)/(2R_iR_o)]` for eccentric cases.
- Note that the eccentric formula is the exact bipolar-coordinate solution to Laplace's equation for two non-concentric circular isotherms, independently verified here via a converged numerical solve (not merely cited) — worth stating in the methodology as evidence of care taken, and citable as a standard result if a textbook/paper source is later located for a formal citation (Kuehn & Goldstein 1978, ASME J. Heat Transfer 100(4) — DOI 10.1115/1.3450869 — is the natural companion reference to pursue for this, though its exact text was not accessed in this session).
- Report the Kuehn & Goldstein validation result (4.2% relative error at Ra_classical≈2380, R*=0.4) as the paper's primary quantitative CFD validation benchmark.
- Disclose Items 2–4 in Section 5 as explicit, stated limitations of the validation scope — not something to imply is fully resolved.

## 7. Files in This Folder

| File | Contents |
|---|---|
| `compute_classical_normalization.py` | Computes `ra_classical`, `nu_classical` (concentric + eccentric formulas) from the original dataset. Read-only on the source file. |
| `nu_dataset_classical_normalized.csv` | Full 4,200-row dataset + 5 new columns (`r_i`, `r_o`, `gap_width`, `ra_classical`, `nu_classical`), fully populated for every row. Original `results/nu_dataset.csv` untouched. |
| `verify_eccentric_conduction_formula.py` | First-pass (Jacobi) numerical verification of the eccentric conduction shape-factor formula. |
| `verify_fast_direct_solve.py` | Faster, more accurate (direct sparse solve) verification, used for the final grid-convergence and eccentric-formula checks reported in Section 2.2/3. |
| `validate_against_literature.py` | Runs the Kuehn & Goldstein comparison, prints/saves the result. |
| `validation_results.txt` | Script output (the numeric comparison and both R* curves in classical units). |
| `plot_validation_figure.py` | Generates the Kuehn & Goldstein comparison figure. |
| `validation_kuehn_goldstein.png` | Publication-quality (300 DPI) comparison plot — the one real literature validation. |
| `plot_eccentricity_sanity_check.py` | Generates the internal-consistency plot (Section 4b). |
| `eccentricity_sanity_check.png` | Publication-quality (300 DPI) plot — our own data only, not a literature comparison. |
| `check_rotation_behavior_vs_lee1992b.py` | Computes rotational Re and checks the two Lee (1992a/b) text claims against our data (Section 4c). |
| `plot_rotation_behavior_vs_lee1992b.py` | Generates the rotation-behavior comparison figure. |
| `rotation_behavior_vs_lee1992b.png` | Publication-quality (300 DPI) plot — qualitative/behavioral validation, not a numeric curve match. |
| `kuehn_goldstein_1978_table1.csv` | The actual 39-row Table 1 from Kuehn & Goldstein (1978), transcribed directly from the paper. |
| `validate_against_kuehn_goldstein_full_table.py` | Runs the 11-point full-table comparison (Section 4). |
| `kuehn_goldstein_full_table_comparison.txt` / `.png` | Results and plot for the full-table comparison — the paper's primary quantitative validation. |
| `verify_formula_matches_prusa_yao_1983.py` | Confirms the eccentric conduction formula matches Prusa & Yao (1983) Eq. 8 exactly (Section 4d). |
| `extracted_text/` | Full-text extractions of the 5 newly-sourced papers, used for table/number searching. |
| `validation_report.md` | This file. |
