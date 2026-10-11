# Mixed Convection in a Rotating Eccentric Horizontal Annulus via Data-Driven Surrogate Modeling

Code and analysis scripts accompanying the manuscript submitted to *International Communications in Heat and Mass Transfer*.

**Authors:** Hussein Kokash, Khalil Khanafer, Mihai Burzo

## What this repository contains

- `03-computation/` — CFD post-processing, physical validation (Kuehn & Goldstein 1978 comparison, Couette-flow check, eccentric-conduction shape-factor derivation and grid-refinement verification), and Grid Convergence Index analysis scripts.
- `04-ml/` — Scalar (XGBoost/KAN/PINN) and spatial-field (per-node MLP/DeepONet/GNN) surrogate training, uncertainty-quantification (bootstrap and deep ensembles), SHAP interpretability, and true out-of-sample blind-test scripts, plus the small logged metrics/results JSON files.
- `05-manuscript/` — Full LaTeX manuscript source (`main.tex`, `sections/`, `references.bib`), all publication figures, and the Python scripts that generate every figure from the underlying result data.

## What this repository does NOT contain

The raw CFD simulation data (35 geometries x 12 Reynolds numbers x 13 Richardson numbers = 5,460 ANSYS Fluent cases, including mesh files, solver transcripts, and per-case field exports) is not included here due to its size (several gigabytes). It is available from the corresponding author upon reasonable request. All processing and analysis of that data is fully reproducible from the scripts in this repository, given the raw simulation outputs.

## Reproducing the results

Each script under `04-ml/scripts/` and `03-computation/` corresponds to one result reported in the manuscript (scalar/field surrogate training, cross-validation, the true out-of-sample test, uncertainty quantification, and all validation checks). Figure-generation scripts under `05-manuscript/scripts/` regenerate every figure in the manuscript from the small, included result JSON files or from the raw CFD data (for the field-comparison and schematic figures) once the latter is obtained from the corresponding author.

Solver: ANSYS Fluent 2026 R1, driven via the PyFluent Python interface.

## Citation

If you use this code, please cite the manuscript (citation details to be added upon publication).

## Contact

Hussein Kokash, Department of Mechanical Engineering, American University of Madaba — h.kokash@aum.edu.jo
