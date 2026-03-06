# Plan: Self-Contained 3D Plate FEA + Adaptive ANN (Primary Solver) Desktop App

## Summary
Build a new offline desktop app in `/Users/nautilus/Desktop/Structures-FEA` using the same general stack as the reference app (`Svelte 5 + TypeScript + Tauri + Rust`).
The app will solve a 3D plate problem (10 in × 4 in × 0.125 in, axial 1000 lbf) and cover:
- Equilibrium equations
- Constitutive equations
- FEA matrix formulation
- Stress tensor + principal stresses
- Failure criteria
- Boundary conditions
- Thermal stress analysis
- Impact/dynamic analysis
with a self-learning ANN that is the primary predictor and grows/adapts during training.

## Scope And Deliverables
1. Project scaffold
- Create `structures-fea-ann/` as the app root under `/Users/nautilus/Desktop/Structures-FEA`.
- Stack: SvelteKit frontend, Tauri shell, Rust compute core.
- No cloud calls; fully local execution/storage.

2. Solver domains
- Static linear elasticity (3D solid, small strain).
- Thermoelastic coupling via thermal strain.
- Transient dynamics (impact pulse) using time integration.
- Failure checks: Von Mises, Max Principal, Tresca.

3. ANN system
- ANN primary inference engine for displacement/stress/failure outputs.
- Continuous local training from FEM-generated truth data.
- Topology growth (neurons/layers) when validation stagnates.
- Adaptive learning-rate schedule and checkpoint rollback.

4. UX/reporting
- Input panel for geometry/material/load/BC/thermal/impact.
- Results views for displacement, stress tensor, principal stresses, failure margins.
- Comparison panel: ANN vs FEM reference error.
- Export JSON/CSV/PDF reports locally.

## Public Interfaces / Types
1. Frontend-to-backend Tauri commands
- `solve_fem_case(input: SolveInput) -> FemResult`
- `train_ann(batch: TrainingBatch) -> TrainResult`
- `infer_ann(input: SolveInput) -> AnnResult`
- `run_dynamic_case(input: DynamicInput) -> DynamicResult`
- `run_thermal_case(input: ThermalInput) -> ThermalResult`
- `evaluate_failure(input: FailureInput) -> FailureResult`
- `get_model_status() -> ModelStatus`
- `export_report(input: ReportInput) -> ExportResult`

2. Core data contracts
- `SolveInput`: geometry, mesh controls, material, BCs, loads, unit system.
- `Material`: `E`, `nu`, `rho`, `alpha`, `yield_strength` (default Al 6061-T6; editable).
- `FemResult`: nodal displacement, element strains/stresses, principal stresses, diagnostics.
- `AnnResult`: same shape as `FemResult` + confidence/uncertainty + model version.
- `DynamicInput`: impact pulse shape/time-step/damping/end-time.
- `ThermalInput`: `deltaT` and optional temperature field/gradient.
- `FailureResult`: criteria values and safety factors.

## Physics/Computation Decisions
1. Geometry and loading baseline
- Plate dimensions: `L=10 in`, `W=4 in`, `t=0.125 in`.
- Axial load: `1000 lbf` on end face normal to length.
- Applied traction default: `1000 / (4*0.125) = 2000 psi`.

2. Equations
- Equilibrium: `∇·σ + b = ρü` (static uses `ü=0`).
- Constitutive (isotropic thermoelastic): `σ = C:(ε - ε_th)`, `ε_th = αΔT I`.
- Strain-displacement: `ε = sym(∇u)`.
- FEA system:
  - Static: `K u = f_mech + f_th`
  - Dynamic: `M ü + C u̇ + K u = f(t) + f_th`

3. Discretization
- 3D solid elements (`hex8` primary, `tet4` fallback).
- Sparse assembly (CSR), direct/iterative solve (Rust/nalgebra + sparse crate).
- Dynamic integration default: implicit Newmark (unconditionally stable baseline).

4. Stress/failure outputs
- Full Cauchy stress tensor per element/Gauss point.
- Principal stresses from eigenvalues of stress tensor.
- Von Mises, Tresca, Max Principal with pass/fail margins.

## ANN Strategy (Primary Solver)
1. Model
- Base MLP regressor predicting displacement/stress fields from encoded input.
- Separate head for confidence score (epistemic proxy from ensemble/dropout).

2. Self-learning loop
- Start from seed FEM dataset generated locally from param sweeps.
- Train ANN; use ANN for primary predictions.
- Periodic FEM audits on sampled cases.
- If error threshold exceeded: trigger retraining and architecture growth.

3. Growth policy
- Trigger: validation plateau for `N` epochs and residual above target.
- Action sequence:
  - increase neurons in last hidden layer;
  - if still plateaued, add one hidden layer;
  - reduce LR on instability; rollback to best checkpoint on divergence.

4. Safety gates
- Hard cap on topology growth.
- Minimum FEM audit frequency.
- Mark low-confidence ANN outputs and auto-run FEM fallback.

## Folder / File Plan
1. App root
- `/Users/nautilus/Desktop/Structures-FEA/structures-fea-ann/`

2. Major modules
- `src/lib/ui/*` input/results/report screens
- `src/lib/types/contracts.ts` shared DTOs
- `src/lib/stores/*` state and run history
- `src-tauri/src/physics/*` elasticity/thermal/dynamic/failure
- `src-tauri/src/fem/*` mesh, element kernels, assembly, solvers
- `src-tauri/src/ann/*` model, trainer, growth policy, inference
- `src-tauri/src/io/*` local persistence/export only
- `docs/` theory notes, validation pack, traceability matrix
- `plans/FEA_ANN_MASTER_PLAN.md` canonical project plan document

## Test Cases And Acceptance Criteria
1. Deterministic physics checks
- Static traction case converges with mesh refinement.
- Stress recovery produces symmetric tensor and valid principal ordering.
- Thermal-only case gives expected restrained/unrestrained behavior.
- Dynamic pulse run is stable and energy trend is physically plausible.

2. Failure checks
- Von Mises/Tresca/Max Principal values match reference formulas on synthetic tensors.
- Boundary conditions correctly enforce fixed end and loaded end traction.

3. ANN checks
- ANN inference latency is lower than FEM baseline for same case.
- ANN median error under target on holdout FEM cases.
- Growth policy triggers correctly and improves validation metrics after expansion.

4. Integration checks
- End-to-end solve/report export works offline.
- No outbound network calls during startup/solve/train/export.

## Rollout Plan
1. Phase 1: skeleton + contracts + static FEM vertical slice.
2. Phase 2: thermal + dynamic + failure modules.
3. Phase 3: ANN training/inference + growth policy.
4. Phase 4: UI visualization + report export.
5. Phase 5: verification pack and performance hardening.

## Assumptions And Defaults
- Canonical formulation is 3D solid (per your choice).
- Default material is Al 6061-T6, user-editable.
- ANN is primary solver with mandatory periodic FEM auditing.
- Unit system default is inch-lbf-second with psi stress output.
- App must be fully self-contained and offline at runtime.
- Plan artifact path is `plans/FEA_ANN_MASTER_PLAN.md`.

## Research Basis (Web, high-value technical sources)
- [Physics-Informed Neural Networks (Raissi et al., JCP 2019)](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [AdaNet: Adaptive Structural Learning of Artificial Neural Networks (ICML/PMLR)](https://proceedings.mlr.press/v70/cortes17a.html)
- [Cascade-Correlation Learning Architecture (NeurIPS)](https://proceedings.neurips.cc/paper/1989/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html)
- [Gridap Linear Elasticity Weak Form/Implementation](https://gridap.github.io/Tutorials/stable/pages/t003_elasticity/)
- [deal.II Step-8 Elasticity Example](https://dealii.org/current/doxygen/deal.II/step_8.html)
- [Abaqus Explicit Central-Difference Operator](https://docs.software.vt.edu/abaqusv2024/English/SIMACAETHERefMap/simathe-c-expdynamic.htm)
- [Abaqus Thermal Expansion/Strain Conventions](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEMATRefMap/simamat-c-thermalexpan.htm)
- [Finite Element Formulations for Plates and Shells (review)](https://www.sciencedirect.com/science/article/pii/S0045782508003907)
- [Thermoelastic PINN Optimization paper (for hybrid training ideas)](https://www.mdpi.com/2673-4931/79/1/76)
