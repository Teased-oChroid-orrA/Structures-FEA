# Direct-Replace 3D Navier-Cauchy PINO Plan

## Summary

Replace the current 2D plate-family PINO with a direct-replace 3D no-assumption PINO that trains and infers on the full 3D linear-elastic Navier-Cauchy formulation. The new primary runtime will solve in volumetric 3D, predict full displacement/stress fields, and let geometry plus boundary conditions reduce the problem naturally instead of hard-coding 1D/2D beam/plate assumptions.

Chosen defaults:
- Cutover: direct replace at the app surface; current 2D PINO becomes internal rollback only.
- Physics v1: full 3D small-strain linear elasticity only.
- Contact, plasticity, and thermal couplings remain disabled in the primary 3D loss until the linear-elastic 3D signoff gate is passing.
- This document replaces the current 2D direct-replace PINO plan rather than living beside it as a competing plan.

## Implementation Changes

1. 3D operator domain and data model
- Replace the 2D midsurface operator encoding in `src-tauri/src/pino.rs` with a structured 3D volumetric lattice driven from `SolveInput.mesh.nx/ny/nz`.
- Canonical sample representation is one active voxel/cell-center sample over the solid domain.
- Mandatory per-cell input channels: normalized `x/y/z`, occupancy, signed distance to exterior, signed distance to hole/void, signed distance to fixed boundary, signed distance to loaded boundary, broadcast geometry scalars, broadcast material scalars, and broadcast load scalars.
- Mandatory operator outputs: `ux/uy/uz`, `sxx/syy/szz/sxy/sxz/syz`, plus derived `von_mises` and `max_principal`.
- Existing UI/result contracts stay additive-compatible by deriving legacy summaries from the 3D field result.

2. Full 3D Navier-Cauchy physics loss
- Replace the current 2D equilibrium and plane-stress loss with full 3D small-strain linear elasticity:
  - kinematics from `u(x,y,z)`
  - isotropic 3D Hooke constitutive law
  - full `div(sigma) + b = 0` equilibrium in x/y/z
  - Dirichlet boundary loss on fixed faces
  - traction/Neumann loss on loaded faces
  - invariant consistency for von Mises and principal stress
- All residual families must be nondimensionalized from sample-specific geometry/material/load scales.

3. 3D PINO operator body
- Replace the current hand-shaped 2D field-head path with a 3D operator body: lift, repeated local/global/spectral residual blocks, layer-wise adaptive activations, and a stabilized output head.
- Spectral mixing must become 3D-aware over `(nx, ny, nz)` low modes.
- Runtime and Burn training paths must stay numerically aligned within test tolerance.

4. Convergence repair
- Convergence policy in `src-tauri/src/pinn.rs` becomes decision-complete:
  - stage 1: coarse 3D grid + Adam + stronger data/observable weighting
  - stage 2: mixed coarse/full 3D supervision + balanced physics weighting
  - stage 3: full 3D grid + L-BFGS + stronger equilibrium/constitutive/boundary weighting
- Use adaptive loss balancing from residual magnitudes.
- Add residual-based adaptive refinement over the 3D domain.
- Preserve checkpoint selection by blended field/signoff/holdout score.
- Enforce stage-3-before-plateau-stop and nontrivial L-BFGS budget.

5. Holdout and signoff
- Replace the current 2D regime-specific holdout logic with 3D regime-agnostic observables selected from actual load/boundary configuration.
- Train on both field targets and signoff observables.
- Required v1 signoff cases remain cantilever, axial plate, and plate-with-hole, but solved and checked in volumetric 3D form.

## Public API / Interface Changes
- Keep existing Tauri command names stable.
- Add additive PINO metadata:
  - `operatorGrid3d`
  - `domainDim`
  - `physicsModel`
  - `spectralModes3d`
  - richer 3D residual-subterm diagnostics
  - 3D holdout/signoff metrics
- `AnnResult` remains the app-facing envelope but now semantically means a 3D surrogate operator result.
- `FemResult` / decoded operator payloads must expose full 3D displacement and symmetric Cauchy stress summaries plus scalar invariants.

## Test Plan
- Unit tests for 3D operator-grid encode/decode, 3D spectral projection invariants, 3D constitutive residuals, and runtime/Burn parity.
- Physics tests for zero-load fixed body, pure axial 3D block, cantilever trend, and plate-with-hole concentration.
- Training tests for finite residuals through Adam and L-BFGS, guaranteed stage-3 entry before plateau-stop, checkpoint/resume after L-BFGS, and deterministic residual refinement.
- Integration/signoff tests for the existing Tauri command flow plus `benchmark:pino-signoff` and `verify:pino-signoff-lock` on the 3D path.
- Acceptance: no 2D plate-specific assumptions in the primary loss, primary runtime reports 3D Navier-Cauchy metadata, signoff passes on the 3D path, and `default-1e9` diagnostics expose the limiting residual family.

## Assumptions and Defaults
- v1 scope is full 3D Navier-Cauchy for linear elasticity only.
- Contact, plasticity, thermal strain, and other nonlinear couplings stay deferred until 3D linear-elastic signoff passes.
- Current 2D PINO stays only behind an internal rollback switch during transition.
- Implementation priority order: 3D operator/data model, 3D physics loss, 3D model/runtime parity, convergence repair, holdout/signoff migration, then UI/telemetry cleanup.
