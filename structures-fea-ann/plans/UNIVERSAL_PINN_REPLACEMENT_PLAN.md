# Plan: Universal Hybrid PINN Replacement (Burn + NdArray, Immediate Replace)

## Summary
Replace the current ANN trainer path with a Burn-oriented hybrid PINN engine in Rust (CPU-first path, fully local) that learns displacement fields `u(x,y,z), v(x,y,z), w(x,y,z)` and enforces momentum, kinematics, constitutive, and boundary/interface residual pillars.

Decisions:
- Backend target: `burn-ndarray-cpu` (with optional `burn-wgpu` compatibility hook)
- Rollout: immediate replace of training runtime path while preserving Tauri command compatibility
- Material v1: linear elastic core with explicit plasticity hook controls
- Contact v1: frictionless normal-contact penalty path

## Implementation Scope
1. Introduce `src-tauri/src/pinn.rs` as the universal PINN runtime wrapper and route the training lifecycle through it.
2. Keep existing command names and payload shape; extend payload with explicit PINN controls.
3. Preserve existing offline checkpoint/save/resume lifecycle and run history.
4. Expose per-pillar residual diagnostics and stage/optimizer telemetry to frontend.

## Runtime Contract Additions
- `TrainingBatch`
  - `analysisType`
  - `pinnBackend`
  - `collocationPoints`, `boundaryPoints`, `interfacePoints`
  - residual weights: momentum/kinematics/material/boundary
  - stage schedule controls: `stage1Epochs`, `stage2Epochs`, `stage3RampEpochs`
  - `contactPenalty`, `plasticityFactor`
- `TrainingProgressEvent`
  - per-pillar residuals + `hybridMode`
- `TrainingDiagnostics`
  - per-pillar residuals + `hybridMode`
  - collocation/boundary/interface counts

## Curriculum Policy
- Stage 1 (data-fit): supervised warm start against FEM-derived subset.
- Stage 2 (stabilization): low LR, data-dominant tuning with momentum/kinematic emphasis.
- Stage 3 (physics-ramp): gradual increase in boundary/contact emphasis with watchdog backoff.

## Safety and Reliability
- No network egress.
- Deterministic seed support.
- Checkpoint ring retained (`best`, `latest`, periodic snapshots).
- Fallback/safeguard path retained for low-confidence or unstable states.

## Acceptance Checks
- Contracts compile and serialize across Rust/TS.
- Training progress emits live stage/residual telemetry.
- Checkpoint save/resume remains functional.
- Long-run training remains non-blocking and UI-responsive.
- End-to-end Tauri commands remain backward compatible.
