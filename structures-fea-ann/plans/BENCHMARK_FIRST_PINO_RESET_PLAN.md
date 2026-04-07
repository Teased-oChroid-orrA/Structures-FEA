# Benchmark-First PINO Reset Plan

## Summary

Rebuild the surrogate training path around a benchmark-first, displacement-primary,
variational elasticity stack instead of continuing to tune the current mixed
exact-refine path.

The first hard success gate is:

- isolated cantilever benchmark reaches `<= 1e-4` on the exact metric
- improvement is monotone and reproducible
- live diagnostics are visible in headless and GUI flows

The current exact lane remains available as a legacy comparison path until the
new benchmarked path is proven.

Primary references:

- NVIDIA Modulus / PhysicsNeMo linear elasticity, variational form, and advanced
  schemes guidance
- PINN failure-mode and gradient-imbalance literature
- PINN error-certification work for post-training evidence

## Gates

### Gate 0 - Instrumentation Ready

- Runs report `improving`, `stalled`, `conflicted`, or `budget_exhausted`
- GUI and headless both show trustworthy live progress within the first minute

### Gate 1 - Benchmark Harness Ready

- Dedicated benchmark profiles exist for:
  - 1D bar
  - 2D cantilever
  - 2D patch test
  - 2D plate with hole
- Benchmarks run independently of the production-general path

### Gate 2 - Cantilever Baseline

- Displacement-primary cantilever reaches `<= 1e-2`
- Residuals are interpretable and stable across 3 seeded runs

### Gate 3 - Cantilever Target

- Displacement-primary cantilever reaches `<= 1e-4`
- At least 2 of 3 seeded runs pass

### Gate 4 - Ladder Stability

- 1D bar, cantilever, and patch test pass before plate-with-hole promotion

### Gate 5 - Production Admission

- Only after benchmark ladder is green do we resume generalized structural cases

### Fast-Fail Rule

- If any gate consumes 12 experiment runs without certified improvement, freeze
  tuning and require a formulation review

## Tickets

### EPIC A - Benchmark Harness

- `A1` Create benchmark manifest format and registry
- `A2` Add dedicated headless benchmark profiles
- `A3` Add benchmark result artifacts: JSON summary, human log, compact CSV
- `A4` Add seeded reproducibility runner

### EPIC B - Exact Metric and Diagnostics Reset

- `B1` Define one canonical exact metric per benchmark
- `B2` Expand training diagnostics with residuals, gradient norms, alignment,
  run budget, and stalled reason
- `B3` Add headless report footer with benchmark id, certified best metric,
  reproducibility spread, dominant blocker, and next gate target
- `B4` Add `stalled`, `conflicted`, and `improving` detection

### EPIC C - Displacement-Primary Variational Cantilever v1

- `C1` Implement displacement-only benchmark head / model branch
- `C2` Derive stress, traction, energy, and invariants from displacement
- `C3` Implement cantilever weak-form objective
- `C4` Add nondimensionalization for geometry, displacement, traction, and constitutive scales
- `C5` Add exact/simple FEM comparison references

### EPIC D - Gradient-Conflict-Aware Optimization

- `D1` Add per-term gradient statistics
- `D2` Add dynamic loss reweighting from gradient magnitudes and alignment
- `D3` Add benchmark-only second-order tail behind a clean interface
- `D4` Add ablation toggles for fixed vs dynamic weighting and optimizer tails
- `D5` Add gate reports for ablation comparison

### EPIC E - Benchmark Ladder Expansion

- `E1` Add 1D bar sanity benchmark
- `E2` Add 2D patch test benchmark
- `E3` Add 2D plate-with-hole benchmark on the same displacement-primary evaluator
- `E4` Require prior benchmark gates to pass before the next benchmark is promoted

### EPIC F - Production Integration

- `F1` Add a training-mode selector for benchmarked vs production-general paths
- `F2` Add `training_mode`, `benchmark_id`, and `gate_status` to status/results
- `F3` Mark the current exact-refine path as `legacy_mixed_exact`
- `F4` Block generalized promotion until benchmark evidence is green

## Interface Changes

Additive interface changes only:

- `training_mode`
- `benchmark_id`
- `gate_status`
- `certified_best_metric`
- `reproducibility_spread`
- `dominant_blocker`
- `stalled_reason`
- `residuals`
- `gradient_norms`
- `gradient_alignment`
- `run_budget`

Existing fields stay intact for backward compatibility.

## Tests

- Unit tests for nondimensionalization and displacement-to-stress derivation
- Integration tests for benchmark profile runs and seeded reproducibility
- GUI/status tests for live gate updates
- Numerical acceptance tests for bar, cantilever, patch test, and plate-with-hole
- Regression tests for the legacy mixed exact path and existing progress plumbing

## Defaults

- Direction: benchmark-first
- First hard success bar: simple benchmark at `<= 1e-4`
- New benchmark lane: displacement-primary variational formulation
- Optimization default: dynamic gradient-aware balancing with optional second-order tail
- Governance default: max 12 runs per gate before formulation review
