# PINO L-BFGS Dense Gradient Enablement Plan

## Summary

Restore the nominal `pino-lbfgs` phase in the Burn-native physics training path by replacing the current Adam fallback with a dense-gradient L-BFGS implementation that is memory-safe, contiguous, and compatible with Burn's gradient container model. The selected solution is dense gradient conversion performed lazily at the L-BFGS step boundary, with explicit buffer reuse and dispatch updates so the physics path can execute a true second-order finetune.

## Decision

Use **Dense Gradient Conversion** as the primary solution.

Why this is the chosen path:
- L-BFGS is mathematically a dense-history optimizer, so a contiguous dense gradient view is the correct representation for curvature updates.
- This preserves the convergence advantage of L-BFGS over Adam in late-stage smooth physics optimization.
- It is the least invasive fix compared with wrapping sparse containers in concurrency scaffolding or extending Burn internals with new gradient APIs.
- It can be implemented incrementally with clear rollback points and measurable gates.

## Problem Statement

Current state in the physics path:
- The nominal `pino-lbfgs` stage in `src-tauri/src/pino_burn_head.rs` is routed through a stable Adam fallback.
- The fallback exists because Burn's sparse-gradient container is not safe or structurally compatible with the custom L-BFGS two-loop recursion branch.
- This leaves the PINO runtime without a real second-order finetune phase, slowing or capping late-stage convergence.

## Implementation Plan

1. **Gate L0: Gradient container audit**
   Identify every place the current custom L-BFGS branch reads, clones, or indexes Burn gradients.
   Required output:
   - exact list of call sites in `src-tauri/src/pino_burn_head.rs`
   - exact assumptions currently made about gradient layout, missing entries, and ownership
   Acceptance gate:
   - one documented map of sparse-gradient touch points and failure modes is committed in this file or linked notes.

2. **Gate L1: Dense gradient extraction layer**
   Add a dedicated helper that converts Burn gradients into a contiguous dense vector only at the L-BFGS step boundary.
   Required behavior:
   - lazily convert sparse/missing parameter gradients into dense zero-filled slices
   - preserve deterministic parameter ordering across steps
   - separate parameter flattening order from optimizer math
   Acceptance gate:
   - helper returns stable dense vectors for repeated identical inputs
   - missing sparse entries are represented as explicit zeros
   - targeted tests prove contiguous ordering and length correctness

3. **Gate L2: Reusable dense buffers**
   Remove per-step heap churn by pre-allocating dense gradient, parameter, and search-direction buffers inside the L-BFGS state.
   Required behavior:
   - reuse the same buffers across optimizer steps
   - resize only when model parameter count changes
   - avoid hidden reallocation in the line-search hot path
   Acceptance gate:
   - benchmark or instrumentation confirms no repeated full-buffer allocation in steady state
   - parameter-count changes trigger safe resize and state reset

4. **Gate L3: Memory-safety and ownership cleanup**
   Ensure the dense conversion and L-BFGS state satisfy Rust aliasing and thread-safety rules without unsafe borrowing of Burn gradient internals.
   Required behavior:
   - no raw pointer aliasing against Burn-owned gradient storage
   - clear ownership handoff from Burn gradients to dense L-BFGS buffers
   - explicit `Send`/`Sync` reasoning documented if the optimizer state crosses threads
   Acceptance gate:
   - `cargo check` and targeted tests pass without fallback paths
   - no custom unsafe block is introduced unless documented and justified

5. **Gate L4: Real L-BFGS step restoration**
   Re-enable the custom two-loop recursion path using the dense gradients from Gate L1.
   Required behavior:
   - restore curvature history updates using dense `s` and `y` vectors
   - run a real backtracking line search over the physics loss
   - remove the Adam fallback from the `BurnFieldHeadOptimizer::Lbfgs` branch in `src-tauri/src/pino_burn_head.rs`
   Acceptance gate:
   - the `pino-lbfgs` branch executes true L-BFGS logic in tests and runtime logs
   - no implicit reroute to Adam remains

6. **Gate L5: Physics-path dispatch integration**
   Update the physics training dispatch so the new dense L-BFGS branch is explicitly allowed and selected when `optimizer_id == "pino-lbfgs"`.
   Required behavior:
   - `src-tauri/src/pinn.rs` and `src-tauri/src/pino_burn_head.rs` use the restored branch
   - runtime notes and progress reporting clearly indicate when L-BFGS is active
   Acceptance gate:
   - headless logs show `pino-lbfgs` as an actual second-order phase, not a label over Adam

7. **Gate L6: Verification and rollback guard**
   Validate correctness and stability before using the restored branch as the default late-stage optimizer.
   Required checks:
   - `cargo check`
   - targeted runtime tests for inference and architecture adaptation
   - headless `default-1e9-fast`
   - headless `default-1e9`
   - one checkpoint save/resume cycle through an L-BFGS phase
   Acceptance gate:
   - no panic, no fallback reroute, no NaN/Inf loss
   - late-stage loss is stable or improved versus the Adam-fallback baseline
   - rollback switch remains documented if signoff fails

## Recommended Implementation Strategy

- Convert gradients to dense only inside the L-BFGS step function, not during Adam epochs.
- Pre-allocate and reuse dense gradient/history buffers.
- Keep deterministic flatten/unflatten helpers as the single source of truth for parameter ordering.
- Do not modify Burn internals unless Gates L1-L4 fail twice without new signal.
- If dense conversion works but line search remains unstable, tune line-search safeguards only after the dense branch is fully active.

## Interaction With Remaining PINO Gates

This plan is not isolated. It unblocks the remaining high-value PINO gates:
- It closes the optimizer gap in the Burn-native physics path.
- It is required before judging whether the current operator body has truly reached its convergence limit.
- After Gate L6 passes, resume the remaining operator-body and holdout/signoff gates from:
  - `plans/DIRECT_REPLACE_PINO_UPGRADE_PLAN.md`
  - `plans/UPGRADE_EXECUTION_PROGRESS.md`

## Definition of Done

- `pino-lbfgs` is a real L-BFGS phase, not an Adam fallback.
- Dense gradient conversion is lazy, reusable, and deterministic.
- Physics-path logs, status, and checkpoints reflect actual L-BFGS execution.
- Headless default training and resume tests pass on the restored branch.
- The remaining PINO gates can continue on top of a true second-order finetune path.

## Status Update 2026-03-14

Implemented so far:
- Dense gradient extraction is now performed safely through `GradientsParams::remove` on Burn's inner backend in `src-tauri/src/pino_burn_head.rs`.
- The `BurnFieldHeadOptimizer::Lbfgs` branch now executes a real dense limited-memory BFGS loop with backtracking line search instead of reusing Adam.
- Direct unit coverage proves the L-BFGS branch reduces physics loss without fallback.
- Runtime coverage proves the explicit engine schedule reaches `pino-lbfgs` and records that phase in diagnostics.

Still pending from this plan:
- full headless signoff through a long-enough run that reaches the L-BFGS window naturally
- checkpoint save/resume across an L-BFGS phase
- final regression signoff against the current best training floor and holdout behavior
