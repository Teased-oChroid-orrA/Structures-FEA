# Benchmark Cantilever Displacement-Primary Execution Plan

## Objective

Replace the current shared-head benchmark cantilever exact lane with a benchmark-specific
displacement-primary path that is judged on one consistent physical metric.

The immediate goal is not to claim generalized success. It is to make
`benchmark_cantilever_2d` numerically honest, diagnostically stable, and structurally
aligned with the physics it is supposed to satisfy.

## Gates

### Gate A - Metric Consistency

- Stage progress, exact-refine, selection, and certification all use the same physical
  cantilever score.
- No stale generic field loss remains in the benchmark GUI/headless path.

Status: completed

### Gate B - 2D Physics Correctness

- Single-layer benchmark uses plane-stress reconstruction and single-layer operator grids.
- No out-of-plane equilibrium, energy, or traction terms remain in the formal 2D benchmark.

Status: completed

### Gate C - Displacement-Primary Exact Surface

- Benchmark exact refine stops treating raw stress channels as the authoritative physical
  state.
- Stress-fit, equilibrium, traction, and observables are evaluated from displacement-derived
  stress on the benchmark exact lane.
- Raw stress-channel constitutive penalties are removed from the benchmark exact lane.

Status: attempted, not kept

### Gate D - Benchmark Floor Revalidation

- Run `benchmark-cantilever-fast` on the real headless path.
- Keep the branch only if the benchmark floor improves or the new floor is demonstrably more
  physically faithful than the legacy branch.

Status: completed

### Gate E - Follow-On Redesign

- If Gate D still plateaus materially above `1e-4`, split the benchmark head itself into a
  true 3-channel displacement head instead of continuing to retrofit the shared 11-output
  head.

Status: in progress

## Tickets

- `C1` Route benchmark exact-surface stress metrics through displacement-derived stress.
- `C2` Disable raw stress constitutive penalties on the benchmark exact surface.
- `C3` Keep legacy/shared exact surface unchanged for non-benchmark paths.
- `D1` Run the real headless benchmark and compare floor/runtime/breakdown to the prior kept
  branch.
- `D2` If improved, keep and document the new dominant blocker.
- `E1` If not improved enough, start a benchmark-only 3-channel head branch.
- `E2` Carry `output_dim=3` through runtime config, forward pass, burn sample prep, and
  exact-refine evaluation while keeping legacy 11-channel paths unchanged.
- `E3` Reconstruct benchmark stresses from displacement before benchmark scoring and
  certification.
- `E4` Keep `benchmark_cantilever_2d` exact-refine isolated on the single anchor case; no
  family broadening on the formal benchmark gate.
- `E5` Add a self-consistency regression so the benchmark target’s own displacement
  reconstruction must remain close to its stored stress field.

## Execution Notes

- 2026-03-30: benchmark-only exact-surface retrofit was implemented and validated on the real
  `benchmark-cantilever-fast` headless path.
- Result: not kept. It zeroed raw-stress constitutive penalties on the benchmark exact lane, but
  the physical benchmark score stayed slightly worse than the current best branch.
- Conclusion: the next keeper cannot be a partial exact-surface retrofit on the shared 11-output
  head. It has to be a true benchmark-only 3-channel displacement head / evaluator.
- 2026-03-30: benchmark-only 3-channel displacement head branch is now partially live.
- Real validation path: `benchmark-cantilever-fast` now runs with `arch=[15, 40, 40, 3]` and
  exact-refine stays on that 3-channel path end to end.
- Current floor on this branch: about `4.200154e0` on the formal benchmark metric, with
  displacement errors improved but mean von Mises still dominating physical failure.
- 2026-03-30: the benchmark exact-refine path no longer broadens to extra cantilever family
  cases; the formal benchmark gate now stays on `evalCases=1, trainCases=1`.
- 2026-03-30: the benchmark exact target itself now reconciles to displacement-derived stress on
  the 3-channel path. The new self-consistency regression passes, and the real
  `benchmark-cantilever-fast` floor improved from about `4.199485e0` to `2.965901e0`.
- 2026-03-30: `benchmark_cantilever_2d` exact-refine is now truly isolated in runtime notes and
  real headless execution: `evalCases=1, trainCases=1`, `arch=[15, 40, 40, 3]`, latest verified
  floor `2.965901e0`.
- Next remaining work on Gate E: benchmark-specific target self-consistency and stress
  reconstruction / scoring quality inside the benchmark head, not more hidden legacy-head
  cleanup.

## Success Criteria

- Short term: lower or more physically trustworthy `benchmark_cantilever_2d` floor than the
  current shared-head path.
- Medium term: benchmark reaches `<= 1e-2` reproducibly.
- Long term: benchmark reaches `<= 1e-4` before promotion to harder families.
