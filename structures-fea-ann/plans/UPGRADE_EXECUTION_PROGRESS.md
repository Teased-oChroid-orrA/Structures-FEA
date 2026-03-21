# Upgrade Execution Progress Board

Updated: 2026-03-19 (America/Chicago)

## Gates

| Gate | Scope | Status | Evidence |
|---|---|---|---|
| G1 | Boundary conditioning | Completed | Exact displacement embedding is implemented in runtime and Burn; clamp-proof tests pass. |
| G2 | Stabilized physics objective | Completed | Weak/energy elasticity support and hotspot-aware checkpoint scoring are implemented and verified at compile/parity level. |
| G3 | Generic residual-driven local refinement | Completed | Residual-ranked hotspot refinement and generic patch-focus weighting are implemented in the compact path, with additive metadata and diagnostics. |
| G4 | Generic local enrichment | Completed | The operator basis now includes generic hotspot-local enrichment features and preserves prior weights during basis expansion. |
| G5 | 3D signoff recovery | In Progress | Compact 3D signoff remains the primary harness. Deep evaluation shows two real blockers: the fast holdout harness is circular/expensive because it nests inner holdout refreshes inside an outer holdout loop, and the current hotspot patch-focus logic is helping interior hole hotspots while over-localizing the boundary-dominated axial/cantilever regimes. |

## Current Blocker

The current blocker is not missing gate implementation. It is branch quality on the combined recovery stack. Gates 1 through 4 are implemented, and two concrete regression fixes have now landed on top of that stack:

1. Field-head realignment now preserves overlapping learned weights and biases when the enrichment basis changes, instead of reinitializing the whole operator shell.
2. Boundary embedding now acts on the learned displacement correction only, not on the entire displacement field, which removes artificial damping of the physical prior away from the clamp.

Targeted compact `plate-hole` verification has completed on that corrected branch and recovered to a narrow remaining gap. The next work is no longer catastrophic-regression recovery; it is finishing the remaining `plate-hole` mean-von-Mises gap, then rerunning axial, cantilever, and full signoff.

## Immediate Next

1. Re-run compact `axial` on the corrected branch with one axial-only lever at a time.
2. If `axial` remains above lock, continue axial-only stress tuning under env-gated experiments.
3. Rerun compact `cantilever` once the axial branch is stable.
4. Keep the fast filtered holdout harness on the new non-circular path, then rerun `axial` and `cantilever` to confirm the speedup did not change winners.
5. Revisit `plate-hole` with a different narrow lever only if axial/cantilever stay healthy and global signoff is still blocked.
6. Run full `benchmark:pino-signoff`.
7. Run `verify:pino-signoff-lock`.
8. Promote the same corrected path into the default training flow.

## Todo Tracker

| Task | Status | Evidence / Next |
|---|---|---|
| S1 regression-gate runner | Done | `scripts/run-pino-regression-gates.sh` added to run `cargo check` plus filtered holdouts in a fixed order. |
| S2 regression-gate verifier | Done | `scripts/verify-pino-regression-gates.mjs` and `docs/baselines/pino-regression-gates.json` added. |
| S3 env-gated behavioral knobs | Done | Hotspot/ring behavioral levers now read from env vars instead of requiring direct path edits. |
| S4 one-lever experiment rule | Active | Defaults stay on the last winning branch values; losing behavioral changes are reverted immediately. Current live lever is `PINO_CANTILEVER_SPAN_BAND_BLEND`, env-gated and isolated to vertical-dominant non-hole cases. |
| S5 non-circular filtered holdout harness | Done | Filtered fast holdout runs no longer rerun inner epoch-level holdout refreshes inside the outer regime-selection loop. |
| G1-T1/T2 displacement embedding | Done | Runtime + Burn embedding landed in `pino.rs`, `pino_burn_head.rs`, `pinn.rs`. |
| G1-T3 metadata | Done | `boundaryMode` added to Rust/TS contracts. |
| G1-T4 clamp proof tests | Done | `embedded_dirichlet_keeps_clamped_displacements_zero` passes. |
| G2-T1 weak/energy term | Done | `weak_energy` added to physics loss and diagnostics. |
| G2-T2 strong/weak rebalance | Done | Constitutive/material pillar now includes weak-energy support. |
| G2-T3 metadata | Done | `objectiveMode` added to runtime metadata. |
| G2-T4 hotspot-aware checkpoint scoring | Done | `hotspot_selection_score(...)` is live in the main training loop. |
| G3-T1 hotspot scoring | Done | Residual hotspot ranking and patch centers are computed per-sample. |
| G3-T2 local refinement | Done | Compact path supports ranked hotspot grid refinement and patch-focus weighting. |
| G3-T3 hard limits | Done | Patch count and refine budget are bounded in metadata/logic. |
| G3-T4 diagnostics | Done | Residual-refine diagnostics and hotspot metadata are emitted. |
| G4-T1 enrichment features | Done | Basis expanded from 27 to 33 with generic hotspot-local features. |
| G4-T2 local correction path | Deferred | Not required yet; only if the preserved-shell + correction-only boundary branch still underperforms. |
| G4-T3 residual-first hotspot init | Done | Patch centers are selected from residual peaks first. |
| G5-T1 targeted compact signoff | In Progress | `plate-hole` recovery is stable at `mean_vm=5.283974e-2`, `p95=5.317193e-2`, `ratio=1.056795`. The current best axial-only lever is `PINO_PATCH_FOCUS_GAIN=0`, which improved axial baseline sharply. Cantilever verification on the same lever improved baseline to `mean_disp=3.253050e-1`, `mean_vm=1.344808e-1`, `p95=3.399719e-1`, `ratio=6.506100`. The next cantilever-only lever, `PINO_CANTILEVER_SPAN_BAND_BLEND=0.35` on top of `PINO_PATCH_FOCUS_GAIN=0`, improved the first filtered `adam` candidate further to `mean_disp=5.646650e-2`, `mean_vm=1.329148e-1`, `p95=1.409909e-1`, `ratio=2.658296`. Deep evaluation shows the remaining issue is regime coupling: generic patch-focus helps interior hole hotspots but over-localizes boundary-dominated axial/cantilever supervision. A generic hotspot-density/interiority auto-gate was tested and reverted after it regressed axial displacement/trust. |
| G5-T2 promote into default training | Pending | Hold until compact signoff is healthy again. |
| G5-T3 remove rescue heuristics | Pending | Hold until generic path passes signoff. |

## Current Verification Notes

- `cargo check` passes on the corrected branch after:
  - preserved-weight basis realignment
  - correction-only displacement embedding
- Latest corrected compact `plate-hole` signals on the preserved-shell branch:
  - trained-shell baseline:
    - `score=4.895880e0`
    - `disp=2.609138e-1`
    - `vm=5.344264e-2`
    - `p95=2.609970e-1`
    - `ratio=5.218275e0`
  - default enriched-shell baseline:
    - `score=3.006900e-1`
    - `disp=7.667417e-5`
    - `vm=5.283974e-2`
    - `p95=5.317193e-2`
    - `ratio=1.056795e0`
- Current implication:
  - the shell-preservation and correction-only boundary fixes worked
  - the compact path must start from the better default enriched shell when it outperforms the trained shell
  - the remaining `plate-hole` blocker is now narrow again: `mean_vm` is still about `0.00284` above the `0.05` lock
  - the stronger ring-loss weight (`0.44`) did not help and was reverted; the branch remains on the better `0.38` setting
  - deep evaluation also identified a harness problem: filtered fast holdout tests were nesting inner epoch-level holdout refreshes inside the outer holdout loop, which wasted time and obscured attribution; that path is now disabled for filtered fast runs
  - axial on the corrected branch is still outside gate:
    - previous baseline:
      - `mean_disp=1.681547e-1`
      - `mean_vm=7.369511e-2`
      - `p95=1.682222e-1`
      - `ratio=3.363094`
    - best current axial-only lever (`PINO_PATCH_FOCUS_GAIN=0`) baseline:
      - `mean_disp=2.948032e-2`
      - `mean_vm=5.343343e-2`
      - `p95=6.519845e-2`
      - `ratio=1.068669`
  - filtered `lbfgs` candidates were all worse than the axial baseline
  - disabling residual refinement for axial did not help
  - disabling patch focus for axial produced the first strong axial improvement; first filtered `lbfgs` candidate was worse than that new baseline
  - a generic hotspot-density/interiority auto-gate was tested as a replacement for the axial-only override, but it regressed axial badly:
    - `mean_disp=2.726620e-1`
    - `mean_vm=4.388425e-2`
    - `p95=2.727652e-1`
    - `ratio=5.453239`
  - that experiment was reverted immediately; the winning axial branch remains the explicit `PINO_PATCH_FOCUS_GAIN=0` lever
  - cantilever on the corrected branch is still outside gate:
    - baseline:
      - `mean_disp=3.676137e-1`
      - `mean_vm=1.360348e-1`
      - `p95=3.827488e-1`
      - `ratio=7.352275`
  - baseline-default improved `mean_vm` alone but badly regressed displacement/trust:
    - `mean_disp=8.251472e-1`
    - `mean_vm=7.693025e-2`
    - `p95=8.453457e-1`
    - `ratio=1.650294e1`
  - filtered `lbfgs` cantilever candidates were all flat/worse than the cantilever baseline
  - active cantilever single-lever test is `PINO_PATCH_FOCUS_GAIN=0`, because the current cantilever path is strongly root-edge hotspot focused and the same lever materially improved axial
  - confirmed cantilever signal on that lever is better than the prior baseline:
    - `mean_disp=3.253050e-1`
    - `mean_vm=1.344808e-1`
    - `p95=3.399719e-1`
    - `ratio=6.506100`
  - independent evaluation also identified a deeper cantilever mismatch:
    - the current cantilever observable geometry is still root-localized, while the holdout gate scores broad bending transfer using tip displacement and global VM
    - that makes cantilever less responsive to generic patch-focus changes than axial

## Follow-On Sequence

1. `plate-hole`: rerun after the ring-local VM weight bump. Result: slightly worse; reverted.
2. `axial`: rerun compact filtered holdout on the corrected branch. Result: baseline won; filtered `lbfgs` candidates were worse.
3. `axial`: disable residual refinement for this regime only. Result: no improvement; baseline still won.
4. `axial`: disable hotspot patch focus for this regime only. Result: strong baseline improvement; first filtered `lbfgs` candidate worse, so this is the current best axial-only branch.
5. `cantilever`: rerun compact filtered holdout on the corrected branch. Result: baseline won; filtered `lbfgs` candidates were flat/worse.
6. `cantilever`: disable hotspot patch focus for this regime only. Result: confirmed improvement over the prior baseline.
7. Do not pursue the generic hotspot-density/interiority gate further; it failed axial and was reverted.
8. `cantilever`: baseline-check the env-gated span-distributed bending-band projection with `PINO_CANTILEVER_SPAN_BAND_BLEND` on top of the current winning `PINO_PATCH_FOCUS_GAIN=0` branch. Result: no regression, but baseline metrics were unchanged.
9. `cantilever`: rerun a small `adam` candidate sweep on the same lever. Result: strong improvement over the current cantilever branch, especially displacement, p95, and ratio.
10. `axial`: confirm the same win under regime-aware patch-focus split using `PINO_PATCH_FOCUS_GAIN_NON_HOLE=0`. Result: exact match to the prior winning axial branch.
11. `plate-hole`: confirm the split with `PINO_PATCH_FOCUS_GAIN_NON_HOLE=0`. Result: exact match to the preserved winning `plate-hole` branch because hole-local patch focus remains active.
12. Current live task: test `PINO_CANTILEVER_FIFTH_OBSERVABLE_SPAN_BLEND=0.45` on top of the shared branch:
    - `PINO_PATCH_FOCUS_GAIN_NON_HOLE=0`
    - `PINO_CANTILEVER_SPAN_BAND_BLEND=0.35`
    - one filtered `adam` candidate only
13. Shared full-signoff env bundle, once cantilever is ready:
    - `PINO_SIGNOFF_FAST_PROFILE=1`
    - `PINO_SIGNOFF_RESIDUAL_REFINE=1`
    - `PINO_PATCH_FOCUS_GAIN_NON_HOLE=0`
    - `PINO_CANTILEVER_SPAN_BAND_BLEND=0.35`
14. If the current fifth-observable cantilever candidate loses, next fallback lever is:
    - `PINO_CANTILEVER_ROOT_EDGE_SCALE`
    - target first test: `0.35`
15. Next: if the active cantilever candidate wins, run the current best shared signoff branch with:
    - hole patch focus kept on
    - non-hole patch focus disabled
    - cantilever span-band observable enabled
10. Full compact signoff: `npm run benchmark:pino-signoff`
11. Lock verification: `npm run verify:pino-signoff-lock`
12. Default-training promotion: carry the corrected compact path back into the default training branch and re-check headless health.

## Experiment Standards

1. Every behavioral experiment must target one lever only.
2. Every behavioral experiment must pass the regression-gate harness before it is kept.
3. Defaults remain pinned to the last winning values; experimentation is env-gated first.
4. Losing behavioral experiments are reverted and logged here before the next task begins.
