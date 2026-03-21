# 3D Navier-Cauchy PINO Recovery Plan

Updated: 2026-03-18 (America/Chicago)

## Summary

The current 3D PINO is blocked by hotspot conditioning, not missing volumetric plumbing. Research and branch evidence point to three main causes:

1. Dirichlet boundary conditions are still enforced mainly through penalties.
2. The physics objective is still too strong-form dominant in stress-concentration regimes.
3. Local refinement and enrichment are not yet strong enough to resolve near-singular fields efficiently.

The implementation order is fixed:

1. Boundary-exact displacement embedding
2. Stabilized mixed strong/weak elasticity objective
3. Generic residual-driven local patch refinement
4. Generic local hotspot enrichment
5. Signoff recovery and rollout into the main path

## Gates And Tickets

### Gate 1: Boundary Conditioning

Exit criteria:
- Displacements are zero by construction on Dirichlet regions in the primary 3D PINO path.
- Clamp penalties are no longer carrying the optimization.
- Runtime and Burn paths remain aligned.

Tickets:
- `G1-T1` Add displacement embedding to the operator decode so `ux/uy/uz` are multiplied by a boundary factor that is exactly zero on clamped cells.
- `G1-T2` Thread the same embedding through Burn physics samples and loss evaluation.
- `G1-T3` Add additive runtime metadata for `boundaryMode`.
- `G1-T4` Add tests proving clamp cells remain exactly zero even with nonzero learned corrections.

### Gate 2: Stabilized Physics Objective

Exit criteria:
- Hotspot regimes are no longer trained with a mostly strong-form-only objective.
- Plate-hole and cantilever remain finite under stronger local supervision.
- Weak/energy terms are visible in diagnostics.

Tickets:
- `G2-T1` Add a weak-form or elastic-energy loss term for 3D linear elasticity.
- `G2-T2` Rebalance mixed displacement/stress supervision around strong plus weak terms.
- `G2-T3` Add additive runtime metadata for `objectiveMode`.
- `G2-T4` Update checkpoint scoring to include bounded hotspot quality, not just diffuse field loss.

### Gate 3: Generic Residual-Driven Local Refinement

Exit criteria:
- Refinement is truly local and applicable to any structural hotspot.
- Compact signoff remains usable as the iteration loop.
- Refinement either improves hotspot score or exits early.

Tickets:
- `G3-T1` Score per-cell residuals and cluster top hotspots into local patches.
- `G3-T2` Upsample only the patch-local grid, train a short correction step, and project back to the base grid.
- `G3-T3` Add hard limits for patch count, refined cells, and patch steps.
- `G3-T4` Log hotspot locations, refined shapes, and before/after hotspot metrics.

### Gate 4: Generic Local Enrichment

Exit criteria:
- The operator has explicit local representational support for hotspot fields.
- The same enrichment path works for holes, roots, notches, and future regimes.

Tickets:
- `G4-T1` Add hotspot-local enrichment features: radial distance, angular coordinates, and normalized boundary-distance gradients.
- `G4-T2` Add a lightweight local correction head that runs only on refined patches.
- `G4-T3` Initialize patch centers from residual clusters first and geometry priors second.

### Gate 5: Signoff Recovery

Exit criteria:
- `benchmark:pino-signoff` and `verify:pino-signoff-lock` pass on the 3D path.
- Compact signoff remains the primary iteration harness.

Tickets:
- `G5-T1` Run compact signoff after each gate and track cantilever, axial, and plate-hole separately.
- `G5-T2` Promote the same refinement/enrichment path into default training.
- `G5-T3` Remove per-regime optimizer rescue logic once the generic path is sufficient.

## Important Interface Changes

- Additive PINO metadata:
  - `boundaryMode`
  - `objectiveMode`
  - `localRefinement`
  - `localEnrichment`
- Keep Tauri command shapes stable.

## Test Plan

- Exact clamp embedding yields zero displacement on Dirichlet cells.
- Runtime/Burn parity remains valid with embedding enabled.
- Compact residual-refine path finishes within a bounded runtime budget.
- Plate-hole hotspot metrics improve without regressing cantilever and axial beyond lock margins.

## Defaults

- Linear elasticity only.
- Keep the current 3D volumetric operator core.
- Prefer residual-driven hotspot detection over geometry-specific heuristics.
- Do not start Gate 4 before Gates 1 through 3 are complete.
