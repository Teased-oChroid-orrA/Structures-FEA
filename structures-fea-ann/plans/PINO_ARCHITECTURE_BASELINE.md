# PINO Architecture Baseline

## Scope
This baseline defines the direct-replace PINO runtime contract for the 2D plate-family path in `structures-fea-ann`.

Supported v1 regimes:
- cantilever
- axial plate
- plate-with-hole

Out of scope for v1:
- true volumetric 3D PINO
- contact-first operator training
- replacing the existing FEM solver

## Runtime Split
- Burn owns model definition, module composition, and autodiff.
- Candle is the accelerated execution backend when available.
- `burn-ndarray` remains the deterministic CPU fallback for tests and local development.

## Canonical Runtime Backends
- `pino-ndarray-cpu`
- `pino-candle-cpu`
- `pino-candle-cuda`
- `pino-candle-metal`

Legacy aliases still accepted during migration:
- `burn-ndarray-cpu` -> `pino-ndarray-cpu`
- `burn-wgpu` -> `pino-candle-cpu`

## Operator Domain
Each structural case is encoded onto a fixed 2D midsurface grid with condition channels.

Current canonical input channels:
1. normalized x
2. normalized y
3. geometry mask
4. thickness
5. signed hole distance
6. axial load
7. vertical load
8. elastic modulus
9. Poisson ratio
10. clamp indicator

Current canonical output channels:
1. `ux`
2. `uy`
3. `sxx`
4. `syy`
5. `txy`
6. `von_mises`
7. `max_principal`

## Upgrade Sequence
1. Shared contract and backend naming
2. Operator-grid encode/decode foundation
3. Spectral/Fourier layer core
4. PINO model assembly and physics residual graph
5. Runtime replacement of current ANN/Burn pilot path
6. Holdout-calibrated safeguards and UI observability

## Acceptance Expectation For This Baseline
- Cargo features define the intended backend matrix.
- The app remains buildable on CPU fallback.
- The runtime and frontend can surface PINO metadata without breaking existing command contracts.
