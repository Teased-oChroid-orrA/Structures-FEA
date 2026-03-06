# Structures FEA + Adaptive ANN

Offline, self-contained desktop application for plate mechanics analysis using Svelte 5 + TypeScript + Tauri 2 + Rust.

## Key capabilities
- 3D plate baseline case (10 in x 4 in x 0.125 in, 1000 lbf axial load)
- FEM solve endpoint with matrix formulation outputs
- Stress tensor, principal stresses, Von Mises, Tresca, max principal
- Thermal stress solver
- Dynamic (impact pulse) solver using implicit Newmark integration
- Failure criteria evaluation
- Adaptive ANN training/inference with topology growth and fallback gates
- One-click auto-training until target loss (with max-epoch safety cap)
- Adaptive topology growth and pruning (hidden neurons/layers can expand or contract)
- Live ANN training telemetry (epoch/loss/lr) with interactive network graph
- Local report export: JSON/CSV/PDF
- No cloud dependencies at runtime

## Run
```bash
npm install
npm run tauri:dev
```

## Check
```bash
npm run check
cd src-tauri && cargo check
```

## Windows compatibility
- Built on Tauri 2 (cross-platform desktop runtime)
- Uses std::path and relative-safe file handling
- No shell-specific runtime dependencies in app logic
