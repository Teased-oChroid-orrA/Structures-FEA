#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/output/benchmarks"
mkdir -p "$OUT_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RAW="$OUT_DIR/benchmark-${TS}.log"
MD="$OUT_DIR/benchmark-${TS}.md"
LATEST="$OUT_DIR/latest.md"

cd "$ROOT/src-tauri"
cargo test ann::tests::benchmark_profile_epoch_windows_manual -- --ignored --nocapture >"$RAW"

{
  echo "# Benchmark Profile"
  echo
  echo "- Timestamp (UTC): ${TS}"
  echo "- Command: \`cargo test ann::tests::benchmark_profile_epoch_windows_manual -- --ignored --nocapture\`"
  echo
  awk '
    /^\| epoch \| val_data_loss \| val_physics_loss \| lr_phase \| optimizer \|/ {p=1}
    /^benchmark-summary:/ {if (!seen++) print; p=0; next}
    p==1 {print}
  ' "$RAW"
} >"$MD"

cp "$MD" "$LATEST"
echo "benchmark-profile: wrote $MD and $LATEST"
