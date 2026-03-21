#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/output/benchmarks"
mkdir -p "$OUT_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RAW="$OUT_DIR/pino-regression-gates-${TS}.log"
MD="$OUT_DIR/pino-regression-gates-${TS}.md"
LATEST="$OUT_DIR/pino-regression-gates-latest.md"

REGIMES="${PINO_REGRESSION_REGIMES:-plate-hole axial cantilever}"
export PINO_SIGNOFF_FAST_PROFILE="${PINO_SIGNOFF_FAST_PROFILE:-1}"
export PINO_SIGNOFF_RESIDUAL_REFINE="${PINO_SIGNOFF_RESIDUAL_REFINE:-1}"

cd "$ROOT/src-tauri"

{
  echo "pino-regression-gates-config: fast_profile=${PINO_SIGNOFF_FAST_PROFILE} residual_refine=${PINO_SIGNOFF_RESIDUAL_REFINE} regimes=${REGIMES}"
  cargo check
} >"$RAW" 2>&1

for regime in $REGIMES; do
  {
    echo "pino-regression-gates-run: regime=${regime}"
    PINO_SIGNOFF_REGIME_FILTER="$regime" cargo test --release pino::tests::pino_holdout_profile_manual -- --ignored --nocapture
  } >>"$RAW" 2>&1
done

{
  echo "# PINO Regression Gates"
  echo
  echo "- Timestamp (UTC): ${TS}"
  echo "- Command 1: \`cargo check\`"
  for regime in $REGIMES; do
    echo "- Command regime ${regime}: \`PINO_SIGNOFF_REGIME_FILTER=${regime} cargo test --release pino::tests::pino_holdout_profile_manual -- --ignored --nocapture\`"
  done
  echo
  awk '/^pino-regression-gates-config:/ {print}' "$RAW"
  awk '/^pino-regression-gates-run:/ {print}' "$RAW"
  awk '/^pino-holdout-target-delta:/ {print}' "$RAW"
  awk '/^pino-holdout-gradnorm:/ {print}' "$RAW"
  awk '/^pino-holdout-candidate:/ {print}' "$RAW"
  awk '/^pino-holdout-train:/ {print}' "$RAW"
  awk '/^pino-holdout-summary:/ {print}' "$RAW"
} >"$MD"

cp "$MD" "$LATEST"
echo "pino-regression-gates: wrote $MD and $LATEST"
