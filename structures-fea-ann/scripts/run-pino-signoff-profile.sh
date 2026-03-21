#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/output/benchmarks"
mkdir -p "$OUT_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RAW="$OUT_DIR/pino-signoff-${TS}.log"
MD="$OUT_DIR/pino-signoff-${TS}.md"
LATEST="$OUT_DIR/pino-signoff-latest.md"

cd "$ROOT/src-tauri"
export PINO_SIGNOFF_FAST_PROFILE="${PINO_SIGNOFF_FAST_PROFILE:-1}"
cargo test --release pinn::tests::pino_release_signoff_profile_manual -- --ignored --nocapture >"$RAW"
cargo test --release pino::tests::pino_holdout_profile_manual -- --ignored --nocapture >>"$RAW"

MAIN_LINE="$(awk '/^pino-signoff-summary:/ {print; exit}' "$RAW")"
if [[ -z "${MAIN_LINE:-}" ]]; then
  echo "pino-signoff-profile: missing pino-signoff-summary in $RAW" >&2
  exit 1
fi

SIGNOFF_LINE="$MAIN_LINE"

{
  echo "# PINO Release Signoff Profile"
  echo
  echo "- Timestamp (UTC): ${TS}"
  echo "- Command 1: \`cargo test --release pinn::tests::pino_release_signoff_profile_manual -- --ignored --nocapture\`"
  echo "- Command 2: \`cargo test --release pino::tests::pino_holdout_profile_manual -- --ignored --nocapture\`"
  echo
  printf '%s\n' "$SIGNOFF_LINE"
  awk '/^pino-holdout-target-delta:/ {print}' "$RAW"
  awk '/^pino-holdout-gradnorm:/ {print}' "$RAW"
  awk '/^pino-holdout-candidate:/ {print}' "$RAW"
  awk '/^pino-holdout-train:/ {print}' "$RAW"
  awk '/^pino-holdout-summary:/ {print}' "$RAW"
} >"$MD"

cp "$MD" "$LATEST"
echo "pino-signoff-profile: wrote $MD and $LATEST"
