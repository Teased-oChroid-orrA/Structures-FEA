#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const root = process.cwd();
const baselinePath = join(root, 'docs/baselines/benchmark-lock.json');
const reportPath = join(root, 'output/benchmarks/latest.md');

const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
const report = readFileSync(reportPath, 'utf8');
const line = report
  .split('\n')
  .find((l) => l.startsWith('benchmark-summary:'));

if (!line) {
  console.error('benchmark-lock: FAILED missing benchmark-summary line');
  process.exit(1);
}

const valLossMatch = line.match(/val_loss=([0-9.eE+-]+)/);
const epochsMatch = line.match(/epochs=([0-9]+)/);
if (!valLossMatch || !epochsMatch) {
  console.error(`benchmark-lock: FAILED malformed summary: ${line}`);
  process.exit(1);
}

const valLoss = Number(valLossMatch[1]);
const epochs = Number(epochsMatch[1]);
if (!Number.isFinite(valLoss) || !Number.isFinite(epochs)) {
  console.error('benchmark-lock: FAILED non-finite metrics');
  process.exit(1);
}

if (valLoss > baseline.maxValLoss) {
  console.error(
    `benchmark-lock: FAILED val_loss ${valLoss} exceeds lock ${baseline.maxValLoss}`
  );
  process.exit(1);
}
if (epochs > baseline.maxEpochs) {
  console.error(`benchmark-lock: FAILED epochs ${epochs} exceeds lock ${baseline.maxEpochs}`);
  process.exit(1);
}

console.log(
  `benchmark-lock: PASS val_loss=${valLoss.toExponential(6)} epochs=${epochs} (limits ${baseline.maxValLoss}, ${baseline.maxEpochs})`
);
