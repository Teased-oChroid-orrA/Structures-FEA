#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const root = process.cwd();
const baselinePath = join(root, 'docs/baselines/pino-regression-gates.json');
const reportPath = join(root, 'output/benchmarks/pino-regression-gates-latest.md');

const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
const report = readFileSync(reportPath, 'utf8');
const lines = report.split('\n');

const fail = (msg) => {
  console.error(`pino-regression-gates: FAILED ${msg}`);
  process.exit(1);
};

const summaryLines = lines.filter((line) => line.startsWith('pino-holdout-summary:'));
if (summaryLines.length === 0) {
  fail('missing pino-holdout-summary lines');
}

const metricsByRegime = new Map();
for (const line of summaryLines) {
  const regime = line.match(/regime=([a-zA-Z0-9-]+)/)?.[1];
  const meanDisp = Number(line.match(/mean_disp=([0-9.eE+-]+)/)?.[1]);
  const meanVm = Number(line.match(/mean_vm=([0-9.eE+-]+)/)?.[1]);
  const p95 = Number(line.match(/p95=([0-9.eE+-]+)/)?.[1]);
  const ratio = Number(line.match(/ratio=([0-9.eE+-]+)/)?.[1]);
  if (
    !regime ||
    !Number.isFinite(meanDisp) ||
    !Number.isFinite(meanVm) ||
    !Number.isFinite(p95) ||
    !Number.isFinite(ratio)
  ) {
    fail(`malformed holdout summary line: ${line}`);
  }
  metricsByRegime.set(regime, { meanDisp, meanVm, p95, ratio });
}

for (const [regime, limits] of Object.entries(baseline.regimes ?? {})) {
  const metrics = metricsByRegime.get(regime);
  if (!metrics) {
    fail(`missing regime ${regime}`);
  }
  if (metrics.meanDisp > limits.maxMeanDisp) {
    fail(`${regime} mean_disp ${metrics.meanDisp} exceeds gate ${limits.maxMeanDisp}`);
  }
  if (metrics.meanVm > limits.maxMeanVm) {
    fail(`${regime} mean_vm ${metrics.meanVm} exceeds gate ${limits.maxMeanVm}`);
  }
  if (metrics.p95 > limits.maxP95) {
    fail(`${regime} p95 ${metrics.p95} exceeds gate ${limits.maxP95}`);
  }
  if (metrics.ratio > limits.maxRatio) {
    fail(`${regime} ratio ${metrics.ratio} exceeds gate ${limits.maxRatio}`);
  }
}

console.log(
  `pino-regression-gates: PASS regimes=${Object.keys(baseline.regimes ?? {}).join(',')}`
);
