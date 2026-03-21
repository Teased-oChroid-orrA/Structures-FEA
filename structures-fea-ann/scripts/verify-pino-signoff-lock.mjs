#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const root = process.cwd();
const baselinePath = join(root, 'docs/baselines/pino-signoff-lock.json');
const reportPath = join(root, 'output/benchmarks/pino-signoff-latest.md');

const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
const report = readFileSync(reportPath, 'utf8');

const mainLine = report.split('\n').find((line) => line.startsWith('pino-signoff-summary:'));
if (!mainLine) {
  console.error('pino-signoff-lock: FAILED missing pino-signoff-summary line');
  process.exit(1);
}

const parseMainMetric = (name) => {
  const match = mainLine.match(new RegExp(`${name}=([0-9.eE+-]+)`));
  return match ? Number(match[1]) : NaN;
};

const throughput = parseMainMetric('throughput_eps');
const inferLatency = parseMainMetric('infer_latency_ms');
const memoryEstimate = parseMainMetric('memory_estimate_mb');
const resumeOk = parseMainMetric('resume_ok');
const valLoss = parseMainMetric('val_loss');
const epochs = parseMainMetric('epochs');

if (
  !Number.isFinite(throughput) ||
  !Number.isFinite(inferLatency) ||
  !Number.isFinite(memoryEstimate) ||
  !Number.isFinite(resumeOk) ||
  !Number.isFinite(valLoss) ||
  !Number.isFinite(epochs)
) {
  console.error(`pino-signoff-lock: FAILED malformed signoff summary: ${mainLine}`);
  process.exit(1);
}

const fail = (msg) => {
  console.error(`pino-signoff-lock: FAILED ${msg}`);
  process.exit(1);
};

const holdoutLines = report
  .split('\n')
  .filter((line) => line.startsWith('pino-holdout-summary:'));
if (holdoutLines.length === 0) {
  fail('missing pino-holdout-summary lines');
}

const holdoutByRegime = new Map();
for (const line of holdoutLines) {
  const regime = line.match(/regime=([a-zA-Z0-9-]+)/)?.[1];
  const meanDisp = Number(line.match(/mean_disp=([0-9.eE+-]+)/)?.[1]);
  const meanVm = Number(line.match(/mean_vm=([0-9.eE+-]+)/)?.[1]);
  const p95 = Number(line.match(/p95=([0-9.eE+-]+)/)?.[1]);
  if (!regime || !Number.isFinite(meanDisp) || !Number.isFinite(meanVm) || !Number.isFinite(p95)) {
    fail(`malformed holdout line: ${line}`);
  }
  holdoutByRegime.set(regime, { meanDisp, meanVm, p95 });
}

const holdoutBaseline = baseline.holdout ?? {};
for (const [regime, limits] of Object.entries(holdoutBaseline)) {
  const metrics = holdoutByRegime.get(regime);
  if (!metrics) {
    fail(`missing holdout regime ${regime}`);
  }
  if (metrics.meanDisp > limits.maxMeanDisp) {
    fail(
      `${regime} mean_disp ${metrics.meanDisp} exceeds lock ${limits.maxMeanDisp}`
    );
  }
  if (metrics.meanVm > limits.maxMeanVm) {
    fail(
      `${regime} mean_vm ${metrics.meanVm} exceeds lock ${limits.maxMeanVm}`
    );
  }
  if (metrics.p95 > limits.maxP95) {
    fail(`${regime} p95 ${metrics.p95} exceeds lock ${limits.maxP95}`);
  }
}

if (throughput < baseline.minThroughputEps) {
  fail(`throughput_eps ${throughput} below lock ${baseline.minThroughputEps}`);
}
if (inferLatency > baseline.maxInferLatencyMs) {
  fail(`infer_latency_ms ${inferLatency} exceeds lock ${baseline.maxInferLatencyMs}`);
}
if (memoryEstimate > baseline.maxMemoryEstimateMb) {
  fail(`memory_estimate_mb ${memoryEstimate} exceeds lock ${baseline.maxMemoryEstimateMb}`);
}
if (Math.round(resumeOk) !== Math.round(baseline.requireResumeOk)) {
  fail(`resume_ok ${resumeOk} does not satisfy lock ${baseline.requireResumeOk}`);
}
if (valLoss > baseline.maxValLoss) {
  fail(`val_loss ${valLoss} exceeds lock ${baseline.maxValLoss}`);
}
if (epochs > baseline.maxEpochs) {
  fail(`epochs ${epochs} exceeds lock ${baseline.maxEpochs}`);
}

console.log(
  `pino-signoff-lock: PASS throughput=${throughput.toFixed(3)} eps infer_latency=${inferLatency.toFixed(
    3
  )} ms memory=${memoryEstimate.toFixed(3)} MB val_loss=${valLoss.toExponential(6)} epochs=${epochs}`
);
