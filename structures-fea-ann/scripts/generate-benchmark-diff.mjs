#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const root = process.cwd();
const baselinePath = join(root, 'docs/baselines/benchmark-lock.json');
const reportPath = join(root, 'output/benchmarks/latest.md');
const outDir = join(root, 'output/benchmarks');
const outPath = join(outDir, 'lock-diff.md');

const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
const report = readFileSync(reportPath, 'utf8');
const summary = report.split('\n').find((l) => l.startsWith('benchmark-summary:')) ?? '';
const valLoss = Number((summary.match(/val_loss=([0-9.eE+-]+)/) ?? [])[1]);
const epochs = Number((summary.match(/epochs=([0-9]+)/) ?? [])[1]);

if (!Number.isFinite(valLoss) || !Number.isFinite(epochs)) {
  console.error('benchmark-diff: FAILED unable to parse benchmark-summary');
  process.exit(1);
}

const valMargin = baseline.maxValLoss - valLoss;
const epochMargin = baseline.maxEpochs - epochs;
const status = valMargin >= 0 && epochMargin >= 0 ? 'PASS' : 'FAIL';

mkdirSync(outDir, { recursive: true });
writeFileSync(
  outPath,
  `# Benchmark Lock Diff\n\n` +
    `- Status: **${status}**\n` +
    `- Summary: \`${summary}\`\n\n` +
    `| Metric | Current | Lock | Margin |\n` +
    `|---|---:|---:|---:|\n` +
    `| val_loss | ${valLoss.toExponential(6)} | ${baseline.maxValLoss} | ${valMargin.toExponential(6)} |\n` +
    `| epochs | ${epochs} | ${baseline.maxEpochs} | ${epochMargin} |\n`
);

console.log(`benchmark-diff: wrote ${outPath}`);
