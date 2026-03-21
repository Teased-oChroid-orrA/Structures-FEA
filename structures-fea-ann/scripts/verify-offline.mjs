#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { execSync } from 'node:child_process';

const root = process.cwd();
const targets = ['src', 'src-tauri/src'];
const forbidden = [
  { name: 'http-url', re: /https?:\/\//g },
  { name: 'ws-url', re: /wss?:\/\//g },
  { name: 'fetch-call', re: /\bfetch\s*\(/g },
  { name: 'xmlhttprequest', re: /XMLHttpRequest/g },
  { name: 'reqwest', re: /\breqwest\b/g }
];

const allow = [
  /http:\/\/127\.0\.0\.1/,
  /https:\/\/tauri\.app/,
  /https:\/\/svelte\.dev/,
  /https:\/\/meshlib\.io/,
  /https:\/\/svelteflow\.dev/
];

const files = execSync(
  `cd ${JSON.stringify(root)} && rg --files ${targets.map((t) => JSON.stringify(t)).join(' ')}`,
  { encoding: 'utf8' }
)
  .split('\n')
  .filter(Boolean)
  .filter((f) => /\.(ts|js|svelte|rs|toml|json|md)$/i.test(f));

const hits = [];
for (const file of files) {
  const text = readFileSync(join(root, file), 'utf8');
  for (const rule of forbidden) {
    for (const m of text.matchAll(rule.re)) {
      const i = m.index ?? 0;
      const line = text.slice(0, i).split('\n').length;
      const sample = text.slice(Math.max(0, i - 20), Math.min(text.length, i + 120));
      if (allow.some((a) => a.test(sample))) continue;
      hits.push({ file, line, rule: rule.name, sample: sample.replace(/\s+/g, ' ').trim() });
    }
  }
}

if (hits.length > 0) {
  console.error('offline-guard: FAILED');
  for (const h of hits.slice(0, 40)) {
    console.error(`- ${h.file}:${h.line} [${h.rule}] ${h.sample}`);
  }
  process.exit(1);
}

console.log(`offline-guard: PASS (${files.length} files scanned)`);
