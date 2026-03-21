#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const pageFile = join(process.cwd(), 'src/routes/+page.svelte');
const txt = readFileSync(pageFile, 'utf8');

const checks = [
  { name: 'tick poll interval', re: /setInterval\(poll,\s*120\)/ },
  { name: 'progress poll interval', re: /setInterval\(poll,\s*280\)/ },
  { name: 'status poll interval', re: /setInterval\(poll,\s*350\)/ },
  { name: 'history cap', re: /trainingHistory\s*=\s*\[\.\.\.trainingHistory\.slice\(-119\)/ },
  { name: 'tick cadence window cap', re: /tickIntervalsMs\s*=\s*\[\.\.\.tickIntervalsMs\.slice\(-31\)/ },
  { name: 'progress cadence window cap', re: /progressIntervalsMs\s*=\s*\[\.\.\.progressIntervalsMs\.slice\(-31\)/ }
];

const missing = checks.filter((c) => !c.re.test(txt)).map((c) => c.name);
if (missing.length) {
  console.error(`ui-budget-guard: FAILED missing checks: ${missing.join(', ')}`);
  process.exit(1);
}

console.log('ui-budget-guard: PASS');
