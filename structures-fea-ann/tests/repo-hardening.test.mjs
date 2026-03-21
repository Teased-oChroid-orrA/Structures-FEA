import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { describe, test } from 'node:test';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..');

async function readText(relPath) {
  return readFile(resolve(repoRoot, relPath), 'utf8');
}

async function readJson(relPath) {
  return JSON.parse(await readText(relPath));
}

describe('repo hardening contracts', () => {
  test('package.json exposes validation scripts', async () => {
    const pkg = await readJson('package.json');
    assert.equal(pkg.scripts.lint, 'npm run check');
    assert.equal(pkg.scripts.test, 'node --test tests/repo-hardening.test.mjs');
    assert.equal(pkg.scripts['test:repo'], 'node --test tests/repo-hardening.test.mjs');
    assert.equal(pkg.scripts.validate, 'npm run check && npm test');
  });

  test('ci workflows include repo and rust gates', async () => {
    const ci = await readText('../.github/workflows/ci-matrix.yml');
    const signoff = await readText('../.github/workflows/release-signoff.yml');
    const windows = await readText('../.github/workflows/windows-build.yml');

    assert.match(ci, /npm run lint/);
    assert.match(ci, /npm test/);
    assert.match(ci, /cargo test --manifest-path src-tauri\/Cargo.toml/);

    assert.match(signoff, /npm test/);
    assert.match(signoff, /cargo test --manifest-path src-tauri\/Cargo.toml/);
    assert.match(signoff, /npm run verify:pino-regression-gates/);

    assert.match(windows, /npm test/);
    assert.match(windows, /npm run lint/);
  });

  test('readme documents validation flow', async () => {
    const readme = await readText('README.md');
    assert.match(readme, /## Prerequisites/);
    assert.match(readme, /Node\.js 20/);
    assert.match(readme, /Rust stable toolchain/);
    assert.match(readme, /## Validate/);
    assert.match(readme, /npm test/);
    assert.match(readme, /cargo test/);
    assert.match(readme, /## CI Gates/);
  });
});
