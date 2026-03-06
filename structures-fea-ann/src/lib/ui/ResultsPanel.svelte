<script lang="ts">
  import NumberFlow from '@number-flow/svelte';
  import type {
    AnnResult,
    DynamicResult,
    FailureResult,
    FemResult,
    ModelStatus,
    ThermalResult
  } from '$lib/types/contracts';
  import MeshFieldView from '$lib/ui/MeshFieldView.svelte';
  import ThrelteMeshViewport from '$lib/ui/ThrelteMeshViewport.svelte';

  type Props = {
    femResult: FemResult | null;
    annResult: AnnResult | null;
    thermalResult: ThermalResult | null;
    dynamicResult: DynamicResult | null;
    failureResult: FailureResult | null;
    modelStatus: ModelStatus | null;
  };

  let { femResult, annResult, thermalResult, dynamicResult, failureResult, modelStatus }: Props = $props();

  const base = $derived(femResult ?? annResult?.femLike ?? null);
  const sigmaX = $derived(base?.stressTensor?.[0]?.[0] ?? 0);
  const uEnd = $derived(base?.displacementVector?.[1] ?? 0);
  const vm = $derived(base?.vonMisesPsi ?? 0);

  const principal = $derived(base?.principalStresses ?? [0, 0, 0]);
  const principalMaxAbs = $derived.by(() => Math.max(1e-9, ...principal.map((x) => Math.abs(x))));

  const dynPolyline = $derived.by(() => {
    if (!dynamicResult || dynamicResult.timeS.length === 0 || dynamicResult.displacementIn.length === 0) return '';
    const w = 520;
    const h = 160;
    const xs = dynamicResult.timeS;
    const ys = dynamicResult.displacementIn;
    const tMax = Math.max(1e-9, xs[xs.length - 1]);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const yRange = Math.max(1e-9, yMax - yMin);
    return ys
      .map((y, i) => {
        const x = (xs[i] / tMax) * w;
        const py = h - ((y - yMin) / yRange) * h;
        return `${x.toFixed(1)},${py.toFixed(1)}`;
      })
      .join(' ');
  });

  const stationPolyline = $derived.by(() => {
    const stations = base?.beamStations ?? [];
    if (stations.length === 0) return '';
    const w = 520;
    const h = 160;
    const xMax = Math.max(1e-9, stations[stations.length - 1].xIn);
    const yMin = Math.min(...stations.map((s) => s.sigmaTopPsi));
    const yMax = Math.max(...stations.map((s) => s.sigmaTopPsi));
    const yRange = Math.max(1e-9, yMax - yMin);
    return stations
      .map((s) => {
        const x = (s.xIn / xMax) * w;
        const y = h - ((s.sigmaTopPsi - yMin) / yRange) * h;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  });

  const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? v.toFixed(digits) : '0.000');
</script>

<div class="panel stack">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem;flex-wrap:wrap;">
    <div class="stack" style="gap:0.35rem;">
      <h2>Result Workbench</h2>
      <p>Mesh, stress, dynamics, thermal, and failure views from FEM and ANN outputs.</p>
    </div>
    <div class="kicker">
      <span class="chip">sigma_x: <NumberFlow value={sigmaX} format={{ maximumFractionDigits: 2 }} /> psi</span>
      <span class="chip">u_end: <NumberFlow value={uEnd} format={{ maximumFractionDigits: 5 }} /> in</span>
      <span class="chip warn">vm: <NumberFlow value={vm} format={{ maximumFractionDigits: 2 }} /> psi</span>
    </div>
  </div>

  <div class="summary-cards">
    <article class="stat"><div class="label">ANN Confidence</div><div class="value"><NumberFlow value={annResult?.confidence ?? 0} format={{ maximumFractionDigits: 4 }} /></div></article>
    <article class="stat"><div class="label">ANN Uncertainty</div><div class="value"><NumberFlow value={annResult?.uncertainty ?? 0} format={{ maximumFractionDigits: 4 }} /></div></article>
    <article class="stat"><div class="label">Model Version</div><div class="value"><NumberFlow value={modelStatus?.modelVersion ?? 0} format={{ maximumFractionDigits: 0 }} /></div></article>
    <article class="stat"><div class="label">Safety (VM)</div><div class="value"><NumberFlow value={failureResult?.safetyFactorVm ?? 0} format={{ maximumFractionDigits: 3 }} /></div></article>
  </div>

  <MeshFieldView nodalDisplacements={base?.nodalDisplacements ?? null} />
  <ThrelteMeshViewport nodalDisplacements={base?.nodalDisplacements ?? null} />

  <section class="stack">
    <h3>Stress Tensor (psi)</h3>
    <table class="tensor-table">
      <tbody>
        {#each base?.stressTensor ?? [[0, 0, 0], [0, 0, 0], [0, 0, 0]] as row, i (`r${i}`)}
          <tr>
            {#each row as v, j (`c${j}`)}
              <td>{fmt(v, 3)}</td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
  </section>

  <section class="stack">
    <h3>Stress Along Beam Length (Top Fiber)</h3>
    <svg viewBox="0 0 520 160" style="width:100%;border:1px solid rgba(120,138,158,0.42);border-radius:10px;background:#101722;">
      {#if stationPolyline}
        <polyline points={stationPolyline} fill="none" stroke="rgba(77,209,255,0.95)" stroke-width="2.2" />
      {/if}
    </svg>
  </section>

  <section class="stack">
    <h3>Principal Stress Distribution</h3>
    {#each principal as s, i (`p${i}`)}
      {@const ratio = Math.abs(s) / principalMaxAbs}
      <div class="bar-row">
        <span class="bar-label">σ{i + 1}</span>
        <div class="bar-track">
          <div class="bar-fill" style={`width:${Math.max(4, ratio * 100)}%;`}></div>
        </div>
        <span class="bar-value">{fmt(s, 2)} psi</span>
      </div>
    {/each}
  </section>

  <div class="mini-grid">
    <section class="panel stack">
      <h3>Dynamic Displacement vs Time</h3>
      <svg viewBox="0 0 520 160" style="width:100%;border:1px solid rgba(120,138,158,0.42);border-radius:10px;background:#101722;">
        {#if dynPolyline}
          <polyline points={dynPolyline} fill="none" stroke="rgba(105,179,255,0.95)" stroke-width="2" />
        {/if}
      </svg>
      {#if dynamicResult}
        <div class="kicker">
          <span class="chip {dynamicResult.stable ? 'ok' : 'warn'}">{dynamicResult.stable ? 'stable' : 'unstable'}</span>
          <span class="chip">steps: {dynamicResult.timeS.length}</span>
        </div>
      {/if}
    </section>

    <section class="panel stack">
      <h3>Thermal + Failure</h3>
      <div class="summary-cards">
        <article class="stat"><div class="label">Thermal Strain X</div><div class="value"><NumberFlow value={thermalResult?.thermalStrainX ?? 0} format={{ maximumFractionDigits: 8 }} /></div></article>
        <article class="stat"><div class="label">Thermal Stress</div><div class="value"><NumberFlow value={thermalResult?.thermalStressPsi ?? 0} format={{ maximumFractionDigits: 2 }} /> psi</div></article>
        <article class="stat"><div class="label">Tresca</div><div class="value"><NumberFlow value={failureResult?.trescaPsi ?? 0} format={{ maximumFractionDigits: 2 }} /> psi</div></article>
        <article class="stat"><div class="label">Max Principal</div><div class="value"><NumberFlow value={failureResult?.maxPrincipalPsi ?? 0} format={{ maximumFractionDigits: 2 }} /> psi</div></article>
      </div>
      <div class="kicker">
        <span class="chip {failureResult?.failed ? 'warn' : 'ok'}">{failureResult ? (failureResult.failed ? 'failure predicted' : 'pass') : 'not evaluated'}</span>
      </div>
    </section>
  </div>

  <details>
    <summary>Diagnostics</summary>
    <div class="stack">
      {#if base?.diagnostics}
        <ul>
          {#each base.diagnostics as d (`d-${d}`)}
            <li>{d}</li>
          {/each}
        </ul>
      {/if}
      {#if thermalResult?.diagnostics}
        <ul>
          {#each thermalResult.diagnostics as d (`t-${d}`)}
            <li>{d}</li>
          {/each}
        </ul>
      {/if}
      {#if dynamicResult?.diagnostics}
        <ul>
          {#each dynamicResult.diagnostics as d (`y-${d}`)}
            <li>{d}</li>
          {/each}
        </ul>
      {/if}
    </div>
  </details>
</div>
