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
    yieldStrengthPsi: number;
  };

  let { femResult, annResult, thermalResult, dynamicResult, failureResult, modelStatus, yieldStrengthPsi }: Props = $props();
  let showViewport = $state(false);

  const base = $derived(annResult?.femLike ?? femResult ?? null);
  const sigmaX = $derived(base?.stressTensor?.[0]?.[0] ?? 0);
  const uEnd = $derived(base?.displacementVector?.[1] ?? 0);
  const vm = $derived(base?.vonMisesPsi ?? 0);
  const liveSafetyVm = $derived.by(() => {
    if (!base) return null;
    return yieldStrengthPsi / Math.max(Math.abs(base.vonMisesPsi), 1e-9);
  });
  const safetyVmLabel = $derived.by(() => {
    if (failureResult) return Number.isFinite(failureResult.safetyFactorVm) ? failureResult.safetyFactorVm.toFixed(3) : 'n/a';
    if (liveSafetyVm == null || !Number.isFinite(liveSafetyVm)) return 'n/a';
    return liveSafetyVm.toFixed(3);
  });
  const safeguardSummary = $derived.by(() => {
    if (!annResult) return null;
    if (annResult.usedFemFallback) {
      return (
        annResult.fallbackReason ??
        `PINO surrogate fallback triggered. Residual ${annResult.residualScore ?? 0} exceeded threshold ${annResult.residualThreshold ?? 0}.`
      );
    }
    return `PINO surrogate accepted. Residual ${annResult.residualScore ?? 0} within threshold ${annResult.residualThreshold ?? 0}.`;
  });
  const residualRatio = $derived.by(() => {
    if (!annResult) return null;
    const score = annResult.residualScore ?? 0;
    const threshold = Math.max(annResult.residualThreshold ?? 0, 1e-9);
    return score / threshold;
  });
  const holdoutValidation = $derived.by(
    () => annResult?.pino?.holdoutValidation ?? modelStatus?.pino?.holdoutValidation ?? null
  );
  const pinoMeta = $derived.by(() => annResult?.pino ?? modelStatus?.pino ?? null);
  const holdoutCriteriaSummary = $derived.by(() => {
    if (!holdoutValidation) return null;
    const failed: string[] = [];
    if (!holdoutValidation.displacementPass) {
      failed.push(
        `disp ${holdoutValidation.meanDisplacementError.toFixed(3)} > ${holdoutValidation.meanErrorLimit.toFixed(3)}`
      );
    }
    if (!holdoutValidation.vonMisesPass) {
      failed.push(
        `vm ${holdoutValidation.meanVonMisesError.toFixed(3)} > ${holdoutValidation.meanErrorLimit.toFixed(3)}`
      );
    }
    if (!holdoutValidation.p95Pass) {
      failed.push(
        `p95 ${holdoutValidation.p95FieldError.toFixed(3)} > ${holdoutValidation.p95ErrorLimit.toFixed(3)}`
      );
    }
    if (!holdoutValidation.residualRatioPass) {
      failed.push(
        `ratio ${holdoutValidation.residualRatio.toFixed(2)} > ${holdoutValidation.residualRatioLimit.toFixed(2)}`
      );
    }
    return failed;
  });
  const safeguardAdvice = $derived.by(() => {
    if (!annResult) {
      return 'Run surrogate inference to compare the residual score against the safeguard threshold.';
    }
    if (
      holdoutValidation &&
      (!holdoutValidation.trusted || !holdoutValidation.acceptedWithoutFallback)
    ) {
      const failed = holdoutCriteriaSummary ?? [];
      if (failed.length > 0) {
        return `Holdout trust gate failed: ${failed.join('; ')}. Add local training coverage around this regime before using PINO-direct inference.`;
      }
      return 'Holdout trust gate failed. Add local training coverage around this regime before using PINO-direct inference.';
    }
    if (!annResult.usedFemFallback) {
      return 'Residual safeguard passed. This inference stayed within the current acceptance envelope.';
    }
    const ratio = residualRatio ?? 0;
    if (ratio <= 1.5) {
      return 'Close miss. Try retraining with more samples near this geometry/load regime or relax the residual threshold slightly if this case is known-valid.';
    }
    if (ratio <= 4) {
      return 'Moderate mismatch. Add nearby training cases, widen the sampled input range, and validate whether the current residual threshold is too strict for this regime.';
    }
    return 'Large mismatch. This case likely sits outside the trained envelope or the prediction is physically inconsistent. Add targeted training data for this regime before trusting surrogate-only inference.';
  });

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
      <p>Mesh, stress, dynamics, thermal, and failure views from FEM and PINO surrogate outputs.</p>
    </div>
    <div class="kicker">
      <span class="chip">sigma_x: <NumberFlow value={sigmaX} format={{ maximumFractionDigits: 2 }} /> psi</span>
      <span class="chip">u_end: <NumberFlow value={uEnd} format={{ maximumFractionDigits: 5 }} /> in</span>
      <span class="chip warn">vm: <NumberFlow value={vm} format={{ maximumFractionDigits: 2 }} /> psi</span>
    </div>
  </div>

  <div class="summary-cards">
    <article class="stat">
      <div class="label">PINO Confidence</div>
      <div class="value"><NumberFlow value={annResult?.confidence ?? 0} format={{ maximumFractionDigits: 4 }} /></div>
      <div class="meta" style="margin-top:0.3rem;">derived from holdout trust and current residual safeguard context</div>
    </article>
    <article class="stat">
      <div class="label">PINO Uncertainty</div>
      <div class="value"><NumberFlow value={annResult?.uncertainty ?? 0} format={{ maximumFractionDigits: 4 }} /></div>
      <div class="meta" style="margin-top:0.3rem;">higher means lower trust for direct surrogate use in this regime</div>
    </article>
    <article class="stat"><div class="label">Model Version</div><div class="value"><NumberFlow value={modelStatus?.modelVersion ?? 0} format={{ maximumFractionDigits: 0 }} /></div></article>
    <article class="stat">
      <div class="label">Operator Runtime</div>
      <div class="value">{annResult?.pino?.backend ?? modelStatus?.pino?.backend ?? 'legacy-compat'}</div>
      <div class="meta" style="margin-top:0.3rem;">
        {annResult?.pino?.engineId ?? modelStatus?.pino?.engineId ?? 'compat runtime'}
      </div>
    </article>
    <article class="stat">
      <div class="label">Operator Grid</div>
      <div class="value">
        {#if annResult?.pino?.operatorGrid ?? modelStatus?.pino?.operatorGrid}
          {(annResult?.pino?.operatorGrid3d ?? modelStatus?.pino?.operatorGrid3d ?? annResult?.pino?.operatorGrid ?? modelStatus?.pino?.operatorGrid)?.nx}
          x
          {(annResult?.pino?.operatorGrid3d ?? modelStatus?.pino?.operatorGrid3d ?? annResult?.pino?.operatorGrid ?? modelStatus?.pino?.operatorGrid)?.ny}
          x
          {(annResult?.pino?.operatorGrid3d ?? modelStatus?.pino?.operatorGrid3d ?? annResult?.pino?.operatorGrid ?? modelStatus?.pino?.operatorGrid)?.nz}
        {:else}
          n/a
        {/if}
      </div>
    </article>
    <article class="stat">
      <div class="label">Physics Model</div>
      <div class="value">{annResult?.pino?.physicsModel ?? modelStatus?.pino?.physicsModel ?? 'legacy-compat'}</div>
    </article>
    <article class="stat">
      <div class="label">Safety (VM)</div>
      <div class="value">{safetyVmLabel}</div>
      <div class="meta" style="margin-top:0.3rem;">{failureResult ? 'from failure evaluation' : 'estimated from displayed result'}</div>
    </article>
    <article class="stat"><div class="label">Infer Path</div><div class="value">{annResult ? (annResult.usedFemFallback ? 'PINO + FEM fallback' : 'PINO direct') : 'n/a'}</div></article>
    <article class="stat">
      <div class="label">Safeguard Gate</div>
      <div class="value">{annResult ? (annResult.usedFemFallback ? 'fallback used' : 'accepted') : 'n/a'}</div>
      <div class="meta" style="margin-top:0.3rem;">{safeguardSummary ?? 'Run surrogate inference to see the guard decision.'}</div>
    </article>
    <article class="stat">
      <div class="label">Residual Ratio</div>
      <div class="value">
        {#if residualRatio != null}
          <NumberFlow value={residualRatio} format={{ maximumFractionDigits: 2 }} />
          <span class="subtle">x threshold</span>
        {:else}
          n/a
        {/if}
      </div>
      <div class="meta" style="margin-top:0.3rem;">{safeguardAdvice}</div>
    </article>
    {#if holdoutValidation}
      <article class="stat">
        <div class="label">Holdout Trust</div>
        <div class="value">
          {holdoutValidation.trusted && holdoutValidation.acceptedWithoutFallback ? 'trusted for direct PINO' : 'fallback-only'}
        </div>
        <div class="meta" style="margin-top:0.3rem;">
          residual ratio {holdoutValidation.residualRatio?.toFixed?.(2) ?? 'n/a'} · accepted without fallback: {holdoutValidation.acceptedWithoutFallback ? 'yes' : 'no'}
        </div>
        <div class="meta" style="margin-top:0.3rem;">
          limits: mean ≤ {holdoutValidation.meanErrorLimit?.toFixed?.(3) ?? 'n/a'}, p95 ≤ {holdoutValidation.p95ErrorLimit?.toFixed?.(3) ?? 'n/a'}, ratio ≤ {holdoutValidation.residualRatioLimit?.toFixed?.(2) ?? 'n/a'}
        </div>
        <div class="meta" style="margin-top:0.3rem;">
          checks: disp {holdoutValidation.displacementPass ? 'pass' : 'fail'} · vm {holdoutValidation.vonMisesPass ? 'pass' : 'fail'} · p95 {holdoutValidation.p95Pass ? 'pass' : 'fail'} · ratio {holdoutValidation.residualRatioPass ? 'pass' : 'fail'}
        </div>
      </article>
      <article class="stat">
        <div class="label">Holdout Mean Errors</div>
        <div class="value">
          disp {(holdoutValidation.meanDisplacementError ?? 0).toFixed(3)} / vm {(holdoutValidation.meanVonMisesError ?? 0).toFixed(3)}
        </div>
        <div class="meta" style="margin-top:0.3rem;">
          p95 {holdoutValidation.p95FieldError?.toFixed?.(3) ?? 'n/a'}
        </div>
      </article>
      <article class="stat">
        <div class="label">Operator Calibration</div>
        <div class="value">
          s {(pinoMeta?.calibrationStressScale ?? 1).toFixed(3)} / d {(pinoMeta?.calibrationDisplacementScale ?? 1).toFixed(3)}
        </div>
      </article>
    {/if}
  </div>

  <MeshFieldView nodalDisplacements={base?.nodalDisplacements ?? null} />
  <details bind:open={showViewport}>
    <summary>3D mesh viewport</summary>
    <div class="stack" style="margin-top:0.8rem;">
      <p>Open this only when you need the 3D view. Keeping it collapsed avoids heavy re-renders during inference updates.</p>
      {#if showViewport}
        <ThrelteMeshViewport nodalDisplacements={base?.nodalDisplacements ?? null} />
      {/if}
    </div>
  </details>

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
    <h3>Stress Along Plate Length (Kirsch-Informed)</h3>
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
