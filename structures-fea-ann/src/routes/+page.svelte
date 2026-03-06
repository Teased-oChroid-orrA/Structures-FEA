<script lang="ts">
  import { onMount } from 'svelte';
  import NumberFlow from '@number-flow/svelte';
  import CaseInputPanel from '$lib/ui/CaseInputPanel.svelte';
  import NeuralNetworkLive from '$lib/ui/NeuralNetworkLive.svelte';
  import ResultsPanel from '$lib/ui/ResultsPanel.svelte';
  import { warmupMeshLib } from '$lib/mesh/meshlibRuntime';
  import {
    solveFemCase,
    trainAnn,
    inferAnn,
    runDynamicCase,
    runThermalCase,
    evaluateFailure,
    getModelStatus,
    exportReport
  } from '$lib/api/commands';
  import { defaultSolveInput } from '$lib/types/contracts';
  import type {
    DynamicInput,
    TrainingBatch,
    ThermalInput,
    ReportInput,
    FemResult,
    FailureResult,
    TrainingProgressEvent,
    TrainResult
  } from '$lib/types/contracts';

  let solveInput = $state(structuredClone(defaultSolveInput));
  let femResult = $state<FemResult | null>(null);
  let annResult = $state<any>(null);
  let thermalResult = $state<any>(null);
  let dynamicResult = $state<any>(null);
  let failureResult = $state<FailureResult | null>(null);
  let modelStatus = $state<any>(null);
  let exportResult = $state<any>(null);
  let err = $state<string>('');
  let appReady = $state(false);
  let startupProgress = $state(0);
  let startupStage = $state('Booting compute core...');
  let startupDetail = $state('');
  let trainingActive = $state(false);
  let trainingProgress = $state<TrainingProgressEvent | null>(null);
  let trainingHistory = $state<TrainingProgressEvent[]>([]);
  let lastTrainResult = $state<TrainResult | null>(null);

  let thermalDelta = $state(40);
  let thermalRestrainedX = $state('true');
  let dynDt = $state(0.001);
  let dynEnd = $state(0.1);
  let dynDamping = $state(0.02);
  let dynPulseDur = $state(0.01);
  let dynPulseScale = $state(1);

  let trainEpochs = $state(20);
  let trainTarget = $state(0.01);
  let trainLr = $state(0.0005);
  let trainAutoMode = $state('true');
  let trainMaxEpochs = $state(800);
  let trainMinImprovement = $state(0.0000001);

  let exportPath = $state('outputs/fea_report.json');
  let exportFormat = $state<'json' | 'csv' | 'pdf'>('json');

  const hasAnyResult = $derived(Boolean(femResult || annResult || thermalResult || dynamicResult || failureResult));
  const startupPercent = $derived(Math.round(startupProgress * 100));

  onMount(() => {
    const isTauriRuntime = typeof window !== 'undefined' && typeof (window as any).__TAURI_INTERNALS__ !== 'undefined';
    let unlistenProgress: (() => void) | undefined;
    let unlistenComplete: (() => void) | undefined;
    let offProgressWeb: (() => void) | undefined;
    let offCompleteWeb: (() => void) | undefined;

    (async () => {
      const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
      startupStage = 'Initializing event streams...';
      startupProgress = 0.2;
      await pause(120);

      if (isTauriRuntime) {
        const { listen } = await import('@tauri-apps/api/event');
        unlistenProgress = await listen<TrainingProgressEvent>('ann-training-progress', (event) => {
          trainingProgress = event.payload;
          trainingHistory = [...trainingHistory.slice(-199), event.payload];
        });

        unlistenComplete = await listen<TrainResult>('ann-training-complete', async (event) => {
          trainingActive = false;
          lastTrainResult = event.payload;
          await onModelStatus();
        });
      } else if (typeof window !== 'undefined') {
        const onProgress = (event: Event) => {
          const payload = (event as CustomEvent<TrainingProgressEvent>).detail;
          trainingProgress = payload;
          trainingHistory = [...trainingHistory.slice(-199), payload];
        };
        const onComplete = async (event: Event) => {
          const payload = (event as CustomEvent<TrainResult>).detail;
          trainingActive = false;
          lastTrainResult = payload;
          await onModelStatus();
        };
        window.addEventListener('ann-training-progress', onProgress as EventListener);
        window.addEventListener('ann-training-complete', onComplete as EventListener);
        offProgressWeb = () => window.removeEventListener('ann-training-progress', onProgress as EventListener);
        offCompleteWeb = () => window.removeEventListener('ann-training-complete', onComplete as EventListener);
      }

      startupStage = 'Loading model state...';
      startupProgress = 0.72;
      await onModelStatus();
      await pause(100);
      startupStage = 'Warming MeshLib kernels...';
      startupProgress = 0.86;
      const meshlib = await warmupMeshLib();
      startupDetail = meshlib.detail;
      await pause(120);
      startupStage = 'Ready';
      startupProgress = 1;
      await pause(120);
      appReady = true;
    })();

    return () => {
      unlistenProgress?.();
      unlistenComplete?.();
      offProgressWeb?.();
      offCompleteWeb?.();
    };
  });

  async function call<T>(fn: () => Promise<T>) {
    err = '';
    try {
      return await fn();
    } catch (e) {
      err = String(e);
      return null;
    }
  }

  async function onSolveFem() {
    const out = await call(() => solveFemCase(solveInput));
    if (out) femResult = out as FemResult;
  }

  async function onTrainAnn() {
    trainingActive = true;
    trainingHistory = [];
    trainingProgress = null;
    const batch: TrainingBatch = {
      cases: [solveInput],
      epochs: trainEpochs,
      targetLoss: trainTarget,
      learningRate: trainLr,
      autoMode: trainAutoMode === 'true',
      maxTotalEpochs: trainMaxEpochs,
      minImprovement: trainMinImprovement
    };
    const result = await call(() => trainAnn(batch));
    if (!result) {
      trainingActive = false;
    }
  }

  async function onInferAnn() {
    const out = await call(() => inferAnn(solveInput));
    if (out) annResult = out;
  }

  async function onThermal() {
    const input: ThermalInput = {
      solveInput,
      deltaTF: thermalDelta,
      restrainedX: thermalRestrainedX === 'true'
    };
    const out = await call(() => runThermalCase(input));
    if (out) thermalResult = out;
  }

  async function onDynamic() {
    const input: DynamicInput = {
      solveInput,
      timeStepS: dynDt,
      endTimeS: dynEnd,
      dampingRatio: dynDamping,
      pulseDurationS: dynPulseDur,
      pulseScale: dynPulseScale
    };
    const out = await call(() => runDynamicCase(input));
    if (out) dynamicResult = out;
  }

  async function onFailure() {
    const source = femResult?.stressTensor ?? annResult?.femLike?.stressTensor;
    if (!source) {
      err = 'Run FEM or ANN first to compute stress tensor.';
      return;
    }
    const out = await call(() =>
      evaluateFailure({
        stressTensor: source,
        yieldStrengthPsi: solveInput.material.yieldStrengthPsi
      })
    );
    if (out) failureResult = out as FailureResult;
  }

  async function onModelStatus() {
    const out = await call(() => getModelStatus());
    if (out) modelStatus = out;
  }

  async function onExport() {
    const payload: ReportInput = {
      path: exportPath,
      format: exportFormat,
      solveInput,
      femResult: femResult ?? undefined,
      annResult: annResult ?? undefined,
      dynamicResult: dynamicResult ?? undefined,
      thermalResult: thermalResult ?? undefined,
      failureResult: failureResult ?? undefined
    };
    const out = await call(() => exportReport(payload));
    if (out) exportResult = out;
  }
</script>

<main>
  {#if !appReady}
    <section class="panel stack startup-screen">
      <h2>Starting Structures FEA + ANN</h2>
      <p>{startupStage}</p>
      {#if startupDetail}
        <p>{startupDetail}</p>
      {/if}
      <div class="startup-progress-track">
        <div class="startup-progress-fill" style={`width:${startupPercent}%;`}></div>
      </div>
      <div class="kicker">
        <span class="chip"><NumberFlow value={startupPercent} format={{ maximumFractionDigits: 0 }} />%</span>
        <span class="chip">local-only runtime</span>
      </div>
    </section>
  {/if}

  <div class="app-shell">
    <section class="panel stack">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem;flex-wrap:wrap;">
        <div class="stack" style="gap:0.35rem;">
          <h1>Structures FEA + Adaptive ANN</h1>
          <p>Modern local desktop workbench for a cantilever beam: fixed left end, vertical point load at right tip/top (center thickness).</p>
        </div>
        <div class="kicker">
          <span class="chip ok">offline</span>
          <span class="chip">windows-ready</span>
          <span class="chip warn">ann primary</span>
        </div>
      </div>
    </section>

    <div class="layout">
      <div class="stack">
        <CaseInputPanel bind:solveInput />

        <section class="panel stack">
          <h2>Solve Controls</h2>
          <div class="actions">
            <button class="btn-primary" onclick={onSolveFem}>Solve FEM Case</button>
            <button class="btn-secondary" onclick={onInferAnn}>Infer ANN</button>
            <button class="btn-secondary" onclick={onTrainAnn} disabled={trainingActive}>
              {trainingActive ? 'Training…' : 'Train ANN'}
            </button>
            <button class="btn-secondary" onclick={onFailure}>Evaluate Failure</button>
            <button class="btn-secondary" onclick={onThermal}>Run Thermal Case</button>
            <button class="btn-secondary" onclick={onDynamic}>Run Dynamic Case</button>
            <button class="btn-secondary" onclick={onModelStatus}>Model Status</button>
          </div>
        </section>

        <section class="panel stack">
          <h2>Scenario Parameters</h2>
          <div class="field-grid">
            <label class="field">
              <span>Delta T (F)</span>
              <input type="number" bind:value={thermalDelta} step="1" />
            </label>
            <label class="field">
              <span>Restrained X</span>
              <select bind:value={thermalRestrainedX}>
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </label>
            <label class="field">
              <span>Dynamic dt (s)</span>
              <input type="number" bind:value={dynDt} step="0.0001" />
            </label>
            <label class="field">
              <span>End Time (s)</span>
              <input type="number" bind:value={dynEnd} step="0.01" />
            </label>
            <label class="field">
              <span>Damping Ratio</span>
              <input type="number" bind:value={dynDamping} step="0.01" />
            </label>
            <label class="field">
              <span>Pulse Duration (s)</span>
              <input type="number" bind:value={dynPulseDur} step="0.001" />
            </label>
            <label class="field">
              <span>Pulse Scale</span>
              <input type="number" bind:value={dynPulseScale} step="0.1" />
            </label>
            <label class="field">
              <span>Train Epochs</span>
              <input type="number" bind:value={trainEpochs} step="1" />
            </label>
            <label class="field">
              <span>Target Loss</span>
              <input type="number" bind:value={trainTarget} step="0.001" />
            </label>
            <label class="field">
              <span>Learning Rate</span>
              <input type="number" bind:value={trainLr} step="0.0001" />
            </label>
            <label class="field">
              <span>Auto Train</span>
              <select bind:value={trainAutoMode}>
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </label>
            <label class="field">
              <span>Max Total Epochs</span>
              <input type="number" bind:value={trainMaxEpochs} step="10" min="1" />
            </label>
            <label class="field">
              <span>Min Improvement</span>
              <input type="number" bind:value={trainMinImprovement} step="0.0000001" />
            </label>
          </div>
        </section>

        <section class="panel stack">
          <h2>Export</h2>
          <div class="field-grid">
            <label class="field" style="grid-column:1/-1;">
              <span>Output Path</span>
              <input type="text" bind:value={exportPath} />
            </label>
            <label class="field">
              <span>Format</span>
              <select bind:value={exportFormat}>
                <option value="json">json</option>
                <option value="csv">csv</option>
                <option value="pdf">pdf</option>
              </select>
            </label>
            <label class="field">
              <span>Report Action</span>
              <button class="btn-primary" onclick={onExport}>Export Report</button>
            </label>
          </div>
          {#if exportResult}
            <div class="summary-cards">
              <article class="stat">
                <div class="label">Export Path</div>
                <div class="value">{exportResult.path}</div>
              </article>
              <article class="stat">
                <div class="label">Bytes Written</div>
                <div class="value">{exportResult.bytesWritten}</div>
              </article>
              <article class="stat">
                <div class="label">Format</div>
                <div class="value">{exportResult.format}</div>
              </article>
            </div>
          {/if}
        </section>
      </div>

      <div class="stack">
        <NeuralNetworkLive
          {trainingActive}
          progress={trainingProgress}
          history={trainingHistory}
        />

        {#if lastTrainResult}
          <section class="panel stack">
            <h3>Last Training Run</h3>
            <div class="summary-cards">
              <article class="stat">
                <div class="label">Stop Reason</div>
                <div class="value">{lastTrainResult.stopReason}</div>
              </article>
              <article class="stat">
                <div class="label">Reached Target</div>
                <div class="value">{lastTrainResult.reachedTarget ? 'yes' : 'no'}</div>
              </article>
              <article class="stat">
                <div class="label">Completed Epochs</div>
                <div class="value"><NumberFlow value={lastTrainResult.completedEpochs} format={{ maximumFractionDigits: 0 }} /></div>
              </article>
              <article class="stat">
                <div class="label">Topology Changes</div>
                <div class="value">{lastTrainResult.grew ? 'grow ' : ''}{lastTrainResult.pruned ? 'prune' : ''}</div>
              </article>
            </div>
          </section>
        {/if}

        <ResultsPanel
          {femResult}
          {annResult}
          {thermalResult}
          {dynamicResult}
          {failureResult}
          {modelStatus}
        />

        {#if err}
          <section class="panel alert stack">
            <h3>Runtime Error</h3>
            <pre>{err}</pre>
          </section>
        {/if}

        {#if !hasAnyResult}
          <section class="panel stack">
            <h3>Quick Start</h3>
            <p>Run <strong>Solve FEM Case</strong> to compute stress/deflection along the beam length, then train ANN for adaptive prediction.</p>
          </section>
        {/if}
      </div>
    </div>
  </div>
</main>
