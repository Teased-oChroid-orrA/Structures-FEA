<script lang="ts">
  import { onMount, tick } from 'svelte';
  import NumberFlow from '@number-flow/svelte';
  import CaseInputPanel from '$lib/ui/CaseInputPanel.svelte';
  import NeuralNetworkLive from '$lib/ui/NeuralNetworkLive.svelte';
  import ResultsPanel from '$lib/ui/ResultsPanel.svelte';
  import { warmupMeshLib } from '$lib/mesh/meshlibRuntime';
  import {
    solveFemCase,
    startAnnTraining,
    stopAnnTraining,
    getTrainingStatus,
    inferAnn,
    runDynamicCase,
    runThermalCase,
    evaluateFailure,
    getModelStatus,
    getRuntimeKind,
    getRuntimeFingerprint,
    setSafeguardSettings,
    getTrainingTick,
    getTrainingProgress,
    saveTrainingCheckpoint,
    listTrainingCheckpoints,
    resumeTrainingFromCheckpoint,
    purgeTrainingCheckpoints,
    resetAnnModel,
    exportReport
  } from '$lib/api/commands';
  import { defaultSolveInput } from '$lib/types/contracts';
  import type {
    AnnResult,
    DynamicInput,
    DynamicResult,
    ExportResult,
    TrainingBatch,
    TrainingRunStatus,
    ThermalInput,
    ThermalResult,
    ReportInput,
    FemResult,
    FailureResult,
    SolveInput,
    TrainingTickEvent,
    TrainingProgressEvent,
    TrainingCheckpointInfo,
    TrainResult,
    RuntimeFingerprint,
    ModelStatus,
    SafeguardSettings,
    ResumeTrainingResult
  } from '$lib/types/contracts';

  let solveInput = $state(structuredClone(defaultSolveInput));
  let femResult = $state<FemResult | null>(null);
  let annResult = $state<AnnResult | null>(null);
  let thermalResult = $state<ThermalResult | null>(null);
  let dynamicResult = $state<DynamicResult | null>(null);
  let failureResult = $state<FailureResult | null>(null);
  let modelStatus = $state<ModelStatus | null>(null);
  let exportResult = $state<ExportResult | null>(null);
  let err = $state<string>('');
  let appReady = $state(false);
  let startupProgress = $state(0);
  let startupStage = $state('Booting compute core...');
  let startupDetail = $state('');
  let trainingActive = $state(false);
  let inferActive = $state(false);
  let trainingTick = $state<TrainingTickEvent | null>(null);
  let trainingProgress = $state<TrainingProgressEvent | null>(null);
  let trainingHistory = $state<TrainingProgressEvent[]>([]);
  let lastHistoryEpoch = $state(0);
  let lastTrainResult = $state<TrainResult | null>(null);
  let trainingStatus = $state<TrainingRunStatus | null>(null);
  let trainingElapsedMs = $state(0);
  let trainingStartMs = $state<number | null>(null);
  let displayEpoch = $state(0);
  let lastEpochRate = $state(0);
  let lastTickUpdatedAtMs = $state(0);
  let lastProgressUpdatedAtMs = $state(0);
  let tickIntervalsMs = $state<number[]>([]);
  let progressIntervalsMs = $state<number[]>([]);
  let tickPollTimeoutsTotal = $state(0);
  let progressPollTimeoutsTotal = $state(0);
  let statusPollTimeoutsTotal = $state(0);
  let tickPollTimeoutRecent = $state<number[]>([]);
  let progressPollTimeoutRecent = $state<number[]>([]);
  let statusPollTimeoutRecent = $state<number[]>([]);

  function extractEpochHintFromPhase(phase: string | null | undefined): number {
    if (!phase) return 0;
    const match = phase.match(/epoch-(\d+)-bootstrap/);
    return match ? Number.parseInt(match[1] ?? '0', 10) || 0 : 0;
  }

  let thermalDelta = $state(40);
  let thermalRestrainedX = $state('true');
  let dynDt = $state(0.001);
  let dynEnd = $state(0.1);
  let dynDamping = $state(0.02);
  let dynPulseDur = $state(0.01);
  let dynPulseScale = $state(1);

  let trainEpochs = $state(40);
  let trainTarget = $state(1e-9);
  let trainLr = $state(0.0005);
  let trainAnalysisType = $state<'general' | 'cantilever' | 'plate-hole'>('cantilever');
  let trainAutoMode = $state<'true' | 'false'>('true');
  let trainMaxEpochs = $state(10000);
  let trainMinImprovement = $state(0.000001);
  let trainUpdateEveryEpochs = $state(1);
  let trainNetworkUpdateEveryEpochs = $state(10);
  let trainOnlineActiveLearning = $state<'true' | 'false'>('true');
  let trainAutonomousMode = $state<'true' | 'false'>('true');
  let trainMaxTopology = $state(128);
  let trainMaxBackoffs = $state(12);
  let trainMaxOptimizerSwitches = $state(8);
  let trainCheckpointEveryEpochs = $state(1000);
  let trainCheckpointRetention = $state(8);
  let trainSeed = $state(42);
  let trainPinnBackend = $state<
    'pino-ndarray-cpu' | 'pino-candle-cpu' | 'pino-candle-cuda' | 'pino-candle-metal'
  >('pino-ndarray-cpu');
  let trainCollocationPoints = $state(4096);
  let trainBoundaryPoints = $state(1024);
  let trainInterfacePoints = $state(512);
  let trainResidualWeightMomentum = $state(1.0);
  let trainResidualWeightKinematics = $state(1.0);
  let trainResidualWeightMaterial = $state(1.0);
  let trainResidualWeightBoundary = $state(1.0);
  let trainCurriculumPreset = $state<'cantilever-data-first' | 'plate-hole-balanced' | 'contact-stabilized'>(
    'cantilever-data-first'
  );
  let trainAutoRecipe = $state<'true' | 'false'>('true');
  let trainStage1Epochs = $state(2500);
  let trainStage2Epochs = $state(4500);
  let trainStage3RampEpochs = $state(3000);
  let trainContactPenalty = $state(10);
  let trainPlasticityFactor = $state(0);
  let runtimeKind = $state<'tauri' | 'mock'>('mock');
  let runtimeFingerprint = $state<RuntimeFingerprint | null>(null);
  let lastRunFingerprint = $state<string>('');
  let checkpoints = $state<TrainingCheckpointInfo[]>([]);
  let safeguardPreset = $state<'balanced' | 'conservative' | 'performance'>('balanced');
  let safeguardUncertainty = $state(0.26);
  let safeguardResidual = $state(0.24);
  let safeguardAdaptive = $state<'true' | 'false'>('true');

  const trainingPollBudgetMs = $derived.by(() => {
    if (runtimeKind !== 'tauri') {
      return { tick: 500, progress: 800, status: 1000 };
    }
    if (trainPinnBackend === 'pino-ndarray-cpu') {
      return { tick: 1800, progress: 2800, status: 3200 };
    }
    return { tick: 1200, progress: 1800, status: 2200 };
  });

  const trainingPollIntervalMs = $derived.by(() => {
    if (runtimeKind !== 'tauri') {
      return { tick: 120, progress: 280, status: 350 };
    }
    if (trainPinnBackend === 'pino-ndarray-cpu') {
      return { tick: 240, progress: 420, status: 700 };
    }
    return { tick: 180, progress: 320, status: 520 };
  });

  const scaleWithin = (value: number, scale: number, min: number, max: number) =>
    Math.max(min, Math.min(max, value * scale));

  const asSafeguardPreset = (value: string): 'balanced' | 'conservative' | 'performance' =>
    value === 'conservative' || value === 'performance' || value === 'balanced'
      ? value
      : 'balanced';

  type TrainingRecipe = {
    seedCases: SolveInput[];
    validationCases: SolveInput[];
    variedInputs: string[];
    rangeLines: string[];
    estimatedFemCases: number;
    validationChecks: string[];
  };

  function toPlainSolveInput(input: SolveInput): SolveInput {
    return {
      geometry: {
        lengthIn: input.geometry.lengthIn,
        widthIn: input.geometry.widthIn,
        thicknessIn: input.geometry.thicknessIn,
        holeDiameterIn: input.geometry.holeDiameterIn
      },
      mesh: {
        nx: input.mesh.nx,
        ny: input.mesh.ny,
        nz: input.mesh.nz,
        elementType: input.mesh.elementType,
        autoAdapt: input.mesh.autoAdapt,
        maxDofs: input.mesh.maxDofs,
        amrEnabled: input.mesh.amrEnabled,
        amrPasses: input.mesh.amrPasses,
        amrMaxNx: input.mesh.amrMaxNx,
        amrRefineRatio: input.mesh.amrRefineRatio
      },
      material: {
        ePsi: input.material.ePsi,
        nu: input.material.nu,
        rhoLbIn3: input.material.rhoLbIn3,
        alphaPerF: input.material.alphaPerF,
        yieldStrengthPsi: input.material.yieldStrengthPsi
      },
      boundaryConditions: {
        fixStartFace: input.boundaryConditions.fixStartFace,
        fixEndFace: input.boundaryConditions.fixEndFace
      },
      load: {
        axialLoadLbf: input.load.axialLoadLbf,
        verticalPointLoadLbf: input.load.verticalPointLoadLbf
      },
      unitSystem: input.unitSystem,
      deltaTF: input.deltaTF
    };
  }

  function buildAutoTrainingRecipe(
    input: SolveInput,
    analysisType: 'general' | 'cantilever' | 'plate-hole',
    enabled: boolean
  ): TrainingRecipe {
    const baseInput = toPlainSolveInput(input);
    if (!enabled) {
      return {
        seedCases: [baseInput],
        validationCases: [],
        variedInputs: ['current case only'],
        rangeLines: ['manual mode: no automatic case expansion'],
        estimatedFemCases: 1,
        validationChecks: [
          'Manual mode keeps the current case only.',
          'Run a reference FEM solve before inferring nearby cases.'
        ]
      };
    }

    const verticalDriven =
      analysisType === 'cantilever' || baseInput.load.verticalPointLoadLbf !== 0;
    const loadLabel = verticalDriven ? 'vertical load' : 'axial load';
    const loadBase = verticalDriven
      ? baseInput.load.verticalPointLoadLbf || 250
      : baseInput.load.axialLoadLbf || 250;

    const seedScales = [0.9, 1.0, 1.1];
    const seedCases = seedScales.map((scale, idx) => {
      const c = toPlainSolveInput(baseInput);
      c.geometry.lengthIn = scaleWithin(baseInput.geometry.lengthIn, idx === 1 ? 1 : scale, 4, 30);
      c.geometry.widthIn = scaleWithin(baseInput.geometry.widthIn, idx === 1 ? 1 : 2 - scale, 1, 12);
      c.geometry.thicknessIn = scaleWithin(baseInput.geometry.thicknessIn, scale, 0.03, 0.75);
      c.material.ePsi = scaleWithin(baseInput.material.ePsi, idx === 0 ? 0.97 : idx === 2 ? 1.03 : 1, 1.0e6, 40.0e6);

      if (analysisType === 'plate-hole' && baseInput.geometry.holeDiameterIn != null) {
        c.geometry.holeDiameterIn = scaleWithin(
          baseInput.geometry.holeDiameterIn,
          idx === 0 ? 0.92 : idx === 2 ? 1.08 : 1,
          0,
          c.geometry.widthIn * 0.95
        );
      }

      if (verticalDriven) {
        c.load.verticalPointLoadLbf = scaleWithin(loadBase, scale, -10000, 10000);
        c.load.axialLoadLbf = 0;
      } else {
        c.load.axialLoadLbf = scaleWithin(Math.abs(loadBase), scale, 25, 100000);
        c.load.verticalPointLoadLbf = 0;
      }

      return c;
    });

    const validationCases = [0.95, 1.05].map((scale, idx) => {
      const c = toPlainSolveInput(baseInput);
      c.geometry.lengthIn = scaleWithin(baseInput.geometry.lengthIn, scale, 4, 30);
      c.geometry.widthIn = scaleWithin(baseInput.geometry.widthIn, idx === 0 ? 1.03 : 0.97, 1, 12);
      c.geometry.thicknessIn = scaleWithin(baseInput.geometry.thicknessIn, idx === 0 ? 0.96 : 1.04, 0.03, 0.75);
      if (analysisType === 'plate-hole' && baseInput.geometry.holeDiameterIn != null) {
        c.geometry.holeDiameterIn = scaleWithin(baseInput.geometry.holeDiameterIn, idx === 0 ? 0.96 : 1.04, 0, c.geometry.widthIn * 0.95);
      }
      if (verticalDriven) {
        c.load.verticalPointLoadLbf = scaleWithin(loadBase, scale, -10000, 10000);
        c.load.axialLoadLbf = 0;
      } else {
        c.load.axialLoadLbf = scaleWithin(Math.abs(loadBase), scale, 25, 100000);
        c.load.verticalPointLoadLbf = 0;
      }
      return c;
    });

    const variedInputs = [
      'length',
      'width',
      'thickness',
      loadLabel,
      analysisType === 'plate-hole' ? 'hole diameter' : null,
      'elastic modulus'
    ].filter(Boolean) as string[];

    return {
      seedCases,
      variedInputs,
      rangeLines: [
        `length: ${scaleWithin(baseInput.geometry.lengthIn, 0.94, 4, 30).toFixed(3)} to ${scaleWithin(baseInput.geometry.lengthIn, 1.06, 4, 30).toFixed(3)} in`,
        `width: ${scaleWithin(baseInput.geometry.widthIn, 0.94, 1, 12).toFixed(3)} to ${scaleWithin(baseInput.geometry.widthIn, 1.06, 1, 12).toFixed(3)} in`,
        `thickness: ${scaleWithin(baseInput.geometry.thicknessIn, 0.94, 0.03, 0.75).toFixed(3)} to ${scaleWithin(baseInput.geometry.thicknessIn, 1.06, 0.03, 0.75).toFixed(3)} in`,
        `${loadLabel}: ${scaleWithin(Math.abs(loadBase), 0.88, 25, 100000).toFixed(1)} to ${scaleWithin(Math.abs(loadBase), 1.12, 25, 100000).toFixed(1)} lbf`,
        analysisType === 'plate-hole' && baseInput.geometry.holeDiameterIn != null
          ? `hole diameter: ${scaleWithin(baseInput.geometry.holeDiameterIn, 0.92, 0, baseInput.geometry.widthIn * 0.95).toFixed(3)} to ${scaleWithin(baseInput.geometry.holeDiameterIn, 1.08, 0, baseInput.geometry.widthIn * 0.95).toFixed(3)} in`
          : null,
        `E: ${(scaleWithin(baseInput.material.ePsi, 0.97, 1.0e6, 40.0e6) / 1_000_000).toFixed(2)} to ${(scaleWithin(baseInput.material.ePsi, 1.03, 1.0e6, 40.0e6) / 1_000_000).toFixed(2)} Mpsi`
      ].filter(Boolean) as string[],
      estimatedFemCases: seedCases.length * 12,
      validationChecks: [
        'Run FEM and PINO surrogate on 2 holdout cases inside this range.',
        'Trust the surrogate only if the holdouts avoid safeguard fallback.',
        'Check tip displacement and von Mises error against FEM; keep them within about 5% before relying on surrogate-only inference.'
      ],
      validationCases
    };
  }

  const autoTrainingRecipe = $derived.by(() =>
    buildAutoTrainingRecipe(solveInput, trainAnalysisType, trainAutoRecipe === 'true')
  );

  let exportPath = $state('outputs/fea_report.json');
  let exportFormat = $state<'json' | 'csv' | 'pdf'>('json');

  const hasAnyResult = $derived(Boolean(femResult || annResult || thermalResult || dynamicResult || failureResult));
  const startupPercent = $derived(Math.round(startupProgress * 100));
  const trainingElapsedS = $derived(trainingElapsedMs / 1000);
  const POLL_WINDOW_MS = 60_000;
  const estimatedEpoch = $derived.by(() => {
    const tickEpoch = trainingTick?.epoch ?? 0;
    const progressEpoch = trainingProgress?.epoch ?? 0;
    const bootstrapEpochHint = extractEpochHintFromPhase(trainingProgress?.lrPhase);
    if (!trainingActive) {
      return Math.max(tickEpoch, progressEpoch, bootstrapEpochHint, displayEpoch, 0);
    }
    return Math.max(tickEpoch, progressEpoch, bootstrapEpochHint, 0);
  });
  const displayTotalEpochs = $derived.by(() => {
    const fromProgress = trainingProgress?.totalEpochs ?? 0;
    const fromTick = trainingTick?.totalEpochs ?? 0;
    if (trainingActive) {
      if (fromProgress > 0) return fromProgress;
      if (fromTick > 0) return fromTick;
      return Math.max(1, trainMaxEpochs);
    }
    if (fromTick > 0) return fromTick;
    if (fromProgress > 0) return fromProgress;
    return Math.max(1, trainMaxEpochs);
  });
  const progressDisplayRatio = $derived.by(() => {
    const fromProgress = trainingProgress?.totalEpochs ?? 0;
    const fromTick = trainingTick?.totalEpochs ?? 0;
    const total = fromProgress > 0 ? fromProgress : fromTick > 0 ? fromTick : displayTotalEpochs;
    if (total > 0) {
      return Math.max(0, Math.min(1, displayEpoch / total));
    }
    if (trainingActive && displayTotalEpochs > 0) {
      return Math.max(0, Math.min(1, displayEpoch / displayTotalEpochs));
    }
    if (!trainingActive) return 0;
    return ((trainingElapsedMs / 1000) % 6) / 6;
  });
  const stageSchedule = $derived.by(() => {
    const total = Math.max(1, displayTotalEpochs);
    const explicit = trainStage1Epochs > 0 || trainStage2Epochs > 0 || trainStage3RampEpochs > 0;
    const s1 = explicit ? Math.max(1, trainStage1Epochs) : Math.max(1, Math.round(total * 0.25));
    const s2 = explicit ? Math.max(1, trainStage2Epochs) : Math.max(1, Math.round(total * 0.45));
    const s3 = explicit ? Math.max(1, trainStage3RampEpochs) : Math.max(1, total - s1 - s2);
    // Literal epoch semantics for stage inputs:
    // if user enters 25, it means 25 epochs (never percentage-scaled).
    const stageTotal = explicit ? Math.max(1, s1 + s2 + s3) : total;
    const epoch = Math.max(0, Math.min(displayEpoch, stageTotal));
    const r1 = Math.max(0, Math.min(1, epoch / s1));
    const r2 = Math.max(0, Math.min(1, (epoch - s1) / s2));
    const r3 = Math.max(0, Math.min(1, (epoch - s1 - s2) / s3));
    return {
      explicit,
      scheduleTotal: total,
      stageTotal,
      stage1: { span: s1, ratio: r1 },
      stage2: { span: s2, ratio: r2 },
      stage3: { span: s3, ratio: r3 }
    };
  });
  const stageGateInspector = $derived.by(() => {
    const e = Math.max(0, displayEpoch);
    const s1Start = 0;
    const s1End = stageSchedule.stage1.span;
    const s2Start = s1End;
    const s2End = s1End + stageSchedule.stage2.span;
    const s3Start = s2End;
    const s3End = s2End + stageSchedule.stage3.span;
    const gateStatus = (start: number, end: number) => {
      if (e >= end) return 'complete';
      if (e >= start) return 'active';
      return 'pending';
    };
    return {
      backendStage: trainingStatus?.diagnostics?.activeStage ?? 'idle',
      rows: [
        {
          id: 'stage-1',
          label: 'Stage 1 (Data-Fit)',
          start: s1Start,
          end: s1End,
          criterion: `epoch >= ${s1End}`,
          status: gateStatus(s1Start, s1End)
        },
        {
          id: 'stage-2',
          label: 'Stage 2 (Stabilization)',
          start: s2Start,
          end: s2End,
          criterion: `epoch >= ${s2End}`,
          status: gateStatus(s2Start, s2End)
        },
        {
          id: 'stage-3',
          label: 'Stage 3 (Physics-Ramp)',
          start: s3Start,
          end: s3End,
          criterion: `epoch >= ${s3End}`,
          status: gateStatus(s3Start, s3End)
        }
      ]
    };
  });
  const telemetry = $derived.by(() => {
    const p = trainingProgress;
    const t = trainingTick;
    const finiteOr = (value: number | undefined | null, fallback = 0) =>
      Number.isFinite(value ?? NaN) ? (value as number) : fallback;
    return {
      status: trainingActive ? trainingPhaseLabel : 'idle',
      epoch: finiteOr(t?.epoch, 0),
      totalEpochs: finiteOr(t?.totalEpochs, displayTotalEpochs),
      loss: finiteOr(t?.loss, 0),
      valLoss: finiteOr(t?.valLoss, 0),
      valDataLoss: finiteOr(p?.valDataLoss, 0),
      valPhysicsLoss: finiteOr(p?.valPhysicsLoss, 0),
      learningRate: finiteOr(t?.learningRate, trainLr),
      lrPhase: p?.lrPhase ?? trainingStatus?.diagnostics?.lrSchedulePhase ?? 'idle',
      optimizer: p?.optimizerId ?? trainingStatus?.diagnostics?.activeOptimizer ?? 'pino-adam',
      stage: p?.stageId ?? trainingStatus?.diagnostics?.activeStage ?? 'idle'
    };
  });
  const displayBestValLoss = $derived.by(() => {
    const v = trainingStatus?.diagnostics?.bestValLoss;
    return Number.isFinite(v ?? NaN) && (v as number) < Number.MAX_VALUE / 2 ? (v as number) : null;
  });
  const displayEpochsSinceImprovement = $derived.by(() => {
    if (!trainingActive && !lastTrainResult) return 0;
    return Math.max(0, trainingStatus?.diagnostics?.epochsSinceImprovement ?? 0);
  });
  const latestObservedLoss = $derived.by(() => {
    const candidates = [
      trainingProgress?.valLoss,
      trainingTick?.valLoss,
      lastTrainResult?.valLoss,
      trainingProgress?.loss,
      trainingTick?.loss,
      lastTrainResult?.loss
    ];
    for (const value of candidates) {
      if (Number.isFinite(value ?? NaN)) return value as number;
    }
    return null;
  });
  const targetLossReached = $derived.by(() => {
    const observed = latestObservedLoss;
    return observed !== null && observed > 0 ? observed <= trainTarget : false;
  });
  const targetLossGap = $derived.by(() => {
    const observed = latestObservedLoss;
    if (observed === null) return null;
    if (trainingActive && trainingHistory.length === 0 && (trainingTick?.epoch ?? 0) === 0) {
      return null;
    }
    return Math.max(0, observed - trainTarget);
  });
  const preflightPhaseLabel = $derived.by(() => {
    if (!trainingActive) return null;
    if ((trainingTick?.epoch ?? 0) > 0 || trainingHistory.length > 0) return null;
    const phase = trainingProgress?.lrPhase ?? '';
    if (!phase || phase === 'idle' || phase === 'queued') return 'booting';
    return phase.replaceAll('-', ' ');
  });
  const trainingPhaseLabel = $derived.by(() => {
    if (!trainingActive) {
      if (lastTrainResult?.reachedTargetLoss || lastTrainResult?.stopReason === 'target-loss-reached') {
        return 'target reached';
      }
      if (lastTrainResult?.stopReason === 'plateau-stop') return 'plateau stop';
      if (lastTrainResult) return 'run complete';
      return 'idle';
    }
    const currentEpoch = trainingTick?.epoch ?? trainingProgress?.epoch ?? 0;
    if (currentEpoch === 0 && trainingHistory.length === 0) {
      return preflightPhaseLabel ?? 'booting';
    }
    if (targetLossReached) return 'at target';
    if (currentEpoch < 10) return 'warming up';
    if (displayEpochsSinceImprovement >= 100) return 'plateau watch';
    if (latestObservedLoss !== null && latestObservedLoss < trainTarget * 10) return 'closing in';
    return 'optimizing';
  });
  const progressEvidenceLabel = $derived.by(() => {
    if (!trainingActive) {
      return lastTrainResult ? `${trainingHistory.length} frames captured` : 'no live frames yet';
    }
    if ((trainingTick?.epoch ?? 0) === 0 && trainingHistory.length === 0) {
      return preflightPhaseLabel
        ? `epoch 0 live: ${preflightPhaseLabel}`
        : 'epoch 0 armed, awaiting first live progress frame';
    }
    return `${trainingHistory.length} live progress frames`;
  });
  const trainedModelReady = $derived.by(() => {
    const completedRun = (lastTrainResult?.completedEpochs ?? 0) > 0;
    const versionReady = (modelStatus?.modelVersion ?? 0) > 1;
    return completedRun || versionReady;
  });
  const inferenceStateLabel = $derived.by(() => {
    if (inferActive) return 'inferring...';
    if (annResult) return 'prediction ready';
    if (trainedModelReady) return 'ready to infer';
    return 'train first';
  });
  const inferenceStatusMessage = $derived.by(() => {
    if (inferActive) return 'Running PINO surrogate inference and applying safeguard checks for the current case.';
    if (annResult?.usedFemFallback) return 'Inference completed with FEM fallback after PINO trust/safeguard gate rejection.';
    if (annResult) return 'Inference completed from the trained PINO surrogate.';
    if (trainedModelReady) return 'Model is trained. Adjust the case and run inference when ready.';
    return 'Train the PINO surrogate before requesting an inference.';
  });
  const nextRecommendedAction = $derived.by(() => {
    if (trainingActive) {
      if (targetLossReached) {
        return 'The live validation loss is at or below target. Let the run finish, checkpoint it, and verify the holdout cases before trusting the surrogate.';
      }
      return `Training is running. Track the live target gap (${targetLossGap !== null ? targetLossGap.toExponential(2) : 'n/a'}) and keep the case in the same regime until validation crosses target.`;
    }
    if (!femResult && !trainedModelReady) {
      return 'Start with Solve FEM Case to establish a reference result for the current case, then train the PINO surrogate.';
    }
    if (femResult && !trainedModelReady) {
      return 'The reference FEM result is ready. Train the PINO surrogate next so you can reuse this regime for fast predictions.';
    }
    if (trainedModelReady && !annResult) {
      return 'The surrogate is ready. Adjust the case within the same regime and click Infer Surrogate for a rapid prediction.';
    }
    return 'Use Solve FEM Case to validate important surrogate predictions or whenever you move outside the trained range.';
  });
  const lastRunOutcomeLabel = $derived.by(() => {
    if (trainingActive) return 'training';
    if (!lastTrainResult) return 'not trained';
    if (lastTrainResult.stopReason === 'target-loss-reached') return 'converged';
    if (lastTrainResult.stopReason === 'plateau-stop') return 'stopped on plateau';
    if (lastTrainResult.stopReason === 'manual-stop') return 'stopped manually';
    return lastTrainResult.stopReason;
  });
  const tickAgeMs = $derived.by(() => {
    if (!trainingActive || lastTickUpdatedAtMs <= 0) return 0;
    return Math.max(0, Date.now() - lastTickUpdatedAtMs);
  });
  const progressAgeMs = $derived.by(() => {
    if (!trainingActive || lastProgressUpdatedAtMs <= 0) return 0;
    return Math.max(0, Date.now() - lastProgressUpdatedAtMs);
  });
  const tickCadenceMs = $derived.by(() =>
    tickIntervalsMs.length
      ? tickIntervalsMs.reduce((acc, v) => acc + v, 0) / tickIntervalsMs.length
      : 0
  );
  const progressCadenceMs = $derived.by(() =>
    progressIntervalsMs.length
      ? progressIntervalsMs.reduce((acc, v) => acc + v, 0) / progressIntervalsMs.length
      : 0
  );
  const throughputHealth = $derived.by(() => {
    if (!trainingActive) return 'idle';
    if (tickAgeMs > 2500 || progressAgeMs > 3500) return 'lagging';
    if (tickCadenceMs > 900 || progressCadenceMs > 1500) return 'degraded';
    return 'stable';
  });
  const tickPollTimeouts = $derived.by(() => tickPollTimeoutRecent.length);
  const progressPollTimeouts = $derived.by(() => progressPollTimeoutRecent.length);
  const statusPollTimeouts = $derived.by(() => statusPollTimeoutRecent.length);
  const pollHealthLabel = $derived.by(() => {
    const total = tickPollTimeouts + progressPollTimeouts + statusPollTimeouts;
    if (!trainingActive) return 'idle';
    if (tickAgeMs > 3_000 || total > 40) return 'stalled';
    if (total > 8) return 'degraded';
    return 'good';
  });
  const sortedCheckpoints = $derived.by(() =>
    [...checkpoints].sort((a, b) => Number(b.createdAtUnixMs ?? 0) - Number(a.createdAtUnixMs ?? 0))
  );
  const checkpointHealth = $derived.by(() => {
    const latest = sortedCheckpoints[0];
    const best =
      sortedCheckpoints.find((cp) => cp.isBest) ??
      [...sortedCheckpoints].sort((a, b) => (a.bestValLoss ?? Number.MAX_VALUE) - (b.bestValLoss ?? Number.MAX_VALUE))[0];
    const ageMs = latest ? Math.max(0, Date.now() - Number(latest.createdAtUnixMs ?? 0)) : Infinity;
    const recoveryState = !latest
      ? 'unavailable'
      : trainingActive && trainCheckpointEveryEpochs <= 0
        ? 'manual-only'
        : trainingActive && ageMs > 15 * 60_000
          ? 'stale-while-running'
          : 'recoverable';
    return { latest, best, ageMs, recoveryState };
  });
  const checkpointAgeLabel = $derived.by(() => {
    const ms = checkpointHealth.ageMs;
    if (!Number.isFinite(ms)) return 'n/a';
    if (ms < 1000) return 'just now';
    const sec = Math.floor(ms / 1000);
    if (sec < 60) return `${sec}s ago`;
    const min = Math.floor(sec / 60);
    if (min < 60) return `${min}m ago`;
    const hr = Math.floor(min / 60);
    return `${hr}h ago`;
  });
  const epochWindowRows = $derived.by(() => {
    const windowSize = 1000;
    const source = [...trainingHistory];
    if (trainingProgress && trainingProgress.epoch > 0) {
      source.push(trainingProgress);
    }
    const byWindow = new Map<number, TrainingProgressEvent>();
    for (const p of source) {
      if (!p || p.epoch < windowSize) continue;
      const w = Math.floor(p.epoch / windowSize) * windowSize;
      const prev = byWindow.get(w);
      if (!prev || p.epoch >= prev.epoch) byWindow.set(w, p);
    }
    return [...byWindow.entries()]
      .sort((a, b) => a[0] - b[0])
      .slice(-10)
      .map(([windowEpoch, p]) => ({
        windowEpoch,
        valDataLoss: p.valDataLoss ?? 0,
        valPhysicsLoss: p.valPhysicsLoss ?? 0,
        lrPhase: p.lrPhase ?? 'steady',
        optimizer: p.optimizerId ?? 'pino-adam',
        stage: p.stageId ?? 'stage-1'
      }));
  });
  $effect(() => {
    if (!trainingActive) return;
    const candidate = Math.max(0, estimatedEpoch);
    displayEpoch = Math.max(displayEpoch, candidate);
  });
  const currentRunFingerprint = $derived.by(() =>
    JSON.stringify(
      {
        runtime: runtimeKind,
        backend: runtimeFingerprint,
        geometry: solveInput.geometry,
        mesh: solveInput.mesh,
        load: solveInput.load,
        material: solveInput.material,
        train: {
          analysisType: trainAnalysisType,
          autoRecipe: trainAutoRecipe === 'true',
          epochs: trainEpochs,
          maxTotalEpochs: trainMaxEpochs,
          targetLoss: trainTarget,
          learningRate: trainLr,
          minImprovement: trainMinImprovement,
          progressEmitEveryEpochs: trainUpdateEveryEpochs,
          networkEmitEveryEpochs: trainNetworkUpdateEveryEpochs,
          onlineActiveLearning: trainOnlineActiveLearning === 'true',
          autonomousMode: trainAutonomousMode === 'true',
          maxTopology: trainMaxTopology,
          maxBackoffs: trainMaxBackoffs,
          maxOptimizerSwitches: trainMaxOptimizerSwitches,
          checkpointEveryEpochs: trainCheckpointEveryEpochs,
          checkpointRetention: trainCheckpointRetention,
          seed: trainSeed,
          pinnBackend: trainPinnBackend,
          collocationPoints: trainCollocationPoints,
          boundaryPoints: trainBoundaryPoints,
          interfacePoints: trainInterfacePoints,
          residualWeightMomentum: trainResidualWeightMomentum,
          residualWeightKinematics: trainResidualWeightKinematics,
          residualWeightMaterial: trainResidualWeightMaterial,
          residualWeightBoundary: trainResidualWeightBoundary,
          stage1Epochs: trainStage1Epochs,
          stage2Epochs: trainStage2Epochs,
          stage3RampEpochs: trainStage3RampEpochs,
          contactPenalty: trainContactPenalty,
          plasticityFactor: trainPlasticityFactor
        }
      },
      null,
      0
    )
  );

  const trimTimeoutWindow = (values: number[], now: number) => values.filter((v) => now - v <= POLL_WINDOW_MS);
  const registerPollTimeout = (kind: 'tick' | 'progress' | 'status') => {
    const now = Date.now();
    if (kind === 'tick') {
      tickPollTimeoutsTotal += 1;
      tickPollTimeoutRecent = [...trimTimeoutWindow(tickPollTimeoutRecent, now), now];
      return;
    }
    if (kind === 'progress') {
      progressPollTimeoutsTotal += 1;
      progressPollTimeoutRecent = [...trimTimeoutWindow(progressPollTimeoutRecent, now), now];
      return;
    }
    statusPollTimeoutsTotal += 1;
    statusPollTimeoutRecent = [...trimTimeoutWindow(statusPollTimeoutRecent, now), now];
  };

  $effect(() => {
    if (!trainingActive) {
      trainingStartMs = null;
      return;
    }
    trainingElapsedMs = 0;
    trainingStartMs = Date.now();
    const id = setInterval(() => {
      trainingElapsedMs = Date.now() - (trainingStartMs ?? Date.now());
    }, 200);
    return () => clearInterval(id);
  });

  $effect(() => {
    if (!trainingActive) return;
    const id = setInterval(() => {
      const now = Date.now();
      tickPollTimeoutRecent = trimTimeoutWindow(tickPollTimeoutRecent, now);
      progressPollTimeoutRecent = trimTimeoutWindow(progressPollTimeoutRecent, now);
      statusPollTimeoutRecent = trimTimeoutWindow(statusPollTimeoutRecent, now);
    }, 1000);
    return () => clearInterval(id);
  });

  $effect(() => {
    if (!trainingActive) return;
    let stopped = false;
    let inFlight = false;
    const poll = async () => {
      if (stopped || inFlight) return;
      inFlight = true;
      try {
        const tick = await withTimeout(() => getTrainingTick(), trainingPollBudgetMs.tick, () => {
          registerPollTimeout('tick');
        });
        if (!stopped && tick && tick.epoch >= (trainingTick?.epoch ?? 0)) {
          if (tick.epoch > (trainingTick?.epoch ?? 0)) {
            const now = Date.now();
            if (lastTickUpdatedAtMs > 0) {
              tickIntervalsMs = [...tickIntervalsMs.slice(-31), now - lastTickUpdatedAtMs];
            }
            lastTickUpdatedAtMs = now;
          }
          trainingTick = {
            ...tick,
            totalEpochs: tick.totalEpochs > 0 ? tick.totalEpochs : Math.max(1, trainMaxEpochs)
          };
        }
      } catch (error) {
        if (!String(error).includes('timeout')) {
          err = `Training tick polling failed: ${String(error)}`;
        }
      } finally {
        inFlight = false;
      }
    };
    const id = setInterval(poll, trainingPollIntervalMs.tick);
    poll();
    return () => {
      stopped = true;
      clearInterval(id);
    };
  });

  $effect(() => {
    if (!trainingActive) return;
    let stopped = false;
    let inFlight = false;
    const poll = async () => {
      if (stopped || inFlight) return;
      inFlight = true;
      try {
        const progress = await withTimeout(
          () => getTrainingProgress(),
          trainingPollBudgetMs.progress,
          () => {
            registerPollTimeout('progress');
          }
        );
        if (stopped || !progress) return;
        const isPreflightProgress =
          progress.epoch === 0 && (progress.stageId !== 'idle' || progress.lrPhase !== 'idle');
        if (!isPreflightProgress && progress.epoch < lastHistoryEpoch) return;
        if (progress.network.layerSizes.length > 0) {
          trainingProgress = progress;
        } else if (trainingProgress) {
          trainingProgress = { ...progress, network: trainingProgress.network };
        } else {
          trainingProgress = progress;
        }
        if (isPreflightProgress) {
          lastProgressUpdatedAtMs = Date.now();
        }
        if (progress.epoch > 0 && progress.epoch !== lastHistoryEpoch) {
          const now = Date.now();
          if (lastProgressUpdatedAtMs > 0) {
            progressIntervalsMs = [...progressIntervalsMs.slice(-31), now - lastProgressUpdatedAtMs];
          }
          lastProgressUpdatedAtMs = now;
          lastHistoryEpoch = progress.epoch;
          const minimal = { ...progress, network: { layerSizes: [], nodes: [], connections: [] } };
          trainingHistory = [...trainingHistory.slice(-119), minimal];
        }
      } catch (error) {
        if (!String(error).includes('timeout')) {
          err = `Training progress polling failed: ${String(error)}`;
        }
      } finally {
        inFlight = false;
      }
    };
    const id = setInterval(poll, trainingPollIntervalMs.progress);
    poll();
    return () => {
      stopped = true;
      clearInterval(id);
    };
  });

  $effect(() => {
    if (!trainingActive) return;
    let stopped = false;
    let inFlight = false;
    const poll = async () => {
      if (stopped || inFlight) return;
      inFlight = true;
      try {
        const status = await withTimeout(() => getTrainingStatus(), trainingPollBudgetMs.status, () => {
          registerPollTimeout('status');
        });
        if (!status) return;
        if (stopped) return;
        trainingStatus = status;
        if (!status.running && trainingStartMs !== null) {
          trainingElapsedMs = Math.max(trainingElapsedMs, Date.now() - trainingStartMs);
        }
        trainingActive = status.running;
        if (!status.running) {
          trainingStartMs = null;
        }
        if (!status.running) {
          if (status.lastResult) {
            lastTrainResult = status.lastResult;
            const elapsedS = Math.max(0.001, trainingElapsedMs / 1000);
            if (status.lastResult.completedEpochs > 0) {
              lastEpochRate = status.lastResult.completedEpochs / elapsedS;
            }
            trainingTick = {
              epoch: status.lastResult.completedEpochs,
              totalEpochs: Math.max(status.lastResult.completedEpochs, trainMaxEpochs),
              loss: status.lastResult.loss,
              valLoss: status.lastResult.valLoss,
              learningRate: status.lastResult.learningRate,
              architecture: status.lastResult.architecture,
              progressRatio:
                status.lastResult.completedEpochs /
                Math.max(1, Math.max(status.lastResult.completedEpochs, trainMaxEpochs))
            };
            displayEpoch = Math.max(displayEpoch, status.lastResult.completedEpochs);
            const now = Date.now();
            lastTickUpdatedAtMs = now;
            lastProgressUpdatedAtMs = now;
            await onModelStatus();
          } else if (status.lastError) {
            err = status.lastError;
          }
        }
      } catch (error) {
        if (!String(error).includes('timeout')) {
          err = `Training status polling failed: ${String(error)}`;
        }
      } finally {
        inFlight = false;
      }
    };
    const id = setInterval(poll, trainingPollIntervalMs.status);
    poll();
    return () => {
      stopped = true;
      clearInterval(id);
    };
  });

  onMount(() => {
    (async () => {
      const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
      async function startupTimeout<T>(task: () => Promise<T>, timeoutMs: number, fallback: T): Promise<T> {
        try {
          return (await withTimeout(task, timeoutMs)) ?? fallback;
        } catch (error) {
          if (String(error).includes('timeout')) {
            return fallback;
          }
          throw error;
        }
      }

      try {
        startupStage = 'Initializing polling channels...';
        startupProgress = 0.2;
        await pause(120);

        startupStage = 'Loading model state...';
        startupProgress = 0.72;
        await startupTimeout(async () => {
          await onModelStatus();
          return true;
        }, 1500, false);
        runtimeKind = getRuntimeKind();
        runtimeFingerprint = await startupTimeout(() => getRuntimeFingerprint(), 1500, null);
        trainingStatus = await startupTimeout(() => getTrainingStatus(), 1500, null);
        if (trainingStatus?.running) {
          trainingActive = true;
        }

        await pause(100);
        startupStage = 'Warming MeshLib kernels...';
        startupProgress = 0.86;
        const meshlib = await startupTimeout(
          () => warmupMeshLib(),
          6000,
          {
            ready: false,
            runtime: 'internal',
            reason: 'timeout',
            detail: 'MeshLib warmup exceeded startup budget. Continuing with internal meshing kernels.'
          }
        );
        startupDetail = meshlib.detail;

        await pause(120);
        await startupTimeout(async () => {
          await onListCheckpoints();
          return true;
        }, 1500, false);
        startupStage = 'Ready';
        startupProgress = 1;
        await pause(120);
      } catch (error) {
        err = `Startup degraded: ${String(error)}`;
        startupStage = 'Ready (degraded)';
        startupProgress = 1;
      } finally {
        appReady = true;
      }
    })();

    return () => {};
  });

  $effect(() => {
    if (!appReady) return;
    let stopped = false;
    let inFlight = false;
    const poll = async () => {
      if (stopped || inFlight) return;
      if (!trainingActive && checkpoints.length > 0) return;
      inFlight = true;
      try {
        const out = await withTimeout(() => listTrainingCheckpoints(), 1200, () => {});
        if (stopped || !out) return;
        checkpoints = out;
      } catch (error) {
        if (!String(error).includes('timeout')) {
          err = `Checkpoint refresh failed: ${String(error)}`;
        }
      } finally {
        inFlight = false;
      }
    };
    const id = setInterval(poll, 8000);
    poll();
    return () => {
      stopped = true;
      clearInterval(id);
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

  async function withTimeout<T>(
    fn: () => Promise<T>,
    timeoutMs: number,
    onTimeout?: () => void
  ): Promise<T | null> {
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    try {
      return await Promise.race([
        fn(),
        new Promise<never>((_, reject) => {
          timeoutId = setTimeout(() => {
            onTimeout?.();
            reject(new Error(`timeout after ${timeoutMs}ms`));
          }, timeoutMs);
        })
      ]);
    } finally {
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
      }
    }
  }

  async function onSolveFem() {
    const out = await call(() => solveFemCase(solveInput));
    if (out) {
      femResult = out;
      annResult = null;
      failureResult = null;
    }
  }

  async function onTrainAnn() {
    trainingActive = true;
    trainingStatus = {
      running: true,
      stopRequested: false,
      completed: false,
      diagnostics: {
        bestValLoss: Number.MAX_VALUE,
        epochsSinceImprovement: 0,
        lrSchedulePhase: 'training',
        currentLearningRate: trainLr,
        dataWeight: 1,
        physicsWeight: 0.15,
        activeLearningRounds: 0,
        activeLearningSamplesAdded: 0,
        safeguardTriggers: 0,
        curriculumBackoffs: 0,
        optimizerSwitches: 0,
        checkpointRollbacks: 0,
        targetFloorEstimate: 0,
        trendStopReason: 'training',
        activeStage: 'stage-1',
        activeOptimizer: 'pino-adam',
        boPresearchUsed: false,
        boSelectedArchitecture: modelStatus?.architecture ?? [],
        residualWeightMomentum: 1,
        residualWeightKinematics: 1,
        residualWeightMaterial: 1,
        residualWeightBoundary: 1,
        momentumResidual: 0,
        kinematicResidual: 0,
        materialResidual: 0,
        boundaryResidual: 0,
        displacementFit: 0,
        stressFit: 0,
        invariantResidual: 0,
        constitutiveNormalResidual: 0,
        constitutiveShearResidual: 0,
        valDisplacementFit: 0,
        valStressFit: 0,
        valInvariantResidual: 0,
        valConstitutiveNormalResidual: 0,
        valConstitutiveShearResidual: 0,
        hybridMode: 'hybrid',
        collocationPoints: trainCollocationPoints,
        boundaryPoints: trainBoundaryPoints,
        interfacePoints: trainInterfacePoints,
        collocationSamplesAdded: 0,
        trainDataSize: 0,
        trainDataCap: 0,
        recentEvents: []
      }
    };
    trainingTick = {
      epoch: 0,
      totalEpochs: Math.max(1, trainMaxEpochs),
      loss: 0,
      valLoss: 0,
      learningRate: trainLr,
      architecture: modelStatus?.architecture ?? [],
      progressRatio: 0
    };
    displayEpoch = 0;
    lastTickUpdatedAtMs = Date.now();
    lastProgressUpdatedAtMs = 0;
    tickIntervalsMs = [];
    progressIntervalsMs = [];
    tickPollTimeoutsTotal = 0;
    progressPollTimeoutsTotal = 0;
    statusPollTimeoutsTotal = 0;
    tickPollTimeoutRecent = [];
    progressPollTimeoutRecent = [];
    statusPollTimeoutRecent = [];
    trainingHistory = [];
    lastHistoryEpoch = 0;
    trainingProgress = null;
    const trainingCases =
      trainAutoRecipe === 'true'
        ? autoTrainingRecipe.seedCases.map((c) => toPlainSolveInput(c))
        : [toPlainSolveInput(solveInput)];
    const batch: TrainingBatch = {
      cases: trainingCases,
      epochs: trainEpochs,
      targetLoss: trainTarget,
      learningRate: trainLr,
      analysisType: trainAnalysisType,
      autoMode: trainAutoMode === 'true',
      maxTotalEpochs: trainMaxEpochs,
      minImprovement: trainMinImprovement,
      progressEmitEveryEpochs: Math.max(1, Math.floor(trainUpdateEveryEpochs)),
      networkEmitEveryEpochs: Math.max(1, Math.floor(trainNetworkUpdateEveryEpochs)),
      onlineActiveLearning: trainOnlineActiveLearning === 'true',
      autonomousMode: trainAutonomousMode === 'true',
      maxTopology: Math.max(8, Math.floor(trainMaxTopology)),
      maxBackoffs: Math.max(1, Math.floor(trainMaxBackoffs)),
      maxOptimizerSwitches: Math.max(1, Math.floor(trainMaxOptimizerSwitches)),
      checkpointEveryEpochs: Math.max(0, Math.floor(trainCheckpointEveryEpochs)),
      checkpointRetention: Math.max(1, Math.floor(trainCheckpointRetention)),
      seed: Math.max(0, Math.floor(trainSeed)),
      pinnBackend: trainPinnBackend,
      collocationPoints: Math.max(64, Math.floor(trainCollocationPoints)),
      boundaryPoints: Math.max(16, Math.floor(trainBoundaryPoints)),
      interfacePoints: Math.max(16, Math.floor(trainInterfacePoints)),
      residualWeightMomentum: Math.max(0, trainResidualWeightMomentum),
      residualWeightKinematics: Math.max(0, trainResidualWeightKinematics),
      residualWeightMaterial: Math.max(0, trainResidualWeightMaterial),
      residualWeightBoundary: Math.max(0, trainResidualWeightBoundary),
      stage1Epochs: Math.max(0, Math.floor(trainStage1Epochs)),
      stage2Epochs: Math.max(0, Math.floor(trainStage2Epochs)),
      stage3RampEpochs: Math.max(0, Math.floor(trainStage3RampEpochs)),
      contactPenalty: Math.max(0, trainContactPenalty),
      plasticityFactor: Math.max(0, Math.min(1, trainPlasticityFactor))
    };
    lastRunFingerprint = currentRunFingerprint;
    const started = await call(() => startAnnTraining(batch));
    if (!started) {
      trainingActive = false;
      err = 'Training is already running or failed to start.';
    }
  }

  async function onStopTrain() {
    const stopped = await call(() => stopAnnTraining());
    if (!stopped) return;
    if (trainingStatus) {
      trainingStatus = { ...trainingStatus, stopRequested: true };
    }
  }

  async function onInferAnn() {
    if (!trainedModelReady) {
      err = 'Train the surrogate before requesting inference.';
      return;
    }
    inferActive = true;
    await tick();
    try {
      const out = await call(() => inferAnn(solveInput));
      if (out) {
        annResult = out;
        failureResult = null;
      }
    } finally {
      inferActive = false;
    }
  }

  async function ensureFailureSource() {
    const existing = femResult?.stressTensor ?? annResult?.femLike?.stressTensor;
    if (existing) {
      return existing;
    }

    if (!trainedModelReady) {
      err = 'Run FEM first or train the PINO surrogate before evaluating failure.';
      return null;
    }

    inferActive = true;
    await tick();
    try {
      const inferred = await call(() => inferAnn(solveInput));
      if (!inferred) {
        return null;
      }
      annResult = inferred;
      failureResult = null;
      return inferred.femLike?.stressTensor ?? null;
    } finally {
      inferActive = false;
    }
  }

  async function onResetModel() {
    const out = await call(() => resetAnnModel(Math.max(0, Math.floor(trainSeed))));
    if (out) {
      modelStatus = out;
      lastTrainResult = null;
      trainingElapsedMs = 0;
      trainingStartMs = null;
      trainingTick = null;
      displayEpoch = 0;
      lastEpochRate = 0;
      trainingProgress = null;
      trainingHistory = [];
      lastHistoryEpoch = 0;
      lastTickUpdatedAtMs = 0;
      lastProgressUpdatedAtMs = 0;
      tickIntervalsMs = [];
      progressIntervalsMs = [];
      tickPollTimeoutsTotal = 0;
      progressPollTimeoutsTotal = 0;
      statusPollTimeoutsTotal = 0;
      tickPollTimeoutRecent = [];
      progressPollTimeoutRecent = [];
      statusPollTimeoutRecent = [];
    }
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
    const source = await ensureFailureSource();
    if (!source) {
      err ||= 'Unable to evaluate failure because no stress tensor is available.';
      return;
    }
    const out = await call(() =>
      evaluateFailure({
        stressTensor: source,
        yieldStrengthPsi: solveInput.material.yieldStrengthPsi
      })
    );
    if (out) failureResult = out;
  }

  async function onModelStatus() {
    const out = await call(() => getModelStatus());
    if (out) {
      modelStatus = out;
      const s = out.safeguardSettings;
      if (s) {
        safeguardPreset = asSafeguardPreset(s.preset ?? safeguardPreset);
        safeguardUncertainty = s.uncertaintyThreshold ?? safeguardUncertainty;
        safeguardResidual = s.residualThreshold ?? safeguardResidual;
        safeguardAdaptive = (s.adaptiveByGeometry ?? true) ? 'true' : 'false';
      }
    }
  }

  async function onApplySafeguardPreset() {
    if (safeguardPreset === 'conservative') {
      safeguardUncertainty = 0.22;
      safeguardResidual = 0.18;
    } else if (safeguardPreset === 'performance') {
      safeguardUncertainty = 0.34;
      safeguardResidual = 0.34;
    } else if (safeguardPreset === 'balanced') {
      safeguardUncertainty = 0.26;
      safeguardResidual = 0.24;
    }
    const out = await call(() =>
      setSafeguardSettings({
        preset: safeguardPreset,
        uncertaintyThreshold: safeguardUncertainty,
        residualThreshold: safeguardResidual,
        adaptiveByGeometry: safeguardAdaptive === 'true'
      })
    );
    if (out) modelStatus = out;
  }

  function onApplyCurriculumPreset() {
    if (trainCurriculumPreset === 'cantilever-data-first') {
      trainAnalysisType = 'cantilever';
      trainLr = 0.0005;
      trainTarget = 1e-9;
      trainUpdateEveryEpochs = 1;
      trainNetworkUpdateEveryEpochs = 10;
      trainMaxEpochs = Math.max(trainMaxEpochs, 12000);
      trainStage1Epochs = 2500;
      trainStage2Epochs = 4500;
      trainStage3RampEpochs = 3000;
      trainResidualWeightMomentum = 1.0;
      trainResidualWeightKinematics = 1.0;
      trainResidualWeightMaterial = 1.0;
      trainResidualWeightBoundary = 0.6;
      trainContactPenalty = 6;
      return;
    }
    if (trainCurriculumPreset === 'plate-hole-balanced') {
      trainAnalysisType = 'plate-hole';
      trainLr = 0.00035;
      trainTarget = 1e-9;
      trainUpdateEveryEpochs = 1;
      trainNetworkUpdateEveryEpochs = 10;
      trainMaxEpochs = Math.max(trainMaxEpochs, 12000);
      trainStage1Epochs = 3000;
      trainStage2Epochs = 4000;
      trainStage3RampEpochs = 3000;
      trainResidualWeightMomentum = 1.0;
      trainResidualWeightKinematics = 1.1;
      trainResidualWeightMaterial = 1.0;
      trainResidualWeightBoundary = 1.0;
      trainContactPenalty = 10;
      return;
    }
    trainAnalysisType = 'general';
    trainLr = 0.00025;
    trainTarget = 1e-8;
    trainUpdateEveryEpochs = 1;
    trainNetworkUpdateEveryEpochs = 10;
    trainStage1Epochs = 4200;
    trainStage2Epochs = 3800;
    trainStage3RampEpochs = 3000;
    trainResidualWeightMomentum = 1.1;
    trainResidualWeightKinematics = 1.2;
    trainResidualWeightMaterial = 1.0;
    trainResidualWeightBoundary = 1.4;
    trainContactPenalty = 18;
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

  async function onSaveCheckpoint() {
    const cp = await call(() => saveTrainingCheckpoint({ tag: 'manual', markBest: false }));
    if (cp) {
      await onListCheckpoints();
    }
  }

  async function onListCheckpoints() {
    const out = await call(() => listTrainingCheckpoints());
    if (out) checkpoints = out;
  }

  async function onResumeLatestCheckpoint() {
    if (!checkpoints.length) {
      await onListCheckpoints();
    }
    if (!checkpoints.length) {
      err = 'No checkpoints found.';
      return;
    }
    const out = await call(() => resumeTrainingFromCheckpoint(checkpoints[0].id));
    if (out) {
      modelStatus = out.modelStatus;
    }
  }

  async function onResumeBestCheckpoint() {
    if (!checkpoints.length) {
      await onListCheckpoints();
    }
    if (!checkpoints.length) {
      err = 'No checkpoints found.';
      return;
    }
    const best =
      checkpoints.find((cp) => cp.isBest) ??
      [...checkpoints].sort((a, b) => (a.bestValLoss ?? Number.MAX_VALUE) - (b.bestValLoss ?? Number.MAX_VALUE))[0];
    if (!best) {
      err = 'No recoverable best checkpoint found.';
      return;
    }
    const out = await call(() => resumeTrainingFromCheckpoint(best.id));
    if (out) {
      modelStatus = out.modelStatus;
    }
  }

  async function onPurgeCheckpoints() {
    const out = await call(() =>
      purgeTrainingCheckpoints({ keepLast: Math.max(1, trainCheckpointRetention), keepBest: 2 })
    );
    if (out) {
      await onListCheckpoints();
    }
  }
</script>

<main>
  {#if !appReady}
    <section class="panel stack startup-screen">
      <h2>Starting Structures FEA + PINO</h2>
      <p>{startupStage}</p>
      {#if startupDetail}
        <p>{startupDetail}</p>
      {/if}
      <div class="startup-progress-track">
        <div class="startup-progress-fill" style={`width:${startupPercent}%;`}></div>
      </div>
      <div class="kicker">
        <span class="chip"><NumberFlow value={startupPercent} format={{ maximumFractionDigits: 0 }} />%</span>
        <span class="chip">runtime: {runtimeKind}</span>
      </div>
    </section>
  {/if}

  <div class="app-shell">
    <section class="panel stack">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem;flex-wrap:wrap;">
        <div class="stack" style="gap:0.35rem;">
          <h1>Structures FEA + Adaptive PINO Surrogate</h1>
          <p>Modern local desktop workbench using a NAFEMS plate-with-hole benchmark (USCS) with Kirsch stress concentration verification.</p>
        </div>
        <div class="kicker">
          <span class="chip ok">offline</span>
          <span class="chip">windows-ready</span>
          <span class="chip warn">pino primary</span>
          <span class="chip">runtime: {runtimeKind}</span>
        </div>
      </div>
    </section>

    <div class="layout">
      <div class="stack">
        <CaseInputPanel bind:solveInput />

        <section class="panel stack">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem;flex-wrap:wrap;">
            <div class="stack" style="gap:0.35rem;">
              <h2>Guided Workflow</h2>
              <p>Use the app in this order: solve a reference case, train the PINO surrogate, then reuse it for fast inference on nearby cases.</p>
            </div>
            <div class="kicker">
              <span class={`chip ${femResult ? 'ok' : ''}`}>{femResult ? 'reference solved' : 'reference pending'}</span>
              <span class={`chip ${trainedModelReady ? 'ok' : ''}`}>{trainedModelReady ? 'model ready' : 'model not trained'}</span>
              <span class={`chip ${annResult ? 'ok' : ''}`}>{inferenceStateLabel}</span>
            </div>
          </div>
          <div
            data-testid="run-fingerprint-json"
            aria-hidden="true"
            style="position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;"
          >
            {currentRunFingerprint}
          </div>
          <div
            data-testid="training-telemetry"
            aria-hidden="true"
            style="position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;"
          >
            <span data-testid="telemetry-status">{telemetry.status}</span>
            <span data-testid="telemetry-epoch">{telemetry.epoch}</span>
            <span data-testid="telemetry-total-epochs">{telemetry.totalEpochs}</span>
            <span data-testid="telemetry-loss">{telemetry.loss}</span>
            <span data-testid="telemetry-val-loss">{telemetry.valLoss}</span>
            <span data-testid="telemetry-val-data-loss">{telemetry.valDataLoss}</span>
            <span data-testid="telemetry-val-physics-loss">{telemetry.valPhysicsLoss}</span>
            <span data-testid="telemetry-learning-rate">{telemetry.learningRate}</span>
            <span data-testid="telemetry-lr-phase">{telemetry.lrPhase}</span>
            <span data-testid="telemetry-optimizer">{telemetry.optimizer}</span>
            <span data-testid="telemetry-stage">{telemetry.stage}</span>
          </div>
          <div class="summary-cards">
            <article class="stat">
              <div class="label">Step 1: Reference Solve</div>
              <div class="value">{femResult ? 'ready' : 'not run'}</div>
              <div class="meta" style="margin-top:0.3rem;">
                Solve FEM once to anchor the current case before training or validating predictions.
              </div>
            </article>
            <article class="stat">
              <div class="label">Step 2: Train Surrogate</div>
              <div class="value">{trainingActive ? trainingPhaseLabel : lastRunOutcomeLabel}</div>
              <div class="meta" style="margin-top:0.3rem;">
                {#if trainingActive}
                  {progressEvidenceLabel}
                  {#if latestObservedLoss !== null && (trainingHistory.length > 0 || (trainingProgress?.epoch ?? 0) > 0)}
                    · observed loss <NumberFlow value={latestObservedLoss} format={{ maximumSignificantDigits: 4 }} />
                  {/if}
                {:else if lastTrainResult}
                  {lastTrainResult.stopReason} after
                  <NumberFlow value={lastTrainResult.completedEpochs} format={{ maximumFractionDigits: 0 }} />
                  epochs
                {:else}
                  Train on this regime so nearby cases can be inferred quickly.
                {/if}
              </div>
            </article>
            <article class="stat">
              <div class="label">Step 3: Infer Surrogate</div>
              <div class="value">{inferenceStateLabel}</div>
              <div class="meta" style="margin-top:0.3rem;">
                After training, change geometry or loads within the same regime and run PINO surrogate inference.
              </div>
            </article>
            <article class="stat">
              <div class="label">Current Recommendation</div>
              <div class="value">{trainingActive ? 'training in progress' : 'next action'}</div>
              <div class="meta" style="margin-top:0.3rem;">{nextRecommendedAction}</div>
            </article>
            <article class="stat">
              <div class="label">Inference Status</div>
              <div class="value">{inferActive ? 'working' : annResult ? 'complete' : 'idle'}</div>
              <div class="meta" style="margin-top:0.3rem;">{inferenceStatusMessage}</div>
              {#if inferActive}
                <div class="startup-progress-track" style="margin-top:0.75rem;">
                  <div class="startup-progress-fill" style="width:68%;"></div>
                </div>
              {/if}
            </article>
          </div>
          <div class="actions">
            <button class="btn-primary" onclick={onSolveFem}>Solve FEM Case</button>
            <button class="btn-secondary" onclick={onTrainAnn} disabled={trainingActive}>
              {trainingActive ? 'Training…' : 'Train Surrogate'}
            </button>
            <button class="btn-secondary" onclick={onInferAnn} disabled={inferActive || !trainedModelReady}>
              {inferActive ? 'Inferring…' : trainedModelReady ? 'Infer Surrogate' : 'Train First'}
            </button>
            <button class="btn-secondary" onclick={onStopTrain} disabled={!trainingActive || trainingStatus?.stopRequested}>
              Stop Training
            </button>
          </div>
          <details open={trainingActive || !!lastTrainResult}>
            <summary>Training summary</summary>
            <div class="stack" style="margin-top:0.75rem;gap:0.75rem;">
              <div style="display:flex;justify-content:space-between;gap:0.75rem;align-items:center;flex-wrap:wrap;">
                <p>Training progress</p>
                <div class="kicker">
                  <span class="chip">{trainingActive ? 'running' : 'idle'}</span>
                  {#if trainingStatus?.stopRequested}
                    <span class="chip warn">stop requested</span>
                  {/if}
                  <span class="chip">
                    <NumberFlow value={displayEpoch} format={{ maximumFractionDigits: 0 }} />
                    /
                    <NumberFlow value={displayTotalEpochs} format={{ maximumFractionDigits: 0 }} />
                  </span>
                  <span class="chip">
                    val loss:
                    {#if trainingTick && trainingTick.valLoss > 0}
                      <NumberFlow value={trainingTick.valLoss} format={{ maximumSignificantDigits: 4 }} />
                    {:else if trainingActive}
                      estimating...
                    {:else}
                      -
                    {/if}
                  </span>
                  <span class="chip">elapsed: <NumberFlow value={trainingElapsedS} format={{ maximumFractionDigits: 1 }} />s</span>
                </div>
              </div>
              <div class="startup-progress-track">
                <div
                  class="startup-progress-fill"
                  style={`width:${
                    trainingActive && displayEpoch === 0
                      ? 2
                      : Math.max(0, Math.min(100, progressDisplayRatio * 100))
                  }%;`}
                ></div>
              </div>
              <div class="summary-cards">
                <article class="stat">
                  <div class="label">Best Val Loss</div>
                  <div class="value">
                    {#if displayBestValLoss !== null}
                      <NumberFlow value={displayBestValLoss} format={{ maximumSignificantDigits: 6 }} />
                    {:else}
                      -
                    {/if}
                  </div>
                </article>
                <article class="stat">
                  <div class="label">Epochs Since Improvement</div>
                  <div class="value">
                    <NumberFlow value={displayEpochsSinceImprovement} format={{ maximumFractionDigits: 0 }} />
                  </div>
                </article>
                <article class="stat">
                  <div class="label">Stage / Optimizer</div>
                  <div class="value">
                    {trainingStatus?.diagnostics?.activeStage ?? telemetry.stage} / {trainingStatus?.diagnostics?.activeOptimizer ?? telemetry.optimizer}
                  </div>
                </article>
                <article class="stat">
                  <div class="label">Current LR</div>
                  <div class="value">
                    <NumberFlow value={trainingStatus?.diagnostics?.currentLearningRate ?? trainLr} format={{ maximumSignificantDigits: 6 }} />
                  </div>
                </article>
                <article class="stat">
                  <div class="label">Training Phase</div>
                  <div class="value">{trainingPhaseLabel}</div>
                  <div class="meta" style="margin-top:0.3rem;">
                    {#if trainingActive}
                      {progressEvidenceLabel}
                    {:else}
                      {lastRunOutcomeLabel}
                    {/if}
                  </div>
                </article>
                <article class="stat">
                  <div class="label">Target Loss</div>
                  <div class="value">{trainTarget.toExponential(0)}</div>
                  <div class="meta" style="margin-top:0.3rem;">live validation must cross this threshold before the run is called converged</div>
                </article>
                <article class="stat">
                  <div class="label">Observed Gap</div>
                  <div class="value">
                    {#if targetLossGap !== null}
                      <NumberFlow value={targetLossGap} format={{ maximumSignificantDigits: 4 }} />
                    {:else}
                      -
                    {/if}
                  </div>
                  <div class="meta" style="margin-top:0.3rem;">latest observed loss minus the target loss</div>
                </article>
                <article class="stat">
                  <div class="label">Progress Frames</div>
                  <div class="value">{progressEvidenceLabel}</div>
                  <div class="meta" style="margin-top:0.3rem;">epoch 0, tick updates, and progress snapshots are surfaced as they arrive</div>
                </article>
                <article class="stat">
                  <div class="label">Poll Health</div>
                  <div class="value" data-testid="poll-health-status">{pollHealthLabel}</div>
                  <div class="meta" style="margin-top:0.3rem;">
                    tick/progress/status: {tickPollTimeouts}/{progressPollTimeouts}/{statusPollTimeouts}
                  </div>
                </article>
                <article class="stat">
                  <div class="label">Throughput</div>
                  <div class="value" data-testid="throughput-health">{throughputHealth}</div>
                  <div class="meta" style="margin-top:0.3rem;">
                    cadence {Math.round(tickCadenceMs)}/{Math.round(progressCadenceMs)} ms
                  </div>
                </article>
              </div>
              <details>
                <summary>Deep diagnostics</summary>
                <div class="stack" style="margin-top:0.75rem;gap:0.75rem;">
                  <div class="stage-progress-grid">
                    <article class="stat">
                      <div class="label">Overall Progress</div>
                      <div class="meta">
                        <span data-testid="overall-progress-epoch">
                          <NumberFlow value={displayEpoch} format={{ maximumFractionDigits: 0 }} />
                        </span>
                        /
                        <span data-testid="overall-progress-total">
                          <NumberFlow value={stageSchedule.scheduleTotal} format={{ maximumFractionDigits: 0 }} />
                        </span>
                        epochs
                      </div>
                      <div class="startup-progress-track stage-track">
                        <div
                          class="startup-progress-fill stage-fill"
                          data-testid="overall-progress-fill"
                          data-epoch={displayEpoch}
                          data-total={stageSchedule.scheduleTotal}
                          style={`width:${
                            trainingActive && displayEpoch === 0
                              ? 2
                              : Math.round(
                                  Math.max(0, Math.min(1, displayEpoch / Math.max(1, stageSchedule.scheduleTotal))) *
                                    100
                                )
                          }%;`}
                        ></div>
                      </div>
                    </article>
                    <article class="stat">
                      <div class="label">Stage 1</div>
                      <div class="meta"><NumberFlow value={stageSchedule.stage1.span} format={{ maximumFractionDigits: 0 }} /> epochs</div>
                      <div class="startup-progress-track stage-track">
                        <div class="startup-progress-fill stage-fill" data-testid="stage1-progress-fill" data-span={stageSchedule.stage1.span} style={`width:${trainingActive && displayEpoch === 0 ? 2 : Math.round(stageSchedule.stage1.ratio * 100)}%;`}></div>
                      </div>
                    </article>
                    <article class="stat">
                      <div class="label">Stage 2</div>
                      <div class="meta"><NumberFlow value={stageSchedule.stage2.span} format={{ maximumFractionDigits: 0 }} /> epochs</div>
                      <div class="startup-progress-track stage-track">
                        <div class="startup-progress-fill stage-fill" data-testid="stage2-progress-fill" data-span={stageSchedule.stage2.span} style={`width:${trainingActive && displayEpoch === 0 ? 2 : Math.round(stageSchedule.stage2.ratio * 100)}%;`}></div>
                      </div>
                    </article>
                    <article class="stat">
                      <div class="label">Stage 3</div>
                      <div class="meta"><NumberFlow value={stageSchedule.stage3.span} format={{ maximumFractionDigits: 0 }} /> epochs</div>
                      <div class="startup-progress-track stage-track">
                        <div class="startup-progress-fill stage-fill" data-testid="stage3-progress-fill" data-span={stageSchedule.stage3.span} style={`width:${trainingActive && displayEpoch === 0 ? 2 : Math.round(stageSchedule.stage3.ratio * 100)}%;`}></div>
                      </div>
                    </article>
                  </div>
                  <article class="stat stack">
                    <div class="label">Curriculum Gate Inspector</div>
                    <div class="meta">backend stage: {stageGateInspector.backendStage}</div>
                    <div style="display:grid;gap:0.28rem;">
                      {#each stageGateInspector.rows as row}
                        <div data-testid={`gate-${row.id}`} style="display:grid;grid-template-columns:1.35fr 0.8fr 1fr 1fr;gap:0.45rem;align-items:center;">
                          <span>{row.label}</span>
                          <span class={`chip ${row.status === 'complete' ? 'ok' : row.status === 'active' ? 'warn' : ''}`}>{row.status}</span>
                          <span class="meta">{row.start} → {row.end}</span>
                          <span class="meta">{row.criterion}</span>
                        </div>
                      {/each}
                    </div>
                  </article>
                  <div class="summary-cards">
                    <article class="stat">
                      <div class="label">Run Fingerprint</div>
                      <div class="meta" style="word-break:break-all;" data-testid="run-fingerprint-current">{currentRunFingerprint}</div>
                    </article>
                    <article class="stat">
                      <div class="label">Last Started Fingerprint</div>
                      <div class="meta" style="word-break:break-all;" data-testid="run-fingerprint-last">{lastRunFingerprint || 'n/a'}</div>
                    </article>
                    <article class="stat">
                      <div class="label">Residual Weights (M/K/MAT/BC)</div>
                      <div class="value">
                        <NumberFlow value={trainingStatus?.diagnostics?.residualWeightMomentum ?? 1} format={{ maximumSignificantDigits: 3 }} />
                        /
                        <NumberFlow value={trainingStatus?.diagnostics?.residualWeightKinematics ?? 1} format={{ maximumSignificantDigits: 3 }} />
                        /
                        <NumberFlow value={trainingStatus?.diagnostics?.residualWeightMaterial ?? 1} format={{ maximumSignificantDigits: 3 }} />
                        /
                        <NumberFlow value={trainingStatus?.diagnostics?.residualWeightBoundary ?? 1} format={{ maximumSignificantDigits: 3 }} />
                      </div>
                    </article>
                    <article class="stat">
                      <div class="label">Target Floor Estimate</div>
                      <div class="value">
                        <NumberFlow value={trainingStatus?.diagnostics?.targetFloorEstimate ?? 0} format={{ maximumSignificantDigits: 6 }} />
                      </div>
                    </article>
                    <article class="stat">
                      <div class="label">Optimizer Switches / Backoffs</div>
                      <div class="value">
                        <NumberFlow value={trainingStatus?.diagnostics?.optimizerSwitches ?? 0} format={{ maximumFractionDigits: 0 }} />
                        /
                        <NumberFlow value={trainingStatus?.diagnostics?.curriculumBackoffs ?? 0} format={{ maximumFractionDigits: 0 }} />
                      </div>
                    </article>
                    <article class="stat">
                      <div class="label">Train Data / Cap</div>
                      <div class="value">
                        <NumberFlow value={trainingStatus?.diagnostics?.trainDataSize ?? trainingProgress?.trainDataSize ?? 0} format={{ maximumFractionDigits: 0 }} />
                        /
                        <NumberFlow value={trainingStatus?.diagnostics?.trainDataCap ?? trainingProgress?.trainDataCap ?? 0} format={{ maximumFractionDigits: 0 }} />
                      </div>
                    </article>
                  </div>
                  <article class="stat stack">
                    <div class="label">Autonomy Event Log (Recent)</div>
                    {#if trainingStatus?.diagnostics?.recentEvents?.length}
                      <div class="value" style="max-height: 140px; overflow: auto; display: grid; gap: 0.25rem;">
                        {#each [...(trainingStatus?.diagnostics?.recentEvents ?? [])].slice(-8).reverse() as eventLine}
                          <div>{eventLine}</div>
                        {/each}
                      </div>
                    {:else}
                      <div class="value">No autonomous events yet.</div>
                    {/if}
                  </article>
                  <article class="stat stack">
                    <div class="label">Epoch Window Table (Last 10 @ 1000)</div>
                    {#if epochWindowRows.length}
                      <div data-testid="epoch-window-table" style="display:grid;gap:0.25rem;max-height:170px;overflow:auto;font-size:0.84rem;">
                        {#each epochWindowRows as row}
                          <div style="display:grid;grid-template-columns:0.9fr 1fr 1fr 1fr 0.9fr 0.9fr;gap:0.35rem;">
                            <span>e{row.windowEpoch}</span>
                            <span><NumberFlow value={row.valDataLoss} format={{ maximumSignificantDigits: 4 }} /></span>
                            <span><NumberFlow value={row.valPhysicsLoss} format={{ maximumSignificantDigits: 4 }} /></span>
                            <span>{row.lrPhase}</span>
                            <span>{row.optimizer}</span>
                            <span>{row.stage}</span>
                          </div>
                        {/each}
                      </div>
                    {:else}
                      <div class="value">No 1000-epoch windows reached yet.</div>
                    {/if}
                  </article>
                </div>
              </details>
            </div>
          </details>
        </section>

        <section class="panel stack">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem;flex-wrap:wrap;">
            <div class="stack" style="gap:0.35rem;">
              <h2>Training Setup</h2>
              <p>Keep this compact area for the settings you are likely to change often. Expert tuning stays below.</p>
            </div>
            <div class="kicker">
              <span class="chip">{trainAnalysisType}</span>
              <span class="chip">target {trainTarget.toExponential(0)}</span>
              <span class="chip">{trainPinnBackend}</span>
            </div>
          </div>
          <div class="field-grid">
            <label class="field">
              <span>Analysis Type</span>
              <select bind:value={trainAnalysisType}>
                <option value="general">general</option>
                <option value="cantilever">cantilever</option>
                <option value="plate-hole">plate-hole</option>
              </select>
            </label>
            <label class="field">
              <span>Curriculum Preset</span>
              <select bind:value={trainCurriculumPreset}>
                <option value="cantilever-data-first">cantilever-data-first</option>
                <option value="plate-hole-balanced">plate-hole-balanced</option>
                <option value="contact-stabilized">contact-stabilized</option>
              </select>
            </label>
            <label class="field">
              <span>Apply Curriculum</span>
              <button class="btn-primary" onclick={onApplyCurriculumPreset}>Apply Preset</button>
            </label>
            <label class="field">
              <span>Auto Training Recipe</span>
              <select bind:value={trainAutoRecipe}>
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </label>
            <label class="field">
              <span>Train Epochs</span>
              <input type="number" bind:value={trainEpochs} step="1" />
            </label>
            <label class="field">
              <span>Target Loss</span>
              <input type="number" bind:value={trainTarget} step="any" min="0" />
            </label>
            <label class="field">
              <span>Learning Rate</span>
              <input type="number" bind:value={trainLr} step="0.0001" />
            </label>
            <label class="field">
              <span>PINN Backend</span>
              <select bind:value={trainPinnBackend}>
                <option value="pino-ndarray-cpu">pino-ndarray-cpu</option>
                <option value="pino-candle-cpu">pino-candle-cpu</option>
                <option value="pino-candle-cuda">pino-candle-cuda</option>
                <option value="pino-candle-metal">pino-candle-metal</option>
              </select>
            </label>
            <label class="field">
              <span>Max Total Epochs</span>
              <input type="number" bind:value={trainMaxEpochs} step="10" min="1" />
            </label>
            <label class="field">
              <span>Checkpoint Every (epochs)</span>
              <input type="number" bind:value={trainCheckpointEveryEpochs} min="0" step="1" />
            </label>
          </div>
          <div class="summary-cards">
            <article class="stat">
              <div class="label">Auto Recipe</div>
              <div class="value">{trainAutoRecipe === 'true' ? 'enabled' : 'manual'}</div>
              <div class="meta" style="margin-top:0.3rem;">
                {trainAutoRecipe === 'true'
                  ? `App expands the current case into 3 training seeds and 2 validation cases, then grows to about ${autoTrainingRecipe.estimatedFemCases} lightweight FEM cases for the local recipe preview.`
                  : 'Training stays on the current case only; automatic expansion is disabled.'}
              </div>
            </article>
            <article class="stat">
              <div class="label">Convergence Policy</div>
              <div class="value">{trainTarget.toExponential(0)}</div>
              <div class="meta" style="margin-top:0.3rem;">
                Visible progress starts at epoch 0, then the run is only called converged when the live validation loss reaches the selected target.
              </div>
            </article>
            <article class="stat">
              <div class="label">Inputs Varied Automatically</div>
              <div class="meta" style="margin-top:0.3rem;">{autoTrainingRecipe.variedInputs.join(', ')}</div>
            </article>
            <article class="stat">
              <div class="label">Validation Before Trusting Surrogate</div>
              <div class="meta" style="margin-top:0.3rem;">{autoTrainingRecipe.validationChecks.join(' ')}</div>
            </article>
          </div>
          <details>
            <summary>Automatic training recipe</summary>
            <div class="stack" style="margin-top:0.8rem;">
              <p>The app derives this recipe from the current structural case and selected analysis type.</p>
              <div class="summary-cards">
                <article class="stat">
                  <div class="label">Suggested Ranges</div>
                  <div class="meta" style="margin-top:0.3rem;">
                    {#each autoTrainingRecipe.rangeLines as line (`range-${line}`)}
                      <div>{line}</div>
                    {/each}
                  </div>
                </article>
                <article class="stat">
                  <div class="label">Seed Cases</div>
                  <div class="meta" style="margin-top:0.3rem;">
                    {#each autoTrainingRecipe.seedCases as c, idx (`seed-${idx}`)}
                      <div>
                        #{idx + 1}: L {c.geometry.lengthIn.toFixed(3)} in, W {c.geometry.widthIn.toFixed(3)} in,
                        t {c.geometry.thicknessIn.toFixed(3)} in,
                        {c.load.verticalPointLoadLbf !== 0
                          ? `P_v ${c.load.verticalPointLoadLbf.toFixed(1)} lbf`
                          : `P_a ${c.load.axialLoadLbf.toFixed(1)} lbf`}
                      </div>
                    {/each}
                  </div>
                </article>
                <article class="stat">
                  <div class="label">Holdout Validation Cases</div>
                  <div class="meta" style="margin-top:0.3rem;">
                    {#each autoTrainingRecipe.validationCases as c, idx (`holdout-${idx}`)}
                      <div>
                        holdout #{idx + 1}: L {c.geometry.lengthIn.toFixed(3)} in, W {c.geometry.widthIn.toFixed(3)} in,
                        t {c.geometry.thicknessIn.toFixed(3)} in,
                        {c.load.verticalPointLoadLbf !== 0
                          ? `P_v ${c.load.verticalPointLoadLbf.toFixed(1)} lbf`
                          : `P_a ${c.load.axialLoadLbf.toFixed(1)} lbf`}
                      </div>
                    {/each}
                  </div>
                </article>
              </div>
            </div>
          </details>
          <details>
            <summary>Advanced training settings</summary>
            <div class="field-grid" style="margin-top:0.9rem;">
              <label class="field">
                <span>Auto Train</span>
                <select bind:value={trainAutoMode}>
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              </label>
              <label class="field">
                <span>Min Improvement</span>
                <input type="number" bind:value={trainMinImprovement} step="0.0000001" />
              </label>
              <label class="field">
                <span>Update Every (epochs)</span>
                <input type="number" bind:value={trainUpdateEveryEpochs} min="1" step="1" />
              </label>
              <label class="field">
                <span>Network Update Every (epochs)</span>
                <input type="number" bind:value={trainNetworkUpdateEveryEpochs} min="1" step="1" />
              </label>
              <label class="field">
                <span>Online Active Learning</span>
                <select bind:value={trainOnlineActiveLearning}>
                  <option value="false">false</option>
                  <option value="true">true</option>
                </select>
              </label>
              <label class="field">
                <span>Autonomous Mode</span>
                <select bind:value={trainAutonomousMode}>
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              </label>
              <label class="field">
                <span>Max Topology</span>
                <input type="number" bind:value={trainMaxTopology} min="8" step="1" />
              </label>
              <label class="field">
                <span>Max Backoffs</span>
                <input type="number" bind:value={trainMaxBackoffs} min="1" step="1" />
              </label>
              <label class="field">
                <span>Max Optimizer Switches</span>
                <input type="number" bind:value={trainMaxOptimizerSwitches} min="1" step="1" />
              </label>
              <label class="field">
                <span>Checkpoint Retention</span>
                <input type="number" bind:value={trainCheckpointRetention} min="1" step="1" />
              </label>
              <label class="field">
                <span>Training Seed</span>
                <input type="number" bind:value={trainSeed} min="0" step="1" />
              </label>
              <label class="field">
                <span>Collocation Points</span>
                <input type="number" bind:value={trainCollocationPoints} min="64" step="64" />
              </label>
              <label class="field">
                <span>Boundary Points</span>
                <input type="number" bind:value={trainBoundaryPoints} min="16" step="16" />
              </label>
              <label class="field">
                <span>Interface Points</span>
                <input type="number" bind:value={trainInterfacePoints} min="16" step="16" />
              </label>
              <label class="field">
                <span>Residual W: Momentum</span>
                <input type="number" bind:value={trainResidualWeightMomentum} min="0" step="0.1" />
              </label>
              <label class="field">
                <span>Residual W: Kinematics</span>
                <input type="number" bind:value={trainResidualWeightKinematics} min="0" step="0.1" />
              </label>
              <label class="field">
                <span>Residual W: Material</span>
                <input type="number" bind:value={trainResidualWeightMaterial} min="0" step="0.1" />
              </label>
              <label class="field">
                <span>Residual W: Boundary</span>
                <input type="number" bind:value={trainResidualWeightBoundary} min="0" step="0.1" />
              </label>
              <label class="field">
                <span>Stage 1 Epochs</span>
                <input type="number" bind:value={trainStage1Epochs} min="0" step="10" />
              </label>
              <label class="field">
                <span>Stage 2 Epochs</span>
                <input type="number" bind:value={trainStage2Epochs} min="0" step="10" />
              </label>
              <label class="field">
                <span>Stage 3 Ramp Epochs</span>
                <input type="number" bind:value={trainStage3RampEpochs} min="0" step="10" />
              </label>
              <label class="field">
                <span>Contact Penalty</span>
                <input type="number" bind:value={trainContactPenalty} min="0" step="0.1" />
              </label>
              <label class="field">
                <span>Plasticity Factor</span>
                <input type="number" bind:value={trainPlasticityFactor} min="0" max="1" step="0.01" />
              </label>
            </div>
          </details>
        </section>

        <section class="panel stack">
          <h2>Use The Trained PINO Surrogate</h2>
          <div class="summary-cards">
            <article class="stat">
              <div class="label">1. Keep The Same Regime</div>
              <div class="meta">Stay within the same case family you trained on, such as cantilever or plate-with-hole.</div>
            </article>
            <article class="stat">
              <div class="label">2. Change Inputs</div>
              <div class="meta">Adjust loads, dimensions, or material values for a nearby case you want to estimate.</div>
            </article>
            <article class="stat">
              <div class="label">3. Run Infer Surrogate</div>
              <div class="meta">Use surrogate inference for a fast prediction, then validate important cases with Solve FEM Case.</div>
            </article>
            <article class="stat">
              <div class="label">4. Re-Train When Needed</div>
              <div class="meta">If the geometry or load range changes materially, retrain before trusting the prediction.</div>
            </article>
          </div>
        </section>

        <details class="panel stack">
          <summary>Recovery, extra studies, and export</summary>
          <div class="stack" style="margin-top:0.9rem;">
            <section class="stack">
              <h3>Recovery</h3>
              <div class="summary-cards">
                <article class="stat">
                  <div class="label">Stored Checkpoints</div>
                  <div class="value"><NumberFlow value={checkpoints.length} format={{ maximumFractionDigits: 0 }} /></div>
                </article>
                <article class="stat">
                  <div class="label">Latest Checkpoint</div>
                  <div class="value" data-testid="checkpoint-latest-tag">
                    {checkpointHealth.latest ? `${checkpointHealth.latest.tag} (${checkpointHealth.latest.id.slice(0, 8)})` : '-'}
                  </div>
                </article>
                <article class="stat">
                  <div class="label">Best Checkpoint</div>
                  <div class="value" data-testid="checkpoint-best-tag">
                    {checkpointHealth.best ? `${checkpointHealth.best.tag} (${checkpointHealth.best.id.slice(0, 8)})` : '-'}
                  </div>
                </article>
                <article class="stat">
                  <div class="label">Recoverability</div>
                  <div class="value" data-testid="checkpoint-recovery">{checkpointHealth.recoveryState}</div>
                </article>
              </div>
              <div class="actions">
                <button class="btn-secondary" onclick={onModelStatus}>Model Status</button>
                <button class="btn-secondary" onclick={onResetModel}>Reset Surrogate Model</button>
                <button class="btn-secondary" onclick={onSaveCheckpoint}>Save Checkpoint</button>
                <button class="btn-secondary" onclick={onResumeLatestCheckpoint}>Resume Latest CP</button>
                <button class="btn-secondary" onclick={onResumeBestCheckpoint}>Resume Best CP</button>
                <button class="btn-secondary" onclick={onListCheckpoints}>List Checkpoints</button>
                <button class="btn-secondary" onclick={onPurgeCheckpoints}>Purge Old CPs</button>
              </div>
            </section>

            <section class="stack">
              <h3>Additional Studies</h3>
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
              </div>
              <div class="actions">
                <button class="btn-secondary" onclick={onFailure}>Evaluate Failure</button>
                <button class="btn-secondary" onclick={onThermal}>Run Thermal Case</button>
                <button class="btn-secondary" onclick={onDynamic}>Run Dynamic Case</button>
              </div>
            </section>

            <section class="stack">
              <h3>Safeguards</h3>
              <div class="field-grid">
                <label class="field">
                  <span>Preset</span>
                  <select bind:value={safeguardPreset}>
                    <option value="conservative">conservative</option>
                    <option value="balanced">balanced</option>
                    <option value="performance">performance</option>
                    <option value="custom">custom</option>
                  </select>
                </label>
                <label class="field">
                  <span>Uncertainty Threshold</span>
                  <input type="number" bind:value={safeguardUncertainty} min="0.01" max="0.99" step="0.01" />
                </label>
                <label class="field">
                  <span>Residual Threshold</span>
                  <input type="number" bind:value={safeguardResidual} min="0.000001" max="10" step="0.001" />
                </label>
                <label class="field">
                  <span>Adaptive by Geometry</span>
                  <select bind:value={safeguardAdaptive}>
                    <option value="true">true</option>
                    <option value="false">false</option>
                  </select>
                </label>
                <label class="field">
                  <span>Apply</span>
                  <button class="btn-primary" onclick={onApplySafeguardPreset}>Apply Safeguards</button>
                </label>
              </div>
            </section>

            <section class="stack">
              <h3>Export</h3>
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
        </details>
      </div>

      <div class="stack">
        <NeuralNetworkLive
          {trainingActive}
          tick={trainingTick}
          progress={trainingProgress}
          history={trainingHistory}
          targetLoss={trainTarget}
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
                <div class="value">
                  {(lastTrainResult.reachedTargetLoss ?? (lastTrainResult.stopReason === 'target-loss-reached')) ? 'yes' : 'no'}
                </div>
              </article>
              <article class="stat">
                <div class="label">Autonomous Converged</div>
                <div class="value">
                  {lastTrainResult.reachedAutonomousConvergence ? 'yes' : 'no'}
                </div>
              </article>
              <article class="stat">
                <div class="label">Completed Epochs</div>
                <div class="value"><NumberFlow value={lastTrainResult.completedEpochs} format={{ maximumFractionDigits: 0 }} /></div>
              </article>
              <article class="stat">
                <div class="label">Topology Changes</div>
                <div class="value">{lastTrainResult.grew ? 'grow ' : ''}{lastTrainResult.pruned ? 'prune' : ''}</div>
              </article>
              {#if lastTrainResult.pino}
                <article class="stat">
                  <div class="label">PINO Engine</div>
                  <div class="value">{lastTrainResult.pino.engineId}</div>
                </article>
                <article class="stat">
                  <div class="label">Operator Grid</div>
                  <div class="value">
                    {lastTrainResult.pino.operatorGrid3d?.nx ?? lastTrainResult.pino.operatorGrid.nx}
                    x
                    {lastTrainResult.pino.operatorGrid3d?.ny ?? lastTrainResult.pino.operatorGrid.ny}
                    x
                    {lastTrainResult.pino.operatorGrid3d?.nz ?? lastTrainResult.pino.operatorGrid.nz}
                  </div>
                </article>
                <article class="stat">
                  <div class="label">Physics Model</div>
                  <div class="value">{lastTrainResult.pino.physicsModel}</div>
                </article>
                <article class="stat">
                  <div class="label">Spectral Modes</div>
                  <div class="value">{lastTrainResult.pino.spectralModes}</div>
                </article>
                <article class="stat">
                  <div class="label">PINO Backend</div>
                  <div class="value">{lastTrainResult.pino.backend}</div>
                </article>
              {/if}
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
          yieldStrengthPsi={solveInput.material.yieldStrengthPsi}
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
            <p>Run <strong>Solve FEM Case</strong> to compute stress/deflection along the beam length, then train the PINO surrogate for adaptive prediction.</p>
          </section>
        {/if}
      </div>
    </div>
  </div>
</main>
