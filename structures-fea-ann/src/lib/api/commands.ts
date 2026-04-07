import { invoke } from '@tauri-apps/api/core';
import type {
  AnnResult,
  CheckpointRetentionPolicy,
  CheckpointSaveInput,
  DynamicInput,
  DynamicResult,
  ExportResult,
  FailureInput,
  FailureResult,
  FemResult,
  ModelStatus,
  PurgeCheckpointsResult,
  ReportInput,
  RuntimeFingerprint,
  ResumeTrainingResult,
  SafeguardSettings,
  SolveInput,
  ThermalInput,
  ThermalResult,
  TrainingBenchmarkManifest,
  TrainResult,
  TrainingBatch,
  TrainingCheckpointInfo,
  TrainingProgressEvent,
  TrainingRunStatus,
  TrainingTickEvent
} from '$lib/types/contracts';

const isTauriRuntime = () =>
  typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

export const getRuntimeKind = () => (isTauriRuntime() ? 'tauri' : 'mock');

export const getRuntimeFingerprint = () => {
  if (isTauriRuntime()) return invoke<RuntimeFingerprint>('get_runtime_fingerprint');
  return Promise.resolve({
    appVersion: '0.1.0',
    buildProfile: 'mock',
    targetOs: (typeof navigator !== 'undefined' ? navigator.platform : 'unknown').toLowerCase(),
    targetArch: 'unknown',
    debugBuild: true,
    gitCommit: 'mock',
    buildTimeUtc: 'mock'
  });
};

type MockState = {
  modelVersion: number;
  architecture: number[];
  learningRate: number;
  lastLoss: number;
  trainSamples: number;
  auditFrequency: number;
  fallbackEnabled: boolean;
};

const mockState: MockState = {
  modelVersion: 1,
  architecture: [19, 16, 16, 9],
  learningRate: 5e-4,
  lastLoss: 0.2,
  trainSamples: 0,
  auditFrequency: 5,
  fallbackEnabled: true
};

const resolveMockTrainingMode = (batch?: TrainingBatch) =>
  batch?.trainingMode ?? (batch?.benchmarkId ? 'benchmark' : 'legacy-mixed-exact');

const mockTrainingBenchmarks: TrainingBenchmarkManifest[] = [
  {
    id: 'benchmark_bar_1d',
    title: '1D Bar',
    description: 'Displacement-primary axial bar sanity benchmark with exact closed-form response.',
    trainingMode: 'benchmark',
    analysisType: 'general',
    gateName: 'Gate 1',
    gateTargetLoss: 1e-3,
    recommendedLearningRate: 5e-4,
    maxRuntimeSeconds: 45,
    recommendedEpochs: 256,
    active: true
  },
  {
    id: 'benchmark_cantilever_2d',
    title: '2D Cantilever',
    description: 'Isolated cantilever benchmark for exact displacement-primary convergence before harder families.',
    trainingMode: 'benchmark',
    analysisType: 'cantilever',
    gateName: 'Gate 3',
    gateTargetLoss: 1e-4,
    recommendedLearningRate: 5e-4,
    maxRuntimeSeconds: 180,
    recommendedEpochs: 1024,
    active: true
  },
  {
    id: 'benchmark_patch_test_2d',
    title: '2D Patch Test',
    description: 'Linear elasticity patch test used to verify exact reproduction and residual consistency.',
    trainingMode: 'benchmark',
    analysisType: 'general',
    gateName: 'Gate 4',
    gateTargetLoss: 1e-4,
    recommendedLearningRate: 4e-4,
    maxRuntimeSeconds: 180,
    recommendedEpochs: 1024,
    active: true
  },
  {
    id: 'benchmark_plate_hole_2d',
    title: 'Plate With Hole',
    description: 'Plate-with-hole benchmark promoted only after simpler benchmark gates are stable.',
    trainingMode: 'benchmark',
    analysisType: 'plate-hole',
    gateName: 'Gate 4',
    gateTargetLoss: 1e-2,
    recommendedLearningRate: 3.5e-4,
    maxRuntimeSeconds: 240,
    recommendedEpochs: 1536,
    active: true
  }
];

let lastMockTick: TrainingTickEvent = {
  epoch: 0,
  totalEpochs: 0,
  loss: 0,
  valLoss: 0,
  learningRate: mockState.learningRate,
  architecture: [...mockState.architecture],
  progressRatio: 0
};
let lastMockProgress: TrainingProgressEvent = {
  epoch: 0,
  totalEpochs: 0,
  loss: 0,
  valLoss: 0,
  dataLoss: 0,
  physicsLoss: 0,
  valDataLoss: 0,
  valPhysicsLoss: 0,
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
  stageId: 'idle',
  optimizerId: 'pino-adam',
  lrPhase: 'idle',
  targetBandLow: 0,
  targetBandHigh: 0,
  trendSlope: 0,
  trendVariance: 0,
  watchdogTriggerCount: 0,
  collocationSamplesAdded: 0,
  trainDataSize: 0,
  trainDataCap: 0,
  residualWeightMomentum: 1,
  residualWeightKinematics: 1,
  residualWeightMaterial: 1,
  residualWeightBoundary: 1,
  learningRate: mockState.learningRate,
  architecture: [...mockState.architecture],
  progressRatio: 0,
  trainingMode: 'legacy-mixed-exact',
  benchmarkId: null,
  gateStatus: 'queued',
  certifiedBestMetric: Number.MAX_VALUE,
  dominantBlocker: null,
  stalledReason: null,
  network: { layerSizes: [], nodes: [], connections: [] }
};
let mockTrainingStatus: TrainingRunStatus = {
  running: false,
  stopRequested: false,
  completed: false,
  diagnostics: {
    bestValLoss: Number.MAX_VALUE,
    epochsSinceImprovement: 0,
    lrSchedulePhase: 'idle',
    currentLearningRate: mockState.learningRate,
    dataWeight: 2,
    physicsWeight: 2,
    residualWeightMomentum: 1,
    residualWeightKinematics: 1,
    residualWeightMaterial: 1,
    residualWeightBoundary: 1,
    activeLearningRounds: 0,
    activeLearningSamplesAdded: 0,
    safeguardTriggers: 0,
    curriculumBackoffs: 0,
    optimizerSwitches: 0,
    checkpointRollbacks: 0,
    targetFloorEstimate: 0,
    trendStopReason: 'idle',
    activeStage: 'idle',
    activeOptimizer: 'pino-adam',
    boPresearchUsed: false,
    boSelectedArchitecture: [...mockState.architecture],
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
    collocationPoints: 512,
    boundaryPoints: 128,
    interfacePoints: 128,
    collocationSamplesAdded: 0,
    trainDataSize: 0,
    trainDataCap: 0,
    trainingMode: 'legacy-mixed-exact',
    benchmarkId: null,
    gateStatus: 'queued',
    certifiedBestMetric: Number.MAX_VALUE,
    reproducibilitySpread: null,
    dominantBlocker: null,
    stalledReason: null,
    runBudgetUsed: 0,
    runBudgetTotal: 0,
    recentEvents: []
  }
};

const mockCheckpoints: TrainingCheckpointInfo[] = [];
let mockSafeguardSettings: SafeguardSettings = {
  preset: 'balanced',
  uncertaintyThreshold: 0.26,
  residualThreshold: 0.24,
  adaptiveByGeometry: true
};

const emitMock = (name: string, payload: unknown) => {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(name, { detail: payload }));
  }
};

const fmtStations = (input: SolveInput) => {
  const L = input.geometry.lengthIn;
  const W = input.geometry.widthIn;
  const t = input.geometry.thicknessIn;
  const P = input.load.axialLoadLbf;
  const V = input.load.verticalPointLoadLbf;
  const area = Math.max(1e-9, W * t);
  const sigmaNom = P / area;
  const i = Math.max(1e-9, (t * W ** 3) / 12);
  const sigmaMax = V !== 0 ? sigmaNom + (V * L * 0.5 * W) / i : 3 * sigmaNom;
  const spread = Math.max(0.12 * L, 1e-6);
  const n = Math.max(8, input.mesh.nx);
  return Array.from({ length: n + 1 }, (_, i) => {
    const x = (L * i) / n;
    const bump = Math.exp(-Math.pow(x - L * 0.5, 2) / (2 * spread * spread));
    const sigmaTop = sigmaNom + (sigmaMax - sigmaNom) * bump;
    return {
      xIn: x,
      shearLbf: 0,
      momentLbIn: 0,
      sigmaTopPsi: sigmaTop,
      sigmaBottomPsi: -sigmaTop,
      deflectionIn: 0
    };
  });
};

const solveLocal = (input: SolveInput): FemResult => {
  const L = input.geometry.lengthIn;
  const W = input.geometry.widthIn;
  const t = input.geometry.thicknessIn;
  const P = input.load.axialLoadLbf;
  const V = input.load.verticalPointLoadLbf;
  const E = input.material.ePsi;
  const nu = input.material.nu;
  const area = Math.max(1e-9, W * t);
  const sigmaNom = P / area;
  const i = Math.max(1e-9, (t * W ** 3) / 12);
  const sigma = V !== 0 ? sigmaNom + (V * L * 0.5 * W) / i : 3 * sigmaNom;
  const eps = sigma / E;
  const tip = input.boundaryConditions.fixEndFace
    ? 0
    : V !== 0
      ? (V * L ** 3) / (3 * E * i)
      : (sigmaNom * L) / E;
  const vm = Math.abs(sigma);
  const k = E * area / Math.max(1e-9, L);
  const scf = sigmaNom === 0 ? 0 : Math.abs(sigma / sigmaNom);
  const fixedTip = input.boundaryConditions.fixEndFace ? 0 : tip;
  const rightFaceFixed = input.boundaryConditions.fixEndFace ? 0 : tip;

  return {
    nodalDisplacements: [
      { nodeId: 0, xIn: 0, yIn: 0, zIn: 0, uxIn: 0, uyIn: 0, uzIn: 0, dispMagIn: 0, vmPsi: vm },
      {
        nodeId: 1,
        xIn: L,
        yIn: W,
        zIn: t / 2,
        uxIn: fixedTip,
        uyIn: V !== 0 && !input.boundaryConditions.fixEndFace ? tip : 0,
        uzIn: 0,
        dispMagIn: Math.abs(fixedTip),
        vmPsi: vm
      }
    ],
    strainTensor: [[eps, 0, 0], [0, -nu * eps, 0], [0, 0, -nu * eps]],
    stressTensor: [[sigma, 0, 0], [0, 0, 0], [0, 0, 0]],
    principalStresses: [sigma, 0, 0],
    vonMisesPsi: vm,
    trescaPsi: Math.abs(sigma),
    maxPrincipalPsi: sigma,
    stiffnessMatrix: [[k, -k], [-k, k]],
    massMatrix: [[1, 0], [0, 1]],
    dampingMatrix: [[0.01, 0], [0, 0.01]],
    forceVector: [P, 0],
    displacementVector: [0, rightFaceFixed],
    beamStations: fmtStations(input),
    diagnostics: [
      'Web preview mode: local mock solver active.',
      `Mock structural solve: sigma_nom=${sigmaNom.toFixed(3)} psi, sigma_max=${sigma.toFixed(3)} psi, SCF=${scf.toFixed(3)}, support=${input.boundaryConditions.fixStartFace ? 'start-fixed' : 'free-start'}-${input.boundaryConditions.fixEndFace ? 'end-fixed' : 'free-end'}`
    ]
  };
};

const networkSnapshot = (): TrainingProgressEvent['network'] => {
  const [a, b, c, d] = mockState.architecture;
  const mkNodes = (layer: number, n: number) =>
    Array.from({ length: n }, (_, i) => ({
      id: `L${layer}N${i}`,
      layer,
      index: i,
      activation: Math.sin((i + 1) * 0.7) * 0.7,
      bias: Math.cos((i + 1) * 0.5) * 0.2,
      importance: 0.2 + Math.random() * 0.8
    }));
  const nodes = [...mkNodes(0, a), ...mkNodes(1, b), ...mkNodes(2, c), ...mkNodes(3, d)];
  const connections: TrainingProgressEvent['network']['connections'] = [];
  for (const from of nodes) {
    for (const to of nodes) {
      if (to.layer === from.layer + 1) {
        const w = (Math.random() - 0.5) * 2;
        connections.push({ fromId: from.id, toId: to.id, weight: w, magnitude: Math.abs(w) });
      }
    }
  }
  return { layerSizes: [...mockState.architecture], nodes, connections };
};

export const solveFemCase = (input: SolveInput) => {
  if (isTauriRuntime()) return invoke<FemResult>('solve_fem_case', { input });
  return Promise.resolve(solveLocal(input));
};

const runMockTrainingLoop = async (batch: TrainingBatch) => {
  if (mockTrainingStatus.running) return false;
  const trainingMode = resolveMockTrainingMode(batch);
  const benchmarkId = batch.benchmarkId ?? null;

  const epochs = Math.max(1, batch.epochs);
  const emitEvery = Math.max(1, batch.progressEmitEveryEpochs ?? 1);
  const networkEvery = Math.max(1, batch.networkEmitEveryEpochs ?? emitEvery * 25);
  const maxTotal = Math.max(epochs, batch.maxTotalEpochs ?? epochs * 20);
  const target = Math.max(0, batch.targetLoss);
  if (batch.learningRate) mockState.learningRate = batch.learningRate;

  let loss = Math.max(mockState.lastLoss, target * 3 + 0.02);
  let valLoss = loss * 1.02;
  let completed = 0;
  let grew = false;
  let pruned = false;
  let bestVal = Number.MAX_VALUE;
  let sinceImprove = 0;
  let lastStage = 'idle';
  let lastOptimizer = 'pino-adam';
  let lastLrPhase = 'idle';
  let recentEvents: string[] = [];
  const pushRecentEvent = (line: string) => {
    recentEvents = [...recentEvents, line].slice(-24);
  };
  mockTrainingStatus = {
    running: true,
    stopRequested: false,
    completed: false,
    diagnostics: {
      ...mockTrainingStatus.diagnostics,
      lrSchedulePhase: 'training',
      currentLearningRate: mockState.learningRate,
      trainingMode,
      benchmarkId,
      gateStatus: 'running',
      certifiedBestMetric: Number.MAX_VALUE,
      reproducibilitySpread: null,
      dominantBlocker: null,
      stalledReason: null,
      runBudgetUsed: 0,
      runBudgetTotal: maxTotal
    }
  };

  while (completed < maxTotal && valLoss > target && !mockTrainingStatus.stopRequested) {
    completed += 1;
    loss = Math.max(target * 0.5, loss * (0.965 + Math.random() * 0.02));
    valLoss = Math.max(target * 0.6, loss * (0.99 + Math.random() * 0.02));
    if (valLoss < bestVal) {
      bestVal = valLoss;
      sinceImprove = 0;
    } else {
      sinceImprove += 1;
    }

    if (completed % 80 === 0 && valLoss > target * 1.5 && mockState.architecture[mockState.architecture.length - 2] < 64) {
      mockState.architecture[mockState.architecture.length - 2] += 4;
      grew = true;
      mockState.modelVersion += 1;
    }
    if (completed % 120 === 0 && valLoss < target * 0.4 && mockState.architecture[mockState.architecture.length - 2] > 6) {
      mockState.architecture[mockState.architecture.length - 2] -= 2;
      pruned = true;
      mockState.modelVersion += 1;
    }

    const progress: TrainingProgressEvent = {
      epoch: completed,
      totalEpochs: maxTotal,
      loss,
      valLoss,
      dataLoss: loss * 0.82,
      physicsLoss: Math.max(1e-8, loss * 0.18),
      valDataLoss: valLoss * 0.8,
      valPhysicsLoss: Math.max(1e-8, valLoss * 0.2),
      momentumResidual: Math.max(1e-8, valLoss * 0.26),
      kinematicResidual: Math.max(1e-8, valLoss * 0.24),
      materialResidual: Math.max(1e-8, valLoss * 0.22),
      boundaryResidual: Math.max(1e-8, valLoss * 0.28),
      displacementFit: Math.max(1e-8, valLoss * 0.3),
      stressFit: Math.max(1e-8, valLoss * 0.34),
      invariantResidual: Math.max(1e-8, valLoss * 0.12),
      constitutiveNormalResidual: Math.max(1e-8, valLoss * 0.14),
      constitutiveShearResidual: Math.max(1e-8, valLoss * 0.08),
      valDisplacementFit: Math.max(1e-8, valLoss * 0.3),
      valStressFit: Math.max(1e-8, valLoss * 0.34),
      valInvariantResidual: Math.max(1e-8, valLoss * 0.12),
      valConstitutiveNormalResidual: Math.max(1e-8, valLoss * 0.14),
      valConstitutiveShearResidual: Math.max(1e-8, valLoss * 0.08),
      hybridMode: valLoss > 0.2 ? 'data-dominant' : valLoss > 0.08 ? 'hybrid' : 'pinn-dominant',
      stageId: completed < Math.max(30, Math.floor(maxTotal * 0.2)) ? 'stage-1' : 'stage-2',
      optimizerId: sinceImprove > 120 ? 'pino-lbfgs' : 'pino-adam',
      lrPhase: sinceImprove > 120 ? 'pino-decay' : 'pino-steady',
      targetBandLow: Math.max(0, bestVal * 0.97),
      targetBandHigh: Math.max(0, bestVal * 1.03),
      trendSlope: -Math.max(0, Math.min(0.01, (bestVal - valLoss) / Math.max(1, completed))),
      trendVariance: Math.max(1e-9, Math.abs(valLoss - bestVal) * 0.01),
      watchdogTriggerCount: 0,
      collocationSamplesAdded: Math.max(64, batch.collocationPoints ?? 512),
      trainDataSize: Math.max(128, Math.floor((batch.collocationPoints ?? 512) * 0.4)),
      trainDataCap: Math.max(256, Math.floor((batch.collocationPoints ?? 512) * 0.8)),
      residualWeightMomentum: 1,
      residualWeightKinematics: 1,
      residualWeightMaterial: 1,
      residualWeightBoundary: 1,
      learningRate: mockState.learningRate,
      architecture: [...mockState.architecture],
      progressRatio: completed / maxTotal,
      trainingMode,
      benchmarkId,
      gateStatus: 'running',
      certifiedBestMetric: bestVal,
      dominantBlocker:
        valLoss * 0.34 >= valLoss * 0.3 ? 'val-stress-fit' : 'val-displacement-fit',
      stalledReason: null,
      network: networkSnapshot()
    };
    const tick: TrainingTickEvent = {
      epoch: completed,
      totalEpochs: maxTotal,
      loss,
      valLoss,
      learningRate: mockState.learningRate,
      architecture: [...mockState.architecture],
      progressRatio: completed / maxTotal
    };
    if (completed === 1 || completed % emitEvery === 0 || completed === maxTotal) {
      if (progress.stageId !== lastStage) {
        pushRecentEvent(`e${completed} stage ${lastStage} -> ${progress.stageId}`);
        lastStage = progress.stageId;
      }
      if (progress.optimizerId !== lastOptimizer) {
        pushRecentEvent(`e${completed} optimizer ${lastOptimizer} -> ${progress.optimizerId}`);
        lastOptimizer = progress.optimizerId;
      }
      if (progress.lrPhase !== lastLrPhase) {
        pushRecentEvent(`e${completed} lr-phase ${lastLrPhase} -> ${progress.lrPhase}`);
        lastLrPhase = progress.lrPhase;
      }
      lastMockTick = tick;
      lastMockProgress = progress;
      mockTrainingStatus = {
        ...mockTrainingStatus,
        diagnostics: {
          ...mockTrainingStatus.diagnostics,
          bestValLoss: bestVal,
          epochsSinceImprovement: sinceImprove,
          currentLearningRate: mockState.learningRate,
          lrSchedulePhase: sinceImprove > 30 ? 'pino-decay' : 'pino-steady',
          targetFloorEstimate: bestVal,
          activeStage: progress.stageId,
          activeOptimizer: progress.optimizerId,
          residualWeightMomentum: progress.residualWeightMomentum,
          residualWeightKinematics: progress.residualWeightKinematics,
          residualWeightMaterial: progress.residualWeightMaterial,
          residualWeightBoundary: progress.residualWeightBoundary,
          dataWeight: progress.residualWeightMomentum + progress.residualWeightKinematics,
          physicsWeight: progress.residualWeightMaterial + progress.residualWeightBoundary,
          momentumResidual: progress.momentumResidual,
          kinematicResidual: progress.kinematicResidual,
          materialResidual: progress.materialResidual,
          boundaryResidual: progress.boundaryResidual,
          hybridMode: progress.hybridMode,
          collocationPoints: Math.max(64, batch.collocationPoints ?? 512),
          boundaryPoints: Math.max(16, batch.boundaryPoints ?? 128),
          interfacePoints: Math.max(16, batch.interfacePoints ?? 128),
          collocationSamplesAdded: progress.collocationSamplesAdded,
          trainDataSize: progress.trainDataSize,
          trainDataCap: progress.trainDataCap,
          trainingMode,
          benchmarkId,
          gateStatus: 'running',
          certifiedBestMetric: bestVal,
          reproducibilitySpread: null,
          dominantBlocker: progress.dominantBlocker,
          stalledReason: null,
          runBudgetUsed: completed,
          runBudgetTotal: maxTotal,
          recentEvents
        }
      };
      emitMock('ann-training-tick', tick);
      if (completed % networkEvery === 0 || completed === 1 || completed === maxTotal) {
        emitMock('ann-training-progress', progress);
      }
    }
    await new Promise((r) => setTimeout(r, 10));
  }

  mockState.lastLoss = valLoss;
  mockState.trainSamples += batch.cases.length;

  const result: TrainResult = {
    modelVersion: mockState.modelVersion,
    loss,
    valLoss,
    architecture: [...mockState.architecture],
    learningRate: mockState.learningRate,
    grew,
    pruned,
    completedEpochs: completed,
    reachedTarget: valLoss <= target,
    reachedTargetLoss: valLoss <= target,
    reachedAutonomousConvergence: false,
    stopReason: mockTrainingStatus.stopRequested
      ? 'manual-stop'
      : valLoss <= target
        ? 'target-loss-reached'
        : 'max-epochs-reached',
    notes: ['Web preview mode: local mock trainer active.'],
    trainingMode,
    benchmarkId,
    gateStatus: mockTrainingStatus.stopRequested
      ? 'stopped'
      : valLoss <= target
        ? 'passed'
        : 'failed',
    certifiedBestMetric: bestVal,
    reproducibilitySpread: null,
    dominantBlocker: lastMockProgress.dominantBlocker ?? null,
    stalledReason: null
  };
  mockTrainingStatus = {
    running: false,
    stopRequested: false,
    completed: true,
    lastResult: result,
    diagnostics: {
      ...mockTrainingStatus.diagnostics,
      bestValLoss: Math.min(mockTrainingStatus.diagnostics.bestValLoss, result.valLoss),
      currentLearningRate: result.learningRate,
      lrSchedulePhase: result.stopReason,
      boSelectedArchitecture: [...result.architecture],
      trainingMode,
      benchmarkId,
      gateStatus: result.gateStatus ?? 'failed',
      certifiedBestMetric: result.certifiedBestMetric ?? result.valLoss,
      reproducibilitySpread: null,
      dominantBlocker: result.dominantBlocker ?? null,
      stalledReason: result.stalledReason ?? null,
      runBudgetUsed: completed,
      runBudgetTotal: maxTotal
    }
  };
  emitMock('ann-training-complete', result);
  return true;
};

export const startAnnTraining = async (batch: TrainingBatch) => {
  if (isTauriRuntime()) return invoke<boolean>('start_ann_training', { batch });
  return runMockTrainingLoop(batch);
};

export const stopAnnTraining = async () => {
  if (isTauriRuntime()) return invoke<boolean>('stop_ann_training');
  if (mockTrainingStatus.running) {
    mockTrainingStatus = { ...mockTrainingStatus, stopRequested: true };
    return true;
  }
  return false;
};

export const listTrainingBenchmarks = () => {
  if (isTauriRuntime()) {
    return invoke<TrainingBenchmarkManifest[]>('list_training_benchmarks_command');
  }
  return Promise.resolve(mockTrainingBenchmarks);
};

export const getTrainingStatus = () => {
  if (isTauriRuntime()) return invoke<TrainingRunStatus>('get_training_status');
  return Promise.resolve(mockTrainingStatus);
};

export const trainAnn = async (batch: TrainingBatch) => {
  if (isTauriRuntime()) return invoke<TrainResult>('train_ann', { batch });
  await runMockTrainingLoop(batch);
  if (mockTrainingStatus.lastResult) return mockTrainingStatus.lastResult;
  throw new Error(mockTrainingStatus.lastError ?? 'Training ended without a result');
};

export const inferAnn = async (input: SolveInput) => {
  if (isTauriRuntime()) return invoke<AnnResult>('infer_ann', { input });
  const fem = solveLocal(input);
  const noise = (Math.random() - 0.5) * 0.06;
  fem.stressTensor[0][0] *= 1 + noise;
  fem.vonMisesPsi = Math.abs(fem.stressTensor[0][0]);
  return {
    femLike: fem,
    confidence: Math.max(0.05, 1 / (1 + mockState.lastLoss * 10)),
    uncertainty: Math.min(0.95, mockState.lastLoss * 2),
    modelVersion: mockState.modelVersion,
    usedFemFallback: false,
    fallbackReason: null,
    residualScore: 0,
    uncertaintyThreshold: mockSafeguardSettings.uncertaintyThreshold,
    residualThreshold: mockSafeguardSettings.residualThreshold,
    diagnostics: ['Web preview mode: local mock PINO infer active.']
  };
};

export const runDynamicCase = (input: DynamicInput) => {
  if (isTauriRuntime()) return invoke<DynamicResult>('run_dynamic_case', { input });
  const n = Math.max(2, Math.floor(input.endTimeS / input.timeStepS));
  const t = Array.from({ length: n + 1 }, (_, i) => i * input.timeStepS);
  const drivingLoad =
    input.solveInput.load.axialLoadLbf !== 0
      ? input.solveInput.load.axialLoadLbf
      : input.solveInput.load.verticalPointLoadLbf;
  const amp = (drivingLoad / 10000) * input.pulseScale;
  const y = t.map((tt) => amp * Math.sin((2 * Math.PI * tt) / Math.max(input.endTimeS, 1e-6)) * Math.exp(-input.dampingRatio * 6 * tt));
  return Promise.resolve({
    timeS: t,
    displacementIn: y,
    velocityInS: y.map((v, i) => (i === 0 ? 0 : (v - y[i - 1]) / input.timeStepS)),
    accelerationInS2: y.map((v, i) => (i < 2 ? 0 : (v - 2 * y[i - 1] + y[i - 2]) / (input.timeStepS * input.timeStepS))),
    stable: true,
    diagnostics: ['Web preview mode: local mock dynamic solver active.']
  });
};

export const runThermalCase = (input: ThermalInput) => {
  if (isTauriRuntime()) return invoke<ThermalResult>('run_thermal_case', { input });
  const eps = input.solveInput.material.alphaPerF * input.deltaTF;
  const sigma = input.restrainedX ? input.solveInput.material.ePsi * eps : 0;
  return Promise.resolve({
    thermalStrainX: eps,
    thermalStressPsi: sigma,
    combinedStressTensor: [[sigma, 0, 0], [0, 0, 0], [0, 0, 0]],
    principalStresses: [sigma, 0, 0],
    diagnostics: ['Web preview mode: local mock thermal solver active.']
  });
};

export const evaluateFailure = (input: FailureInput) => {
  if (isTauriRuntime()) return invoke<FailureResult>('evaluate_failure', { input });
  const sx = input.stressTensor[0][0] ?? 0;
  const vm = Math.abs(sx);
  const tresca = Math.abs(sx);
  const sf = input.yieldStrengthPsi / Math.max(vm, 1e-9);
  return Promise.resolve({
    vonMisesPsi: vm,
    trescaPsi: tresca,
    maxPrincipalPsi: sx,
    safetyFactorVm: sf,
    safetyFactorTresca: sf,
    safetyFactorPrincipal: sf,
    failed: sf < 1
  });
};

export const getModelStatus = () => {
  if (isTauriRuntime()) return invoke<ModelStatus>('get_model_status');
  return Promise.resolve({
    modelVersion: mockState.modelVersion,
    architecture: [...mockState.architecture],
    learningRate: mockState.learningRate,
    lastLoss: mockState.lastLoss,
    trainSamples: mockState.trainSamples,
    auditFrequency: mockState.auditFrequency,
    fallbackEnabled: mockState.fallbackEnabled,
    safeguardSettings: { ...mockSafeguardSettings }
  });
};

export const setSafeguardSettings = (settings: SafeguardSettings) => {
  if (isTauriRuntime()) return invoke<ModelStatus>('set_safeguard_settings', { settings });
  mockSafeguardSettings = {
    ...settings,
    uncertaintyThreshold: Math.max(0.01, Math.min(0.99, settings.uncertaintyThreshold)),
    residualThreshold: Math.max(1e-6, Math.min(10, settings.residualThreshold))
  };
  return getModelStatus();
};

export const resetAnnModel = (seed?: number) => {
  if (isTauriRuntime()) return invoke<ModelStatus>('reset_ann_model', { seed });
  mockState.modelVersion = 1;
  mockState.architecture = [19, 16, 16, 9];
  mockState.learningRate = 5e-4;
  mockState.lastLoss = 0.2;
  mockState.trainSamples = 0;
  mockTrainingStatus = {
    running: false,
    stopRequested: false,
    completed: false,
    diagnostics: {
      bestValLoss: Number.MAX_VALUE,
      epochsSinceImprovement: 0,
      lrSchedulePhase: 'idle',
      currentLearningRate: mockState.learningRate,
      dataWeight: 2,
      physicsWeight: 2,
      residualWeightMomentum: 1,
      residualWeightKinematics: 1,
      residualWeightMaterial: 1,
      residualWeightBoundary: 1,
      activeLearningRounds: 0,
      activeLearningSamplesAdded: 0,
      safeguardTriggers: 0,
      curriculumBackoffs: 0,
      optimizerSwitches: 0,
      checkpointRollbacks: 0,
      targetFloorEstimate: 0,
      trendStopReason: 'idle',
      activeStage: 'idle',
      activeOptimizer: 'pino-adam',
      boPresearchUsed: false,
      boSelectedArchitecture: [...mockState.architecture],
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
      collocationPoints: 512,
      boundaryPoints: 128,
      interfacePoints: 128,
      collocationSamplesAdded: 0,
      trainDataSize: 0,
      trainDataCap: 0,
      recentEvents: []
    }
  };
  lastMockTick = {
    epoch: 0,
    totalEpochs: 0,
    loss: 0,
    valLoss: 0,
    learningRate: mockState.learningRate,
    architecture: [...mockState.architecture],
    progressRatio: 0
  };
  lastMockProgress = {
    ...lastMockProgress,
    epoch: 0,
    totalEpochs: 0,
    loss: 0,
    valLoss: 0,
    dataLoss: 0,
    physicsLoss: 0,
    valDataLoss: 0,
    valPhysicsLoss: 0,
    momentumResidual: 0,
    kinematicResidual: 0,
    materialResidual: 0,
    boundaryResidual: 0,
    hybridMode: 'hybrid',
    stageId: 'idle',
    optimizerId: 'pino-adam',
    lrPhase: 'idle',
    targetBandLow: 0,
    targetBandHigh: 0,
    trendSlope: 0,
    trendVariance: 0,
    watchdogTriggerCount: 0,
    collocationSamplesAdded: 0,
    trainDataSize: 0,
    trainDataCap: 0,
    residualWeightMomentum: 1,
    residualWeightKinematics: 1,
    residualWeightMaterial: 1,
    residualWeightBoundary: 1,
    learningRate: mockState.learningRate,
    architecture: [...mockState.architecture],
    progressRatio: 0
  };
  return getModelStatus();
};

export const getTrainingTick = () => {
  if (isTauriRuntime()) return invoke<TrainingTickEvent>('get_training_tick');
  return Promise.resolve(lastMockTick);
};

export const getTrainingProgress = () => {
  if (isTauriRuntime()) return invoke<TrainingProgressEvent>('get_training_progress');
  return Promise.resolve(lastMockProgress);
};

export const saveTrainingCheckpoint = (input: CheckpointSaveInput = {}) => {
  if (isTauriRuntime()) return invoke<TrainingCheckpointInfo>('save_training_checkpoint', { input });
  const now = Date.now();
  const cp: TrainingCheckpointInfo = {
    id: `mock-${now}`,
    tag: input.tag ?? 'manual',
    path: `mock://checkpoint/${now}`,
    createdEpoch: lastMockTick.epoch,
    modelVersion: mockState.modelVersion,
    bestValLoss: mockTrainingStatus.diagnostics.bestValLoss,
    isBest: Boolean(input.markBest),
    createdAtUnixMs: now
  };
  mockCheckpoints.unshift(cp);
  return Promise.resolve(cp);
};

export const listTrainingCheckpoints = () => {
  if (isTauriRuntime()) return invoke<TrainingCheckpointInfo[]>('list_training_checkpoints');
  return Promise.resolve([...mockCheckpoints]);
};

export const resumeTrainingFromCheckpoint = (id: string) => {
  if (isTauriRuntime()) return invoke<ResumeTrainingResult>('resume_training_from_checkpoint', { id });
  const cp = mockCheckpoints.find((c) => c.id === id);
  if (!cp) return Promise.reject(new Error('Checkpoint not found'));
  return Promise.resolve({
    checkpoint: cp,
    modelStatus: {
      modelVersion: mockState.modelVersion,
      architecture: [...mockState.architecture],
      learningRate: mockState.learningRate,
      lastLoss: mockState.lastLoss,
      trainSamples: mockState.trainSamples,
      auditFrequency: mockState.auditFrequency,
      fallbackEnabled: mockState.fallbackEnabled,
      safeguardSettings: { ...mockSafeguardSettings }
    }
  });
};

export const purgeTrainingCheckpoints = (retentionPolicy: CheckpointRetentionPolicy) => {
  if (isTauriRuntime()) return invoke<PurgeCheckpointsResult>('purge_training_checkpoints', { retentionPolicy });
  const keep = Math.max(1, retentionPolicy.keepLast ?? 5);
  const removed = Math.max(0, mockCheckpoints.length - keep);
  mockCheckpoints.splice(keep);
  return Promise.resolve({ removed, kept: mockCheckpoints.length });
};

export const exportReport = (input: ReportInput) => {
  if (isTauriRuntime()) return invoke<ExportResult>('export_report', { input });
  return Promise.resolve({
    path: input.path,
    bytesWritten: JSON.stringify(input).length,
    format: input.format
  });
};
