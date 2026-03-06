import { invoke } from '@tauri-apps/api/core';
import type {
  AnnResult,
  DynamicInput,
  DynamicResult,
  ExportResult,
  FailureInput,
  FailureResult,
  FemResult,
  ModelStatus,
  ReportInput,
  SolveInput,
  ThermalInput,
  ThermalResult,
  TrainResult,
  TrainingBatch,
  TrainingProgressEvent
} from '$lib/types/contracts';

const isTauriRuntime = () =>
  typeof window !== 'undefined' && typeof (window as any).__TAURI_INTERNALS__ !== 'undefined';

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
  architecture: [8, 12, 12, 6],
  learningRate: 5e-4,
  lastLoss: 0.2,
  trainSamples: 0,
  auditFrequency: 5,
  fallbackEnabled: true
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
  const P = input.load.verticalPointLoadLbf;
  const E = input.material.ePsi;
  const I = (t * Math.pow(W, 3)) / 12;
  const c = W / 2;
  const n = Math.max(8, input.mesh.nx);
  return Array.from({ length: n + 1 }, (_, i) => {
    const x = (L * i) / n;
    const M = P * (L - x);
    const sigmaTop = (M * c) / I;
    const defl = (P * x * x * (3 * L - x)) / (6 * E * I);
    return {
      xIn: x,
      shearLbf: P,
      momentLbIn: M,
      sigmaTopPsi: sigmaTop,
      sigmaBottomPsi: -sigmaTop,
      deflectionIn: defl
    };
  });
};

const solveLocal = (input: SolveInput): FemResult => {
  const L = input.geometry.lengthIn;
  const W = input.geometry.widthIn;
  const t = input.geometry.thicknessIn;
  const P = input.load.verticalPointLoadLbf;
  const E = input.material.ePsi;
  const nu = input.material.nu;
  const I = (t * Math.pow(W, 3)) / 12;
  const c = W / 2;
  const M0 = P * L;
  const sigma = (M0 * c) / I;
  const eps = sigma / E;
  const tip = (P * Math.pow(L, 3)) / (3 * E * I);
  const vm = Math.abs(sigma);
  const k = (3 * E * I) / Math.pow(L, 3);

  return {
    nodalDisplacements: [
      { nodeId: 0, xIn: 0, yIn: 0, zIn: 0, uxIn: 0, uyIn: 0, uzIn: 0, dispMagIn: 0, vmPsi: vm },
      { nodeId: 1, xIn: L, yIn: W, zIn: t / 2, uxIn: 0, uyIn: tip, uzIn: 0, dispMagIn: Math.abs(tip), vmPsi: vm }
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
    forceVector: [0, P],
    displacementVector: [0, tip],
    beamStations: fmtStations(input),
    diagnostics: ['Web preview mode: local mock solver active.']
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

export const trainAnn = async (batch: TrainingBatch) => {
  if (isTauriRuntime()) return invoke<TrainResult>('train_ann', { batch });

  const epochs = Math.max(1, batch.epochs);
  const maxTotal = Math.max(epochs, batch.maxTotalEpochs ?? epochs * 20);
  const target = Math.max(0, batch.targetLoss);
  if (batch.learningRate) mockState.learningRate = batch.learningRate;

  let loss = Math.max(mockState.lastLoss, target * 3 + 0.02);
  let valLoss = loss * 1.02;
  let completed = 0;
  let grew = false;
  let pruned = false;

  while (completed < maxTotal && valLoss > target) {
    completed += 1;
    loss = Math.max(target * 0.5, loss * (0.965 + Math.random() * 0.02));
    valLoss = Math.max(target * 0.6, loss * (0.99 + Math.random() * 0.02));

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
      learningRate: mockState.learningRate,
      architecture: [...mockState.architecture],
      progressRatio: completed / maxTotal,
      network: networkSnapshot()
    };
    emitMock('ann-training-progress', progress);
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
    stopReason: valLoss <= target ? 'target-loss-reached' : 'max-epochs-reached',
    notes: ['Web preview mode: local mock trainer active.']
  };
  emitMock('ann-training-complete', result);
  return result;
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
    diagnostics: ['Web preview mode: local mock ANN infer active.']
  };
};

export const runDynamicCase = (input: DynamicInput) => {
  if (isTauriRuntime()) return invoke<DynamicResult>('run_dynamic_case', { input });
  const n = Math.max(2, Math.floor(input.endTimeS / input.timeStepS));
  const t = Array.from({ length: n + 1 }, (_, i) => i * input.timeStepS);
  const amp = (input.solveInput.load.verticalPointLoadLbf / 10000) * input.pulseScale;
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
    fallbackEnabled: mockState.fallbackEnabled
  });
};

export const exportReport = (input: ReportInput) => {
  if (isTauriRuntime()) return invoke<ExportResult>('export_report', { input });
  return Promise.resolve({
    path: input.path,
    bytesWritten: JSON.stringify(input).length,
    format: input.format
  });
};
