export type GeometryInput = {
  lengthIn: number;
  widthIn: number;
  thicknessIn: number;
};

export type MeshControls = {
  nx: number;
  ny: number;
  nz: number;
  elementType: string;
  autoAdapt: boolean;
  maxDofs: number;
  amrEnabled: boolean;
  amrPasses: number;
  amrMaxNx: number;
  amrRefineRatio: number;
};

export type BoundaryConditionInput = {
  fixStartFace: boolean;
  fixEndFace: boolean;
};

export type LoadInput = {
  axialLoadLbf: number;
  verticalPointLoadLbf: number;
};

export type Material = {
  ePsi: number;
  nu: number;
  rhoLbIn3: number;
  alphaPerF: number;
  yieldStrengthPsi: number;
};

export type SolveInput = {
  geometry: GeometryInput;
  mesh: MeshControls;
  material: Material;
  boundaryConditions: BoundaryConditionInput;
  load: LoadInput;
  unitSystem: string;
  deltaTF?: number;
};

export type NodalDisplacement = {
  nodeId: number;
  xIn: number;
  yIn: number;
  zIn: number;
  uxIn: number;
  uyIn: number;
  uzIn: number;
  dispMagIn: number;
  vmPsi: number;
};

export type FemResult = {
  nodalDisplacements: NodalDisplacement[];
  strainTensor: number[][];
  stressTensor: number[][];
  principalStresses: number[];
  vonMisesPsi: number;
  trescaPsi: number;
  maxPrincipalPsi: number;
  stiffnessMatrix: number[][];
  massMatrix: number[][];
  dampingMatrix: number[][];
  forceVector: number[];
  displacementVector: number[];
  beamStations: BeamStationResult[];
  diagnostics: string[];
};

export type BeamStationResult = {
  xIn: number;
  shearLbf: number;
  momentLbIn: number;
  sigmaTopPsi: number;
  sigmaBottomPsi: number;
  deflectionIn: number;
};

export type ThermalInput = {
  solveInput: SolveInput;
  deltaTF: number;
  restrainedX: boolean;
};

export type ThermalResult = {
  thermalStrainX: number;
  thermalStressPsi: number;
  combinedStressTensor: number[][];
  principalStresses: number[];
  diagnostics: string[];
};

export type DynamicInput = {
  solveInput: SolveInput;
  timeStepS: number;
  endTimeS: number;
  dampingRatio: number;
  pulseDurationS: number;
  pulseScale: number;
};

export type DynamicResult = {
  timeS: number[];
  displacementIn: number[];
  velocityInS: number[];
  accelerationInS2: number[];
  stable: boolean;
  diagnostics: string[];
};

export type FailureInput = {
  stressTensor: number[][];
  yieldStrengthPsi: number;
};

export type FailureResult = {
  vonMisesPsi: number;
  trescaPsi: number;
  maxPrincipalPsi: number;
  safetyFactorVm: number;
  safetyFactorTresca: number;
  safetyFactorPrincipal: number;
  failed: boolean;
};

export type TrainingBatch = {
  cases: SolveInput[];
  epochs: number;
  targetLoss: number;
  learningRate?: number;
  autoMode?: boolean;
  maxTotalEpochs?: number;
  minImprovement?: number;
};

export type NetworkNodeSnapshot = {
  id: string;
  layer: number;
  index: number;
  activation: number;
  bias: number;
  importance: number;
};

export type NetworkConnectionSnapshot = {
  fromId: string;
  toId: string;
  weight: number;
  magnitude: number;
};

export type NetworkSnapshot = {
  layerSizes: number[];
  nodes: NetworkNodeSnapshot[];
  connections: NetworkConnectionSnapshot[];
};

export type TrainingProgressEvent = {
  epoch: number;
  totalEpochs: number;
  loss: number;
  valLoss: number;
  learningRate: number;
  architecture: number[];
  progressRatio: number;
  network: NetworkSnapshot;
};

export type TrainingTickEvent = {
  epoch: number;
  totalEpochs: number;
  loss: number;
  valLoss: number;
  learningRate: number;
  architecture: number[];
  progressRatio: number;
};

export type TrainingRunStatus = {
  running: boolean;
  stopRequested: boolean;
  completed: boolean;
  lastResult?: TrainResult;
  lastError?: string;
};

export type TrainResult = {
  modelVersion: number;
  loss: number;
  valLoss: number;
  architecture: number[];
  learningRate: number;
  grew: boolean;
  pruned: boolean;
  completedEpochs: number;
  reachedTarget: boolean;
  stopReason: string;
  notes: string[];
};

export type AnnResult = {
  femLike: FemResult;
  confidence: number;
  uncertainty: number;
  modelVersion: number;
  usedFemFallback: boolean;
  diagnostics: string[];
};

export type ModelStatus = {
  modelVersion: number;
  architecture: number[];
  learningRate: number;
  lastLoss: number;
  trainSamples: number;
  auditFrequency: number;
  fallbackEnabled: boolean;
};

export type ReportInput = {
  path: string;
  format: 'json' | 'csv' | 'pdf';
  solveInput: SolveInput;
  femResult?: FemResult;
  annResult?: AnnResult;
  dynamicResult?: DynamicResult;
  thermalResult?: ThermalResult;
  failureResult?: FailureResult;
};

export type ExportResult = {
  path: string;
  bytesWritten: number;
  format: string;
};

export const defaultSolveInput: SolveInput = {
  geometry: {
    lengthIn: 10,
    widthIn: 4,
    thicknessIn: 0.125
  },
  mesh: {
    nx: 10,
    ny: 4,
    nz: 1,
    elementType: 'hex8',
    autoAdapt: true,
    maxDofs: 12000,
    amrEnabled: true,
    amrPasses: 2,
    amrMaxNx: 28,
    amrRefineRatio: 1.2
  },
  material: {
    ePsi: 10_000_000,
    nu: 0.33,
    rhoLbIn3: 0.0975,
    alphaPerF: 13e-6,
    yieldStrengthPsi: 40_000
  },
  boundaryConditions: {
    fixStartFace: true,
    fixEndFace: false
  },
  load: {
    axialLoadLbf: 0,
    verticalPointLoadLbf: -1000
  },
  unitSystem: 'inch-lbf-second',
  deltaTF: 0
};
