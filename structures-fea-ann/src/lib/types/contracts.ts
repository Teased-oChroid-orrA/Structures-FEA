export type GeometryInput = {
  lengthIn: number;
  widthIn: number;
  thicknessIn: number;
  holeDiameterIn?: number;
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

export type TrainingBenchmarkManifest = {
  id: string;
  title: string;
  description: string;
  trainingMode: string;
  analysisType: string;
  gateName: string;
  gateTargetLoss: number;
  recommendedLearningRate: number;
  maxRuntimeSeconds: number;
  recommendedEpochs: number;
  active: boolean;
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
  trainingMode?: 'legacy-mixed-exact' | 'benchmark' | 'production-generalized';
  benchmarkId?: string;
  seed?: number;
  analysisType?: 'general' | 'cantilever' | 'plate-hole';
  pinnBackend?:
    | 'pino-ndarray-cpu'
    | 'pino-candle-cpu'
    | 'pino-candle-cuda'
    | 'pino-candle-metal'
    | 'burn-ndarray-cpu'
    | 'burn-wgpu';
  collocationPoints?: number;
  boundaryPoints?: number;
  interfacePoints?: number;
  residualWeightMomentum?: number;
  residualWeightKinematics?: number;
  residualWeightMaterial?: number;
  residualWeightBoundary?: number;
  stage1Epochs?: number;
  stage2Epochs?: number;
  stage3RampEpochs?: number;
  contactPenalty?: number;
  plasticityFactor?: number;
  learningRate?: number;
  autoMode?: boolean;
  maxTotalEpochs?: number;
  minImprovement?: number;
  progressEmitEveryEpochs?: number;
  networkEmitEveryEpochs?: number;
  onlineActiveLearning?: boolean;
  autonomousMode?: boolean;
  maxTopology?: number;
  maxBackoffs?: number;
  maxOptimizerSwitches?: number;
  checkpointEveryEpochs?: number;
  checkpointRetention?: number;
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

export type OperatorGridSpec = {
  nx: number;
  ny: number;
  nz: number;
  inputChannels: number;
  outputChannels: number;
};

export type HoldoutValidationSummary = {
  trusted: boolean;
  trainingSeedCases: number;
  holdoutCases: number;
  meanDisplacementError: number;
  meanVonMisesError: number;
  p95FieldError: number;
  residualRatio: number;
  acceptedWithoutFallback: boolean;
  meanErrorLimit: number;
  p95ErrorLimit: number;
  residualRatioLimit: number;
  displacementPass: boolean;
  vonMisesPass: boolean;
  p95Pass: boolean;
  residualRatioPass: boolean;
};

export type SurrogateDomainSummary = {
  featureLabels: string[];
  featureMins: number[];
  featureMaxs: number[];
  coverageTags: string[];
  trainingSeedCases: number;
  expandedCases: number;
  mixedLoadCases: number;
  holeCases: number;
  dualFixedCases: number;
};

export type PinoRuntimeMetadata = {
  engineId: string;
  backend: string;
  spectralModes: number;
  operatorGrid: OperatorGridSpec;
  domainDim: number;
  physicsModel: string;
  spectralModes3d: [number, number, number];
  operatorGrid3d?: OperatorGridSpec | null;
  boundaryMode?: string | null;
  objectiveMode?: string | null;
  localRefinement?: {
    enabled: boolean;
    strategy: string;
    maxPatches: number;
    maxPatchCells: number;
  } | null;
  localEnrichment?: {
    enabled: boolean;
    strategy: string;
  } | null;
  calibrationStressScale?: number | null;
  calibrationDisplacementScale?: number | null;
  holdoutValidation?: HoldoutValidationSummary | null;
};

export type TrainingProgressEvent = {
  epoch: number;
  totalEpochs: number;
  loss: number;
  valLoss: number;
  dataLoss: number;
  physicsLoss: number;
  valDataLoss: number;
  valPhysicsLoss: number;
  momentumResidual: number;
  kinematicResidual: number;
  materialResidual: number;
  boundaryResidual: number;
  displacementFit: number;
  stressFit: number;
  invariantResidual: number;
  constitutiveNormalResidual: number;
  constitutiveShearResidual: number;
  valDisplacementFit: number;
  valStressFit: number;
  valInvariantResidual: number;
  valConstitutiveNormalResidual: number;
  valConstitutiveShearResidual: number;
  hybridMode: string;
  stageId: string;
  optimizerId: string;
  lrPhase: string;
  targetBandLow: number;
  targetBandHigh: number;
  trendSlope: number;
  trendVariance: number;
  watchdogTriggerCount: number;
  collocationSamplesAdded: number;
  trainDataSize: number;
  trainDataCap: number;
  residualWeightMomentum: number;
  residualWeightKinematics: number;
  residualWeightMaterial: number;
  residualWeightBoundary: number;
  learningRate: number;
  architecture: number[];
  progressRatio: number;
  trainingMode?: string;
  benchmarkId?: string | null;
  gateStatus?: string;
  certifiedBestMetric?: number;
  dominantBlocker?: string | null;
  stalledReason?: string | null;
  network: NetworkSnapshot;
  pino?: PinoRuntimeMetadata | null;
};

export type BenchmarkCertification = {
  status: string;
  summary: string;
  suggestedTargetLoss: number;
  tipDisplacementRelativeError?: number | null;
  maxDisplacementRelativeError?: number | null;
  meanVonMisesRelativeError?: number | null;
  maxSigmaXxRelativeError?: number | null;
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
  diagnostics: TrainingDiagnostics;
};

export type TrainingDiagnostics = {
  bestValLoss: number;
  epochsSinceImprovement: number;
  lrSchedulePhase: string;
  currentLearningRate: number;
  dataWeight: number;
  physicsWeight: number;
  residualWeightMomentum: number;
  residualWeightKinematics: number;
  residualWeightMaterial: number;
  residualWeightBoundary: number;
  activeLearningRounds: number;
  activeLearningSamplesAdded: number;
  safeguardTriggers: number;
  curriculumBackoffs: number;
  optimizerSwitches: number;
  checkpointRollbacks: number;
  targetFloorEstimate: number;
  trendStopReason: string;
  activeStage: string;
  activeOptimizer: string;
  boPresearchUsed: boolean;
  boSelectedArchitecture: number[];
  momentumResidual: number;
  kinematicResidual: number;
  materialResidual: number;
  boundaryResidual: number;
  displacementFit: number;
  stressFit: number;
  invariantResidual: number;
  constitutiveNormalResidual: number;
  constitutiveShearResidual: number;
  valDisplacementFit: number;
  valStressFit: number;
  valInvariantResidual: number;
  valConstitutiveNormalResidual: number;
  valConstitutiveShearResidual: number;
  hybridMode: string;
  collocationPoints: number;
  boundaryPoints: number;
  interfacePoints: number;
  collocationSamplesAdded: number;
  trainDataSize: number;
  trainDataCap: number;
  trainingMode?: string;
  benchmarkId?: string | null;
  gateStatus?: string;
  certifiedBestMetric?: number;
  reproducibilitySpread?: number | null;
  dominantBlocker?: string | null;
  stalledReason?: string | null;
  benchmarkCertification?: BenchmarkCertification | null;
  runBudgetUsed?: number;
  runBudgetTotal?: number;
  recentEvents: string[];
  pino?: PinoRuntimeMetadata | null;
};

export type TrainingCheckpointInfo = {
  id: string;
  tag: string;
  path: string;
  createdEpoch: number;
  modelVersion: number;
  bestValLoss: number;
  isBest: boolean;
  createdAtUnixMs: number;
};

export type CheckpointSaveInput = {
  tag?: string;
  markBest?: boolean;
};

export type CheckpointRetentionPolicy = {
  keepLast: number;
  keepBest: number;
};

export type ResumeTrainingResult = {
  checkpoint: TrainingCheckpointInfo;
  modelStatus: ModelStatus;
};

export type PurgeCheckpointsResult = {
  removed: number;
  kept: number;
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
  reachedTargetLoss?: boolean;
  reachedAutonomousConvergence?: boolean;
  stopReason: string;
  notes: string[];
  trainingMode?: string | null;
  benchmarkId?: string | null;
  gateStatus?: string | null;
  certifiedBestMetric?: number | null;
  reproducibilitySpread?: number | null;
  dominantBlocker?: string | null;
  stalledReason?: string | null;
  benchmarkCertification?: BenchmarkCertification | null;
  pino?: PinoRuntimeMetadata | null;
};

export type AnnResult = {
  femLike: FemResult;
  confidence: number;
  uncertainty: number;
  modelVersion: number;
  usedFemFallback: boolean;
  fallbackReason?: string | null;
  domainExtrapolationScore?: number;
  residualScore?: number;
  uncertaintyThreshold?: number;
  residualThreshold?: number;
  diagnostics: string[];
  surrogateDomain?: SurrogateDomainSummary | null;
  pino?: PinoRuntimeMetadata | null;
};

export type ModelStatus = {
  modelVersion: number;
  architecture: number[];
  learningRate: number;
  lastLoss: number;
  trainSamples: number;
  auditFrequency: number;
  fallbackEnabled: boolean;
  safeguardSettings: SafeguardSettings;
  surrogateDomain?: SurrogateDomainSummary | null;
  pino?: PinoRuntimeMetadata | null;
};

export type SafeguardSettings = {
  preset: string;
  uncertaintyThreshold: number;
  residualThreshold: number;
  adaptiveByGeometry: boolean;
};

export type RuntimeFingerprint = {
  appVersion: string;
  buildProfile: string;
  targetOs: string;
  targetArch: string;
  debugBuild: boolean;
  gitCommit: string;
  buildTimeUtc: string;
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

export const MAX_DENSE_SOLVER_DOFS = 3200;

export const defaultSolveInput: SolveInput = {
  geometry: {
    lengthIn: 11.811,
    widthIn: 4.724,
    thicknessIn: 0.25,
    holeDiameterIn: 2.362
  },
  mesh: {
    nx: 28,
    ny: 14,
    nz: 1,
    elementType: 'hex8',
    autoAdapt: true,
    maxDofs: MAX_DENSE_SOLVER_DOFS,
    amrEnabled: true,
    amrPasses: 3,
    amrMaxNx: 40,
    amrRefineRatio: 1.15
  },
  material: {
    ePsi: 29_000_000,
    nu: 0.3,
    rhoLbIn3: 0.283,
    alphaPerF: 6.5e-6,
    yieldStrengthPsi: 36_000
  },
  boundaryConditions: {
    fixStartFace: true,
    fixEndFace: false
  },
  load: {
    axialLoadLbf: 1712,
    verticalPointLoadLbf: 0
  },
  unitSystem: 'inch-lbf-second',
  deltaTF: 0
};
