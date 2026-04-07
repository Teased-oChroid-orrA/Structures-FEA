<script lang="ts">
  import NumberFlow from '@number-flow/svelte';
  import type { TrainingProgressEvent, TrainingTickEvent } from '$lib/types/contracts';

  type Props = {
    trainingActive: boolean;
    tick: TrainingTickEvent | null;
    progress: TrainingProgressEvent | null;
    history: TrainingProgressEvent[];
    targetLoss: number;
  };

  let { trainingActive, tick, progress, history, targetLoss }: Props = $props();

  let renderedNodes = $state<
    Array<{
      id: string;
      x: number;
      y: number;
      size: number;
      label: string;
      connected: boolean;
      selected: boolean;
      activation: number;
      layer: number;
      impactRatio: number;
      impactTier: 'high' | 'mid' | 'low';
    }>
  >([]);
  let overlayConnections = $state<
    Array<{
      id: string;
      d: string;
      labelX: number;
      labelY: number;
      magnitude: number;
      positive: boolean;
      connected: boolean;
      labeled: boolean;
      strengthBand: 'strong' | 'mid' | 'light';
    }>
  >([]);
  let selectedNodeId = $state<string | null>(null);
  let hoveredNodeId = $state<string | null>(null);
  const CANVAS_WIDTH = 900;
  const CANVAS_HEIGHT = 430;
  const CANVAS_X_PAD = 72;
  const CANVAS_Y_PAD = 52;

  function canonicalNodeId(rawId: string, fallbackLayer?: number, fallbackIndex?: number) {
    const normalized = rawId.trim();
    const canonicalMatch = normalized.match(/^L(\d+)N(\d+)$/);
    if (canonicalMatch) {
      return normalized;
    }

    const legacyMatch = normalized.match(/^l(\d+)_n(\d+)$/);
    if (legacyMatch) {
      return `L${legacyMatch[1]}N${legacyMatch[2]}`;
    }

    if (fallbackLayer != null && fallbackIndex != null) {
      return `L${fallbackLayer}N${fallbackIndex}`;
    }

    return normalized;
  }

  function extractEpochHintFromPhase(phase: string | null | undefined): number {
    if (!phase) return 0;
    const match = phase.match(/epoch-(\d+)-bootstrap/);
    return match ? Number.parseInt(match[1] ?? '0', 10) || 0 : 0;
  }

  const network = $derived.by(() => {
    if (!progress?.network || progress.network.layerSizes.length === 0) {
      return null;
    }

    return {
      ...progress.network,
      nodes: progress.network.nodes.map((node) => ({
        ...node,
        id: canonicalNodeId(node.id, node.layer, node.index)
      })),
      connections: progress.network.connections.map((connection) => ({
        ...connection,
        fromId: canonicalNodeId(connection.fromId),
        toId: canonicalNodeId(connection.toId)
      }))
    };
  });
  const currentEpoch = $derived.by(() =>
    Math.max(tick?.epoch ?? 0, progress?.epoch ?? 0, extractEpochHintFromPhase(progress?.lrPhase))
  );
  const totalEpochs = $derived(tick?.totalEpochs ?? progress?.totalEpochs ?? 0);
  const learningRate = $derived(tick?.learningRate ?? progress?.learningRate ?? 0);
  const progressPercent = $derived.by(() =>
    Math.max(
      trainingActive && currentEpoch === 0 ? 2 : 0,
      Math.min(100, ((tick?.progressRatio ?? progress?.progressRatio) ?? 0) * 100)
    )
  );
  const currentLoss = $derived(tick?.loss ?? progress?.loss ?? 0);
  const validationLoss = $derived.by(() => {
    const candidates = [progress?.valLoss, progress?.loss, tick?.valLoss, tick?.loss];
    for (const value of candidates) {
      if (Number.isFinite(value ?? NaN) && (value as number) > 0) return value as number;
    }
    return 0;
  });
  const latestObservedLoss = $derived.by(() => {
    const candidates = [progress?.valLoss, progress?.loss, tick?.valLoss, tick?.loss];
    for (const value of candidates) {
      if (Number.isFinite(value ?? NaN)) return value as number;
    }
    return null;
  });
  const targetLossReached = $derived.by(() =>
    latestObservedLoss !== null && latestObservedLoss > 0 ? latestObservedLoss <= targetLoss : false
  );
  const architectureLabel = $derived.by(() => (tick?.architecture ?? progress?.architecture)?.join(' → ') ?? '-');
  const stageLabel = $derived.by(() => (trainingActive ? progress?.stageId ?? 'stage-1' : 'ready'));
  const optimizerLabel = $derived.by(() => (trainingActive ? progress?.optimizerId ?? 'pino-adam' : 'standby'));
  const lrPhaseLabel = $derived.by(() => (trainingActive ? progress?.lrPhase ?? 'pino-steady' : 'idle'));
  const historyDepth = $derived(history.length);
  const activeNodeCount = $derived(network?.nodes.length ?? 0);
  const activeEdgeCount = $derived(network?.connections.length ?? 0);
  const activeFocusNodeId = $derived(selectedNodeId ?? hoveredNodeId);
  const nodeInfluenceMap = $derived.by(() => {
    if (!network) return new Map<string, number>();
    const totals = new Map<string, number>();
    for (const node of network.nodes) {
      totals.set(node.id, Math.abs(node.importance ?? 0));
    }
    for (const edge of network.connections) {
      totals.set(edge.fromId, (totals.get(edge.fromId) ?? 0) + edge.magnitude);
      totals.set(edge.toId, (totals.get(edge.toId) ?? 0) + edge.magnitude);
    }
    return totals;
  });
  const activeTrace = $derived.by(() => {
    if (!network || !activeFocusNodeId) {
      return {
        nodeIds: new Set<string>(),
        edgeIds: new Set<string>(),
        labeledEdgeIds: new Set<string>()
      };
    }

    const nodeIds = new Set<string>([activeFocusNodeId]);
    const edgeIds = new Set<string>();
    const labeledEdgeIds = new Set<string>();
    const directEdges = network.connections
      .filter((edge) => edge.fromId === activeFocusNodeId || edge.toId === activeFocusNodeId)
      .sort((a, b) => b.magnitude - a.magnitude);

    for (const edge of directEdges) {
      const edgeId = `${edge.fromId}->${edge.toId}`;
      edgeIds.add(edgeId);
      nodeIds.add(edge.fromId);
      nodeIds.add(edge.toId);
    }

    for (const edge of directEdges.slice(0, 4)) {
      labeledEdgeIds.add(`${edge.fromId}->${edge.toId}`);
    }

    return { nodeIds, edgeIds, labeledEdgeIds };
  });
  const strongestConnections = $derived.by(() => {
    if (!network) return [];
    return [...network.connections]
      .sort((a, b) => b.magnitude - a.magnitude)
      .slice(0, 5)
      .map((edge) => ({
        id: `${edge.fromId}->${edge.toId}`,
        from: edge.fromId,
        to: edge.toId,
        magnitude: edge.magnitude,
        signedWeight: edge.weight
      }));
  });
  const neuronImpactLeaders = $derived.by(() => {
    if (!network) return [];
    const incomingByNode = new Map<string, number>();
    const outgoingByNode = new Map<string, number>();
    for (const edge of network.connections) {
      incomingByNode.set(edge.toId, (incomingByNode.get(edge.toId) ?? 0) + edge.magnitude);
      outgoingByNode.set(edge.fromId, (outgoingByNode.get(edge.fromId) ?? 0) + edge.magnitude);
    }
    return [...network.nodes]
      .map((node) => {
        const incoming = incomingByNode.get(node.id) ?? 0;
        const outgoing = outgoingByNode.get(node.id) ?? 0;
        const totalInfluence = incoming + outgoing + Math.abs(node.importance ?? 0);
        return {
          id: node.id,
          label: node.id,
          activation: Math.abs(node.activation ?? 0),
          importance: Math.abs(node.importance ?? 0),
          incoming,
          outgoing,
          totalInfluence
        };
      })
      .sort((a, b) => b.totalInfluence - a.totalInfluence)
      .slice(0, 6);
  });
  const connectionDensityByLayer = $derived.by(() => {
    if (!network) return [];
    const density: Array<{ label: string; count: number; weight: number }> = [];
    for (let layer = 0; layer < network.layerSizes.length - 1; layer++) {
      const fromPrefix = `L${layer}`;
      const toPrefix = `L${layer + 1}`;
      const matching = network.connections.filter(
        (edge) => edge.fromId.startsWith(fromPrefix) && edge.toId.startsWith(toPrefix)
      );
      density.push({
        label: `${layer} → ${layer + 1}`,
        count: matching.length,
        weight: matching.reduce((sum, edge) => sum + edge.magnitude, 0)
      });
    }
    return density;
  });
  const selectedNeuronDetail = $derived.by(() => {
    const activeNodeId = selectedNodeId ?? hoveredNodeId;
    if (!network || !activeNodeId) return null;
    const node = network.nodes.find((item) => item.id === activeNodeId);
    if (!node) return null;
    const incoming = network.connections
      .filter((edge) => edge.toId === activeNodeId)
      .sort((a, b) => b.magnitude - a.magnitude);
    const outgoing = network.connections
      .filter((edge) => edge.fromId === activeNodeId)
      .sort((a, b) => b.magnitude - a.magnitude);
    const rank = neuronImpactLeaders.findIndex((item) => item.id === activeNodeId);
    const totalInfluence =
      incoming.reduce((sum, edge) => sum + edge.magnitude, 0) +
      outgoing.reduce((sum, edge) => sum + edge.magnitude, 0) +
      Math.abs(node.importance ?? 0);
    const maxInfluence = Math.max(1e-9, ...neuronImpactLeaders.map((item) => item.totalInfluence));
    return {
      id: activeNodeId,
      previewOnly: !selectedNodeId && !!hoveredNodeId,
      rank: rank >= 0 ? rank + 1 : null,
      totalInfluence,
      influenceRatio: totalInfluence / maxInfluence,
      activation: Math.abs(node.activation ?? 0),
      importance: Math.abs(node.importance ?? 0),
      incoming,
      outgoing
    };
  });
  const signalStrength = $derived.by(() => {
    const loss = Math.max(validationLoss, currentLoss, 1e-12);
    const score = Math.max(0, Math.min(1, 1 - Math.log10(loss + 1e-12) / 8));
    return score;
  });
  const heroTitle = $derived.by(() => (trainingActive ? 'Neural Field In Motion' : 'Inference-Ready Model Map'));
  const heroBlurb = $derived.by(() => {
    if (trainingActive) {
      return 'Live topology, loss balance, and optimizer phase are updating in one place, and convergence is only claimed when the selected target loss is actually crossed.';
    }
    if (network) {
      return 'The latest learned topology stays visible between runs so the model still feels inspectable after convergence.';
    }
    return 'Train the PINO surrogate to light up the live network map, optimization timeline, and residual balance trends.';
  });
  const statusTone = $derived.by(() => {
    if (!trainingActive) return targetLossReached ? 'target' : 'standby';
    if (targetLossReached) return 'target';
    if ((tick?.epoch ?? progress?.epoch ?? 0) < 10) return 'booting';
    const recent = history.slice(-4);
    if (recent.length >= 2) {
      const first = recent[0]?.valLoss ?? recent[0]?.loss ?? 0;
      const last = recent[recent.length - 1]?.valLoss ?? recent[recent.length - 1]?.loss ?? 0;
      if (Number.isFinite(first) && Number.isFinite(last)) {
        if (last <= targetLoss) return 'target';
        if (last < first * 0.995) return 'improving';
        if (Math.abs(last - first) / Math.max(first, 1e-9) < 0.002) return 'plateau';
      }
    }
    return 'exploring';
  });
  const progressPhaseLabel = $derived.by(() => {
    if (!trainingActive) return targetLossReached ? 'target achieved' : 'standby';
    if ((tick?.epoch ?? progress?.epoch ?? 0) === 0 && historyDepth === 0) return 'awaiting first signal';
    if (targetLossReached) return 'at target';
    const recent = history.slice(-4);
    if (recent.length >= 2) {
      const first = recent[0]?.valLoss ?? recent[0]?.loss ?? 0;
      const last = recent[recent.length - 1]?.valLoss ?? recent[recent.length - 1]?.loss ?? 0;
      if (Number.isFinite(first) && Number.isFinite(last)) {
        if (last < first * 0.995) return 'improving';
        if (Math.abs(last - first) / Math.max(first, 1e-9) < 0.002) return 'plateau watch';
      }
    }
    return 'warming up';
  });

  function focusNode(nodeId: string | null) {
    selectedNodeId = nodeId && selectedNodeId === nodeId ? null : nodeId;
    hoveredNodeId = null;
  }

  function previewNode(nodeId: string | null) {
    if (selectedNodeId) return;
    hoveredNodeId = nodeId;
  }

  function clearPreview() {
    if (selectedNodeId) return;
    hoveredNodeId = null;
  }

  function clearSelection() {
    selectedNodeId = null;
    hoveredNodeId = null;
  }

  $effect(() => {
    if (!network) {
      renderedNodes = [];
      overlayConnections = [];
      return;
    }

    const builtRenderedNodes: typeof renderedNodes = [];
    const builtOverlayConnections: typeof overlayConnections = [];
    const nodeCenters = new Map<string, { x: number; y: number; size: number }>();
    const layerCount = Math.max(1, network.layerSizes.length);
    const layerGap =
      layerCount > 1 ? (CANVAS_WIDTH - CANVAS_X_PAD * 2) / (layerCount - 1) : 0;
    const usableHeight = CANVAS_HEIGHT - CANVAS_Y_PAD * 2;
    const maxInfluence = Math.max(1e-9, ...nodeInfluenceMap.values());

    const activeNodeId = selectedNodeId ?? hoveredNodeId;

    for (let layer = 0; layer < network.layerSizes.length; layer++) {
      const count = network.layerSizes[layer] ?? 1;
      const slotHeight = usableHeight / Math.max(count, 1);
      for (let idx = 0; idx < count; idx++) {
        const id = `L${layer}N${idx}`;
        const n = network.nodes.find((it) => it.id === id);
        const importance = Math.max(0, Math.min(1, (n?.importance ?? 0) / 4));
        const activation = Math.abs(n?.activation ?? 0);
        const connected =
          !activeNodeId || activeTrace.nodeIds.has(id);
        const selected = id === activeNodeId;
        const impactRatio = (nodeInfluenceMap.get(id) ?? 0) / maxInfluence;
        const impactTier =
          impactRatio >= 0.78 ? 'high' : impactRatio >= 0.45 ? 'mid' : 'low';
        const maxVisualSize = Math.max(18, Math.min(54, slotHeight - 10));
        const size = Math.min(maxVisualSize, 24 + importance * 14 + impactRatio * 12);
        const x = CANVAS_X_PAD + layer * layerGap;
        const y = CANVAS_Y_PAD + slotHeight * (idx + 0.5);
        nodeCenters.set(id, { x, y, size });
        builtRenderedNodes.push({
          id,
          x,
          y,
          size,
          label:
            layer === 0
              ? `in ${idx + 1}`
              : layer === network.layerSizes.length - 1
                ? `out ${idx + 1}`
                : `h${layer}.${idx + 1}`,
          connected,
          selected,
          activation,
          layer,
          impactRatio,
          impactTier
        });
      }
    }

    for (const c of network.connections) {
      const positive = c.weight >= 0;
      const connected =
        !!activeNodeId &&
        activeTrace.edgeIds.has(`${c.fromId}->${c.toId}`);
      const sourceCenter = nodeCenters.get(c.fromId);
      const targetCenter = nodeCenters.get(c.toId);
      if (sourceCenter && targetCenter) {
        const dx = targetCenter.x - sourceCenter.x;
        const controlOffset = Math.max(56, Math.abs(dx) * 0.42);
        const controlX1 = sourceCenter.x + controlOffset;
        const controlX2 = targetCenter.x - controlOffset;
        builtOverlayConnections.push({
          id: `${c.fromId}->${c.toId}`,
          d: `M ${sourceCenter.x} ${sourceCenter.y} C ${controlX1} ${sourceCenter.y}, ${controlX2} ${targetCenter.y}, ${targetCenter.x} ${targetCenter.y}`,
          labelX: (sourceCenter.x + targetCenter.x) / 2,
          labelY: (sourceCenter.y + targetCenter.y) / 2,
          magnitude: c.magnitude,
          positive,
          connected,
          labeled: activeTrace.labeledEdgeIds.has(`${c.fromId}->${c.toId}`),
          strengthBand: c.magnitude >= 0.75 ? 'strong' : c.magnitude >= 0.42 ? 'mid' : 'light'
        });
      }
    }

    renderedNodes = builtRenderedNodes;
    overlayConnections = builtOverlayConnections;
  });

  const sparkline = $derived.by(() => {
    const points = history.slice(-40);
    if (points.length === 0) return '';
    const w = 360;
    const h = 88;
    const maxLoss = Math.max(1e-9, ...points.map((p) => Math.max(p.loss, p.valLoss)));
    const step = points.length > 1 ? w / (points.length - 1) : w;
    return points
      .map((p, i) => {
        const x = i * step;
        const y = h - (p.valLoss / maxLoss) * h;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  });

  const decomposition = $derived.by(() => {
    const points = history.slice(-80);
    if (points.length === 0) {
      return { dataLine: '', physicsLine: '' };
    }
    const w = 360;
    const h = 88;
    const maxY = Math.max(
      1e-9,
      ...points.map((p) =>
        Math.max(
          p.dataLoss ?? 0,
          p.physicsLoss ?? 0,
          p.valDataLoss ?? 0,
          p.valPhysicsLoss ?? 0
        )
      )
    );
    const step = points.length > 1 ? w / (points.length - 1) : w;
    const line = (pick: (p: TrainingProgressEvent) => number) =>
      points
        .map((p, i) => `${(i * step).toFixed(1)},${(h - (pick(p) / maxY) * h).toFixed(1)}`)
        .join(' ');
    return {
      dataLine: line((p) => p.valDataLoss ?? 0),
      physicsLine: line((p) => p.valPhysicsLoss ?? 0)
    };
  });

  const regimeEvents = $derived.by(() => {
    const points = history.slice(-80);
    if (points.length < 2) return [];
    const events: Array<{ x: number; kind: 'stage' | 'optimizer' | 'watchdog'; label: string }> = [];
    const w = 360;
    const step = points.length > 1 ? w / (points.length - 1) : w;
    for (let i = 1; i < points.length; i++) {
      const prev = points[i - 1];
      const curr = points[i];
      if ((curr.stageId ?? '') !== (prev.stageId ?? '')) {
        events.push({ x: i * step, kind: 'stage', label: curr.stageId ?? 'stage' });
      }
      if ((curr.optimizerId ?? '') !== (prev.optimizerId ?? '')) {
        events.push({ x: i * step, kind: 'optimizer', label: curr.optimizerId ?? 'opt' });
      }
      if ((curr.watchdogTriggerCount ?? 0) > (prev.watchdogTriggerCount ?? 0)) {
        events.push({ x: i * step, kind: 'watchdog', label: 'rollback' });
      }
    }
    return events.slice(-12);
  });

  const residualPillars = $derived.by(() => {
    const points = history.slice(-80);
    if (points.length === 0) {
      return { momentum: '', kinematic: '', material: '', boundary: '' };
    }
    const w = 360;
    const h = 88;
    const maxY = Math.max(
      1e-9,
      ...points.map((p) =>
        Math.max(
          p.momentumResidual ?? 0,
          p.kinematicResidual ?? 0,
          p.materialResidual ?? 0,
          p.boundaryResidual ?? 0
        )
      )
    );
    const step = points.length > 1 ? w / (points.length - 1) : w;
    const mk = (val: (p: TrainingProgressEvent) => number) =>
      points
        .map((p, i) => `${(i * step).toFixed(1)},${(h - (val(p) / maxY) * h).toFixed(1)}`)
        .join(' ');
    return {
      momentum: mk((p) => p.momentumResidual ?? 0),
      kinematic: mk((p) => p.kinematicResidual ?? 0),
      material: mk((p) => p.materialResidual ?? 0),
      boundary: mk((p) => p.boundaryResidual ?? 0)
    };
  });

  const residualSubterms = $derived.by(() => {
    const latest = progress ?? history[history.length - 1] ?? null;
    if (!latest) {
      return {
        focus: 'idle',
        items: [] as Array<{ label: string; value: number; tone: string }>
      };
    }

    const items = [
      {
        key: 'disp',
        label: 'disp fit',
        value: latest.valDisplacementFit ?? latest.displacementFit ?? 0,
        tone: 'momentum'
      },
      {
        key: 'stress',
        label: 'stress fit',
        value: latest.valStressFit ?? latest.stressFit ?? 0,
        tone: 'kinematics'
      },
      {
        key: 'cn',
        label: 'constitutive normal',
        value: latest.valConstitutiveNormalResidual ?? latest.constitutiveNormalResidual ?? 0,
        tone: 'material'
      },
      {
        key: 'cs',
        label: 'constitutive shear',
        value: latest.valConstitutiveShearResidual ?? latest.constitutiveShearResidual ?? 0,
        tone: 'material'
      },
      {
        key: 'inv',
        label: 'invariant',
        value: latest.valInvariantResidual ?? latest.invariantResidual ?? 0,
        tone: 'boundary'
      }
    ];
    const dominant = items.reduce((best, item) => (item.value > best.value ? item : best), items[0]);
    const focus =
      dominant.key === 'disp'
        ? 'kinematic-fit'
        : dominant.key === 'stress'
          ? 'stress-fit'
          : dominant.key === 'inv'
            ? 'invariant-check'
            : 'local-material';
    return { focus, items };
  });
</script>

<section class="network-panel">
  <div class="hero-shell">
    <div class="hero-copy">
      <div class="eyebrow-row">
        <span class={`signal-pill ${trainingActive ? 'live' : 'idle'}`}>{trainingActive ? 'live training' : 'standby'}</span>
        <span class="signal-pill ghost">{statusTone}</span>
        <span class="signal-pill ghost">{stageLabel}</span>
        <span class="signal-pill ghost">{progressPhaseLabel}</span>
      </div>
      <h2>{heroTitle}</h2>
      <p>{heroBlurb}</p>

      <div class="focus-grid">
        <article class="focus-card">
          <div class="label">Validation Loss</div>
          <div class="value"><NumberFlow value={validationLoss} format={{ maximumSignificantDigits: 4 }} /></div>
          <div class="meta">primary quality signal</div>
        </article>
        <article class="focus-card">
          <div class="label">Epoch</div>
          <div class="value">
            <NumberFlow value={currentEpoch} format={{ maximumFractionDigits: 0 }} />
            <span class="subtle"> / <NumberFlow value={totalEpochs} format={{ maximumFractionDigits: 0 }} /></span>
          </div>
          <div class="meta">current run position</div>
        </article>
        <article class="focus-card">
          <div class="label">Optimizer</div>
          <div class="value">{optimizerLabel}</div>
          <div class="meta">{lrPhaseLabel}</div>
        </article>
        <article class="focus-card">
          <div class="label">Target Loss</div>
          <div class="value">{targetLoss.toExponential(0)}</div>
          <div class="meta">{targetLossReached ? 'target crossed' : 'live threshold'}</div>
        </article>
      </div>

      <div class="hero-progress">
        <div class="hero-progress-copy">
          <span>model energy</span>
          <span><NumberFlow value={progressPercent} format={{ maximumFractionDigits: 1 }} />%</span>
        </div>
        <div class="hero-track">
          <div class="hero-fill" style={`width:${progressPercent}%; --signal:${signalStrength};`}></div>
        </div>
      </div>

      <div class="hero-microgrid">
        <article class="micro-card">
          <span>frames</span>
          <strong><NumberFlow value={historyDepth} format={{ maximumFractionDigits: 0 }} /></strong>
        </article>
        <article class="micro-card">
          <span>visible nodes</span>
          <strong><NumberFlow value={activeNodeCount} format={{ maximumFractionDigits: 0 }} /></strong>
        </article>
        <article class="micro-card">
          <span>visible edges</span>
          <strong><NumberFlow value={activeEdgeCount} format={{ maximumFractionDigits: 0 }} /></strong>
        </article>
      </div>
    </div>

    <div class="hero-radar">
      <div class="radar-ring ring-a"></div>
      <div class="radar-ring ring-b"></div>
      <div class="radar-core" style={`--signal:${signalStrength};`}>
        <span>{trainingActive ? 'pulse' : 'ready'}</span>
      </div>
      <div class="radar-stat">
        <div class="label">Architecture</div>
        <div class="value">{architectureLabel}</div>
      </div>
      <div class="radar-stat">
        <div class="label">Current Loss</div>
        <div class="value"><NumberFlow value={currentLoss} format={{ maximumSignificantDigits: 4 }} /></div>
      </div>
      <div class="radar-stat">
        <div class="label">Learning Rate</div>
        <div class="value"><NumberFlow value={learningRate} format={{ maximumSignificantDigits: 4 }} /></div>
      </div>
    </div>
  </div>

  <div class="flow-shell">
    <div class="flow-header">
      <div>
        <h3>Topology Theater</h3>
        <p>Weight sign, node importance, and stage shifts stay visible instead of disappearing into logs.</p>
      </div>
      <div class="legend-row">
        <span class="legend-pill positive">positive weights</span>
        <span class="legend-pill negative">negative weights</span>
        <span class="legend-pill neutral">importance glow</span>
        <span class="legend-pill influence">impact ranking below</span>
      </div>
    </div>

    <div class="flow-canvas">
      <div class="flow-atmosphere"></div>
      <div class="flow-spotlight" style={`--signal:${signalStrength};`}></div>
      <div class="flow-grid-glow"></div>
      <svg class="topology-svg" viewBox="0 0 900 430" preserveAspectRatio="none" aria-hidden="true">
        {#each overlayConnections as link (link.id)}
          <path
            d={link.d}
            class:active-path={link.connected}
            class:inactive-path={!link.connected}
            class:positive-path={link.positive}
            class:negative-path={!link.positive}
            class:strong-path={link.strengthBand === 'strong'}
            class:mid-path={link.strengthBand === 'mid'}
            class:light-path={link.strengthBand === 'light'}
            style={`--path-width:${link.connected ? Math.max(1.8, 1.2 + link.magnitude * 2.2) : Math.max(0.4, 0.18 + link.magnitude * 0.75)};`}
          />
          {#if activeFocusNodeId && link.connected && link.labeled}
            <g transform={`translate(${link.labelX}, ${link.labelY})`}>
              <rect
                x="-20"
                y="-10"
                width="40"
                height="20"
                rx="10"
                class:positive-label={link.positive}
                class:negative-label={!link.positive}
              />
              <text y="4" text-anchor="middle" class="path-label-text">{link.magnitude.toFixed(2)}</text>
            </g>
          {/if}
        {/each}
      </svg>
      <div class="node-layer">
        {#each renderedNodes as node (node.id)}
          <button
            class="topology-node"
            class:connected={node.connected}
            class:selected={node.selected}
            class:previewed={!node.selected && activeFocusNodeId === node.id}
            class:impact-high={node.impactTier === 'high'}
            class:impact-mid={node.impactTier === 'mid'}
            type="button"
            style={`left:${(node.x / 900) * 100}%; top:${(node.y / 430) * 100}%; width:${node.size}px; height:${node.size}px; --node-layer:${node.layer}; --node-activation:${Math.min(1, node.activation)}; --impact:${node.impactRatio};`}
            onmouseenter={() => previewNode(node.id)}
            onmouseleave={clearPreview}
            onclick={() => focusNode(node.id)}
            aria-label={`Neuron ${node.id}`}
          >
            <span>{node.label}</span>
          </button>
        {/each}
      </div>
      <div class="flow-hud">
        <div class="hud-card primary">
          <span>regime</span>
          <strong>{trainingActive ? progressPhaseLabel : 'ready'}</strong>
          <small>{trainingActive ? `${optimizerLabel} · target ${targetLoss.toExponential(0)}` : 'awaiting next run'}</small>
        </div>
        <div class="hud-stack">
          <div class="hud-card">
            <span>val loss</span>
            <strong><NumberFlow value={validationLoss} format={{ maximumSignificantDigits: 4 }} /></strong>
          </div>
          <div class="hud-card">
            <span>flow map</span>
            <strong>{activeNodeCount}N / {activeEdgeCount}E</strong>
          </div>
        </div>
      </div>
      {#if network}
        <div class="canvas-callout {selectedNeuronDetail ? 'active' : ''}">
          {#if selectedNeuronDetail}
            <strong>{selectedNeuronDetail.id}</strong>
            <span>
              tracing {selectedNeuronDetail.incoming.length} inbound and {selectedNeuronDetail.outgoing.length}
              outbound links
            </span>
          {:else}
            <strong>Click Any Neuron</strong>
            <span>Reveal its incoming and outgoing links directly on the graph.</span>
          {/if}
        </div>
      {/if}
      {#if !network}
        <div class="flow-overlay-hint">
          <strong>{trainingActive ? 'Listening for live topology...' : 'No live topology yet'}</strong>
          <span>{trainingActive ? 'The panel will bloom as soon as the next network snapshot arrives.' : 'Run Train PINO to animate the model map and trend wall.'}</span>
        </div>
      {/if}
    </div>
    <div class="selection-strip">
      {#if selectedNeuronDetail}
        <div class="selection-card">
          <div class="selection-header">
            <div>
              <h4>{selectedNeuronDetail.id}</h4>
              <p>
                {#if selectedNeuronDetail.rank}
                  impact rank #{selectedNeuronDetail.rank}
                {:else}
                  custom selection
                {/if}
                {#if selectedNeuronDetail.previewOnly}
                  · hover preview
                {/if}
              </p>
            </div>
            {#if !selectedNeuronDetail.previewOnly}
              <button class="selection-clear" type="button" onclick={clearSelection}>Clear</button>
            {/if}
          </div>
          <div class="selection-grid">
            <article class="selection-stat">
              <span>Total Influence</span>
              <strong><NumberFlow value={selectedNeuronDetail.totalInfluence} format={{ maximumFractionDigits: 2 }} /></strong>
            </article>
            <article class="selection-stat">
              <span>Relative Impact</span>
              <strong><NumberFlow value={selectedNeuronDetail.influenceRatio * 100} format={{ maximumFractionDigits: 1 }} />%</strong>
            </article>
            <article class="selection-stat">
              <span>Activation</span>
              <strong><NumberFlow value={selectedNeuronDetail.activation} format={{ maximumFractionDigits: 3 }} /></strong>
            </article>
            <article class="selection-stat">
              <span>Importance</span>
              <strong><NumberFlow value={selectedNeuronDetail.importance} format={{ maximumFractionDigits: 3 }} /></strong>
            </article>
          </div>
          <div class="selection-columns">
            <div class="selection-column">
              <h5>Inbound Connections</h5>
              {#if selectedNeuronDetail.incoming.length}
                {#each selectedNeuronDetail.incoming.slice(0, 6) as edge (`in-${edge.fromId}-${edge.toId}`)}
                  <div class="selection-link">
                    <span>{edge.fromId}</span>
                    <strong><NumberFlow value={edge.magnitude} format={{ maximumFractionDigits: 3 }} /></strong>
                  </div>
                {/each}
              {:else}
                <p class="empty-copy">No inbound links in the current snapshot.</p>
              {/if}
            </div>
            <div class="selection-column">
              <h5>Outbound Connections</h5>
              {#if selectedNeuronDetail.outgoing.length}
                {#each selectedNeuronDetail.outgoing.slice(0, 6) as edge (`out-${edge.fromId}-${edge.toId}`)}
                  <div class="selection-link">
                    <span>{edge.toId}</span>
                    <strong><NumberFlow value={edge.magnitude} format={{ maximumFractionDigits: 3 }} /></strong>
                  </div>
                {/each}
              {:else}
                <p class="empty-copy">No outbound links in the current snapshot.</p>
              {/if}
            </div>
          </div>
          <div class="selection-route">
            <div class="route-column">
              <span class="route-label">Inbound path</span>
              <div class="route-chips">
                {#if selectedNeuronDetail.incoming.length}
                  {#each selectedNeuronDetail.incoming.slice(0, 6) as edge (`route-in-${edge.fromId}-${edge.toId}`)}
                    <span class="route-chip">{edge.fromId}</span>
                  {/each}
                {:else}
                  <span class="route-chip muted">none</span>
                {/if}
              </div>
            </div>
            <div class="route-focus">
              <span>{selectedNeuronDetail.id}</span>
            </div>
            <div class="route-column">
              <span class="route-label">Outbound path</span>
              <div class="route-chips">
                {#if selectedNeuronDetail.outgoing.length}
                  {#each selectedNeuronDetail.outgoing.slice(0, 6) as edge (`route-out-${edge.fromId}-${edge.toId}`)}
                    <span class="route-chip">{edge.toId}</span>
                  {/each}
                {:else}
                  <span class="route-chip muted">none</span>
                {/if}
              </div>
            </div>
          </div>
        </div>
      {:else}
        <div class="selection-card idle">
          <h4>Inspect Any Neuron</h4>
          <p>Click a node in the flow map to isolate its inbound and outbound links and compare its influence against the rest of the network.</p>
        </div>
      {/if}
    </div>
  </div>

  <div class="telemetry-grid">
    <article class="telemetry-card">
      <div class="chart-topline">
        <h3>Validation Arc</h3>
        <span>does the model keep improving?</span>
      </div>
      <svg viewBox="0 0 360 88" class="trend-svg">
        {#if sparkline}
          <polyline points={sparkline} fill="none" stroke="rgba(251, 191, 36, 0.98)" stroke-width="2.6" />
        {/if}
      </svg>
    </article>

    <article class="telemetry-card">
      <div class="chart-topline">
        <h3>Loss Balance</h3>
        <span>data vs physics on validation</span>
      </div>
      <div class="legend-row">
        <span class="legend-pill data">data</span>
        <span class="legend-pill physics">physics</span>
      </div>
      <svg viewBox="0 0 360 88" class="trend-svg">
        {#if decomposition.dataLine}
          <polyline points={decomposition.dataLine} fill="none" stroke="rgba(96, 165, 250, 0.98)" stroke-width="2.3" />
          <polyline points={decomposition.physicsLine} fill="none" stroke="rgba(248, 113, 113, 0.98)" stroke-width="2.3" />
        {/if}
      </svg>
    </article>

    <article class="telemetry-card wide">
      <div class="chart-topline">
        <h3>Training Storyline</h3>
        <span>stage changes, optimizer swaps, and watchdog events on one rail</span>
      </div>
      <svg viewBox="0 0 360 52" class="timeline-svg">
        <line x1="0" y1="26" x2="360" y2="26" stroke="rgba(148, 163, 184, 0.34)" stroke-width="1.2" />
        {#each regimeEvents as ev (`${ev.kind}-${ev.label}-${ev.x}`)}
          <line
            x1={ev.x}
            y1="10"
            x2={ev.x}
            y2="42"
            stroke={ev.kind === 'stage' ? 'rgba(96,165,250,0.95)' : ev.kind === 'optimizer' ? 'rgba(251,191,36,0.95)' : 'rgba(248,113,113,0.95)'}
            stroke-width="2"
          />
        {/each}
      </svg>
      <div class="legend-row">
        <span class="legend-pill stage">stage change</span>
        <span class="legend-pill optimizer">optimizer switch</span>
        <span class="legend-pill watchdog">watchdog rollback</span>
      </div>
    </article>

    <article class="telemetry-card wide">
      <div class="chart-topline">
        <h3>Neuron Impact Ladder</h3>
        <span>incoming weight + outgoing weight + node importance</span>
      </div>
      <div class="impact-list">
        {#if neuronImpactLeaders.length}
          {#each neuronImpactLeaders as neuron (neuron.id)}
            <button
              class="impact-row impact-button"
              class:selected={activeFocusNodeId === neuron.id}
              type="button"
              aria-pressed={activeFocusNodeId === neuron.id}
              onclick={() => {
                selectedNodeId = neuron.id;
                hoveredNodeId = null;
              }}
            >
              <div class="impact-id">
                <strong>{neuron.label}</strong>
                <span>act {neuron.activation.toFixed(3)}</span>
              </div>
              <div class="impact-bars">
                <div class="impact-track">
                  <div class="impact-fill" style={`width:${Math.max(8, neuron.totalInfluence * 100)}%`}></div>
                </div>
                <div class="impact-meta">
                  <span>in <NumberFlow value={neuron.incoming} format={{ maximumFractionDigits: 2 }} /></span>
                  <span>out <NumberFlow value={neuron.outgoing} format={{ maximumFractionDigits: 2 }} /></span>
                  <span>importance <NumberFlow value={neuron.importance} format={{ maximumFractionDigits: 2 }} /></span>
                </div>
              </div>
            </button>
          {/each}
        {:else}
          <p class="empty-copy">Neuron impact ranking appears when a live network snapshot is available.</p>
        {/if}
      </div>
    </article>

    <article class="telemetry-card wide">
      <div class="chart-topline">
        <h3>Connection Influence Map</h3>
        <span>the strongest links and the busiest layer bridges</span>
      </div>
      <div class="connection-layout">
        <div class="connection-column">
          <h4>Strongest Connections</h4>
          {#if strongestConnections.length}
            {#each strongestConnections as edge (edge.id)}
              <div class="connection-row">
                <div>
                  <strong>{edge.from}</strong>
                  <span>{edge.to}</span>
                </div>
                <div class="connection-metrics">
                  <span class:positive-weight={edge.signedWeight >= 0} class:negative-weight={edge.signedWeight < 0}>
                    {edge.signedWeight >= 0 ? 'positive' : 'negative'}
                  </span>
                  <strong><NumberFlow value={edge.magnitude} format={{ maximumFractionDigits: 3 }} /></strong>
                </div>
              </div>
            {/each}
          {:else}
            <p class="empty-copy">Connection rankings appear when the graph has streamed in.</p>
          {/if}
        </div>
        <div class="connection-column">
          <h4>Layer Bridge Density</h4>
          {#if connectionDensityByLayer.length}
            {#each connectionDensityByLayer as bridge (bridge.label)}
              <div class="bridge-row">
                <div class="bridge-head">
                  <strong>{bridge.label}</strong>
                  <span>{bridge.count} links</span>
                </div>
                <div class="impact-track compact">
                  <div class="impact-fill amber" style={`width:${Math.max(10, bridge.weight * 100)}%`}></div>
                </div>
                <span class="bridge-weight">total influence <NumberFlow value={bridge.weight} format={{ maximumFractionDigits: 2 }} /></span>
              </div>
            {/each}
          {/if}
        </div>
      </div>
    </article>

    <article class="telemetry-card wide">
      <div class="chart-topline">
        <h3>Residual Pillars</h3>
        <span>where the physics pressure is building</span>
      </div>
      <div class="legend-row">
        <span class="legend-pill momentum">momentum</span>
        <span class="legend-pill kinematics">kinematics</span>
        <span class="legend-pill material">material</span>
        <span class="legend-pill boundary">boundary</span>
      </div>
      <svg viewBox="0 0 360 88" class="trend-svg" data-testid="residual-pillars-trend">
        {#if residualPillars.momentum}
          <polyline points={residualPillars.momentum} fill="none" stroke="rgba(56, 189, 248, 0.96)" stroke-width="2" />
          <polyline points={residualPillars.kinematic} fill="none" stroke="rgba(250, 204, 21, 0.96)" stroke-width="2" />
          <polyline points={residualPillars.material} fill="none" stroke="rgba(74, 222, 128, 0.96)" stroke-width="2" />
          <polyline points={residualPillars.boundary} fill="none" stroke="rgba(248, 113, 113, 0.96)" stroke-width="2" />
        {/if}
      </svg>
      <div class="residual-subterms">
        <div class="focus-chip">dominant focus: {residualSubterms.focus}</div>
        <div class="subterm-grid">
          {#each residualSubterms.items as item (item.label)}
            <div class="subterm-card">
              <span>{item.label}</span>
              <strong class={`tone-${item.tone}`}>
                <NumberFlow value={item.value} format={{ maximumSignificantDigits: 4 }} />
              </strong>
            </div>
          {/each}
        </div>
      </div>
    </article>
  </div>
</section>

<style>
  .network-panel {
    display: grid;
    gap: 1rem;
    padding: 1rem;
    border-radius: 28px;
    border: 1px solid rgba(148, 163, 184, 0.14);
    background:
      radial-gradient(circle at top left, rgba(59, 130, 246, 0.16), transparent 32%),
      radial-gradient(circle at top right, rgba(20, 184, 166, 0.14), transparent 28%),
      linear-gradient(180deg, rgba(7, 14, 27, 0.98), rgba(10, 19, 33, 0.98));
    box-shadow: 0 30px 80px rgba(2, 6, 23, 0.42);
  }

  .hero-shell {
    display: grid;
    grid-template-columns: minmax(0, 1.6fr) minmax(280px, 0.9fr);
    gap: 1rem;
  }

  .hero-copy,
  .hero-radar,
  .flow-shell,
  .telemetry-card {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    border: 1px solid rgba(148, 163, 184, 0.14);
    background: linear-gradient(180deg, rgba(13, 22, 38, 0.92), rgba(9, 16, 28, 0.98));
  }

  .hero-copy {
    padding: 1.35rem;
  }

  .hero-copy h2,
  .flow-shell h3,
  .telemetry-card h3 {
    margin: 0;
    font-family: 'Avenir Next', 'Segoe UI', sans-serif;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #f8fafc;
  }

  .hero-copy h2 {
    font-size: clamp(1.8rem, 2vw, 2.45rem);
    line-height: 0.96;
  }

  .hero-copy p,
  .flow-shell p,
  .telemetry-card span,
  .telemetry-card .chart-topline span {
    margin: 0;
    color: rgba(203, 213, 225, 0.82);
  }

  .eyebrow-row,
  .legend-row {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    align-items: center;
  }

  .signal-pill,
  .legend-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.42rem 0.7rem;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #dbeafe;
    background: rgba(15, 23, 42, 0.66);
  }

  .signal-pill.live {
    color: #ddfff4;
    background: rgba(13, 61, 49, 0.75);
    border-color: rgba(45, 212, 191, 0.34);
    box-shadow: 0 0 20px rgba(45, 212, 191, 0.18);
  }

  .signal-pill.ghost {
    color: rgba(226, 232, 240, 0.88);
  }

  .residual-subterms {
    display: grid;
    gap: 0.75rem;
    margin-top: 0.9rem;
  }

  .focus-chip {
    display: inline-flex;
    align-items: center;
    width: fit-content;
    padding: 0.4rem 0.7rem;
    border-radius: 999px;
    border: 1px solid rgba(96, 165, 250, 0.24);
    background: rgba(15, 23, 42, 0.78);
    color: rgba(191, 219, 254, 0.94);
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .subterm-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.65rem;
  }

  .subterm-card {
    display: grid;
    gap: 0.3rem;
    padding: 0.8rem 0.9rem;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.14);
    background: rgba(8, 15, 27, 0.72);
  }

  .subterm-card strong {
    font-size: 1.05rem;
    line-height: 1;
  }

  .tone-momentum {
    color: rgba(125, 211, 252, 0.95);
  }

  .tone-kinematics {
    color: rgba(253, 224, 71, 0.95);
  }

  .tone-material {
    color: rgba(134, 239, 172, 0.95);
  }

  .tone-boundary {
    color: rgba(252, 165, 165, 0.95);
  }

  .focus-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
  }

  .focus-card {
    padding: 0.95rem;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(17, 26, 45, 0.92), rgba(10, 17, 29, 0.98));
    border: 1px solid rgba(148, 163, 184, 0.12);
  }

  .label {
    color: rgba(148, 163, 184, 0.92);
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .value {
    margin-top: 0.28rem;
    color: #f8fafc;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.04em;
  }

  .meta,
  .subtle {
    color: rgba(148, 163, 184, 0.82);
    font-size: 0.8rem;
  }

  .hero-progress {
    margin-top: 1rem;
  }

  .hero-microgrid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.65rem;
    margin-top: 0.9rem;
  }

  .micro-card {
    padding: 0.72rem 0.8rem;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    background: rgba(8, 15, 27, 0.56);
  }

  .micro-card span {
    display: block;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(148, 163, 184, 0.88);
  }

  .micro-card strong {
    display: block;
    margin-top: 0.22rem;
    font-size: 1.05rem;
    color: #f8fafc;
    letter-spacing: -0.03em;
  }

  .hero-progress-copy {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    color: rgba(226, 232, 240, 0.9);
    font-size: 0.84rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .hero-track {
    margin-top: 0.45rem;
    height: 13px;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid rgba(148, 163, 184, 0.12);
    overflow: hidden;
  }

  .hero-fill {
    height: 100%;
    min-width: 4%;
    border-radius: inherit;
    background:
      linear-gradient(90deg, rgba(59, 130, 246, 0.96), rgba(45, 212, 191, calc(0.55 + var(--signal) * 0.35)), rgba(251, 191, 36, 0.95));
    box-shadow: 0 0 28px rgba(45, 212, 191, calc(0.2 + var(--signal) * 0.22));
  }

  .hero-radar {
    min-height: 280px;
    display: grid;
    place-items: center;
    padding: 1.2rem;
    background:
      radial-gradient(circle at center, rgba(20, 184, 166, 0.12), transparent 38%),
      linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(5, 10, 18, 0.98));
  }

  .radar-ring,
  .radar-core {
    position: absolute;
    border-radius: 999px;
  }

  .radar-ring {
    border: 1px solid rgba(96, 165, 250, 0.16);
  }

  .ring-a {
    width: 220px;
    height: 220px;
  }

  .ring-b {
    width: 152px;
    height: 152px;
    border-color: rgba(45, 212, 191, 0.2);
  }

  .radar-core {
    width: 108px;
    height: 108px;
    display: grid;
    place-items: center;
    color: #f8fafc;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background:
      radial-gradient(circle at 32% 30%, rgba(255,255,255,0.26), transparent 40%),
      linear-gradient(160deg, rgba(96,165,250, calc(0.35 + var(--signal) * 0.22)), rgba(13,148,136, calc(0.46 + var(--signal) * 0.22)));
    box-shadow:
      0 0 45px rgba(45, 212, 191, calc(0.16 + var(--signal) * 0.26)),
      inset 0 0 18px rgba(255,255,255,0.12);
  }

  .radar-stat {
    position: relative;
    z-index: 1;
    width: min(260px, 100%);
    margin-top: auto;
    padding: 0.75rem 0.95rem;
    border-radius: 16px;
    background: rgba(8, 15, 27, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.12);
  }

  .flow-shell {
    padding: 1rem;
  }

  .flow-header,
  .chart-topline {
    display: flex;
    justify-content: space-between;
    gap: 0.9rem;
    align-items: center;
    flex-wrap: wrap;
  }

  .flow-canvas {
    position: relative;
    height: 430px;
    margin-top: 0.85rem;
    border-radius: 22px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.12);
    background: linear-gradient(180deg, rgba(5, 10, 19, 0.96), rgba(12, 19, 33, 0.98));
  }

  .flow-atmosphere {
    position: absolute;
    inset: 0;
    background:
      radial-gradient(circle at 15% 15%, rgba(96, 165, 250, 0.12), transparent 20%),
      radial-gradient(circle at 82% 24%, rgba(45, 212, 191, 0.11), transparent 18%),
      radial-gradient(circle at 48% 82%, rgba(251, 191, 36, 0.09), transparent 24%);
    pointer-events: none;
    z-index: 0;
  }

  .flow-spotlight {
    position: absolute;
    inset: 8% 14%;
    border-radius: 28px;
    background:
      radial-gradient(circle at center, rgba(96, 165, 250, calc(0.09 + var(--signal) * 0.12)), transparent 38%),
      radial-gradient(circle at center, rgba(45, 212, 191, calc(0.08 + var(--signal) * 0.12)), transparent 54%);
    filter: blur(16px);
    pointer-events: none;
    z-index: 0;
    animation: spotlight-drift 9s ease-in-out infinite;
  }

  .flow-grid-glow {
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(rgba(148, 163, 184, 0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(148, 163, 184, 0.04) 1px, transparent 1px);
    background-size: 32px 32px;
    mask-image: radial-gradient(circle at center, black 35%, transparent 88%);
    pointer-events: none;
    z-index: 0;
  }

  .topology-svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
  }

  .topology-svg path {
    fill: none;
    stroke-width: var(--path-width);
    stroke-linecap: round;
    stroke-linejoin: round;
    transition:
      opacity 180ms ease,
      stroke-width 180ms ease,
      filter 180ms ease;
  }

  .positive-path {
    stroke: rgba(94, 234, 212, 0.34);
    filter: drop-shadow(0 0 4px rgba(45, 212, 191, 0.12));
  }

  .negative-path {
    stroke: rgba(251, 146, 170, 0.3);
    filter: drop-shadow(0 0 4px rgba(251, 113, 133, 0.1));
  }

  .active-path.positive-path {
    stroke: rgba(125, 255, 232, 0.94);
    filter: drop-shadow(0 0 9px rgba(45, 212, 191, 0.22));
  }

  .active-path.negative-path {
    stroke: rgba(251, 132, 151, 0.92);
    filter: drop-shadow(0 0 9px rgba(251, 113, 133, 0.2));
  }

  .inactive-path {
    opacity: 0.22;
  }

  .active-path {
    opacity: 1;
  }

  .strong-path.inactive-path {
    opacity: 0.32;
  }

  .mid-path.inactive-path {
    opacity: 0.2;
  }

  .light-path.inactive-path {
    opacity: 0.12;
  }

  .positive-label {
    fill: rgba(6, 20, 18, 0.92);
    stroke: rgba(45, 212, 191, 0.34);
  }

  .negative-label {
    fill: rgba(29, 8, 14, 0.92);
    stroke: rgba(251, 113, 133, 0.34);
  }

  .path-label-text {
    fill: #f8fafc;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.04em;
  }

  .node-layer {
    position: absolute;
    inset: 0;
    z-index: 2;
  }

  .topology-node {
    position: absolute;
    transform: translate(-50%, -50%);
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    color: rgba(248, 250, 252, 0.45);
    background:
      radial-gradient(circle at 30% 30%, rgba(255,255,255,0.18), transparent 55%),
      linear-gradient(160deg, rgba(10, 17, 30, 0.96), rgba(24, 37, 58, 0.98));
    box-shadow:
      0 0 0 1px rgba(255,255,255,0.03) inset,
      0 18px 30px rgba(3,7,18,0.28),
      0 0 calc(12px + var(--impact) * 18px) rgba(96, 165, 250, calc(0.08 + var(--impact) * 0.16));
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    transition:
      transform 140ms ease,
      opacity 140ms ease,
      box-shadow 180ms ease,
      border-color 180ms ease,
      color 180ms ease;
  }

  .topology-node span {
    pointer-events: none;
  }

  .topology-node.connected {
    color: #f8fafc;
    opacity: 1;
  }

  .topology-node:not(.connected) {
    opacity: 0.22;
  }

  .topology-node.previewed,
  .topology-node.selected {
    transform: translate(-50%, -50%) scale(1.08);
    color: #fff8db;
    border-color: rgba(251, 191, 36, 0.9);
    box-shadow:
      0 0 0 1px rgba(255,255,255,0.04) inset,
      0 18px 28px rgba(3,7,18,0.34),
      0 0 18px rgba(251, 191, 36, 0.26);
  }

  .topology-node.selected {
    background:
      radial-gradient(circle at 30% 30%, rgba(255,255,255,0.26), transparent 55%),
      linear-gradient(160deg, rgba(51, 30, 8, 0.96), rgba(36, 33, 56, 0.98));
  }

  .topology-node.impact-high {
    border-color: rgba(251, 191, 36, 0.34);
    box-shadow:
      0 0 0 1px rgba(255,255,255,0.03) inset,
      0 18px 30px rgba(3,7,18,0.28),
      0 0 16px rgba(251, 191, 36, 0.12),
      0 0 calc(12px + var(--impact) * 16px) rgba(96, 165, 250, calc(0.08 + var(--impact) * 0.16));
  }

  .topology-node.impact-mid {
    border-color: rgba(125, 211, 252, 0.24);
  }

  .flow-hud {
    position: absolute;
    inset: 1rem 1rem auto 1rem;
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    align-items: flex-start;
    pointer-events: none;
    z-index: 4;
  }

  .hud-stack {
    display: grid;
    gap: 0.5rem;
  }

  .hud-card {
    min-width: 128px;
    padding: 0.78rem 0.85rem;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.14);
    background: rgba(4, 10, 20, 0.7);
    backdrop-filter: blur(12px);
    box-shadow: 0 18px 30px rgba(2, 6, 23, 0.24);
  }

  .hud-card.primary {
    min-width: 180px;
    background:
      linear-gradient(160deg, rgba(8, 15, 27, 0.8), rgba(17, 24, 39, 0.68)),
      radial-gradient(circle at top left, rgba(96, 165, 250, 0.16), transparent 42%);
  }

  .hud-card span,
  .hud-card small {
    display: block;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.68rem;
    color: rgba(191, 219, 254, 0.82);
  }

  .hud-card strong {
    display: block;
    margin-top: 0.2rem;
    color: #f8fafc;
    font-size: 1rem;
    letter-spacing: -0.03em;
  }

  .flow-overlay-hint {
    position: absolute;
    inset: auto 1rem 1rem 1rem;
    display: grid;
    gap: 0.25rem;
    padding: 0.95rem 1rem;
    border-radius: 16px;
    background: rgba(6, 12, 23, 0.82);
    border: 1px solid rgba(148, 163, 184, 0.18);
    color: #e2e8f0;
    backdrop-filter: blur(10px);
    z-index: 4;
  }

  .canvas-callout {
    position: absolute;
    inset: auto auto 1rem 1rem;
    display: grid;
    gap: 0.18rem;
    max-width: min(420px, calc(100% - 2rem));
    padding: 0.8rem 0.95rem;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(5, 12, 24, 0.8);
    backdrop-filter: blur(12px);
    color: #f8fafc;
    z-index: 4;
    pointer-events: none;
  }

  .canvas-callout strong {
    font-size: 0.84rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #fef3c7;
  }

  .canvas-callout span {
    color: rgba(226, 232, 240, 0.86);
    font-size: 0.85rem;
  }

  .canvas-callout.active {
    border-color: rgba(251, 191, 36, 0.28);
    box-shadow: 0 0 22px rgba(251, 191, 36, 0.16);
  }

  .selection-strip {
    margin-top: 0.85rem;
  }

  .selection-card {
    padding: 0.95rem 1rem;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.12);
    background: rgba(8, 15, 27, 0.62);
  }

  .selection-card.idle {
    background: rgba(8, 15, 27, 0.42);
  }

  .selection-header,
  .selection-grid,
  .selection-columns {
    display: grid;
    gap: 0.75rem;
  }

  .selection-header {
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: start;
  }

  .selection-header h4,
  .selection-column h5 {
    margin: 0;
    color: #f8fafc;
  }

  .selection-header p,
  .selection-card p {
    margin: 0.2rem 0 0 0;
    color: rgba(191, 219, 254, 0.78);
    font-size: 0.85rem;
  }

  .selection-clear {
    padding: 0.55rem 0.8rem;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(15, 23, 42, 0.7);
    color: #f8fafc;
    cursor: pointer;
  }

  .selection-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
    margin-top: 0.85rem;
  }

  .selection-stat {
    padding: 0.8rem;
    border-radius: 14px;
    background: rgba(15, 23, 42, 0.74);
    border: 1px solid rgba(148, 163, 184, 0.08);
  }

  .selection-stat span {
    display: block;
    color: rgba(148, 163, 184, 0.84);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .selection-stat strong {
    display: block;
    margin-top: 0.22rem;
    color: #f8fafc;
    font-size: 1rem;
  }

  .selection-columns {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    margin-top: 0.85rem;
  }

  .selection-column {
    padding: 0.85rem;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.62);
    border: 1px solid rgba(148, 163, 184, 0.08);
  }

  .selection-link {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 0.58rem 0;
    border-top: 1px solid rgba(148, 163, 184, 0.08);
    color: rgba(226, 232, 240, 0.9);
    font-size: 0.84rem;
  }

  .selection-link:first-of-type {
    border-top: 0;
    padding-top: 0.2rem;
  }

  .selection-route {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
    gap: 0.9rem;
    align-items: center;
    margin-top: 0.95rem;
    padding: 0.95rem;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    background:
      linear-gradient(180deg, rgba(7, 13, 24, 0.9), rgba(12, 20, 36, 0.94)),
      radial-gradient(circle at center, rgba(251, 191, 36, 0.08), transparent 55%);
  }

  .route-column {
    display: grid;
    gap: 0.55rem;
  }

  .route-label {
    color: rgba(148, 163, 184, 0.92);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .route-chips {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
  }

  .route-chip,
  .route-focus {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 2.25rem;
    padding: 0.45rem 0.72rem;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(15, 23, 42, 0.82);
    color: #f8fafc;
    font-size: 0.82rem;
    font-weight: 700;
  }

  .route-chip.muted {
    color: rgba(148, 163, 184, 0.88);
  }

  .route-focus {
    min-width: 84px;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.92), rgba(45, 212, 191, 0.92));
    border-color: rgba(191, 219, 254, 0.22);
    box-shadow: 0 0 24px rgba(45, 212, 191, 0.18);
  }

  .telemetry-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }

  .telemetry-card {
    padding: 1rem;
  }

  .telemetry-card.wide {
    grid-column: span 2;
  }

  .trend-svg,
  .timeline-svg {
    width: 100%;
    height: auto;
    margin-top: 0.8rem;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.12);
    background: linear-gradient(180deg, rgba(7, 12, 22, 0.98), rgba(14, 22, 36, 0.96));
  }

  .legend-pill.positive { color: #99f6e4; border-color: rgba(45, 212, 191, 0.28); }
  .legend-pill.negative { color: #fecdd3; border-color: rgba(251, 113, 133, 0.28); }
  .legend-pill.neutral { color: #dbeafe; border-color: rgba(96, 165, 250, 0.25); }
  .legend-pill.influence { color: #fef3c7; border-color: rgba(251, 191, 36, 0.25); }
  .legend-pill.data { color: #bfdbfe; border-color: rgba(96, 165, 250, 0.28); }
  .legend-pill.physics { color: #fecaca; border-color: rgba(248, 113, 113, 0.28); }
  .legend-pill.stage { color: #bfdbfe; border-color: rgba(96, 165, 250, 0.28); }
  .legend-pill.optimizer { color: #fde68a; border-color: rgba(251, 191, 36, 0.28); }
  .legend-pill.watchdog { color: #fecaca; border-color: rgba(248, 113, 113, 0.28); }
  .legend-pill.momentum { color: #bae6fd; border-color: rgba(56, 189, 248, 0.28); }
  .legend-pill.kinematics { color: #fef08a; border-color: rgba(250, 204, 21, 0.28); }
  .legend-pill.material { color: #bbf7d0; border-color: rgba(74, 222, 128, 0.28); }
  .legend-pill.boundary { color: #fecaca; border-color: rgba(248, 113, 113, 0.28); }

  @media (max-width: 900px) {
    .hero-shell,
    .telemetry-grid {
      grid-template-columns: 1fr;
    }

    .telemetry-card.wide {
      grid-column: span 1;
    }

    .focus-grid {
      grid-template-columns: 1fr;
    }

    .hero-microgrid {
      grid-template-columns: 1fr;
    }

    .flow-canvas {
      height: 360px;
    }

    .flow-hud {
      position: absolute;
      inset: auto 0.75rem 0.75rem 0.75rem;
      flex-direction: column;
    }

    .hud-stack,
    .hud-card,
    .hud-card.primary {
      width: 100%;
      min-width: 0;
    }
  }

  .impact-list,
  .connection-layout {
    margin-top: 0.9rem;
  }

  .impact-list {
    display: grid;
    gap: 0.7rem;
  }

  .impact-row {
    display: grid;
    grid-template-columns: minmax(110px, 0.35fr) minmax(0, 1fr);
    gap: 0.8rem;
    align-items: center;
    padding: 0.78rem 0.85rem;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    background: rgba(8, 15, 27, 0.56);
  }

  .impact-button {
    width: 100%;
    text-align: left;
    cursor: pointer;
  }

  .impact-button.selected {
    border-color: rgba(251, 191, 36, 0.28);
    background:
      linear-gradient(180deg, rgba(27, 19, 8, 0.92), rgba(17, 18, 25, 0.92)),
      radial-gradient(circle at left center, rgba(251, 191, 36, 0.12), transparent 36%);
    box-shadow: 0 0 22px rgba(251, 191, 36, 0.12);
  }

  .impact-id strong,
  .connection-row strong,
  .bridge-head strong,
  .connection-column h4 {
    display: block;
    color: #f8fafc;
    margin: 0;
  }

  .impact-id span,
  .impact-meta span,
  .connection-row span,
  .bridge-head span,
  .bridge-weight,
  .empty-copy {
    color: rgba(191, 219, 254, 0.8);
    font-size: 0.78rem;
  }

  .impact-bars {
    display: grid;
    gap: 0.45rem;
  }

  .impact-track {
    height: 10px;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.88);
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.08);
  }

  .impact-track.compact {
    height: 8px;
  }

  .impact-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, rgba(96, 165, 250, 0.96), rgba(45, 212, 191, 0.96));
    box-shadow: 0 0 18px rgba(45, 212, 191, 0.2);
  }

  .impact-fill.amber {
    background: linear-gradient(90deg, rgba(251, 191, 36, 0.96), rgba(249, 115, 22, 0.96));
    box-shadow: 0 0 18px rgba(251, 191, 36, 0.18);
  }

  .impact-meta {
    display: flex;
    gap: 0.65rem;
    flex-wrap: wrap;
  }

  .connection-layout {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }

  .connection-column {
    padding: 0.9rem;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    background: rgba(8, 15, 27, 0.56);
  }

  .connection-column h4 {
    margin: 0 0 0.8rem 0;
    font-size: 0.95rem;
  }

  .connection-row,
  .bridge-row {
    display: grid;
    gap: 0.4rem;
    padding: 0.68rem 0;
  }

  .connection-row {
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    border-top: 1px solid rgba(148, 163, 184, 0.08);
  }

  .connection-row:first-of-type {
    border-top: 0;
    padding-top: 0;
  }

  .connection-metrics {
    display: grid;
    justify-items: end;
    gap: 0.2rem;
  }

  .positive-weight {
    color: #99f6e4;
  }

  .negative-weight {
    color: #fecdd3;
  }

  .bridge-head {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    align-items: center;
  }

  @media (max-width: 900px) {
    .impact-row,
    .connection-layout,
    .selection-grid,
    .selection-columns,
    .selection-route {
      grid-template-columns: 1fr;
    }
  }

  @keyframes spotlight-drift {
    0%, 100% {
      transform: translate3d(-2%, -1%, 0) scale(0.98);
    }
    50% {
      transform: translate3d(2%, 1%, 0) scale(1.03);
    }
  }
</style>
