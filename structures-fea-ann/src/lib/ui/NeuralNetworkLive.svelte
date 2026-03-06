<script lang="ts">
  import NumberFlow from '@number-flow/svelte';
  import {
    SvelteFlow,
    Controls,
    Background,
    MiniMap,
    BackgroundVariant,
    MarkerType,
    type Node,
    type Edge
  } from '@xyflow/svelte';
  import type { TrainingProgressEvent } from '$lib/types/contracts';

  type Props = {
    trainingActive: boolean;
    progress: TrainingProgressEvent | null;
    history: TrainingProgressEvent[];
  };

  let { trainingActive, progress, history }: Props = $props();

  let nodes = $state<Node[]>([]);
  let edges = $state<Edge[]>([]);

  const network = $derived(progress?.network ?? null);

  $effect(() => {
    if (!network) {
      nodes = [];
      edges = [];
      return;
    }

    const builtNodes: Node[] = [];
    const builtEdges: Edge[] = [];
    const layerGap = 220;
    const yPad = 56;

    for (let layer = 0; layer < network.layerSizes.length; layer++) {
      const count = network.layerSizes[layer] ?? 1;
      const innerH = Math.max(160, count * 42);
      for (let idx = 0; idx < count; idx++) {
        const id = `L${layer}N${idx}`;
        const n = network.nodes.find((it) => it.id === id);
        const importance = Math.max(0, Math.min(1, (n?.importance ?? 0) / 4));
        const size = 34 + importance * 18;
        builtNodes.push({
          id,
          position: {
            x: 34 + layer * layerGap,
            y: yPad + ((idx + 1) / (count + 1)) * innerH
          },
          data: { label: `${layer}:${idx}` },
          draggable: false,
          selectable: false,
          style: `width:${size}px;height:${size}px;border-radius:999px;border:1px solid rgba(119,193,255,0.65);color:#d9e7f5;background:linear-gradient(140deg, rgba(18,27,38,0.95), rgba(35,50,70,0.95));font-size:10px;display:flex;align-items:center;justify-content:center;`
        });
      }
    }

    for (const c of network.connections) {
      builtEdges.push({
        id: `${c.fromId}->${c.toId}`,
        source: c.fromId,
        target: c.toId,
        type: 'smoothstep',
        animated: trainingActive,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 14,
          height: 14,
          color: c.weight >= 0 ? 'rgba(82,204,153,0.95)' : 'rgba(249,115,122,0.95)'
        },
        style: `stroke:${c.weight >= 0 ? 'rgba(82,204,153,0.8)' : 'rgba(249,115,122,0.8)'};stroke-width:${0.6 + c.magnitude * 3.6};`
      });
    }

    nodes = builtNodes;
    edges = builtEdges;
  });

  const sparkline = $derived.by(() => {
    const points = history.slice(-40);
    if (points.length === 0) return '';
    const w = 320;
    const h = 78;
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
</script>

<section class="panel stack">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem;flex-wrap:wrap;">
    <div class="stack" style="gap:0.35rem;">
      <h2>Live Neural Network (Svelte Flow)</h2>
      <p>Interactive live graph of topology and weight impact while training is running.</p>
    </div>
    <div class="kicker">
      <span class="chip {trainingActive ? 'ok' : ''}">{trainingActive ? 'training' : 'idle'}</span>
      <span class="chip">epoch: <NumberFlow value={progress?.epoch ?? 0} format={{ maximumFractionDigits: 0 }} />/<NumberFlow value={progress?.totalEpochs ?? 0} format={{ maximumFractionDigits: 0 }} /></span>
      <span class="chip">lr: <NumberFlow value={progress?.learningRate ?? 0} format={{ maximumSignificantDigits: 4 }} /></span>
    </div>
  </div>

  <div class="summary-cards">
    <article class="stat">
      <div class="label">Current Loss</div>
      <div class="value"><NumberFlow value={progress?.loss ?? 0} format={{ maximumSignificantDigits: 4 }} /></div>
    </article>
    <article class="stat">
      <div class="label">Validation Loss</div>
      <div class="value"><NumberFlow value={progress?.valLoss ?? 0} format={{ maximumSignificantDigits: 4 }} /></div>
    </article>
    <article class="stat">
      <div class="label">Architecture</div>
      <div class="value">{progress?.architecture?.join(' → ') ?? '-'}</div>
    </article>
    <article class="stat">
      <div class="label">Progress</div>
      <div class="value"><NumberFlow value={(progress?.progressRatio ?? 0) * 100} format={{ maximumFractionDigits: 1 }} />%</div>
    </article>
  </div>

  <div class="flow-canvas">
    <SvelteFlow {nodes} {edges} fitView minZoom={0.2} maxZoom={2} nodesConnectable={false} nodesDraggable={false} elementsSelectable={false}>
      <Controls />
      <MiniMap pannable zoomable />
      <Background variant={BackgroundVariant.Dots} gap={18} size={1.2} />
    </SvelteFlow>
    {#if !network}
      <div class="flow-overlay-hint">
        {trainingActive ? 'Waiting for live training events...' : 'Run Train ANN to stream live topology updates.'}
      </div>
    {/if}
  </div>

  <div style="display:flex;justify-content:space-between;gap:0.6rem;flex-wrap:wrap;align-items:center;">
    <p>Validation loss trend</p>
    <svg viewBox="0 0 320 78" style="width:320px;height:78px;border:1px solid rgba(120,138,158,0.42);border-radius:8px;background:#101722;">
      {#if sparkline}
        <polyline points={sparkline} fill="none" stroke="rgba(255,185,95,0.95)" stroke-width="2" />
      {/if}
    </svg>
  </div>
</section>
