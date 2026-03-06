<script lang="ts">
  import { extent, interpolateTurbo, scaleLinear, scaleSequential } from 'd3';
  import type { NodalDisplacement } from '$lib/types/contracts';
  import { extractMidThicknessSlice } from '$lib/mesh/meshlib';

  type Props = {
    nodalDisplacements: NodalDisplacement[] | null;
  };

  let { nodalDisplacements }: Props = $props();

  let deformMultiplier = $state(1);

  const sliceData = $derived.by(() => extractMidThicknessSlice(nodalDisplacements));

  const deformScale = $derived.by(() => {
    if (!sliceData) return 1;
    const modelSpan = Math.max(1e-9, sliceData.xMax - sliceData.xMin);
    return (0.15 * modelSpan / sliceData.maxDisp) * deformMultiplier;
  });

  const render = $derived.by(() => {
    if (!sliceData) return null;

    const width = 760;
    const height = 340;
    const pad = 34;

    const deformedCoords = sliceData.nodes.map((n) => ({
      x: n.xIn + n.uxIn * deformScale,
      y: n.yIn + n.uyIn * deformScale
    }));

    const [dxMin, dxMax] = extent(deformedCoords, (p) => p.x) as [number, number];
    const [dyMin, dyMax] = extent(deformedCoords, (p) => p.y) as [number, number];

    const xMin = Math.min(sliceData.xMin, dxMin ?? sliceData.xMin);
    const xMax = Math.max(sliceData.xMax, dxMax ?? sliceData.xMax);
    const yMin = Math.min(sliceData.yMin, dyMin ?? sliceData.yMin);
    const yMax = Math.max(sliceData.yMax, dyMax ?? sliceData.yMax);

    const sx = scaleLinear().domain([xMin, xMax]).range([pad, width - pad]);
    const sy = scaleLinear().domain([yMin, yMax]).range([height - pad, pad]);

    const vmHi = Math.max(sliceData.minVm + 1e-9, sliceData.maxVm);
    const vmScale = scaleSequential(interpolateTurbo).domain([sliceData.minVm, vmHi]);
    const rScale = scaleLinear().domain([0, sliceData.maxDisp]).range([2.2, 7.2]);

    const undeformed = sliceData.segments.map((s) => ({
      x1: sx(s.a.xIn),
      y1: sy(s.a.yIn),
      x2: sx(s.b.xIn),
      y2: sy(s.b.yIn)
    }));

    const deformed = sliceData.segments.map((s) => ({
      x1: sx(s.a.xIn + s.a.uxIn * deformScale),
      y1: sy(s.a.yIn + s.a.uyIn * deformScale),
      x2: sx(s.b.xIn + s.b.uxIn * deformScale),
      y2: sy(s.b.yIn + s.b.uyIn * deformScale)
    }));

    const points = sliceData.nodes.map((n) => ({
      x: sx(n.xIn + n.uxIn * deformScale),
      y: sy(n.yIn + n.uyIn * deformScale),
      vm: n.vmPsi,
      r: rScale(n.dispMagIn),
      fill: vmScale(n.vmPsi)
    }));

    return {
      width,
      height,
      undeformed,
      deformed,
      points,
      vmMin: sliceData.minVm,
      vmMax: vmHi,
      sliceZ: sliceData.sliceZ
    };
  });

  const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '0.00');
</script>

<section class="panel stack">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem;flex-wrap:wrap;">
    <div class="stack" style="gap:0.35rem;">
      <h3>3D Mesh Slice With Superimposed Results</h3>
      <p>Mid-thickness slice showing undeformed mesh, deformed mesh, and Von Mises stress field at nodes.</p>
    </div>
    <div class="kicker">
      <span class="chip">deformation scale: {fmt(deformScale, 2)}x</span>
      <span class="chip">slice z: {fmt(render?.sliceZ ?? 0, 4)} in</span>
    </div>
  </div>

  <label class="field">
    <span>Visual Amplification</span>
    <input type="range" min="0.2" max="2.4" step="0.05" bind:value={deformMultiplier} />
  </label>

  {#if render}
    <svg viewBox={`0 0 ${render.width} ${render.height}`} style="width:100%;border:1px solid rgba(120,138,158,0.42);border-radius:12px;background:#101722;">
      <defs>
        <linearGradient id="stressLegend" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#30123b" />
          <stop offset="20%" stop-color="#4145ab" />
          <stop offset="40%" stop-color="#2f89d9" />
          <stop offset="60%" stop-color="#21c7a8" />
          <stop offset="80%" stop-color="#9bd93c" />
          <stop offset="100%" stop-color="#f9c80e" />
        </linearGradient>
      </defs>

      {#each render.undeformed as seg, i (`u-${i}`)}
        <line x1={seg.x1} y1={seg.y1} x2={seg.x2} y2={seg.y2} stroke="rgba(132,148,168,0.45)" stroke-width="1" />
      {/each}

      {#each render.deformed as seg, i (`d-${i}`)}
        <line x1={seg.x1} y1={seg.y1} x2={seg.x2} y2={seg.y2} stroke="rgba(77,209,255,0.72)" stroke-width="1.2" />
      {/each}

      {#each render.points as p, i (`p-${i}`)}
        <circle cx={p.x} cy={p.y} r={p.r} fill={p.fill} stroke="rgba(3,10,16,0.8)" stroke-width="0.8" />
      {/each}

      <rect x="24" y={render.height - 24} width="220" height="9" fill="url(#stressLegend)" rx="5" ry="5" />
      <text x="24" y={render.height - 30} fill="#b7c8d8" font-size="10">Von Mises stress</text>
      <text x="24" y={render.height - 4} fill="#b7c8d8" font-size="10">{fmt(render.vmMin, 1)} psi</text>
      <text x="244" y={render.height - 4} fill="#b7c8d8" font-size="10" text-anchor="end">{fmt(render.vmMax, 1)} psi</text>
    </svg>
  {:else}
    <div class="panel stack">
      <p>Run FEM or ANN solve to generate nodal mesh and stress data.</p>
    </div>
  {/if}
</section>
