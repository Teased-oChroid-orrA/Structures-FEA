<script lang="ts">
  import { onMount } from 'svelte';
  import { Canvas, T } from '@threlte/core';
  import { Grid, OrbitControls } from '@threlte/extras';
  import type { NodalDisplacement } from '$lib/types/contracts';

  type Props = {
    nodalDisplacements: NodalDisplacement[] | null;
  };

  let { nodalDisplacements }: Props = $props();

  let mounted = $state(false);
  let webglAvailable = $state(false);
  let webglDetail = $state('');
  let deformAmplification = $state(80);
  let hoveredNodeId = $state<number | null>(null);
  let selectedNodeId = $state<number | null>(null);

  onMount(() => {
    mounted = true;
    const canvas = document.createElement('canvas');
    const gl =
      canvas.getContext('webgl2', { failIfMajorPerformanceCaveat: true }) ??
      canvas.getContext('webgl', { failIfMajorPerformanceCaveat: true }) ??
      canvas.getContext('experimental-webgl');
    webglAvailable = Boolean(gl);
    if (!webglAvailable) {
      webglDetail = 'WebGL is unavailable in this runtime. Falling back to the 2D mesh and scalar views.';
    }
  });

  const vmRange = $derived.by(() => {
    const nodes = nodalDisplacements ?? [];
    if (nodes.length === 0) return { min: 0, max: 1 };
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const n of nodes) {
      if (n.vmPsi < min) min = n.vmPsi;
      if (n.vmPsi > max) max = n.vmPsi;
    }
    return { min, max: Math.max(max, min + 1e-9) };
  });

  const renderNodes = $derived.by(() => {
    const nodes = nodalDisplacements ?? [];
    if (nodes.length === 0) return [] as {
      nodeId: number;
      undeformed: [number, number, number];
      deformed: [number, number, number];
      color: string;
      radius: number;
    }[];

    const vmSpan = vmRange.max - vmRange.min;
    const maxDisp = Math.max(1e-12, ...nodes.map((n) => n.dispMagIn));

    return nodes.map((n) => {
      const t = Math.max(0, Math.min(1, (n.vmPsi - vmRange.min) / vmSpan));
      const hue = 220 - t * 220;
      const color = `hsl(${hue} 86% 56%)`;
      const radius = 0.018 + (n.dispMagIn / maxDisp) * 0.05;
      return {
        nodeId: n.nodeId,
        undeformed: [n.xIn, n.yIn, n.zIn] as [number, number, number],
        deformed: [
          n.xIn + n.uxIn * deformAmplification,
          n.yIn + n.uyIn * deformAmplification,
          n.zIn + n.uzIn * deformAmplification
        ] as [number, number, number],
        color,
        radius
      };
    });
  });

  const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '0.00');
</script>

<section class="panel stack">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem;flex-wrap:wrap;">
    <div class="stack" style="gap:0.35rem;">
      <h3>3D Mesh + Deflection View (Threlte)</h3>
      <p>Geometry state rendered declaratively with Threlte and interactive orbit controls.</p>
    </div>
    <div class="kicker">
      <span class="chip">amplification: {fmt(deformAmplification, 1)}x</span>
      <span class="chip">nodes: {renderNodes.length}</span>
      <span class="chip">selected: {selectedNodeId ?? '-'}</span>
    </div>
  </div>

  <label class="field">
    <span>Deflection Amplification</span>
    <input type="range" min="1" max="260" step="1" bind:value={deformAmplification} />
  </label>

  {#if mounted && webglAvailable}
    <div class="threlte-canvas">
      <Canvas dpr={1.5} renderMode="on-demand" colorManagementEnabled={true}>
        <T.PerspectiveCamera makeDefault position={[14, 9, 14]} fov={40} near={0.01} far={2000}>
          <OrbitControls enableDamping dampingFactor={0.08} />
        </T.PerspectiveCamera>
        <T.AmbientLight intensity={0.55} />
        <T.DirectionalLight position={[14, 10, 8]} intensity={1.2} />
        <T.DirectionalLight position={[-9, 6, -10]} intensity={0.55} />

        <Grid cellSize={0.5} sectionSize={2} infiniteGrid fadeDistance={35} fadeStrength={1.8} />

        {#each renderNodes as n (`u-${n.nodeId}`)}
          <T.Mesh position={n.undeformed}>
            <T.SphereGeometry args={[Math.max(0.012, n.radius * 0.35), 10, 10]} />
            <T.MeshStandardMaterial color="#6f7e92" transparent opacity={0.25} />
          </T.Mesh>
        {/each}

        {#each renderNodes as n (`d-${n.nodeId}`)}
          <T.Mesh
            position={n.deformed}
            onpointerenter={() => (hoveredNodeId = n.nodeId)}
            onpointerleave={() => (hoveredNodeId = null)}
            onclick={() => (selectedNodeId = n.nodeId)}
          >
            <T.SphereGeometry args={[n.radius, 12, 12]} />
            <T.MeshStandardMaterial
              color={n.color}
              emissive={selectedNodeId === n.nodeId ? '#ffd166' : hoveredNodeId === n.nodeId ? '#b6f0ff' : '#101722'}
              emissiveIntensity={selectedNodeId === n.nodeId ? 0.55 : hoveredNodeId === n.nodeId ? 0.3 : 0.08}
              metalness={0.08}
              roughness={0.45}
            />
          </T.Mesh>
        {/each}
      </Canvas>
    </div>
  {:else if mounted}
    <p>{webglDetail || 'WebGL is unavailable in this runtime. Falling back to the 2D mesh and scalar views.'}</p>
  {:else}
    <p>Preparing 3D scene...</p>
  {/if}
</section>
