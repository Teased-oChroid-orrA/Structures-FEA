<script lang="ts">
  import { onMount } from 'svelte';
  import { MAX_DENSE_SOLVER_DOFS, type SolveInput } from '$lib/types/contracts';

  type Props = {
    solveInput: SolveInput;
  };

  let { solveInput = $bindable() }: Props = $props();

  let selectedPreset = $state<'nafems_plate_hole' | 'simple_cantilever'>('simple_cantilever');

  const applyPreset = (preset: 'nafems_plate_hole' | 'simple_cantilever') => {
    selectedPreset = preset;
    if (preset === 'simple_cantilever') {
      Object.assign(solveInput.geometry, {
        lengthIn: 10,
        widthIn: 1,
        thicknessIn: 0.25,
        holeDiameterIn: 0
      });
      Object.assign(solveInput.mesh, {
        nx: 18,
        ny: 4,
        nz: 1,
        elementType: 'hex8',
        autoAdapt: true,
        maxDofs: MAX_DENSE_SOLVER_DOFS,
        amrEnabled: false,
        amrPasses: 0,
        amrMaxNx: 30,
        amrRefineRatio: 1.15
      });
      Object.assign(solveInput.material, {
        ePsi: 29_000_000,
        nu: 0.3,
        rhoLbIn3: 0.283,
        alphaPerF: 6.5e-6,
        yieldStrengthPsi: 36_000
      });
      Object.assign(solveInput.boundaryConditions, {
        fixStartFace: true,
        fixEndFace: false
      });
      Object.assign(solveInput.load, {
        axialLoadLbf: 0,
        verticalPointLoadLbf: -100
      });
      solveInput.unitSystem = 'inch-lbf-second';
      solveInput.deltaTF = 0;
      return;
    }

    Object.assign(solveInput.geometry, {
      lengthIn: 11.811,
      widthIn: 4.724,
      thicknessIn: 0.25,
      holeDiameterIn: 2.362
    });
    Object.assign(solveInput.mesh, {
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
    });
    Object.assign(solveInput.material, {
      ePsi: 29_000_000,
      nu: 0.3,
      rhoLbIn3: 0.283,
      alphaPerF: 6.5e-6,
      yieldStrengthPsi: 36_000
    });
    Object.assign(solveInput.boundaryConditions, {
      fixStartFace: true,
      fixEndFace: false
    });
    Object.assign(solveInput.load, {
      axialLoadLbf: 1712,
      verticalPointLoadLbf: 0
    });
    solveInput.unitSystem = 'inch-lbf-second';
    solveInput.deltaTF = 0;
  };

  const tractionPsi = $derived.by(() => {
    const area = solveInput.geometry.widthIn * solveInput.geometry.thicknessIn;
    if (area <= 0) return 0;
    return solveInput.load.axialLoadLbf / area;
  });

  const supportLabel = $derived.by(() => {
    if (solveInput.boundaryConditions.fixStartFace && solveInput.boundaryConditions.fixEndFace) {
      return 'both faces fixed';
    }
    if (solveInput.boundaryConditions.fixStartFace) {
      return 'start face fixed';
    }
    if (solveInput.boundaryConditions.fixEndFace) {
      return 'end face fixed';
    }
    return 'unsupported';
  });

  onMount(() => {
    applyPreset(selectedPreset);
  });
</script>

<div class="panel stack">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem;flex-wrap:wrap;">
    <div class="stack" style="gap:0.35rem;">
      <h2>Input Deck</h2>
      <p>Select a benchmark preset, then tune material law and mesh/discretization controls for the current dense solver path.</p>
    </div>
    <div class="kicker">
      <span class="chip">unit: inch-lbf-s</span>
      <span class="chip ok">equiv. stress: {tractionPsi.toFixed(2)} psi</span>
      <span class="chip">dense cap: {MAX_DENSE_SOLVER_DOFS.toLocaleString()} DOFs</span>
      <span class="chip">{supportLabel}</span>
    </div>
  </div>

  <div class="field-grid">
    <label class="field">
      <span>Benchmark Preset</span>
      <select bind:value={selectedPreset}>
        <option value="nafems_plate_hole">NAFEMS Plate with Hole (SCF)</option>
        <option value="simple_cantilever">Simple Cantilever Tip Load (Verification)</option>
      </select>
    </label>
    <label class="field">
      <span>Preset Action</span>
      <button class="btn-secondary" type="button" onclick={() => applyPreset(selectedPreset)}>Apply Preset</button>
    </label>
  </div>

  <div class="field-grid">
    <label class="field">
      <span>Length (in)</span>
      <input type="number" bind:value={solveInput.geometry.lengthIn} step="0.001" />
    </label>
    <label class="field">
      <span>Width (in)</span>
      <input type="number" bind:value={solveInput.geometry.widthIn} step="0.001" />
    </label>
    <label class="field">
      <span>Thickness (in)</span>
      <input type="number" bind:value={solveInput.geometry.thicknessIn} step="0.001" />
    </label>
    <label class="field">
      <span>Hole Diameter (in)</span>
      <input type="number" bind:value={solveInput.geometry.holeDiameterIn} step="0.001" />
    </label>
    <label class="field">
      <span>Axial Tension Load (lbf)</span>
      <input type="number" bind:value={solveInput.load.axialLoadLbf} step="1" />
    </label>
    <label class="field">
      <span>Vertical Point Load (lbf)</span>
      <input type="number" bind:value={solveInput.load.verticalPointLoadLbf} step="1" />
    </label>
  </div>

  <div class="field-grid">
    <label class="field">
      <span>Fix Start Face</span>
      <select bind:value={solveInput.boundaryConditions.fixStartFace}>
        <option value={true}>true</option>
        <option value={false}>false</option>
      </select>
    </label>
    <label class="field">
      <span>Fix End Face</span>
      <select bind:value={solveInput.boundaryConditions.fixEndFace}>
        <option value={true}>true</option>
        <option value={false}>false</option>
      </select>
    </label>
  </div>

  <div class="field-grid">
    <label class="field">
      <span>E (psi)</span>
      <input type="number" bind:value={solveInput.material.ePsi} step="1" />
    </label>
    <label class="field">
      <span>Poisson (nu)</span>
      <input type="number" bind:value={solveInput.material.nu} step="0.001" />
    </label>
    <label class="field">
      <span>Density (lb/in^3)</span>
      <input type="number" bind:value={solveInput.material.rhoLbIn3} step="0.0001" />
    </label>
    <label class="field">
      <span>Thermal alpha (/F)</span>
      <input type="number" bind:value={solveInput.material.alphaPerF} step="0.000001" />
    </label>
    <label class="field">
      <span>Yield Strength (psi)</span>
      <input type="number" bind:value={solveInput.material.yieldStrengthPsi} step="1" />
    </label>
    <label class="field">
      <span>Element Type</span>
      <select bind:value={solveInput.mesh.elementType}>
        <option value="hex8">hex8</option>
      </select>
    </label>
    <label class="field">
      <span>Mesh NX</span>
      <input type="number" bind:value={solveInput.mesh.nx} min="1" step="1" />
    </label>
    <label class="field">
      <span>Mesh NY</span>
      <input type="number" bind:value={solveInput.mesh.ny} min="1" step="1" />
    </label>
    <label class="field">
      <span>Mesh NZ</span>
      <input type="number" bind:value={solveInput.mesh.nz} min="1" step="1" />
    </label>
    <label class="field">
      <span>Auto Mesh Adapt</span>
      <select bind:value={solveInput.mesh.autoAdapt}>
        <option value={true}>true</option>
        <option value={false}>false</option>
      </select>
    </label>
    <label class="field">
      <span>Max DOFs</span>
      <input type="number" bind:value={solveInput.mesh.maxDofs} min="300" max={MAX_DENSE_SOLVER_DOFS} step="100" />
    </label>
    <label class="field">
      <span>AMR Enabled</span>
      <select bind:value={solveInput.mesh.amrEnabled}>
        <option value={true}>true</option>
        <option value={false}>false</option>
      </select>
    </label>
    <label class="field">
      <span>AMR Passes</span>
      <input type="number" bind:value={solveInput.mesh.amrPasses} min="0" step="1" />
    </label>
    <label class="field">
      <span>AMR Max NX</span>
      <input type="number" bind:value={solveInput.mesh.amrMaxNx} min="2" step="1" />
    </label>
    <label class="field">
      <span>AMR Refine Ratio</span>
      <input type="number" bind:value={solveInput.mesh.amrRefineRatio} min="1" step="0.05" />
    </label>
  </div>
</div>
