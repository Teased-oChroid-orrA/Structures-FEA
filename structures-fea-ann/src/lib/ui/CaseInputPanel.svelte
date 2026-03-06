<script lang="ts">
  import type { SolveInput } from '$lib/types/contracts';

  type Props = {
    solveInput: SolveInput;
  };

  let { solveInput = $bindable() }: Props = $props();

  const tractionPsi = $derived.by(() => {
    const area = solveInput.geometry.widthIn * solveInput.geometry.thicknessIn;
    if (area <= 0) return 0;
    return solveInput.load.verticalPointLoadLbf / area;
  });
</script>

<div class="panel stack">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem;flex-wrap:wrap;">
    <div class="stack" style="gap:0.35rem;">
      <h2>Input Deck</h2>
      <p>Cantilever beam setup, material law, and mesh/discretization controls.</p>
    </div>
    <div class="kicker">
      <span class="chip">unit: inch-lbf-s</span>
      <span class="chip ok">equiv. stress: {tractionPsi.toFixed(2)} psi</span>
    </div>
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
      <span>Vertical Point Load (lbf)</span>
      <input type="number" bind:value={solveInput.load.verticalPointLoadLbf} step="1" />
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
        <option value="tet4">tet4</option>
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
      <input type="number" bind:value={solveInput.mesh.maxDofs} min="300" step="100" />
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
