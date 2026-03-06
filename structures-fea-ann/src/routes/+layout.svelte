<script lang="ts">
  import '../app.css';
  import { styles as numberFlowStyles } from '@number-flow/svelte';
  import '@xyflow/svelte/dist/style.css';
</script>

<svelte:head>
  {@html `<style>${numberFlowStyles}</style>`}
</svelte:head>

<slot />
