export type MeshLibStatus = {
  ready: boolean;
  detail: string;
};

let cached: MeshLibStatus | null = null;

export async function warmupMeshLib(): Promise<MeshLibStatus> {
  if (cached) return cached;

  try {
    const isTauriRuntime =
      typeof window !== 'undefined' && typeof (window as any).__TAURI_INTERNALS__ !== 'undefined';
    if (isTauriRuntime) {
      cached = {
        ready: false,
        detail:
          'MeshLib warmup skipped in Tauri runtime to avoid worker clone incompatibilities. Using internal meshing kernels.'
      };
      return cached;
    }

    // MeshLib WASM/thread boot requires cross-origin isolation and SharedArrayBuffer support.
    // In Tauri/WebView contexts this may not be available; skip hard init in that case.
    const hasSharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';
    const isIsolated = typeof crossOriginIsolated !== 'undefined' ? crossOriginIsolated : false;
    if (!hasSharedArrayBuffer || !isIsolated) {
      cached = {
        ready: false,
        detail:
          'MeshLib runtime requires cross-origin isolation (SharedArrayBuffer). Skipping MeshLib boot and using internal meshing kernels.'
      };
      return cached;
    }

    const mod = await import('@alpinebuster/meshlib');
    const create = (mod as any).createMeshLib ?? (mod as any).default;
    if (typeof create !== 'function') {
      cached = { ready: false, detail: 'MeshLib loaded but initializer not found.' };
      return cached;
    }

    const timeoutMs = 5000;
    const ready = await Promise.race([
      Promise.resolve()
        .then(() => create())
        .then(() => true)
        .catch(() => false),
      new Promise<boolean>((resolve) => setTimeout(() => resolve(false), timeoutMs))
    ]);

    if (!ready) {
      cached = {
        ready: false,
        detail: 'MeshLib warmup timed out. Continuing with internal meshing kernels.'
      };
      return cached;
    }

    cached = { ready: true, detail: 'MeshLib kernels warmed up.' };
    return cached;
  } catch (error) {
    cached = {
      ready: false,
      detail: `MeshLib unavailable in this runtime (${String(error)}). Using internal meshing kernels.`
    };
    return cached;
  }
}
