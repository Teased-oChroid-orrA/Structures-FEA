export type MeshLibStatus = {
  ready: boolean;
  detail: string;
  runtime: 'meshlib' | 'internal';
  reason:
    | 'active'
    | 'missing-window'
    | 'missing-worker'
    | 'missing-shared-array-buffer'
    | 'missing-cross-origin-isolation'
    | 'load-failed'
    | 'init-failed'
    | 'timeout';
};

let cached: MeshLibStatus | null = null;
let initPromise: Promise<MeshLibStatus> | null = null;

async function loadMeshLibFactory(): Promise<() => Promise<unknown>> {
  const specifier = '@alpinebuster/meshlib';
  const mod = await import(/* @vite-ignore */ specifier);
  const create = mod.createMeshLib ?? mod.create ?? mod.MRMesh?.create ?? mod.default;
  if (typeof create !== 'function') {
    throw new Error('createMeshLib export not found');
  }
  return create as () => Promise<unknown>;
}

function internal(reason: MeshLibStatus['reason'], detail: string): MeshLibStatus {
  return {
    ready: false,
    runtime: 'internal',
    reason,
    detail
  };
}

export async function warmupMeshLib(): Promise<MeshLibStatus> {
  if (cached) return cached;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      if (!import.meta.env.DEV) {
        cached = internal(
          'active',
          'MeshLib warmup is disabled in production builds. Using internal meshing kernels.'
        );
        return cached;
      }

      if (typeof window === 'undefined') {
        cached = internal('missing-window', 'MeshLib requires a browser window runtime. Using internal meshing kernels.');
        return cached;
      }

      if (typeof Worker === 'undefined') {
        cached = internal('missing-worker', 'MeshLib requires Web Worker support. Using internal meshing kernels.');
        return cached;
      }

      const hasSharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';
      if (!hasSharedArrayBuffer) {
        cached = internal(
          'missing-shared-array-buffer',
          'MeshLib requires SharedArrayBuffer for threaded WASM. Using internal meshing kernels.'
        );
        return cached;
      }

      const isIsolated = typeof crossOriginIsolated !== 'undefined' ? crossOriginIsolated : false;
      if (!isIsolated) {
        cached = internal(
          'missing-cross-origin-isolation',
          'MeshLib requires cross-origin isolation. Using internal meshing kernels.'
        );
        return cached;
      }

      const timeoutMs = 5000;
      const createMeshLib = await Promise.race([
        loadMeshLibFactory(),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('meshlib-import-timeout')), timeoutMs)
        )
      ]);

      await Promise.race([
        Promise.resolve().then(() => createMeshLib()),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('meshlib-init-timeout')), timeoutMs)
        )
      ]);

      cached = {
        ready: true,
        runtime: 'meshlib',
        reason: 'active',
        detail: 'MeshLib kernels warmed up and active in this runtime.'
      };
      return cached;
    } catch (error) {
      const message = String(error);
      cached = message.includes('timeout')
        ? internal('timeout', 'MeshLib warmup timed out. Using internal meshing kernels.')
        : message.includes('createMeshLib export not found')
          ? internal('load-failed', 'MeshLib package loaded but no compatible factory was exported. Using internal meshing kernels.')
          : internal('init-failed', `MeshLib initialization failed (${message}). Using internal meshing kernels.`);
      return cached;
    } finally {
      initPromise = null;
    }
  })();

  return initPromise;
}
