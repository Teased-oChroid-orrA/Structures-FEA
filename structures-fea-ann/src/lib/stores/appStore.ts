import { writable } from 'svelte/store';
import type {
  AnnResult,
  DynamicResult,
  ExportResult,
  FailureResult,
  FemResult,
  ModelStatus,
  ThermalResult
} from '$lib/types/contracts';

export const femResultStore = writable<FemResult | null>(null);
export const annResultStore = writable<AnnResult | null>(null);
export const thermalResultStore = writable<ThermalResult | null>(null);
export const dynamicResultStore = writable<DynamicResult | null>(null);
export const failureResultStore = writable<FailureResult | null>(null);
export const modelStatusStore = writable<ModelStatus | null>(null);
export const exportResultStore = writable<ExportResult | null>(null);
export const busyStore = writable(false);
export const errorStore = writable<string | null>(null);
