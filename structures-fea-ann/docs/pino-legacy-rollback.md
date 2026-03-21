# PINO Legacy Rollback (Internal Only)

## Purpose

Primary runtime is PINO-only. Legacy ANN compatibility is retained only as an internal rollback harness for emergency diagnostics during the deprecation window.

## Default Behavior

- Unknown backend strings no longer route to legacy ANN runtime.
- They fall back to `pino-ndarray-cpu`.

## Internal Rollback Switch

Set the environment variable below before launching the app or tests:

`PINO_ENABLE_LEGACY_ROLLBACK=1`

With this flag set, unknown backend strings can route to the legacy compat runtime (`CompatAnn`) for rollback testing.

## Safety Notes

- This flag is not exposed in the UI.
- Do not enable it for standard production runs.
- Remove the rollback path after one stable release cycle.
