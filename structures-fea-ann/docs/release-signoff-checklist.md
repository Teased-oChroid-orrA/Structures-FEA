# Release Signoff Checklist

## Functional
- [ ] `Solve FEM Case` completes for cantilever and plate-hole presets.
- [ ] `Train PINO` starts/stops/resumes without UI lockup.
- [ ] Stage progress bars and overall progress remain monotonic.
- [ ] Live network panel updates during training.

## Numerical
- [ ] Plate-hole benchmark reports SCF trend near Kirsch reference.
- [ ] Residual pillars remain finite during long training.
- [ ] Epoch-window table shows the last 10 windows at 1000-epoch increments.
- [ ] Holdout validation baselines pass for cantilever, axial, and plate-with-hole.
- [ ] PINO signoff lock gate passes (`npm run verify:pino-signoff-lock`).

## Reliability
- [ ] Checkpoint save/list/resume latest/resume best/purge all succeed.
- [ ] Checkpoint health panel shows recoverability state correctly.
- [ ] Crash-restart resume tested from a saved checkpoint.

## Security / Offline
- [ ] `npm run verify:offline` passes.
- [ ] Runtime operations perform without outbound network calls.

## Platform
- [ ] `ci-matrix` passes on Linux/macOS/Windows.
- [ ] `windows-build` workflow produces bundle artifact.
- [ ] `release-signoff` workflow passes and publishes benchmark + PINO signoff artifacts.

## Performance
- [ ] `npm run verify:ui-budgets` passes.
- [ ] Poll health and telemetry throughput remain `stable/degraded` (not persistent `lagging`) in long run.
- [ ] PINO signoff profile records throughput, inference latency, memory estimate, and checkpoint-resume correctness.
