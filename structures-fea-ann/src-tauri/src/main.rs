#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ann;
mod benchmarks;
mod contracts;
mod fem;
mod io;
mod physics;
mod pinn;
mod pinn_burn;
mod pino;
mod pino_burn_head;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use benchmarks::{apply_training_benchmark, get_training_benchmark, list_training_benchmarks};
use contracts::*;
use pinn::UniversalPinnEngine;
use tauri::{Emitter, State};

struct AppState {
    model: Arc<Mutex<UniversalPinnEngine>>,
    training_tick: Arc<Mutex<TrainingTickEvent>>,
    training_progress: Arc<Mutex<TrainingProgressEvent>>,
    training_status: Arc<Mutex<TrainingRunStatus>>,
    training_stop: Arc<AtomicBool>,
}

fn parse_optimizer_id_from_notes(notes: &[String]) -> String {
    notes
        .iter()
        .find_map(|n| n.split("optimizerId=").nth(1))
        .map(|tail| tail.split(',').next().unwrap_or("pino-adam").to_string())
        .unwrap_or_else(|| "pino-adam".to_string())
}

fn parse_stage_id_from_notes(notes: &[String]) -> String {
    if let Some(stage) = notes
        .iter()
        .find_map(|n| n.split("curriculumStage=").nth(1))
        .map(|tail| tail.split(',').next().unwrap_or("1"))
    {
        return match stage {
            "1" => "stage-1".to_string(),
            "2" => "stage-2".to_string(),
            "3" => "stage-3".to_string(),
            _ => "idle".to_string(),
        };
    }
    "idle".to_string()
}

fn resolve_training_mode(batch: &TrainingBatch) -> String {
    batch.training_mode.clone().unwrap_or_else(|| {
        if batch.benchmark_id.is_some() {
            "benchmark".to_string()
        } else if batch.target_loss.is_finite() && batch.target_loss > 0.0 && batch.target_loss <= 1e-8 {
            "legacy-mixed-exact".to_string()
        } else {
            "production-generalized".to_string()
        }
    })
}

fn resolve_gate_status(running: bool, completed: bool, reached_target: bool, stop_reason: &str) -> String {
    if running {
        "running".to_string()
    } else if !completed {
        "queued".to_string()
    } else if reached_target {
        "passed".to_string()
    } else if stop_reason.contains("plateau") {
        "stalled".to_string()
    } else if stop_reason.contains("manual") {
        "stopped".to_string()
    } else {
        "failed".to_string()
    }
}

fn dominant_blocker_from_progress(progress: &TrainingProgressEvent) -> Option<String> {
    [
        ("val-stress-fit", progress.val_stress_fit),
        ("val-constitutive-normal", progress.val_constitutive_normal_residual),
        ("val-constitutive-shear", progress.val_constitutive_shear_residual),
        ("val-invariant", progress.val_invariant_residual),
        ("boundary", progress.boundary_residual),
        ("material", progress.material_residual),
        ("kinematic", progress.kinematic_residual),
        ("momentum", progress.momentum_residual),
    ]
    .into_iter()
    .filter(|(_, value)| value.is_finite())
    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    .and_then(|(name, value)| if value > 0.0 { Some(name.to_string()) } else { None })
}

fn model_status_from_snapshot(state: &pinn::UniversalPinnState) -> ModelStatus {
    if let Some(burn) = &state.burn {
        return ModelStatus {
            model_version: burn.model_version,
            architecture: burn.architecture.clone(),
            learning_rate: burn.learning_rate,
            last_loss: burn.last_loss,
            train_samples: burn.train_samples,
            audit_frequency: state.ann.audit_frequency,
            fallback_enabled: state.ann.fallback_enabled,
            safeguard_settings: SafeguardSettings {
                preset: state.ann.safeguard_preset.clone(),
                uncertainty_threshold: state.ann.safeguard_uncertainty_threshold,
                residual_threshold: state.ann.safeguard_residual_threshold,
                adaptive_by_geometry: state.ann.safeguard_adaptive_by_geometry,
            },
            surrogate_domain: state.ann.training_domain.clone(),
            pino: burn.pino.clone(),
        };
    }
    ModelStatus {
        model_version: state.ann.model_version,
        architecture: state.ann.layer_sizes.clone(),
        learning_rate: state.ann.learning_rate,
        last_loss: state.ann.last_loss,
        train_samples: state.ann.train_samples,
        audit_frequency: state.ann.audit_frequency,
        fallback_enabled: state.ann.fallback_enabled,
        safeguard_settings: SafeguardSettings {
            preset: state.ann.safeguard_preset.clone(),
            uncertainty_threshold: state.ann.safeguard_uncertainty_threshold,
            residual_threshold: state.ann.safeguard_residual_threshold,
            adaptive_by_geometry: state.ann.safeguard_adaptive_by_geometry,
        },
        surrogate_domain: state.ann.training_domain.clone(),
        pino: None,
    }
}

fn default_diagnostics(
    lr: f64,
    arch: Vec<usize>,
    training_mode: String,
    benchmark_id: Option<String>,
    run_budget_total: usize,
) -> TrainingDiagnostics {
    TrainingDiagnostics {
        best_val_loss: f64::MAX,
        epochs_since_improvement: 0,
        lr_schedule_phase: "idle".to_string(),
        current_learning_rate: lr,
        data_weight: 1.0,
        physics_weight: 0.15,
        active_learning_rounds: 0,
        active_learning_samples_added: 0,
        safeguard_triggers: 0,
        curriculum_backoffs: 0,
        optimizer_switches: 0,
        checkpoint_rollbacks: 0,
        target_floor_estimate: 0.0,
        trend_stop_reason: "idle".to_string(),
        active_stage: "idle".to_string(),
        active_optimizer: "pino-adam".to_string(),
        bo_presearch_used: false,
        bo_selected_architecture: arch,
        residual_weight_momentum: 1.0,
        residual_weight_kinematics: 1.0,
        residual_weight_material: 1.0,
        residual_weight_boundary: 1.0,
        momentum_residual: 0.0,
        kinematic_residual: 0.0,
        material_residual: 0.0,
        boundary_residual: 0.0,
        displacement_fit: 0.0,
        stress_fit: 0.0,
        invariant_residual: 0.0,
        constitutive_normal_residual: 0.0,
        constitutive_shear_residual: 0.0,
        val_displacement_fit: 0.0,
        val_stress_fit: 0.0,
        val_invariant_residual: 0.0,
        val_constitutive_normal_residual: 0.0,
        val_constitutive_shear_residual: 0.0,
        hybrid_mode: "hybrid".to_string(),
        collocation_points: 0,
        boundary_points: 0,
        interface_points: 0,
        collocation_samples_added: 0,
        train_data_size: 0,
        train_data_cap: 0,
        training_mode,
        benchmark_id,
        gate_status: "queued".to_string(),
        certified_best_metric: f64::MAX,
        reproducibility_spread: None,
        dominant_blocker: None,
        stalled_reason: None,
        benchmark_certification: None,
        run_budget_used: 0,
        run_budget_total,
        recent_events: vec![],
        pino: None,
    }
}

fn push_recent_event(events: &mut Vec<String>, msg: String) {
    events.push(msg);
    if events.len() > 24 {
        let trim = events.len() - 24;
        events.drain(0..trim);
    }
}

fn extract_metric(notes: &[String], key: &str) -> Option<f64> {
    for note in notes {
        if let Some(idx) = note.find(key) {
            let start = idx + key.len();
            let tail = &note[start..];
            let value = tail
                .split(|c: char| c == ',' || c.is_whitespace())
                .find(|s| !s.is_empty())?;
            if let Ok(v) = value.parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

fn merge_training_tick(current: &TrainingTickEvent, incoming: &TrainingTickEvent) -> TrainingTickEvent {
    if incoming.epoch > 0 || current.epoch == 0 {
        return incoming.clone();
    }
    TrainingTickEvent {
        epoch: current.epoch,
        total_epochs: current.total_epochs.max(incoming.total_epochs),
        loss: current.loss,
        val_loss: current.val_loss,
        learning_rate: if incoming.learning_rate > 0.0 {
            incoming.learning_rate
        } else {
            current.learning_rate
        },
        architecture: if incoming.architecture.is_empty() {
            current.architecture.clone()
        } else {
            incoming.architecture.clone()
        },
        progress_ratio: current.progress_ratio.max(incoming.progress_ratio),
    }
}

fn merge_training_progress(
    current: &TrainingProgressEvent,
    incoming: &TrainingProgressEvent,
) -> TrainingProgressEvent {
    let keep_completed_epoch = incoming.epoch == 0 && current.epoch > 0;
    let mut merged = if keep_completed_epoch {
        let mut next = current.clone();
        next.total_epochs = next.total_epochs.max(incoming.total_epochs);
        next.hybrid_mode = incoming.hybrid_mode.clone();
        next.stage_id = incoming.stage_id.clone();
        next.optimizer_id = incoming.optimizer_id.clone();
        next.lr_phase = incoming.lr_phase.clone();
        next.watchdog_trigger_count = incoming.watchdog_trigger_count;
        next.collocation_samples_added = incoming.collocation_samples_added;
        next.train_data_size = incoming.train_data_size;
        next.train_data_cap = incoming.train_data_cap;
        if incoming.loss.is_finite() && incoming.loss > 0.0 {
            next.loss = incoming.loss;
        }
        if incoming.val_loss.is_finite() && incoming.val_loss > 0.0 {
            next.val_loss = incoming.val_loss;
        }
        if incoming.data_loss.is_finite() && incoming.data_loss > 0.0 {
            next.data_loss = incoming.data_loss;
        }
        if incoming.physics_loss.is_finite() && incoming.physics_loss > 0.0 {
            next.physics_loss = incoming.physics_loss;
        }
        if incoming.val_data_loss.is_finite() && incoming.val_data_loss > 0.0 {
            next.val_data_loss = incoming.val_data_loss;
        }
        if incoming.val_physics_loss.is_finite() && incoming.val_physics_loss > 0.0 {
            next.val_physics_loss = incoming.val_physics_loss;
        }
        next.residual_weight_momentum = incoming.residual_weight_momentum;
        next.residual_weight_kinematics = incoming.residual_weight_kinematics;
        next.residual_weight_material = incoming.residual_weight_material;
        next.residual_weight_boundary = incoming.residual_weight_boundary;
        next.learning_rate = if incoming.learning_rate > 0.0 {
            incoming.learning_rate
        } else {
            current.learning_rate
        };
        if !incoming.architecture.is_empty() {
            next.architecture = incoming.architecture.clone();
        }
        if incoming.progress_ratio > 0.0 {
            next.progress_ratio = next.progress_ratio.max(incoming.progress_ratio);
        }
        if !incoming.training_mode.is_empty() {
            next.training_mode = incoming.training_mode.clone();
        }
        if incoming.benchmark_id.is_some() {
            next.benchmark_id = incoming.benchmark_id.clone();
        }
        if !incoming.gate_status.is_empty() {
            next.gate_status = incoming.gate_status.clone();
        }
        if incoming.certified_best_metric.is_finite() {
            next.certified_best_metric =
                next.certified_best_metric.min(incoming.certified_best_metric);
        }
        if incoming.dominant_blocker.is_some() {
            next.dominant_blocker = incoming.dominant_blocker.clone();
        }
        if incoming.stalled_reason.is_some() {
            next.stalled_reason = incoming.stalled_reason.clone();
        }
        next.target_band_low = incoming.target_band_low;
        next.target_band_high = incoming.target_band_high;
        next.trend_slope = incoming.trend_slope;
        next.trend_variance = incoming.trend_variance;
        next.pino = incoming.pino.clone().or_else(|| current.pino.clone());
        next
    } else {
        incoming.clone()
    };
    if merged.network.layer_sizes.is_empty() && !current.network.layer_sizes.is_empty() {
        merged.network = current.network.clone();
    }
    if merged.pino.is_none() {
        merged.pino = current.pino.clone();
    }
    merged
}

fn update_epochs_since_improvement(
    best_val: &mut f64,
    since_improve: &mut usize,
    last_epoch_counted: &mut usize,
    epoch: usize,
    metric: f64,
) {
    if metric.is_finite() && metric > 0.0 && metric + 1e-12 < *best_val {
        *best_val = metric;
        *since_improve = 0;
    } else if epoch > 0 && epoch > *last_epoch_counted {
        *since_improve = since_improve.saturating_add(epoch - *last_epoch_counted);
    }

    if epoch > 0 {
        *last_epoch_counted = (*last_epoch_counted).max(epoch);
    }
}

#[tauri::command(rename_all = "camelCase")]
fn solve_fem_case(input: SolveInput) -> Result<FemResult, String> {
    fem::try_solve_case(&input)
}

#[tauri::command(rename_all = "camelCase")]
fn run_thermal_case(input: ThermalInput) -> Result<ThermalResult, String> {
    let fem = fem::try_solve_case(&input.solve_input)?;
    Ok(physics::thermal::run(&input, fem.stress_tensor))
}

#[tauri::command(rename_all = "camelCase")]
fn run_dynamic_case(input: DynamicInput) -> Result<DynamicResult, String> {
    physics::dynamic::run(&input)
}

#[tauri::command(rename_all = "camelCase")]
fn evaluate_failure(input: FailureInput) -> Result<FailureResult, String> {
    Ok(physics::failure::evaluate(&input))
}

#[tauri::command(rename_all = "camelCase")]
fn start_ann_training(
    batch: TrainingBatch,
    state: State<'_, AppState>,
    app: tauri::AppHandle,
) -> Result<bool, String> {
    let batch = apply_training_benchmark(batch)?;
    batch.validate()?;
    if let Some(benchmark_id) = batch.benchmark_id.as_deref() {
        if get_training_benchmark(benchmark_id).is_none() {
            return Err(format!(
                "Unknown benchmarkId '{benchmark_id}'. Call listTrainingBenchmarks for supported benchmark profiles."
            ));
        }
    }
    let training_mode = resolve_training_mode(&batch);
    let benchmark_id = batch.benchmark_id.clone();
    let init_total = batch.max_total_epochs.unwrap_or(batch.epochs.max(1));
    {
        let mut status = state
            .training_status
            .lock()
            .map_err(|_| "Training status lock poisoned".to_string())?;
        if status.running {
            return Ok(false);
        }
        *status = TrainingRunStatus {
            running: true,
            stop_requested: false,
            completed: false,
            last_result: None,
            last_error: None,
            diagnostics: default_diagnostics(
                batch.learning_rate.unwrap_or(5e-4),
                vec![],
                training_mode.clone(),
                benchmark_id.clone(),
                init_total,
            ),
        };
    }

    if let Ok(mut tick) = state.training_tick.lock() {
        *tick = TrainingTickEvent {
            epoch: 0,
            total_epochs: init_total,
            loss: 0.0,
            val_loss: 0.0,
            learning_rate: batch.learning_rate.unwrap_or(5e-4),
            architecture: vec![],
            progress_ratio: 0.0,
        };
    }
    if let Ok(mut progress) = state.training_progress.lock() {
        *progress = TrainingProgressEvent {
            epoch: 0,
            total_epochs: init_total,
            loss: 0.0,
            val_loss: 0.0,
            data_loss: 0.0,
            physics_loss: 0.0,
            val_data_loss: 0.0,
            val_physics_loss: 0.0,
            momentum_residual: 0.0,
            kinematic_residual: 0.0,
            material_residual: 0.0,
            boundary_residual: 0.0,
            displacement_fit: 0.0,
            stress_fit: 0.0,
            invariant_residual: 0.0,
            constitutive_normal_residual: 0.0,
            constitutive_shear_residual: 0.0,
            val_displacement_fit: 0.0,
            val_stress_fit: 0.0,
            val_invariant_residual: 0.0,
            val_constitutive_normal_residual: 0.0,
            val_constitutive_shear_residual: 0.0,
            hybrid_mode: "booting".to_string(),
            stage_id: "preflight".to_string(),
            optimizer_id: "pino-adam".to_string(),
            lr_phase: "queued".to_string(),
            target_band_low: 0.0,
            target_band_high: 0.0,
            trend_slope: 0.0,
            trend_variance: 0.0,
            watchdog_trigger_count: 0,
            collocation_samples_added: 0,
            train_data_size: 0,
            train_data_cap: 0,
            residual_weight_momentum: 1.0,
            residual_weight_kinematics: 1.0,
            residual_weight_material: 1.0,
            residual_weight_boundary: 1.0,
            learning_rate: batch.learning_rate.unwrap_or(5e-4),
            architecture: vec![],
            progress_ratio: 0.0,
            training_mode: training_mode.clone(),
            benchmark_id: benchmark_id.clone(),
            gate_status: "queued".to_string(),
            certified_best_metric: f64::MAX,
            dominant_blocker: None,
            stalled_reason: None,
            network: NetworkSnapshot {
                layer_sizes: vec![],
                nodes: vec![],
                connections: vec![],
            },
            pino: None,
        };
    }
    state.training_stop.store(false, Ordering::Relaxed);

    let model = state.model.clone();
    let tick_state = state.training_tick.clone();
    let progress_state = state.training_progress.clone();
    let status_state = state.training_status.clone();
    let stop_flag = state.training_stop.clone();
    let app_handle = app.clone();
    let batch_owned = batch.clone();
    let training_mode_owned = training_mode.clone();
    let benchmark_id_owned = benchmark_id.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let sanitize = |v: f64| if v.is_finite() { v } else { 0.0 };
        let mut best_val = f64::MAX;
        let mut since_improve = 0usize;
        let mut last_epoch_counted = 0usize;
        let mut last_lr = batch_owned.learning_rate.unwrap_or(5e-4);
        let mut last_stage = "idle".to_string();
        let mut last_optimizer = "adamw".to_string();
        let mut last_lr_phase = "idle".to_string();
        let mut last_watchdog_count = 0usize;
        let mut recent_events: Vec<String> = vec![];
        let run_result =
            (|| -> Result<(TrainResult, pinn::UniversalPinnState, ModelStatus), String> {
                let mut model = model
                    .lock()
                    .map_err(|_| "ANN state lock poisoned".to_string())?;
                let result = model.train_with_progress_with_checkpoint(
                    &batch_owned,
                    |progress| {
                        let incoming_tick = TrainingTickEvent {
                            epoch: progress.epoch,
                            total_epochs: progress.total_epochs,
                            loss: sanitize(progress.loss),
                            val_loss: sanitize(progress.val_loss),
                            learning_rate: sanitize(progress.learning_rate),
                            architecture: progress.architecture.clone(),
                            progress_ratio: sanitize(progress.progress_ratio).clamp(0.0, 1.0),
                        };
                        if let Ok(mut shared_tick) = tick_state.lock() {
                            *shared_tick = merge_training_tick(&shared_tick, &incoming_tick);
                        }
                        let safe_progress = TrainingProgressEvent {
                            loss: sanitize(progress.loss),
                            val_loss: sanitize(progress.val_loss),
                            data_loss: sanitize(progress.data_loss),
                            physics_loss: sanitize(progress.physics_loss),
                            val_data_loss: sanitize(progress.val_data_loss),
                            val_physics_loss: sanitize(progress.val_physics_loss),
                            target_band_low: sanitize(progress.target_band_low),
                            target_band_high: sanitize(progress.target_band_high),
                            trend_slope: sanitize(progress.trend_slope),
                            trend_variance: sanitize(progress.trend_variance),
                            learning_rate: sanitize(progress.learning_rate),
                            progress_ratio: sanitize(progress.progress_ratio).clamp(0.0, 1.0),
                            training_mode: training_mode_owned.clone(),
                            benchmark_id: benchmark_id_owned.clone(),
                            gate_status: "running".to_string(),
                            certified_best_metric: if progress.certified_best_metric.is_finite()
                                && progress.certified_best_metric > 0.0
                            {
                                progress.certified_best_metric
                            } else {
                                f64::MAX
                            },
                            dominant_blocker: dominant_blocker_from_progress(&progress),
                            stalled_reason: None,
                            ..progress
                        };
                        let safe_arch = safe_progress.architecture.clone();
                        if let Ok(mut shared_progress) = progress_state.lock() {
                            *shared_progress =
                                merge_training_progress(&shared_progress, &safe_progress);
                        }
                        let v = sanitize(progress.val_loss);
                        let live_metric = safe_progress.certified_best_metric.min(if v > 0.0 {
                            v
                        } else {
                            f64::MAX
                        });
                        update_epochs_since_improvement(
                            &mut best_val,
                            &mut since_improve,
                            &mut last_epoch_counted,
                            safe_progress.epoch,
                            live_metric,
                        );
                        let lr = sanitize(progress.learning_rate);
                        last_lr = lr;
                        if let Ok(mut status) = status_state.lock() {
                            if safe_progress.stage_id != last_stage {
                                push_recent_event(
                                    &mut recent_events,
                                    format!(
                                        "e{} stage {} -> {}",
                                        safe_progress.epoch, last_stage, safe_progress.stage_id
                                    ),
                                );
                                last_stage = safe_progress.stage_id.clone();
                            }
                            if safe_progress.optimizer_id != last_optimizer {
                                push_recent_event(
                                    &mut recent_events,
                                    format!(
                                        "e{} optimizer {} -> {}",
                                        safe_progress.epoch,
                                        last_optimizer,
                                        safe_progress.optimizer_id
                                    ),
                                );
                                last_optimizer = safe_progress.optimizer_id.clone();
                            }
                            if safe_progress.lr_phase != last_lr_phase {
                                push_recent_event(
                                    &mut recent_events,
                                    format!(
                                        "e{} lr-phase {} -> {}",
                                        safe_progress.epoch, last_lr_phase, safe_progress.lr_phase
                                    ),
                                );
                                last_lr_phase = safe_progress.lr_phase.clone();
                            }
                            if safe_progress.watchdog_trigger_count > last_watchdog_count {
                                push_recent_event(
                                    &mut recent_events,
                                    format!(
                                        "e{} watchdog trigger +{} (total {})",
                                        safe_progress.epoch,
                                        safe_progress.watchdog_trigger_count - last_watchdog_count,
                                        safe_progress.watchdog_trigger_count
                                    ),
                                );
                                last_watchdog_count = safe_progress.watchdog_trigger_count;
                            }
                            status.running = true;
                            status.diagnostics.best_val_loss = best_val;
                            status.diagnostics.epochs_since_improvement = since_improve;
                            status.diagnostics.current_learning_rate = lr;
                            status.diagnostics.lr_schedule_phase = safe_progress.lr_phase.clone();
                            status.diagnostics.active_stage = safe_progress.stage_id.clone();
                            status.diagnostics.active_optimizer =
                                safe_progress.optimizer_id.clone();
                            status.diagnostics.target_floor_estimate =
                                safe_progress.target_band_low;
                            status.diagnostics.bo_selected_architecture = safe_arch;
                            status.diagnostics.residual_weight_momentum =
                                safe_progress.residual_weight_momentum;
                            status.diagnostics.residual_weight_kinematics =
                                safe_progress.residual_weight_kinematics;
                            status.diagnostics.residual_weight_material =
                                safe_progress.residual_weight_material;
                            status.diagnostics.residual_weight_boundary =
                                safe_progress.residual_weight_boundary;
                            status.diagnostics.data_weight = safe_progress.residual_weight_momentum
                                + safe_progress.residual_weight_kinematics;
                            status.diagnostics.physics_weight = safe_progress
                                .residual_weight_material
                                + safe_progress.residual_weight_boundary;
                            status.diagnostics.momentum_residual = safe_progress.momentum_residual;
                            status.diagnostics.kinematic_residual =
                                safe_progress.kinematic_residual;
                            status.diagnostics.material_residual = safe_progress.material_residual;
                            status.diagnostics.boundary_residual = safe_progress.boundary_residual;
                            status.diagnostics.displacement_fit = safe_progress.displacement_fit;
                            status.diagnostics.stress_fit = safe_progress.stress_fit;
                            status.diagnostics.invariant_residual =
                                safe_progress.invariant_residual;
                            status.diagnostics.constitutive_normal_residual =
                                safe_progress.constitutive_normal_residual;
                            status.diagnostics.constitutive_shear_residual =
                                safe_progress.constitutive_shear_residual;
                            status.diagnostics.val_displacement_fit =
                                safe_progress.val_displacement_fit;
                            status.diagnostics.val_stress_fit = safe_progress.val_stress_fit;
                            status.diagnostics.val_invariant_residual =
                                safe_progress.val_invariant_residual;
                            status.diagnostics.val_constitutive_normal_residual =
                                safe_progress.val_constitutive_normal_residual;
                            status.diagnostics.val_constitutive_shear_residual =
                                safe_progress.val_constitutive_shear_residual;
                            status.diagnostics.hybrid_mode = safe_progress.hybrid_mode.clone();
                            status.diagnostics.collocation_points =
                                batch_owned.collocation_points.unwrap_or(4096);
                            status.diagnostics.boundary_points =
                                batch_owned.boundary_points.unwrap_or(1024);
                            status.diagnostics.interface_points =
                                batch_owned.interface_points.unwrap_or(512);
                            status.diagnostics.collocation_samples_added =
                                safe_progress.collocation_samples_added;
                            status.diagnostics.train_data_size = safe_progress.train_data_size;
                            status.diagnostics.train_data_cap = safe_progress.train_data_cap;
                            status.diagnostics.training_mode = training_mode_owned.clone();
                            status.diagnostics.benchmark_id = benchmark_id_owned.clone();
                            status.diagnostics.gate_status = "running".to_string();
                            status.diagnostics.certified_best_metric = best_val;
                            status.diagnostics.dominant_blocker =
                                dominant_blocker_from_progress(&safe_progress);
                            status.diagnostics.stalled_reason = None;
                            status.diagnostics.benchmark_certification = None;
                            status.diagnostics.run_budget_used = safe_progress.epoch;
                            status.diagnostics.run_budget_total = safe_progress.total_epochs;
                            status.diagnostics.recent_events = recent_events.clone();
                        }
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    },
                    || stop_flag.load(Ordering::Relaxed),
                    |epoch, snapshot, is_best| {
                        if batch_owned.checkpoint_every_epochs.unwrap_or(0) == 0 {
                            return;
                        }
                        let status = model_status_from_snapshot(&snapshot);
                        let _ = io::save_training_checkpoint(
                            Some(if is_best {
                                format!("best-e{epoch}")
                            } else {
                                format!("auto-e{epoch}")
                            }),
                            &snapshot,
                            &status,
                            epoch,
                            is_best,
                        );
                        let _ = io::purge_training_checkpoints(&CheckpointRetentionPolicy {
                            keep_last: batch_owned.checkpoint_retention.unwrap_or(8).max(1),
                            keep_best: 2,
                        });
                    },
                );
                let snapshot = model.snapshot_state();
                let status = model.status();
                Ok((result, snapshot, status))
            })();

        match run_result {
            Ok((result, snapshot, model_status)) => {
                let final_progress = progress_state.lock().ok().map(|p| p.clone());
                let final_gate_status = resolve_gate_status(
                    false,
                    true,
                    result.reached_target_loss,
                    &result.stop_reason,
                );
                let final_dominant_blocker = final_progress
                    .as_ref()
                    .and_then(dominant_blocker_from_progress)
                    .or_else(|| result.dominant_blocker.clone());
                let final_stalled_reason = if final_gate_status == "stalled" {
                    Some(result.stop_reason.clone())
                } else {
                    None
                };
                let enriched_result = TrainResult {
                    training_mode: Some(training_mode_owned.clone()),
                    benchmark_id: benchmark_id_owned.clone(),
                    gate_status: Some(final_gate_status.clone()),
                    certified_best_metric: Some(best_val.min(result.val_loss)),
                    reproducibility_spread: None,
                    dominant_blocker: final_dominant_blocker.clone(),
                    stalled_reason: final_stalled_reason.clone(),
                    ..result.clone()
                };
                if let Ok(mut status) = status_state.lock() {
                    *status = TrainingRunStatus {
                        running: false,
                        stop_requested: false,
                        completed: true,
                        last_result: Some(enriched_result.clone()),
                        last_error: None,
                        diagnostics: TrainingDiagnostics {
                            best_val_loss: best_val,
                            epochs_since_improvement: since_improve,
                            lr_schedule_phase: result.stop_reason.clone(),
                            current_learning_rate: result.learning_rate,
                            data_weight: extract_metric(&result.notes, "momentumWeight=")
                                .zip(extract_metric(&result.notes, "kinematicsWeight="))
                                .map(|(m, k)| m + k)
                                .or_else(|| extract_metric(&result.notes, "dataW="))
                                .unwrap_or(2.0),
                            physics_weight: extract_metric(&result.notes, "materialWeight=")
                                .zip(extract_metric(&result.notes, "boundaryWeight="))
                                .map(|(m, b)| m + b)
                                .or_else(|| extract_metric(&result.notes, "physW="))
                                .unwrap_or(2.0),
                            residual_weight_momentum: extract_metric(
                                &result.notes,
                                "momentumWeight=",
                            )
                            .unwrap_or(1.0),
                            residual_weight_kinematics: extract_metric(
                                &result.notes,
                                "kinematicsWeight=",
                            )
                            .unwrap_or(1.0),
                            residual_weight_material: extract_metric(
                                &result.notes,
                                "materialWeight=",
                            )
                            .unwrap_or(1.0),
                            residual_weight_boundary: extract_metric(
                                &result.notes,
                                "boundaryWeight=",
                            )
                            .unwrap_or(1.0),
                            active_learning_rounds: extract_metric(&result.notes, "AL rounds=")
                                .unwrap_or(0.0)
                                as usize,
                            active_learning_samples_added: extract_metric(
                                &result.notes,
                                "AL samples=",
                            )
                            .unwrap_or(0.0)
                                as usize,
                            safeguard_triggers: extract_metric(&result.notes, "safeguards=")
                                .unwrap_or(0.0)
                                as usize,
                            curriculum_backoffs: extract_metric(
                                &result.notes,
                                "curriculumBackoffs=",
                            )
                            .unwrap_or(0.0)
                                as usize,
                            optimizer_switches: extract_metric(&result.notes, "optimizerSwitches=")
                                .unwrap_or(0.0)
                                as usize,
                            checkpoint_rollbacks: extract_metric(
                                &result.notes,
                                "checkpointRollbacks=",
                            )
                            .unwrap_or(0.0)
                                as usize,
                            target_floor_estimate: extract_metric(
                                &result.notes,
                                "targetFloorEstimate=",
                            )
                            .unwrap_or(best_val),
                            trend_stop_reason: result.stop_reason.clone(),
                            active_stage: final_progress
                                .as_ref()
                                .map(|p| p.stage_id.clone())
                                .unwrap_or_else(|| parse_stage_id_from_notes(&result.notes)),
                            active_optimizer: final_progress
                                .as_ref()
                                .map(|p| p.optimizer_id.clone())
                                .unwrap_or_else(|| parse_optimizer_id_from_notes(&result.notes)),
                            bo_presearch_used: result
                                .notes
                                .iter()
                                .any(|n| n.contains("boUsed=true")),
                            bo_selected_architecture: result.architecture.clone(),
                            momentum_residual: extract_metric(&result.notes, "momentumResidual=")
                                .unwrap_or(0.0),
                            kinematic_residual: extract_metric(&result.notes, "kinematicResidual=")
                                .unwrap_or(0.0),
                            material_residual: extract_metric(&result.notes, "materialResidual=")
                                .unwrap_or(0.0),
                            boundary_residual: extract_metric(&result.notes, "boundaryResidual=")
                                .unwrap_or(0.0),
                            displacement_fit: final_progress
                                .as_ref()
                                .map(|p| p.displacement_fit)
                                .unwrap_or_else(|| {
                                    extract_metric(&result.notes, "dispFit=").unwrap_or(0.0)
                                }),
                            stress_fit: final_progress
                                .as_ref()
                                .map(|p| p.stress_fit)
                                .unwrap_or_else(|| {
                                    extract_metric(&result.notes, "stressFit=").unwrap_or(0.0)
                                }),
                            invariant_residual: final_progress
                                .as_ref()
                                .map(|p| p.invariant_residual)
                                .unwrap_or_else(|| {
                                    extract_metric(&result.notes, "invResidual=")
                                        .unwrap_or(0.0)
                                }),
                            constitutive_normal_residual: final_progress
                                .as_ref()
                                .map(|p| p.constitutive_normal_residual)
                                .unwrap_or_else(|| {
                                    extract_metric(&result.notes, "constitutiveNormal=")
                                        .unwrap_or(0.0)
                                }),
                            constitutive_shear_residual: final_progress
                                .as_ref()
                                .map(|p| p.constitutive_shear_residual)
                                .unwrap_or_else(|| {
                                    extract_metric(&result.notes, "constitutiveShear=")
                                        .unwrap_or(0.0)
                                }),
                            val_displacement_fit: final_progress
                                .as_ref()
                                .map(|p| p.val_displacement_fit)
                                .unwrap_or_else(|| {
                                    extract_metric(&result.notes, "valDispFit=").unwrap_or(0.0)
                                }),
                            val_stress_fit: final_progress
                                .as_ref()
                                .map(|p| p.val_stress_fit)
                                .unwrap_or_else(|| {
                                    extract_metric(&result.notes, "valStressFit=").unwrap_or(0.0)
                                }),
                            val_invariant_residual: final_progress
                                .as_ref()
                                .map(|p| p.val_invariant_residual)
                                .unwrap_or_else(|| {
                                    extract_metric(&result.notes, "valInvResidual=")
                                        .unwrap_or(0.0)
                                }),
                            val_constitutive_normal_residual: final_progress
                                .as_ref()
                                .map(|p| p.val_constitutive_normal_residual)
                                .unwrap_or_else(|| {
                                    extract_metric(&result.notes, "valConstitutiveNormal=")
                                        .unwrap_or(0.0)
                                }),
                            val_constitutive_shear_residual: final_progress
                                .as_ref()
                                .map(|p| p.val_constitutive_shear_residual)
                                .unwrap_or_else(|| {
                                    extract_metric(&result.notes, "valConstitutiveShear=")
                                        .unwrap_or(0.0)
                                }),
                            hybrid_mode: result
                                .notes
                                .iter()
                                .find_map(|n| n.split("hybridMode=").nth(1))
                                .map(|s| s.split(',').next().unwrap_or("hybrid").to_string())
                                .unwrap_or_else(|| "hybrid".to_string()),
                            collocation_points: batch_owned.collocation_points.unwrap_or(4096),
                            boundary_points: batch_owned.boundary_points.unwrap_or(1024),
                            interface_points: batch_owned.interface_points.unwrap_or(512),
                            collocation_samples_added: extract_metric(
                                &result.notes,
                                "CollocationSamplesAdded:",
                            )
                            .unwrap_or(0.0)
                                as usize,
                            train_data_cap: extract_metric(&result.notes, "TrainDataCap:")
                                .unwrap_or(0.0)
                                as usize,
                            train_data_size: extract_metric(&result.notes, "TrainDataFinal:")
                                .unwrap_or(0.0)
                                as usize,
                            training_mode: training_mode_owned.clone(),
                            benchmark_id: benchmark_id_owned.clone(),
                            gate_status: final_gate_status.clone(),
                            certified_best_metric: best_val.min(enriched_result.val_loss),
                            reproducibility_spread: None,
                            dominant_blocker: final_dominant_blocker.clone(),
                            stalled_reason: final_stalled_reason.clone(),
                            benchmark_certification: enriched_result.benchmark_certification.clone(),
                            run_budget_used: enriched_result.completed_epochs,
                            run_budget_total: batch_owned
                                .max_total_epochs
                                .unwrap_or(batch_owned.epochs.max(1)),
                            recent_events: recent_events.clone(),
                            pino: result.pino.clone(),
                        },
                    };
                }
                let _ = io::save_training_checkpoint(
                    Some(format!("auto-{}", result.stop_reason)),
                    &snapshot,
                    &model_status,
                    result.completed_epochs,
                    true,
                );
                let _ = io::purge_training_checkpoints(&CheckpointRetentionPolicy {
                    keep_last: batch_owned.checkpoint_retention.unwrap_or(8).max(1),
                    keep_best: 2,
                });
                let _ = app_handle.emit("ann-training-complete", &enriched_result);
            }
            Err(err) => {
                if let Ok(mut status) = status_state.lock() {
                    *status = TrainingRunStatus {
                        running: false,
                        stop_requested: false,
                        completed: true,
                        last_result: None,
                        last_error: Some(err.clone()),
                        diagnostics: default_diagnostics(
                            last_lr,
                            vec![],
                            training_mode_owned.clone(),
                            benchmark_id_owned.clone(),
                            batch_owned.max_total_epochs.unwrap_or(batch_owned.epochs.max(1)),
                        ),
                    };
                    status.diagnostics.gate_status = "failed".to_string();
                    status.diagnostics.stalled_reason = Some(err.clone());
                    status.diagnostics.benchmark_certification = None;
                }
                let _ = app_handle.emit("ann-training-error", err);
            }
        }
    });

    Ok(true)
}

#[tauri::command(rename_all = "camelCase")]
fn stop_ann_training(state: State<'_, AppState>) -> Result<bool, String> {
    state.training_stop.store(true, Ordering::Relaxed);
    let mut status = state
        .training_status
        .lock()
        .map_err(|_| "Training status lock poisoned".to_string())?;
    if status.running {
        status.stop_requested = true;
        Ok(true)
    } else {
        Ok(false)
    }
}

#[tauri::command(rename_all = "camelCase")]
fn list_training_benchmarks_command() -> Result<Vec<TrainingBenchmarkManifest>, String> {
    Ok(list_training_benchmarks())
}

#[tauri::command(rename_all = "camelCase")]
fn get_training_status(state: State<'_, AppState>) -> Result<TrainingRunStatus, String> {
    let status = state
        .training_status
        .lock()
        .map_err(|_| "Training status lock poisoned".to_string())?;
    Ok(status.clone())
}

#[tauri::command(rename_all = "camelCase")]
fn train_ann(
    batch: TrainingBatch,
    state: State<'_, AppState>,
    app: tauri::AppHandle,
) -> Result<TrainResult, String> {
    let started = start_ann_training(batch, state.clone(), app)?;
    if !started {
        return Err("Training is already running".to_string());
    }
    loop {
        {
            let status = state
                .training_status
                .lock()
                .map_err(|_| "Training status lock poisoned".to_string())?;
            if !status.running {
                if let Some(err) = &status.last_error {
                    return Err(err.clone());
                }
                if let Some(result) = &status.last_result {
                    return Ok(result.clone());
                }
                return Err("Training finished without a result".to_string());
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

#[tauri::command(rename_all = "camelCase")]
fn get_training_tick(state: State<'_, AppState>) -> Result<TrainingTickEvent, String> {
    let tick = state
        .training_tick
        .lock()
        .map_err(|_| "Training tick state lock poisoned".to_string())?;
    Ok(tick.clone())
}

#[tauri::command(rename_all = "camelCase")]
fn get_training_progress(state: State<'_, AppState>) -> Result<TrainingProgressEvent, String> {
    let progress = state
        .training_progress
        .lock()
        .map_err(|_| "Training progress state lock poisoned".to_string())?;
    Ok(progress.clone())
}

#[tauri::command(rename_all = "camelCase")]
async fn infer_ann(input: SolveInput, state: State<'_, AppState>) -> Result<AnnResult, String> {
    input.validate()?;
    let model = state
        .model
        .lock()
        .map_err(|_| "ANN state lock poisoned".to_string())?
        .clone();
    tauri::async_runtime::spawn_blocking(move || model.infer(&input))
        .await
        .map_err(|e| format!("ANN inference task failed: {e}"))
}

#[tauri::command(rename_all = "camelCase")]
fn get_model_status(state: State<'_, AppState>) -> Result<ModelStatus, String> {
    let model = state
        .model
        .lock()
        .map_err(|_| "ANN state lock poisoned".to_string())?;
    Ok(model.status())
}

#[tauri::command(rename_all = "camelCase")]
fn reset_ann_model(seed: Option<u64>, state: State<'_, AppState>) -> Result<ModelStatus, String> {
    {
        let mut model = state
            .model
            .lock()
            .map_err(|_| "ANN state lock poisoned".to_string())?;
        model.reset(seed);
    }
    if let Ok(mut tick) = state.training_tick.lock() {
        *tick = TrainingTickEvent {
            epoch: 0,
            total_epochs: 0,
            loss: 0.0,
            val_loss: 0.0,
            learning_rate: 0.0,
            architecture: vec![],
            progress_ratio: 0.0,
        };
    }
    if let Ok(mut progress) = state.training_progress.lock() {
        *progress = TrainingProgressEvent {
            epoch: 0,
            total_epochs: 0,
            loss: 0.0,
            val_loss: 0.0,
            data_loss: 0.0,
            physics_loss: 0.0,
            val_data_loss: 0.0,
            val_physics_loss: 0.0,
            momentum_residual: 0.0,
            kinematic_residual: 0.0,
            material_residual: 0.0,
            boundary_residual: 0.0,
            displacement_fit: 0.0,
            stress_fit: 0.0,
            invariant_residual: 0.0,
            constitutive_normal_residual: 0.0,
            constitutive_shear_residual: 0.0,
            val_displacement_fit: 0.0,
            val_stress_fit: 0.0,
            val_invariant_residual: 0.0,
            val_constitutive_normal_residual: 0.0,
            val_constitutive_shear_residual: 0.0,
            hybrid_mode: "hybrid".to_string(),
            stage_id: "idle".to_string(),
            optimizer_id: "pino-adam".to_string(),
            lr_phase: "idle".to_string(),
            target_band_low: 0.0,
            target_band_high: 0.0,
            trend_slope: 0.0,
            trend_variance: 0.0,
            watchdog_trigger_count: 0,
            collocation_samples_added: 0,
            train_data_size: 0,
            train_data_cap: 0,
            residual_weight_momentum: 1.0,
            residual_weight_kinematics: 1.0,
            residual_weight_material: 1.0,
            residual_weight_boundary: 1.0,
            learning_rate: 0.0,
            architecture: vec![],
            progress_ratio: 0.0,
            training_mode: "legacy-mixed-exact".to_string(),
            benchmark_id: None,
            gate_status: "queued".to_string(),
            certified_best_metric: f64::MAX,
            dominant_blocker: None,
            stalled_reason: None,
            network: NetworkSnapshot {
                layer_sizes: vec![],
                nodes: vec![],
                connections: vec![],
            },
            pino: None,
        };
    }
    if let Ok(mut status) = state.training_status.lock() {
        *status = TrainingRunStatus {
            running: false,
            stop_requested: false,
            completed: false,
            last_result: None,
            last_error: None,
            diagnostics: default_diagnostics(
                5e-4,
                vec![],
                "legacy-mixed-exact".to_string(),
                None,
                0,
            ),
        };
    }
    let model = state
        .model
        .lock()
        .map_err(|_| "ANN state lock poisoned".to_string())?;
    Ok(model.status())
}

#[tauri::command(rename_all = "camelCase")]
fn set_safeguard_settings(
    settings: SafeguardSettings,
    state: State<'_, AppState>,
) -> Result<ModelStatus, String> {
    let mut model = state
        .model
        .lock()
        .map_err(|_| "ANN state lock poisoned".to_string())?;
    model.set_safeguard_settings(settings);
    Ok(model.status())
}

#[tauri::command(rename_all = "camelCase")]
fn save_training_checkpoint(
    input: CheckpointSaveInput,
    state: State<'_, AppState>,
) -> Result<TrainingCheckpointInfo, String> {
    let model = state
        .model
        .lock()
        .map_err(|_| "ANN state lock poisoned".to_string())?;
    let tick = state
        .training_tick
        .lock()
        .map_err(|_| "Training tick state lock poisoned".to_string())?;
    let snapshot = model.snapshot_state();
    let model_status = model.status();
    io::save_training_checkpoint(
        input.tag,
        &snapshot,
        &model_status,
        tick.epoch,
        input.mark_best.unwrap_or(false),
    )
}

#[tauri::command(rename_all = "camelCase")]
fn list_training_checkpoints() -> Result<Vec<TrainingCheckpointInfo>, String> {
    io::list_training_checkpoints()
}

#[tauri::command(rename_all = "camelCase")]
fn resume_training_from_checkpoint(
    id: String,
    state: State<'_, AppState>,
) -> Result<ResumeTrainingResult, String> {
    let (checkpoint, model_state, model_status) = io::resume_training_from_checkpoint(&id)?;
    let mut model = state
        .model
        .lock()
        .map_err(|_| "ANN state lock poisoned".to_string())?;
    model.load_state(model_state);
    Ok(ResumeTrainingResult {
        checkpoint,
        model_status,
    })
}

#[tauri::command(rename_all = "camelCase")]
fn purge_training_checkpoints(
    retention_policy: CheckpointRetentionPolicy,
) -> Result<PurgeCheckpointsResult, String> {
    io::purge_training_checkpoints(&retention_policy)
}

#[tauri::command(rename_all = "camelCase")]
fn export_report(input: ReportInput) -> Result<ExportResult, String> {
    io::export_report(&input)
}

#[tauri::command(rename_all = "camelCase")]
fn get_runtime_fingerprint() -> Result<RuntimeFingerprint, String> {
    let profile = if cfg!(debug_assertions) {
        "debug".to_string()
    } else {
        "release".to_string()
    };
    Ok(RuntimeFingerprint {
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        build_profile: profile,
        target_os: std::env::consts::OS.to_string(),
        target_arch: std::env::consts::ARCH.to_string(),
        debug_build: cfg!(debug_assertions),
        git_commit: option_env!("GIT_COMMIT_HASH")
            .unwrap_or("unknown")
            .to_string(),
        build_time_utc: option_env!("BUILD_TIME_UTC")
            .unwrap_or("unknown")
            .to_string(),
    })
}

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            model: Arc::new(Mutex::new(UniversalPinnEngine::default())),
            training_tick: Arc::new(Mutex::new(TrainingTickEvent {
                epoch: 0,
                total_epochs: 0,
                loss: 0.0,
                val_loss: 0.0,
                learning_rate: 0.0,
                architecture: vec![],
                progress_ratio: 0.0,
            })),
            training_progress: Arc::new(Mutex::new(TrainingProgressEvent {
                epoch: 0,
                total_epochs: 0,
                loss: 0.0,
                val_loss: 0.0,
                data_loss: 0.0,
                physics_loss: 0.0,
                val_data_loss: 0.0,
                val_physics_loss: 0.0,
                momentum_residual: 0.0,
                kinematic_residual: 0.0,
                material_residual: 0.0,
                boundary_residual: 0.0,
                displacement_fit: 0.0,
                stress_fit: 0.0,
                invariant_residual: 0.0,
                constitutive_normal_residual: 0.0,
                constitutive_shear_residual: 0.0,
                val_displacement_fit: 0.0,
                val_stress_fit: 0.0,
                val_invariant_residual: 0.0,
                val_constitutive_normal_residual: 0.0,
                val_constitutive_shear_residual: 0.0,
                hybrid_mode: "hybrid".to_string(),
                stage_id: "idle".to_string(),
                optimizer_id: "pino-adam".to_string(),
                lr_phase: "idle".to_string(),
                target_band_low: 0.0,
                target_band_high: 0.0,
                trend_slope: 0.0,
                trend_variance: 0.0,
                watchdog_trigger_count: 0,
                collocation_samples_added: 0,
                train_data_size: 0,
                train_data_cap: 0,
                residual_weight_momentum: 1.0,
                residual_weight_kinematics: 1.0,
                residual_weight_material: 1.0,
                residual_weight_boundary: 1.0,
                learning_rate: 0.0,
                architecture: vec![],
                progress_ratio: 0.0,
                training_mode: "legacy-mixed-exact".to_string(),
                benchmark_id: None,
                gate_status: "queued".to_string(),
                certified_best_metric: f64::MAX,
                dominant_blocker: None,
                stalled_reason: None,
                network: NetworkSnapshot {
                    layer_sizes: vec![],
                    nodes: vec![],
                    connections: vec![],
                },
                pino: None,
            })),
            training_status: Arc::new(Mutex::new(TrainingRunStatus {
                running: false,
                stop_requested: false,
                completed: false,
                last_result: None,
                last_error: None,
                diagnostics: default_diagnostics(
                    5e-4,
                    vec![],
                    "legacy-mixed-exact".to_string(),
                    None,
                    0,
                ),
            })),
            training_stop: Arc::new(AtomicBool::new(false)),
        })
        .invoke_handler(tauri::generate_handler![
            solve_fem_case,
            train_ann,
            start_ann_training,
            stop_ann_training,
            list_training_benchmarks_command,
            get_training_status,
            infer_ann,
            run_dynamic_case,
            run_thermal_case,
            evaluate_failure,
            get_model_status,
            reset_ann_model,
            set_safeguard_settings,
            save_training_checkpoint,
            list_training_checkpoints,
            resume_training_from_checkpoint,
            purge_training_checkpoints,
            get_training_tick,
            get_training_progress,
            get_runtime_fingerprint,
            export_report
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tick(epoch: usize, loss: f64, val_loss: f64) -> TrainingTickEvent {
        TrainingTickEvent {
            epoch,
            total_epochs: 10_000,
            loss,
            val_loss,
            learning_rate: 5e-4,
            architecture: vec![15, 48, 48, 48, 11],
            progress_ratio: epoch as f64 / 10_000.0,
        }
    }

    fn sample_progress(epoch: usize, lr_phase: &str) -> TrainingProgressEvent {
        TrainingProgressEvent {
            epoch,
            total_epochs: 10_000,
            loss: if epoch > 0 { 0.25 } else { 0.0 },
            val_loss: if epoch > 0 { 0.30 } else { 0.0 },
            data_loss: if epoch > 0 { 0.11 } else { 0.0 },
            physics_loss: if epoch > 0 { 0.14 } else { 0.0 },
            val_data_loss: if epoch > 0 { 0.12 } else { 0.0 },
            val_physics_loss: if epoch > 0 { 0.18 } else { 0.0 },
            momentum_residual: 0.0,
            kinematic_residual: 0.0,
            material_residual: 0.0,
            boundary_residual: 0.0,
            displacement_fit: 0.0,
            stress_fit: 0.0,
            invariant_residual: 0.0,
            constitutive_normal_residual: 0.0,
            constitutive_shear_residual: 0.0,
            val_displacement_fit: 0.0,
            val_stress_fit: 0.0,
            val_invariant_residual: 0.0,
            val_constitutive_normal_residual: 0.0,
            val_constitutive_shear_residual: 0.0,
            hybrid_mode: "pino-ndarray-cpu+booting".to_string(),
            stage_id: if epoch > 0 {
                "stage-1".to_string()
            } else {
                "preflight".to_string()
            },
            optimizer_id: "pino-adam".to_string(),
            lr_phase: lr_phase.to_string(),
            target_band_low: 0.0,
            target_band_high: 0.0,
            trend_slope: 0.0,
            trend_variance: 0.0,
            watchdog_trigger_count: 0,
            collocation_samples_added: 0,
            train_data_size: 0,
            train_data_cap: 0,
            residual_weight_momentum: 1.0,
            residual_weight_kinematics: 1.0,
            residual_weight_material: 1.0,
            residual_weight_boundary: 1.0,
            learning_rate: 5e-4,
            architecture: vec![15, 48, 48, 48, 11],
            progress_ratio: if epoch > 0 { epoch as f64 / 10_000.0 } else { 0.0 },
            training_mode: "legacy-mixed-exact".to_string(),
            benchmark_id: None,
            gate_status: if epoch > 0 { "running".to_string() } else { "queued".to_string() },
            certified_best_metric: if epoch > 0 { 0.30 } else { f64::MAX },
            dominant_blocker: None,
            stalled_reason: None,
            network: NetworkSnapshot {
                layer_sizes: vec![],
                nodes: vec![],
                connections: vec![],
            },
            pino: None,
        }
    }

    #[test]
    fn bootstrap_tick_does_not_erase_completed_epoch() {
        let current = sample_tick(15, 0.25, 0.30);
        let incoming = sample_tick(0, 0.0, 0.0);
        let merged = merge_training_tick(&current, &incoming);
        assert_eq!(merged.epoch, 15);
        assert!((merged.loss - 0.25).abs() <= 1e-12);
        assert!((merged.val_loss - 0.30).abs() <= 1e-12);
    }

    #[test]
    fn bootstrap_progress_preserves_completed_metrics_and_updates_phase() {
        let current = sample_progress(15, "pino-steady");
        let incoming = sample_progress(0, "epoch-16-bootstrap");
        let merged = merge_training_progress(&current, &incoming);
        assert_eq!(merged.epoch, 15);
        assert!((merged.loss - current.loss).abs() <= 1e-12);
        assert!((merged.val_loss - current.val_loss).abs() <= 1e-12);
        assert_eq!(merged.lr_phase, "epoch-16-bootstrap");
        assert_eq!(merged.stage_id, "preflight");
    }

    #[test]
    fn exact_refine_progress_preserves_epoch_but_updates_live_losses() {
        let current = sample_progress(48, "pino-steady");
        let mut incoming = sample_progress(0, "candidate-r1-s1-adam");
        incoming.stage_id = "exact-refine".to_string();
        incoming.loss = 4.176160;
        incoming.val_loss = 3.699585;

        let merged = merge_training_progress(&current, &incoming);
        assert_eq!(merged.epoch, 48);
        assert!((merged.loss - 4.176160).abs() <= 1e-12);
        assert!((merged.val_loss - 3.699585).abs() <= 1e-12);
        assert_eq!(merged.stage_id, "exact-refine");
        assert_eq!(merged.lr_phase, "candidate-r1-s1-adam");
    }

    #[test]
    fn epochs_since_improvement_ignores_epoch_zero_refine_frames() {
        let mut best_val = 0.25;
        let mut since_improve = 3usize;
        let mut last_epoch_counted = 15usize;

        update_epochs_since_improvement(
            &mut best_val,
            &mut since_improve,
            &mut last_epoch_counted,
            0,
            f64::MAX,
        );

        assert!((best_val - 0.25).abs() <= 1e-12);
        assert_eq!(since_improve, 3);
        assert_eq!(last_epoch_counted, 15);
    }

    #[test]
    fn epochs_since_improvement_counts_only_new_completed_epochs() {
        let mut best_val = 0.25;
        let mut since_improve = 0usize;
        let mut last_epoch_counted = 15usize;

        update_epochs_since_improvement(
            &mut best_val,
            &mut since_improve,
            &mut last_epoch_counted,
            16,
            0.30,
        );
        update_epochs_since_improvement(
            &mut best_val,
            &mut since_improve,
            &mut last_epoch_counted,
            16,
            0.31,
        );

        assert!((best_val - 0.25).abs() <= 1e-12);
        assert_eq!(since_improve, 1);
        assert_eq!(last_epoch_counted, 16);
    }

    #[test]
    fn epoch_zero_certified_metric_can_improve_best_without_counting_epoch() {
        let mut best_val = 10.0;
        let mut since_improve = 8usize;
        let mut last_epoch_counted = 48usize;

        update_epochs_since_improvement(
            &mut best_val,
            &mut since_improve,
            &mut last_epoch_counted,
            0,
            3.7,
        );

        assert!((best_val - 3.7).abs() <= 1e-12);
        assert_eq!(since_improve, 0);
        assert_eq!(last_epoch_counted, 48);
    }
}
