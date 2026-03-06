#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ann;
mod contracts;
mod fem;
mod io;
mod physics;

use std::sync::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ann::AnnModel;
use contracts::*;
use tauri::{Emitter, State};

struct AppState {
    model: Arc<Mutex<AnnModel>>,
    training_tick: Arc<Mutex<TrainingTickEvent>>,
    training_status: Arc<Mutex<TrainingRunStatus>>,
    training_stop: Arc<AtomicBool>,
}

#[tauri::command(rename_all = "camelCase")]
fn solve_fem_case(input: SolveInput) -> Result<FemResult, String> {
    Ok(fem::solve_case(&input))
}

#[tauri::command(rename_all = "camelCase")]
fn run_thermal_case(input: ThermalInput) -> Result<ThermalResult, String> {
    let fem = fem::solve_case(&input.solve_input);
    Ok(physics::thermal::run(&input, fem.stress_tensor))
}

#[tauri::command(rename_all = "camelCase")]
fn run_dynamic_case(input: DynamicInput) -> Result<DynamicResult, String> {
    Ok(physics::dynamic::run(&input))
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
        };
    }

    let init_total = batch.max_total_epochs.unwrap_or(batch.epochs.max(1));
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
    state.training_stop.store(false, Ordering::Relaxed);

    let model = state.model.clone();
    let tick_state = state.training_tick.clone();
    let status_state = state.training_status.clone();
    let stop_flag = state.training_stop.clone();
    let app_handle = app.clone();
    let batch_owned = batch.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let sanitize = |v: f64| if v.is_finite() { v } else { 0.0 };
        let run_result = (|| -> Result<TrainResult, String> {
            let mut model = model
                .lock()
                .map_err(|_| "ANN state lock poisoned".to_string())?;
            let result = model.train_with_progress(&batch_owned, |progress| {
                let tick = TrainingTickEvent {
                    epoch: progress.epoch,
                    total_epochs: progress.total_epochs,
                    loss: sanitize(progress.loss),
                    val_loss: sanitize(progress.val_loss),
                    learning_rate: sanitize(progress.learning_rate),
                    architecture: progress.architecture.clone(),
                    progress_ratio: sanitize(progress.progress_ratio).clamp(0.0, 1.0),
                };
                if let Ok(mut shared_tick) = tick_state.lock() {
                    *shared_tick = tick.clone();
                }
                let _ = app_handle.emit("ann-training-tick", tick);
                let safe_progress = TrainingProgressEvent {
                    loss: sanitize(progress.loss),
                    val_loss: sanitize(progress.val_loss),
                    learning_rate: sanitize(progress.learning_rate),
                    progress_ratio: sanitize(progress.progress_ratio).clamp(0.0, 1.0),
                    ..progress
                };
                let _ = app_handle.emit("ann-training-progress", safe_progress);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }, || stop_flag.load(Ordering::Relaxed));
            Ok(result)
        })();

        match run_result {
            Ok(result) => {
                if let Ok(mut status) = status_state.lock() {
                    *status = TrainingRunStatus {
                        running: false,
                        stop_requested: false,
                        completed: true,
                        last_result: Some(result.clone()),
                        last_error: None,
                    };
                }
                let _ = app_handle.emit("ann-training-complete", &result);
            }
            Err(err) => {
                if let Ok(mut status) = status_state.lock() {
                    *status = TrainingRunStatus {
                        running: false,
                        stop_requested: false,
                        completed: true,
                        last_result: None,
                        last_error: Some(err.clone()),
                    };
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
fn infer_ann(input: SolveInput, state: State<'_, AppState>) -> Result<AnnResult, String> {
    let model = state
        .model
        .lock()
        .map_err(|_| "ANN state lock poisoned".to_string())?;
    Ok(model.infer(&input))
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
fn export_report(input: ReportInput) -> Result<ExportResult, String> {
    io::export_report(&input)
}

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            model: Arc::new(Mutex::new(AnnModel::default())),
            training_tick: Arc::new(Mutex::new(TrainingTickEvent {
                epoch: 0,
                total_epochs: 0,
                loss: 0.0,
                val_loss: 0.0,
                learning_rate: 0.0,
                architecture: vec![],
                progress_ratio: 0.0,
            })),
            training_status: Arc::new(Mutex::new(TrainingRunStatus {
                running: false,
                stop_requested: false,
                completed: false,
                last_result: None,
                last_error: None,
            })),
            training_stop: Arc::new(AtomicBool::new(false)),
        })
        .invoke_handler(tauri::generate_handler![
            solve_fem_case,
            train_ann,
            start_ann_training,
            stop_ann_training,
            get_training_status,
            infer_ann,
            run_dynamic_case,
            run_thermal_case,
            evaluate_failure,
            get_model_status,
            get_training_tick,
            export_report
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
