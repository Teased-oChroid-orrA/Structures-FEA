#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ann;
mod contracts;
mod fem;
mod io;
mod physics;

use std::sync::Mutex;
use std::sync::Arc;

use ann::AnnModel;
use contracts::*;
use tauri::{Emitter, State};

struct AppState {
    model: Arc<Mutex<AnnModel>>,
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
async fn train_ann(
    batch: TrainingBatch,
    state: State<'_, AppState>,
    app: tauri::AppHandle,
) -> Result<TrainResult, String> {
    let model = state.model.clone();
    let app_handle = app.clone();
    let batch_owned = batch.clone();

    let result = tauri::async_runtime::spawn_blocking(move || {
        let mut model = model
            .lock()
            .map_err(|_| "ANN state lock poisoned".to_string())?;
        let result = model.train_with_progress(&batch_owned, |progress| {
            let _ = app_handle.emit("ann-training-progress", progress);
            std::thread::sleep(std::time::Duration::from_millis(1));
        });
        let _ = app_handle.emit("ann-training-complete", &result);
        Ok::<TrainResult, String>(result)
    })
    .await
    .map_err(|e| format!("ANN worker task failed: {e}"))??;

    Ok(result)
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
        })
        .invoke_handler(tauri::generate_handler![
            solve_fem_case,
            train_ann,
            infer_ann,
            run_dynamic_case,
            run_thermal_case,
            evaluate_failure,
            get_model_status,
            export_report
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
