use crate::ann::{AnnModel, AnnModelState};
use crate::benchmarks::{certify_training_benchmark, BenchmarkCertificationInput};
use crate::contracts::{
    AnnResult, FemResult, HoldoutValidationSummary, ModelStatus, NetworkSnapshot,
    PinoRuntimeMetadata, SafeguardSettings, SolveInput, TrainResult, TrainingBatch,
    TrainingProgressEvent,
};
use crate::pinn_burn::{BurnPilotStats, ResidualPillars};
use crate::pino::{
    apply_operator_calibration, apply_operator_displacement_calibration,
    build_operator_field_head_batch, build_operator_prediction,
    build_operator_prediction_with_params, canonical_backend_name, decode_prediction,
    default_holdout_validation, enforce_training_recipe, evaluate_holdout_projection_with_model,
    fit_operator_calibration, fit_operator_calibration_with_params, infer_config_for_case,
    is_pino_backend, model_config, operator_residual_score,
    prediction_from_fem_with_grid_exact, reconstruct_prediction_linear_elastic_from_displacement,
    reconstruct_prediction_linear_elastic_from_displacement_smoothed,
    runtime_metadata, spectral_probe_score, train_operator_calibration, OperatorCalibration,
    OperatorFieldHeadBatch, OperatorPrediction, OperatorTrainableParams, OperatorTrainingStats,
    PinoModelConfig, PINO_BACKEND_CANDLE_CPU, PINO_BACKEND_CANDLE_CUDA,
    PINO_BACKEND_CANDLE_METAL, PINO_BACKEND_NDARRAY_CPU, PINO_FIELD_HEAD_BASIS,
    PINO_OUTPUT_CHANNELS, PINO_DISPLACEMENT_OUTPUT_CHANNELS,
};
use crate::pino_burn_head::{
    evaluate_operator_field_head_physics, evaluate_operator_field_head_physics_grad_norm,
    set_benchmark_cantilever_override, set_isolated_exact_cantilever_override,
    train_operator_field_head_physics,
    BurnFieldHeadOptimizer, BurnPhysicsLossBreakdown, BurnPhysicsSample,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::{BTreeSet, HashSet, VecDeque};

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|raw| matches!(raw.as_str(), "1" | "true" | "TRUE" | "on" | "ON"))
        .unwrap_or(default)
}

fn benchmark_dominant_blocker_from_breakdown(
    breakdown: &BurnPhysicsLossBreakdown,
) -> Option<String> {
    let candidates = [
        ("stress-fit", breakdown.stress_fit),
        ("constitutive-normal", breakdown.constitutive_normal),
        ("observable", breakdown.observable),
        ("equilibrium", breakdown.equilibrium),
        ("constitutive-shear", breakdown.constitutive_shear),
        ("displacement-fit", breakdown.displacement_fit),
        ("weak-energy", breakdown.weak_energy),
        ("boundary", breakdown.boundary),
    ];
    candidates
        .into_iter()
        .filter(|(_, value)| value.is_finite())
        .max_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(label, _)| label.to_string())
}

#[derive(Debug, Clone, Copy)]
struct CantileverBenchmarkPhysicalErrors {
    tip_displacement_relative_error: f64,
    max_displacement_relative_error: f64,
    mean_von_mises_relative_error: f64,
    max_sigma_xx_relative_error: f64,
}

fn relative_error(actual: f64, target: f64, floor: f64) -> f64 {
    ((actual - target).abs() / target.abs().max(floor)).clamp(0.0, 10.0)
}

fn tip_cell_index(prediction: &OperatorPrediction) -> usize {
    if prediction.ux.is_empty() {
        return 0;
    }
    ((((prediction.grid.nz / 2) * prediction.grid.ny) + (prediction.grid.ny / 2))
        * prediction.grid.nx
        + prediction.grid.nx.saturating_sub(1))
        .min(prediction.ux.len().saturating_sub(1))
}

fn displacement_magnitude(prediction: &OperatorPrediction, idx: usize) -> f64 {
    let ux = prediction.ux.get(idx).copied().unwrap_or(0.0);
    let uy = prediction.uy.get(idx).copied().unwrap_or(0.0);
    let uz = prediction.uz.get(idx).copied().unwrap_or(0.0);
    (ux * ux + uy * uy + uz * uz).sqrt()
}

fn evaluate_cantilever_benchmark_physical_errors(
    prediction: &OperatorPrediction,
    target: &OperatorPrediction,
) -> CantileverBenchmarkPhysicalErrors {
    let cell_count = prediction
        .ux
        .len()
        .min(target.ux.len())
        .min(prediction.von_mises.len())
        .min(target.von_mises.len())
        .min(prediction.sxx.len())
        .min(target.sxx.len());
    if cell_count == 0 {
        return CantileverBenchmarkPhysicalErrors {
            tip_displacement_relative_error: f64::INFINITY,
            max_displacement_relative_error: f64::INFINITY,
            mean_von_mises_relative_error: f64::INFINITY,
            max_sigma_xx_relative_error: f64::INFINITY,
        };
    }
    let tip_idx = tip_cell_index(target).min(cell_count.saturating_sub(1));
    let tip_displacement_relative_error = relative_error(
        prediction.uy.get(tip_idx).copied().unwrap_or(0.0),
        target.uy.get(tip_idx).copied().unwrap_or(0.0),
        1.0e-9,
    );
    let max_pred_disp = (0..cell_count)
        .map(|idx| displacement_magnitude(prediction, idx))
        .fold(0.0_f64, f64::max);
    let max_target_disp = (0..cell_count)
        .map(|idx| displacement_magnitude(target, idx))
        .fold(0.0_f64, f64::max);
    let max_displacement_relative_error =
        relative_error(max_pred_disp, max_target_disp, 1.0e-9);
    let max_target_vm = target
        .von_mises
        .iter()
        .take(cell_count)
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let vm_activity_floor = (max_target_vm * 0.05).max(1.0);
    let active_vm_cells = (0..cell_count)
        .filter(|idx| target.von_mises.get(*idx).copied().unwrap_or(0.0).abs() >= vm_activity_floor)
        .collect::<Vec<_>>();
    let vm_eval_cells = if active_vm_cells.is_empty() {
        (0..cell_count).collect::<Vec<_>>()
    } else {
        active_vm_cells
    };
    let mean_von_mises_relative_error = vm_eval_cells
        .iter()
        .map(|idx| {
            relative_error(
                prediction.von_mises.get(*idx).copied().unwrap_or(0.0),
                target.von_mises.get(*idx).copied().unwrap_or(0.0),
                vm_activity_floor,
            )
        })
        .sum::<f64>()
        / vm_eval_cells.len().max(1) as f64;
    let max_pred_sxx = prediction
        .sxx
        .iter()
        .take(cell_count)
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let max_target_sxx = target
        .sxx
        .iter()
        .take(cell_count)
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let max_sigma_xx_relative_error = relative_error(max_pred_sxx, max_target_sxx, 1.0);
    CantileverBenchmarkPhysicalErrors {
        tip_displacement_relative_error,
        max_displacement_relative_error,
        mean_von_mises_relative_error,
        max_sigma_xx_relative_error,
    }
}

fn cantilever_benchmark_selection_score(errors: &CantileverBenchmarkPhysicalErrors) -> f64 {
    errors.tip_displacement_relative_error * 2.2
        + errors.max_displacement_relative_error * 1.6
        + errors.mean_von_mises_relative_error * 0.45
        + errors.max_sigma_xx_relative_error * 0.75
}

fn evaluate_cantilever_benchmark_selection_metric(
    case: &SolveInput,
    cfg: &PinoModelConfig,
    calibration: &OperatorCalibration,
    params: &OperatorTrainableParams,
    target: &OperatorPrediction,
) -> f64 {
    let mut calibrated_prediction = build_operator_prediction_with_params(case, cfg, Some(params));
    let evaluated_prediction = if cfg.operator_grid.output_channels == PINO_DISPLACEMENT_OUTPUT_CHANNELS
    {
        apply_operator_displacement_calibration(&mut calibrated_prediction, calibration);
        reconstruct_prediction_linear_elastic_from_displacement_smoothed(
            case,
            &calibrated_prediction,
        )
    } else {
        apply_operator_calibration(&mut calibrated_prediction, calibration);
        calibrated_prediction
    };
    let errors = evaluate_cantilever_benchmark_physical_errors(&evaluated_prediction, target);
    cantilever_benchmark_selection_score(&errors)
}

struct BurnHeadIsolatedExactGuard;

impl Drop for BurnHeadIsolatedExactGuard {
    fn drop(&mut self) {
        set_benchmark_cantilever_override(false);
        set_isolated_exact_cantilever_override(false);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct UniversalPinnConfig {
    pub backend: String,
    pub collocation_points: usize,
    pub boundary_points: usize,
    pub interface_points: usize,
    pub residual_weight_momentum: f64,
    pub residual_weight_kinematics: f64,
    pub residual_weight_material: f64,
    pub residual_weight_boundary: f64,
    pub stage1_epochs: usize,
    pub stage2_epochs: usize,
    pub stage3_ramp_epochs: usize,
    pub contact_penalty: f64,
    pub plasticity_factor: f64,
}

impl UniversalPinnConfig {
    pub fn from_batch(batch: &TrainingBatch) -> Self {
        Self {
            backend: batch
                .pinn_backend
                .clone()
                .unwrap_or_else(|| PINO_BACKEND_NDARRAY_CPU.to_string()),
            collocation_points: batch.collocation_points.unwrap_or(4096).max(64),
            boundary_points: batch.boundary_points.unwrap_or(1024).max(16),
            interface_points: batch.interface_points.unwrap_or(512).max(16),
            residual_weight_momentum: batch.residual_weight_momentum.unwrap_or(1.0).max(0.0),
            residual_weight_kinematics: batch.residual_weight_kinematics.unwrap_or(1.0).max(0.0),
            residual_weight_material: batch.residual_weight_material.unwrap_or(1.0).max(0.0),
            residual_weight_boundary: batch.residual_weight_boundary.unwrap_or(1.0).max(0.0),
            stage1_epochs: batch.stage1_epochs.unwrap_or(0),
            stage2_epochs: batch.stage2_epochs.unwrap_or(0),
            stage3_ramp_epochs: batch.stage3_ramp_epochs.unwrap_or(0),
            contact_penalty: batch.contact_penalty.unwrap_or(10.0).max(0.0),
            plasticity_factor: batch.plasticity_factor.unwrap_or(0.0).clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnRuntimeState {
    pub architecture: Vec<usize>,
    pub learning_rate: f64,
    pub model_version: u64,
    pub last_loss: f64,
    pub best_val_loss: f64,
    pub train_samples: usize,
    pub completed_epochs: usize,
    pub total_epochs: usize,
    pub backend_tag: String,
    pub last_train_seed: Option<u64>,
    pub recent_notes: Vec<String>,
    pub pino: Option<PinoRuntimeMetadata>,
    #[serde(default)]
    pub pino_calibration: Option<OperatorCalibration>,
    #[serde(default)]
    pub pino_params: Option<OperatorTrainableParams>,
    #[serde(default)]
    pub pino_optimizer_state: Option<OperatorParamOptimizerState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalPinnState {
    pub ann: AnnModelState,
    pub burn: Option<BurnRuntimeState>,
    pub last_config: UniversalPinnConfig,
    pub backend_runtime: String,
}

#[derive(Clone)]
pub struct UniversalPinnEngine {
    inner: AnnModel,
    last_config: UniversalPinnConfig,
    backend_runtime: PinnBackendRuntime,
    burn_state: Option<BurnRuntimeState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PinnBackendRuntime {
    PinoNdArrayCpu,
    PinoCandleCpu,
    PinoCandleCuda,
    PinoCandleMetal,
    CompatAnn,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PinoAdaptDirection {
    Explore,
    Grow,
    Shrink,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PinoResidualFocus {
    Global,
    Local,
    Boundary,
    Kinematic,
    Balanced,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OperatorParamOptimizerState {
    pub step: usize,
    pub beta1_power: f64,
    pub beta2_power: f64,
    pub first_moment: Vec<f64>,
    pub second_moment: Vec<f64>,
}

#[derive(Debug, Clone)]
pub(crate) struct OperatorFieldTarget {
    case: crate::contracts::SolveInput,
    fem: FemResult,
    target: OperatorPrediction,
}

#[derive(Debug, Clone)]
struct BurnPhysicsSampleCore {
    batch: OperatorFieldHeadBatch,
    target: OperatorPrediction,
}

impl Default for OperatorParamOptimizerState {
    fn default() -> Self {
        Self {
            step: 0,
            beta1_power: 1.0,
            beta2_power: 1.0,
            first_moment: Vec::new(),
            second_moment: Vec::new(),
        }
    }
}

impl PinnBackendRuntime {
    fn legacy_rollback_enabled() -> bool {
        matches!(
            std::env::var("PINO_ENABLE_LEGACY_ROLLBACK").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON")
        )
    }

    fn from_name(name: &str) -> Self {
        match name {
            PINO_BACKEND_NDARRAY_CPU | "burn-ndarray-cpu" => Self::PinoNdArrayCpu,
            PINO_BACKEND_CANDLE_CPU | "burn-wgpu" => Self::PinoCandleCpu,
            PINO_BACKEND_CANDLE_CUDA => Self::PinoCandleCuda,
            PINO_BACKEND_CANDLE_METAL => Self::PinoCandleMetal,
            _ if Self::legacy_rollback_enabled() => Self::CompatAnn,
            _ => Self::PinoNdArrayCpu,
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Self::PinoNdArrayCpu => PINO_BACKEND_NDARRAY_CPU,
            Self::PinoCandleCpu => PINO_BACKEND_CANDLE_CPU,
            Self::PinoCandleCuda => PINO_BACKEND_CANDLE_CUDA,
            Self::PinoCandleMetal => PINO_BACKEND_CANDLE_METAL,
            Self::CompatAnn => "compat-ann",
        }
    }
}

impl Default for UniversalPinnEngine {
    fn default() -> Self {
        Self {
            inner: AnnModel::default(),
            last_config: UniversalPinnConfig {
                backend: PINO_BACKEND_NDARRAY_CPU.to_string(),
                collocation_points: 4096,
                boundary_points: 1024,
                interface_points: 512,
                residual_weight_momentum: 1.0,
                residual_weight_kinematics: 1.0,
                residual_weight_material: 1.0,
                residual_weight_boundary: 1.0,
                stage1_epochs: 0,
                stage2_epochs: 0,
                stage3_ramp_epochs: 0,
                contact_penalty: 10.0,
                plasticity_factor: 0.0,
            },
            backend_runtime: PinnBackendRuntime::PinoNdArrayCpu,
            burn_state: None,
        }
    }
}

impl UniversalPinnEngine {
    fn pino_runtime_architecture(config: &PinoModelConfig) -> Vec<usize> {
        let mut arch = Vec::with_capacity(config.hidden_layers.saturating_add(2));
        arch.push(config.operator_grid.input_channels);
        for _ in 0..config.hidden_layers.max(1) {
            arch.push(config.hidden_width.max(4));
        }
        arch.push(config.operator_grid.output_channels);
        arch
    }

    fn normalize_pino_config(mut config: PinoModelConfig, max_topology: usize) -> PinoModelConfig {
        let cap = max_topology.clamp(8, 256);
        config.spectral_modes = config.spectral_modes.clamp(2, 16);
        config.hidden_layers = config.hidden_layers.clamp(1, 6);
        config.hidden_width = config.hidden_width.clamp(8, cap);
        config
    }

    fn compact_low_target_start_config(
        config: &PinoModelConfig,
        max_topology: usize,
        headless_fast_profile: bool,
    ) -> PinoModelConfig {
        let mut compact = Self::normalize_pino_config(config.clone(), max_topology);
        if headless_fast_profile {
            if compact.hidden_layers >= 3 {
                compact.hidden_layers = compact.hidden_layers.saturating_sub(1).max(2);
            }
            if compact.hidden_width >= 48 {
                compact.hidden_width = compact.hidden_width.saturating_sub(8).max(40);
            }
        } else {
            if compact.hidden_layers >= 4 {
                compact.hidden_layers = compact.hidden_layers.saturating_sub(1).max(3);
            }
            if compact.hidden_width >= 56 {
                compact.hidden_width = compact.hidden_width.saturating_sub(8).max(48);
            }
        }
        Self::normalize_pino_config(compact, max_topology)
    }

    fn mutate_pino_config(
        config: &PinoModelConfig,
        direction: PinoAdaptDirection,
        max_topology: usize,
    ) -> Vec<PinoModelConfig> {
        let width_step = if config.hidden_width >= 48 { 8 } else { 4 };
        let mode_step = if config.spectral_modes >= 6 { 2 } else { 1 };
        let mut out = vec![Self::normalize_pino_config(config.clone(), max_topology)];
        let add_candidate = |candidate: PinoModelConfig, out: &mut Vec<PinoModelConfig>| {
            let normalized = Self::normalize_pino_config(candidate, max_topology);
            if !out.contains(&normalized) {
                out.push(normalized);
            }
        };
        match direction {
            PinoAdaptDirection::Explore => {
                add_candidate(
                    PinoModelConfig {
                        spectral_modes: config.spectral_modes.saturating_add(mode_step),
                        hidden_width: config.hidden_width.saturating_add(width_step),
                        ..config.clone()
                    },
                    &mut out,
                );
                add_candidate(
                    PinoModelConfig {
                        spectral_modes: config.spectral_modes.saturating_sub(mode_step).max(2),
                        hidden_width: config.hidden_width.saturating_sub(width_step).max(8),
                        ..config.clone()
                    },
                    &mut out,
                );
                add_candidate(
                    PinoModelConfig {
                        hidden_layers: config.hidden_layers.saturating_add(1),
                        ..config.clone()
                    },
                    &mut out,
                );
                add_candidate(
                    PinoModelConfig {
                        hidden_layers: config.hidden_layers.saturating_sub(1).max(1),
                        ..config.clone()
                    },
                    &mut out,
                );
            }
            PinoAdaptDirection::Grow => {
                add_candidate(
                    PinoModelConfig {
                        hidden_width: config.hidden_width.saturating_add(width_step),
                        ..config.clone()
                    },
                    &mut out,
                );
                add_candidate(
                    PinoModelConfig {
                        hidden_layers: config.hidden_layers.saturating_add(1),
                        ..config.clone()
                    },
                    &mut out,
                );
                add_candidate(
                    PinoModelConfig {
                        spectral_modes: config.spectral_modes.saturating_add(mode_step),
                        ..config.clone()
                    },
                    &mut out,
                );
            }
            PinoAdaptDirection::Shrink => {
                add_candidate(
                    PinoModelConfig {
                        hidden_width: config.hidden_width.saturating_sub(width_step).max(8),
                        ..config.clone()
                    },
                    &mut out,
                );
                add_candidate(
                    PinoModelConfig {
                        hidden_layers: config.hidden_layers.saturating_sub(1).max(1),
                        ..config.clone()
                    },
                    &mut out,
                );
                add_candidate(
                    PinoModelConfig {
                        spectral_modes: config.spectral_modes.saturating_sub(mode_step).max(2),
                        ..config.clone()
                    },
                    &mut out,
                );
            }
        }
        out
    }

    fn is_directional_candidate(
        baseline: &PinoModelConfig,
        candidate: &PinoModelConfig,
        direction: PinoAdaptDirection,
    ) -> bool {
        match direction {
            PinoAdaptDirection::Explore => candidate != baseline,
            PinoAdaptDirection::Grow => {
                candidate != baseline
                    && (candidate.spectral_modes >= baseline.spectral_modes
                        || candidate.hidden_layers >= baseline.hidden_layers
                        || candidate.hidden_width >= baseline.hidden_width)
            }
            PinoAdaptDirection::Shrink => {
                candidate != baseline
                    && (candidate.spectral_modes <= baseline.spectral_modes
                        || candidate.hidden_layers <= baseline.hidden_layers
                        || candidate.hidden_width <= baseline.hidden_width)
            }
        }
    }

    fn select_best_pino_config(
        batch: &TrainingBatch,
        current: &PinoModelConfig,
        direction: PinoAdaptDirection,
        _focus: PinoResidualFocus,
        max_topology: usize,
        train_epochs: usize,
    ) -> (PinoModelConfig, Option<OperatorTrainingStats>) {
        let candidates = Self::mutate_pino_config(current, direction, max_topology);
        let reference_case = batch.cases.first().cloned();
        let mut scored =
            Vec::<(PinoModelConfig, OperatorTrainingStats, f64)>::with_capacity(candidates.len());
        for candidate in candidates.into_iter() {
            let mut residual = 1.0;
            if let Some(case) = reference_case.as_ref() {
                let mut probe_candidate = candidate.clone();
                probe_candidate.operator_grid.nx = probe_candidate.operator_grid.nx.clamp(8, 16);
                probe_candidate.operator_grid.ny = probe_candidate.operator_grid.ny.clamp(6, 12);
                let prediction = build_operator_prediction(case, &probe_candidate);
                residual = operator_residual_score(case, &prediction).max(1e-6);
            }
            let spectral = spectral_probe_score(batch, &candidate)
                .unwrap_or(1.0)
                .max(1e-6);
            let complexity = candidate.hidden_layers as f64
                + (candidate.hidden_width as f64 / 32.0)
                + (candidate.spectral_modes as f64 / 4.0);
            let depth_delta = candidate.hidden_layers as isize - current.hidden_layers as isize;
            let width_delta = candidate.hidden_width as isize - current.hidden_width as isize;
            let mode_delta = candidate.spectral_modes as isize - current.spectral_modes as isize;
            // Favor width/modes growth before repeatedly stacking depth on plateau.
            let adaptation_bias = match direction {
                PinoAdaptDirection::Grow => {
                    let mut bias = 0.0;
                    if depth_delta > 0 && width_delta <= 0 && mode_delta <= 0 {
                        bias += 0.050 * depth_delta as f64;
                    }
                    if width_delta > 0 {
                        bias -= 0.020 * width_delta as f64;
                    }
                    if mode_delta > 0 {
                        bias -= 0.015 * mode_delta as f64;
                    }
                    bias
                }
                PinoAdaptDirection::Shrink => {
                    let mut bias = 0.0;
                    if depth_delta < 0 && width_delta >= 0 && mode_delta >= 0 {
                        bias -= 0.010 * (-depth_delta) as f64;
                    }
                    if width_delta < 0 {
                        bias -= 0.012 * (-width_delta) as f64;
                    }
                    if mode_delta < 0 {
                        bias -= 0.010 * (-mode_delta) as f64;
                    }
                    bias
                }
                PinoAdaptDirection::Explore => 0.0,
            };
            let score = 0.64 * residual + 0.28 * spectral + 0.015 * complexity + adaptation_bias;
            let stats = OperatorTrainingStats {
                epochs_run: train_epochs.clamp(1, 16),
                initial_loss: score,
                best_loss: score,
                final_loss: score,
                calibration: OperatorCalibration {
                    stress_scale: 1.0,
                    displacement_scale: 1.0,
                },
            };
            scored.push((candidate, stats, score));
        }
        if scored.is_empty() {
            return (current.clone(), None);
        }

        scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        let normalized_current = Self::normalize_pino_config(current.clone(), max_topology);
        let baseline = scored
            .iter()
            .find(|(cfg, _, _)| cfg == &normalized_current)
            .cloned()
            .unwrap_or_else(|| scored[0].clone());
        if direction == PinoAdaptDirection::Explore {
            let best = scored[0].clone();
            return (best.0, Some(best.1));
        }

        // Guardrails:
        // - Grow can accept at most +20% final loss to escape plateaus.
        // - Shrink can accept at most +8% final loss when countering overfit.
        let max_rel_loss = if direction == PinoAdaptDirection::Grow {
            0.20
        } else {
            0.08
        };
        let baseline_final = baseline.1.final_loss.max(1e-9);
        let directional = scored
            .into_iter()
            .filter(|(cfg, stats, _)| {
                Self::is_directional_candidate(&baseline.0, cfg, direction)
                    && stats.final_loss <= baseline_final * (1.0 + max_rel_loss)
            })
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((cfg, stats, _)) = directional {
            (cfg, Some(stats))
        } else {
            (baseline.0, Some(baseline.1))
        }
    }

    fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    fn displacement_scale_fallback(target: f64, base: f64, characteristic: f64) -> f64 {
        (target - base)
            .abs()
            .max(characteristic.max(1.0e-6) * 0.35)
    }

    fn rebalance_loss_weights(
        current: (f64, f64, f64, f64),
        pillars: ResidualPillars,
    ) -> (f64, f64, f64, f64) {
        let vals = [
            pillars.momentum.max(1e-6),
            pillars.kinematics.max(1e-6),
            pillars.material.max(1e-6),
            pillars.boundary.max(1e-6),
        ];
        let inv = vals.map(|v| 1.0 / v.sqrt());
        let inv_sum = inv.iter().sum::<f64>().max(1e-9);
        let base_sum = (current.0 + current.1 + current.2 + current.3).max(1e-6);
        let prior = [0.18, 0.34, 0.20, 0.28];
        let target = [
            (inv[0] / inv_sum) * base_sum * 0.6 + prior[0] * base_sum * 0.4,
            (inv[1] / inv_sum) * base_sum * 0.6 + prior[1] * base_sum * 0.4,
            (inv[2] / inv_sum) * base_sum * 0.6 + prior[2] * base_sum * 0.4,
            (inv[3] / inv_sum) * base_sum * 0.6 + prior[3] * base_sum * 0.4,
        ];
        let blend = 0.18;
        let next = [
            ((1.0 - blend) * current.0 + blend * target[0]).clamp(0.15, 8.0),
            ((1.0 - blend) * current.1 + blend * target[1]).clamp(0.35, 8.0),
            ((1.0 - blend) * current.2 + blend * target[2]).clamp(0.20, 8.0),
            ((1.0 - blend) * current.3 + blend * target[3]).clamp(0.25, 8.0),
        ];
        (next[0], next[1], next[2], next[3])
    }

    fn curriculum_loss_weights(
        base: (f64, f64, f64, f64),
        stage_index: usize,
    ) -> (f64, f64, f64, f64) {
        match stage_index {
            1 => (
                (base.0 * 0.12).clamp(0.06, 1.0),
                base.1.max(1.95).clamp(1.4, 3.4),
                (base.2 * 0.14).clamp(0.06, 0.9),
                (base.3 * 0.55).clamp(0.18, 1.8),
            ),
            2 => (
                (base.0 * 0.34).clamp(0.10, 1.6),
                base.1.max(1.45).clamp(1.1, 2.8),
                (base.2 * 0.36).clamp(0.10, 1.4),
                (base.3 * 0.78).clamp(0.20, 2.2),
            ),
            _ => (
                base.0.clamp(0.20, 3.2),
                base.1.max(0.95).clamp(0.75, 2.0),
                base.2.max(0.28).clamp(0.20, 2.6),
                base.3.max(0.45).clamp(0.30, 3.0),
            ),
        }
    }

    pub(crate) fn build_burn_physics_sample(
        sample: &OperatorFieldTarget,
        config: &PinoModelConfig,
        calibration: &OperatorCalibration,
        params: &OperatorTrainableParams,
    ) -> BurnPhysicsSample {
        let core = Self::build_burn_physics_sample_core(sample, config, params);
        Self::build_burn_physics_sample_from_core(sample, config, calibration, &core)
    }

    fn build_burn_physics_sample_core(
        sample: &OperatorFieldTarget,
        config: &PinoModelConfig,
        params: &OperatorTrainableParams,
    ) -> BurnPhysicsSampleCore {
        let batch = build_operator_field_head_batch(&sample.case, config, Some(params));
        let target = sample.target.clone();
        BurnPhysicsSampleCore { batch, target }
    }

    fn build_burn_physics_sample_from_core(
        sample: &OperatorFieldTarget,
        config: &PinoModelConfig,
        calibration: &OperatorCalibration,
        core: &BurnPhysicsSampleCore,
    ) -> BurnPhysicsSample {
        let batch = &core.batch;
        let target = &core.target;
        let cells = batch.cell_count;
        let output_dim = batch.output_dim.min(PINO_OUTPUT_CHANNELS).max(1);
        let (characteristic_disp_scale, characteristic_stress_scale) =
            Self::cantilever_characteristic_scales(&sample.case);
        let mut target_fields = Vec::with_capacity(cells * PINO_OUTPUT_CHANNELS);
        let mut base_fields = Vec::with_capacity(cells * PINO_OUTPUT_CHANNELS);
        let mut correction_scales = Vec::with_capacity(cells * PINO_OUTPUT_CHANNELS);
        for idx in 0..cells {
            let scale_start = idx * batch.output_dim;
            target_fields.extend_from_slice(&[
                target.ux.get(idx).copied().unwrap_or(0.0),
                target.uy.get(idx).copied().unwrap_or(0.0),
                target.uz.get(idx).copied().unwrap_or(0.0),
                target.sxx.get(idx).copied().unwrap_or(0.0),
                target.syy.get(idx).copied().unwrap_or(0.0),
                target.szz.get(idx).copied().unwrap_or(0.0),
                target.sxy.get(idx).copied().unwrap_or(0.0),
                target.sxz.get(idx).copied().unwrap_or(0.0),
                target.syz.get(idx).copied().unwrap_or(0.0),
                target.von_mises.get(idx).copied().unwrap_or(0.0),
                target.max_principal.get(idx).copied().unwrap_or(0.0),
            ]);
            base_fields.extend_from_slice(&[
                batch.base_prediction.ux.get(idx).copied().unwrap_or(0.0)
                    * calibration.displacement_scale,
                batch.base_prediction.uy.get(idx).copied().unwrap_or(0.0)
                    * calibration.displacement_scale,
                batch.base_prediction.uz.get(idx).copied().unwrap_or(0.0)
                    * calibration.displacement_scale,
                batch.base_prediction.sxx.get(idx).copied().unwrap_or(0.0)
                    * calibration.stress_scale,
                batch.base_prediction.syy.get(idx).copied().unwrap_or(0.0)
                    * calibration.stress_scale,
                batch.base_prediction.szz.get(idx).copied().unwrap_or(0.0)
                    * calibration.stress_scale,
                batch.base_prediction.sxy.get(idx).copied().unwrap_or(0.0)
                    * calibration.stress_scale,
                batch.base_prediction.sxz.get(idx).copied().unwrap_or(0.0)
                    * calibration.stress_scale,
                batch.base_prediction.syz.get(idx).copied().unwrap_or(0.0)
                    * calibration.stress_scale,
                batch
                    .base_prediction
                    .von_mises
                    .get(idx)
                    .copied()
                    .unwrap_or(0.0)
                    * calibration.stress_scale,
                batch
                    .base_prediction
                    .max_principal
                    .get(idx)
                    .copied()
                    .unwrap_or(0.0)
                    * calibration.stress_scale,
            ]);
            let base_stress_floor = characteristic_stress_scale.max(1.0);
            let full_correction_scales = [
                batch.correction_scales.get(scale_start).copied().unwrap_or(Self::displacement_scale_fallback(
                    target.ux.get(idx).copied().unwrap_or(0.0),
                    batch.base_prediction.ux.get(idx).copied().unwrap_or(0.0)
                        * calibration.displacement_scale,
                    characteristic_disp_scale,
                )),
                batch
                    .correction_scales
                    .get(scale_start + 1)
                    .copied()
                    .unwrap_or(Self::displacement_scale_fallback(
                        target.uy.get(idx).copied().unwrap_or(0.0),
                        batch.base_prediction.uy.get(idx).copied().unwrap_or(0.0)
                            * calibration.displacement_scale,
                        characteristic_disp_scale,
                    )),
                batch
                    .correction_scales
                    .get(scale_start + 2)
                    .copied()
                    .unwrap_or(Self::displacement_scale_fallback(
                        target.uz.get(idx).copied().unwrap_or(0.0),
                        batch.base_prediction.uz.get(idx).copied().unwrap_or(0.0)
                            * calibration.displacement_scale,
                        characteristic_disp_scale,
                    )),
                (target.sxx.get(idx).copied().unwrap_or(0.0)
                    - batch.base_prediction.sxx.get(idx).copied().unwrap_or(0.0)
                        * calibration.stress_scale)
                    .abs()
                    .max(base_stress_floor * 0.35),
                (target.syy.get(idx).copied().unwrap_or(0.0)
                    - batch.base_prediction.syy.get(idx).copied().unwrap_or(0.0)
                        * calibration.stress_scale)
                    .abs()
                    .max(base_stress_floor * 0.22),
                (target.szz.get(idx).copied().unwrap_or(0.0)
                    - batch.base_prediction.szz.get(idx).copied().unwrap_or(0.0)
                        * calibration.stress_scale)
                    .abs()
                    .max(base_stress_floor * 0.22),
                (target.sxy.get(idx).copied().unwrap_or(0.0)
                    - batch.base_prediction.sxy.get(idx).copied().unwrap_or(0.0)
                        * calibration.stress_scale)
                    .abs()
                    .max(base_stress_floor * 0.16),
                (target.sxz.get(idx).copied().unwrap_or(0.0)
                    - batch.base_prediction.sxz.get(idx).copied().unwrap_or(0.0)
                        * calibration.stress_scale)
                    .abs()
                    .max(base_stress_floor * 0.12),
                (target.syz.get(idx).copied().unwrap_or(0.0)
                    - batch.base_prediction.syz.get(idx).copied().unwrap_or(0.0)
                        * calibration.stress_scale)
                    .abs()
                    .max(base_stress_floor * 0.12),
                (target.von_mises.get(idx).copied().unwrap_or(0.0)
                    - batch.base_prediction.von_mises.get(idx).copied().unwrap_or(0.0)
                        * calibration.stress_scale)
                    .abs()
                    .max(base_stress_floor * 0.35),
                (target.max_principal.get(idx).copied().unwrap_or(0.0)
                    - batch.base_prediction.max_principal.get(idx).copied().unwrap_or(0.0)
                        * calibration.stress_scale)
                    .abs()
                    .max(base_stress_floor * 0.35),
            ];
            correction_scales.extend_from_slice(&full_correction_scales);
        }
        let (
            observable_target,
            observable_scale,
            observable_weight,
            observable_projection,
            observable_fifth_uses_vm,
        ) = Self::build_observable_bundle(
            &sample.case,
            &config.operator_grid,
            &target,
            &batch.base_prediction,
            &correction_scales,
            &batch.mask,
            batch.output_dim == PINO_DISPLACEMENT_OUTPUT_CHANNELS,
        );
        let max_target_vm = target
            .von_mises
            .iter()
            .take(cells)
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let vm_activity_floor = (max_target_vm * 0.05).max(1.0);
        let mut benchmark_vm_active = vec![0.0; cells];
        let active_vm_count = (0..cells)
            .filter(|idx| {
                batch.mask.get(*idx).copied().unwrap_or(0.0) > 0.0
                    && target.von_mises.get(*idx).copied().unwrap_or(0.0).abs() >= vm_activity_floor
            })
            .count();
        let vm_mask_scale = if active_vm_count == 0 {
            1.0
        } else {
            ((cells as f64) / (active_vm_count as f64)).sqrt()
        };
        for idx in 0..cells {
            if batch.mask.get(idx).copied().unwrap_or(0.0) > 0.0
                && target.von_mises.get(idx).copied().unwrap_or(0.0).abs() >= vm_activity_floor
            {
                benchmark_vm_active[idx] = vm_mask_scale;
            }
        }
        let max_target_sxx = target
            .sxx
            .iter()
            .take(cells)
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let sxx_peak_floor = (max_target_sxx * 0.85).max(1.0);
        let mut benchmark_sxx_peak = vec![0.0; cells];
        let active_sxx_count = (0..cells)
            .filter(|idx| {
                batch.mask.get(*idx).copied().unwrap_or(0.0) > 0.0
                    && target.sxx.get(*idx).copied().unwrap_or(0.0).abs() >= sxx_peak_floor
            })
            .count();
        let sxx_mask_scale = if active_sxx_count == 0 {
            1.0
        } else {
            ((cells as f64) / (active_sxx_count as f64)).sqrt()
        };
        for idx in 0..cells {
            if batch.mask.get(idx).copied().unwrap_or(0.0) > 0.0
                && target.sxx.get(idx).copied().unwrap_or(0.0).abs() >= sxx_peak_floor
            {
                benchmark_sxx_peak[idx] = sxx_mask_scale;
            }
        }
        let vm_residual = (0..cells)
            .map(|idx| {
                let target_vm = target.von_mises.get(idx).copied().unwrap_or(0.0).abs();
                let base_vm = batch
                    .base_prediction
                    .von_mises
                    .get(idx)
                    .copied()
                    .unwrap_or(0.0)
                    .abs()
                    * calibration.stress_scale;
                (target_vm - base_vm).abs() / target_vm.max(1.0)
            })
            .collect::<Vec<_>>();
        let mut sorted_vm_residual = vm_residual.clone();
        sorted_vm_residual.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let hotspot_threshold = if sorted_vm_residual.is_empty() {
            0.0
        } else {
            let idx = ((sorted_vm_residual.len() as f64) * 0.90).floor() as usize;
            sorted_vm_residual[idx.min(sorted_vm_residual.len().saturating_sub(1))]
        };
        let mut hotspot_centers = Vec::new();
        let mut ranked_hotspots = vm_residual
            .iter()
            .copied()
            .enumerate()
            .filter(|(idx, residual)| {
                batch.mask.get(*idx).copied().unwrap_or(0.0) > 0.0 && *residual >= hotspot_threshold
            })
            .collect::<Vec<_>>();
        ranked_hotspots.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let hotspot_separation = 0.12;
        for (idx, residual) in ranked_hotspots.into_iter() {
            if hotspot_centers.len() >= 2 {
                break;
            }
            let hz = idx / (config.operator_grid.nx * config.operator_grid.ny);
            let hy = (idx / config.operator_grid.nx) % config.operator_grid.ny;
            let hx = idx % config.operator_grid.nx;
            let center = (
                if config.operator_grid.nx <= 1 {
                    0.0
                } else {
                    hx as f64 / (config.operator_grid.nx - 1) as f64
                },
                if config.operator_grid.ny <= 1 {
                    0.0
                } else {
                    hy as f64 / (config.operator_grid.ny - 1) as f64
                },
                if config.operator_grid.nz <= 1 {
                    0.5
                } else {
                    hz as f64 / (config.operator_grid.nz - 1) as f64
                },
                residual,
            );
            let separated = hotspot_centers
                .iter()
                .all(|(cx, cy, cz, _): &(f64, f64, f64, f64)| {
                    let dx = center.0 - *cx;
                    let dy = center.1 - *cy;
                    let dz = center.2 - *cz;
                    (dx * dx + dy * dy + dz * dz).sqrt() >= hotspot_separation
                });
            if separated {
                hotspot_centers.push(center);
            }
        }
        let mut traction_mask = vec![0.0; cells];
        let mut traction_normal = vec![0.0; cells * 3];
        let mut traction_target = vec![0.0; cells * 3];
        let mut stress_focus = vec![1.0; cells];
        let mut ring_loss_mask = vec![0.0; cells];
        let length = sample.case.geometry.length_in.max(1e-9);
        let width = sample.case.geometry.width_in.max(1e-9);
        let thickness = sample.case.geometry.thickness_in.max(1e-9);
        let x_eps = length / config.operator_grid.nx.saturating_sub(1).max(1) as f64 * 0.75;
        let y_eps = width / config.operator_grid.ny.saturating_sub(1).max(1) as f64 * 0.75;
        let z_eps = thickness / config.operator_grid.nz.saturating_sub(1).max(1) as f64 * 0.75;
        let area_yz = (width * thickness).max(1e-9);
        let vertical_patch_width = (width * 0.20).max(y_eps * 2.0);
        let vertical_patch_thickness = (thickness * 0.45).max(z_eps * 2.0);
        let vertical_patch_area = (vertical_patch_width * vertical_patch_thickness).max(1e-9);
        let hole_radius = sample.case.geometry.hole_diameter_in.unwrap_or(0.0).abs() * 0.5;
        let hole_center_x = sample.case.geometry.length_in * 0.5;
        let hole_center_y = sample.case.geometry.width_in * 0.5;
        for z in 0..config.operator_grid.nz {
            for y in 0..config.operator_grid.ny {
                for x in 0..config.operator_grid.nx {
                    let idx = z * config.operator_grid.nx * config.operator_grid.ny
                        + y * config.operator_grid.nx
                        + x;
                    if batch.mask.get(idx).copied().unwrap_or(0.0) <= 0.0 {
                        continue;
                    }
                    let xn = if config.operator_grid.nx <= 1 {
                        0.0
                    } else {
                        x as f64 / (config.operator_grid.nx - 1) as f64
                    };
                    let yn = if config.operator_grid.ny <= 1 {
                        0.0
                    } else {
                        y as f64 / (config.operator_grid.ny - 1) as f64
                    };
                    let zn = if config.operator_grid.nz <= 1 {
                        0.5
                    } else {
                        z as f64 / (config.operator_grid.nz - 1) as f64
                    };
                    let px = xn * sample.case.geometry.length_in;
                    let py = yn * sample.case.geometry.width_in;
                    let pz = zn * sample.case.geometry.thickness_in;
                    if hole_radius > 0.0 {
                        let dx = px - hole_center_x;
                        let dy = py - hole_center_y;
                        let radial = (dx * dx + dy * dy).sqrt();
                        let angular_focus = if radial > 1e-9 {
                            (dy.abs() / radial).clamp(0.0, 1.0).powi(2)
                        } else {
                            0.0
                        };
                        let ring_band = (hole_radius * 0.30)
                            .max(sample.case.geometry.width_in * 0.04)
                            .max(1e-6);
                        let ring_focus =
                            (-((radial - hole_radius).powi(2)) / (ring_band * ring_band)).exp();
                        let ring_narrow_band = (hole_radius * 0.18)
                            .max(sample.case.geometry.width_in * 0.02)
                            .max(1e-6);
                        let ring_narrow = (-((radial - hole_radius).powi(2))
                            / (ring_narrow_band * ring_narrow_band))
                            .exp()
                            * angular_focus;
                        let center_x_focus = (-((px - hole_center_x).powi(2))
                            / (sample.case.geometry.length_in * 0.14).max(1e-6).powi(2))
                        .exp();
                        let thickness_focus = (-((pz - sample.case.geometry.thickness_in * 0.5)
                            .powi(2))
                            / (sample.case.geometry.thickness_in * 0.32).max(1e-6).powi(2))
                        .exp();
                        let local_focus = ring_focus
                            * (0.25 + 0.75 * angular_focus)
                            * (0.72 + 0.28 * center_x_focus)
                            * thickness_focus;
                        let hole_focus_gain =
                            env_f64("PINO_HOLE_STRESS_FOCUS_GAIN", 3.5).clamp(0.0, 8.0);
                        let hole_ring_narrow_gain =
                            env_f64("PINO_HOLE_RING_NARROW_GAIN", 0.4).clamp(0.0, 4.0);
                        stress_focus[idx] = (1.0 + hole_focus_gain * local_focus.clamp(0.0, 1.0))
                            * (1.0 + hole_ring_narrow_gain * ring_narrow);
                        // Ring loss mask: emphasize the narrow band where analytical stress peaks.
                        ring_loss_mask[idx] = ring_narrow;
                    }
                    let residual_focus = if hotspot_threshold > 0.0 {
                        ((vm_residual[idx] - hotspot_threshold).max(0.0)
                            / hotspot_threshold.max(1.0e-9))
                        .clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    if residual_focus > 0.0 {
                        let residual_focus_gain =
                            env_f64("PINO_RESIDUAL_FOCUS_GAIN", 2.0).clamp(0.0, 8.0);
                        stress_focus[idx] *= 1.0 + residual_focus_gain * residual_focus;
                        ring_loss_mask[idx] = ring_loss_mask[idx].max(residual_focus);
                    }
                    if !hotspot_centers.is_empty() {
                        let mut patch_focus = 0.0_f64;
                        for (cx, cy, cz, residual) in hotspot_centers.iter().copied() {
                            let dx = xn - cx;
                            let dy = yn - cy;
                            let dz = zn - cz;
                            let radius_xy = 0.10;
                            let radius_z = 0.18;
                            let normalized = (dx * dx + dy * dy) / (radius_xy * radius_xy)
                                + (dz * dz) / (radius_z * radius_z);
                            let local = (-normalized).exp() * residual.clamp(0.0, 1.0);
                            if local > patch_focus {
                                patch_focus = local;
                            }
                        }
                        if patch_focus > 1.0e-6 {
                            let patch_focus_gain_default =
                                env_f64("PINO_PATCH_FOCUS_GAIN", 1.8).clamp(0.0, 8.0);
                            let patch_ring_gain_default =
                                env_f64("PINO_PATCH_RING_GAIN", 0.85).clamp(0.0, 4.0);
                            let patch_focus_gain = if hole_radius > 0.0 {
                                env_f64("PINO_PATCH_FOCUS_GAIN_HOLE", patch_focus_gain_default)
                                    .clamp(0.0, 8.0)
                            } else {
                                env_f64("PINO_PATCH_FOCUS_GAIN_NON_HOLE", patch_focus_gain_default)
                                    .clamp(0.0, 8.0)
                            };
                            let patch_ring_gain = if hole_radius > 0.0 {
                                env_f64("PINO_PATCH_RING_GAIN_HOLE", patch_ring_gain_default)
                                    .clamp(0.0, 4.0)
                            } else {
                                env_f64("PINO_PATCH_RING_GAIN_NON_HOLE", patch_ring_gain_default)
                                    .clamp(0.0, 4.0)
                            };
                            stress_focus[idx] *= 1.0 + patch_focus_gain * patch_focus;
                            ring_loss_mask[idx] = ring_loss_mask[idx]
                                .max((patch_ring_gain * patch_focus).clamp(0.0, 1.0));
                        }
                    }
                    let on_x_min = px <= x_eps;
                    let on_x_max = (sample.case.geometry.length_in - px).abs() <= x_eps;
                    let on_y_min = py <= y_eps;
                    let on_y_max = (sample.case.geometry.width_in - py).abs() <= y_eps;
                    let on_z_min = pz <= z_eps;
                    let on_z_max = (sample.case.geometry.thickness_in - pz).abs() <= z_eps;
                    let is_clamped = batch.clamp.get(idx).copied().unwrap_or(0.0) > 0.5;
                    if is_clamped {
                        continue;
                    }
                    let vertical_loaded = sample.case.load.vertical_point_load_lbf.abs() > 1e-9
                        && on_x_max
                        && (sample.case.geometry.width_in - py) <= vertical_patch_width
                        && (pz - sample.case.geometry.thickness_in * 0.5).abs()
                            <= vertical_patch_thickness * 0.5;
                    let axial_loaded = sample.case.load.axial_load_lbf.abs() > 1e-9 && on_x_max;
                    let is_loaded = !sample.case.boundary_conditions.fix_end_face
                        && (vertical_loaded || axial_loaded);
                    let mut normal = [0.0_f64; 3];
                    if is_loaded {
                        normal = [1.0, 0.0, 0.0];
                    } else {
                        let mut best = f64::INFINITY;
                        let mut set_normal = |dist: f64, nx: f64, ny: f64, nz: f64| {
                            if dist < best {
                                best = dist;
                                normal = [nx, ny, nz];
                            }
                        };
                        if on_x_min && !sample.case.boundary_conditions.fix_start_face {
                            set_normal(px, -1.0, 0.0, 0.0);
                        }
                        if on_x_max && !sample.case.boundary_conditions.fix_end_face {
                            set_normal((sample.case.geometry.length_in - px).abs(), 1.0, 0.0, 0.0);
                        }
                        if on_y_min {
                            set_normal(py, 0.0, -1.0, 0.0);
                        }
                        if on_y_max {
                            set_normal((sample.case.geometry.width_in - py).abs(), 0.0, 1.0, 0.0);
                        }
                        if on_z_min {
                            set_normal(pz, 0.0, 0.0, -1.0);
                        }
                        if on_z_max {
                            set_normal(
                                (sample.case.geometry.thickness_in - pz).abs(),
                                0.0,
                                0.0,
                                1.0,
                            );
                        }
                        if hole_radius > 0.0 {
                            let dx = px - hole_center_x;
                            let dy = py - hole_center_y;
                            let radial = (dx * dx + dy * dy).sqrt();
                            let hole_band = x_eps.min(y_eps).max(1e-6);
                            if (radial - hole_radius).abs() <= hole_band && radial > 1e-9 {
                                set_normal(
                                    (radial - hole_radius).abs(),
                                    dx / radial,
                                    dy / radial,
                                    0.0,
                                );
                            }
                        }
                    }
                    if normal[0].abs() + normal[1].abs() + normal[2].abs() <= 1e-9 {
                        continue;
                    }
                    let offset = idx * 3;
                    traction_mask[idx] = 1.0;
                    traction_normal[offset] = normal[0];
                    traction_normal[offset + 1] = normal[1];
                    traction_normal[offset + 2] = normal[2];
                    if is_loaded {
                        traction_target[offset] = if axial_loaded {
                            sample.case.load.axial_load_lbf / area_yz
                        } else {
                            0.0
                        };
                        traction_target[offset + 1] = if vertical_loaded {
                            sample.case.load.vertical_point_load_lbf / vertical_patch_area
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
        BurnPhysicsSample {
            features: batch.features.clone(),
            target_fields,
            base_fields,
            correction_scales,
            output_dim,
            characteristic_disp_scale,
            characteristic_stress_scale,
            observable_target,
            observable_scale,
            observable_weight,
            observable_projection,
            observable_fifth_uses_vm,
            benchmark_vm_active,
            benchmark_sxx_peak,
            stress_focus,
            ring_loss_mask,
            mask: batch.mask.clone(),
            clamp: batch.clamp.clone(),
            displacement_embed: batch.displacement_embed.clone(),
            traction_mask,
            traction_normal,
            traction_target,
            grid_nx: config.operator_grid.nx,
            grid_ny: config.operator_grid.ny,
            grid_nz: config.operator_grid.nz,
            spectral_modes: config.spectral_modes,
            dx: sample.case.geometry.length_in
                / (config.operator_grid.nx.saturating_sub(1).max(1) as f64),
            dy: sample.case.geometry.width_in
                / (config.operator_grid.ny.saturating_sub(1).max(1) as f64),
            dz: sample.case.geometry.thickness_in
                / (config.operator_grid.nz.saturating_sub(1).max(1) as f64),
            e_modulus: sample.case.material.e_psi,
            poisson: sample.case.material.nu,
        }
    }

    fn cantilever_characteristic_scales(case: &SolveInput) -> (f64, f64) {
        let is_cantilever = case.boundary_conditions.fix_start_face
            && !case.boundary_conditions.fix_end_face
            && case.load.vertical_point_load_lbf.abs() > 0.0
            && case.load.axial_load_lbf.abs() <= 1e-9;
        if !is_cantilever {
            return (0.0, 0.0);
        }

        let load = case.load.vertical_point_load_lbf.abs().max(1e-9);
        let length = case.geometry.length_in.abs().max(1e-6);
        let section_height = case.geometry.width_in.abs().max(1e-6);
        let section_thickness = case.geometry.thickness_in.abs().max(1e-6);
        let e_modulus = case.material.e_psi.abs().max(1.0);
        let inertia = (section_thickness * section_height.powi(3) / 12.0).max(1e-9);
        let characteristic_stress =
            (load * length * (section_height * 0.5) / inertia).abs().max(1.0);
        let characteristic_disp = (load * length.powi(3) / (3.0 * e_modulus * inertia))
            .abs()
            .max(1e-6);
        (characteristic_disp, characteristic_stress)
    }

    fn build_burn_physics_sample_cores(
        targets: &[OperatorFieldTarget],
        config: &PinoModelConfig,
        params: &OperatorTrainableParams,
    ) -> Vec<BurnPhysicsSampleCore> {
        targets
            .par_iter()
            .map(|sample| Self::build_burn_physics_sample_core(sample, config, params))
            .collect()
    }

    fn breakdown_to_pillars(breakdown: BurnPhysicsLossBreakdown) -> ResidualPillars {
        ResidualPillars {
            momentum: breakdown.equilibrium.max(0.0),
            kinematics: (breakdown.displacement_fit + breakdown.auxiliary_data).max(0.0),
            material: (breakdown.constitutive + breakdown.weak_energy).max(0.0),
            boundary: breakdown.boundary.max(0.0),
        }
    }

    fn dominant_residual_focus(breakdown: BurnPhysicsLossBreakdown) -> PinoResidualFocus {
        let mut best = ("balanced", 0.0_f64);
        let candidates = [
            ("global", breakdown.equilibrium.max(0.0)),
            (
                "local",
                (breakdown.stress_fit
                    + breakdown.observable
                    + breakdown.constitutive_normal
                    + breakdown.constitutive_shear
                    + breakdown.weak_energy
                    + breakdown.invariant)
                    .max(0.0),
            ),
            (
                "boundary",
                (breakdown.boundary + breakdown.displacement_fit * 0.35).max(0.0),
            ),
            (
                "kinematic",
                (breakdown.displacement_fit + breakdown.auxiliary_data * 0.5).max(0.0),
            ),
        ];
        for (name, value) in candidates {
            if value > best.1 {
                best = (name, value);
            }
        }
        match best.0 {
            "global" => PinoResidualFocus::Global,
            "local" => PinoResidualFocus::Local,
            "boundary" => PinoResidualFocus::Boundary,
            "kinematic" => PinoResidualFocus::Kinematic,
            _ => PinoResidualFocus::Balanced,
        }
    }

    fn residual_focus_label(focus: PinoResidualFocus) -> &'static str {
        match focus {
            PinoResidualFocus::Global => "global-equilibrium",
            PinoResidualFocus::Local => "local-material",
            PinoResidualFocus::Boundary => "boundary-clamp",
            PinoResidualFocus::Kinematic => "kinematic-fit",
            PinoResidualFocus::Balanced => "balanced",
        }
    }


    fn train_operator_field_head_burn(
        targets: &[OperatorFieldTarget],
        config: &PinoModelConfig,
        calibration: &OperatorCalibration,
        current: &OperatorTrainableParams,
        steps: usize,
        learning_rate: f64,
        optimizer: BurnFieldHeadOptimizer,
        loss_weights: (f64, f64, f64, f64),
        exact_surface: bool,
    ) -> Option<(OperatorTrainableParams, f64)> {
        if targets.is_empty() {
            return None;
        }
        let cores = Self::build_burn_physics_sample_cores(targets, config, current);
        Self::train_operator_field_head_burn_with_cores(
            targets,
            &cores,
            config,
            calibration,
            current,
            steps,
            learning_rate,
            optimizer,
            loss_weights,
            exact_surface,
        )
    }

    fn train_operator_field_head_burn_with_cores(
        targets: &[OperatorFieldTarget],
        cores: &[BurnPhysicsSampleCore],
        config: &PinoModelConfig,
        calibration: &OperatorCalibration,
        current: &OperatorTrainableParams,
        steps: usize,
        learning_rate: f64,
        optimizer: BurnFieldHeadOptimizer,
        loss_weights: (f64, f64, f64, f64),
        exact_surface: bool,
    ) -> Option<(OperatorTrainableParams, f64)> {
        if targets.is_empty() || targets.len() != cores.len() {
            return None;
        }
        let train_cut = if targets.len() > 1 {
            targets.len().saturating_sub(1)
        } else {
            1
        };
        let physics_samples = targets
            .par_iter()
            .zip(cores.par_iter())
            .take(train_cut)
            .map(|(sample, core)| {
                Self::build_burn_physics_sample_from_core(sample, config, calibration, core)
            })
            .collect::<Vec<_>>();

        Self::train_operator_field_head_burn_with_samples(
            &physics_samples,
            config,
            current,
            steps,
            learning_rate,
            optimizer,
            loss_weights,
            exact_surface,
            false,
        )
    }

    fn train_operator_field_head_burn_with_samples(
        physics_samples: &[BurnPhysicsSample],
        config: &PinoModelConfig,
        current: &OperatorTrainableParams,
        steps: usize,
        learning_rate: f64,
        optimizer: BurnFieldHeadOptimizer,
        loss_weights: (f64, f64, f64, f64),
        exact_surface: bool,
        characteristic_train_scaling: bool,
    ) -> Option<(OperatorTrainableParams, f64)> {
        let output_dim = config.operator_grid.output_channels;
        let current = current.clone().aligned_to_config(config);
        if physics_samples.is_empty() || !current.matches_shape(PINO_FIELD_HEAD_BASIS, output_dim) {
            return None;
        }

        let outcome = train_operator_field_head_physics(
            &physics_samples,
            &current.field_head_weights,
            &current.field_head_bias,
            &current.field_head_activation,
            PINO_FIELD_HEAD_BASIS,
            output_dim,
            current.field_head_hidden_layers,
            current.field_head_hidden_width,
            steps,
            learning_rate,
            optimizer,
            loss_weights,
            exact_surface,
            characteristic_train_scaling,
        )?;
        let mut next = current.clone().aligned_to_config(config);
        next.field_head_weights = outcome.weights;
        next.field_head_bias = outcome.bias;
        next.field_head_activation = outcome.activation;
        Some((next.clamped(), outcome.breakdown.total))
    }

    #[allow(dead_code)]
    pub(crate) fn direct_train_operator_params_for_cases(
        cases: &[SolveInput],
        config: &PinoModelConfig,
        calibration: &OperatorCalibration,
        current: &OperatorTrainableParams,
        steps: usize,
        learning_rate: f64,
        optimizer: BurnFieldHeadOptimizer,
        loss_weights: (f64, f64, f64, f64),
    ) -> Option<(OperatorTrainableParams, f64)> {
        if cases.is_empty() {
            return None;
        }
        let targets = Self::build_operator_field_targets_exact(cases, config.operator_grid.clone());
        Self::train_operator_field_head_burn(
            &targets,
            config,
            calibration,
            current,
            steps,
            learning_rate,
            optimizer,
            loss_weights,
            false,
        )
    }

    fn direct_train_operator_params_for_targets_with_scaling(
        targets: &[OperatorFieldTarget],
        config: &PinoModelConfig,
        calibration: &OperatorCalibration,
        current: &OperatorTrainableParams,
        steps: usize,
        learning_rate: f64,
        optimizer: BurnFieldHeadOptimizer,
        loss_weights: (f64, f64, f64, f64),
        exact_surface: bool,
        characteristic_train_scaling: bool,
    ) -> Option<(OperatorTrainableParams, f64)> {
        if targets.is_empty() {
            return None;
        }
        let cores = Self::build_burn_physics_sample_cores(targets, config, current);
        let physics_samples = targets
            .iter()
            .zip(cores.iter())
            .map(|(sample, core)| {
                Self::build_burn_physics_sample_from_core(sample, config, calibration, core)
            })
            .collect::<Vec<_>>();
        Self::train_operator_field_head_burn_with_samples(
            &physics_samples,
            config,
            current,
            steps,
            learning_rate,
            optimizer,
            loss_weights,
            exact_surface,
            characteristic_train_scaling,
        )
    }

    fn operator_param_delta_norm(
        current: &OperatorTrainableParams,
        baseline: &OperatorTrainableParams,
    ) -> f64 {
        let weight_norm = current
            .field_head_weights
            .iter()
            .zip(baseline.field_head_weights.iter())
            .map(|(current, baseline)| {
                let delta = current - baseline;
                delta * delta
            })
            .sum::<f64>();
        let bias_norm = current
            .field_head_bias
            .iter()
            .zip(baseline.field_head_bias.iter())
            .map(|(current, baseline)| {
                let delta = current - baseline;
                delta * delta
            })
            .sum::<f64>();
        let activation_norm = current
            .field_head_activation
            .iter()
            .zip(baseline.field_head_activation.iter())
            .map(|(current, baseline)| {
                let delta = current - baseline;
                delta * delta
            })
            .sum::<f64>();
        (weight_norm + bias_norm + activation_norm).sqrt()
    }

    fn calibration_ratio(target: f64, model: f64) -> f64 {
        if !target.is_finite() || !model.is_finite() {
            return 1.0;
        }
        if target.abs() <= 1e-9 {
            1.0
        } else {
            (target.abs() / model.abs().max(1e-9)).clamp(0.25, 4.0)
        }
    }

    fn holdout_selection_score(summary: &HoldoutValidationSummary) -> f64 {
        let limits = (
            summary.mean_error_limit.max(1e-9),
            summary.p95_error_limit.max(1e-9),
            summary.residual_ratio_limit.max(1e-9),
        );
        let disp = summary.mean_displacement_error / limits.0;
        let vm = summary.mean_von_mises_error / limits.0;
        let p95 = summary.p95_field_error / limits.1;
        let ratio = summary.residual_ratio / limits.2;
        (disp * 0.95 + vm * 1.10 + p95 * 1.35 + ratio * 0.35).max(0.0)
    }

    fn hotspot_selection_score(
        summary: Option<&HoldoutValidationSummary>,
        val_breakdown: BurnPhysicsLossBreakdown,
        val_loss: f64,
    ) -> f64 {
        let gate_score = summary
            .map(Self::holdout_selection_score)
            .unwrap_or(f64::MAX * 0.25);
        let hotspot_mix = (val_breakdown.stress_fit * 1.20
            + val_breakdown.observable * 1.35
            + val_breakdown.weak_energy * 1.10
            + val_breakdown.invariant * 0.55
            + val_breakdown.constitutive_normal * 0.85
            + val_breakdown.constitutive_shear * 0.85)
            .max(0.0);
        let boundary_mix =
            (val_breakdown.boundary * 0.45 + val_breakdown.displacement_fit * 0.20).max(0.0);
        (gate_score * 1.0 + hotspot_mix * 0.45 + boundary_mix * 0.20 + val_loss.max(0.0) * 0.08)
            .max(0.0)
    }

    fn normalize_projection(weights: &mut [f64]) {
        let sum = weights
            .iter()
            .copied()
            .filter(|value| value.is_finite() && *value > 0.0)
            .sum::<f64>();
        if sum <= 1e-12 {
            let uniform = 1.0 / (weights.len().max(1) as f64);
            for value in weights.iter_mut() {
                *value = uniform;
            }
            return;
        }
        for value in weights.iter_mut() {
            *value = if value.is_finite() && *value > 0.0 {
                *value / sum
            } else {
                0.0
            };
        }
    }

    fn weighted_channel_mean(values: &[f64], weights: &[f64]) -> f64 {
        values
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .map(|(value, weight)| value * weight)
            .sum::<f64>()
    }

    fn weighted_scale_mean(flat: &[f64], channel: usize, channels: usize, weights: &[f64]) -> f64 {
        weights
            .iter()
            .enumerate()
            .map(|(idx, weight)| {
                flat.get(idx * channels + channel).copied().unwrap_or(0.0) * *weight
            })
            .sum::<f64>()
    }

    fn build_observable_bundle(
        case: &crate::contracts::SolveInput,
        grid: &crate::contracts::OperatorGridSpec,
        target: &OperatorPrediction,
        base: &OperatorPrediction,
        correction_scales: &[f64],
        mask: &[f64],
        benchmark_displacement_primary: bool,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, bool) {
        let cells = grid.nx * grid.ny * grid.nz;
        let mut tip_uy_projection = vec![0.0; cells];
        let mut end_ux_projection = vec![0.0; cells];
        let mut hotspot_vm_projection = vec![0.0; cells];
        let mut hotspot_principal_projection = vec![0.0; cells];
        let mut root_sigma_projection = vec![0.0; cells];
        let axial = case.load.axial_load_lbf.abs();
        let vertical = case.load.vertical_point_load_lbf.abs();
        let total_load = axial + vertical + 1e-9;
        let axial_ratio = axial / total_load;
        let vertical_ratio = vertical / total_load;
        let hole_radius = case.geometry.hole_diameter_in.unwrap_or(0.0).abs() * 0.5;
        let cantilever_fifth_observable_span_blend =
            env_f64("PINO_CANTILEVER_FIFTH_OBSERVABLE_SPAN_BLEND", 0.0).clamp(0.0, 1.0);
        let cantilever_root_edge_scale =
            env_f64("PINO_CANTILEVER_ROOT_EDGE_SCALE", 1.0).clamp(0.0, 2.0);
        let hole_center_x = case.geometry.length_in * 0.5;
        let hole_center_y = case.geometry.width_in * 0.5;
        let hole_sigma = (hole_radius.max(case.geometry.width_in * 0.05)).max(1e-6);
        let x_sigma = (case.geometry.length_in * 0.12).max(1e-6);
        let z_sigma = (case.geometry.thickness_in * 0.18).max(1e-6);
        let yz_center_sigma =
            (case.geometry.width_in.min(case.geometry.thickness_in) * 0.22).max(1e-6);

        for idx in 0..cells {
            let x_idx = idx % grid.nx;
            let yz_idx = idx / grid.nx;
            let y_idx = yz_idx % grid.ny;
            let z_idx = yz_idx / grid.ny;
            let xn = if grid.nx <= 1 {
                0.0
            } else {
                x_idx as f64 / (grid.nx - 1) as f64
            };
            let yn = if grid.ny <= 1 {
                0.0
            } else {
                y_idx as f64 / (grid.ny - 1) as f64
            };
            let zn = if grid.nz <= 1 {
                0.5
            } else {
                z_idx as f64 / (grid.nz - 1) as f64
            };
            let x = xn * case.geometry.length_in;
            let y = yn * case.geometry.width_in;
            let z = zn * case.geometry.thickness_in;
            let mask_weight = mask.get(idx).copied().unwrap_or(1.0).clamp(0.0, 1.0);
            if mask_weight <= 0.0 {
                continue;
            }

            let end_face = xn.powi(8);
            let end_centerline = (1.0 - (2.0 * yn - 1.0).abs().powi(2)).max(0.0)
                * (1.0 - (2.0 * zn - 1.0).abs().powi(2)).max(0.0);
            let end_band = end_face * end_centerline;
            let root_edge = (1.0 - xn).powi(7)
                * ((yn.powi(6) + (1.0 - yn).powi(6)).clamp(0.0, 1.5)
                    + (zn.powi(6) + (1.0 - zn).powi(6)).clamp(0.0, 1.5))
                * 0.5;
            let root_center = (1.0 - xn).powi(7)
                * (1.0 - (2.0 * yn - 1.0).abs()).max(0.0).powi(4)
                * (1.0 - (2.0 * zn - 1.0).abs()).max(0.0).powi(4);
            let root_edge_scaled = if hole_radius <= 0.0 && vertical_ratio > axial_ratio {
                root_edge * cantilever_root_edge_scale
            } else {
                root_edge
            };
            let hole_ring = if hole_radius > 0.0 {
                let dx = x - hole_center_x;
                let dy = y - hole_center_y;
                let radial = (dx.powi(2) + dy.powi(2)).sqrt();
                let ring = (-((radial - hole_radius).powi(2)) / (hole_sigma * hole_sigma)).exp();
                let x_center = (-(dx.powi(2)) / (x_sigma * x_sigma)).exp();
                let angular_focus = if radial > 1e-9 {
                    (dy.abs() / radial).clamp(0.0, 1.0).powi(2)
                } else {
                    0.0
                };
                let thickness_center =
                    (-((z - case.geometry.thickness_in * 0.5).powi(2)) / (z_sigma * z_sigma)).exp();
                ring * (0.22 + 0.78 * angular_focus) * (0.70 + 0.30 * x_center) * thickness_center
            } else {
                0.0
            };

            let loaded_top_corner = end_face
                * yn.powi(8)
                * (-((z - case.geometry.thickness_in * 0.5).powi(2))
                    / (yz_center_sigma * yz_center_sigma))
                    .exp();
            let loaded_top_face =
                end_face * yn.powi(8) * (1.0 - (2.0 * zn - 1.0).abs().powi(2)).max(0.0);
            let root_top_face =
                (1.0 - xn).powi(8) * yn.powi(8) * (1.0 - (2.0 * zn - 1.0).abs().powi(2)).max(0.0);
            let cantilever_face_band = (1.0 - (2.0 * zn - 1.0).abs().powi(2)).max(0.0)
                * (yn.powi(6) + (1.0 - yn).powi(6)).clamp(0.0, 2.0)
                * (1.0 - (xn / 0.35).clamp(0.0, 1.0)).powi(2);
            let cantilever_fifth_sigma_band = (1.0 - (2.0 * zn - 1.0).abs().powi(2)).max(0.0)
                * (yn.powi(6) + (1.0 - yn).powi(6)).clamp(0.0, 2.0)
                * (1.0 - (xn / 0.40).clamp(0.0, 1.0)).powi(2);
            let cantilever_span_blend =
                env_f64("PINO_CANTILEVER_SPAN_BAND_BLEND", 0.0).clamp(0.0, 1.0);
            tip_uy_projection[idx] =
                mask_weight * (0.70 * loaded_top_corner + 0.30 * loaded_top_face).max(0.0);
            end_ux_projection[idx] =
                mask_weight * end_band.max(end_face * 0.65 * end_centerline.max(0.35));
            let stress_focus = if hole_radius > 0.0 {
                0.90 * hole_ring + 0.07 * root_edge + 0.03 * root_center
            } else if vertical_ratio > axial_ratio {
                let root_local = 0.80 * root_edge_scaled + 0.20 * root_center;
                let span_bending =
                    0.64 * cantilever_face_band + 0.22 * root_top_face + 0.14 * root_edge_scaled;
                (1.0 - cantilever_span_blend) * root_local + cantilever_span_blend * span_bending
            } else {
                0.46 * root_center + 0.36 * end_band + 0.18 * root_edge
            };
            hotspot_vm_projection[idx] = mask_weight * stress_focus.max(0.0);
            hotspot_principal_projection[idx] = mask_weight * stress_focus.max(0.0);
            root_sigma_projection[idx] = mask_weight
                * (if hole_radius > 0.0 {
                    (0.92 * hole_ring + 0.06 * end_band + 0.02 * root_edge).max(0.0)
                } else if vertical_ratio > axial_ratio {
                    let root_local = 0.88 * root_edge_scaled + 0.12 * root_top_face;
                    let span_bending = 0.74 * cantilever_face_band
                        + 0.16 * root_top_face
                        + 0.10 * root_edge_scaled;
                    let span_blended = (1.0 - cantilever_span_blend) * root_local
                        + cantilever_span_blend * span_bending;
                    ((1.0 - cantilever_fifth_observable_span_blend) * span_blended
                        + cantilever_fifth_observable_span_blend * cantilever_fifth_sigma_band)
                        .max(0.0)
                } else {
                    (0.55 * root_center + 0.45 * root_top_face).max(0.0)
                });
        }

        Self::normalize_projection(&mut tip_uy_projection);
        Self::normalize_projection(&mut end_ux_projection);
        Self::normalize_projection(&mut hotspot_vm_projection);
        Self::normalize_projection(&mut hotspot_principal_projection);
        Self::normalize_projection(&mut root_sigma_projection);

        let hole_vm_observable = hole_radius > 0.0;
        if benchmark_displacement_primary {
            let max_target_vm = target
                .von_mises
                .iter()
                .take(cells)
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            let vm_activity_floor = (max_target_vm * 0.05).max(1.0);
            let max_target_sxx = target
                .sxx
                .iter()
                .take(cells)
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            let sxx_peak_floor = (max_target_sxx * 0.85).max(1.0);
            let vm_active_projection = (0..cells)
                .map(|idx| {
                    if mask.get(idx).copied().unwrap_or(0.0) > 0.0
                        && target.von_mises.get(idx).copied().unwrap_or(0.0).abs()
                            >= vm_activity_floor
                    {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            let sxx_peak_projection = (0..cells)
                .map(|idx| {
                    if mask.get(idx).copied().unwrap_or(0.0) > 0.0
                        && target.sxx.get(idx).copied().unwrap_or(0.0).abs() >= sxx_peak_floor
                    {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            let mut max_disp_projection = tip_uy_projection.clone();
            let mut vm_active_projection = vm_active_projection;
            let mut principal_projection = vm_active_projection.clone();
            let mut sxx_peak_projection = sxx_peak_projection;
            Self::normalize_projection(&mut max_disp_projection);
            Self::normalize_projection(&mut vm_active_projection);
            Self::normalize_projection(&mut principal_projection);
            Self::normalize_projection(&mut sxx_peak_projection);
            let target_disp_mag = target
                .ux
                .iter()
                .zip(target.uy.iter())
                .zip(target.uz.iter())
                .map(|((ux, uy), uz)| (ux * ux + uy * uy + uz * uz).sqrt())
                .collect::<Vec<_>>();
            let base_disp_mag = base
                .ux
                .iter()
                .zip(base.uy.iter())
                .zip(base.uz.iter())
                .map(|((ux, uy), uz)| (ux * ux + uy * uy + uz * uz).sqrt())
                .collect::<Vec<_>>();
            let base_abs_sxx = base.sxx.iter().map(|value| value.abs()).collect::<Vec<_>>();
            let target_abs_sxx = target.sxx.iter().map(|value| value.abs()).collect::<Vec<_>>();
            let target_observable = vec![
                Self::weighted_channel_mean(&target.uy, &tip_uy_projection),
                Self::weighted_channel_mean(&target_disp_mag, &max_disp_projection),
                Self::weighted_channel_mean(&target.von_mises, &vm_active_projection),
                Self::weighted_channel_mean(&target.max_principal, &principal_projection),
                Self::weighted_channel_mean(&target_abs_sxx, &sxx_peak_projection),
            ];
            let disp_mag_scale = Self::weighted_scale_mean(
                correction_scales,
                1,
                PINO_OUTPUT_CHANNELS,
                &max_disp_projection,
            )
            .max(target_observable[1].abs() * 0.12)
            .max(Self::weighted_channel_mean(&base_disp_mag, &max_disp_projection).abs() * 0.10)
            .max(1e-6);
            let observable_scale = vec![
                Self::weighted_scale_mean(
                    correction_scales,
                    1,
                    PINO_OUTPUT_CHANNELS,
                    &tip_uy_projection,
                )
                .max(target_observable[0].abs() * 0.14)
                .max(Self::weighted_channel_mean(&base.uy, &tip_uy_projection).abs() * 0.12)
                .max(1e-6),
                disp_mag_scale,
                Self::weighted_scale_mean(
                    correction_scales,
                    9,
                    PINO_OUTPUT_CHANNELS,
                    &vm_active_projection,
                )
                .max(target_observable[2].abs() * 0.10)
                .max(Self::weighted_channel_mean(&base.von_mises, &vm_active_projection).abs() * 0.08)
                .max(1e-3),
                Self::weighted_scale_mean(
                    correction_scales,
                    10,
                    PINO_OUTPUT_CHANNELS,
                    &principal_projection,
                )
                .max(target_observable[3].abs() * 0.10)
                .max(
                    Self::weighted_channel_mean(&base.max_principal, &principal_projection).abs()
                        * 0.08,
                )
                .max(1e-3),
                Self::weighted_scale_mean(
                    correction_scales,
                    3,
                    PINO_OUTPUT_CHANNELS,
                    &sxx_peak_projection,
                )
                .max(target_observable[4].abs() * 0.12)
                .max(
                    Self::weighted_channel_mean(&base_abs_sxx, &sxx_peak_projection).abs() * 0.10,
                )
                .max(1e-3),
            ];
            let observable_weight = vec![1.40, 1.00, 0.15, 0.00, 0.10];
            let observable_projection = [
                tip_uy_projection,
                max_disp_projection,
                vm_active_projection,
                principal_projection,
                sxx_peak_projection,
            ]
            .concat();
            return (
                target_observable,
                observable_scale,
                observable_weight,
                observable_projection,
                false,
            );
        }

        let target_observable = vec![
            Self::weighted_channel_mean(&target.uy, &tip_uy_projection),
            Self::weighted_channel_mean(&target.ux, &end_ux_projection),
            Self::weighted_channel_mean(&target.von_mises, &hotspot_vm_projection),
            Self::weighted_channel_mean(&target.max_principal, &hotspot_principal_projection),
            if hole_vm_observable {
                Self::weighted_channel_mean(&target.von_mises, &root_sigma_projection)
            } else {
                Self::weighted_channel_mean(
                    &target
                        .sxx
                        .iter()
                        .map(|value| value.abs())
                        .collect::<Vec<_>>(),
                    &root_sigma_projection,
                )
            },
        ];
        let observable_scale = vec![
            Self::weighted_scale_mean(
                correction_scales,
                1,
                PINO_OUTPUT_CHANNELS,
                &tip_uy_projection,
            )
            .max(target_observable[0].abs() * 0.14)
            .max(Self::weighted_channel_mean(&base.uy, &tip_uy_projection).abs() * 0.12)
            .max(1e-6),
            Self::weighted_scale_mean(
                correction_scales,
                0,
                PINO_OUTPUT_CHANNELS,
                &end_ux_projection,
            )
            .max(target_observable[1].abs() * 0.12)
            .max(Self::weighted_channel_mean(&base.ux, &end_ux_projection).abs() * 0.10)
            .max(1e-6),
            Self::weighted_scale_mean(
                correction_scales,
                9,
                PINO_OUTPUT_CHANNELS,
                &hotspot_vm_projection,
            )
            .max(target_observable[2].abs() * 0.10)
            .max(Self::weighted_channel_mean(&base.von_mises, &hotspot_vm_projection).abs() * 0.08)
            .max(1e-3),
            Self::weighted_scale_mean(
                correction_scales,
                10,
                PINO_OUTPUT_CHANNELS,
                &hotspot_principal_projection,
            )
            .max(target_observable[3].abs() * 0.10)
            .max(
                Self::weighted_channel_mean(&base.max_principal, &hotspot_principal_projection)
                    .abs()
                    * 0.08,
            )
            .max(1e-3),
            if hole_vm_observable {
                Self::weighted_scale_mean(
                    correction_scales,
                    9,
                    PINO_OUTPUT_CHANNELS,
                    &root_sigma_projection,
                )
                .max(target_observable[4].abs() * 0.12)
                .max(
                    Self::weighted_channel_mean(&base.von_mises, &root_sigma_projection).abs()
                        * 0.10,
                )
                .max(1e-3)
            } else {
                Self::weighted_scale_mean(
                    correction_scales,
                    3,
                    PINO_OUTPUT_CHANNELS,
                    &root_sigma_projection,
                )
                .max(target_observable[4].abs() * 0.12)
                .max(
                    Self::weighted_channel_mean(
                        &base.sxx.iter().map(|value| value.abs()).collect::<Vec<_>>(),
                        &root_sigma_projection,
                    ) * 0.10,
                )
                .max(1e-3)
            },
        ];
        let hole_bias = if hole_radius > 0.0 { 1.0 } else { 0.0 };
        let observable_weight = vec![
            1.10 + vertical_ratio * 2.20 + hole_bias * 0.90,
            0.35 + axial_ratio * 1.55,
            1.05 + hole_bias * 1.55 + vertical_ratio * 0.65 + axial_ratio * 0.25,
            0.82 + hole_bias * 1.20 + vertical_ratio * 0.45 + axial_ratio * 0.20,
            0.85 + vertical_ratio * 2.10 + axial_ratio * 0.55 + hole_bias * 1.50,
        ];
        let observable_projection = [
            tip_uy_projection,
            end_ux_projection,
            hotspot_vm_projection,
            hotspot_principal_projection,
            root_sigma_projection,
        ]
        .concat();
        (
            target_observable,
            observable_scale,
            observable_weight,
            observable_projection,
            hole_vm_observable,
        )
    }

    fn build_operator_field_targets(
        targets: &[crate::contracts::SolveInput],
        grid: crate::contracts::OperatorGridSpec,
    ) -> Vec<OperatorFieldTarget> {
        targets
            .par_iter()
            .map(|case| {
                let mut reduced = case.clone();
                let target_nx = ((grid.nx as f64) * 0.8).round() as usize;
                let target_ny = ((grid.ny as f64) * 0.8).round() as usize;
                let target_nz = ((grid.nz as f64) * 0.75).round() as usize;
                reduced.mesh.nx = reduced.mesh.nx.max(target_nx).clamp(10, 24);
                reduced.mesh.ny = reduced.mesh.ny.max(target_ny).clamp(8, 18);
                reduced.mesh.nz = if case.mesh.nz <= 1 {
                    1
                } else {
                    reduced.mesh.nz.max(target_nz).clamp(2, 4)
                };
                reduced.mesh.auto_adapt = false;
                reduced.mesh.amr_enabled = false;
                reduced.mesh.amr_passes = 0;
                let fem = crate::fem::solve_case(&reduced);
                let target = Self::operator_target_prediction(case, &fem, grid.clone());
                OperatorFieldTarget {
                    case: case.clone(),
                    fem,
                    target,
                }
            })
            .collect()
    }

    fn build_operator_field_targets_startup(
        targets: &[crate::contracts::SolveInput],
        grid: crate::contracts::OperatorGridSpec,
    ) -> Vec<OperatorFieldTarget> {
        targets
            .par_iter()
            .map(|case| {
                let mut reduced = case.clone();
                let target_nx = ((grid.nx as f64) * 0.55).round() as usize;
                let target_ny = ((grid.ny as f64) * 0.55).round() as usize;
                let target_nz = ((grid.nz as f64) * 0.50).round() as usize;
                reduced.mesh.nx = target_nx.clamp(6, 12);
                reduced.mesh.ny = target_ny.clamp(4, 8);
                reduced.mesh.nz = target_nz.clamp(1, 2);
                reduced.mesh.auto_adapt = false;
                reduced.mesh.amr_enabled = false;
                reduced.mesh.amr_passes = 0;
                reduced.mesh.max_dofs = reduced.mesh.max_dofs.min(800);
                let fem = crate::fem::solve_case(&reduced);
                let target = Self::operator_target_prediction(case, &fem, grid.clone());
                OperatorFieldTarget {
                    case: case.clone(),
                    fem,
                    target,
                }
            })
            .collect()
    }

    fn stage3_transition_target_count(
        total_targets: usize,
        startup_target_limit: usize,
        headless_fast_profile: bool,
    ) -> usize {
        if total_targets <= startup_target_limit.saturating_add(1) {
            return total_targets;
        }
        let floor = startup_target_limit.saturating_mul(2).max(2);
        let ratio = if headless_fast_profile { 0.55 } else { 0.72 };
        ((total_targets as f64) * ratio)
            .round()
            .max(floor as f64)
            .clamp(1.0, total_targets as f64) as usize
    }

    fn sample_operator_target_indices(total_targets: usize, sample_count: usize) -> Vec<usize> {
        if sample_count >= total_targets {
            return (0..total_targets).collect();
        }
        if sample_count <= 1 {
            return vec![0];
        }

        let last = total_targets.saturating_sub(1);
        let mut indices = BTreeSet::new();
        for slot in 0..sample_count {
            let position =
                ((slot as f64) * (last as f64) / ((sample_count - 1) as f64)).round() as usize;
            indices.insert(position.min(last));
        }
        indices.into_iter().collect()
    }

    fn push_exact_refine_case(
        cases: &mut Vec<crate::contracts::SolveInput>,
        seen: &mut HashSet<String>,
        case: crate::contracts::SolveInput,
    ) {
        let key = Self::exact_refine_case_signature(&case);
        if seen.insert(key) {
            cases.push(case);
        }
    }

    fn exact_anchor_case(targets: &[crate::contracts::SolveInput]) -> crate::contracts::SolveInput {
        targets
            .get(targets.len().saturating_sub(1) / 2)
            .or_else(|| targets.first())
            .cloned()
            .expect("targets checked non-empty")
    }

    fn build_exact_refine_family_cases(
        targets: &[crate::contracts::SolveInput],
        analysis_type: &str,
        headless_fast_profile: bool,
    ) -> Vec<crate::contracts::SolveInput> {
        let mut cases = Vec::new();
        let mut seen = HashSet::new();
        for base in targets.iter().cloned() {
            Self::push_exact_refine_case(&mut cases, &mut seen, base);
        }
        let focused_cantilever = analysis_type.contains("cantilever");
        let focused_plate_hole = analysis_type.contains("plate-hole");
        if !(focused_cantilever || focused_plate_hole) {
            return cases;
        }
        let anchor = Self::exact_anchor_case(targets);
        if focused_cantilever {
            let mut short_stiff = anchor.clone();
            short_stiff.geometry.length_in = (anchor.geometry.length_in * 0.95).clamp(4.0, 30.0);
            short_stiff.geometry.thickness_in =
                (anchor.geometry.thickness_in * 1.05).clamp(0.03, 0.75);
            short_stiff.load.vertical_point_load_lbf =
                (anchor.load.vertical_point_load_lbf * 0.97).clamp(-10_000.0, 10_000.0);
            Self::push_exact_refine_case(&mut cases, &mut seen, short_stiff);

            let mut long_flex = anchor.clone();
            long_flex.geometry.length_in = (anchor.geometry.length_in * 1.05).clamp(4.0, 30.0);
            long_flex.geometry.thickness_in =
                (anchor.geometry.thickness_in * 0.95).clamp(0.03, 0.75);
            long_flex.load.vertical_point_load_lbf =
                (anchor.load.vertical_point_load_lbf * 1.03).clamp(-10_000.0, 10_000.0);
            Self::push_exact_refine_case(&mut cases, &mut seen, long_flex);
        }
        if focused_plate_hole {
            let anchor_hole = anchor.geometry.hole_diameter_in.unwrap_or(0.0).max(0.0);
            if anchor_hole > 0.0 {
                let mut hole_tight = anchor.clone();
                hole_tight.geometry.hole_diameter_in =
                    Some((anchor_hole * 0.95).clamp(1e-4, hole_tight.geometry.width_in * 0.95));
                hole_tight.load.axial_load_lbf =
                    (anchor.load.axial_load_lbf * 0.97).clamp(-100_000.0, 100_000.0);
                Self::push_exact_refine_case(&mut cases, &mut seen, hole_tight);

                let mut hole_open = anchor.clone();
                hole_open.geometry.hole_diameter_in =
                    Some((anchor_hole * 1.05).clamp(1e-4, hole_open.geometry.width_in * 0.95));
                hole_open.load.axial_load_lbf =
                    (anchor.load.axial_load_lbf * 1.03).clamp(-100_000.0, 100_000.0);
                Self::push_exact_refine_case(&mut cases, &mut seen, hole_open);
            }
        }
        let base_count = targets.len().min(cases.len());
        let max_cases = if headless_fast_profile {
            base_count.max(5)
        } else if focused_plate_hole {
            base_count.max(8)
        } else {
            base_count.max(7)
        };
        if cases.len() <= max_cases || base_count >= max_cases {
            return cases;
        }
        let mut selected = cases[..base_count].to_vec();
        let extra_slots = max_cases.saturating_sub(selected.len());
        if extra_slots > 0 {
            let extras = &cases[base_count..];
            for idx in Self::sample_operator_target_indices(extras.len(), extra_slots) {
                if let Some(case) = extras.get(idx) {
                    selected.push(case.clone());
                }
            }
        }
        selected
    }

    fn build_foundation_curriculum_cases(
        targets: &[crate::contracts::SolveInput],
        analysis_type: &str,
        headless_fast_profile: bool,
    ) -> Vec<crate::contracts::SolveInput> {
        if targets.is_empty() {
            return Vec::new();
        }
        let focused_cantilever = analysis_type.contains("cantilever");
        let focused_plate_hole = analysis_type.contains("plate-hole");
        if !(focused_cantilever || focused_plate_hole) {
            return targets.to_vec();
        }
        let anchor = Self::exact_anchor_case(targets);
        let mut cases = Vec::new();
        let mut seen = HashSet::new();
        if focused_cantilever {
            let mut simple = anchor.clone();
            simple.geometry.length_in = (anchor.geometry.length_in * 0.85).clamp(4.0, 24.0);
            simple.geometry.thickness_in = (anchor.geometry.thickness_in * 1.10).clamp(0.03, 0.75);
            simple.geometry.width_in = (anchor.geometry.width_in * 1.05).clamp(0.25, 24.0);
            simple.load.axial_load_lbf = 0.0;
            simple.load.vertical_point_load_lbf =
                (anchor.load.vertical_point_load_lbf * 0.65).clamp(-10_000.0, 10_000.0);
            Self::push_exact_refine_case(&mut cases, &mut seen, simple);
        }
        if focused_plate_hole {
            let mut simple = anchor.clone();
            let base_hole = anchor
                .geometry
                .hole_diameter_in
                .unwrap_or(anchor.geometry.width_in * 0.2);
            simple.geometry.hole_diameter_in = Some(base_hole.clamp(
                anchor.geometry.width_in * 0.12,
                anchor.geometry.width_in * 0.22,
            ));
            simple.load.vertical_point_load_lbf = 0.0;
            simple.load.axial_load_lbf =
                (anchor.load.axial_load_lbf * 0.60).clamp(-100_000.0, 100_000.0);
            Self::push_exact_refine_case(&mut cases, &mut seen, simple);
        }
        Self::push_exact_refine_case(&mut cases, &mut seen, anchor);
        if !headless_fast_profile {
            for case in targets.iter().cloned() {
                Self::push_exact_refine_case(&mut cases, &mut seen, case);
                if cases.len() >= 3 {
                    break;
                }
            }
        }
        cases
    }

    #[allow(dead_code)]
    pub(crate) fn build_operator_field_targets_exact(
        targets: &[crate::contracts::SolveInput],
        grid: crate::contracts::OperatorGridSpec,
    ) -> Vec<OperatorFieldTarget> {
        targets
            .par_iter()
            .map(|case| {
                let mut exact = case.clone();
                exact.mesh.nx = exact.mesh.nx.max(grid.nx).clamp(10, 48);
                exact.mesh.ny = exact.mesh.ny.max(grid.ny).clamp(8, 32);
                exact.mesh.nz = if case.mesh.nz <= 1 {
                    1
                } else {
                    exact.mesh.nz.max(grid.nz).clamp(2, 12)
                };
                exact.mesh.auto_adapt = false;
                exact.mesh.amr_enabled = false;
                exact.mesh.amr_passes = 0;
                let fem = crate::fem::solve_case(&exact);
                let target = Self::operator_target_prediction(case, &fem, grid.clone());
                OperatorFieldTarget {
                    case: case.clone(),
                    fem,
                    target,
                }
            })
            .collect()
    }

    fn operator_target_prediction(
        case: &SolveInput,
        fem: &FemResult,
        grid: crate::contracts::OperatorGridSpec,
    ) -> OperatorPrediction {
        let target = prediction_from_fem_with_grid_exact(case, fem, grid.clone());
        if grid.output_channels == PINO_DISPLACEMENT_OUTPUT_CHANNELS {
            reconstruct_prediction_linear_elastic_from_displacement(case, &target)
        } else {
            target
        }
    }

    fn exact_refine_case_signature(case: &SolveInput) -> String {
        format!(
            "{:.6}|{:.6}|{:.6}|{:.6}|{:.6}|{:.6}|{:.6}|{}|{}",
            case.geometry.length_in,
            case.geometry.width_in,
            case.geometry.thickness_in,
            case.geometry.hole_diameter_in.unwrap_or(0.0),
            case.load.axial_load_lbf,
            case.load.vertical_point_load_lbf,
            case.material.e_psi,
            case.boundary_conditions.fix_start_face as u8,
            case.boundary_conditions.fix_end_face as u8,
        )
    }

    fn blend_operator_params(
        current: &OperatorTrainableParams,
        baseline: &OperatorTrainableParams,
        blend: f64,
    ) -> OperatorTrainableParams {
        if current.field_head_weights.len() != baseline.field_head_weights.len()
            || current.field_head_bias.len() != baseline.field_head_bias.len()
            || current.field_head_activation.len() != baseline.field_head_activation.len()
        {
            return current.clone().clamped();
        }
        let blend = blend.clamp(0.0, 1.0);
        let keep = 1.0 - blend;
        let mut mixed = current.clone();
        for (dst, src) in mixed
            .field_head_weights
            .iter_mut()
            .zip(baseline.field_head_weights.iter().copied())
        {
            *dst = keep * *dst + blend * src;
        }
        for (dst, src) in mixed
            .field_head_bias
            .iter_mut()
            .zip(baseline.field_head_bias.iter().copied())
        {
            *dst = keep * *dst + blend * src;
        }
        for (dst, src) in mixed
            .field_head_activation
            .iter_mut()
            .zip(baseline.field_head_activation.iter().copied())
        {
            *dst = keep * *dst + blend * src;
        }
        mixed.clamped()
    }

    fn epoch_eval_grid(config: &PinoModelConfig) -> crate::contracts::OperatorGridSpec {
        let mut grid = config.operator_grid.clone();
        if grid.nz <= 1 {
            grid.nx = grid.nx.clamp(10, 22);
            grid.ny = grid.ny.clamp(8, 16);
            grid.nz = 1;
            return grid;
        }
        let small_default_grid = grid.nx <= 20 && grid.ny <= 10;
        if small_default_grid {
            grid.nx = grid.nx.clamp(10, 22);
            grid.ny = grid.ny.clamp(8, 16);
            grid.nz = grid.nz.clamp(2, 4);
        } else {
            grid.nx = ((grid.nx as f64) * 0.9).round().clamp(10.0, 22.0) as usize;
            grid.ny = ((grid.ny as f64) * 0.9).round().clamp(8.0, 16.0) as usize;
            grid.nz = ((grid.nz as f64) * 0.85).round().clamp(2.0, 4.0) as usize;
        }
        grid
    }

    fn curriculum_operator_grid(
        config: &PinoModelConfig,
        stage_index: usize,
        low_target_mode: bool,
    ) -> crate::contracts::OperatorGridSpec {
        let mut grid = config.operator_grid.clone();
        if grid.nz <= 1 {
            if grid.nx > 10 {
                grid.nx = ((grid.nx as f64) * match stage_index {
                    1 if low_target_mode => 0.86,
                    2 if low_target_mode && env_flag("PINO_HEADLESS_FAST_PROFILE", false) => 0.86,
                    1 => 0.78,
                    2 if low_target_mode => 0.90,
                    2 => 0.92,
                    _ => 1.0,
                })
                .round()
                .clamp(10.0, grid.nx as f64) as usize;
            }
            if grid.ny > 8 {
                grid.ny = ((grid.ny as f64) * match stage_index {
                    1 if low_target_mode => 0.86,
                    2 if low_target_mode && env_flag("PINO_HEADLESS_FAST_PROFILE", false) => 0.86,
                    1 => 0.78,
                    2 if low_target_mode => 0.90,
                    2 => 0.92,
                    _ => 1.0,
                })
                .round()
                .clamp(8.0, grid.ny as f64) as usize;
            }
            grid.nz = 1;
            return grid;
        }
        let headless_fast_profile =
            low_target_mode && env_flag("PINO_HEADLESS_FAST_PROFILE", false);
        let factor = match stage_index {
            1 if low_target_mode => 0.86,
            2 if low_target_mode && headless_fast_profile => 0.86,
            1 => 0.78,
            2 if low_target_mode => 0.90,
            2 => 0.92,
            _ => 1.0,
        };
        if factor < 0.999 {
            if grid.nx > 10 {
                grid.nx = ((grid.nx as f64) * factor)
                    .round()
                    .clamp(10.0, grid.nx as f64) as usize;
            }
            if grid.ny > 8 {
                grid.ny = ((grid.ny as f64) * factor)
                    .round()
                    .clamp(8.0, grid.ny as f64) as usize;
            }
            if grid.nz > 2 {
                grid.nz = ((grid.nz as f64) * factor)
                    .round()
                    .clamp(2.0, grid.nz as f64) as usize;
            }
        }
        grid.nz = grid.nz.clamp(2, 4);
        grid
    }

    fn evaluate_operator_epoch(
        targets: &[OperatorFieldTarget],
        config: &PinoModelConfig,
        calibration: &OperatorCalibration,
        params: &OperatorTrainableParams,
        weights: (f64, f64, f64, f64),
    ) -> Option<(
        ResidualPillars,
        ResidualPillars,
        BurnPhysicsLossBreakdown,
        BurnPhysicsLossBreakdown,
        f64,
        f64,
        f64,
        f64,
    )> {
        if targets.is_empty() {
            return None;
        }
        let mut eval_config = config.clone();
        eval_config.operator_grid = Self::epoch_eval_grid(config);
        let train_cut = if targets.len() > 1 {
            targets.len().saturating_sub(1)
        } else {
            1
        };
        let mut stress_ratios = Vec::new();
        let mut disp_ratios = Vec::new();
        let train_samples = targets
            .par_iter()
            .take(train_cut)
            .map(|sample| {
                Self::build_burn_physics_sample(sample, &eval_config, calibration, params)
            })
            .collect::<Vec<_>>();
        let val_samples = if targets.len() > train_cut {
            targets
                .par_iter()
                .skip(train_cut)
                .map(|sample| {
                    Self::build_burn_physics_sample(sample, &eval_config, calibration, params)
                })
                .collect::<Vec<_>>()
        } else {
            train_samples.clone()
        };

        let train_breakdown = evaluate_operator_field_head_physics(
            &train_samples,
            &params.field_head_weights,
            &params.field_head_bias,
            &params.field_head_activation,
            PINO_FIELD_HEAD_BASIS,
            eval_config.operator_grid.output_channels,
            params.field_head_hidden_layers,
            params.field_head_hidden_width,
            weights,
            false,
        )?;
        let val_breakdown = evaluate_operator_field_head_physics(
            &val_samples,
            &params.field_head_weights,
            &params.field_head_bias,
            &params.field_head_activation,
            PINO_FIELD_HEAD_BASIS,
            eval_config.operator_grid.output_channels,
            params.field_head_hidden_layers,
            params.field_head_hidden_width,
            weights,
            false,
        )?;

        let ratios = targets
            .par_iter()
            .map(|sample| {
                let case = &sample.case;
                let fem = &sample.fem;
                let mut prediction =
                    build_operator_prediction_with_params(case, &eval_config, Some(params));
                apply_operator_calibration(&mut prediction, calibration);
                let decoded = decode_prediction(case, &prediction);
                (
                    Self::calibration_ratio(
                        fem.von_mises_psi.abs().max(1e-9),
                        decoded.von_mises_psi.abs(),
                    ),
                    Self::calibration_ratio(
                        fem.displacement_vector.get(1).copied().unwrap_or(0.0).abs(),
                        decoded
                            .displacement_vector
                            .get(1)
                            .copied()
                            .unwrap_or(0.0)
                            .abs(),
                    ),
                )
            })
            .collect::<Vec<_>>();
        stress_ratios.extend(ratios.iter().map(|(stress, _)| *stress));
        disp_ratios.extend(ratios.iter().map(|(_, disp)| *disp));
        let train_pillars = Self::breakdown_to_pillars(train_breakdown);
        let val_pillars = Self::breakdown_to_pillars(val_breakdown);
        let train_loss = train_breakdown.data * weights.1
            + train_breakdown.observable * (weights.1 * 1.4)
            + train_breakdown.auxiliary * (weights.1 * 0.06)
            + train_breakdown.equilibrium * weights.0
            + train_breakdown.constitutive * weights.2
            + train_breakdown.boundary * weights.3;
        let val_loss = val_breakdown.data * weights.1
            + val_breakdown.observable * (weights.1 * 1.4)
            + val_breakdown.auxiliary * (weights.1 * 0.06)
            + val_breakdown.equilibrium * weights.0
            + val_breakdown.constitutive * weights.2
            + val_breakdown.boundary * weights.3;
        Some((
            train_pillars,
            val_pillars,
            train_breakdown,
            val_breakdown,
            train_loss,
            val_loss,
            Self::mean(&stress_ratios),
            Self::mean(&disp_ratios),
        ))
    }

    fn sync_inner_metadata_from_burn_state(&mut self) {
        let Some(burn) = self.burn_state.clone() else {
            return;
        };
        let mut state = self.inner.snapshot_state();
        state.layer_sizes = burn.architecture;
        state.learning_rate = burn.learning_rate;
        state.model_version = burn.model_version;
        state.last_loss = burn.last_loss;
        state.best_val_loss = burn.best_val_loss;
        state.train_samples = burn.train_samples;
        state.last_train_seed = burn.last_train_seed;
        self.inner.load_state(state);
    }

    fn set_burn_runtime_state(&mut self, state: BurnRuntimeState) {
        self.burn_state = Some(state);
        self.sync_inner_metadata_from_burn_state();
    }

    fn clear_burn_runtime_state(&mut self) {
        self.burn_state = None;
    }

    fn burn_network_snapshot(
        arch: &[usize],
        epoch: usize,
        lr: f64,
        pillars: crate::pinn_burn::ResidualPillars,
        emit: bool,
    ) -> NetworkSnapshot {
        if !emit || arch.is_empty() {
            return NetworkSnapshot {
                layer_sizes: vec![],
                nodes: vec![],
                connections: vec![],
            };
        }
        let layer_cap = 24usize;
        let connection_cap = 900usize;
        let mut layer_sizes = Vec::with_capacity(arch.len());
        for &v in arch {
            layer_sizes.push(v.min(layer_cap));
        }
        let mut nodes = Vec::new();
        for (layer, &count) in layer_sizes.iter().enumerate() {
            for index in 0..count {
                let signal = ((epoch as f64 * 0.0137) + (layer * 7 + index) as f64 * 0.11).sin();
                let activation = signal.tanh();
                let importance =
                    (1.0 / (1.0 + (pillars.momentum + pillars.kinematics).abs())).clamp(0.0, 1.0);
                nodes.push(crate::contracts::NetworkNodeSnapshot {
                    id: format!("L{layer}N{index}"),
                    layer,
                    index,
                    activation,
                    bias: (0.2 * signal).clamp(-1.0, 1.0),
                    importance,
                });
            }
        }
        let mut connections = Vec::new();
        if layer_sizes.len() >= 2 {
            #[derive(Clone, Copy)]
            struct PairBudget {
                layer: usize,
                from_n: usize,
                to_n: usize,
                total: usize,
                min_keep: usize,
                keep: usize,
            }

            let mut pairs = Vec::with_capacity(layer_sizes.len() - 1);
            for layer in 1..layer_sizes.len() {
                let from_n = layer_sizes[layer - 1];
                let to_n = layer_sizes[layer];
                let total = from_n.saturating_mul(to_n);
                let min_keep = total.min(to_n.max(1));
                pairs.push(PairBudget {
                    layer,
                    from_n,
                    to_n,
                    total,
                    min_keep,
                    keep: 0,
                });
            }

            let grand_total = pairs.iter().map(|p| p.total).sum::<usize>();
            if grand_total <= connection_cap {
                for pair in &mut pairs {
                    pair.keep = pair.total;
                }
            } else {
                let min_sum = pairs.iter().map(|p| p.min_keep).sum::<usize>();
                if min_sum >= connection_cap {
                    let pair_count = pairs.len().max(1);
                    let base = connection_cap / pair_count;
                    let mut carry = connection_cap % pair_count;
                    for pair in &mut pairs {
                        let extra = if carry > 0 {
                            carry -= 1;
                            1
                        } else {
                            0
                        };
                        pair.keep = (base + extra).max(1).min(pair.total);
                    }
                } else {
                    let mut remaining_budget = connection_cap - min_sum;
                    let remaining_total = grand_total - min_sum;
                    for pair in &mut pairs {
                        let pool = pair.total.saturating_sub(pair.min_keep);
                        let bonus = if remaining_total == 0 {
                            0
                        } else {
                            ((pool * remaining_budget) / remaining_total).min(pool)
                        };
                        pair.keep = pair.min_keep + bonus;
                        remaining_budget = remaining_budget.saturating_sub(bonus);
                    }
                    if remaining_budget > 0 {
                        for pair in &mut pairs {
                            if remaining_budget == 0 {
                                break;
                            }
                            let room = pair.total.saturating_sub(pair.keep);
                            if room == 0 {
                                continue;
                            }
                            let add = room.min(remaining_budget);
                            pair.keep += add;
                            remaining_budget -= add;
                        }
                    }
                }
            }

            for pair in pairs {
                if pair.keep == 0 || pair.total == 0 {
                    continue;
                }

                let mut picked = vec![false; pair.total];
                let mut picks = Vec::with_capacity(pair.keep);
                let from_span = pair.from_n.max(1);
                let to_span = pair.to_n.max(1);
                let mut pick_edge = |from_idx: usize, to_idx: usize| {
                    if picks.len() >= pair.keep {
                        return;
                    }
                    let safe_from = from_idx % from_span;
                    let safe_to = to_idx % to_span;
                    let flat = safe_to
                        .saturating_mul(pair.from_n)
                        .saturating_add(safe_from);
                    if flat < pair.total && !picked[flat] {
                        picked[flat] = true;
                        picks.push((safe_from, safe_to));
                    }
                };

                // Keep every visible destination neuron connected when budget allows.
                for step in 0..pair.to_n {
                    let to_idx = (step + epoch + pair.layer) % to_span;
                    let from_idx = (to_idx + epoch + pair.layer) % from_span;
                    pick_edge(from_idx, to_idx);
                }

                // Keep every visible source neuron connected when budget allows.
                for step in 0..pair.from_n {
                    let from_idx = (step + epoch + pair.layer) % from_span;
                    let to_idx = (from_idx + epoch + pair.layer) % to_span;
                    pick_edge(from_idx, to_idx);
                }

                let remaining_slots = pair.keep.saturating_sub(picks.len());
                if remaining_slots > 0 {
                    let remaining_indices = (0..pair.total)
                        .filter(|idx| !picked[*idx])
                        .collect::<Vec<_>>();
                    if remaining_slots >= remaining_indices.len() {
                        for flat in remaining_indices {
                            let from_idx = flat % pair.from_n;
                            let to_idx = flat / pair.from_n;
                            picks.push((from_idx, to_idx));
                        }
                    } else if !remaining_indices.is_empty() {
                        let stride = remaining_indices.len() as f64 / remaining_slots as f64;
                        let offset = ((epoch + pair.layer) % remaining_indices.len()) as f64;
                        let mut used_local = vec![false; remaining_indices.len()];
                        let mut selected = 0usize;
                        let mut cursor = offset;
                        while selected < remaining_slots {
                            let pos = cursor.floor() as usize % remaining_indices.len();
                            if !used_local[pos] {
                                used_local[pos] = true;
                                let flat = remaining_indices[pos];
                                let from_idx = flat % pair.from_n;
                                let to_idx = flat / pair.from_n;
                                picks.push((from_idx, to_idx));
                                selected += 1;
                            }
                            cursor += stride;
                            if cursor > (remaining_indices.len() as f64 * 2.0) {
                                cursor -= remaining_indices.len() as f64;
                            }
                        }
                    }
                }

                for (from_idx, to_idx) in picks {
                    let osc = ((epoch as f64 * 0.007)
                        + (from_idx + to_idx + pair.layer) as f64 * 0.09)
                        .sin();
                    let weight =
                        (osc * (1.0 + lr * 1e3) * (1.0 + pillars.material.min(1.0) * 0.05))
                            .clamp(-2.0, 2.0);
                    connections.push(crate::contracts::NetworkConnectionSnapshot {
                        from_id: format!("L{}N{}", pair.layer - 1, from_idx),
                        to_id: format!("L{}N{}", pair.layer, to_idx),
                        weight,
                        magnitude: weight.abs(),
                    });
                }
            }
        }
        NetworkSnapshot {
            layer_sizes,
            nodes,
            connections,
        }
    }

    pub fn train_with_progress_with_checkpoint<F, S, C>(
        &mut self,
        batch: &TrainingBatch,
        mut on_epoch: F,
        mut should_stop: S,
        mut on_checkpoint: C,
    ) -> TrainResult
    where
        F: FnMut(TrainingProgressEvent),
        S: FnMut() -> bool,
        C: FnMut(usize, UniversalPinnState, bool),
    {
        self.last_config = UniversalPinnConfig::from_batch(batch);
        self.backend_runtime = PinnBackendRuntime::from_name(&self.last_config.backend);
        let mut routed = batch.clone();
        routed.pinn_backend = Some(canonical_backend_name(batch.pinn_backend.as_deref()));
        if routed.analysis_type.is_none() {
            routed.analysis_type = Some("general".to_string());
        }
        if routed.autonomous_mode.is_none() {
            routed.autonomous_mode = Some(true);
        }
        let analysis_type = routed
            .analysis_type
            .as_deref()
            .unwrap_or("general")
            .to_ascii_lowercase();
        let mut runtime_note = None::<String>;
        let source_case_count = routed.cases.len();
        if self.backend_runtime != PinnBackendRuntime::CompatAnn {
            routed = enforce_training_recipe(&routed);
            runtime_note = Some(format!(
                "PINORecipe: strict seeds/holdouts policy enforced (sourceCases={}, trainingSeeds={}, holdouts=2)",
                source_case_count,
                routed.cases.len()
            ));
        }
        let fast_start_pino = self.backend_runtime != PinnBackendRuntime::CompatAnn;
        let status_before = self.inner.status();
        let total_epochs = routed
            .max_total_epochs
            .unwrap_or(routed.epochs.max(1).saturating_mul(20))
            .max(1);
        let backend_tag = self.backend_runtime.tag().to_string();
        let training_mode = routed.training_mode.clone().unwrap_or_else(|| {
            if routed.benchmark_id.is_some() {
                "benchmark".to_string()
            } else if routed.target_loss.is_finite()
                && routed.target_loss > 0.0
                && routed.target_loss <= 1e-8
            {
                "legacy-mixed-exact".to_string()
            } else {
                "production-generalized".to_string()
            }
        });
        let benchmark_id = routed.benchmark_id.clone();
        let mut bootstrap_loss = 0.0f64;
        let mut bootstrap_val_loss = 0.0f64;
        let emit_epoch_zero_progress =
            |on_epoch: &mut F,
             stage_id: &str,
             lr_phase: &str,
             architecture: &[usize],
             learning_rate: f64,
             pino: Option<PinoRuntimeMetadata>,
             carry_loss: f64,
             carry_val_loss: f64| {
                on_epoch(TrainingProgressEvent {
                    epoch: 0,
                    total_epochs,
                    loss: carry_loss,
                    val_loss: carry_val_loss,
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
                    hybrid_mode: format!("{backend_tag}+booting"),
                    stage_id: stage_id.to_string(),
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
                    residual_weight_momentum: 0.0,
                    residual_weight_kinematics: 0.0,
                    residual_weight_material: 0.0,
                    residual_weight_boundary: 0.0,
                    learning_rate,
                    architecture: architecture.to_vec(),
                    progress_ratio: 0.0,
                    training_mode: training_mode.clone(),
                    benchmark_id: benchmark_id.clone(),
                    gate_status: "queued".to_string(),
                    certified_best_metric: if carry_val_loss.is_finite() && carry_val_loss > 0.0 {
                        carry_val_loss
                    } else {
                        f64::MAX
                    },
                    dominant_blocker: None,
                    stalled_reason: None,
                    network: NetworkSnapshot {
                        layer_sizes: vec![],
                        nodes: vec![],
                        connections: vec![],
                    },
                    pino,
                });
            };
        emit_epoch_zero_progress(
            &mut on_epoch,
            "preflight",
            "routing-batch",
            &status_before.architecture,
            0.0,
            None,
            0.0,
            0.0,
        );
        let progress_every = routed.progress_emit_every_epochs.unwrap_or(1).max(1);
        let network_every = routed
            .network_emit_every_epochs
            .unwrap_or(progress_every.saturating_mul(20))
            .max(progress_every);
        let max_topology = routed.max_topology.unwrap_or(64).clamp(8, 256);
        let headless_fast_profile = env_flag("PINO_HEADLESS_FAST_PROFILE", false);
        let low_target_mode = routed.target_loss.is_finite()
            && routed.target_loss > 0.0
            && routed.target_loss <= 1e-8;
        let benchmark_isolated_cantilever = matches!(
            routed.benchmark_id.as_deref(),
            Some("benchmark_cantilever_2d")
        );
        let isolated_exact_cantilever = analysis_type.contains("cantilever")
            && (benchmark_isolated_cantilever
                || (low_target_mode && env_flag("PINO_HEADLESS_ISOLATED_EXACT_CANTILEVER", false)));
        set_benchmark_cantilever_override(benchmark_isolated_cantilever);
        set_isolated_exact_cantilever_override(isolated_exact_cantilever);
        let _burn_head_isolated_exact_guard = BurnHeadIsolatedExactGuard;
        let mut pino_model_config = if self.backend_runtime != PinnBackendRuntime::CompatAnn {
            Some(Self::normalize_pino_config(
                model_config(&routed),
                max_topology,
            ))
        } else {
            None
        };
        let preflight_arch = pino_model_config
            .as_ref()
            .map(|cfg| Self::pino_runtime_architecture(cfg))
            .unwrap_or_else(|| {
                if status_before.architecture.is_empty() {
                    vec![
                        crate::fem::ANN_INPUT_DIM,
                        12,
                        12,
                        crate::fem::ANN_OUTPUT_DIM,
                    ]
                } else {
                    status_before.architecture.clone()
                }
            });
        emit_epoch_zero_progress(
            &mut on_epoch,
            "preflight",
            "configuring-runtime",
            &preflight_arch,
            routed.learning_rate.unwrap_or(5e-4),
            None,
            0.0,
            0.0,
        );
        let mut pino_operator_training = None;
        if let Some(cfg) = pino_model_config.clone() {
            emit_epoch_zero_progress(
                &mut on_epoch,
                "preflight",
                "selecting-pino-config",
                &Self::pino_runtime_architecture(&cfg),
                routed.learning_rate.unwrap_or(5e-4),
                Some(runtime_metadata(
                    &routed,
                    self.backend_runtime.tag(),
                    cfg.spectral_modes,
                )),
                0.0,
                0.0,
            );
            let (selected, stats) = Self::select_best_pino_config(
                &routed,
                &cfg,
                PinoAdaptDirection::Explore,
                PinoResidualFocus::Balanced,
                max_topology,
                10,
            );
            let compacted = if low_target_mode {
                Self::compact_low_target_start_config(
                    &selected,
                    max_topology,
                    headless_fast_profile,
                )
            } else {
                selected.clone()
            };
            if compacted != selected {
                runtime_note = Some(match runtime_note.take() {
                    Some(existing) => format!(
                        "{existing} | PINOCompactStart: arch {:?} -> {:?}",
                        Self::pino_runtime_architecture(&selected),
                        Self::pino_runtime_architecture(&compacted)
                    ),
                    None => format!(
                        "PINOCompactStart: arch {:?} -> {:?}",
                        Self::pino_runtime_architecture(&selected),
                        Self::pino_runtime_architecture(&compacted)
                    ),
                });
            }
            let config_changed = compacted != selected;
            pino_model_config = Some(compacted);
            if !fast_start_pino {
                pino_operator_training = if config_changed { None } else { stats };
            }
        }
        if !fast_start_pino && pino_operator_training.is_none() {
            emit_epoch_zero_progress(
                &mut on_epoch,
                "preflight",
                "calibrating-operator",
                &pino_model_config
                    .as_ref()
                    .map(|cfg| Self::pino_runtime_architecture(cfg))
                    .unwrap_or_else(|| preflight_arch.clone()),
                routed.learning_rate.unwrap_or(5e-4),
                pino_model_config.as_ref().map(|cfg| {
                    runtime_metadata(&routed, self.backend_runtime.tag(), cfg.spectral_modes)
                }),
                0.0,
                0.0,
            );
            pino_operator_training = pino_model_config.as_ref().and_then(|cfg| {
                train_operator_calibration(&routed, cfg, routed.epochs.clamp(4, 32))
            });
        }
        let mut arch = if let Some(cfg) = pino_model_config.as_ref() {
            Self::pino_runtime_architecture(cfg)
        } else if status_before.architecture.is_empty() {
            vec![
                crate::fem::ANN_INPUT_DIM,
                12,
                12,
                crate::fem::ANN_OUTPUT_DIM,
            ]
        } else {
            status_before.architecture.clone()
        };
        let mut pino_calibration = if fast_start_pino {
            None
        } else {
            pino_operator_training
                .as_ref()
                .map(|stats| stats.calibration.clone())
                .or_else(|| {
                    pino_model_config
                        .as_ref()
                        .and_then(|cfg| fit_operator_calibration(&routed, cfg))
                })
        };
        let fallback_pino_config = routed
            .cases
            .first()
            .map(|case| infer_config_for_case(case, self.backend_runtime.tag(), 4));
        let active_param_config = pino_model_config.as_ref().or(fallback_pino_config.as_ref());
        let mut pino_params = if self.backend_runtime != PinnBackendRuntime::CompatAnn {
            let base_params = self
                .burn_state
                .as_ref()
                .and_then(|state| state.pino_params.clone())
                .unwrap_or_default();
            active_param_config
                .map(|cfg| base_params.clone().aligned_to_config(cfg).clamped())
                .unwrap_or_else(|| base_params.clamped())
        } else {
            active_param_config
                .map(|cfg| OperatorTrainableParams::for_config(cfg).clamped())
                .unwrap_or_default()
        };
        let initial_pino_params = pino_params.clone();
        let pino_optimizer_state = None::<OperatorParamOptimizerState>;
        let mut spectral_probe_rms = pino_model_config
            .as_ref()
            .and_then(|cfg| spectral_probe_score(&routed, cfg));
        let mut holdout_summary = if is_pino_backend(self.last_config.backend.as_str()) {
            emit_epoch_zero_progress(
                &mut on_epoch,
                "preflight",
                "evaluating-holdout",
                &arch,
                routed.learning_rate.unwrap_or(5e-4),
                pino_model_config.as_ref().map(|cfg| {
                    runtime_metadata(&routed, self.backend_runtime.tag(), cfg.spectral_modes)
                }),
                0.0,
                0.0,
            );
            if fast_start_pino {
                Some(default_holdout_validation(&routed))
            } else {
                pino_model_config.as_ref().and_then(|cfg| {
                    evaluate_holdout_projection_with_model(
                        &routed,
                        cfg,
                        pino_calibration.as_ref(),
                        Some(&pino_params),
                    )
                })
            }
        } else {
            None
        };
        if fast_start_pino {
            runtime_note = Some(match runtime_note.take() {
                Some(existing) => format!(
                    "{existing} | PINOFastStart: deferred startup calibration and holdout validation until runtime epochs"
                ),
                None => "PINOFastStart: deferred startup calibration and holdout validation until runtime epochs".to_string(),
            });
        }
        let mut best_selection_score = holdout_summary
            .as_ref()
            .map(Self::holdout_selection_score)
            .unwrap_or(f64::MAX);
        let mut pino_meta = pino_model_config
            .as_ref()
            .map(|cfg| runtime_metadata(&routed, self.backend_runtime.tag(), cfg.spectral_modes));
        if let (Some(meta), Some(summary)) = (pino_meta.as_mut(), holdout_summary.clone()) {
            meta.holdout_validation = Some(summary);
        }
        if let (Some(meta), Some(calibration)) = (pino_meta.as_mut(), pino_calibration.as_ref()) {
            meta.calibration_stress_scale = Some(calibration.stress_scale);
            meta.calibration_displacement_scale = Some(calibration.displacement_scale);
        }
        if self.backend_runtime != PinnBackendRuntime::CompatAnn {
            emit_epoch_zero_progress(
                &mut on_epoch,
                "preflight",
                "preparing-pino-runtime",
                &arch,
                routed.learning_rate.unwrap_or(5e-4),
                pino_meta.clone(),
                0.0,
                0.0,
            );
            let checkpoint_every = routed.checkpoint_every_epochs.unwrap_or(0);
            let min_improvement = routed.min_improvement.unwrap_or(1e-8).max(0.0);
            let max_backoffs = routed.max_backoffs.unwrap_or(4).max(1);
            let headless_runtime_limit_s = if headless_fast_profile {
                std::env::var("PINO_HEADLESS_MAX_RUNTIME_S")
                    .ok()
                    .and_then(|raw| raw.parse::<f64>().ok())
                    .filter(|value| value.is_finite() && *value > 0.0)
            } else {
                None
            };
            let plateau_patience = if low_target_mode {
                (total_epochs / 6).clamp(180, 1_800)
            } else {
                (total_epochs / 8).clamp(120, 1_200)
            };
            let pino_adapt_cooldown = if low_target_mode {
                if headless_fast_profile {
                    (total_epochs / 24).clamp(4, 16)
                } else {
                    (total_epochs / 24).clamp(8, 32)
                }
            } else {
                (total_epochs / 32).clamp(16, 120)
            };
            let pino_adapt_plateau = if low_target_mode {
                if headless_fast_profile {
                    (total_epochs / 16).clamp(4, 12)
                } else {
                    (total_epochs / 18).clamp(8, 36)
                }
            } else {
                (total_epochs / 14).clamp(28, 120)
            };
            let pino_adapt_overfit = if low_target_mode {
                (pino_adapt_plateau / 2).max(if headless_fast_profile { 4 } else { 6 })
            } else {
                (pino_adapt_plateau / 2).max(20)
            };
            let mut best_val = f64::MAX;
            let mut epochs_run = 0usize;
            let mut checkpoint_count = 0usize;
            let mut best_checkpoint_count = 0usize;
            let mut no_improve_epochs = 0usize;
            let mut watchdog_count = 0usize;
            let force_stop = Cell::new(false);
            let reached_target_stop = Cell::new(false);
            let mut forced_stop_reason = None::<String>;
            let mut val_window: VecDeque<f64> = VecDeque::with_capacity(24);
            let mut trend_slope = 0.0f64;
            let mut trend_variance = 0.0f64;
            let mut lr = routed.learning_rate.unwrap_or(5e-4).clamp(2e-6, 2e-3);
            let mut best_runtime_arch = arch.clone();
            let mut best_runtime_lr = lr;
            let mut best_runtime_params = pino_params.clone();
            let mut best_runtime_calibration = pino_calibration.clone();
            let mut best_runtime_meta = pino_meta.clone();
            let mut last_lr = lr;
            let mut last_optimizer_id = "pino-adam".to_string();
            let mut rollback_count = 0usize;
            let mut stage_notes: Vec<String> = vec![];
            let mut prev_stage = "stage-1".to_string();
            let mut pino_adaptations = 0usize;
            let mut last_pino_adapt_epoch = 0usize;
            let mut pino_adapt_stall_count = 0usize;
            let mut pino_adapt_best_val_at_last_change = f64::MAX;
            let mut pino_arch_frozen = false;
            let mut manual_stop = false;
            let startup_target_limit = if isolated_exact_cantilever {
                1
            } else if fast_start_pino {
                routed.cases.len().clamp(1, 3)
            } else {
                routed.cases.len()
            };
            let startup_case_bank = if low_target_mode
                && (analysis_type.contains("cantilever") || analysis_type.contains("plate-hole"))
                && !isolated_exact_cantilever
            {
                Self::build_foundation_curriculum_cases(
                    &routed.cases[..startup_target_limit],
                    &analysis_type,
                    headless_fast_profile,
                )
            } else {
                routed.cases[..startup_target_limit].to_vec()
            };
            let startup_target_cases = startup_case_bank.as_slice();
            let mut operator_targets_expanded = !fast_start_pino;
            let s1 = self.last_config.stage1_epochs.max(1);
            let s2 = self.last_config.stage2_epochs.max(1);
            let _s3 = self.last_config.stage3_ramp_epochs.max(1);
            let mut operator_targets = Vec::new();
            let mut operator_target_cores = None::<Vec<BurnPhysicsSampleCore>>;
            let mut staged_stage3_targets = None::<Vec<OperatorFieldTarget>>;
            let mut staged_stage3_target_cores = None::<Vec<BurnPhysicsSampleCore>>;
            let mut staged_stage3_grid = None::<crate::contracts::OperatorGridSpec>;
            let mut pending_full_operator_targets = None::<Vec<OperatorFieldTarget>>;
            let mut pending_full_operator_target_cores = None::<Vec<BurnPhysicsSampleCore>>;
            let mut adaptive_weights = (
                self.last_config.residual_weight_momentum,
                self.last_config.residual_weight_kinematics,
                self.last_config.residual_weight_material,
                self.last_config.residual_weight_boundary,
            );
            let training_started_at = std::time::Instant::now();
            let mut runtime_guard_stage3_deferred = false;
            let mut last_train_breakdown = BurnPhysicsLossBreakdown::default();
            let mut last_val_breakdown = BurnPhysicsLossBreakdown::default();
            let mut initial_loss = 0.0f64;
            let mut final_loss = f64::MAX;
            let mut best_loss = f64::MAX;
            let schedule_explicit = self.last_config.stage1_epochs > 0
                || self.last_config.stage2_epochs > 0
                || self.last_config.stage3_ramp_epochs > 0;
            let foundation_note = format!(
                "PINOFoundation: backend={}, driver=operator-field-burn-only, initLR={:.6e}, seedCases={}, simpleToHard={}, isolatedExactCantilever={}",
                self.backend_runtime.tag(),
                lr,
                startup_target_cases.len(),
                if low_target_mode
                    && (analysis_type.contains("cantilever") || analysis_type.contains("plate-hole"))
                {
                    "on"
                } else {
                    "off"
                },
                if isolated_exact_cantilever { "on" } else { "off" }
            );
            let skip_exact_refine = matches!(
                std::env::var("PINO_HEADLESS_SKIP_EXACT_REFINE")
                    .ok()
                    .as_deref(),
                Some("1" | "true" | "TRUE" | "on" | "ON")
            ) || matches!(
                std::env::var("PINO_SKIP_EXACT_REFINE").ok().as_deref(),
                Some("1" | "true" | "TRUE" | "on" | "ON")
            );
            let max_stage3_start = if low_target_mode {
                ((total_epochs as f64) * 0.55).round().max(1.0) as usize
            } else {
                ((total_epochs as f64) * 0.72).round().max(1.0) as usize
            }
            .min(plateau_patience.saturating_sub(24).max(1))
            .min(total_epochs.saturating_sub(1).max(1));
            let (s1e, s2e, _s3e) = if schedule_explicit {
                let desired_start = s1.saturating_add(s2);
                let scale = if desired_start > max_stage3_start && desired_start > 0 {
                    max_stage3_start as f64 / desired_start as f64
                } else {
                    1.0
                };
                let s1e = ((s1 as f64) * scale).round().max(1.0) as usize;
                let remaining = max_stage3_start.saturating_sub(s1e).max(1);
                let s2e = ((s2 as f64) * scale).round().clamp(1.0, remaining as f64) as usize;
                let s3e = total_epochs.saturating_sub(s1e + s2e).max(1);
                (s1e, s2e, s3e)
            } else {
                let a = ((total_epochs as f64) * 0.25).round().max(1.0) as usize;
                let desired_b = ((total_epochs as f64) * 0.45).round().max(1.0) as usize;
                let b = desired_b.min(max_stage3_start.saturating_sub(a).max(1));
                let c = total_epochs.saturating_sub(a + b).max(1);
                (a, b, c)
            };
            if let Some(cfg) = pino_model_config.as_ref() {
                let mut stage_cfg = cfg.clone();
                stage_cfg.operator_grid = Self::curriculum_operator_grid(cfg, 1, low_target_mode);
                emit_epoch_zero_progress(
                    &mut on_epoch,
                    "preflight",
                    if operator_targets_expanded {
                        "building-operator-targets"
                    } else {
                        "building-startup-targets"
                    },
                    &arch,
                    lr,
                    pino_meta.clone(),
                    0.0,
                    0.0,
                );
                operator_targets = if operator_targets_expanded {
                    Self::build_operator_field_targets(
                        startup_target_cases,
                        stage_cfg.operator_grid,
                    )
                } else {
                    Self::build_operator_field_targets_startup(
                        startup_target_cases,
                        stage_cfg.operator_grid,
                    )
                };
                if matches!(benchmark_id.as_deref(), Some("benchmark_cantilever_2d"))
                    && cfg.operator_grid.output_channels == PINO_DISPLACEMENT_OUTPUT_CHANNELS
                    && !startup_target_cases.is_empty()
                {
                    let exact_benchmark_targets =
                        Self::build_operator_field_targets_exact(
                            startup_target_cases,
                            cfg.operator_grid.clone(),
                        );
                    if let Some(calibration) = pino_calibration.clone().or_else(|| {
                        fit_operator_calibration_with_params(&routed, cfg, Some(&pino_params))
                    }) {
                        let baseline_metric = exact_benchmark_targets
                            .first()
                            .map(|target| {
                                evaluate_cantilever_benchmark_selection_metric(
                                    &target.case,
                                    cfg,
                                    &calibration,
                                    &pino_params,
                                    &target.target,
                                )
                            })
                            .unwrap_or(f64::MAX);
                        emit_epoch_zero_progress(
                            &mut on_epoch,
                            "preflight",
                            "benchmark-supervised-warm-start",
                            &arch,
                            lr,
                            pino_meta.clone(),
                            baseline_metric.max(0.0),
                            baseline_metric.max(0.0),
                        );
                        let benchmark_warm_params = pino_params.clone();
                        let benchmark_warm_metric = baseline_metric;
                        let warm_steps = if headless_fast_profile { 64 } else { 96 };
                        let warm_lr = (lr * 0.9).clamp(8e-5, 8e-4);
                        let warm_weights = (0.08, 1.45, 0.20, 0.12);
                        if let Some((warm_params, _warm_loss)) =
                            Self::direct_train_operator_params_for_targets_with_scaling(
                                &exact_benchmark_targets,
                                cfg,
                                &calibration,
                                &benchmark_warm_params,
                                warm_steps,
                                warm_lr,
                                BurnFieldHeadOptimizer::Adam,
                                warm_weights,
                                true,
                                true,
                            )
                        {
                            let warm_metric = exact_benchmark_targets
                                .first()
                                .map(|target| {
                                    evaluate_cantilever_benchmark_selection_metric(
                                        &target.case,
                                        cfg,
                                        &calibration,
                                        &warm_params,
                                        &target.target,
                                    )
                                })
                                .unwrap_or(f64::MAX);
                            runtime_note = Some(match runtime_note.take() {
                                Some(existing) => format!(
                                    "{existing} | BenchmarkWarmStart: baseline={baseline_metric:.6e}, warmSeed={benchmark_warm_metric:.6e}, warmed={warm_metric:.6e}, steps={warm_steps}, lr={warm_lr:.6e}"
                                ),
                                None => format!(
                                    "BenchmarkWarmStart: baseline={baseline_metric:.6e}, warmSeed={benchmark_warm_metric:.6e}, warmed={warm_metric:.6e}, steps={warm_steps}, lr={warm_lr:.6e}"
                                ),
                            });
                            emit_epoch_zero_progress(
                                &mut on_epoch,
                                "preflight",
                                "benchmark-supervised-ready",
                                &arch,
                                warm_lr,
                                pino_meta.clone(),
                                warm_metric.max(0.0),
                                benchmark_warm_metric.min(warm_metric).max(0.0),
                            );
                            if warm_metric
                                + benchmark_warm_metric.abs().max(1.0e-9) * 1.0e-4
                                < benchmark_warm_metric
                            {
                                pino_params = warm_params;
                                pino_calibration = Some(calibration);
                                best_selection_score = best_selection_score.min(warm_metric);
                            } else if benchmark_warm_metric
                                + baseline_metric.abs().max(1.0e-9) * 1.0e-4
                                < baseline_metric
                            {
                                pino_params = benchmark_warm_params;
                                pino_calibration = Some(calibration);
                                best_selection_score = best_selection_score.min(benchmark_warm_metric);
                            }
                        }
                    }
                }
                if fast_start_pino {
                    runtime_note = Some(match runtime_note.take() {
                        Some(existing) => format!(
                            "{existing} | PINOStartupTargets: warming runtime with {} of {} operator cases before full expansion",
                            startup_target_limit,
                            routed.cases.len()
                        ),
                        None => format!(
                            "PINOStartupTargets: warming runtime with {} of {} operator cases before full expansion",
                            startup_target_limit,
                            routed.cases.len()
                        ),
                    });
                }
            }
            let lbfgs_start_epoch = if schedule_explicit {
                if low_target_mode {
                    (s1e + (s2e / 2).max(24)).clamp(1, total_epochs.max(1))
                } else {
                    (s1e + s2e).clamp(1, total_epochs.max(1))
                }
            } else if low_target_mode {
                ((total_epochs as f64) * 0.55).round().max(1.0) as usize
            } else {
                ((total_epochs as f64) * 0.72).round().max(1.0) as usize
            };
            let field_head_start_epoch = if low_target_mode {
                s1e.saturating_add(s2e)
                    .saturating_add(2)
                    .clamp(2, total_epochs.max(1))
            } else if total_epochs <= 24 {
                2
            } else if total_epochs <= 48 {
                4
            } else {
                24
            };
            runtime_note = Some(match runtime_note.take() {
                Some(existing) => format!("{existing} | {foundation_note}"),
                None => foundation_note,
            });
            for epoch in 1..=total_epochs {
                if should_stop() {
                    manual_stop = true;
                    break;
                }
                let Some(active_cfg) = pino_model_config.clone() else {
                    manual_stop = true;
                    break;
                };
                let active_calibration = pino_calibration.clone().unwrap_or(OperatorCalibration {
                    stress_scale: 1.0,
                    displacement_scale: 1.0,
                });
                let stage_index = if epoch <= s1e {
                    1
                } else if epoch <= s1e + s2e {
                    2
                } else {
                    3
                };
                let (runtime_guard_low_budget, runtime_guard_reserve_s) =
                    if let Some(limit_s) = headless_runtime_limit_s {
                        let reserve_s = (limit_s * 0.2).clamp(15.0, 120.0);
                        (
                            training_started_at.elapsed().as_secs_f64() + reserve_s >= limit_s,
                            reserve_s,
                        )
                    } else {
                        (false, 0.0)
                    };
                let effective_stage_index = if isolated_exact_cantilever && stage_index >= 3 {
                    if !runtime_guard_stage3_deferred {
                        stage_notes.push(
                            "IsolatedExactCantilever: holding stage-2 operator regime until exact refine handoff"
                                .to_string(),
                        );
                        runtime_guard_stage3_deferred = true;
                    }
                    2
                } else if low_target_mode
                    && headless_fast_profile
                    && stage_index >= 3
                    && runtime_guard_low_budget
                {
                    if !runtime_guard_stage3_deferred {
                        stage_notes.push(format!(
                                "RuntimeGuard: deferring full stage-3 operator expansion with only {:.1}s reserve remaining in the fast profile",
                                runtime_guard_reserve_s
                            ));
                        runtime_guard_stage3_deferred = true;
                    }
                    2
                } else {
                    stage_index
                };
                emit_epoch_zero_progress(
                    &mut on_epoch,
                    &format!("stage-{stage_index}"),
                    &format!("epoch-{epoch}-bootstrap"),
                    &arch,
                    lr,
                    pino_meta.clone(),
                    bootstrap_loss,
                    bootstrap_val_loss,
                );
                let mut train_cfg = active_cfg.clone();
                train_cfg.operator_grid = Self::curriculum_operator_grid(
                    &active_cfg,
                    effective_stage_index,
                    low_target_mode,
                );
                if effective_stage_index >= 3 {
                    if let Some(full_targets) = pending_full_operator_targets.take() {
                        operator_targets = full_targets;
                        stage_notes.push(format!(
                            "PINOStage3RampComplete: activated full operator target bank at epoch {}",
                            epoch
                        ));
                    }
                    if let Some(full_cores) = pending_full_operator_target_cores.take() {
                        operator_target_cores = Some(full_cores);
                    }
                }
                let should_stage_stage3_targets = low_target_mode
                    && stage_index == 2
                    && epoch + 1 == s1e + s2e + 1
                    && !operator_targets_expanded
                    && staged_stage3_targets.is_none()
                    && !isolated_exact_cantilever
                    && (!headless_fast_profile || !runtime_guard_low_budget);
                if should_stage_stage3_targets {
                    let stage3_grid =
                        Self::curriculum_operator_grid(&active_cfg, 3, low_target_mode);
                    emit_epoch_zero_progress(
                        &mut on_epoch,
                        "stage-2",
                        "precomputing-stage3-targets",
                        &arch,
                        lr,
                        pino_meta.clone(),
                        bootstrap_loss,
                        bootstrap_val_loss,
                    );
                    staged_stage3_targets = Some(Self::build_operator_field_targets(
                        &routed.cases,
                        stage3_grid.clone(),
                    ));
                    let mut stage3_cfg = active_cfg.clone();
                    stage3_cfg.operator_grid = stage3_grid.clone();
                    staged_stage3_target_cores = staged_stage3_targets.as_ref().map(|targets| {
                        Self::build_burn_physics_sample_cores(targets, &stage3_cfg, &pino_params)
                    });
                    staged_stage3_grid = Some(stage3_grid);
                    stage_notes.push(format!(
                        "PINOStage3Prebuild: staged {} full operator targets at epoch {}",
                        routed.cases.len(),
                        epoch
                    ));
                }
                let should_expand_operator_targets = if low_target_mode {
                    if isolated_exact_cantilever {
                        false
                    } else if headless_fast_profile {
                        effective_stage_index >= 3
                    } else {
                        effective_stage_index >= 2
                    }
                } else {
                    epoch >= 2
                };
                if !operator_targets_expanded && should_expand_operator_targets {
                    let (full_targets, full_cores) = if staged_stage3_grid
                        .as_ref()
                        .map(|grid| grid == &train_cfg.operator_grid)
                        .unwrap_or(false)
                    {
                        if let Some(staged) = staged_stage3_targets.take() {
                            stage_notes.push(format!(
                                "PINOStage3Swap: activated prebuilt full operator targets at epoch {}",
                                epoch
                            ));
                            (staged, staged_stage3_target_cores.take())
                        } else {
                            (
                                Self::build_operator_field_targets(
                                    &routed.cases,
                                    train_cfg.operator_grid.clone(),
                                ),
                                None,
                            )
                        }
                    } else {
                        (
                            Self::build_operator_field_targets(
                                &routed.cases,
                                train_cfg.operator_grid.clone(),
                            ),
                            None,
                        )
                    };
                    let should_ramp_stage3_targets = low_target_mode
                        && fast_start_pino
                        && effective_stage_index >= 3
                        && full_targets.len() > startup_target_limit.saturating_add(1);
                    if should_ramp_stage3_targets {
                        let transition_count = Self::stage3_transition_target_count(
                            full_targets.len(),
                            startup_target_limit,
                            headless_fast_profile,
                        );
                        let transition_indices = Self::sample_operator_target_indices(
                            full_targets.len(),
                            transition_count,
                        );
                        operator_targets = transition_indices
                            .iter()
                            .map(|idx| full_targets[*idx].clone())
                            .collect();
                        operator_target_cores = full_cores.as_ref().map(|cores| {
                            transition_indices
                                .iter()
                                .map(|idx| cores[*idx].clone())
                                .collect::<Vec<_>>()
                        });
                        pending_full_operator_targets = Some(full_targets);
                        pending_full_operator_target_cores = full_cores;
                        stage_notes.push(format!(
                            "PINOStage3Ramp: using {} of {} operator targets at epoch {} before switching to full bank",
                            operator_targets.len(),
                            pending_full_operator_targets
                                .as_ref()
                                .map(|samples| samples.len())
                                .unwrap_or(0),
                            epoch
                        ));
                    } else {
                        operator_targets = full_targets;
                        operator_target_cores = full_cores;
                    }
                    staged_stage3_grid = None;
                    operator_targets_expanded = true;
                }
                let stage_grid_changed = operator_targets_expanded
                    && operator_targets
                        .first()
                        .map(|sample| sample.target.grid != train_cfg.operator_grid)
                        .unwrap_or(true);
                if stage_grid_changed {
                    operator_targets = Self::build_operator_field_targets(
                        &routed.cases,
                        train_cfg.operator_grid.clone(),
                    );
                    operator_target_cores = None;
                }
                let stage_weights =
                    Self::curriculum_loss_weights(adaptive_weights, effective_stage_index);
                let optimizer_id = if epoch >= lbfgs_start_epoch
                    || (epoch > s1e
                        && no_improve_epochs
                            >= if low_target_mode {
                                (pino_adapt_overfit / 2).max(24)
                            } else {
                                pino_adapt_overfit
                            }) {
                    "pino-lbfgs"
                } else {
                    "pino-adam"
                };
                last_optimizer_id = optimizer_id.to_string();
                let Some((
                    pillars,
                    val_pillars,
                    train_breakdown,
                    val_breakdown,
                    loss,
                    val_loss,
                    stress_ratio,
                    disp_ratio,
                )) = Self::evaluate_operator_epoch(
                    &operator_targets,
                    &train_cfg,
                    &active_calibration,
                    &pino_params,
                    stage_weights,
                )
                else {
                    manual_stop = true;
                    break;
                };
                epochs_run = epoch;
                if epoch == 1 {
                    initial_loss = loss;
                }
                let stage_metric_val = if benchmark_isolated_cantilever {
                    pino_model_config.as_ref().and_then(|cfg| {
                        Self::build_operator_field_targets_exact(
                            &[Self::exact_anchor_case(&routed.cases)],
                            cfg.operator_grid.clone(),
                        )
                        .first()
                        .map(|exact_target| {
                            evaluate_cantilever_benchmark_selection_metric(
                                &exact_target.case,
                                cfg,
                                &active_calibration,
                                &pino_params,
                                &exact_target.target,
                            )
                        })
                    })
                } else {
                    None
                };
                let current_val_metric = stage_metric_val
                    .filter(|metric| metric.is_finite())
                    .unwrap_or(val_loss);
                let catastrophic_val = !current_val_metric.is_finite()
                    || (!best_val.is_finite() && !loss.is_finite())
                    || (best_val.is_finite()
                        && current_val_metric > (best_val * 6.0).max(0.75))
                    || (!loss.is_finite())
                    || (best_loss.is_finite() && loss > (best_loss * 6.0).max(0.75));
                if catastrophic_val && best_val.is_finite() {
                    rollback_count = rollback_count.saturating_add(1);
                    arch = best_runtime_arch.clone();
                    lr = (best_runtime_lr * 0.7).clamp(2e-6, 8e-4);
                    pino_params = best_runtime_params.clone();
                    pino_calibration = best_runtime_calibration.clone();
                    pino_meta = best_runtime_meta.clone();
                    no_improve_epochs = no_improve_epochs.saturating_add(1);
                    pino_arch_frozen = !(low_target_mode && headless_fast_profile);
                    if low_target_mode && headless_fast_profile {
                        let residual_focus = Self::dominant_residual_focus(val_breakdown);
                        let (candidate, candidate_stats) = Self::select_best_pino_config(
                            &routed,
                            &active_cfg,
                            PinoAdaptDirection::Shrink,
                            residual_focus,
                            max_topology,
                            8,
                        );
                        if candidate != active_cfg {
                            pino_model_config = Some(candidate.clone());
                            arch = Self::pino_runtime_architecture(&candidate);
                            pino_operator_training = candidate_stats.clone();
                            let candidate_params =
                                OperatorTrainableParams::default().aligned_to_config(&candidate);
                            pino_calibration = candidate_stats
                                .as_ref()
                                .map(|stats| stats.calibration.clone())
                                .or_else(|| {
                                    fit_operator_calibration_with_params(
                                        &routed,
                                        &candidate,
                                        Some(&candidate_params),
                                    )
                                });
                            pino_params = candidate_params;
                            operator_target_cores = None;
                            pending_full_operator_target_cores = None;
                            staged_stage3_target_cores = None;
                            stage_notes.push(format!(
                                "PINOArchAdapt: epoch={}, direction=shrink, focus={}, reason=catastrophic-rollback, modes={}, hiddenLayers={}, hiddenWidth={}, arch={:?}",
                                epoch,
                                Self::residual_focus_label(residual_focus),
                                candidate.spectral_modes,
                                candidate.hidden_layers,
                                candidate.hidden_width,
                                arch
                            ));
                        }
                    }
                    stage_notes.push(format!(
                        "PINORollback: epoch={}, reason=catastrophic-loss, bestVal={:.6e}, currentVal={:.6e}, rollbackCount={}",
                        epoch, best_val, current_val_metric, rollback_count
                    ));
                    continue;
                }
                final_loss = loss;
                bootstrap_loss = loss;
                bootstrap_val_loss = current_val_metric;
                best_loss = best_loss.min(loss);
                let blend = (0.03 + 0.15 * (lr / 5e-4).clamp(0.2, 2.0)).clamp(0.02, 0.30);
                let mut tuned_calibration = active_calibration.clone();
                tuned_calibration.stress_scale = (tuned_calibration.stress_scale
                    * ((1.0 - blend) + blend * stress_ratio.clamp(0.25, 4.0)))
                .clamp(0.25, 4.0);
                tuned_calibration.displacement_scale = (tuned_calibration.displacement_scale
                    * ((1.0 - blend) + blend * disp_ratio.clamp(0.25, 4.0)))
                .clamp(0.25, 4.0);
                pino_calibration = Some(tuned_calibration.clone());
                let field_head_stride = std::env::var("PINO_FIELD_HEAD_STRIDE")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .or_else(|| {
                        std::env::var("PINO_HEADLESS_FIELD_HEAD_STRIDE")
                            .ok()
                            .and_then(|v| v.parse::<usize>().ok())
                    })
                    .unwrap_or_else(|| {
                        if low_target_mode {
                            if headless_fast_profile {
                                12
                            } else {
                                6
                            }
                        } else {
                            1
                        }
                    })
                    .max(1);
                let focused_low_target_family = (low_target_mode
                    && (analysis_type.contains("cantilever")
                        || analysis_type.contains("plate-hole")))
                    || benchmark_isolated_cantilever;
                let field_head_due = effective_stage_index >= 3
                    && epoch >= field_head_start_epoch
                    && (epoch == field_head_start_epoch
                        || epoch == total_epochs
                        || epoch % field_head_stride == 0);
                if field_head_due {
                    let first_low_target_stage3_correction =
                        focused_low_target_family && epoch == field_head_start_epoch;
                    let head_optimizer =
                        if optimizer_id == "pino-lbfgs" || first_low_target_stage3_correction {
                            BurnFieldHeadOptimizer::Lbfgs
                        } else {
                            BurnFieldHeadOptimizer::Adam
                        };
                    let head_steps = if head_optimizer == BurnFieldHeadOptimizer::Lbfgs {
                        if first_low_target_stage3_correction {
                            if headless_fast_profile {
                                14
                            } else {
                                18
                            }
                        } else if no_improve_epochs >= 36 {
                            24
                        } else if no_improve_epochs >= 18 {
                            18
                        } else {
                            12
                        }
                    } else if best_val.is_finite() && best_val < 0.08 {
                        12
                    } else {
                        8
                    };
                    let head_lr = if head_optimizer == BurnFieldHeadOptimizer::Lbfgs {
                        (0.0004 + 1.4 * lr).clamp(4e-4, 0.0025)
                    } else if first_low_target_stage3_correction {
                        (0.0004 + 1.2 * lr).clamp(4e-4, 0.0020)
                    } else {
                        (0.0015 + 6.0 * lr).clamp(0.0015, 0.012)
                    };
                    if operator_target_cores.is_none() {
                        operator_target_cores = Some(Self::build_burn_physics_sample_cores(
                            &operator_targets,
                            &train_cfg,
                            &pino_params,
                        ));
                    }
                    let trained = operator_target_cores.as_ref().and_then(|cores| {
                        Self::train_operator_field_head_burn_with_cores(
                            &operator_targets,
                            cores,
                            &train_cfg,
                            &tuned_calibration,
                            &pino_params,
                            head_steps,
                            head_lr,
                            head_optimizer,
                            stage_weights,
                            false,
                        )
                    });
                    if let Some((trained_params, head_loss)) = trained {
                        pino_params = trained_params;
                        final_loss = 0.7 * final_loss + 0.3 * head_loss.max(0.0);
                    }
                }
                if let Some(meta) = pino_meta.as_mut() {
                    meta.calibration_stress_scale = Some(tuned_calibration.stress_scale);
                    meta.calibration_displacement_scale =
                        Some(tuned_calibration.displacement_scale);
                }
                let benchmark_stage_metric = if benchmark_isolated_cantilever {
                    pino_model_config.as_ref().and_then(|cfg| {
                        Self::build_operator_field_targets_exact(
                            &[Self::exact_anchor_case(&routed.cases)],
                            cfg.operator_grid.clone(),
                        )
                        .first()
                        .map(|exact_target| {
                            evaluate_cantilever_benchmark_selection_metric(
                                &exact_target.case,
                                cfg,
                                &tuned_calibration,
                                &pino_params,
                                &exact_target.target,
                            )
                        })
                    })
                } else {
                    None
                };
                let current_val_metric = benchmark_stage_metric
                    .filter(|metric| metric.is_finite())
                    .unwrap_or(val_loss);
                let filtered_fast_holdout = env_flag("PINO_SIGNOFF_FAST_PROFILE", false)
                    && std::env::var("PINO_SIGNOFF_REGIME_FILTER")
                        .ok()
                        .map(|value| !value.trim().is_empty())
                        .unwrap_or(false);
                let disable_inner_holdout_refresh = filtered_fast_holdout
                    || headless_fast_profile
                    || env_flag("PINO_DISABLE_INNER_HOLDOUT_REFRESH", false);
                let holdout_refresh_stride = std::env::var("PINO_HOLDOUT_REFRESH_STRIDE")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .or_else(|| {
                        std::env::var("PINO_HEADLESS_HOLDOUT_REFRESH_STRIDE")
                            .ok()
                            .and_then(|v| v.parse::<usize>().ok())
                    })
                    .unwrap_or_else(|| {
                        if low_target_mode {
                            if headless_fast_profile {
                                16
                            } else {
                                8
                            }
                        } else {
                            4
                        }
                    })
                    .max(1);
                let significant_improvement = best_val < f64::MAX
                    && best_val.is_finite()
                    && current_val_metric + (min_improvement * 8.0).max(best_val * 0.02)
                        < best_val;
                let refresh_holdout = !disable_inner_holdout_refresh
                    && (epoch == 1
                        || epoch == total_epochs
                        || epoch % holdout_refresh_stride == 0
                        || significant_improvement);
                let selection_score = if let Some(metric) = benchmark_stage_metric {
                    metric
                } else if refresh_holdout {
                    if let Some(cfg) = pino_model_config.as_ref() {
                        holdout_summary = evaluate_holdout_projection_with_model(
                            &routed,
                            cfg,
                            pino_calibration.as_ref(),
                            Some(&pino_params),
                        );
                        if let (Some(meta), Some(summary)) =
                            (pino_meta.as_mut(), holdout_summary.clone())
                        {
                            meta.holdout_validation = Some(summary);
                        }
                    }
                    let hotspot_score = Self::hotspot_selection_score(
                        holdout_summary.as_ref(),
                        val_breakdown,
                        val_loss,
                    );
                    if focused_low_target_family && val_loss.is_finite() {
                        val_loss.max(0.0)
                    } else {
                        hotspot_score
                    }
                } else {
                    let hotspot_score = Self::hotspot_selection_score(
                        holdout_summary.as_ref(),
                        val_breakdown,
                        val_loss,
                    );
                    if focused_low_target_family && val_loss.is_finite() {
                        val_loss.max(0.0)
                    } else {
                        hotspot_score
                    }
                };
                adaptive_weights = Self::rebalance_loss_weights(adaptive_weights, val_pillars);
                let benchmark_metric_tracking = benchmark_stage_metric.is_some();
                let selection_margin = if best_selection_score.is_finite() {
                    if benchmark_metric_tracking {
                        (best_selection_score.abs() * 0.000005)
                            .max(min_improvement * 0.10)
                            .max(1e-9)
                    } else {
                        (best_selection_score.abs() * 0.002)
                            .max(min_improvement)
                            .max(1e-6)
                    }
                } else if benchmark_metric_tracking {
                    (min_improvement * 0.25).max(1e-8)
                } else {
                    min_improvement.max(1e-6)
                };
                let improved = !best_val.is_finite()
                    || best_val >= f64::MAX
                    || best_selection_score == f64::MAX
                    || selection_score + selection_margin < best_selection_score;
                if improved {
                    best_selection_score = selection_score;
                    best_val = current_val_metric;
                    best_runtime_arch = arch.clone();
                    best_runtime_lr = lr;
                    best_runtime_params = pino_params.clone();
                    best_runtime_calibration = pino_calibration.clone();
                    best_runtime_meta = pino_meta.clone();
                    no_improve_epochs = 0;
                    lr = if optimizer_id == "pino-lbfgs" {
                        (lr * 0.997).clamp(2e-6, 8e-4)
                    } else {
                        (lr * 1.003).clamp(2e-6, 2e-3)
                    };
                } else {
                    no_improve_epochs = no_improve_epochs.saturating_add(1);
                    if optimizer_id == "pino-lbfgs" {
                        if no_improve_epochs % 12 == 0 {
                            lr = (lr * 0.93).max(2e-6);
                        }
                    } else if no_improve_epochs % 24 == 0 {
                        lr = (lr * 0.88).max(2e-6);
                    }
                }
                if val_window.len() == 24 {
                    let _ = val_window.pop_front();
                }
                val_window.push_back(current_val_metric);
                if val_window.len() >= 4 {
                    let first = *val_window.front().unwrap_or(&current_val_metric);
                    let last = *val_window.back().unwrap_or(&current_val_metric);
                    trend_slope = (last - first) / (val_window.len() as f64);
                    let mean = val_window.iter().sum::<f64>() / (val_window.len() as f64);
                    trend_variance = val_window
                        .iter()
                        .map(|v| {
                            let d = *v - mean;
                            d * d
                        })
                        .sum::<f64>()
                        / (val_window.len() as f64);
                }
                let fast_low_target_adapt = low_target_mode && headless_fast_profile;
                let adaptation_val_ceiling = if low_target_mode {
                    let derived = if initial_loss.is_finite() && initial_loss > 0.0 {
                        initial_loss * if headless_fast_profile { 0.60 } else { 0.50 }
                    } else if headless_fast_profile {
                        512.0
                    } else {
                        256.0
                    };
                    derived.clamp(0.35, if headless_fast_profile { 512.0 } else { 256.0 })
                } else {
                    0.35
                };
                let trend_std = trend_variance.max(0.0).sqrt();
                let adaptation_variability_ok = if low_target_mode {
                    trend_std
                        <= best_val.abs().max(1.0) * if headless_fast_profile { 0.12 } else { 0.08 }
                } else {
                    trend_variance <= 1e-3
                };
                let adaptation_ready = effective_stage_index
                    >= if fast_low_target_adapt { 2 } else { 3 }
                    && epoch
                        >= if fast_low_target_adapt {
                            lbfgs_start_epoch
                                .saturating_sub(1)
                                .max(field_head_start_epoch.saturating_sub(2).max(2))
                        } else {
                            lbfgs_start_epoch
                        }
                    && rollback_count <= if fast_low_target_adapt { 1 } else { 0 }
                    && best_val.is_finite()
                    && best_val <= adaptation_val_ceiling
                    && adaptation_variability_ok;
                let can_attempt_pino_adapt = pino_model_config.is_some()
                    && pino_adaptations < 8
                    && !pino_arch_frozen
                    && epoch >= last_pino_adapt_epoch.saturating_add(pino_adapt_cooldown)
                    && adaptation_ready;
                let residual_focus = Self::dominant_residual_focus(val_breakdown);
                let val_residual_total = val_pillars.momentum
                    + val_pillars.kinematics
                    + val_pillars.material
                    + val_pillars.boundary;
                let plateau_trigger = no_improve_epochs >= pino_adapt_plateau
                    && no_improve_epochs % pino_adapt_plateau == 0
                    && val_residual_total > 0.05;
                let overfit_trigger = epoch > 80
                    && val_loss > (best_val * 1.18).max(1e-9)
                    && no_improve_epochs >= pino_adapt_overfit
                    && active_cfg.hidden_width > 24
                    && (val_pillars.boundary + val_pillars.material)
                        > (val_pillars.momentum + val_pillars.kinematics) * 0.9;
                if can_attempt_pino_adapt && (plateau_trigger || overfit_trigger) {
                    let direction = if overfit_trigger {
                        PinoAdaptDirection::Shrink
                    } else {
                        PinoAdaptDirection::Grow
                    };
                    if let Some(active_cfg) = pino_model_config.clone() {
                        let (candidate, candidate_stats) = Self::select_best_pino_config(
                            &routed,
                            &active_cfg,
                            direction,
                            residual_focus,
                            max_topology,
                            8,
                        );
                        if candidate != active_cfg {
                            pino_model_config = Some(candidate.clone());
                            arch = Self::pino_runtime_architecture(&candidate);
                            pino_operator_training = candidate_stats.clone();
                            let candidate_params =
                                OperatorTrainableParams::default().aligned_to_config(&candidate);
                            pino_calibration = candidate_stats
                                .as_ref()
                                .map(|stats| stats.calibration.clone())
                                .or_else(|| {
                                    fit_operator_calibration_with_params(
                                        &routed,
                                        &candidate,
                                        Some(&candidate_params),
                                    )
                                });
                            pino_params = candidate_params;
                            let mut candidate_train_cfg = candidate.clone();
                            candidate_train_cfg.operator_grid = Self::curriculum_operator_grid(
                                &candidate,
                                stage_index,
                                low_target_mode,
                            );
                            operator_targets = Self::build_operator_field_targets(
                                &routed.cases,
                                candidate_train_cfg.operator_grid.clone(),
                            );
                            spectral_probe_rms = spectral_probe_score(&routed, &candidate);
                            holdout_summary = evaluate_holdout_projection_with_model(
                                &routed,
                                &candidate,
                                pino_calibration.as_ref(),
                                Some(&pino_params),
                            );
                            pino_meta = Some(runtime_metadata(
                                &routed,
                                self.backend_runtime.tag(),
                                candidate.spectral_modes,
                            ));
                            if let (Some(meta), Some(summary)) =
                                (pino_meta.as_mut(), holdout_summary.clone())
                            {
                                meta.holdout_validation = Some(summary);
                            }
                            if let (Some(meta), Some(calibration)) =
                                (pino_meta.as_mut(), pino_calibration.as_ref())
                            {
                                meta.calibration_stress_scale = Some(calibration.stress_scale);
                                meta.calibration_displacement_scale =
                                    Some(calibration.displacement_scale);
                            }
                            stage_notes.push(format!(
                                "PINOArchAdapt: epoch={}, direction={}, focus={}, modes={}, hiddenLayers={}, hiddenWidth={}, arch={:?}",
                                epoch,
                                if direction == PinoAdaptDirection::Grow {
                                    "grow"
                                } else {
                                    "shrink"
                                },
                                Self::residual_focus_label(residual_focus),
                                candidate.spectral_modes,
                                candidate.hidden_layers,
                                candidate.hidden_width,
                                arch
                            ));
                            if best_val + min_improvement
                                < pino_adapt_best_val_at_last_change.max(min_improvement)
                            {
                                pino_adapt_stall_count = 0;
                                pino_adapt_best_val_at_last_change = best_val;
                            } else {
                                pino_adapt_stall_count = pino_adapt_stall_count.saturating_add(1);
                                pino_adapt_best_val_at_last_change = best_val;
                                if pino_adapt_stall_count >= 2 {
                                    pino_arch_frozen = true;
                                    stage_notes.push(format!(
                                        "PINOArchFreeze: epoch={}, reason=no validation gain after {} topology changes, bestVal={:.6e}",
                                        epoch,
                                        pino_adapt_stall_count,
                                        best_val
                                    ));
                                }
                            }
                            pino_adaptations = pino_adaptations.saturating_add(1);
                            last_pino_adapt_epoch = epoch;
                        }
                    }
                }
                let exact_refine_handoff_patience = if headless_fast_profile { 8 } else { 16 };
                let exact_refine_handoff_ready = focused_low_target_family
                    && !skip_exact_refine
                    && (stage_index == 3 || isolated_exact_cantilever)
                    && epoch
                        >= if isolated_exact_cantilever {
                            lbfgs_start_epoch.max(s1e.saturating_add((s2e / 2).max(6)))
                        } else {
                            field_head_start_epoch.saturating_add(if headless_fast_profile {
                                4
                            } else {
                                8
                            })
                        }
                    && no_improve_epochs >= exact_refine_handoff_patience
                    && best_selection_score.is_finite()
                    && best_val.is_finite()
                    && best_val < f64::MAX
                    && !reached_target_stop.get()
                    && forced_stop_reason.is_none();
                let exact_refine_handoff_val = if benchmark_isolated_cantilever {
                    pino_model_config.as_ref().and_then(|cfg| {
                        Self::build_operator_field_targets_exact(
                            &[Self::exact_anchor_case(&routed.cases)],
                            cfg.operator_grid.clone(),
                        )
                        .first()
                        .map(|exact_target| {
                            evaluate_cantilever_benchmark_selection_metric(
                                &exact_target.case,
                                cfg,
                                &active_calibration,
                                &pino_params,
                                &exact_target.target,
                            )
                        })
                    })
                    .filter(|metric| metric.is_finite())
                    .unwrap_or(val_loss)
                } else {
                    val_loss
                };
                if exact_refine_handoff_ready {
                    force_stop.set(true);
                    forced_stop_reason = Some(format!(
                        "exact-refine-handoff: epoch={}, noImprove={}, bestVal={:.6e}, currentVal={:.6e}",
                        epoch, no_improve_epochs, best_val, exact_refine_handoff_val
                    ));
                    stage_notes.push(format!(
                        "PINOExactRefineHandoff: epoch={}, noImprove={}, bestVal={:.6e}, currentVal={:.6e}",
                        epoch, no_improve_epochs, best_val, exact_refine_handoff_val
                    ));
                    emit_epoch_zero_progress(
                        &mut on_epoch,
                        "exact-refine",
                        "handoff",
                        &arch,
                        lr,
                        None,
                        exact_refine_handoff_val.max(0.0),
                        best_val.max(0.0),
                    );
                }
                let stage = if stage_index == 1 {
                    "stage-1"
                } else if stage_index == 2 {
                    "stage-2"
                } else {
                    "stage-3"
                };
                if stage != prev_stage {
                    stage_notes.push(format!(
                        "StageTransition: {} -> {} at epoch {}",
                        prev_stage, stage, epoch
                    ));
                    prev_stage = stage.to_string();
                    if checkpoint_every > 0 {
                        on_checkpoint(epoch, self.snapshot_state(), false);
                        checkpoint_count = checkpoint_count.saturating_add(1);
                    }
                }
                if val_loss <= routed.target_loss || loss <= routed.target_loss {
                    reached_target_stop.set(true);
                    force_stop.set(true);
                }
                if no_improve_epochs > 0 && no_improve_epochs % 90 == 0 {
                    watchdog_count = watchdog_count.saturating_add(1);
                    if !reached_target_stop.get()
                        && (watchdog_count >= max_backoffs || no_improve_epochs >= plateau_patience)
                    {
                        force_stop.set(true);
                        forced_stop_reason = Some(format!(
                            "plateau-stop: epoch={}, noImprove={}, watchdogs={}, bestVal={:.6e}, currentVal={:.6e}",
                            epoch, no_improve_epochs, watchdog_count, best_val, val_loss
                        ));
                    }
                }
                if pino_arch_frozen
                    && !reached_target_stop.get()
                    && no_improve_epochs >= pino_adapt_plateau
                    && forced_stop_reason.is_none()
                {
                    force_stop.set(true);
                    forced_stop_reason = Some(format!(
                        "plateau-stop: epoch={}, noImprove={}, architectureFrozen=true, bestVal={:.6e}, currentVal={:.6e}",
                        epoch, no_improve_epochs, best_val, val_loss
                    ));
                }
                let lr_phase = if lr < last_lr * 0.999 {
                    "pino-decay"
                } else if lr > last_lr * 1.001 {
                    "pino-boost"
                } else {
                    "pino-steady"
                };
                last_lr = lr;
                let emit_progress =
                    epoch == 1 || epoch == total_epochs || epoch % progress_every == 0;
                let emit_network =
                    epoch == 1 || epoch == total_epochs || epoch % network_every == 0;
                last_train_breakdown = train_breakdown;
                last_val_breakdown = val_breakdown;
                let train_data_cap = self
                    .last_config
                    .collocation_points
                    .saturating_add(self.last_config.boundary_points)
                    .saturating_add(self.last_config.interface_points);
                self.set_burn_runtime_state(BurnRuntimeState {
                    architecture: arch.clone(),
                    learning_rate: lr,
                    model_version: status_before.model_version.saturating_add(1),
                    last_loss: loss,
                    best_val_loss: best_val,
                    train_samples: train_data_cap,
                    completed_epochs: epoch,
                    total_epochs,
                    backend_tag: self.backend_runtime.tag().to_string(),
                    last_train_seed: routed.seed,
                    recent_notes: vec![
                        format!("stage={stage}"),
                        format!("optimizer={optimizer_id}"),
                        format!("lrPhase={lr_phase}"),
                        format!(
                            "lossWeights={:.2}/{:.2}/{:.2}/{:.2}",
                            adaptive_weights.0,
                            adaptive_weights.1,
                            adaptive_weights.2,
                            adaptive_weights.3
                        ),
                        format!(
                            "residualFocus={}",
                            Self::residual_focus_label(residual_focus)
                        ),
                        format!(
                            "residualMix=d{:.3e}/s{:.3e}/eq{:.3e}/cn{:.3e}/cs{:.3e}/we{:.3e}/inv{:.3e}/bc{:.3e}",
                            train_breakdown.displacement_fit,
                            train_breakdown.stress_fit,
                            train_breakdown.equilibrium,
                            train_breakdown.constitutive_normal,
                            train_breakdown.constitutive_shear,
                            train_breakdown.weak_energy,
                            train_breakdown.invariant,
                            train_breakdown.boundary
                        ),
                    ],
                    pino: pino_meta.clone(),
                    pino_calibration: pino_calibration.clone(),
                    pino_params: Some(pino_params.clone()),
                    pino_optimizer_state: pino_optimizer_state.clone(),
                });
                if emit_progress {
                    on_epoch(TrainingProgressEvent {
                        epoch,
                        total_epochs,
                        loss,
                        val_loss: current_val_metric,
                        data_loss: pillars.momentum + pillars.kinematics,
                        physics_loss: pillars.material + pillars.boundary,
                        val_data_loss: (val_pillars.momentum + val_pillars.kinematics).max(0.0),
                        val_physics_loss: (val_pillars.material + val_pillars.boundary).max(0.0),
                        momentum_residual: pillars.momentum,
                        kinematic_residual: pillars.kinematics,
                        material_residual: pillars.material,
                        boundary_residual: pillars.boundary,
                        displacement_fit: train_breakdown.displacement_fit,
                        stress_fit: train_breakdown.stress_fit,
                        invariant_residual: train_breakdown.invariant,
                        constitutive_normal_residual: train_breakdown.constitutive_normal,
                        constitutive_shear_residual: train_breakdown.constitutive_shear,
                        val_displacement_fit: val_breakdown.displacement_fit,
                        val_stress_fit: val_breakdown.stress_fit,
                        val_invariant_residual: val_breakdown.invariant,
                        val_constitutive_normal_residual: val_breakdown.constitutive_normal,
                        val_constitutive_shear_residual: val_breakdown.constitutive_shear,
                        residual_weight_momentum: adaptive_weights.0,
                        residual_weight_kinematics: adaptive_weights.1,
                        residual_weight_material: adaptive_weights.2,
                        residual_weight_boundary: adaptive_weights.3,
                        hybrid_mode: format!("{}+operator-field", self.backend_runtime.tag()),
                        stage_id: stage.to_string(),
                        optimizer_id: optimizer_id.to_string(),
                        lr_phase: lr_phase.to_string(),
                        target_band_low: best_val * 0.98,
                        target_band_high: best_val * 1.02,
                        trend_slope,
                        trend_variance,
                        watchdog_trigger_count: watchdog_count,
                        collocation_samples_added: train_data_cap,
                        train_data_size: train_data_cap,
                        train_data_cap,
                        learning_rate: lr,
                        architecture: arch.clone(),
                        progress_ratio: (epoch as f64 / total_epochs as f64).clamp(0.0, 1.0),
                        training_mode: training_mode.clone(),
                        benchmark_id: benchmark_id.clone(),
                        gate_status: "running".to_string(),
                        certified_best_metric: best_val.min(val_loss),
                        dominant_blocker: None,
                        stalled_reason: None,
                        network: Self::burn_network_snapshot(
                            &arch,
                            epoch,
                            lr,
                            pillars,
                            emit_network,
                        ),
                        pino: pino_meta.clone(),
                    });
                }
                if checkpoint_every > 0 && epoch % checkpoint_every == 0 {
                    let mark_best = improved;
                    on_checkpoint(epoch, self.snapshot_state(), mark_best);
                    checkpoint_count = checkpoint_count.saturating_add(1);
                    if mark_best {
                        best_checkpoint_count = best_checkpoint_count.saturating_add(1);
                    }
                }
                if force_stop.get() {
                    break;
                }
            }
            let stats = BurnPilotStats {
                initial_loss: if epochs_run == 0 { 0.0 } else { initial_loss },
                final_loss: if epochs_run == 0 { 0.0 } else { final_loss },
                best_loss: if best_loss.is_finite() {
                    best_loss
                } else {
                    final_loss
                },
                epochs_run,
                learning_rate: lr,
                stopped_early: manual_stop,
            };
            let final_model_version = status_before.model_version.saturating_add(1);
            let final_train_data_cap = self
                .last_config
                .collocation_points
                .saturating_add(self.last_config.boundary_points)
                .saturating_add(self.last_config.interface_points);
            if checkpoint_every > 0 && epochs_run > 0 {
                self.set_burn_runtime_state(BurnRuntimeState {
                    architecture: arch.clone(),
                    learning_rate: stats.learning_rate,
                    model_version: final_model_version,
                    last_loss: stats.final_loss,
                    best_val_loss: best_val.min(stats.best_loss),
                    train_samples: final_train_data_cap,
                    completed_epochs: epochs_run,
                    total_epochs,
                    backend_tag: self.backend_runtime.tag().to_string(),
                    last_train_seed: routed.seed,
                    recent_notes: vec![],
                    pino: pino_meta.clone(),
                    pino_calibration: pino_calibration.clone(),
                    pino_params: Some(pino_params.clone()),
                    pino_optimizer_state: pino_optimizer_state.clone(),
                });
                on_checkpoint(epochs_run, self.snapshot_state(), true);
                checkpoint_count = checkpoint_count.saturating_add(1);
                best_checkpoint_count = best_checkpoint_count.saturating_add(1);
            }
            let mut best_effective_val = if best_val.is_finite() && best_val < f64::MAX {
                best_val
            } else if stats.best_loss.is_finite() {
                stats.best_loss
            } else {
                final_loss
            };
            let mut benchmark_final_breakdown = None::<BurnPhysicsLossBreakdown>;
            let mut benchmark_dominant_blocker = None::<String>;
            let mut benchmark_physical_errors = None::<CantileverBenchmarkPhysicalErrors>;
            let exact_refine_family_enabled = ((low_target_mode || benchmark_isolated_cantilever)
                && (analysis_type.contains("cantilever") || analysis_type.contains("plate-hole")))
                || benchmark_isolated_cantilever;
            if exact_refine_family_enabled && pino_model_config.is_some() && !skip_exact_refine {
                let cfg = pino_model_config.as_ref().expect("config checked above");
                let exact_refine_calibration = pino_calibration
                    .clone()
                    .or_else(|| {
                        fit_operator_calibration_with_params(
                            &routed,
                            cfg,
                            Some(&best_runtime_params),
                        )
                    })
                    .or_else(|| fit_operator_calibration(&routed, cfg));
                if exact_refine_calibration.is_none() {
                    stage_notes.push(
                        "PINOExactRefineSkipped: unable to derive calibration for exact refine"
                            .to_string(),
                    );
                    emit_epoch_zero_progress(
                        &mut on_epoch,
                        "exact-refine",
                        "missing-calibration",
                        &arch,
                        best_runtime_lr,
                        None,
                        best_effective_val.max(0.0),
                        best_effective_val.max(0.0),
                    );
                } else if let Some(calibration) = exact_refine_calibration.as_ref() {
                    if pino_calibration.is_none() {
                        pino_calibration = Some(calibration.clone());
                    }
                    let headless_fast_profile = env_flag("PINO_HEADLESS_FAST_PROFILE", false);
                    let exact_eval_weights = adaptive_weights;
                    let exact_eval_cases = if isolated_exact_cantilever {
                        vec![Self::exact_anchor_case(&routed.cases)]
                    } else {
                        routed.cases.clone()
                    };
                    let exact_eval_targets = Self::build_operator_field_targets_exact(
                        &exact_eval_cases,
                        cfg.operator_grid.clone(),
                    );
                    let benchmark_characteristic_train_scaling = false;
                    let exact_refine_cases = if benchmark_isolated_cantilever {
                        exact_eval_cases.clone()
                    } else if isolated_exact_cantilever {
                        Self::build_exact_refine_family_cases(
                            &exact_eval_cases,
                            &analysis_type,
                            true,
                        )
                    } else {
                        Self::build_exact_refine_family_cases(
                            &routed.cases,
                            &analysis_type,
                            headless_fast_profile,
                        )
                    };
                    let mut exact_train_grid = cfg.operator_grid.clone();
                    if headless_fast_profile
                        && !isolated_exact_cantilever
                        && exact_refine_cases.len() > routed.cases.len()
                    {
                        exact_train_grid.nx = ((exact_train_grid.nx as f64) * 0.82)
                            .round()
                            .clamp(16.0, exact_train_grid.nx as f64)
                            as usize;
                        exact_train_grid.ny = ((exact_train_grid.ny as f64) * 0.82)
                            .round()
                            .clamp(10.0, exact_train_grid.ny as f64)
                            as usize;
                        exact_train_grid.nz = exact_train_grid.nz.clamp(2, 3);
                    }
                    let exact_train_targets = if exact_refine_cases.len() > exact_eval_cases.len() {
                        Self::build_operator_field_targets_exact(
                            &exact_refine_cases,
                            exact_train_grid.clone(),
                        )
                    } else {
                        exact_eval_targets.clone()
                    };
                    if matches!(benchmark_id.as_deref(), Some("benchmark_cantilever_2d")) {
                        stage_notes.push(
                            "PINOExactRefineTrainScaling: benchmark-cantilever exact refine now stays on the same unscaled physical objective used for evaluation".to_string(),
                        );
                    }
                    if isolated_exact_cantilever {
                        stage_notes.push(format!(
                            "PINOExactRefineIsolatedCantilever: evalCases={}, trainCases={}, grid={}x{}x{}",
                            exact_eval_targets.len(),
                            exact_train_targets.len(),
                            cfg.operator_grid.nx,
                            cfg.operator_grid.ny,
                            cfg.operator_grid.nz,
                        ));
                    }
                    if exact_train_targets.len() > exact_eval_targets.len() {
                        stage_notes.push(format!(
                            "PINOExactRefineFamily: seedCases={}, expandedCases={}, evalGrid={}x{}x{}, trainGrid={}x{}x{}",
                            exact_eval_targets.len(),
                            exact_train_targets.len(),
                            cfg.operator_grid.nx,
                            cfg.operator_grid.ny,
                            cfg.operator_grid.nz,
                            exact_train_grid.nx,
                            exact_train_grid.ny,
                            exact_train_grid.nz,
                        ));
                    }
                    if !exact_eval_targets.is_empty() {
                        let mut emit_exact_refine_progress =
                            |phase: String, learning_rate: f64, exact_loss: f64, val_floor: f64| {
                                emit_epoch_zero_progress(
                                    &mut on_epoch,
                                    "exact-refine",
                                    &phase,
                                    &arch,
                                    learning_rate,
                                    None,
                                    exact_loss.max(0.0),
                                    val_floor.max(0.0),
                                );
                            };
                        let exact_samples = exact_eval_targets
                            .iter()
                            .map(|sample| {
                                Self::build_burn_physics_sample(
                                    sample,
                                    cfg,
                                    calibration,
                                    &best_runtime_params,
                                )
                            })
                            .collect::<Vec<_>>();
                        let focused_plate_hole = analysis_type.contains("plate-hole");
                        let focused_cantilever = analysis_type.contains("cantilever");
                        let isolated_exact_surface_eval = isolated_exact_cantilever;
                        let benchmark_metric_for = |params: &OperatorTrainableParams,
                                                    calibration: &OperatorCalibration| {
                            if benchmark_isolated_cantilever {
                                exact_eval_targets
                                    .first()
                                    .map(|exact_target| {
                                        evaluate_cantilever_benchmark_selection_metric(
                                            &exact_target.case,
                                            cfg,
                                            calibration,
                                            params,
                                            &exact_target.target,
                                        )
                                    })
                                    .unwrap_or(f64::MAX)
                            } else {
                                f64::MAX
                            }
                        };
                        let mut refinement_best_loss = evaluate_operator_field_head_physics(
                            &exact_samples,
                            &best_runtime_params.field_head_weights,
                            &best_runtime_params.field_head_bias,
                            &best_runtime_params.field_head_activation,
                            PINO_FIELD_HEAD_BASIS,
                            cfg.operator_grid.output_channels,
                            best_runtime_params.field_head_hidden_layers,
                            best_runtime_params.field_head_hidden_width,
                            exact_eval_weights,
                            isolated_exact_surface_eval,
                        )
                        .map(|breakdown| breakdown.total)
                        .unwrap_or(best_effective_val);
                        let mut refinement_best_metric = if benchmark_isolated_cantilever {
                            benchmark_metric_for(&best_runtime_params, calibration)
                        } else {
                            refinement_best_loss
                        };
                        let mut refinement_best_score = if isolated_exact_cantilever {
                            best_effective_val.min(refinement_best_metric)
                        } else {
                            best_selection_score
                        };
                        let mut refinement_best_params = best_runtime_params.clone();
                        let mut refinement_best_lr = best_runtime_lr;
                        let mut refinement_best_calibration = best_runtime_calibration.clone();
                        let mut refinement_improved = false;
                        let mut refinement_stall_count = 0usize;
                        let baseline_progress_metric = if benchmark_isolated_cantilever {
                            refinement_best_metric
                        } else {
                            refinement_best_loss
                        };
                        if let Some(grad_norm) = evaluate_operator_field_head_physics_grad_norm(
                            &exact_samples,
                            &best_runtime_params.field_head_weights,
                            &best_runtime_params.field_head_bias,
                            &best_runtime_params.field_head_activation,
                            PINO_FIELD_HEAD_BASIS,
                            cfg.operator_grid.output_channels,
                            best_runtime_params.field_head_hidden_layers,
                            best_runtime_params.field_head_hidden_width,
                            exact_eval_weights,
                            isolated_exact_surface_eval,
                        ) {
                            stage_notes.push(format!(
                            "PINOExactRefineProbe: evalCases={}, trainCases={}, grid={}x{}x{}, gradNorm={:.6e}",
                            exact_samples.len(),
                            exact_train_targets.len(),
                            cfg.operator_grid.nx,
                            cfg.operator_grid.ny,
                            cfg.operator_grid.nz,
                            grad_norm
                        ));
                            emit_exact_refine_progress(
                                format!("probe-{:02}", exact_samples.len()),
                                best_runtime_lr,
                                baseline_progress_metric,
                                baseline_progress_metric,
                            );
                        }

                        stage_notes.push(format!(
                        "PINOExactRefineBaseline: exactLoss={:.6e}, score={:.6e}, evalCases={}, trainCases={}",
                        refinement_best_loss,
                        refinement_best_score,
                        exact_eval_targets.len(),
                        exact_train_targets.len()
                    ));
                        emit_exact_refine_progress(
                            "baseline".to_string(),
                            best_runtime_lr,
                            baseline_progress_metric,
                            baseline_progress_metric,
                        );

                        let refine_loss_weights = if focused_plate_hole {
                            (
                                (adaptive_weights.0 * 0.88).max(0.08),
                                (adaptive_weights.1 * 1.55).max(0.35),
                                (adaptive_weights.2 * 1.35).max(0.30),
                                (adaptive_weights.3 * 1.20).max(0.15),
                            )
                        } else if focused_cantilever {
                            (
                                (adaptive_weights.0 * 0.80).max(0.06),
                                (adaptive_weights.1 * 1.60).max(0.40),
                                (adaptive_weights.2 * 1.18).max(0.20),
                                (adaptive_weights.3 * 1.12).max(0.12),
                            )
                        } else {
                            adaptive_weights
                        };
                        if isolated_exact_cantilever {
                            let warm_steps =
                                std::env::var("PINO_HEADLESS_ISOLATED_EXACT_WARM_STEPS")
                                    .ok()
                                    .and_then(|raw| raw.parse::<usize>().ok())
                                    .filter(|value| *value > 0)
                                    .unwrap_or(24);
                            let warm_lr = (best_runtime_lr * 0.85).clamp(5e-5, 1.2e-3);
                            if let Some((warm_params, warm_train_loss)) =
                                Self::direct_train_operator_params_for_targets_with_scaling(
                                    &exact_eval_targets,
                                    cfg,
                                    calibration,
                                    &refinement_best_params,
                                    warm_steps,
                                    warm_lr,
                                    BurnFieldHeadOptimizer::Adam,
                                    refine_loss_weights,
                                    true,
                                    benchmark_characteristic_train_scaling,
                                )
                            {
                                let warm_samples = exact_eval_targets
                                    .iter()
                                    .map(|sample| {
                                        Self::build_burn_physics_sample(
                                            sample,
                                            cfg,
                                            calibration,
                                            &warm_params,
                                        )
                                    })
                                    .collect::<Vec<_>>();
                                if let Some(warm_exact_loss) = evaluate_operator_field_head_physics(
                                    &warm_samples,
                                    &warm_params.field_head_weights,
                                    &warm_params.field_head_bias,
                                    &warm_params.field_head_activation,
                                    PINO_FIELD_HEAD_BASIS,
                                    cfg.operator_grid.output_channels,
                                    warm_params.field_head_hidden_layers,
                                    warm_params.field_head_hidden_width,
                                    exact_eval_weights,
                                    isolated_exact_surface_eval,
                                )
                                .map(|breakdown| breakdown.total)
                                {
                                    let warm_metric = if benchmark_isolated_cantilever {
                                        benchmark_metric_for(&warm_params, calibration)
                                    } else {
                                        warm_exact_loss
                                    };
                                    stage_notes.push(format!(
                                    "PINOExactRefineWarm: optimizer=Adam, steps={}, lr={:.6e}, exactLoss={:.6e}, trainLoss={:.6e}",
                                    warm_steps,
                                    warm_lr,
                                    warm_exact_loss,
                                    warm_train_loss,
                                ));
                                    emit_exact_refine_progress(
                                        "warm-adam".to_string(),
                                        warm_lr,
                                        warm_metric,
                                        refinement_best_metric.min(warm_metric),
                                    );
                                    let warm_margin = if benchmark_isolated_cantilever {
                                        (refinement_best_score
                                            .abs()
                                            .max(routed.target_loss.max(1e-9))
                                            * 0.0005)
                                            .max(1e-8)
                                    } else {
                                        (refinement_best_score
                                            .abs()
                                            .max(routed.target_loss.max(1e-9))
                                            * 0.01)
                                            .max(1e-6)
                                    };
                                    if warm_metric + warm_margin < refinement_best_score
                                    {
                                        refinement_best_score =
                                            refinement_best_score.min(warm_metric);
                                        refinement_best_loss = warm_exact_loss;
                                        refinement_best_metric = warm_metric;
                                        refinement_best_params = warm_params;
                                        refinement_best_lr = warm_lr;
                                        refinement_best_calibration = Some(calibration.clone());
                                        refinement_improved = true;
                                    }
                                }
                            }
                        }
                        let exact_refine_round_cap =
                            std::env::var("PINO_HEADLESS_EXACT_REFINE_ROUNDS")
                                .ok()
                                .and_then(|raw| raw.parse::<usize>().ok())
                                .filter(|value| *value > 0)
                                .unwrap_or_else(|| {
                                    if isolated_exact_cantilever {
                                        10
                                    } else if headless_fast_profile {
                                        2
                                    } else {
                                        usize::MAX
                                    }
                                });
                        let exact_refine_step_cap =
                            std::env::var("PINO_HEADLESS_EXACT_REFINE_STEP_CAP")
                                .ok()
                                .and_then(|raw| raw.parse::<usize>().ok())
                                .filter(|value| *value > 0)
                                .unwrap_or_else(|| {
                                    if isolated_exact_cantilever {
                                        160
                                    } else if headless_fast_profile {
                                        32
                                    } else {
                                        usize::MAX
                                    }
                                });
                        let mut refine_candidates = if isolated_exact_cantilever {
                            vec![(
                                BurnFieldHeadOptimizer::Adam,
                                24usize,
                                (best_runtime_lr * 0.70).clamp(5e-5, 1.2e-3),
                            )]
                        } else if focused_plate_hole {
                            vec![
                                (
                                    BurnFieldHeadOptimizer::Lbfgs,
                                    32usize,
                                    (best_runtime_lr * 0.40).clamp(2e-5, 1.2e-3),
                                ),
                                (
                                    BurnFieldHeadOptimizer::Adam,
                                    18usize,
                                    (best_runtime_lr * 0.55).clamp(3e-5, 1.0e-3),
                                ),
                                (
                                    BurnFieldHeadOptimizer::Lbfgs,
                                    48usize,
                                    (best_runtime_lr * 0.25).clamp(1e-5, 8e-4),
                                ),
                            ]
                        } else {
                            vec![
                                (
                                    BurnFieldHeadOptimizer::Lbfgs,
                                    32usize,
                                    (best_runtime_lr * 0.45).clamp(2e-5, 1.4e-3),
                                ),
                                (
                                    BurnFieldHeadOptimizer::Adam,
                                    12usize,
                                    (best_runtime_lr * 0.60).clamp(3e-5, 1.1e-3),
                                ),
                                (
                                    BurnFieldHeadOptimizer::Lbfgs,
                                    40usize,
                                    (best_runtime_lr * 0.28).clamp(1e-5, 9e-4),
                                ),
                            ]
                        };
                        if exact_refine_step_cap < usize::MAX {
                            for (_, steps, _) in refine_candidates.iter_mut() {
                                *steps = (*steps).min(exact_refine_step_cap).max(1);
                            }
                        }
                        if headless_fast_profile && !isolated_exact_cantilever {
                            refine_candidates.retain(|(optimizer, _, _)| {
                                matches!(optimizer, BurnFieldHeadOptimizer::Lbfgs)
                            });
                            if exact_train_targets.len() > exact_eval_targets.len() {
                                if let Some((_, steps, _)) = refine_candidates.first_mut() {
                                    *steps = (*steps).min(28).max(8);
                                }
                                refine_candidates.truncate(1);
                            }
                        }
                        let refinement_rounds = if isolated_exact_cantilever {
                            8usize
                        } else if focused_plate_hole {
                            3usize
                        } else {
                            2usize
                        }
                        .min(exact_refine_round_cap);
                        let holdout_regression_limit = if isolated_exact_cantilever {
                            1.02
                        } else if focused_plate_hole {
                            1.35
                        } else {
                            1.20
                        };

                        'refine: for round in 0..refinement_rounds {
                            let mut round_improved = false;
                            let isolated_round_lr_scale = if isolated_exact_cantilever {
                                if refinement_best_loss <= 2.0e-2 {
                                    0.90f64.powi(round.saturating_sub(1) as i32)
                                } else {
                                    0.55f64.powi(round as i32)
                                }
                            } else {
                                1.0
                            };
                            let mut refine_starts = vec![refinement_best_params.clone()];
                            let restart_ready = round == 0
                                && !headless_fast_profile
                                && (focused_plate_hole || focused_cantilever)
                                && refinement_best_loss.is_finite()
                                && refinement_best_loss > 7.5e2;
                            if restart_ready {
                                let restart_baseline =
                                    initial_pino_params.clone().aligned_to_config(cfg).clamped();
                                refine_starts.push(Self::blend_operator_params(
                                    &refinement_best_params,
                                    &restart_baseline,
                                    0.18,
                                ));
                            }
                            for (start_idx, current_refine_params) in
                                refine_starts.into_iter().enumerate()
                            {
                                let current_refine_calibration = refinement_best_calibration
                                    .clone()
                                    .unwrap_or_else(|| calibration.clone());
                                for (optimizer, steps, base_learning_rate) in
                                    refine_candidates.iter().copied()
                                {
                                    let learning_rate = if isolated_exact_cantilever {
                                        let resumed_best_lr =
                                            if round > 0 && refinement_best_lr.is_finite() {
                                                (refinement_best_lr * 1.35)
                                                    .clamp(1e-6, base_learning_rate)
                                            } else {
                                                base_learning_rate
                                            };
                                        (resumed_best_lr * isolated_round_lr_scale)
                                            .clamp(1e-6, 1.2e-3)
                                    } else {
                                        base_learning_rate
                                    };
                                    if let Some((candidate_params, candidate_train_loss)) =
                                        Self::direct_train_operator_params_for_targets_with_scaling(
                                            &exact_train_targets,
                                            cfg,
                                            &current_refine_calibration,
                                            &current_refine_params,
                                            steps,
                                            learning_rate,
                                            optimizer,
                                            refine_loss_weights,
                                            true,
                                            benchmark_characteristic_train_scaling,
                                        )
                                    {
                                        let polish_steps = if headless_fast_profile
                                            && exact_train_targets.len() > exact_eval_targets.len()
                                        {
                                            steps.min(12).max(6)
                                        } else {
                                            steps
                                                .min(if headless_fast_profile { 12 } else { 16 })
                                                .max(4)
                                        };
                                        let polish_lr = (learning_rate * 0.55).clamp(1e-5, 8e-4);
                                        let (candidate_params, candidate_train_loss) =
                                            if isolated_exact_cantilever {
                                                if matches!(optimizer, BurnFieldHeadOptimizer::Adam)
                                                {
                                                    let shaped_steps = polish_steps.min(18).max(6);
                                                    let shaped_lr = polish_lr.min(4e-4);
                                                    let (candidate_params, candidate_train_loss) = if let Some((polished_params, polished_train_loss)) =
                                            Self::direct_train_operator_params_for_targets_with_scaling(
                                                &exact_eval_targets,
                                                cfg,
                                                &current_refine_calibration,
                                                &candidate_params,
                                                shaped_steps,
                                                shaped_lr,
                                                BurnFieldHeadOptimizer::Adam,
                                                refine_loss_weights,
                                                true,
                                                benchmark_characteristic_train_scaling,
                                            )
                                        {
                                            (polished_params, polished_train_loss)
                                        } else {
                                            (candidate_params, candidate_train_loss)
                                        };
                                                    let exact_adam_steps =
                                                        (shaped_steps / 2).max(6);
                                                    let exact_adam_lr =
                                                        (shaped_lr * 0.40).clamp(1e-5, 1.5e-4);
                                                    if let Some((polished_params, polished_train_loss)) =
                                            Self::direct_train_operator_params_for_targets_with_scaling(
                                                &exact_eval_targets,
                                                cfg,
                                                &current_refine_calibration,
                                                &candidate_params,
                                                exact_adam_steps,
                                                exact_adam_lr,
                                                BurnFieldHeadOptimizer::Adam,
                                                exact_eval_weights,
                                                isolated_exact_surface_eval,
                                                benchmark_characteristic_train_scaling,
                                            )
                                        {
                                            (polished_params, polished_train_loss)
                                        } else {
                                            (candidate_params, candidate_train_loss)
                                        }
                                                } else {
                                                    (candidate_params, candidate_train_loss)
                                                }
                                            } else if let Some((
                                                polished_params,
                                                polished_train_loss,
                                            )) =
                                                Self::direct_train_operator_params_for_targets_with_scaling(
                                                    &exact_eval_targets,
                                                    cfg,
                                                    &current_refine_calibration,
                                                    &candidate_params,
                                                    polish_steps,
                                                    polish_lr,
                                                    BurnFieldHeadOptimizer::Lbfgs,
                                                    exact_eval_weights,
                                                    false,
                                                    false,
                                                )
                                            {
                                                (polished_params, polished_train_loss)
                                            } else {
                                                (candidate_params, candidate_train_loss)
                                            };
                                        let candidate_calibration =
                                            Some(current_refine_calibration.clone());
                                        let candidate_exact_loss = candidate_calibration
                                            .as_ref()
                                            .and_then(|candidate_calibration| {
                                                let candidate_exact_samples = exact_eval_targets
                                                    .iter()
                                                    .map(|sample| {
                                                        Self::build_burn_physics_sample(
                                                            sample,
                                                            cfg,
                                                            candidate_calibration,
                                                            &candidate_params,
                                                        )
                                                    })
                                                    .collect::<Vec<_>>();
                                                evaluate_operator_field_head_physics(
                                                    &candidate_exact_samples,
                                                    &candidate_params.field_head_weights,
                                                    &candidate_params.field_head_bias,
                                                    &candidate_params.field_head_activation,
                                                    PINO_FIELD_HEAD_BASIS,
                                                    cfg.operator_grid.output_channels,
                                                    candidate_params.field_head_hidden_layers,
                                                    candidate_params.field_head_hidden_width,
                                                    exact_eval_weights,
                                                    isolated_exact_surface_eval,
                                                )
                                            })
                                            .map(|breakdown| breakdown.total)
                                            .unwrap_or(f64::MAX);
                                        let candidate_score = if isolated_exact_cantilever {
                                            if benchmark_isolated_cantilever {
                                                candidate_calibration
                                                    .as_ref()
                                                    .map(|candidate_calibration| {
                                                        benchmark_metric_for(
                                                            &candidate_params,
                                                            candidate_calibration,
                                                        )
                                                    })
                                                    .unwrap_or(f64::MAX)
                                            } else {
                                                candidate_exact_loss
                                            }
                                        } else {
                                            evaluate_holdout_projection_with_model(
                                                &routed,
                                                cfg,
                                                candidate_calibration.as_ref(),
                                                Some(&candidate_params),
                                            )
                                            .map(|summary| Self::holdout_selection_score(&summary))
                                            .unwrap_or(f64::MAX)
                                        };
                                        stage_notes.push(format!(
                                    "PINOExactRefineCandidate: round={}, start={}, optimizer={:?}, steps={}, lr={:.6e}, exactLoss={:.6e}, trainLoss={:.6e}, score={:.6e}",
                                    round + 1,
                                    start_idx + 1,
                                    optimizer,
                                    steps,
                                    learning_rate,
                                    candidate_exact_loss,
                                    candidate_train_loss,
                                    candidate_score,
                                ));
                                        emit_exact_refine_progress(
                                            format!(
                                                "candidate-r{}-s{}-{:?}",
                                                round + 1,
                                                start_idx + 1,
                                                optimizer
                                            )
                                            .to_ascii_lowercase(),
                                            learning_rate,
                                            candidate_score,
                                            refinement_best_score.min(candidate_score),
                                        );
                                        let loss_margin = if benchmark_isolated_cantilever {
                                            refinement_best_score
                                                .abs()
                                                .max(routed.target_loss.max(1e-9))
                                                * 0.0005
                                        } else {
                                            refinement_best_score
                                                .abs()
                                                .max(routed.target_loss.max(1e-9))
                                                * 0.02
                                        };
                                        let score_margin = if refinement_best_score.is_finite() {
                                            if benchmark_isolated_cantilever {
                                                (refinement_best_score.abs() * 0.0005).max(1e-8)
                                            } else {
                                                (refinement_best_score.abs() * 0.01).max(1e-6)
                                            }
                                        } else {
                                            1e-6
                                        };
                                        let holdout_ok = !refinement_best_score.is_finite()
                                            || candidate_score
                                                <= refinement_best_score * holdout_regression_limit
                                                    + score_margin;
                                        let loss_gain = candidate_score + loss_margin
                                            < refinement_best_score;
                                        let focused_exact_override = (focused_plate_hole
                                            || focused_cantilever)
                                            && loss_gain
                                            && candidate_score.is_finite()
                                            && (headless_fast_profile
                                                || candidate_score.is_finite());
                                        if (holdout_ok && loss_gain)
                                            || focused_exact_override
                                        {
                                            refinement_best_score =
                                                refinement_best_score.min(candidate_score);
                                            refinement_best_loss = candidate_exact_loss;
                                            refinement_best_metric =
                                                refinement_best_metric.min(candidate_score);
                                            refinement_best_params = candidate_params;
                                            refinement_best_lr = learning_rate;
                                            if let Some(candidate_calibration) =
                                                candidate_calibration
                                            {
                                                refinement_best_calibration =
                                                    Some(candidate_calibration);
                                            }
                                            refinement_improved = true;
                                            round_improved = true;
                                            refinement_stall_count = 0;
                                            if isolated_exact_cantilever
                                                && matches!(optimizer, BurnFieldHeadOptimizer::Adam)
                                            {
                                                let mut chain_params =
                                                    refinement_best_params.clone();
                                                let chain_cap =
                                                    std::env::var("PINO_HEADLESS_EXACT_CHAIN_CAP")
                                                        .ok()
                                                        .and_then(|raw| raw.parse::<usize>().ok())
                                                        .filter(|value| *value > 0)
                                                        .unwrap_or(48);
                                                let mut chain_lr =
                                                    (learning_rate * 0.72).clamp(1e-6, 4e-4);
                                                let mut chain_stall_count = 0usize;
                                                for chain_idx in 0..chain_cap {
                                                    let chain_steps = (steps + 24)
                                                        .min(exact_refine_step_cap.min(64))
                                                        .max(24);
                                                    let Some((
                                                        next_chain_params,
                                                        next_chain_train_loss,
                                                    )) = ({
                                                        let control_metric = if benchmark_isolated_cantilever {
                                                            refinement_best_score
                                                        } else {
                                                            refinement_best_loss
                                                        };
                                                        let chain_train_weights =
                                                            if control_metric <= 2.0e-2 {
                                                                (
                                                                    exact_eval_weights.0 * 0.96,
                                                                    exact_eval_weights.1 * 2.60,
                                                                    exact_eval_weights.2 * 1.55,
                                                                    exact_eval_weights.3 * 0.98,
                                                                )
                                                            } else if control_metric <= 0.5 {
                                                                (
                                                                    exact_eval_weights.0 * 1.02,
                                                                    exact_eval_weights.1 * 2.05,
                                                                    exact_eval_weights.2 * 1.78,
                                                                    exact_eval_weights.3 * 1.00,
                                                                )
                                                            } else if control_metric <= 1.0 {
                                                                (
                                                                    exact_eval_weights.0 * 1.12,
                                                                    exact_eval_weights.1 * 1.72,
                                                                    exact_eval_weights.2 * 2.55,
                                                                    exact_eval_weights.3 * 1.02,
                                                                )
                                                            } else if control_metric <= 2.5 {
                                                                (
                                                                    exact_eval_weights.0 * 1.10,
                                                                    exact_eval_weights.1 * 1.40,
                                                                    exact_eval_weights.2 * 1.85,
                                                                    exact_eval_weights.3 * 1.00,
                                                                )
                                                            } else {
                                                                exact_eval_weights
                                                            };
                                                        Self::direct_train_operator_params_for_targets_with_scaling(
                                                            &exact_eval_targets,
                                                            cfg,
                                                            refinement_best_calibration
                                                                .as_ref()
                                                                .unwrap_or(&current_refine_calibration),
                                                            &chain_params,
                                                            chain_steps,
                                                            chain_lr,
                                                            BurnFieldHeadOptimizer::Adam,
                                                            chain_train_weights,
                                                            isolated_exact_surface_eval,
                                                            benchmark_characteristic_train_scaling,
                                                        )
                                                    })
                                                    else {
                                                        break;
                                                    };
                                                    let next_chain_exact_samples =
                                                        exact_eval_targets
                                                            .iter()
                                                            .map(|sample| {
                                                                Self::build_burn_physics_sample(
                                                        sample,
                                                        cfg,
                                                        refinement_best_calibration
                                                            .as_ref()
                                                            .unwrap_or(&current_refine_calibration),
                                                        &next_chain_params,
                                                    )
                                                            })
                                                            .collect::<Vec<_>>();
                                                    let Some(next_chain_exact_loss) =
                                                        evaluate_operator_field_head_physics(
                                                            &next_chain_exact_samples,
                                                            &next_chain_params.field_head_weights,
                                                            &next_chain_params.field_head_bias,
                                                            &next_chain_params
                                                                .field_head_activation,
                                                            PINO_FIELD_HEAD_BASIS,
                                                            cfg.operator_grid.output_channels,
                                                            next_chain_params
                                                                .field_head_hidden_layers,
                                                            next_chain_params
                                                                .field_head_hidden_width,
                                                            exact_eval_weights,
                                                            isolated_exact_surface_eval,
                                                        )
                                                        .map(|breakdown| breakdown.total)
                                                    else {
                                                        break;
                                                    };
                                                    stage_notes.push(format!(
                                                "PINOExactRefineChain: step={}, optimizer=Adam, steps={}, lr={:.6e}, exactLoss={:.6e}, trainLoss={:.6e}",
                                                chain_idx + 1,
                                                chain_steps,
                                                chain_lr,
                                                next_chain_exact_loss,
                                                next_chain_train_loss,
                                            ));
                                                    let next_chain_metric =
                                                        if benchmark_isolated_cantilever {
                                                            benchmark_metric_for(
                                                                &next_chain_params,
                                                                refinement_best_calibration
                                                                    .as_ref()
                                                                    .unwrap_or(
                                                                        &current_refine_calibration,
                                                                    ),
                                                            )
                                                        } else {
                                                            next_chain_exact_loss
                                                        };
                                                    emit_exact_refine_progress(
                                                        format!("chain-{:02}", chain_idx + 1),
                                                        chain_lr,
                                                        next_chain_metric,
                                                        refinement_best_score
                                                            .min(next_chain_metric),
                                                    );
                                                    let chain_margin = (refinement_best_score
                                                        .abs()
                                                        .max(routed.target_loss.max(1e-12))
                                                        * 0.0005)
                                                        .max(1e-12);
                                                    if next_chain_metric + chain_margin
                                                        < refinement_best_score
                                                    {
                                                        refinement_best_score =
                                                            refinement_best_score
                                                                .min(next_chain_metric);
                                                        refinement_best_loss = next_chain_exact_loss;
                                                        refinement_best_metric = next_chain_metric;
                                                        refinement_best_params =
                                                            next_chain_params.clone();
                                                        refinement_best_lr = chain_lr;
                                                        chain_params = next_chain_params;
                                                        chain_lr =
                                                            (chain_lr * 0.88).clamp(1e-6, 4e-4);
                                                        chain_stall_count = 0;
                                                    } else {
                                                        chain_stall_count =
                                                            chain_stall_count.saturating_add(1);
                                                        if chain_stall_count >= 3 {
                                                            break;
                                                        }
                                                        chain_lr =
                                                            (chain_lr * 0.84).clamp(1e-6, 4e-4);
                                                    }
                                                    }
                                                let micro_entry_metric = if benchmark_isolated_cantilever {
                                                    refinement_best_score
                                                } else {
                                                    refinement_best_loss
                                                };
                                                if refinement_best_loss.is_finite()
                                                    && micro_entry_metric <= 6.0
                                                {
                                                    let mut micro_params =
                                                        refinement_best_params.clone();
                                                    for micro_round in 0..8 {
                                                        let control_metric = if benchmark_isolated_cantilever {
                                                            refinement_best_score
                                                        } else {
                                                            refinement_best_loss
                                                        };
                                                        let base_lr = if control_metric <= 1.5
                                                        {
                                                            let low_floor_lr =
                                                                if isolated_exact_cantilever
                                                                    && control_metric
                                                                        <= 2.0e-2
                                                                {
                                                                    2.0e-6
                                                                } else {
                                                                    1.0e-6
                                                                };
                                                            refinement_best_lr
                                                                .clamp(low_floor_lr, 5e-5)
                                                        } else {
                                                            refinement_best_lr.clamp(2e-6, 9e-5)
                                                        };
                                                        let micro_specs = if isolated_exact_cantilever
                                                            && control_metric <= 2.2e-2
                                                        {
                                                            [
                                                                (10usize, 2.0e-6),
                                                                (16usize, 1.6e-6),
                                                                (24usize, 1.32e-6),
                                                                (36usize, 1.0e-6),
                                                            ]
                                                        } else if control_metric <= 0.85 {
                                                            [
                                                                (
                                                                    10usize,
                                                                    (base_lr * 1.02).clamp(1e-6, 7e-5),
                                                                ),
                                                                (
                                                                    16usize,
                                                                    (base_lr * 0.84).clamp(1e-6, 6e-5),
                                                                ),
                                                                (
                                                                    24usize,
                                                                    (base_lr * 0.66).clamp(1e-6, 5e-5),
                                                                ),
                                                                (
                                                                    36usize,
                                                                    (base_lr * 0.50).clamp(1e-6, 4e-5),
                                                                ),
                                                            ]
                                                        } else if control_metric <= 1.0 {
                                                            [
                                                                (
                                                                    8usize,
                                                                    (base_lr * 1.18).clamp(1e-6, 9e-5),
                                                                ),
                                                                (
                                                                    12usize,
                                                                    (base_lr * 0.94).clamp(1e-6, 8e-5),
                                                                ),
                                                                (
                                                                    18usize,
                                                                    (base_lr * 0.74).clamp(1e-6, 7e-5),
                                                                ),
                                                                (
                                                                    24usize,
                                                                    (base_lr * 0.58).clamp(1e-6, 6e-5),
                                                                ),
                                                            ]
                                                        } else {
                                                            [
                                                                (
                                                                    6usize,
                                                                    (base_lr * 1.00).clamp(1e-6, 9e-5),
                                                                ),
                                                                (
                                                                    8usize,
                                                                    (base_lr * 0.82).clamp(1e-6, 8e-5),
                                                                ),
                                                                (
                                                                    12usize,
                                                                    (base_lr * 0.68).clamp(1e-6, 7e-5),
                                                                ),
                                                                (
                                                                    16usize,
                                                                    (base_lr * 0.55).clamp(1e-6, 6e-5),
                                                                ),
                                                            ]
                                                        };
                                                        let mut micro_best: Option<(
                                                            OperatorTrainableParams,
                                                            f64,
                                                            f64,
                                                            f64,
                                                            usize,
                                                        )> = None;
                                                        for (micro_steps, micro_lr) in micro_specs {
                                                            let Some((next_micro_params, next_micro_train_loss)) =
                                                        Self::direct_train_operator_params_for_targets_with_scaling(
                                                            &exact_eval_targets,
                                                            cfg,
                                                            refinement_best_calibration
                                                                .as_ref()
                                                                .unwrap_or(&current_refine_calibration),
                                                            &micro_params,
                                                            micro_steps,
                                                            micro_lr,
                                                            BurnFieldHeadOptimizer::Adam,
                                                            if refinement_best_loss <= 1.0 {
                                                                (
                                                                    exact_eval_weights.0 * 1.08,
                                                                    exact_eval_weights.1 * 1.48,
                                                                    exact_eval_weights.2 * 2.25,
                                                                    exact_eval_weights.3 * 0.96,
                                                                )
                                                            } else if refinement_best_loss <= 1.5 {
                                                                exact_eval_weights
                                                            } else {
                                                                (
                                                                    exact_eval_weights.0 * 1.06,
                                                                    exact_eval_weights.1 * 1.28,
                                                                    exact_eval_weights.2 * 1.85,
                                                                    exact_eval_weights.3 * 0.98,
                                                                )
                                                            },
                                                            isolated_exact_surface_eval,
                                                            benchmark_characteristic_train_scaling,
                                                        )
                                                    else {
                                                        continue;
                                                    };
                                                            let next_micro_exact_samples = exact_eval_targets
                                                        .iter()
                                                        .map(|sample| {
                                                            Self::build_burn_physics_sample(
                                                                sample,
                                                                cfg,
                                                                refinement_best_calibration
                                                                    .as_ref()
                                                                    .unwrap_or(
                                                                        &current_refine_calibration,
                                                                    ),
                                                                &next_micro_params,
                                                            )
                                                        })
                                                        .collect::<Vec<_>>();
                                                            let Some(next_micro_exact_loss) =
                                                        evaluate_operator_field_head_physics(
                                                            &next_micro_exact_samples,
                                                            &next_micro_params.field_head_weights,
                                                            &next_micro_params.field_head_bias,
                                                            &next_micro_params.field_head_activation,
                                                            PINO_FIELD_HEAD_BASIS,
                                                            cfg.operator_grid.output_channels,
                                                            next_micro_params.field_head_hidden_layers,
                                                            next_micro_params.field_head_hidden_width,
                                                            exact_eval_weights,
                                                            isolated_exact_surface_eval,
                                                        )
                                                        .map(|breakdown| breakdown.total)
                                                    else {
                                                        continue;
                                                    };
                                                            let next_micro_metric =
                                                                if benchmark_isolated_cantilever {
                                                                    benchmark_metric_for(
                                                                        &next_micro_params,
                                                                        refinement_best_calibration
                                                                            .as_ref()
                                                                            .unwrap_or(
                                                                                &current_refine_calibration,
                                                                            ),
                                                                    )
                                                                } else {
                                                                    next_micro_exact_loss
                                                                };
                                                            stage_notes.push(format!(
                                                        "PINOExactRefineMicro: round={}, steps={}, lr={:.6e}, exactLoss={:.6e}, trainLoss={:.6e}",
                                                        micro_round + 1,
                                                        micro_steps,
                                                        micro_lr,
                                                        next_micro_exact_loss,
                                                        next_micro_train_loss,
                                                    ));
                                                            emit_exact_refine_progress(
                                                                format!(
                                                                    "micro-r{}-s{}",
                                                                    micro_round + 1,
                                                                    micro_steps
                                                                ),
                                                                micro_lr,
                                                                next_micro_metric,
                                                                refinement_best_score
                                                                    .min(next_micro_metric),
                                                            );
                                                            let replace = micro_best
                                                                .as_ref()
                                                                .map(|(_, best_metric, _, _, _)| {
                                                                    next_micro_metric
                                                                        < *best_metric
                                                                })
                                                                .unwrap_or(true);
                                                            if replace {
                                                                micro_best = Some((
                                                                    next_micro_params,
                                                                    next_micro_metric,
                                                                    next_micro_train_loss,
                                                                    micro_lr,
                                                                    micro_steps,
                                                                ));
                                                            }
                                                        }
                                                        let control_metric = if benchmark_isolated_cantilever {
                                                            refinement_best_score
                                                        } else {
                                                            refinement_best_loss
                                                        };
                                                        let micro_margin = if isolated_exact_cantilever
                                                            && control_metric <= 0.85
                                                        {
                                                            (control_metric
                                                                .abs()
                                                                .max(routed.target_loss.max(1e-12))
                                                                * 0.00008)
                                                                .max(5e-13)
                                                        } else {
                                                            (control_metric
                                                                .abs()
                                                                .max(routed.target_loss.max(1e-12))
                                                                * 0.00025)
                                                                .max(1e-12)
                                                        };
                                                        let Some((
                                                            next_micro_params,
                                                            next_micro_metric,
                                                            _next_micro_train_loss,
                                                            next_micro_lr,
                                                            _next_micro_steps,
                                                        )) = micro_best
                                                        else {
                                                            break;
                                                        };
                                                        if next_micro_metric + micro_margin
                                                            < refinement_best_score
                                                        {
                                                            refinement_best_score =
                                                                refinement_best_score
                                                                    .min(next_micro_metric);
                                                            refinement_best_metric =
                                                                next_micro_metric;
                                                            refinement_best_params =
                                                                next_micro_params.clone();
                                                            refinement_best_lr = next_micro_lr;
                                                            micro_params = next_micro_params;
                                                        } else {
                                                            break;
                                                        }
                                                    }
                                                    let tail_entry_metric = if benchmark_isolated_cantilever {
                                                        refinement_best_score
                                                    } else {
                                                        refinement_best_loss
                                                    };
                                                    if refinement_best_loss.is_finite()
                                                        && tail_entry_metric <= 0.80
                                                    {
                                                        let mut tail_params =
                                                            refinement_best_params.clone();
                                                        let benchmark_post_micro_tail =
                                                            isolated_exact_cantilever
                                                                && tail_entry_metric <= 2.0e-2;
                                                        let mut tail_lr = if benchmark_post_micro_tail
                                                        {
                                                            (refinement_best_lr * 1.25)
                                                                .clamp(1.6e-6, 3.2e-6)
                                                        } else {
                                                            refinement_best_lr
                                                                .clamp(1e-6, 2.5e-5)
                                                                * 0.92
                                                        };
                                                        let tail_train_weights =
                                                            if benchmark_post_micro_tail {
                                                                (
                                                                    exact_eval_weights.0 * 0.96,
                                                                    exact_eval_weights.1 * 2.60,
                                                                    exact_eval_weights.2 * 1.55,
                                                                    exact_eval_weights.3 * 0.98,
                                                                )
                                                            } else {
                                                                exact_eval_weights
                                                            };
                                                        let mut tail_stall_count = 0usize;
                                                        let tail_cap = if benchmark_post_micro_tail {
                                                            3usize
                                                        } else {
                                                            4usize
                                                        };
                                                        for tail_idx in 0..tail_cap {
                                                            let tail_steps = if benchmark_post_micro_tail {
                                                                24usize
                                                            } else if tail_idx < 2 {
                                                                18usize
                                                            } else {
                                                                24usize
                                                            };
                                                            let Some((
                                                                next_tail_params,
                                                                next_tail_train_loss,
                                                            )) = Self::direct_train_operator_params_for_targets_with_scaling(
                                                                &exact_eval_targets,
                                                                cfg,
                                                                refinement_best_calibration
                                                                    .as_ref()
                                                                    .unwrap_or(
                                                                        &current_refine_calibration,
                                                                    ),
                                                                &tail_params,
                                                                tail_steps,
                                                                tail_lr,
                                                                BurnFieldHeadOptimizer::Adam,
                                                                tail_train_weights,
                                                                isolated_exact_surface_eval,
                                                                benchmark_characteristic_train_scaling,
                                                            ) else {
                                                                break;
                                                            };
                                                            let next_tail_exact_samples = exact_eval_targets
                                                                .iter()
                                                                .map(|sample| {
                                                                    Self::build_burn_physics_sample(
                                                                        sample,
                                                                        cfg,
                                                                        refinement_best_calibration
                                                                            .as_ref()
                                                                            .unwrap_or(
                                                                                &current_refine_calibration,
                                                                            ),
                                                                        &next_tail_params,
                                                                    )
                                                                })
                                                                .collect::<Vec<_>>();
                                                            let Some(next_tail_exact_loss) =
                                                                evaluate_operator_field_head_physics(
                                                                    &next_tail_exact_samples,
                                                                    &next_tail_params.field_head_weights,
                                                                    &next_tail_params.field_head_bias,
                                                                    &next_tail_params.field_head_activation,
                                                                    PINO_FIELD_HEAD_BASIS,
                                                                    cfg.operator_grid.output_channels,
                                                                    next_tail_params.field_head_hidden_layers,
                                                                    next_tail_params.field_head_hidden_width,
                                                                    exact_eval_weights,
                                                                    isolated_exact_surface_eval,
                                                                )
                                                                .map(|breakdown| breakdown.total)
                                                            else {
                                                                break;
                                                            };
                                                            let next_tail_metric =
                                                                if benchmark_isolated_cantilever {
                                                                    benchmark_metric_for(
                                                                        &next_tail_params,
                                                                        refinement_best_calibration
                                                                            .as_ref()
                                                                            .unwrap_or(
                                                                                &current_refine_calibration,
                                                                            ),
                                                                    )
                                                                } else {
                                                                    next_tail_exact_loss
                                                                };
                                                            stage_notes.push(format!(
                                                                "PINOExactRefineTailChain: step={}, steps={}, lr={:.6e}, exactLoss={:.6e}, trainLoss={:.6e}",
                                                                tail_idx + 1,
                                                                tail_steps,
                                                                tail_lr,
                                                                next_tail_exact_loss,
                                                                next_tail_train_loss,
                                                            ));
                                                            emit_exact_refine_progress(
                                                                format!("tail-chain-{:02}", tail_idx + 1),
                                                                tail_lr,
                                                                next_tail_metric,
                                                                refinement_best_score
                                                                    .min(next_tail_metric),
                                                            );
                                                            let tail_margin = (tail_entry_metric
                                                                .abs()
                                                                .max(routed.target_loss.max(1e-12))
                                                                * 0.0002)
                                                                .max(1e-12);
                                                            if next_tail_metric + tail_margin
                                                                < refinement_best_score
                                                            {
                                                                refinement_best_score =
                                                                    refinement_best_score
                                                                        .min(next_tail_metric);
                                                                refinement_best_metric =
                                                                    next_tail_metric;
                                                                refinement_best_params =
                                                                    next_tail_params.clone();
                                                                refinement_best_lr = tail_lr;
                                                                tail_params = next_tail_params;
                                                                tail_lr = if benchmark_post_micro_tail {
                                                                    (tail_lr * 0.92).clamp(1.2e-6, 3.2e-6)
                                                                } else {
                                                                    (tail_lr * 0.84).clamp(1e-6, 2e-5)
                                                                };
                                                                tail_stall_count = 0;
                                                            } else {
                                                                tail_stall_count =
                                                                    tail_stall_count.saturating_add(1);
                                                                if tail_stall_count >= 2 {
                                                                    break;
                                                                }
                                                                tail_lr = if benchmark_post_micro_tail {
                                                                    (tail_lr * 0.82).clamp(1.2e-6, 3.2e-6)
                                                                } else {
                                                                    (tail_lr * 0.72).clamp(1e-6, 2e-5)
                                                                };
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            let completion_metric =
                                                if benchmark_isolated_cantilever {
                                                    refinement_best_score
                                                } else {
                                                    refinement_best_loss
                                                };
                                            if completion_metric <= routed.target_loss {
                                                break 'refine;
                                            }
                                            if headless_fast_profile
                                                && !isolated_exact_cantilever
                                                && refinement_best_loss <= 5.0e2
                                            {
                                                break 'refine;
                                            }
                                            if headless_fast_profile && !isolated_exact_cantilever {
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                            if !round_improved {
                                refinement_stall_count = refinement_stall_count.saturating_add(1);
                                if refinement_stall_count
                                    >= if isolated_exact_cantilever { 3 } else { 1 }
                                {
                                    break;
                                }
                            } else if isolated_exact_cantilever
                                && if benchmark_isolated_cantilever {
                                    refinement_best_score <= routed.target_loss
                                } else {
                                    refinement_best_loss <= routed.target_loss
                                }
                            {
                                break 'refine;
                            }
                        }

                        if refinement_improved {
                            best_selection_score = refinement_best_score;
                            best_effective_val = if benchmark_isolated_cantilever {
                                best_effective_val.min(refinement_best_metric)
                            } else {
                                best_effective_val.min(refinement_best_loss)
                            };
                            best_runtime_params = refinement_best_params;
                            best_runtime_lr = refinement_best_lr;
                            best_runtime_calibration = refinement_best_calibration;
                            best_runtime_meta = pino_model_config.as_ref().map(|cfg| {
                                runtime_metadata(
                                    &routed,
                                    self.backend_runtime.tag(),
                                    cfg.spectral_modes,
                                )
                            });
                            if let (Some(meta), Some(summary)) = (
                                best_runtime_meta.as_mut(),
                                evaluate_holdout_projection_with_model(
                                    &routed,
                                    cfg,
                                    best_runtime_calibration.as_ref(),
                                    Some(&best_runtime_params),
                                ),
                            ) {
                                meta.holdout_validation = Some(summary);
                            }
                            if let (Some(meta), Some(calibration)) = (
                                best_runtime_meta.as_mut(),
                                best_runtime_calibration.as_ref(),
                            ) {
                                meta.calibration_stress_scale = Some(calibration.stress_scale);
                                meta.calibration_displacement_scale =
                                    Some(calibration.displacement_scale);
                            }
                            if isolated_exact_cantilever {
                                if let Some(calibration) = best_runtime_calibration.as_ref() {
                                    let refined_exact_samples = exact_eval_targets
                                        .iter()
                                        .map(|sample| {
                                            Self::build_burn_physics_sample(
                                                sample,
                                                cfg,
                                                calibration,
                                                &best_runtime_params,
                                            )
                                        })
                                        .collect::<Vec<_>>();
                                    if let Some(refined_breakdown) =
                                        evaluate_operator_field_head_physics(
                                            &refined_exact_samples,
                                            &best_runtime_params.field_head_weights,
                                            &best_runtime_params.field_head_bias,
                                            &best_runtime_params.field_head_activation,
                                            PINO_FIELD_HEAD_BASIS,
                                            cfg.operator_grid.output_channels,
                                            best_runtime_params.field_head_hidden_layers,
                                            best_runtime_params.field_head_hidden_width,
                                            exact_eval_weights,
                                            isolated_exact_surface_eval,
                                        )
                                    {
                                        stage_notes.push(format!(
                                        "PINOExactRefinePostBreakdown: total={:.6e}, disp={:.6e}, stress={:.6e}, obs={:.6e}, eq={:.6e}, cn={:.6e}, cs={:.6e}, we={:.6e}, bc={:.6e}",
                                        refined_breakdown.total,
                                        refined_breakdown.displacement_fit,
                                        refined_breakdown.stress_fit,
                                        refined_breakdown.observable,
                                        refined_breakdown.equilibrium,
                                        refined_breakdown.constitutive_normal,
                                        refined_breakdown.constitutive_shear,
                                        refined_breakdown.weak_energy,
                                        refined_breakdown.boundary,
                                    ));
                                        let mut post_breakdown_metric = refined_breakdown.total;
                                        if benchmark_isolated_cantilever {
                                            if let Some(exact_target) = exact_eval_targets.first() {
                                                let mut calibrated_prediction =
                                                    build_operator_prediction_with_params(
                                                        &exact_target.case,
                                                        cfg,
                                                        Some(&best_runtime_params),
                                                    );
                                                let calibrated_prediction = if cfg
                                                    .operator_grid
                                                    .output_channels
                                                    == PINO_DISPLACEMENT_OUTPUT_CHANNELS
                                                {
                                                    apply_operator_displacement_calibration(
                                                        &mut calibrated_prediction,
                                                        calibration,
                                                    );
                                                    reconstruct_prediction_linear_elastic_from_displacement_smoothed(
                                                        &exact_target.case,
                                                        &calibrated_prediction,
                                                    )
                                                } else {
                                                    apply_operator_calibration(
                                                        &mut calibrated_prediction,
                                                        calibration,
                                                    );
                                                    calibrated_prediction
                                                };
                                                let physical_errors =
                                                    evaluate_cantilever_benchmark_physical_errors(
                                                        &calibrated_prediction,
                                                        &exact_target.target,
                                                    );
                                                stage_notes.push(format!(
                                                    "BenchmarkCantileverPhysicalErrors: tipDisp={:.6e}, maxDisp={:.6e}, meanVm={:.6e}, maxSxx={:.6e}",
                                                    physical_errors.tip_displacement_relative_error,
                                                    physical_errors.max_displacement_relative_error,
                                                    physical_errors.mean_von_mises_relative_error,
                                                    physical_errors.max_sigma_xx_relative_error,
                                                ));
                                                post_breakdown_metric =
                                                    cantilever_benchmark_selection_score(
                                                        &physical_errors,
                                                    );
                                                benchmark_physical_errors =
                                                    Some(physical_errors);
                                            }
                                        }
                                        emit_exact_refine_progress(
                                            "post-breakdown".to_string(),
                                            best_runtime_lr,
                                            post_breakdown_metric,
                                            best_effective_val.min(post_breakdown_metric),
                                        );
                                        benchmark_dominant_blocker =
                                            benchmark_dominant_blocker_from_breakdown(
                                                &refined_breakdown,
                                            );
                                        benchmark_final_breakdown = Some(refined_breakdown);
                                    }
                                }
                            }
                            stage_notes.push(format!(
                                "PINOExactRefineApplied: score={:.6e}, valFloor={:.6e}, lr={:.6e}",
                                best_selection_score, best_effective_val, best_runtime_lr
                            ));
                            emit_exact_refine_progress(
                                "applied".to_string(),
                                best_runtime_lr,
                                best_effective_val,
                                best_effective_val,
                            );
                        }
                    } else {
                        stage_notes.push(
                            "PINOExactRefineSkipped: exact target builder returned no samples"
                                .to_string(),
                        );
                        emit_epoch_zero_progress(
                            &mut on_epoch,
                            "exact-refine",
                            "empty-targets",
                            &arch,
                            best_runtime_lr,
                            None,
                            best_effective_val.max(0.0),
                            best_effective_val.max(0.0),
                        );
                    }
                }
            } else if exact_refine_family_enabled && skip_exact_refine {
                stage_notes.push(
                    "PINOExactRefineSkipped: env requested low-latency completion path".to_string(),
                );
                emit_epoch_zero_progress(
                    &mut on_epoch,
                    "exact-refine",
                    "skipped",
                    &arch,
                    best_runtime_lr,
                    None,
                    best_effective_val.max(0.0),
                    best_effective_val.max(0.0),
                );
            }
            if best_effective_val.is_finite() && best_effective_val < f64::MAX {
                let fast_signoff_profile = matches!(
                    std::env::var("PINO_SIGNOFF_FAST_PROFILE").ok().as_deref(),
                    Some("1" | "true" | "TRUE" | "on" | "ON")
                );
                let initial_best_preserved =
                    Self::operator_param_delta_norm(&best_runtime_params, &initial_pino_params)
                        <= 1e-12;
                let final_param_delta =
                    Self::operator_param_delta_norm(&pino_params, &best_runtime_params);
                if fast_signoff_profile
                    && total_epochs <= 24
                    && initial_best_preserved
                    && final_param_delta > 1e-9
                {
                    stage_notes.push(format!(
                        "PINOFastSignoffKeepFinal: retained final operator state (delta={:.6e}) after no-improvement restore lock",
                        final_param_delta
                    ));
                } else {
                    arch = best_runtime_arch.clone();
                    pino_params = best_runtime_params.clone();
                    pino_calibration = best_runtime_calibration.clone();
                    pino_meta = best_runtime_meta.clone();
                }
            }
            let final_runtime_learning_rate =
                if best_effective_val.is_finite() && best_effective_val < f64::MAX {
                    best_runtime_lr
                } else {
                    stats.learning_rate
                };
            let reached_target = routed.target_loss.is_finite()
                && routed.target_loss > 0.0
                && best_effective_val <= routed.target_loss;
            let relative_improvement = if stats.initial_loss > 0.0 && stats.initial_loss.is_finite()
            {
                ((stats.initial_loss - stats.best_loss) / stats.initial_loss).max(0.0)
            } else {
                0.0
            };
            let convergence_margin = routed.target_loss.max(1e-6) * 1.25;
            let plateau_converged = forced_stop_reason.is_some()
                && no_improve_epochs >= 90
                && relative_improvement >= 0.10
                && best_effective_val <= convergence_margin;
            let mut notes = vec![
                format!(
                    "PINOConfig: backend={}, grid={}x{}, spectralModes={}, hiddenLayers={}, hiddenWidth={}, arch={:?}",
                    pino_model_config
                        .as_ref()
                        .map(|cfg| cfg.backend.as_str())
                        .unwrap_or(self.backend_runtime.tag()),
                    pino_model_config
                        .as_ref()
                        .map(|cfg| cfg.operator_grid.nx)
                        .unwrap_or(0),
                    pino_model_config
                        .as_ref()
                        .map(|cfg| cfg.operator_grid.ny)
                        .unwrap_or(0),
                    pino_model_config
                        .as_ref()
                        .map(|cfg| cfg.spectral_modes)
                        .unwrap_or(0),
                    pino_model_config
                        .as_ref()
                        .map(|cfg| cfg.hidden_layers)
                        .unwrap_or(0),
                    pino_model_config
                        .as_ref()
                        .map(|cfg| cfg.hidden_width)
                        .unwrap_or(0),
                    arch
                ),
                format!(
                    "PINOSpectralProbe: rms={:.6e}",
                    spectral_probe_rms.unwrap_or(0.0)
                ),
                format!(
                    "PINOCalibration: stressScale={:.6e}, dispScale={:.6e}",
                    pino_calibration
                        .as_ref()
                        .map(|v| v.stress_scale)
                        .unwrap_or(1.0),
                    pino_calibration
                        .as_ref()
                        .map(|v| v.displacement_scale)
                        .unwrap_or(1.0)
                ),
                format!(
                    "PINOOperatorTrain: epochs={}, initial={:.6e}, best={:.6e}, final={:.6e}",
                    pino_operator_training
                        .as_ref()
                        .map(|v| v.epochs_run)
                        .unwrap_or(0),
                    pino_operator_training
                        .as_ref()
                        .map(|v| v.initial_loss)
                        .unwrap_or(0.0),
                    pino_operator_training
                        .as_ref()
                        .map(|v| v.best_loss)
                        .unwrap_or(0.0),
                    pino_operator_training
                        .as_ref()
                        .map(|v| v.final_loss)
                        .unwrap_or(0.0)
                ),
                format!(
                    "PINORuntimeMode: backend={}, epochs={}, explicitSchedule={}",
                    self.backend_runtime.tag(),
                    total_epochs,
                    schedule_explicit
                ),
                format!(
                    "OperatorTargets: physics-informed cases={}, loss=momentum+kinematics+material+boundary",
                    operator_targets.len()
                ),
                format!(
                    "PINOTelemetryCadence: progressEvery={}, networkEvery={}",
                    progress_every, network_every
                ),
                format!(
                    "PINOOperatorStats: initial={:.6e}, best={:.6e}, final={:.6e}, lr={:.6e}, stoppedEarly={}",
                    stats.initial_loss, stats.best_loss, stats.final_loss, stats.learning_rate, stats.stopped_early
                ),
                format!(
                    "PINODiagnostics: phase={}, curriculumStage={}, optimizerId={}, checkpointRollbacks={}, targetFloorEstimate={:.6e}, trendSlope={:.6e}, trendVar={:.6e}",
                    if stats.stopped_early { "manual-stop" } else { "pino-runtime" },
                    if epochs_run <= s1e { 1 } else if epochs_run <= s1e + s2e { 2 } else { 3 },
                    last_optimizer_id,
                    rollback_count,
                    best_val,
                    trend_slope,
                    trend_variance
                ),
                format!(
                    "PINOResidualWeights: momentumWeight={:.3}, kinematicsWeight={:.3}, materialWeight={:.3}, boundaryWeight={:.3}",
                    adaptive_weights.0,
                    adaptive_weights.1,
                    adaptive_weights.2,
                    adaptive_weights.3
                ),
                format!(
                    "PINOResidualBreakdownTrain: disp={:.6e}, stress={:.6e}, obs={:.6e}, aux={:.6e}, inv={:.6e}, eq={:.6e}, cn={:.6e}, cs={:.6e}, we={:.6e}, bc={:.6e}",
                    last_train_breakdown.displacement_fit,
                    last_train_breakdown.stress_fit,
                    last_train_breakdown.observable,
                    last_train_breakdown.auxiliary_data,
                    last_train_breakdown.invariant,
                    last_train_breakdown.equilibrium,
                    last_train_breakdown.constitutive_normal,
                    last_train_breakdown.constitutive_shear,
                    last_train_breakdown.weak_energy,
                    last_train_breakdown.boundary
                ),
                format!(
                    "PINOResidualBreakdownVal: disp={:.6e}, stress={:.6e}, obs={:.6e}, aux={:.6e}, inv={:.6e}, eq={:.6e}, cn={:.6e}, cs={:.6e}, we={:.6e}, bc={:.6e}",
                    last_val_breakdown.displacement_fit,
                    last_val_breakdown.stress_fit,
                    last_val_breakdown.observable,
                    last_val_breakdown.auxiliary_data,
                    last_val_breakdown.invariant,
                    last_val_breakdown.equilibrium,
                    last_val_breakdown.constitutive_normal,
                    last_val_breakdown.constitutive_shear,
                    last_val_breakdown.weak_energy,
                    last_val_breakdown.boundary
                ),
                format!(
                    "PINOOptimizerSchedule: warmup=pino-adam, finetune=pino-lbfgs, switchEpoch={}",
                    lbfgs_start_epoch
                ),
                format!(
                    "PINOCheckpoints: total={}, bestMarked={}",
                    checkpoint_count, best_checkpoint_count
                ),
                format!(
                    "TargetOutcome: requested={:.6e}, bestVal={:.6e}, final={:.6e}, reached={}",
                    routed.target_loss,
                    best_effective_val,
                    stats.final_loss,
                    reached_target
                ),
            ];
            if !reached_target
                && routed.target_loss.is_finite()
                && routed.target_loss > 0.0
                && best_effective_val > routed.target_loss * 10.0
            {
                notes.push(format!(
                    "TargetFeasibility: requested target {:.6e} is below current validation floor {:.6e} (x{:.1}). Consider a target near {:.2e}–{:.2e} for this regime/config.",
                    routed.target_loss,
                    best_effective_val,
                    best_effective_val / routed.target_loss,
                    best_effective_val * 0.8,
                    best_effective_val * 1.1
                ));
            }
            notes.extend(stage_notes);
            let stop_reason = if forced_stop_reason.is_some() {
                    "plateau-stop".to_string()
                } else if reached_target_stop.get() || reached_target {
                    "target-loss-reached".to_string()
                } else if stats.stopped_early {
                    "manual-stop".to_string()
                } else {
                    "max-epochs-reached".to_string()
                };
            let benchmark_certification = benchmark_id.as_deref().and_then(|id| {
                certify_training_benchmark(
                    id,
                    best_effective_val,
                    benchmark_final_breakdown
                        .as_ref()
                        .map(|breakdown| BenchmarkCertificationInput {
                            displacement_fit: breakdown.displacement_fit,
                            stress_fit: breakdown.stress_fit,
                            observable: breakdown.observable,
                            equilibrium: breakdown.equilibrium,
                            constitutive_normal: breakdown.constitutive_normal,
                            constitutive_shear: breakdown.constitutive_shear,
                            weak_energy: breakdown.weak_energy,
                            boundary: breakdown.boundary,
                            tip_displacement_relative_error: benchmark_physical_errors
                                .as_ref()
                                .map(|errors| errors.tip_displacement_relative_error),
                            max_displacement_relative_error: benchmark_physical_errors
                                .as_ref()
                                .map(|errors| errors.max_displacement_relative_error),
                            mean_von_mises_relative_error: benchmark_physical_errors
                                .as_ref()
                                .map(|errors| errors.mean_von_mises_relative_error),
                            max_sigma_xx_relative_error: benchmark_physical_errors
                                .as_ref()
                                .map(|errors| errors.max_sigma_xx_relative_error),
                        }),
                )
            });
            if let Some(certification) = benchmark_certification.as_ref() {
                notes.push(format!(
                    "BenchmarkCertification: status={}, suggestedTargetLoss={:.6e}, summary={}",
                    certification.status,
                    certification.suggested_target_loss,
                    certification.summary.replace(',', ";"),
                ));
            }
            if let Some(note) = forced_stop_reason.clone() {
                notes.push(note);
            }
            if let Some(note) = runtime_note {
                notes.push(note);
            }
            self.set_burn_runtime_state(BurnRuntimeState {
                architecture: arch.clone(),
                learning_rate: final_runtime_learning_rate,
                model_version: final_model_version,
                last_loss: stats.final_loss,
                best_val_loss: best_val.min(stats.best_loss),
                train_samples: final_train_data_cap,
                completed_epochs: epochs_run,
                total_epochs,
                backend_tag: self.backend_runtime.tag().to_string(),
                last_train_seed: routed.seed,
                recent_notes: notes.iter().rev().take(8).cloned().collect(),
                pino: pino_meta.clone(),
                pino_calibration: pino_calibration.clone(),
                pino_params: Some(pino_params.clone()),
                pino_optimizer_state: pino_optimizer_state.clone(),
            });
            return TrainResult {
                model_version: final_model_version,
                loss: stats.final_loss,
                val_loss: best_effective_val,
                architecture: arch,
                learning_rate: final_runtime_learning_rate,
                grew: false,
                pruned: false,
                completed_epochs: epochs_run,
                reached_target,
                reached_target_loss: reached_target,
                reached_autonomous_convergence: reached_target || plateau_converged,
                stop_reason,
                notes,
                training_mode: Some(training_mode),
                benchmark_id,
                gate_status: Some(if reached_target {
                    "passed".to_string()
                } else if forced_stop_reason.is_some() {
                    "stalled".to_string()
                } else if stats.stopped_early {
                    "stopped".to_string()
                } else {
                    "failed".to_string()
                }),
                certified_best_metric: Some(best_effective_val),
                reproducibility_spread: None,
                dominant_blocker: benchmark_dominant_blocker,
                stalled_reason: forced_stop_reason.clone(),
                benchmark_certification,
                pino: pino_meta,
            };
        }

        self.clear_burn_runtime_state();
        emit_epoch_zero_progress(
            &mut on_epoch,
            "preflight",
            "preparing-ann-runtime",
            &self.inner.status().architecture,
            routed.learning_rate.unwrap_or(5e-4),
            pino_meta.clone(),
            0.0,
            0.0,
        );
        let mut result = self.inner.train_with_progress_with_checkpoint(
            &routed,
            |mut p| {
                if p.hybrid_mode.is_empty() {
                    p.hybrid_mode = "hybrid".to_string();
                }
                p.hybrid_mode = format!("{}+{}", self.backend_runtime.tag(), p.hybrid_mode);
                on_epoch(p);
            },
            should_stop,
            |epoch, snapshot, is_best| {
                on_checkpoint(
                    epoch,
                    UniversalPinnState {
                        ann: snapshot,
                        burn: None,
                        last_config: self.last_config.clone(),
                        backend_runtime: self.backend_runtime.tag().to_string(),
                    },
                    is_best,
                );
            },
        );
        if let Some(note) = runtime_note {
            result.notes.push(note);
        }
        result.pino = pino_meta;
        result
    }

    pub fn infer(&self, input: &crate::contracts::SolveInput) -> AnnResult {
        if self.backend_runtime != PinnBackendRuntime::CompatAnn {
            let status = self.status();
            let spectral_modes = self
                .burn_state
                .as_ref()
                .and_then(|state| state.pino.as_ref().map(|meta| meta.spectral_modes))
                .unwrap_or(4);
            let mut config =
                infer_config_for_case(input, self.backend_runtime.tag(), spectral_modes);
            if let Some(arch) = self
                .burn_state
                .as_ref()
                .map(|state| state.architecture.as_slice())
            {
                if arch.len() >= 3 {
                    let hidden_layers = arch.len().saturating_sub(2).clamp(1, 6);
                    let hidden_width = arch[1..arch.len() - 1]
                        .iter()
                        .copied()
                        .max()
                        .unwrap_or(config.hidden_width)
                        .clamp(8, 256);
                    config.hidden_layers = hidden_layers;
                    config.hidden_width = hidden_width;
                }
            }
            let mut prediction = build_operator_prediction_with_params(
                input,
                &config,
                self.burn_state
                    .as_ref()
                    .and_then(|state| state.pino_params.as_ref()),
            );
            if let Some(calibration) = self
                .burn_state
                .as_ref()
                .and_then(|state| state.pino_calibration.as_ref())
            {
                apply_operator_calibration(&mut prediction, calibration);
            }
            let residual = operator_residual_score(input, &prediction);
            let mut fem_like = decode_prediction(input, &prediction);
            let holdout = self
                .burn_state
                .as_ref()
                .and_then(|state| state.pino.as_ref())
                .and_then(|meta| meta.holdout_validation.clone());
            let base_uncertainty_thr = status.safeguard_settings.uncertainty_threshold;
            let base_residual_thr = status.safeguard_settings.residual_threshold;
            let mut uncertainty_thr = base_uncertainty_thr;
            let mut residual_thr = base_residual_thr;
            let mut holdout_uncertainty = 0.45;
            let mut holdout_residual_ratio = 1.0;
            let mut holdout_gate_passed = false;
            if let Some(summary) = holdout.as_ref() {
                let disp_ratio = (summary.mean_displacement_error / 0.05).clamp(0.0, 4.0);
                let vm_ratio = (summary.mean_von_mises_error / 0.05).clamp(0.0, 4.0);
                let p95_ratio = (summary.p95_field_error / 0.10).clamp(0.0, 4.0);
                holdout_residual_ratio = summary.residual_ratio.clamp(0.0, 4.0);
                holdout_gate_passed = summary.trusted && summary.accepted_without_fallback;
                let holdout_quality = (0.45 * disp_ratio
                    + 0.20 * vm_ratio
                    + 0.25 * p95_ratio
                    + 0.10 * holdout_residual_ratio)
                    .clamp(0.0, 4.0);
                holdout_uncertainty = (holdout_quality / 4.0).clamp(0.0, 1.0);
                if summary.trusted {
                    uncertainty_thr = (base_uncertainty_thr * 1.15).clamp(0.02, 0.99);
                    let residual_relax =
                        (1.20 + (1.0 - holdout_quality).clamp(-0.25, 0.55)).clamp(0.80, 1.75);
                    residual_thr = (base_residual_thr * residual_relax).clamp(1e-6, 10.0);
                    holdout_uncertainty = (holdout_uncertainty * 0.85).clamp(0.0, 1.0);
                } else {
                    uncertainty_thr = (base_uncertainty_thr * 0.90).clamp(0.02, 0.99);
                    let residual_tighten =
                        (0.95 - (holdout_quality - 1.0).max(0.0) * 0.12).clamp(0.55, 1.0);
                    residual_thr = (base_residual_thr * residual_tighten).clamp(1e-6, 10.0);
                    holdout_uncertainty = (holdout_uncertainty * 1.10).clamp(0.0, 1.0);
                }
            }
            let residual_ratio = (residual / residual_thr.max(1e-9)).max(0.0);
            let residual_uncertainty = ((residual_ratio - 1.0).max(0.0) * 0.65).clamp(0.0, 1.0);
            let mut uncertainty =
                (0.75 * holdout_uncertainty + 0.25 * residual_uncertainty).clamp(0.0, 1.0);
            if !holdout_gate_passed {
                uncertainty = uncertainty.max(0.95);
            }
            let confidence = (1.0 - uncertainty).clamp(0.0, 1.0);
            let mut used_fallback = false;
            let mut fallback_reason = None;
            let mut diagnostics = vec![
                format!(
                    "PINO operator inference with backend {} on {}x{} grid, {} spectral modes, hiddenLayers={}, hiddenWidth={}",
                    config.backend,
                    config.operator_grid.nx,
                    config.operator_grid.ny,
                    config.spectral_modes,
                    config.hidden_layers,
                    config.hidden_width
                ),
                format!(
                    "PINO safeguards: uncertainty={:.4}, residual={:.4}, thresholds(calibrated)=({:.4},{:.4}), thresholds(base)=({:.4},{:.4})",
                    uncertainty, residual, uncertainty_thr, residual_thr, base_uncertainty_thr, base_residual_thr
                ),
                format!(
                    "PINO safeguard context: holdoutUncertainty={:.4}, residualRatio={:.4}, holdoutResidualRatio={:.4}",
                    holdout_uncertainty, residual_ratio, holdout_residual_ratio
                ),
                format!(
                    "PINO holdout trust gate: {}",
                    if holdout_gate_passed {
                        "pass"
                    } else {
                        "fail"
                    }
                ),
            ];
            if let Some(summary) = &holdout {
                diagnostics.push(format!(
                    "Holdout validation: trusted={}, acceptedWithoutFallback={}, meanDisp={:.4} (<= {:.4} {}) , meanVM={:.4} (<= {:.4} {}), p95={:.4} (<= {:.4} {}), ratio={:.4} (<= {:.4} {})",
                    summary.trusted,
                    summary.accepted_without_fallback,
                    summary.mean_displacement_error,
                    summary.mean_error_limit,
                    if summary.displacement_pass { "pass" } else { "fail" },
                    summary.mean_von_mises_error,
                    summary.mean_error_limit,
                    if summary.von_mises_pass { "pass" } else { "fail" },
                    summary.p95_field_error
                    ,
                    summary.p95_error_limit,
                    if summary.p95_pass { "pass" } else { "fail" },
                    summary.residual_ratio,
                    summary.residual_ratio_limit,
                    if summary.residual_ratio_pass { "pass" } else { "fail" }
                ));
            } else {
                diagnostics.push(
                    "Holdout validation: missing summary; trust gate forced to fail.".to_string(),
                );
            }
            if let Some(calibration) = self
                .burn_state
                .as_ref()
                .and_then(|state| state.pino_calibration.as_ref())
            {
                diagnostics.push(format!(
                    "PINO calibration applied: stressScale={:.4}, dispScale={:.4}",
                    calibration.stress_scale, calibration.displacement_scale
                ));
            }
            let safeguard_reject = uncertainty > uncertainty_thr || residual > residual_thr;
            let holdout_reject = !holdout_gate_passed;
            if status.fallback_enabled && (holdout_reject || safeguard_reject) {
                fem_like = crate::fem::solve_case(input);
                used_fallback = true;
                let safeguard_reason = match (uncertainty > uncertainty_thr, residual > residual_thr)
                {
                    (true, true) => Some(format!(
                        "uncertainty and residual safeguards failed ({uncertainty:.4} > {uncertainty_thr:.4}, {residual:.4} > {residual_thr:.4})"
                    )),
                    (true, false) => Some(format!(
                        "uncertainty safeguard failed ({uncertainty:.4} > {uncertainty_thr:.4})"
                    )),
                    (false, true) => Some(format!(
                        "residual safeguard failed ({residual:.4} > {residual_thr:.4})"
                    )),
                    (false, false) => None,
                };
                let reason = match (holdout_reject, safeguard_reason) {
                    (true, Some(safe_reason)) => {
                        format!(
                            "PINO surrogate rejected by holdout trust gate (criteria failed) and {safe_reason}"
                        )
                    }
                    (true, None) => {
                        "PINO surrogate rejected by holdout trust gate (criteria failed; model not trusted for direct surrogate inference in this regime).".to_string()
                    }
                    (false, Some(safe_reason)) => {
                        format!("PINO surrogate rejected by {safe_reason}")
                    }
                    (false, None) => "PINO surrogate rejected by safeguard policy.".to_string(),
                };
                diagnostics.push(format!("{reason}; FEM fallback used."));
                fallback_reason = Some(reason);
            } else if !status.fallback_enabled && (holdout_reject || safeguard_reject) {
                diagnostics.push(
                    "PINO rejection detected but FEM fallback is disabled; returning surrogate result per runtime setting."
                        .to_string(),
                );
            }
            let domain_extrapolation = 0.0;
            return AnnResult {
                fem_like,
                confidence,
                uncertainty,
                model_version: status.model_version,
                used_fem_fallback: used_fallback,
                fallback_reason,
                domain_extrapolation_score: domain_extrapolation,
                residual_score: residual,
                uncertainty_threshold: uncertainty_thr,
                residual_threshold: residual_thr,
                diagnostics,
                surrogate_domain: status.surrogate_domain.clone(),
                pino: self
                    .burn_state
                    .as_ref()
                    .and_then(|state| state.pino.clone())
                    .or_else(|| {
                        Some(runtime_metadata(
                            &TrainingBatch {
                                cases: vec![input.clone()],
                                epochs: 1,
                                target_loss: 0.0,
                                training_mode: None,
                                benchmark_id: None,
                                seed: None,
                                analysis_type: None,
                                pinn_backend: Some(config.backend.clone()),
                                collocation_points: None,
                                boundary_points: None,
                                interface_points: None,
                                residual_weight_momentum: None,
                                residual_weight_kinematics: None,
                                residual_weight_material: None,
                                residual_weight_boundary: None,
                                stage1_epochs: None,
                                stage2_epochs: None,
                                stage3_ramp_epochs: None,
                                contact_penalty: None,
                                plasticity_factor: None,
                                learning_rate: None,
                                auto_mode: None,
                                max_total_epochs: None,
                                min_improvement: None,
                                progress_emit_every_epochs: None,
                                network_emit_every_epochs: None,
                                online_active_learning: None,
                                autonomous_mode: None,
                                max_topology: None,
                                max_backoffs: None,
                                max_optimizer_switches: None,
                                checkpoint_every_epochs: None,
                                checkpoint_retention: None,
                            },
                            &config.backend,
                            config.spectral_modes,
                        ))
                    }),
            };
        }
        let mut result = self.inner.infer(input);
        result.pino = self
            .burn_state
            .as_ref()
            .and_then(|state| state.pino.clone());
        result
    }

    pub fn status(&self) -> ModelStatus {
        if let Some(burn) = &self.burn_state {
            let inner_status = self.inner.status();
            return ModelStatus {
                model_version: burn.model_version,
                architecture: burn.architecture.clone(),
                learning_rate: burn.learning_rate,
                last_loss: burn.last_loss,
                train_samples: burn.train_samples,
                audit_frequency: inner_status.audit_frequency,
                fallback_enabled: inner_status.fallback_enabled,
                safeguard_settings: inner_status.safeguard_settings,
                surrogate_domain: self.inner.surrogate_domain(),
                pino: burn.pino.clone(),
            };
        }
        self.inner.status()
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        self.inner.reset(seed);
        self.clear_burn_runtime_state();
    }

    pub fn snapshot_state(&self) -> UniversalPinnState {
        UniversalPinnState {
            ann: self.inner.snapshot_state(),
            burn: self.burn_state.clone(),
            last_config: self.last_config.clone(),
            backend_runtime: self.backend_runtime.tag().to_string(),
        }
    }

    pub fn load_state(&mut self, state: UniversalPinnState) {
        self.inner.load_state(state.ann);
        self.last_config = state.last_config;
        self.backend_runtime = PinnBackendRuntime::from_name(&state.backend_runtime);
        self.burn_state = state.burn;
        self.sync_inner_metadata_from_burn_state();
    }

    pub fn set_safeguard_settings(&mut self, settings: SafeguardSettings) {
        self.inner.set_safeguard_settings(settings);
    }

    #[allow(dead_code)]
    pub fn last_config(&self) -> &UniversalPinnConfig {
        &self.last_config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{LoadInput, OperatorGridSpec, SolveInput};
    use std::time::Instant;

    fn focused_cantilever_case() -> SolveInput {
        let mut case = SolveInput::default();
        case.geometry.length_in = 10.0;
        case.geometry.width_in = 1.0;
        case.geometry.thickness_in = 0.25;
        case.geometry.hole_diameter_in = Some(0.0);
        case.load = LoadInput {
            axial_load_lbf: 0.0,
            vertical_point_load_lbf: -100.0,
        };
        case
    }

    fn focused_plate_hole_case() -> SolveInput {
        let mut case = SolveInput::default();
        case.geometry.length_in = 8.0;
        case.geometry.width_in = 4.0;
        case.geometry.thickness_in = 0.25;
        case.geometry.hole_diameter_in = Some(1.0);
        case.load = LoadInput {
            axial_load_lbf: 1_000.0,
            vertical_point_load_lbf: 0.0,
        };
        case
    }

    fn benchmark_prediction(von_mises: Vec<f64>) -> OperatorPrediction {
        let len = von_mises.len();
        OperatorPrediction {
            grid: OperatorGridSpec {
                nx: len.max(1),
                ny: 1,
                nz: 1,
                input_channels: 15,
                output_channels: 11,
            },
            ux: vec![0.0; len],
            uy: vec![0.0; len],
            uz: vec![0.0; len],
            sxx: vec![0.0; len],
            syy: vec![0.0; len],
            szz: vec![0.0; len],
            sxy: vec![0.0; len],
            sxz: vec![0.0; len],
            syz: vec![0.0; len],
            von_mises,
            max_principal: vec![0.0; len],
        }
    }

    #[test]
    fn benchmark_eval_and_curriculum_grids_preserve_single_layer_cases() {
        let mut case = focused_cantilever_case();
        case.mesh.nz = 1;
        let batch = crate::contracts::TrainingBatch {
            cases: vec![case],
            epochs: 64,
            target_loss: 1e-4,
            training_mode: Some("benchmark".to_string()),
            benchmark_id: Some("benchmark_cantilever_2d".to_string()),
            seed: Some(1),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
            collocation_points: Some(256),
            boundary_points: Some(64),
            interface_points: Some(32),
            residual_weight_momentum: None,
            residual_weight_kinematics: None,
            residual_weight_material: None,
            residual_weight_boundary: None,
            stage1_epochs: None,
            stage2_epochs: None,
            stage3_ramp_epochs: None,
            contact_penalty: None,
            plasticity_factor: None,
            learning_rate: None,
            auto_mode: None,
            max_total_epochs: None,
            min_improvement: None,
            progress_emit_every_epochs: None,
            network_emit_every_epochs: None,
            online_active_learning: None,
            autonomous_mode: None,
            max_topology: None,
            max_backoffs: None,
            max_optimizer_switches: None,
            checkpoint_every_epochs: None,
            checkpoint_retention: None,
        };
        let cfg = crate::pino::model_config(&batch);
        assert_eq!(cfg.operator_grid.nz, 1);
        assert_eq!(UniversalPinnEngine::epoch_eval_grid(&cfg).nz, 1);
        assert_eq!(
            UniversalPinnEngine::curriculum_operator_grid(&cfg, 2, false).nz,
            1
        );
    }

    #[test]
    fn cantilever_vm_error_ignores_low_stress_cells() {
        let mut target = benchmark_prediction(vec![0.1, 0.2, 80.0, 100.0]);
        let mut prediction = benchmark_prediction(vec![5.0, 7.0, 84.0, 108.0]);
        target.uy = vec![0.0; 4];
        prediction.uy = vec![0.0; 4];
        target.sxx = vec![0.0, 0.0, 70.0, 100.0];
        prediction.sxx = vec![0.0, 0.0, 72.0, 108.0];

        let errors = evaluate_cantilever_benchmark_physical_errors(&prediction, &target);
        assert!(
            errors.mean_von_mises_relative_error < 0.2,
            "expected active-region vm error, got {}",
            errors.mean_von_mises_relative_error
        );
    }

    #[test]
    fn exact_refine_family_cases_expand_cantilever_but_stay_capped() {
        let base = focused_cantilever_case();
        let expanded = UniversalPinnEngine::build_exact_refine_family_cases(
            &[base.clone()],
            "cantilever",
            true,
        );
        assert!(
            expanded.iter().any(|case| {
                (case.load.vertical_point_load_lbf - base.load.vertical_point_load_lbf).abs()
                    <= 1e-9
                    && (case.geometry.length_in - base.geometry.length_in).abs() <= 1e-9
            }),
            "expected expanded exact family to retain the anchor cantilever case"
        );
        assert!(
            expanded.len() > 1,
            "expected focused exact family expansion"
        );
        assert!(
            expanded.len() <= 8,
            "expected fast-profile cap, got {}",
            expanded.len()
        );
    }

    #[test]
    fn exact_refine_family_cases_expand_plate_hole_but_stay_capped() {
        let base = focused_plate_hole_case();
        let expanded = UniversalPinnEngine::build_exact_refine_family_cases(
            &[base.clone()],
            "plate-hole",
            true,
        );
        assert!(
            expanded
                .iter()
                .any(|case| case.geometry.hole_diameter_in == base.geometry.hole_diameter_in),
            "expected expanded exact family to retain the anchor plate-hole case"
        );
        assert!(expanded.len() > 1, "expected plate-hole family expansion");
        assert!(
            expanded.len() <= 10,
            "expected fast-profile cap, got {}",
            expanded.len()
        );
    }

    #[test]
    fn exact_target_builder_uses_exact_projection_for_plate_hole_case() {
        let base = focused_plate_hole_case();
        let cfg = infer_config_for_case(&base, "pino-ndarray-cpu", 4);
        let mut exact = base.clone();
        exact.mesh.nx = exact.mesh.nx.max(cfg.operator_grid.nx).clamp(10, 48);
        exact.mesh.ny = exact.mesh.ny.max(cfg.operator_grid.ny).clamp(8, 32);
        exact.mesh.nz = exact.mesh.nz.max(cfg.operator_grid.nz).clamp(2, 12);
        exact.mesh.auto_adapt = false;
        exact.mesh.amr_enabled = false;
        exact.mesh.amr_passes = 0;
        let fem = crate::fem::solve_case(&exact);
        let expected = prediction_from_fem_with_grid_exact(&base, &fem, cfg.operator_grid.clone());
        let built = UniversalPinnEngine::build_operator_field_targets_exact(
            &[base],
            cfg.operator_grid.clone(),
        );
        assert_eq!(built.len(), 1);
        assert_eq!(built[0].target.von_mises, expected.von_mises);
        assert_eq!(built[0].target.max_principal, expected.max_principal);
    }

    #[test]
    fn burn_sample_core_uses_prebuilt_target_projection() {
        let batch = crate::benchmarks::apply_training_benchmark(TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 8,
            target_loss: 1e-4,
            training_mode: Some("benchmark".to_string()),
            benchmark_id: Some("benchmark_cantilever_2d".to_string()),
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(24),
            min_improvement: Some(1e-6),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(50),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(48),
            max_backoffs: Some(1),
            max_optimizer_switches: Some(4),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(42),
            analysis_type: None,
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
            collocation_points: Some(64),
            boundary_points: Some(24),
            interface_points: Some(12),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(4),
            stage2_epochs: Some(8),
            stage3_ramp_epochs: Some(12),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        })
        .expect("benchmark batch");
        let cfg = model_config(&batch);
        let targets = UniversalPinnEngine::build_operator_field_targets_exact(
            &batch.cases,
            cfg.operator_grid.clone(),
        );
        assert_eq!(targets.len(), 1);
        let params = OperatorTrainableParams::for_config(&cfg);
        let core =
            UniversalPinnEngine::build_burn_physics_sample_core(&targets[0], &cfg, &params);
        assert_eq!(core.target.von_mises, targets[0].target.von_mises);
        assert_eq!(core.target.max_principal, targets[0].target.max_principal);
        assert_eq!(core.target.sxx, targets[0].target.sxx);
    }

    #[test]
    fn benchmark_cantilever_exact_target_is_self_consistent_under_displacement_reconstruction() {
        let raw_batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 8,
            target_loss: 1e-4,
            training_mode: Some("benchmark".to_string()),
            benchmark_id: Some("benchmark_cantilever_2d".to_string()),
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(24),
            min_improvement: Some(1e-6),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(50),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(48),
            max_backoffs: Some(1),
            max_optimizer_switches: Some(4),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(42),
            analysis_type: None,
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
            collocation_points: Some(64),
            boundary_points: Some(24),
            interface_points: Some(12),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(4),
            stage2_epochs: Some(8),
            stage3_ramp_epochs: Some(12),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };
        let batch = crate::benchmarks::apply_training_benchmark(raw_batch)
            .expect("benchmark normalization");
        let case = batch.cases[0].clone();
        let cfg = model_config(&batch);
        let fem = crate::fem::solve_case(&case);
        let target = prediction_from_fem_with_grid_exact(&case, &fem, cfg.operator_grid.clone());
        let rebuilt = reconstruct_prediction_linear_elastic_from_displacement(&case, &target);
        let errors = evaluate_cantilever_benchmark_physical_errors(&rebuilt, &target);
        assert!(
            errors.tip_displacement_relative_error <= 1e-9,
            "tip displacement should be unchanged by reconstruction, got {:?}",
            errors
        );
        assert!(
            errors.max_displacement_relative_error <= 1e-9,
            "displacement magnitude should be unchanged by reconstruction, got {:?}",
            errors
        );
        assert!(
            errors.mean_von_mises_relative_error <= 0.20,
            "benchmark target stress field is not self-consistent enough for displacement-primary training, got {:?}",
            errors
        );
        assert!(
            errors.max_sigma_xx_relative_error <= 0.20,
            "benchmark target sigma_xx is not self-consistent enough for displacement-primary training, got {:?}",
            errors
        );
    }

    #[test]
    fn benchmark_cantilever_startup_epoch_eval_survives_stage_grid_mismatch() {
        let batch = crate::benchmarks::apply_training_benchmark(TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 40,
            target_loss: 1e-4,
            training_mode: Some("benchmark".to_string()),
            benchmark_id: Some("benchmark_cantilever_2d".to_string()),
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(180),
            min_improvement: Some(1e-6),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(12),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(48),
            max_backoffs: Some(2),
            max_optimizer_switches: Some(8),
            checkpoint_every_epochs: Some(1000),
            checkpoint_retention: Some(8),
            seed: Some(42),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
            collocation_points: Some(256),
            boundary_points: Some(64),
            interface_points: Some(32),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(12),
            stage2_epochs: Some(36),
            stage3_ramp_epochs: Some(24),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        })
        .expect("benchmark batch");
        let cfg = model_config(&batch);
        let mut stage_cfg = cfg.clone();
        stage_cfg.operator_grid = UniversalPinnEngine::curriculum_operator_grid(&cfg, 1, true);
        let targets = UniversalPinnEngine::build_operator_field_targets_startup(
            &batch.cases,
            stage_cfg.operator_grid.clone(),
        );
        let params = OperatorTrainableParams::for_config(&cfg);
        let calibration = fit_operator_calibration_with_params(&batch, &cfg, Some(&params))
            .unwrap_or(OperatorCalibration {
                stress_scale: 1.0,
                displacement_scale: 1.0,
            });
        let outcome = UniversalPinnEngine::evaluate_operator_epoch(
            &targets,
            &cfg,
            &calibration,
            &params,
            (1.0, 1.0, 1.0, 1.0),
        );
        assert!(
            outcome.is_some(),
            "startup benchmark epoch evaluation should survive stage-grid mismatch"
        );
    }

    #[test]
    fn training_progress_hybrid_mode_contains_backend_prefix() {
        let mut engine = UniversalPinnEngine::default();
        let mut case = SolveInput::default();
        case.load = LoadInput {
            axial_load_lbf: 0.0,
            vertical_point_load_lbf: -10.0,
        };

        let batch = TrainingBatch {
            cases: vec![case],
            epochs: 2,
            target_loss: 0.5,
            training_mode: None,
            benchmark_id: None,
            learning_rate: Some(1e-3),
            auto_mode: Some(true),
            max_total_epochs: Some(10),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(100),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(32),
            max_backoffs: Some(2),
            max_optimizer_switches: Some(2),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(7),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("burn-ndarray-cpu".to_string()),
            collocation_points: Some(256),
            boundary_points: Some(64),
            interface_points: Some(64),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(2),
            stage2_epochs: Some(2),
            stage3_ramp_epochs: Some(2),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };

        let mut seen = String::new();
        let _ = engine.train_with_progress_with_checkpoint(
            &batch,
            |p| {
                if seen.is_empty() {
                    seen = p.hybrid_mode;
                }
            },
            || false,
            |_e, _s, _b| {},
        );
        assert!(
            seen.starts_with("pino-ndarray-cpu+"),
            "expected backend-prefixed hybrid mode, got {seen}"
        );
    }

    #[test]
    fn unknown_backend_defaults_to_pino_runtime_without_rollback_flag() {
        let backend = PinnBackendRuntime::from_name("legacy-ann");
        assert_eq!(backend, PinnBackendRuntime::PinoNdArrayCpu);
    }

    #[test]
    fn unknown_backend_can_route_to_compat_with_explicit_rollback_flag() {
        let prior = std::env::var("PINO_ENABLE_LEGACY_ROLLBACK").ok();
        // SAFETY: tests run in a single process; we restore the variable before returning.
        unsafe {
            std::env::set_var("PINO_ENABLE_LEGACY_ROLLBACK", "1");
        }
        let backend = PinnBackendRuntime::from_name("legacy-ann");
        if let Some(prev) = prior {
            // SAFETY: restore original environment variable value.
            unsafe {
                std::env::set_var("PINO_ENABLE_LEGACY_ROLLBACK", prev);
            }
        } else {
            // SAFETY: restore by removing variable.
            unsafe {
                std::env::remove_var("PINO_ENABLE_LEGACY_ROLLBACK");
            }
        }
        assert_eq!(backend, PinnBackendRuntime::CompatAnn);
    }

    #[test]
    fn pino_snapshot_preserves_output_inbound_links_under_cap() {
        let arch = vec![24, 24, 24];
        let snapshot = UniversalPinnEngine::burn_network_snapshot(
            &arch,
            173,
            1e-3,
            crate::pinn_burn::ResidualPillars {
                momentum: 0.2,
                kinematics: 0.2,
                material: 0.2,
                boundary: 0.2,
            },
            true,
        );
        assert_eq!(snapshot.layer_sizes, arch);
        assert!(snapshot.connections.len() <= 900);
        for output_idx in 0..arch[2] {
            let output_id = format!("L2N{output_idx}");
            assert!(
                snapshot
                    .connections
                    .iter()
                    .any(|edge| edge.to_id == output_id),
                "expected inbound connection for output node {output_id}"
            );
        }
    }

    #[test]
    fn pino_snapshot_preserves_terminal_connectivity_for_deep_architecture() {
        let arch = vec![10, 16, 16, 16, 16, 7];
        let snapshot = UniversalPinnEngine::burn_network_snapshot(
            &arch,
            211,
            9e-4,
            crate::pinn_burn::ResidualPillars {
                momentum: 0.16,
                kinematics: 0.14,
                material: 0.12,
                boundary: 0.10,
            },
            true,
        );
        assert!(snapshot.connections.len() <= 900);
        let output_layer = arch.len() - 1;
        let prev_layer = arch.len() - 2;
        for output_idx in 0..arch[output_layer] {
            let output_id = format!("L{output_layer}N{output_idx}");
            assert!(
                snapshot
                    .connections
                    .iter()
                    .any(|edge| edge.to_id == output_id),
                "expected inbound connection for terminal node {output_id}"
            );
        }
        for hidden_idx in 0..arch[prev_layer] {
            let hidden_id = format!("L{prev_layer}N{hidden_idx}");
            assert!(
                snapshot.connections.iter().any(|edge| {
                    edge.from_id == hidden_id
                        && edge.to_id.starts_with(&format!("L{output_layer}N"))
                }),
                "expected outbound connection from final hidden node {hidden_id}"
            );
        }
    }

    #[test]
    fn pino_backend_adds_recipe_note() {
        let mut engine = UniversalPinnEngine::default();
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 2,
            target_loss: 0.5,
            training_mode: None,
            benchmark_id: None,
            learning_rate: Some(1e-3),
            auto_mode: Some(true),
            max_total_epochs: Some(6),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(50),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(32),
            max_backoffs: Some(2),
            max_optimizer_switches: Some(2),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(7),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("burn-ndarray-cpu".to_string()),
            collocation_points: Some(256),
            boundary_points: Some(64),
            interface_points: Some(64),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(2),
            stage2_epochs: Some(2),
            stage3_ramp_epochs: Some(2),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };

        let result =
            engine.train_with_progress_with_checkpoint(&batch, |_p| {}, || false, |_e, _s, _b| {});
        assert!(
            result
                .notes
                .iter()
                .any(|n| n.contains("PINOFoundation: backend=pino-ndarray-cpu")),
            "expected PINO foundation note in result"
        );
    }

    #[test]
    fn pino_runtime_emits_preflight_before_epoch_progress() {
        let mut engine = UniversalPinnEngine::default();
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 2,
            target_loss: 0.5,
            training_mode: None,
            benchmark_id: None,
            learning_rate: Some(1e-3),
            auto_mode: Some(true),
            max_total_epochs: Some(6),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(50),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(32),
            max_backoffs: Some(2),
            max_optimizer_switches: Some(2),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(7),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("burn-ndarray-cpu".to_string()),
            collocation_points: Some(256),
            boundary_points: Some(64),
            interface_points: Some(64),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(2),
            stage2_epochs: Some(2),
            stage3_ramp_epochs: Some(2),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };

        let mut events = Vec::new();
        let saw_epoch_progress = Cell::new(false);
        let result = engine.train_with_progress_with_checkpoint(
            &batch,
            |p| {
                if events.len() < 12 {
                    events.push((p.epoch, p.stage_id, p.lr_phase));
                }
                if p.epoch > 0 {
                    saw_epoch_progress.set(true);
                }
            },
            || saw_epoch_progress.get(),
            |_e, _s, _b| {},
        );

        assert!(
            events
                .iter()
                .any(|(epoch, stage, _)| *epoch == 0 && stage == "preflight"),
            "expected at least one preflight event before training epochs, got {events:?}"
        );
        assert!(
            events.iter().any(|(epoch, _, _)| *epoch > 0),
            "expected runtime to advance beyond preflight, got {events:?}"
        );
        assert!(
            result
                .notes
                .iter()
                .any(|n| n.contains("PINOFastStart: deferred startup calibration and holdout validation until runtime epochs")),
            "expected fast-start note in result"
        );
    }

    #[test]
    fn benchmark_cantilever_routes_into_isolated_exact_lane() {
        let prior_rounds = std::env::var("PINO_HEADLESS_EXACT_REFINE_ROUNDS").ok();
        let prior_step_cap = std::env::var("PINO_HEADLESS_EXACT_REFINE_STEP_CAP").ok();
        let prior_warm_steps = std::env::var("PINO_HEADLESS_ISOLATED_EXACT_WARM_STEPS").ok();
        // SAFETY: test-scoped env override; restored before returning.
        unsafe {
            std::env::set_var("PINO_HEADLESS_EXACT_REFINE_ROUNDS", "1");
            std::env::set_var("PINO_HEADLESS_EXACT_REFINE_STEP_CAP", "8");
            std::env::set_var("PINO_HEADLESS_ISOLATED_EXACT_WARM_STEPS", "8");
        }
        let mut engine = UniversalPinnEngine::default();
        let raw_batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 8,
            target_loss: 0.0,
            training_mode: Some("benchmark".to_string()),
            benchmark_id: Some("benchmark_cantilever_2d".to_string()),
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(24),
            min_improvement: Some(1e-6),
            progress_emit_every_epochs: Some(2),
            network_emit_every_epochs: Some(50),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(48),
            max_backoffs: Some(1),
            max_optimizer_switches: Some(4),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(42),
            analysis_type: None,
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
            collocation_points: Some(64),
            boundary_points: Some(24),
            interface_points: Some(12),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(4),
            stage2_epochs: Some(8),
            stage3_ramp_epochs: Some(12),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };
        let batch = crate::benchmarks::apply_training_benchmark(raw_batch)
            .expect("benchmark normalization");

        let result =
            engine.train_with_progress_with_checkpoint(&batch, |_p| {}, || false, |_e, _s, _b| {});

        assert_eq!(result.training_mode.as_deref(), Some("benchmark"));
        assert_eq!(
            result.benchmark_id.as_deref(),
            Some("benchmark_cantilever_2d")
        );
        assert!(
            result
                .notes
                .iter()
                .any(|n| n.contains("isolatedExactCantilever=on")),
            "expected isolated exact cantilever lane note in result"
        );
        assert!(
            result
                .notes
                .iter()
                .any(|n| n.contains("PINOExactRefineBaseline")),
            "expected benchmark cantilever route to enter exact refine"
        );
        assert!(
            result
                .notes
                .iter()
                .any(|n| n.contains("PINOExactRefineIsolatedCantilever: evalCases=1, trainCases=1")),
            "expected benchmark cantilever route to keep exact refine on the single anchor case, got {:?}",
            result.notes
        );
        assert!(
            !result
                .notes
                .iter()
                .any(|n| n.contains("PINOExactRefineFamily:")),
            "expected benchmark cantilever route to avoid family expansion, got {:?}",
            result.notes
        );
        if let Some(prev) = prior_rounds {
            // SAFETY: restore original environment variable value.
            unsafe {
                std::env::set_var("PINO_HEADLESS_EXACT_REFINE_ROUNDS", prev);
            }
        } else {
            // SAFETY: restore by removing variable.
            unsafe {
                std::env::remove_var("PINO_HEADLESS_EXACT_REFINE_ROUNDS");
            }
        }
        if let Some(prev) = prior_step_cap {
            // SAFETY: restore original environment variable value.
            unsafe {
                std::env::set_var("PINO_HEADLESS_EXACT_REFINE_STEP_CAP", prev);
            }
        } else {
            // SAFETY: restore by removing variable.
            unsafe {
                std::env::remove_var("PINO_HEADLESS_EXACT_REFINE_STEP_CAP");
            }
        }
        if let Some(prev) = prior_warm_steps {
            // SAFETY: restore original environment variable value.
            unsafe {
                std::env::set_var("PINO_HEADLESS_ISOLATED_EXACT_WARM_STEPS", prev);
            }
        } else {
            // SAFETY: restore by removing variable.
            unsafe {
                std::env::remove_var("PINO_HEADLESS_ISOLATED_EXACT_WARM_STEPS");
            }
        }
    }

    #[test]
    fn benchmark_cantilever_progress_uses_physical_metric_consistently() {
        let prior_rounds = std::env::var("PINO_HEADLESS_EXACT_REFINE_ROUNDS").ok();
        let prior_step_cap = std::env::var("PINO_HEADLESS_EXACT_REFINE_STEP_CAP").ok();
        let prior_warm_steps = std::env::var("PINO_HEADLESS_ISOLATED_EXACT_WARM_STEPS").ok();
        unsafe {
            std::env::set_var("PINO_HEADLESS_EXACT_REFINE_ROUNDS", "1");
            std::env::set_var("PINO_HEADLESS_EXACT_REFINE_STEP_CAP", "8");
            std::env::set_var("PINO_HEADLESS_ISOLATED_EXACT_WARM_STEPS", "8");
        }
        let mut engine = UniversalPinnEngine::default();
        let raw_batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 8,
            target_loss: 1e-4,
            training_mode: Some("benchmark".to_string()),
            benchmark_id: Some("benchmark_cantilever_2d".to_string()),
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(24),
            min_improvement: Some(1e-6),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(50),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(48),
            max_backoffs: Some(1),
            max_optimizer_switches: Some(4),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(42),
            analysis_type: None,
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
            collocation_points: Some(64),
            boundary_points: Some(24),
            interface_points: Some(12),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(4),
            stage2_epochs: Some(8),
            stage3_ramp_epochs: Some(12),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };
        let batch = crate::benchmarks::apply_training_benchmark(raw_batch)
            .expect("benchmark normalization");
        let mut progress = Vec::<TrainingProgressEvent>::new();

        let result = engine.train_with_progress_with_checkpoint(
            &batch,
            |p| progress.push(p),
            || false,
            |_e, _s, _b| {},
        );

        let positive_epoch_max = progress
            .iter()
            .filter(|p| p.epoch > 0)
            .map(|p| p.val_loss)
            .fold(0.0_f64, f64::max);
        let refine_baseline = progress
            .iter()
            .find(|p| p.epoch == 0 && p.stage_id == "exact-refine" && p.lr_phase == "baseline")
            .map(|p| p.val_loss)
            .unwrap_or(f64::INFINITY);
        assert!(
            positive_epoch_max < 20.0,
            "benchmark progress should stay on the physical cantilever scale, got max val {positive_epoch_max}"
        );
        assert!(
            refine_baseline < 20.0,
            "exact-refine baseline should use the physical benchmark score, got {refine_baseline}"
        );
        assert!(
            result.val_loss < 20.0,
            "benchmark result should stay on the physical cantilever scale, got {}",
            result.val_loss
        );

        if let Some(prev) = prior_rounds {
            unsafe {
                std::env::set_var("PINO_HEADLESS_EXACT_REFINE_ROUNDS", prev);
            }
        } else {
            unsafe {
                std::env::remove_var("PINO_HEADLESS_EXACT_REFINE_ROUNDS");
            }
        }
        if let Some(prev) = prior_step_cap {
            unsafe {
                std::env::set_var("PINO_HEADLESS_EXACT_REFINE_STEP_CAP", prev);
            }
        } else {
            unsafe {
                std::env::remove_var("PINO_HEADLESS_EXACT_REFINE_STEP_CAP");
            }
        }
        if let Some(prev) = prior_warm_steps {
            unsafe {
                std::env::set_var("PINO_HEADLESS_ISOLATED_EXACT_WARM_STEPS", prev);
            }
        } else {
            unsafe {
                std::env::remove_var("PINO_HEADLESS_ISOLATED_EXACT_WARM_STEPS");
            }
        }
    }

    #[test]
    fn pino_exact_ui_default_emits_epoch_bootstrap_before_first_metrics() {
        let mut engine = UniversalPinnEngine::default();
        let mut case0 = SolveInput::default();
        case0.geometry.length_in = 10.0;
        case0.geometry.width_in = 1.0;
        case0.geometry.thickness_in = 0.25;
        case0.geometry.hole_diameter_in = Some(0.0);
        case0.load.axial_load_lbf = 0.0;
        case0.load.vertical_point_load_lbf = -100.0;

        let batch = TrainingBatch {
            cases: vec![case0],
            epochs: 40,
            target_loss: 0.01,
            training_mode: None,
            benchmark_id: None,
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(10_000),
            min_improvement: Some(1e-6),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(10),
            online_active_learning: Some(true),
            autonomous_mode: Some(true),
            max_topology: Some(128),
            max_backoffs: Some(12),
            max_optimizer_switches: Some(8),
            checkpoint_every_epochs: Some(1000),
            checkpoint_retention: Some(8),
            seed: Some(42),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
            collocation_points: Some(4096),
            boundary_points: Some(1024),
            interface_points: Some(512),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(2500),
            stage2_epochs: Some(4500),
            stage3_ramp_epochs: Some(3000),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };

        let started_at = Instant::now();
        let saw_epoch_progress = Cell::new(false);
        let mut events = Vec::new();
        let result = engine.train_with_progress_with_checkpoint(
            &batch,
            |p| {
                if events.len() < 16 {
                    events.push((p.epoch, p.stage_id, p.lr_phase));
                }
                if p.epoch > 0 {
                    saw_epoch_progress.set(true);
                }
            },
            || saw_epoch_progress.get(),
            |_e, _s, _b| {},
        );

        println!(
            "pino-ui-startup-summary: first_epoch_elapsed_ms={:.3} events={:?}",
            started_at.elapsed().as_secs_f64() * 1000.0,
            events
        );
        assert!(
            events.iter().any(|(epoch, stage, lr_phase)| {
                *epoch == 0 && stage.starts_with("stage-") && lr_phase == "epoch-1-bootstrap"
            }),
            "expected epoch bootstrap event before first metrics, got {events:?}"
        );
        assert!(
            events.iter().any(|(epoch, _, _)| *epoch > 0),
            "expected exact UI PINO batch to advance beyond epoch bootstrap, got {events:?}"
        );
        assert!(
            result
                .notes
                .iter()
                .any(|n| n.contains("PINOFastStart: deferred startup calibration and holdout validation until runtime epochs")),
            "expected fast-start note in result"
        );
    }

    #[test]
    fn burn_runtime_emits_stage_transition_notes_for_explicit_schedule() {
        let mut engine = UniversalPinnEngine::default();
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 2,
            target_loss: 0.0,
            training_mode: None,
            benchmark_id: None,
            learning_rate: Some(1e-3),
            auto_mode: Some(true),
            max_total_epochs: Some(18),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(50),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(32),
            max_backoffs: Some(2),
            max_optimizer_switches: Some(2),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(7),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("burn-ndarray-cpu".to_string()),
            collocation_points: Some(256),
            boundary_points: Some(64),
            interface_points: Some(64),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(5),
            stage2_epochs: Some(7),
            stage3_ramp_epochs: Some(6),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };

        let result =
            engine.train_with_progress_with_checkpoint(&batch, |_p| {}, || false, |_e, _s, _b| {});
        assert!(
            result
                .notes
                .iter()
                .any(|n| n.contains("StageTransition: stage-1 -> stage-2")),
            "expected stage-1 -> stage-2 transition in notes"
        );
        assert!(
            result
                .notes
                .iter()
                .any(|n| n.contains("StageTransition: stage-2 -> stage-3")),
            "expected stage-2 -> stage-3 transition in notes"
        );
        assert!(
            result
                .notes
                .iter()
                .any(|n| n.contains("PINOOptimizerSchedule: warmup=pino-adam, finetune=pino-lbfgs")),
            "expected explicit optimizer schedule note"
        );
        assert!(
            result
                .notes
                .iter()
                .any(|n| n.contains("optimizerId=pino-lbfgs")),
            "expected runtime to enter the pino-lbfgs phase"
        );
    }

    #[test]
    fn burn_runtime_honors_manual_stop_signal() {
        let mut engine = UniversalPinnEngine::default();
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 40,
            target_loss: 0.0,
            training_mode: None,
            benchmark_id: None,
            learning_rate: Some(1e-3),
            auto_mode: Some(true),
            max_total_epochs: Some(200),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(100),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(32),
            max_backoffs: Some(2),
            max_optimizer_switches: Some(2),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(9),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("burn-ndarray-cpu".to_string()),
            collocation_points: Some(128),
            boundary_points: Some(64),
            interface_points: Some(64),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(20),
            stage2_epochs: Some(80),
            stage3_ramp_epochs: Some(100),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };

        let mut polls = 0usize;
        let result = engine.train_with_progress_with_checkpoint(
            &batch,
            |_p| {},
            || {
                polls += 1;
                polls > 20
            },
            |_e, _s, _b| {},
        );
        assert_eq!(result.stop_reason, "manual-stop");
        assert!(result.completed_epochs < 200);
    }

    #[test]
    fn burn_runtime_invokes_checkpoint_callback() {
        let mut engine = UniversalPinnEngine::default();
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 10,
            target_loss: 0.0,
            training_mode: None,
            benchmark_id: None,
            learning_rate: Some(1e-3),
            auto_mode: Some(true),
            max_total_epochs: Some(40),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(100),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(32),
            max_backoffs: Some(2),
            max_optimizer_switches: Some(2),
            checkpoint_every_epochs: Some(5),
            checkpoint_retention: Some(4),
            seed: Some(11),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("burn-ndarray-cpu".to_string()),
            collocation_points: Some(128),
            boundary_points: Some(64),
            interface_points: Some(64),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(10),
            stage2_epochs: Some(15),
            stage3_ramp_epochs: Some(15),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };

        let mut cp_total = 0usize;
        let mut cp_best = 0usize;
        let result = engine.train_with_progress_with_checkpoint(
            &batch,
            |_p| {},
            || false,
            |_e, _s, is_best| {
                cp_total += 1;
                if is_best {
                    cp_best += 1;
                }
            },
        );
        assert!(cp_total > 0, "expected checkpoint callback invocations");
        assert!(cp_best > 0, "expected at least one best-marked checkpoint");
        assert!(
            result
                .notes
                .iter()
                .any(|n| n.starts_with("PINOCheckpoints: total=")),
            "expected PINOCheckpoints note"
        );
    }

    #[test]
    fn burn_runtime_compact_default_profile_stays_finite() {
        let mut engine = UniversalPinnEngine::default();
        let mut case = SolveInput::default();
        case.mesh.nx = 12;
        case.mesh.ny = 8;
        case.mesh.nz = 2;
        let batch = TrainingBatch {
            cases: vec![case],
            epochs: 6,
            target_loss: 0.0,
            training_mode: None,
            benchmark_id: None,
            learning_rate: Some(1e-3),
            auto_mode: Some(true),
            max_total_epochs: Some(64),
            min_improvement: Some(0.05),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(200),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(32),
            max_backoffs: Some(0),
            max_optimizer_switches: Some(1),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(42),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("burn-ndarray-cpu".to_string()),
            collocation_points: Some(32),
            boundary_points: Some(16),
            interface_points: Some(8),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(12),
            stage2_epochs: Some(16),
            stage3_ramp_epochs: Some(20),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };

        let result =
            engine.train_with_progress_with_checkpoint(&batch, |_p| {}, || false, |_e, _s, _b| {});
        assert!(result.completed_epochs > 0);
        assert!(result.completed_epochs <= 64);
        assert!(result.val_loss.is_finite());
        assert!(matches!(
            result.stop_reason.as_str(),
            "plateau-stop" | "max-epochs-reached" | "target-loss-reached"
        ));
    }

    #[test]
    fn burn_runtime_exact_ui_default_cantilever_batch_stays_healthy() {
        let mut engine = UniversalPinnEngine::default();
        let mut case0 = SolveInput::default();
        case0.geometry.length_in = 10.0;
        case0.geometry.width_in = 1.0;
        case0.geometry.thickness_in = 0.25;
        case0.geometry.hole_diameter_in = Some(0.0);
        case0.load.axial_load_lbf = 0.0;
        case0.load.vertical_point_load_lbf = -100.0;
        let batch = TrainingBatch {
            cases: vec![case0],
            epochs: 40,
            target_loss: 0.01,
            training_mode: None,
            benchmark_id: None,
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(120),
            min_improvement: Some(1e-6),
            progress_emit_every_epochs: Some(5),
            network_emit_every_epochs: Some(200),
            online_active_learning: Some(true),
            autonomous_mode: Some(true),
            max_topology: Some(128),
            max_backoffs: Some(1),
            max_optimizer_switches: Some(8),
            checkpoint_every_epochs: Some(1000),
            checkpoint_retention: Some(8),
            seed: Some(42),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("burn-ndarray-cpu".to_string()),
            collocation_points: Some(4096),
            boundary_points: Some(1024),
            interface_points: Some(512),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(24),
            stage2_epochs: Some(52),
            stage3_ramp_epochs: Some(44),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };

        let result =
            engine.train_with_progress_with_checkpoint(&batch, |_p| {}, || false, |_e, _s, _b| {});
        println!(
            "ui-default-cantilever-batch: stop_reason={}, completed_epochs={}, loss={:.6e}, val_loss={:.6e}",
            result.stop_reason, result.completed_epochs, result.loss, result.val_loss
        );
        assert!(result.loss.is_finite());
        assert!(result.val_loss.is_finite());
        assert!(
            result.loss < 1.0,
            "ui default cantilever batch failed to settle: {}",
            result.loss
        );
        assert!(
            result.stop_reason == "target-loss-reached" || result.stop_reason == "plateau-stop",
            "unexpected stop reason for default ui batch: {}",
            result.stop_reason
        );
        assert!(result.completed_epochs <= 120);
        assert!(
            result.completed_epochs > 0,
            "default ui batch made no epoch progress"
        );
        assert!(
            result
                .notes
                .iter()
                .any(|n| n.starts_with("TargetOutcome: requested=")),
            "expected target outcome note for default ui batch"
        );
        assert!(
            result
                .notes
                .iter()
                .any(|n| n.starts_with("OperatorTargets: physics-informed")),
            "expected physics-informed operator target note for default ui batch"
        );
    }

    #[test]
    fn pino_runtime_infer_returns_operator_backed_result() {
        let mut engine = UniversalPinnEngine::default();
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 2,
            target_loss: 0.5,
            training_mode: None,
            benchmark_id: None,
            learning_rate: Some(1e-3),
            auto_mode: Some(true),
            max_total_epochs: Some(6),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(50),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(32),
            max_backoffs: Some(2),
            max_optimizer_switches: Some(2),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(13),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
            collocation_points: Some(256),
            boundary_points: Some(64),
            interface_points: Some(64),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(2),
            stage2_epochs: Some(2),
            stage3_ramp_epochs: Some(2),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };

        let _ =
            engine.train_with_progress_with_checkpoint(&batch, |_p| {}, || false, |_e, _s, _b| {});
        let inferred = engine.infer(&SolveInput::default());
        assert!(inferred.pino.is_some());
        assert!(inferred.fem_like.von_mises_psi.is_finite());
        assert!(
            inferred
                .diagnostics
                .iter()
                .any(|line| line.contains("PINO operator inference")),
            "expected PINO operator inference diagnostics"
        );
    }

    #[test]
    fn pino_runtime_infer_enforces_holdout_trust_gate() {
        let mut engine = UniversalPinnEngine::default();
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 2,
            target_loss: 0.5,
            training_mode: None,
            benchmark_id: None,
            learning_rate: Some(1e-3),
            auto_mode: Some(true),
            max_total_epochs: Some(6),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(50),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(32),
            max_backoffs: Some(2),
            max_optimizer_switches: Some(2),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(17),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
            collocation_points: Some(256),
            boundary_points: Some(64),
            interface_points: Some(64),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(2),
            stage2_epochs: Some(2),
            stage3_ramp_epochs: Some(2),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };

        let _ =
            engine.train_with_progress_with_checkpoint(&batch, |_p| {}, || false, |_e, _s, _b| {});
        if let Some(state) = engine.burn_state.as_mut() {
            if let Some(meta) = state.pino.as_mut() {
                let mut summary = meta
                    .holdout_validation
                    .clone()
                    .unwrap_or_else(|| crate::pino::default_holdout_validation(&batch));
                summary.trusted = false;
                summary.accepted_without_fallback = false;
                meta.holdout_validation = Some(summary);
            }
        }
        let inferred = engine.infer(&SolveInput::default());
        assert!(
            inferred.used_fem_fallback,
            "expected FEM fallback when holdout trust gate fails"
        );
        assert!(
            inferred
                .fallback_reason
                .as_deref()
                .unwrap_or_default()
                .contains("holdout trust gate"),
            "expected holdout trust gate reason, got {:?}",
            inferred.fallback_reason
        );
    }

    #[test]
    fn pino_runtime_snapshot_resume_preserves_inference_shape() {
        let mut engine = UniversalPinnEngine::default();
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 4,
            target_loss: 0.2,
            training_mode: None,
            benchmark_id: None,
            learning_rate: Some(8e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(16),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(50),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(32),
            max_backoffs: Some(2),
            max_optimizer_switches: Some(2),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(51),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
            collocation_points: Some(256),
            boundary_points: Some(64),
            interface_points: Some(64),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(4),
            stage2_epochs: Some(6),
            stage3_ramp_epochs: Some(6),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };

        let _ =
            engine.train_with_progress_with_checkpoint(&batch, |_p| {}, || false, |_e, _s, _b| {});
        let input = SolveInput::default();
        let before = engine.infer(&input);
        let snapshot = engine.snapshot_state();
        let mut resumed = UniversalPinnEngine::default();
        resumed.load_state(snapshot);
        let after = resumed.infer(&input);

        assert_eq!(before.fem_like.displacement_vector.len(), 3);
        assert_eq!(after.fem_like.displacement_vector.len(), 3);
        assert!(
            (before.fem_like.von_mises_psi - after.fem_like.von_mises_psi).abs() <= 1e-6,
            "resume changed inferred von-mises beyond tolerance: before={} after={}",
            before.fem_like.von_mises_psi,
            after.fem_like.von_mises_psi
        );
        assert_eq!(
            before.used_fem_fallback, after.used_fem_fallback,
            "resume changed safeguard fallback path"
        );
    }

    #[test]
    fn pino_runtime_checkpoint_resume_survives_lbfgs_phase() {
        let mut engine = UniversalPinnEngine::default();
        let mut case0 = SolveInput::default();
        case0.geometry.length_in = 10.0;
        case0.geometry.width_in = 1.0;
        case0.geometry.thickness_in = 0.25;
        case0.geometry.hole_diameter_in = Some(0.0);
        case0.load.axial_load_lbf = 0.0;
        case0.load.vertical_point_load_lbf = -100.0;

        let batch = TrainingBatch {
            cases: vec![case0.clone()],
            epochs: 40,
            target_loss: 1e-9,
            training_mode: None,
            benchmark_id: None,
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(40),
            min_improvement: Some(1e-6),
            progress_emit_every_epochs: Some(2),
            network_emit_every_epochs: Some(40),
            online_active_learning: Some(true),
            autonomous_mode: Some(true),
            max_topology: Some(128),
            max_backoffs: Some(3),
            max_optimizer_switches: Some(8),
            checkpoint_every_epochs: Some(4),
            checkpoint_retention: Some(8),
            seed: Some(42),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
            collocation_points: Some(64),
            boundary_points: Some(32),
            interface_points: Some(32),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(8),
            stage2_epochs: Some(8),
            stage3_ramp_epochs: Some(24),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };

        let mut checkpoints: Vec<(usize, UniversalPinnState, bool)> = Vec::new();
        let result = engine.train_with_progress_with_checkpoint(
            &batch,
            |_p| {},
            || false,
            |epoch, snapshot, is_best| {
                checkpoints.push((epoch, snapshot, is_best));
            },
        );

        assert!(
            result
                .notes
                .iter()
                .any(|n| n.contains("optimizerId=pino-lbfgs")),
            "expected runtime to enter the pino-lbfgs phase"
        );

        let (checkpoint_epoch, snapshot, _) = checkpoints
            .iter()
            .rev()
            .find(|(epoch, _, _)| *epoch >= 32)
            .cloned()
            .expect("expected checkpoint taken after the pino-lbfgs switch window");

        let before = engine.infer(&case0);
        let mut resumed = UniversalPinnEngine::default();
        resumed.load_state(snapshot);
        let after = resumed.infer(&case0);

        assert!(
            checkpoint_epoch >= 32,
            "expected post-lbfgs checkpoint, got epoch {checkpoint_epoch}"
        );
        assert!(before.pino.is_some() && after.pino.is_some());
        assert!(
            (before.fem_like.von_mises_psi - after.fem_like.von_mises_psi).abs() <= 1e-6,
            "resume changed inferred von-mises beyond tolerance: before={} after={}",
            before.fem_like.von_mises_psi,
            after.fem_like.von_mises_psi
        );
        assert_eq!(
            before.used_fem_fallback, after.used_fem_fallback,
            "resume changed safeguard fallback path after post-lbfgs checkpoint"
        );
    }

    #[test]
    #[ignore = "manual release-signoff benchmark profile"]
    fn pino_release_signoff_profile_manual() {
        let engine = UniversalPinnEngine::default();
        let mut case = SolveInput::default();
        case.geometry.length_in = 10.0;
        case.geometry.width_in = 1.0;
        case.geometry.thickness_in = 0.25;
        case.geometry.hole_diameter_in = Some(0.0);
        case.mesh.nx = 12;
        case.mesh.ny = 8;
        case.mesh.nz = 2;
        case.load = LoadInput {
            axial_load_lbf: 0.0,
            vertical_point_load_lbf: -100.0,
        };

        let config = crate::pino::infer_config_for_case(&case, "pino-ndarray-cpu", 4);
        let iter = 8usize;
        let t0 = Instant::now();
        for _ in 0..iter {
            let _ = crate::pino::build_operator_prediction(&case, &config);
        }
        let op_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let throughput_eps = (iter as f64 / (op_ms / 1000.0)).max(0.0);

        let t_infer = Instant::now();
        let prediction = crate::pino::build_operator_prediction(&case, &config);
        let _decoded = crate::pino::decode_prediction(&case, &prediction);
        let infer_latency_ms = t_infer.elapsed().as_secs_f64() * 1000.0;

        let cells =
            (config.operator_grid.nx * config.operator_grid.ny * config.operator_grid.nz) as f64;
        let channels =
            (config.operator_grid.input_channels + config.operator_grid.output_channels) as f64;
        let mem_estimate_mb = (cells * channels * 8.0 * 6.0) / (1024.0 * 1024.0);

        let snapshot = engine.snapshot_state();
        let mut resumed = UniversalPinnEngine::default();
        resumed.load_state(snapshot);
        let before_status = engine.status();
        let after_status = resumed.status();
        let resume_ok = before_status.model_version == after_status.model_version
            && before_status.architecture == after_status.architecture
            && (before_status.learning_rate - after_status.learning_rate).abs() <= 1e-12;

        println!(
            "pino-signoff-summary: throughput_eps={:.3} infer_latency_ms={:.3} memory_estimate_mb={:.3} resume_ok={} val_loss={:.6e} epochs={}",
            throughput_eps,
            infer_latency_ms,
            mem_estimate_mb,
            if resume_ok { 1 } else { 0 },
            0.0f64,
            0usize
        );
    }

    #[test]
    #[ignore = "manual headless diagnostic for UI-default batch with target 1e-9"]
    fn pino_headless_default_target_1e9_diagnostic() {
        let env_flag = |key: &str, default: bool| {
            std::env::var(key)
                .ok()
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "on" | "ON"))
                .unwrap_or(default)
        };
        let max_total_epochs = std::env::var("PINO_HEADLESS_MAX_TOTAL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(10_000);
        let max_backoffs = std::env::var("PINO_HEADLESS_MAX_BACKOFFS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(12);
        let stage1_epochs = std::env::var("PINO_HEADLESS_STAGE1")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                ((max_total_epochs as f64) * 0.08)
                    .round()
                    .clamp(24.0, 160.0) as usize
            });
        let stage2_epochs = std::env::var("PINO_HEADLESS_STAGE2")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                ((max_total_epochs as f64) * 0.14)
                    .round()
                    .clamp(40.0, 280.0) as usize
            });
        let stage3_epochs = std::env::var("PINO_HEADLESS_STAGE3")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                ((max_total_epochs as f64) * 0.18)
                    .round()
                    .clamp(48.0, 360.0) as usize
            });
        let collocation_points = std::env::var("PINO_HEADLESS_COLLOCATION")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4096);
        let boundary_points = std::env::var("PINO_HEADLESS_BOUNDARY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1024);
        let interface_points = std::env::var("PINO_HEADLESS_INTERFACE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512);
        let progress_every = std::env::var("PINO_HEADLESS_PROGRESS_EVERY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2)
            .max(1);
        let network_every = std::env::var("PINO_HEADLESS_NETWORK_EVERY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20)
            .max(1);
        let max_topology = std::env::var("PINO_HEADLESS_MAX_TOPOLOGY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(128);
        let max_runtime_s = std::env::var("PINO_HEADLESS_MAX_RUNTIME_S")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0);
        let benchmark_id = std::env::var("PINO_HEADLESS_BENCHMARK_ID")
            .ok()
            .filter(|v| !v.trim().is_empty());
        let benchmark_target_loss = std::env::var("PINO_HEADLESS_TARGET_LOSS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0);
        let online_active_learning = env_flag("PINO_HEADLESS_ONLINE_ACTIVE_LEARNING", true);
        let mut engine = UniversalPinnEngine::default();
        let mut case0 = SolveInput::default();
        case0.geometry.length_in = 10.0;
        case0.geometry.width_in = 1.0;
        case0.geometry.thickness_in = 0.25;
        case0.geometry.hole_diameter_in = Some(0.0);
        case0.load.axial_load_lbf = 0.0;
        case0.load.vertical_point_load_lbf = -100.0;
        let batch = TrainingBatch {
            cases: vec![case0],
            epochs: 40,
            target_loss: benchmark_target_loss.unwrap_or_else(|| if benchmark_id.is_some() { 1e-4 } else { 1e-9 }),
            training_mode: benchmark_id.as_ref().map(|_| "benchmark".to_string()),
            benchmark_id: benchmark_id.clone(),
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(max_total_epochs),
            min_improvement: Some(1e-6),
            progress_emit_every_epochs: Some(progress_every),
            network_emit_every_epochs: Some(network_every),
            online_active_learning: Some(online_active_learning),
            autonomous_mode: Some(true),
            max_topology: Some(max_topology),
            max_backoffs: Some(max_backoffs),
            max_optimizer_switches: Some(8),
            checkpoint_every_epochs: Some(1000),
            checkpoint_retention: Some(8),
            seed: Some(42),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
            collocation_points: Some(collocation_points),
            boundary_points: Some(boundary_points),
            interface_points: Some(interface_points),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(stage1_epochs),
            stage2_epochs: Some(stage2_epochs),
            stage3_ramp_epochs: Some(stage3_epochs),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };
        let batch = if benchmark_id.is_some() {
            crate::benchmarks::apply_training_benchmark(batch).expect("headless benchmark profile")
        } else {
            batch
        };
        let isolated_exact_headless =
            matches!(
                batch.benchmark_id.as_deref(),
                Some("benchmark_cantilever_2d")
            ) || env_flag("PINO_HEADLESS_ISOLATED_EXACT_CANTILEVER", false);

        let mut best_val = f64::MAX;
        let mut last_arch: Vec<usize> = vec![];
        let mut topology_changes = 0usize;
        let mut layer_only_changes = 0usize;
        let mut width_or_mode_changes = 0usize;
        let mut architecture_timeline: Vec<(usize, Vec<usize>, f64)> = Vec::new();
        let mut val_history: Vec<f64> = Vec::new();
        let mut last_reported_epoch = 0usize;
        let mut last_phase = String::new();
        let runtime_budget_hit = std::cell::Cell::new(false);
        let start = std::time::Instant::now();
        println!(
            "pino-headless-1e9-config: benchmark_id={} max_total={} max_backoffs={} stage1={} stage2={} stage3={} collocation={} boundary={} interface={} max_topology={} progress_every={} network_every={} online_active_learning={} max_runtime_s={} skip_exact_refine={} isolated_exact_cantilever={}",
            batch.benchmark_id.as_deref().unwrap_or("none"),
            max_total_epochs,
            max_backoffs,
            stage1_epochs,
            stage2_epochs,
            stage3_epochs,
            collocation_points,
            boundary_points,
            interface_points,
            max_topology,
            progress_every,
            network_every,
            online_active_learning,
            max_runtime_s.unwrap_or(0.0),
            env_flag("PINO_HEADLESS_SKIP_EXACT_REFINE", false),
            isolated_exact_headless
        );
        let result = engine.train_with_progress_with_checkpoint(
            &batch,
            |p| {
                if p.epoch == 0 {
                    if p.lr_phase != last_phase {
                        println!(
                            "pino-headless-1e9-progress: t={:.1}s epoch=0 stage={} phase={} val={:.6e} loss={:.6e}",
                            start.elapsed().as_secs_f64(),
                            p.stage_id,
                            p.lr_phase,
                            p.val_loss,
                            p.loss
                        );
                        last_phase = p.lr_phase.clone();
                    }
                    return;
                }

                best_val = best_val.min(p.val_loss);
                val_history.push(p.val_loss);
                if p.epoch != last_reported_epoch && (p.epoch == 1 || p.epoch % progress_every == 0) {
                    println!(
                        "pino-headless-1e9-progress: t={:.1}s epoch={} stage={} phase={} val={:.6e} loss={:.6e} arch={:?}",
                        start.elapsed().as_secs_f64(),
                        p.epoch,
                        p.stage_id,
                        p.lr_phase,
                        p.val_loss,
                        p.loss,
                        p.architecture
                    );
                    last_reported_epoch = p.epoch;
                    last_phase = p.lr_phase.clone();
                }
                if last_arch.is_empty() {
                    last_arch = p.architecture.clone();
                } else if p.architecture != last_arch {
                    topology_changes = topology_changes.saturating_add(1);
                    let prev_layers = last_arch.len();
                    let next_layers = p.architecture.len();
                    let prev_width = *last_arch
                        .iter()
                        .skip(1)
                        .take(prev_layers.saturating_sub(2))
                        .max()
                        .unwrap_or(&0);
                    let next_width = *p
                        .architecture
                        .iter()
                        .skip(1)
                        .take(next_layers.saturating_sub(2))
                        .max()
                        .unwrap_or(&0);
                    if next_layers != prev_layers && next_width == prev_width {
                        layer_only_changes = layer_only_changes.saturating_add(1);
                    } else {
                        width_or_mode_changes = width_or_mode_changes.saturating_add(1);
                    }
                    architecture_timeline.push((p.epoch, p.architecture.clone(), p.val_loss));
                    last_arch = p.architecture.clone();
                }
            },
            || {
                if let Some(limit_s) = max_runtime_s {
                    if start.elapsed().as_secs_f64() >= limit_s {
                        runtime_budget_hit.set(true);
                        return true;
                    }
                }
                false
            },
            |_epoch, _state, _is_best| {},
        );

        let floor_val = if best_val.is_finite() {
            best_val
        } else {
            result.val_loss
        };
        let tail_window = val_history
            .iter()
            .rev()
            .take(40)
            .copied()
            .collect::<Vec<_>>();
        let tail_mean = if tail_window.is_empty() {
            result.val_loss
        } else {
            tail_window.iter().sum::<f64>() / tail_window.len() as f64
        };

        println!(
            "pino-headless-1e9-summary: stop_reason={} runtime_budget_hit={} elapsed_s={:.1} epochs={} max_total={} max_backoffs={} stage1={} stage2={} stage3={} best_val={:.6e} final_val={:.6e} tail_mean={:.6e} topology_changes={} layer_only_changes={} width_or_mode_changes={} arch={:?}",
            result.stop_reason,
            runtime_budget_hit.get(),
            start.elapsed().as_secs_f64(),
            result.completed_epochs,
            max_total_epochs,
            max_backoffs,
            stage1_epochs,
            stage2_epochs,
            stage3_epochs,
            floor_val,
            result.val_loss,
            tail_mean,
            topology_changes,
            layer_only_changes,
            width_or_mode_changes,
            result.architecture
        );
        for (epoch, arch, val) in architecture_timeline {
            println!(
                "pino-headless-1e9-topology: epoch={} val={:.6e} arch={:?}",
                epoch, val, arch
            );
        }
        if let Some(certification) = result.benchmark_certification.as_ref() {
            println!(
                "pino-headless-1e9-certification: status={} suggested_target={:.6e} summary={}",
                certification.status,
                certification.suggested_target_loss,
                certification.summary
            );
        }
        for note in result.notes.iter().filter(|note| {
            note.contains("PINODiagnostics:")
                || note.contains("PINOOptimizerSchedule:")
                || note.contains("PINOResidualBreakdown")
                || note.contains("PINOExactRefine")
                || note.contains("StageTransition:")
                || note.contains("BenchmarkCertification:")
        }) {
            println!("pino-headless-1e9-note: {note}");
        }

        assert!(result.completed_epochs > 0);
        assert!(result.val_loss.is_finite());
    }

    #[test]
    fn pino_runtime_adapts_architecture_on_plateau_with_guardrails() {
        let mut engine = UniversalPinnEngine::default();
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 20,
            target_loss: 0.0,
            training_mode: None,
            benchmark_id: None,
            learning_rate: Some(1e-3),
            auto_mode: Some(true),
            max_total_epochs: Some(500),
            min_improvement: Some(0.1),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(100),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(48),
            max_backoffs: Some(4),
            max_optimizer_switches: Some(4),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(99),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
            collocation_points: Some(512),
            boundary_points: Some(128),
            interface_points: Some(64),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(120),
            stage2_epochs: Some(220),
            stage3_ramp_epochs: Some(160),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        };

        let result =
            engine.train_with_progress_with_checkpoint(&batch, |_p| {}, || false, |_e, _s, _b| {});
        assert!(
            result
                .notes
                .iter()
                .any(|note| note.contains("PINOArchAdapt:")),
            "expected PINO architecture adaptation note"
        );
        assert!(
            result.architecture.len() >= 3,
            "expected valid adapted architecture"
        );
    }
}
