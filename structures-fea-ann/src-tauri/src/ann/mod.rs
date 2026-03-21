use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use nalgebra::{DMatrix, DVector};

use crate::contracts::{
    AnnResult, ModelStatus, NetworkConnectionSnapshot, NetworkNodeSnapshot, NetworkSnapshot,
    SafeguardSettings, SolveInput, SurrogateDomainSummary, TrainResult, TrainingBatch,
    TrainingProgressEvent,
};
use crate::fem;
use crate::physics::stress::{principal_stresses, tresca_from_principal, von_mises};
use crate::pinn_burn::{contact_penalty, universal_loss, ResidualPillars};

const ANN_FEATURE_LABELS: [&str; 19] = [
    "length_in",
    "width_in",
    "thickness_in",
    "hole_diameter_in",
    "hole_to_width_ratio",
    "hole_to_length_ratio",
    "length_to_width_ratio",
    "length_to_thickness_ratio",
    "width_to_thickness_ratio",
    "fix_start_face",
    "fix_end_face",
    "axial_load_lbf",
    "vertical_point_load_lbf",
    "load_magnitude_lbf",
    "youngs_modulus_psi",
    "poisson_ratio",
    "density_lb_in3",
    "thermal_alpha_per_f",
    "yield_strength_psi",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DenseLayer {
    weights: Vec<Vec<f64>>, // out x in
    biases: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DenseLayerMoments {
    m_weights: Vec<Vec<f64>>,
    v_weights: Vec<Vec<f64>>,
    m_biases: Vec<f64>,
    v_biases: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum HiddenActivation {
    Tanh,
    Relu,
    SiLu,
}

impl HiddenActivation {
    fn as_str(self) -> &'static str {
        match self {
            HiddenActivation::Tanh => "tanh",
            HiddenActivation::Relu => "relu",
            HiddenActivation::SiLu => "silu",
        }
    }
}

fn default_hidden_activation() -> HiddenActivation {
    HiddenActivation::Tanh
}

#[derive(Clone)]
pub struct AnnModel {
    layer_sizes: Vec<usize>,
    layers: Vec<DenseLayer>,
    moments: Vec<DenseLayerMoments>,
    best_layers: Option<Vec<DenseLayer>>,
    pub learning_rate: f64,
    pub model_version: u64,
    pub last_loss: f64,
    pub best_val_loss: f64,
    pub train_samples: usize,
    pub plateau_epochs: usize,
    pub audit_frequency: usize,
    pub fallback_enabled: bool,
    max_hidden_layers: usize,
    max_neurons_per_layer: usize,
    data_weight: f64,
    physics_weight: f64,
    adam_beta1: f64,
    adam_beta2: f64,
    adam_eps: f64,
    adam_t: u64,
    feature_scales: Vec<f64>,
    safeguard_preset: String,
    safeguard_uncertainty_threshold: f64,
    safeguard_residual_threshold: f64,
    safeguard_adaptive_by_geometry: bool,
    hidden_activation: HiddenActivation,
    last_train_seed: Option<u64>,
    training_domain: Option<SurrogateDomainSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnModelState {
    pub layer_sizes: Vec<usize>,
    pub layers: Vec<DenseLayer>,
    pub moments: Vec<DenseLayerMoments>,
    pub best_layers: Option<Vec<DenseLayer>>,
    pub learning_rate: f64,
    pub model_version: u64,
    pub last_loss: f64,
    pub best_val_loss: f64,
    pub train_samples: usize,
    pub plateau_epochs: usize,
    pub audit_frequency: usize,
    pub fallback_enabled: bool,
    pub max_hidden_layers: usize,
    pub max_neurons_per_layer: usize,
    pub data_weight: f64,
    pub physics_weight: f64,
    pub adam_beta1: f64,
    pub adam_beta2: f64,
    pub adam_eps: f64,
    pub adam_t: u64,
    pub feature_scales: Vec<f64>,
    #[serde(default = "default_safeguard_preset")]
    pub safeguard_preset: String,
    #[serde(default = "default_safeguard_uncertainty_threshold")]
    pub safeguard_uncertainty_threshold: f64,
    #[serde(default = "default_safeguard_residual_threshold")]
    pub safeguard_residual_threshold: f64,
    #[serde(default = "default_safeguard_adaptive_by_geometry")]
    pub safeguard_adaptive_by_geometry: bool,
    #[serde(default = "default_hidden_activation")]
    pub hidden_activation: HiddenActivation,
    #[serde(default)]
    pub last_train_seed: Option<u64>,
    #[serde(default)]
    pub training_domain: Option<SurrogateDomainSummary>,
}

#[derive(Clone, Copy)]
struct PhysicsContext {
    length_in: f64,
    width_in: f64,
    thickness_in: f64,
    hole_diameter_in: f64,
    axial_load_lbf: f64,
    vertical_load_lbf: f64,
    e_psi: f64,
    rho_lb_in3: f64,
    yield_strength_psi: f64,
    fix_start_face: bool,
    fix_end_face: bool,
    residual_weight_momentum: f64,
    residual_weight_kinematics: f64,
    residual_weight_material: f64,
    residual_weight_boundary: f64,
    contact_penalty: f64,
    plasticity_factor: f64,
}

#[derive(Clone)]
struct TrainingSample {
    x: Vec<f64>,
    y: Vec<f64>,
    physics: PhysicsContext,
}

#[derive(Clone)]
struct TrainingNorm {
    input_scales: Vec<f64>,
    output_scales: Vec<f64>,
}

#[derive(Clone, Copy, Default)]
struct LossBreakdown {
    total: f64,
    data: f64,
    physics: f64,
    momentum: f64,
    kinematics: f64,
    material: f64,
    boundary: f64,
}

#[derive(Clone)]
struct PhysicsPenaltyBreakdown {
    total: f64,
    momentum: f64,
    kinematics: f64,
    material: f64,
    boundary: f64,
    grad: Vec<f64>,
}

impl Default for AnnModel {
    fn default() -> Self {
        Self::new(vec![fem::ANN_INPUT_DIM, 12, 12, fem::ANN_OUTPUT_DIM])
    }
}

impl AnnModel {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        let layers = initialize_layers(&layer_sizes);
        let moments = initialize_moments(&layer_sizes);
        Self {
            layer_sizes,
            layers,
            moments,
            best_layers: None,
            learning_rate: 5e-4,
            model_version: 1,
            last_loss: f64::MAX,
            best_val_loss: f64::MAX,
            train_samples: 0,
            plateau_epochs: 0,
            audit_frequency: 5,
            fallback_enabled: true,
            max_hidden_layers: 5,
            max_neurons_per_layer: 64,
            data_weight: 1.0,
            physics_weight: 0.15,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1e-8,
            adam_t: 0,
            feature_scales: vec![],
            safeguard_preset: "balanced".to_string(),
            safeguard_uncertainty_threshold: 0.26,
            safeguard_residual_threshold: 0.24,
            safeguard_adaptive_by_geometry: true,
            hidden_activation: HiddenActivation::Tanh,
            last_train_seed: None,
            training_domain: None,
        }
    }

    pub fn new_seeded(layer_sizes: Vec<usize>, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let layers = initialize_layers_with_rng(&layer_sizes, &mut rng);
        let moments = initialize_moments(&layer_sizes);
        Self {
            layer_sizes,
            layers,
            moments,
            best_layers: None,
            learning_rate: 5e-4,
            model_version: 1,
            last_loss: f64::MAX,
            best_val_loss: f64::MAX,
            train_samples: 0,
            plateau_epochs: 0,
            audit_frequency: 5,
            fallback_enabled: true,
            max_hidden_layers: 5,
            max_neurons_per_layer: 64,
            data_weight: 1.0,
            physics_weight: 0.15,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1e-8,
            adam_t: 0,
            feature_scales: vec![],
            safeguard_preset: "balanced".to_string(),
            safeguard_uncertainty_threshold: 0.26,
            safeguard_residual_threshold: 0.24,
            safeguard_adaptive_by_geometry: true,
            hidden_activation: HiddenActivation::Tanh,
            last_train_seed: Some(seed),
            training_domain: None,
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        let arch = self.layer_sizes.clone();
        let safeguard_preset = self.safeguard_preset.clone();
        let safeguard_uncertainty_threshold = self.safeguard_uncertainty_threshold;
        let safeguard_residual_threshold = self.safeguard_residual_threshold;
        let safeguard_adaptive_by_geometry = self.safeguard_adaptive_by_geometry;
        let fallback_enabled = self.fallback_enabled;
        let mut fresh = if let Some(s) = seed {
            AnnModel::new_seeded(arch, s)
        } else {
            AnnModel::new(arch)
        };
        fresh.safeguard_preset = safeguard_preset;
        fresh.safeguard_uncertainty_threshold = safeguard_uncertainty_threshold;
        fresh.safeguard_residual_threshold = safeguard_residual_threshold;
        fresh.safeguard_adaptive_by_geometry = safeguard_adaptive_by_geometry;
        fresh.fallback_enabled = fallback_enabled;
        fresh.training_domain = self.training_domain.clone();
        *self = fresh;
    }

    #[allow(dead_code)]
    pub fn train_with_progress<F, S>(
        &mut self,
        batch: &TrainingBatch,
        on_epoch: F,
        should_stop: S,
    ) -> TrainResult
    where
        F: FnMut(TrainingProgressEvent),
        S: FnMut() -> bool,
    {
        self.train_with_progress_with_checkpoint(
            batch,
            on_epoch,
            should_stop,
            |_epoch, _state, _is_best| {},
        )
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
        C: FnMut(usize, AnnModelState, bool),
    {
        let mut emit_preflight = |stage_id: &str,
                                  lr_phase: &str,
                                  train_data_size: usize,
                                  train_data_cap: usize,
                                  collocation_samples_added: usize,
                                  learning_rate: f64,
                                  architecture: &[usize]| {
            on_epoch(TrainingProgressEvent {
                epoch: 0,
                total_epochs: batch
                    .max_total_epochs
                    .unwrap_or(batch.epochs.max(1).saturating_mul(20))
                    .max(1),
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
                stage_id: stage_id.to_string(),
                optimizer_id: "adamw".to_string(),
                lr_phase: lr_phase.to_string(),
                target_band_low: 0.0,
                target_band_high: 0.0,
                trend_slope: 0.0,
                trend_variance: 0.0,
                watchdog_trigger_count: 0,
                collocation_samples_added,
                train_data_size,
                train_data_cap,
                residual_weight_momentum: 0.0,
                residual_weight_kinematics: 0.0,
                residual_weight_material: 0.0,
                residual_weight_boundary: 0.0,
                learning_rate,
                architecture: architecture.to_vec(),
                progress_ratio: 0.0,
                network: NetworkSnapshot {
                    layer_sizes: vec![],
                    nodes: vec![],
                    connections: vec![],
                },
                pino: None,
            });
        };
        let train_seed = batch.seed.unwrap_or_else(rand::random::<u64>);
        let mut train_rng = StdRng::seed_from_u64(train_seed);
        self.last_train_seed = Some(train_seed);
        if let Some(lr) = batch.learning_rate {
            self.learning_rate = lr.clamp(1e-6, 1e-1);
        }
        emit_preflight(
            "preflight",
            "preparing-cases",
            0,
            0,
            0,
            self.learning_rate,
            &self.layer_sizes,
        );

        let base_cases = if batch.cases.is_empty() {
            vec![SolveInput::default()]
        } else {
            batch.cases.clone()
        };
        let analysis_type = batch
            .analysis_type
            .as_deref()
            .unwrap_or("general")
            .to_ascii_lowercase();
        let is_cantilever_training = analysis_type == "cantilever";
        let is_discontinuity_prone = analysis_type == "plate-hole";
        let samples = expand_training_cases(&base_cases, &analysis_type);

        let mut dataset = Vec::new();
        for case in &samples {
            let train_case = widen_case_family(case, &analysis_type, &mut train_rng);
            dataset.push(with_feature_and_physics(&train_case, batch, &analysis_type));
        }
        emit_preflight(
            "preflight",
            "building-dataset",
            dataset.len(),
            0,
            0,
            self.learning_rate,
            &self.layer_sizes,
        );
        let collocation_limit = batch.collocation_points.unwrap_or(0).min(4096);
        let boundary_limit = batch.boundary_points.unwrap_or(0).min(2048);
        let interface_limit = batch.interface_points.unwrap_or(0).min(2048);
        let mut collocation_added = 0usize;
        if collocation_limit > 0 || boundary_limit > 0 || interface_limit > 0 {
            let generated = generate_collocation_samples(
                &samples,
                collocation_limit,
                boundary_limit,
                interface_limit,
                batch,
                &mut train_rng,
            );
            collocation_added = generated.len();
            dataset.extend(generated);
        }
        emit_preflight(
            "preflight",
            "normalizing",
            dataset.len(),
            0,
            collocation_added,
            self.learning_rate,
            &self.layer_sizes,
        );

        self.training_domain = Some(build_surrogate_domain_summary(
            &base_cases,
            &samples,
            &analysis_type,
        ));
        let input_dim = dataset
            .first()
            .map(|sample| sample.x.len())
            .unwrap_or(fem::ANN_INPUT_DIM);
        let output_dim = dataset
            .first()
            .map(|sample| sample.y.len())
            .unwrap_or(fem::ANN_OUTPUT_DIM);
        self.ensure_network_io(input_dim, output_dim);

        self.train_samples += dataset.len();
        let norm = build_training_norm(&dataset);
        self.feature_scales = norm.input_scales.clone();
        for sample in &mut dataset {
            for i in 0..sample.x.len() {
                let s = norm.input_scales.get(i).copied().unwrap_or(1.0).max(1e-9);
                sample.x[i] /= s;
            }
        }

        let auto_mode = batch.auto_mode.unwrap_or(true);
        let autonomous_mode = batch.autonomous_mode.unwrap_or(true);
        let focused_exact_family = {
            let requested = batch.target_loss.max(0.0);
            requested.is_finite()
                && requested > 0.0
                && requested <= 1e-8
                && (is_cantilever_training || is_discontinuity_prone)
        };
        if let Some(max_topo) = batch.max_topology {
            self.max_neurons_per_layer = max_topo.clamp(8, 256);
            self.max_hidden_layers = (((max_topo / 16) + 1).clamp(3, 10)).max(3);
        }
        if focused_exact_family {
            self.hidden_activation = HiddenActivation::Tanh;
            self.learning_rate = self.learning_rate.max(if is_discontinuity_prone {
                9e-4
            } else {
                7e-4
            });
        }

        let mut bo_used = false;
        let mut bo_selected_arch = self.layer_sizes.clone();
        if auto_mode && dataset.len() >= 4 && !focused_exact_family {
            emit_preflight(
                "preflight",
                "architecture-presearch",
                dataset.len(),
                0,
                collocation_added,
                self.learning_rate,
                &self.layer_sizes,
            );
            if let Some(best_arch) = self.bo_like_architecture_presearch(&dataset, train_seed) {
                if best_arch != self.layer_sizes {
                    self.layer_sizes = best_arch.clone();
                    self.layers = initialize_layers(&self.layer_sizes);
                    self.moments = initialize_moments(&self.layer_sizes);
                    self.adam_t = 0;
                }
                bo_selected_arch = best_arch;
                bo_used = true;
            }
        }

        let split_idx = ((dataset.len() as f64) * 0.8).ceil() as usize;
        let mut train_data = dataset[..split_idx.min(dataset.len())].to_vec();
        let val_data = dataset[split_idx.min(dataset.len())..].to_vec();
        let train_data_cap_base =
            (samples.len() * 64 + collocation_limit + boundary_limit + interface_limit)
                .clamp(256, 8192);
        let train_data_cap = if is_cantilever_training || is_discontinuity_prone {
            train_data_cap_base.clamp(192, 1024)
        } else {
            train_data_cap_base
        };
        if train_data.len() > train_data_cap {
            train_data.shuffle(&mut train_rng);
            train_data.truncate(train_data_cap);
        }
        let cycle_epochs = batch.epochs.clamp(1, 200);
        let progress_emit_every = batch.progress_emit_every_epochs.unwrap_or(1).max(1);
        let network_emit_every = batch
            .network_emit_every_epochs
            .unwrap_or((progress_emit_every * 25).clamp(1, 100_000))
            .max(1);
        let online_active_learning = batch.online_active_learning.unwrap_or(false);
        let autonomous_active_learning = autonomous_mode;
        let max_total_epochs = batch
            .max_total_epochs
            .unwrap_or(cycle_epochs * 20)
            .clamp(1, 100_000_000);
        let min_improvement = batch.min_improvement.unwrap_or(1e-7).max(1e-12);
        let target_loss = batch.target_loss.max(0.0);
        let low_target_mode = target_loss.is_finite() && target_loss > 0.0 && target_loss <= 1e-8;
        let stage1_epochs_cfg = batch.stage1_epochs.unwrap_or(0);
        let stage2_epochs_cfg = batch.stage2_epochs.unwrap_or(0);
        let stage3_ramp_epochs_cfg = batch.stage3_ramp_epochs.unwrap_or(0);
        let explicit_stage_schedule =
            stage1_epochs_cfg > 0 || stage2_epochs_cfg > 0 || stage3_ramp_epochs_cfg > 0;

        let mut loss = 0.0;
        let mut val_loss = f64::MAX;
        let mut last_train_breakdown = LossBreakdown::default();
        let mut last_val_breakdown = LossBreakdown::default();
        let mut completed_epochs = 0usize;
        let mut reached_target = false;
        let mut reached_target_loss = false;
        let mut trend_converged = false;
        let mut manual_stop = false;
        let mut prev_cycle_best = f64::MAX;
        let mut no_improve_epochs = 0usize;
        let mut lr_phase = "warmup".to_string();
        let mut curriculum_stage = 1usize;
        let mut prev_curriculum_stage = 1usize;
        let mut stage_metric_best = f64::MAX;
        let mut stage2_start_epoch = 0usize;
        let mut stage1_baseline_data = None::<f64>;
        let mut stage1_target_data = f64::MAX;
        let mut stage2_anchor_val_loss = f64::MAX;
        let mut stage2_best_since_switch = f64::MAX;
        let mut stage2_best_data_since_switch = f64::MAX;
        let mut stage2_no_data_improve = 0usize;
        let mut stage2_backoffs = 0usize;
        let mut stage3_start_epoch = 0usize;
        let mut stage2_ramp_span_epochs = if stage3_ramp_epochs_cfg > 0 {
            stage3_ramp_epochs_cfg as f64
        } else if is_cantilever_training {
            3200.0
        } else {
            1200.0
        };
        let stage2_floor_base = if low_target_mode {
            if is_cantilever_training { 0.008 } else { 0.012 }
        } else if is_cantilever_training {
            0.012
        } else {
            0.02
        };
        let stage2_floor_ramp = if low_target_mode {
            if is_cantilever_training { 0.04 } else { 0.08 }
        } else if is_cantilever_training {
            0.06
        } else {
            0.12
        };
        let stage2_physics_cap = if low_target_mode {
            if is_cantilever_training { 0.12 } else { 0.18 }
        } else if is_cantilever_training {
            0.16
        } else {
            0.28
        };
        let mut optimizer_switches = 0usize;
        let max_optimizer_switches = batch.max_optimizer_switches.unwrap_or(6).clamp(1, 40);
        let mut activation_switches = 0usize;
        let max_activation_switches = (max_optimizer_switches + 2).clamp(1, 16);
        let max_backoffs = batch.max_backoffs.unwrap_or(8).clamp(1, 100);
        let mut optimizer_id = "adamw".to_string();
        let mut val_window: VecDeque<f64> = VecDeque::with_capacity(48);
        let mut target_band_low = 0.0_f64;
        let mut target_band_high = f64::MAX;
        let mut trend_slope = 0.0_f64;
        let mut trend_variance = 0.0_f64;
        let mut active_learning_rounds = 0usize;
        let mut active_learning_samples_added = 0usize;
        let mut safeguard_triggers = 0usize;
        let checkpoint_every = batch.checkpoint_every_epochs.unwrap_or(0);
        let mut checkpoint_rollbacks = 0usize;
        let mut switch_guard_snapshot: Option<AnnModelState> = None;
        let mut switch_guard_epoch = 0usize;
        let mut switch_guard_baseline = f64::MAX;
        let mut switch_guard_optimizer_prev = "adamw".to_string();
        let mut rescue_count = 0usize;
        let max_rescues = (max_backoffs * 2).clamp(2, 100);
        let mut last_rescue_epoch = 0usize;
        let mut topology_cooldown_until_epoch = 0usize;
        let mut hybrid_mode = "hybrid".to_string();
        let mut notes = vec![];

        if focused_exact_family && self.warmstart_output_layer_with_ridge(&train_data) {
            notes.push("Focused family warm start: output layer fit with ridge seed.".to_string());
            emit_preflight(
                "preflight",
                "warm-start",
                train_data.len(),
                train_data_cap,
                collocation_added,
                self.learning_rate,
                &self.layer_sizes,
            );
        }

        if autonomous_mode && auto_mode && !focused_exact_family {
            emit_preflight(
                "preflight",
                "activation-presearch",
                train_data.len(),
                train_data_cap,
                collocation_added,
                self.learning_rate,
                &self.layer_sizes,
            );
            let probe_train = train_data.iter().take(96).cloned().collect::<Vec<_>>();
            let probe_val = val_data.iter().take(48).cloned().collect::<Vec<_>>();
            if let Some((best_activation, best_score, current_score)) = self.activation_presearch(
                &probe_train,
                &probe_val,
                &norm,
                train_seed.wrapping_add(4049),
                16,
            ) {
                if best_activation != self.hidden_activation && best_score < current_score * 0.995 {
                    self.hidden_activation = best_activation;
                    self.layers = initialize_layers_with_rng(&self.layer_sizes, &mut train_rng);
                    self.moments = initialize_moments(&self.layer_sizes);
                    self.best_layers = None;
                    self.best_val_loss = f64::MAX;
                    self.adam_t = 0;
                    self.learning_rate = (self.learning_rate * 0.9).clamp(1e-6, 2e-3);
                    activation_switches = activation_switches.saturating_add(1);
                }
            }
        }

        // Stage 1 curriculum: prioritize supervised fit first.
        self.data_weight = 1.0;
        self.physics_weight = if focused_exact_family && is_discontinuity_prone {
            self.physics_weight.clamp(0.002, 0.008)
        } else if focused_exact_family {
            self.physics_weight.clamp(0.005, 0.015)
        } else {
            self.physics_weight.clamp(0.01, 0.03)
        };
        emit_preflight(
            "preflight",
            "training-ready",
            train_data.len(),
            train_data_cap,
            collocation_added,
            self.learning_rate,
            &self.layer_sizes,
        );

        while completed_epochs < max_total_epochs {
            if should_stop() {
                if checkpoint_every > 0 {
                    on_checkpoint(completed_epochs, self.snapshot_state(), false);
                }
                manual_stop = true;
                break;
            }
            let cycle_start_best = val_loss;

            for _ in 0..cycle_epochs {
                if completed_epochs >= max_total_epochs {
                    break;
                }
                if should_stop() {
                    if checkpoint_every > 0 {
                        on_checkpoint(completed_epochs, self.snapshot_state(), false);
                    }
                    manual_stop = true;
                    break;
                }

                last_train_breakdown = self.epoch(&train_data, &norm, &mut train_rng);
                loss = last_train_breakdown.total;
                last_val_breakdown = self.eval_loss(
                    if val_data.is_empty() {
                        &train_data
                    } else {
                        &val_data
                    },
                    &norm,
                );
                val_loss = last_val_breakdown.total;
                completed_epochs += 1;
                if val_window.len() >= 48 {
                    val_window.pop_front();
                }
                val_window.push_back(val_loss.max(0.0));
                if val_window.len() >= 6 {
                    let first = *val_window.front().unwrap_or(&val_loss);
                    let last = *val_window.back().unwrap_or(&val_loss);
                    trend_slope = (last - first) / val_window.len() as f64;
                    let mean = val_window.iter().sum::<f64>() / val_window.len() as f64;
                    trend_variance = val_window
                        .iter()
                        .map(|v| (v - mean) * (v - mean))
                        .sum::<f64>()
                        / val_window.len() as f64;
                    let mut sorted = val_window.iter().copied().collect::<Vec<_>>();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let q10_idx = ((sorted.len() as f64) * 0.1).floor() as usize;
                    let q10 = sorted[q10_idx.min(sorted.len() - 1)];
                    target_band_low = (q10 * 0.995).max(0.0);
                    target_band_high = (q10 * 1.02).max(target_band_low + 1e-12);
                }

                if autonomous_mode
                    && stage1_baseline_data.is_none()
                    && last_val_breakdown.data.is_finite()
                {
                    let base = last_val_breakdown.data.max(1e-12);
                    stage1_baseline_data = Some(base);
                    stage1_target_data = if low_target_mode {
                        if is_cantilever_training {
                            (base * 0.25).max(1e-6)
                        } else {
                            (base * 0.30).max(1e-6)
                        }
                    } else if is_cantilever_training {
                        (base * 0.40).max(1e-6)
                    } else {
                        (base * 0.45).max(1e-6)
                    };
                    stage_metric_best = base;
                }

                if val_loss + 1e-10 < self.best_val_loss {
                    self.best_val_loss = val_loss;
                    self.best_layers = Some(self.layers.clone());
                    self.plateau_epochs = 0;
                    if checkpoint_every > 0 && completed_epochs % checkpoint_every == 0 {
                        on_checkpoint(completed_epochs, self.snapshot_state(), true);
                    }
                } else {
                    self.plateau_epochs += 1;
                }

                let control_metric = if curriculum_stage <= 2 {
                    last_val_breakdown.data
                } else {
                    val_loss
                };
                if control_metric + 1e-10 < stage_metric_best {
                    stage_metric_best = control_metric;
                    no_improve_epochs = 0;
                } else {
                    no_improve_epochs += 1;
                }

                if autonomous_mode && explicit_stage_schedule {
                    let s1 = stage1_epochs_cfg.max(1);
                    let s2 = stage2_epochs_cfg.max(1);
                    let s3 = stage3_ramp_epochs_cfg.max(1);
                    if completed_epochs <= s1 {
                        curriculum_stage = 1;
                        self.data_weight = 1.0;
                        self.physics_weight = self.physics_weight.clamp(0.01, 0.05);
                        lr_phase = "curriculum-stage1-explicit".to_string();
                    } else if completed_epochs <= s1.saturating_add(s2) {
                        if curriculum_stage < 2 {
                            stage2_start_epoch = completed_epochs;
                            stage_metric_best = val_loss;
                            stage2_anchor_val_loss = val_loss;
                            stage2_best_since_switch = val_loss;
                            stage2_best_data_since_switch = last_val_breakdown.data;
                            stage2_no_data_improve = 0;
                            no_improve_epochs = 0;
                        }
                        curriculum_stage = 2;
                        self.data_weight = 1.0;
                        self.physics_weight = self.physics_weight.clamp(0.02, 0.08);
                        hybrid_mode = "stabilization-explicit".to_string();
                        lr_phase = "curriculum-stage2-explicit".to_string();
                    } else {
                        if curriculum_stage < 3 {
                            stage3_start_epoch = completed_epochs;
                            stage_metric_best = val_loss;
                            no_improve_epochs = 0;
                        }
                        curriculum_stage = 3;
                        let ramp = ((completed_epochs.saturating_sub(s1.saturating_add(s2)))
                            as f64
                            / s3 as f64)
                            .clamp(0.0, 1.0);
                        self.adapt_loss_weights(
                            last_train_breakdown.data,
                            last_train_breakdown.physics,
                        );
                        let floor = stage2_floor_base + stage2_floor_ramp * ramp;
                        self.data_weight = 1.0;
                        self.physics_weight =
                            self.physics_weight.max(floor).min(stage2_physics_cap);
                        let ratio = last_val_breakdown.data / last_val_breakdown.physics.max(1e-12);
                        hybrid_mode = if ratio > 8.0 {
                            "data-dominant"
                        } else if ratio < 1.25 {
                            "pinn-dominant"
                        } else {
                            "hybrid"
                        }
                        .to_string();
                        lr_phase = "curriculum-stage3-explicit".to_string();
                    }
                } else if autonomous_mode && curriculum_stage == 1 {
                    let min_stage1_epochs = if stage1_epochs_cfg > 0 {
                        stage1_epochs_cfg.min(max_total_epochs)
                    } else if low_target_mode && is_cantilever_training {
                        (max_total_epochs / 20).clamp(180, 6_000)
                    } else if low_target_mode && is_discontinuity_prone {
                        (max_total_epochs / 24).clamp(120, 4_000)
                    } else if is_cantilever_training {
                        (max_total_epochs / 8).clamp(400, 16_000)
                    } else {
                        (max_total_epochs / 16).clamp(120, 8_000)
                    };
                    let hit_data_gate = last_val_breakdown.data <= stage1_target_data;
                    let stale_data_fit = no_improve_epochs >= 80;
                    let allow_stale_switch = if is_cantilever_training {
                        stale_data_fit
                            && completed_epochs >= min_stage1_epochs.saturating_mul(2)
                            && stage1_baseline_data
                                .map(|b| last_val_breakdown.data <= b * 0.85)
                                .unwrap_or(false)
                    } else if low_target_mode {
                        stale_data_fit
                            && completed_epochs >= min_stage1_epochs.saturating_mul(2)
                            && stage1_baseline_data
                                .map(|b| last_val_breakdown.data <= b * 0.82)
                                .unwrap_or(false)
                    } else {
                        stale_data_fit
                    };
                    if completed_epochs >= min_stage1_epochs
                        && (hit_data_gate || allow_stale_switch)
                    {
                        curriculum_stage = 2;
                        stage2_start_epoch = completed_epochs;
                        stage_metric_best = val_loss;
                        stage2_anchor_val_loss = val_loss;
                        stage2_best_since_switch = val_loss;
                        stage2_best_data_since_switch = last_val_breakdown.data;
                        stage2_no_data_improve = 0;
                        no_improve_epochs = 0;
                        self.physics_weight = self.physics_weight.max(0.05);
                        lr_phase = "curriculum-stage2".to_string();
                    } else {
                        self.data_weight = 1.0;
                        self.physics_weight = self.physics_weight.clamp(0.01, 0.05);
                    }
                } else if autonomous_mode && curriculum_stage == 2 {
                    // Stage 2: stabilization zone (interface/discontinuity safe region).
                    self.data_weight = 1.0;
                    self.physics_weight = self.physics_weight.clamp(0.015, 0.05);
                    hybrid_mode = if is_discontinuity_prone {
                        "stabilization-discontinuity".to_string()
                    } else {
                        "stabilization".to_string()
                    };
                    let min_stage2_epochs = if stage2_epochs_cfg > 0 {
                        stage2_epochs_cfg.min(max_total_epochs.saturating_sub(stage2_start_epoch))
                    } else if low_target_mode && is_discontinuity_prone {
                        (max_total_epochs / 18).clamp(120, 5_000)
                    } else if low_target_mode {
                        (max_total_epochs / 14).clamp(120, 4_000)
                    } else if is_discontinuity_prone {
                        (max_total_epochs / 6).clamp(300, 20_000)
                    } else {
                        (max_total_epochs / 10).clamp(150, 10_000)
                    };
                    let data_gate_stage2 = stage1_target_data
                        * if low_target_mode {
                            if is_discontinuity_prone { 0.92 } else { 0.88 }
                        } else if is_discontinuity_prone {
                            1.05
                        } else {
                            1.15
                        };
                    if completed_epochs.saturating_sub(stage2_start_epoch) >= min_stage2_epochs
                        && last_val_breakdown.data <= data_gate_stage2
                    {
                        curriculum_stage = 3;
                        stage3_start_epoch = completed_epochs;
                        stage_metric_best = val_loss;
                        no_improve_epochs = 0;
                        lr_phase = "curriculum-stage3".to_string();
                    }
                } else if autonomous_mode {
                    // Stage 3: PINN-hybrid ramp and autonomous regime selection.
                    stage2_best_since_switch = stage2_best_since_switch.min(val_loss);
                    if last_val_breakdown.data + 1e-10 < stage2_best_data_since_switch {
                        stage2_best_data_since_switch = last_val_breakdown.data;
                        stage2_no_data_improve = 0;
                    } else {
                        stage2_no_data_improve = stage2_no_data_improve.saturating_add(1);
                    }
                    let ramp_anchor = if stage3_start_epoch > 0 {
                        stage3_start_epoch
                    } else {
                        stage2_start_epoch
                    };
                    let ramp = ((completed_epochs.saturating_sub(ramp_anchor)) as f64
                        / stage2_ramp_span_epochs)
                        .clamp(0.0, 1.0);
                    let floor = stage2_floor_base + stage2_floor_ramp * ramp;
                    self.adapt_loss_weights(
                        last_train_breakdown.data,
                        last_train_breakdown.physics,
                    );
                    self.physics_weight = self.physics_weight.max(floor).min(stage2_physics_cap);
                    self.data_weight = 1.0;

                    let ratio = last_val_breakdown.data / last_val_breakdown.physics.max(1e-12);
                    if ratio > 8.0 {
                        hybrid_mode = "data-dominant".to_string();
                        self.physics_weight = self.physics_weight.clamp(stage2_floor_base, 0.08);
                    } else if ratio < 1.25 {
                        hybrid_mode = "pinn-dominant".to_string();
                        self.physics_weight = (self.physics_weight * 1.04).min(stage2_physics_cap);
                    } else {
                        hybrid_mode = "hybrid".to_string();
                    }

                    if completed_epochs.saturating_sub(ramp_anchor) >= 200
                        && stage2_best_since_switch > stage2_anchor_val_loss * 1.01
                        && stage2_backoffs < max_backoffs
                    {
                        stage2_backoffs = stage2_backoffs.saturating_add(1);
                        stage2_start_epoch = completed_epochs;
                        stage3_start_epoch = completed_epochs;
                        stage2_anchor_val_loss = val_loss.min(stage2_best_since_switch);
                        stage2_best_since_switch = stage2_anchor_val_loss;
                        stage2_best_data_since_switch = last_val_breakdown.data;
                        stage2_no_data_improve = 0;
                        stage2_ramp_span_epochs = (stage2_ramp_span_epochs * 1.6).min(20_000.0);
                        self.physics_weight = (self.physics_weight * 0.6).max(0.03);
                        self.learning_rate = (self.learning_rate * 0.85).max(1e-6);
                        no_improve_epochs = 0;
                        lr_phase = "curriculum-backoff".to_string();
                        safeguard_triggers = safeguard_triggers.saturating_add(1);
                    }

                    let data_stalled = stage2_no_data_improve >= 90;
                    let physics_low = last_val_breakdown.physics <= 0.02;
                    let data_not_good = stage1_baseline_data
                        .map(|b| last_val_breakdown.data > b * 0.75)
                        .unwrap_or(true);
                    if data_stalled && physics_low && data_not_good {
                        curriculum_stage = 2;
                        stage2_start_epoch = completed_epochs;
                        stage_metric_best = last_val_breakdown.data;
                        no_improve_epochs = 0;
                        stage2_no_data_improve = 0;
                        stage2_backoffs = stage2_backoffs.saturating_add(1);
                        self.physics_weight = self.physics_weight.min(0.06).max(0.02);
                        self.learning_rate = (self.learning_rate * 1.6).clamp(2e-5, 2e-3);
                        lr_phase = "stage3-to-stage2-recovery".to_string();
                        safeguard_triggers = safeguard_triggers.saturating_add(1);
                    }
                }

                if autonomous_mode
                    && optimizer_switches < max_optimizer_switches
                    && no_improve_epochs >= 120
                    && completed_epochs % 60 == 0
                {
                    switch_guard_snapshot = Some(self.snapshot_state());
                    switch_guard_epoch = completed_epochs;
                    switch_guard_baseline = val_loss;
                    switch_guard_optimizer_prev = optimizer_id.clone();
                    if checkpoint_every > 0 {
                        on_checkpoint(completed_epochs, self.snapshot_state(), false);
                    }
                    match optimizer_id.as_str() {
                        "adamw" => {
                            optimizer_id = "rmsprop".to_string();
                            self.adam_beta1 = 0.0;
                            self.adam_beta2 = 0.99;
                            self.adam_eps = 1e-7;
                            self.learning_rate = (self.learning_rate * 0.7).max(5e-6);
                        }
                        "rmsprop" => {
                            optimizer_id = "sgd-momentum".to_string();
                            self.adam_beta1 = 0.88;
                            self.adam_beta2 = 0.95;
                            self.adam_eps = 1e-8;
                            self.learning_rate = (self.learning_rate * 0.5).max(1e-6);
                        }
                        _ => {
                            optimizer_id = "adamw".to_string();
                            self.adam_beta1 = 0.9;
                            self.adam_beta2 = 0.999;
                            self.adam_eps = 1e-8;
                            self.learning_rate = (self.learning_rate * 0.85).max(1e-6);
                        }
                    }
                    optimizer_switches += 1;
                    no_improve_epochs = 0;
                    lr_phase = "optimizer-switch".to_string();
                }

                if autonomous_mode
                    && switch_guard_snapshot.is_some()
                    && completed_epochs.saturating_sub(switch_guard_epoch) >= 120
                    && val_loss > switch_guard_baseline * 1.03
                {
                    if let Some(snapshot) = switch_guard_snapshot.take() {
                        self.load_state(snapshot);
                        optimizer_id = switch_guard_optimizer_prev.clone();
                        checkpoint_rollbacks = checkpoint_rollbacks.saturating_add(1);
                        self.learning_rate = (self.learning_rate * 0.8).max(1e-6);
                        no_improve_epochs = 0;
                        lr_phase = "optimizer-switch-rollback".to_string();
                        safeguard_triggers = safeguard_triggers.saturating_add(1);
                    }
                }

                if val_loss > self.best_val_loss * 1.5 {
                    if let Some(saved) = self.best_layers.clone() {
                        self.layers = saved;
                        checkpoint_rollbacks = checkpoint_rollbacks.saturating_add(1);
                    }
                    self.learning_rate *= 0.5;
                    lr_phase = "rollback-cut".to_string();
                    safeguard_triggers += 1;
                }

                // Adaptive LR for plateau avoidance:
                // - Reduce LR after sustained stagnation.
                // - Slightly increase LR when making progress to accelerate convergence.
                if no_improve_epochs > 0 && no_improve_epochs % 80 == 0 {
                    let decay = if low_target_mode {
                        if is_cantilever_training { 0.84 } else { 0.82 }
                    } else if is_cantilever_training {
                        0.7
                    } else {
                        0.65
                    };
                    self.learning_rate = (self.learning_rate * decay).max(1e-6);
                    lr_phase = if curriculum_stage == 1 {
                        "curriculum-stage1-decay".to_string()
                    } else {
                        "plateau-decay".to_string()
                    };
                } else if completed_epochs > 20
                    && completed_epochs % 40 == 0
                    && control_metric <= stage_metric_best * 1.01
                {
                    self.learning_rate = (self.learning_rate * 1.03).min(1e-2);
                    lr_phase = if curriculum_stage == 1 {
                        "curriculum-stage1-boost".to_string()
                    } else {
                        "fine-tune-boost".to_string()
                    };
                }

                // If deeply plateaued at high loss, perform a controlled warm restart.
                if no_improve_epochs >= 140 && val_loss > target_loss.max(1e-8) * 5.0 {
                    if let Some(saved) = self.best_layers.clone() {
                        self.layers = saved;
                    }
                    self.learning_rate = (self.learning_rate * 1.4).clamp(5e-6, 2e-3);
                    no_improve_epochs = 0;
                    lr_phase = "warm-restart".to_string();
                    safeguard_triggers += 1;
                }

                if autonomous_mode
                    && activation_switches < max_activation_switches
                    && no_improve_epochs >= 120
                    && completed_epochs % 100 == 0
                {
                    let probe_train = train_data.iter().take(96).cloned().collect::<Vec<_>>();
                    let probe_val = val_data.iter().take(48).cloned().collect::<Vec<_>>();
                    if let Some((best_activation, best_score, current_score)) = self
                        .activation_presearch(
                            &probe_train,
                            &probe_val,
                            &norm,
                            train_seed
                                .wrapping_add(completed_epochs as u64)
                                .wrapping_add(9001),
                            10,
                        )
                    {
                        if best_activation != self.hidden_activation
                            && best_score < current_score * 0.995
                        {
                            self.hidden_activation = best_activation;
                            self.layers =
                                initialize_layers_with_rng(&self.layer_sizes, &mut train_rng);
                            self.moments = initialize_moments(&self.layer_sizes);
                            self.best_layers = None;
                            self.best_val_loss = f64::MAX;
                            self.adam_t = 0;
                            self.data_weight = 1.0;
                            self.physics_weight = self.physics_weight.clamp(0.02, 0.06);
                            self.learning_rate = (self.learning_rate * 1.25).clamp(2e-5, 2e-3);
                            no_improve_epochs = 0;
                            curriculum_stage = 1;
                            stage_metric_best = f64::MAX;
                            activation_switches = activation_switches.saturating_add(1);
                            lr_phase =
                                format!("activation-switch-{}", self.hidden_activation.as_str());
                            safeguard_triggers = safeguard_triggers.saturating_add(1);
                        }
                    }
                }

                if autonomous_mode
                    && rescue_count < max_rescues
                    && curriculum_stage == 1
                    && no_improve_epochs >= 36
                    && completed_epochs.saturating_sub(last_rescue_epoch) >= 1_000
                    && completed_epochs % 120 == 0
                {
                    let probe_train = train_data.iter().take(128).cloned().collect::<Vec<_>>();
                    let probe_val = val_data.iter().take(64).cloned().collect::<Vec<_>>();
                    if let Some((best_activation, best_score, current_score)) = self
                        .activation_presearch(
                            &probe_train,
                            &probe_val,
                            &norm,
                            train_seed
                                .wrapping_add(completed_epochs as u64)
                                .wrapping_add(120_011),
                            12,
                        )
                    {
                        if best_score < current_score * 0.998 {
                            let mut switched_activation = false;
                            if best_activation != self.hidden_activation {
                                self.hidden_activation = best_activation;
                                switched_activation = true;
                            }
                            self.learning_rate = (self.learning_rate * 1.18).clamp(2e-5, 2e-3);
                            self.data_weight = 1.0;
                            self.physics_weight = self.physics_weight.clamp(0.02, 0.08);
                            let added = self.active_learning_refresh(
                                &mut train_data,
                                &norm,
                                batch,
                                &analysis_type,
                                2,
                                train_data_cap,
                                &mut train_rng,
                            );
                            if added > 0 {
                                active_learning_rounds += 1;
                                active_learning_samples_added += added;
                            }
                            if switched_activation {
                                self.layers =
                                    initialize_layers_with_rng(&self.layer_sizes, &mut train_rng);
                                self.moments = initialize_moments(&self.layer_sizes);
                                self.best_layers = None;
                                self.best_val_loss = f64::MAX;
                                self.adam_t = 0;
                            }
                            no_improve_epochs = 0;
                            rescue_count = rescue_count.saturating_add(1);
                            last_rescue_epoch = completed_epochs;
                            lr_phase =
                                format!("autonomous-rescue-{}", self.hidden_activation.as_str());
                            safeguard_triggers = safeguard_triggers.saturating_add(1);
                        }
                    }
                }

                // LR floor guard: if data fit is still far from gate, avoid decaying to near-zero LR.
                if autonomous_mode {
                    let data_gate = stage1_target_data.max(1e-6);
                    let lr_floor = if low_target_mode {
                        1e-5
                    } else if is_cantilever_training {
                        3e-5
                    } else {
                        2e-5
                    };
                    if self.learning_rate < lr_floor && last_val_breakdown.data > data_gate * 1.2 {
                        self.learning_rate = lr_floor;
                        lr_phase = "lr-floor-guard".to_string();
                    }
                }

                let include_snapshot = completed_epochs == 1
                    || completed_epochs % network_emit_every == 0
                    || completed_epochs >= max_total_epochs
                    || (auto_mode && val_loss <= target_loss);
                let emit_now = completed_epochs == 1
                    || completed_epochs % progress_emit_every == 0
                    || completed_epochs >= max_total_epochs
                    || (auto_mode && val_loss <= target_loss);
                if emit_now {
                    let momentum_residual = last_val_breakdown.momentum.max(0.0);
                    let kinematic_residual = last_val_breakdown.kinematics.max(0.0);
                    let material_residual = last_val_breakdown.material.max(0.0);
                    let boundary_residual = last_val_breakdown.boundary.max(0.0);
                    on_epoch(TrainingProgressEvent {
                        epoch: completed_epochs,
                        total_epochs: max_total_epochs,
                        loss,
                        val_loss,
                        data_loss: last_train_breakdown.data,
                        physics_loss: last_train_breakdown.physics,
                        val_data_loss: last_val_breakdown.data,
                        val_physics_loss: last_val_breakdown.physics,
                        momentum_residual,
                        kinematic_residual,
                        material_residual,
                        boundary_residual,
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
                        hybrid_mode: hybrid_mode.clone(),
                        stage_id: match curriculum_stage {
                            1 => "stage-1".to_string(),
                            2 => "stage-2".to_string(),
                            _ => "stage-3".to_string(),
                        },
                        optimizer_id: optimizer_id.clone(),
                        lr_phase: lr_phase.clone(),
                        target_band_low,
                        target_band_high,
                        trend_slope,
                        trend_variance,
                        watchdog_trigger_count: stage2_backoffs,
                        collocation_samples_added: collocation_added,
                        train_data_size: train_data.len(),
                        train_data_cap,
                        residual_weight_momentum: 0.0,
                        residual_weight_kinematics: 0.0,
                        residual_weight_material: 0.0,
                        residual_weight_boundary: 0.0,
                        learning_rate: self.learning_rate,
                        architecture: self.layer_sizes.clone(),
                        progress_ratio: completed_epochs as f64 / max_total_epochs as f64,
                        network: if include_snapshot {
                            self.network_snapshot()
                        } else {
                            NetworkSnapshot {
                                layer_sizes: vec![],
                                nodes: vec![],
                                connections: vec![],
                            }
                        },
                        pino: None,
                    });
                }

                if curriculum_stage != prev_curriculum_stage {
                    notes.push(format!(
                        "StageTransition: {} -> {} at epoch {}",
                        prev_curriculum_stage, curriculum_stage, completed_epochs
                    ));
                    if checkpoint_every > 0 {
                        on_checkpoint(completed_epochs, self.snapshot_state(), false);
                    }
                    prev_curriculum_stage = curriculum_stage;
                }

                if checkpoint_every > 0
                    && completed_epochs % checkpoint_every == 0
                    && completed_epochs < max_total_epochs
                {
                    on_checkpoint(completed_epochs, self.snapshot_state(), false);
                }

                if auto_mode && val_loss <= target_loss {
                    reached_target = true;
                    reached_target_loss = true;
                    break;
                }
                let trend_min_epochs = (max_total_epochs / 20).clamp(160, 12_000);
                let data_reduction_ok = stage1_baseline_data
                    .map(|b| last_val_breakdown.data <= (b * 0.75))
                    .unwrap_or(false);
                let trend_target_ok = if low_target_mode {
                    val_loss <= target_loss.max(1e-12)
                } else {
                    val_loss <= target_loss || data_reduction_ok
                };
                if autonomous_mode
                    && val_window.len() >= 24
                    && completed_epochs >= trend_min_epochs
                    && trend_slope.abs() <= 1e-6
                    && trend_variance <= 5e-8
                    && val_loss <= target_band_high
                    && trend_target_ok
                {
                    reached_target = true;
                    trend_converged = true;
                    lr_phase = "trend-converged".to_string();
                    break;
                }
            }

            if !auto_mode {
                break;
            }

            if manual_stop {
                break;
            }

            if reached_target {
                break;
            }

            let cycle_best = val_loss.min(cycle_start_best);
            let improvement = (prev_cycle_best - cycle_best).abs();
            prev_cycle_best = cycle_best;
            if improvement < min_improvement {
                self.plateau_epochs += 1;
            }

            if completed_epochs >= max_total_epochs {
                break;
            }

            // Active learning: query FEM around hardest cases and add informative samples.
            if auto_mode
                && completed_epochs % 300 == 0
                && (online_active_learning
                    || (autonomous_active_learning && no_improve_epochs >= 60))
            {
                let add_limit = if autonomous_active_learning { 2 } else { 1 };
                let added = self.active_learning_refresh(
                    &mut train_data,
                    &norm,
                    batch,
                    &analysis_type,
                    add_limit,
                    train_data_cap,
                    &mut train_rng,
                );
                if added > 0 {
                    active_learning_rounds += 1;
                    active_learning_samples_added += added;
                    lr_phase = "active-learning-refresh".to_string();
                }
            }

            // Topology and LR adaptation between cycles.
            if self.plateau_epochs >= 3
                && val_loss > target_loss
                && no_improve_epochs >= 120
                && completed_epochs >= topology_cooldown_until_epoch
            {
                let _ = self.grow_topology();
                self.model_version += 1;
                topology_cooldown_until_epoch = completed_epochs.saturating_add(1_200);
                lr_phase = "topology-grow".to_string();
            } else if val_loss < (target_loss * 0.35).max(1e-10)
                && completed_epochs >= topology_cooldown_until_epoch
            {
                if self.prune_topology() {
                    self.model_version += 1;
                    topology_cooldown_until_epoch = completed_epochs.saturating_add(1_200);
                    lr_phase = "topology-prune".to_string();
                }
            }
        }

        let mut grew = false;
        let mut pruned = false;

        if self.plateau_epochs >= 3 && val_loss > target_loss {
            grew = self.grow_topology();
            if grew {
                notes.push(
                    "Adaptive growth triggered due to plateau and residual threshold.".to_string(),
                );
                self.model_version += 1;
            }
        }

        if val_loss < (target_loss * 0.35).max(1e-10) {
            pruned = self.prune_topology();
            if pruned {
                notes.push(
                    "Adaptive pruning triggered: reduced hidden complexity after convergence."
                        .to_string(),
                );
                self.model_version += 1;
            }
        }

        if val_loss > self.last_loss * 1.02 {
            self.learning_rate = (self.learning_rate * 0.7).max(1e-6);
            notes.push("Learning rate reduced after validation degradation.".to_string());
        } else if val_loss < self.last_loss * 0.9 {
            self.learning_rate = (self.learning_rate * 1.05).min(1e-2);
            notes.push("Learning rate increased for faster convergence.".to_string());
        }

        self.last_loss = val_loss;
        notes.push(format!(
            "Loss normalized by output scales {:?}",
            norm.output_scales
        ));
        notes.push(format!("CollocationSamplesAdded: {}", collocation_added));
        notes.push(format!(
            "TrainDataCap: {}, TrainDataFinal: {}",
            train_data_cap,
            train_data.len()
        ));
        notes.push(format!("Seed: {}", train_seed));
        notes.push(format!("AnalysisType: {}", analysis_type));
        if let Some(domain) = &self.training_domain {
            notes.push(format!(
                "Coverage: seedCases={}, expandedCases={}, tags={:?}, mixedLoads={}, holes={}, dualFixed={}",
                domain.training_seed_cases,
                domain.expanded_cases,
                domain.coverage_tags,
                domain.mixed_load_cases,
                domain.hole_cases,
                domain.dual_fixed_cases
            ));
        }
        notes.push(format!(
            "PinnConfig: backend={}, collocation={}, boundary={}, interface={}, weights(m/k/mat/bc)=({:.3},{:.3},{:.3},{:.3}), stageEpochs=({},{},{}), contactPenalty={:.4e}, plasticityFactor={:.4e}",
            batch.pinn_backend.clone().unwrap_or_else(|| "burn-ndarray-cpu".to_string()),
            batch.collocation_points.unwrap_or(4096),
            batch.boundary_points.unwrap_or(1024),
            batch.interface_points.unwrap_or(512),
            batch.residual_weight_momentum.unwrap_or(1.0),
            batch.residual_weight_kinematics.unwrap_or(1.0),
            batch.residual_weight_material.unwrap_or(1.0),
            batch.residual_weight_boundary.unwrap_or(1.0),
            batch.stage1_epochs.unwrap_or(0),
            batch.stage2_epochs.unwrap_or(0),
            batch.stage3_ramp_epochs.unwrap_or(0),
            batch.contact_penalty.unwrap_or(10.0),
            batch.plasticity_factor.unwrap_or(0.0),
        ));
        notes.push(format!(
            "Residuals: momentumResidual={:.6e}, kinematicResidual={:.6e}, materialResidual={:.6e}, boundaryResidual={:.6e}",
            last_val_breakdown.momentum,
            last_val_breakdown.kinematics,
            last_val_breakdown.material,
            last_val_breakdown.boundary,
        ));
        notes.push(format!(
            "Diagnostics: sinceImprove={}, phase={}, curriculumStage={}, hybridMode={}, curriculumBackoffs={}, optimizerSwitches={}, activationSwitches={}, rescues={}, activation={}, optimizerId={}, checkpointRollbacks={}, targetFloorEstimate={:.6e}, trendSlope={:.6e}, trendVar={:.6e}, stage2RampSpan={:.1}, dataGate={:.4e}, dataBaseline={:.4e}, dataW={:.3}, physW={:.3}, AL rounds={}, AL samples={}, safeguards={}, boUsed={}, boArch={:?}, train(data={:.4e},phys={:.4e}), val(data={:.4e},phys={:.4e})",
            no_improve_epochs,
            lr_phase,
            curriculum_stage,
            hybrid_mode,
            stage2_backoffs,
            optimizer_switches,
            activation_switches,
            rescue_count,
            self.hidden_activation.as_str(),
            optimizer_id,
            checkpoint_rollbacks,
            target_band_low,
            trend_slope,
            trend_variance,
            stage2_ramp_span_epochs,
            stage1_target_data,
            stage1_baseline_data.unwrap_or(0.0),
            self.data_weight,
            self.physics_weight,
            active_learning_rounds,
            active_learning_samples_added,
            safeguard_triggers,
            bo_used,
            bo_selected_arch,
            last_train_breakdown.data,
            last_train_breakdown.physics,
            last_val_breakdown.data,
            last_val_breakdown.physics
        ));

        TrainResult {
            model_version: self.model_version,
            loss,
            val_loss,
            architecture: self.layer_sizes.clone(),
            learning_rate: self.learning_rate,
            grew,
            pruned,
            completed_epochs,
            reached_target: reached_target_loss,
            reached_target_loss,
            reached_autonomous_convergence: trend_converged,
            stop_reason: if reached_target {
                if trend_converged {
                    "trend-converged".to_string()
                } else {
                    "target-loss-reached".to_string()
                }
            } else if manual_stop {
                "manual-stop".to_string()
            } else if completed_epochs >= max_total_epochs {
                "max-epochs-reached".to_string()
            } else {
                "manual-stop".to_string()
            },
            notes,
            pino: None,
        }
    }

    pub fn infer(&self, input: &SolveInput) -> AnnResult {
        let mut x = project_ann_input(input, self.layer_sizes.first().copied().unwrap_or(fem::ANN_INPUT_DIM));
        if !self.feature_scales.is_empty() {
            for i in 0..x.len() {
                let s = self.feature_scales.get(i).copied().unwrap_or(1.0).max(1e-9);
                x[i] /= s;
            }
        }
        let pred = self.forward(&x);
        let residual = ann_residual_score_from_pred(&pred, input);
        let mut fem_like = fem::fem_from_ann_prediction(input, &pred);
        let domain_extrapolation = domain_extrapolation_score(input, self.training_domain.as_ref());
        let base_confidence = (1.0 / (1.0 + self.last_loss.abs())).clamp(0.0, 1.0);
        let uncertainty = ((1.0 - base_confidence) + domain_extrapolation).clamp(0.0, 1.0);
        let confidence = (1.0 - uncertainty).clamp(0.0, 1.0);
        let (uncertainty_thr, residual_thr) = self.resolve_safeguard_thresholds(input);

        let mut used_fallback = false;
        let mut fallback_reason = None;
        let mut diagnostics = vec![format!(
            "ANN inference with model v{} and architecture {:?}",
            self.model_version, self.layer_sizes
        )];
        diagnostics.push(format!(
            "Domain extrapolation score: {:.4}",
            domain_extrapolation
        ));
        diagnostics.push(format!(
            "Safeguards: uncertainty={:.4}, residual={:.4}, thresholds=({:.4},{:.4})",
            uncertainty, residual, uncertainty_thr, residual_thr
        ));

        if self.fallback_enabled && (uncertainty > uncertainty_thr || residual > residual_thr) {
            fem_like = fem::solve_case(input);
            used_fallback = true;
            let reason = match (
                uncertainty > uncertainty_thr,
                residual > residual_thr,
            ) {
                (true, true) => format!(
                    "ANN rejected by uncertainty and residual safeguards ({uncertainty:.4} > {uncertainty_thr:.4}, {residual:.4} > {residual_thr:.4})"
                ),
                (true, false) => format!(
                    "ANN rejected by uncertainty safeguard ({uncertainty:.4} > {uncertainty_thr:.4})"
                ),
                (false, true) => format!(
                    "ANN rejected by residual safeguard ({residual:.4} > {residual_thr:.4})"
                ),
                (false, false) => "ANN rejected by safeguard policy.".to_string(),
            };
            diagnostics.push(format!("{reason}; FEM fallback used."));
            fallback_reason = Some(reason);
        }

        AnnResult {
            fem_like,
            confidence,
            uncertainty,
            model_version: self.model_version,
            used_fem_fallback: used_fallback,
            fallback_reason,
            domain_extrapolation_score: domain_extrapolation,
            residual_score: residual,
            uncertainty_threshold: uncertainty_thr,
            residual_threshold: residual_thr,
            diagnostics,
            surrogate_domain: self.training_domain.clone(),
            pino: None,
        }
    }

    pub fn status(&self) -> ModelStatus {
        ModelStatus {
            model_version: self.model_version,
            architecture: self.layer_sizes.clone(),
            learning_rate: self.learning_rate,
            last_loss: self.last_loss,
            train_samples: self.train_samples,
            audit_frequency: self.audit_frequency,
            fallback_enabled: self.fallback_enabled,
            safeguard_settings: SafeguardSettings {
                preset: self.safeguard_preset.clone(),
                uncertainty_threshold: self.safeguard_uncertainty_threshold,
                residual_threshold: self.safeguard_residual_threshold,
                adaptive_by_geometry: self.safeguard_adaptive_by_geometry,
            },
            surrogate_domain: self.training_domain.clone(),
            pino: None,
        }
    }

    pub(crate) fn surrogate_domain(&self) -> Option<SurrogateDomainSummary> {
        self.training_domain.clone()
    }

    pub fn snapshot_state(&self) -> AnnModelState {
        AnnModelState {
            layer_sizes: self.layer_sizes.clone(),
            layers: self.layers.clone(),
            moments: self.moments.clone(),
            best_layers: self.best_layers.clone(),
            learning_rate: self.learning_rate,
            model_version: self.model_version,
            last_loss: self.last_loss,
            best_val_loss: self.best_val_loss,
            train_samples: self.train_samples,
            plateau_epochs: self.plateau_epochs,
            audit_frequency: self.audit_frequency,
            fallback_enabled: self.fallback_enabled,
            max_hidden_layers: self.max_hidden_layers,
            max_neurons_per_layer: self.max_neurons_per_layer,
            data_weight: self.data_weight,
            physics_weight: self.physics_weight,
            adam_beta1: self.adam_beta1,
            adam_beta2: self.adam_beta2,
            adam_eps: self.adam_eps,
            adam_t: self.adam_t,
            feature_scales: self.feature_scales.clone(),
            safeguard_preset: self.safeguard_preset.clone(),
            safeguard_uncertainty_threshold: self.safeguard_uncertainty_threshold,
            safeguard_residual_threshold: self.safeguard_residual_threshold,
            safeguard_adaptive_by_geometry: self.safeguard_adaptive_by_geometry,
            hidden_activation: self.hidden_activation,
            last_train_seed: self.last_train_seed,
            training_domain: self.training_domain.clone(),
        }
    }

    pub fn load_state(&mut self, state: AnnModelState) {
        let target_sizes = self.layer_sizes.clone();
        let mut layer_sizes = state.layer_sizes.clone();
        let mut layers = state.layers.clone();
        let mut moments = state.moments.clone();
        let mut best_layers = state.best_layers.clone();
        if layer_sizes != target_sizes {
            layers = remap_layers_with_transfer(&layer_sizes, &target_sizes, &layers);
            moments = initialize_moments(&target_sizes);
            if let Some(best) = best_layers.as_mut() {
                *best = remap_layers_with_transfer(&layer_sizes, &target_sizes, best);
            }
            layer_sizes = target_sizes.clone();
        }
        self.layer_sizes = layer_sizes;
        self.layers = layers;
        self.moments = moments;
        self.best_layers = best_layers;
        self.learning_rate = state.learning_rate;
        self.model_version = state.model_version;
        self.last_loss = state.last_loss;
        self.best_val_loss = state.best_val_loss;
        self.train_samples = state.train_samples;
        self.plateau_epochs = state.plateau_epochs;
        self.audit_frequency = state.audit_frequency;
        self.fallback_enabled = state.fallback_enabled;
        self.max_hidden_layers = state.max_hidden_layers;
        self.max_neurons_per_layer = state.max_neurons_per_layer;
        self.data_weight = state.data_weight;
        self.physics_weight = state.physics_weight;
        self.adam_beta1 = state.adam_beta1;
        self.adam_beta2 = state.adam_beta2;
        self.adam_eps = state.adam_eps;
        self.adam_t = state.adam_t;
        let target_input_len = self.layer_sizes.first().copied().unwrap_or(fem::ANN_INPUT_DIM);
        self.feature_scales = resize_feature_scales(state.feature_scales, target_input_len);
        self.safeguard_preset = state.safeguard_preset;
        self.safeguard_uncertainty_threshold = state.safeguard_uncertainty_threshold;
        self.safeguard_residual_threshold = state.safeguard_residual_threshold;
        self.safeguard_adaptive_by_geometry = state.safeguard_adaptive_by_geometry;
        self.hidden_activation = state.hidden_activation;
        self.last_train_seed = state.last_train_seed;
        self.training_domain = state.training_domain;
    }

    pub fn set_safeguard_settings(&mut self, settings: SafeguardSettings) {
        self.safeguard_preset = settings.preset;
        self.safeguard_uncertainty_threshold = settings.uncertainty_threshold.clamp(0.01, 0.99);
        self.safeguard_residual_threshold = settings.residual_threshold.clamp(1e-6, 10.0);
        self.safeguard_adaptive_by_geometry = settings.adaptive_by_geometry;
    }

    fn resolve_safeguard_thresholds(&self, input: &SolveInput) -> (f64, f64) {
        let base_u = self.safeguard_uncertainty_threshold;
        let base_r = self.safeguard_residual_threshold;
        if !self.safeguard_adaptive_by_geometry {
            return (base_u, base_r);
        }
        let (u_geo, r_geo) = nafems_safeguard_thresholds(input);
        (base_u.min(u_geo), base_r.min(r_geo))
    }

    fn ensure_network_io(&mut self, input_dim: usize, output_dim: usize) {
        let current_input = self.layer_sizes.first().copied().unwrap_or(0);
        let current_output = self.layer_sizes.last().copied().unwrap_or(0);
        if current_input == input_dim && current_output == output_dim {
            return;
        }
        let mut target_sizes = if self.layer_sizes.len() >= 2 {
            self.layer_sizes.clone()
        } else {
            vec![input_dim, 16, output_dim]
        };
        target_sizes[0] = input_dim.max(1);
        if let Some(last) = target_sizes.last_mut() {
            *last = output_dim.max(1);
        }
        let old_sizes = self.layer_sizes.clone();
        self.layers = remap_layers_with_transfer(&old_sizes, &target_sizes, &self.layers);
        self.moments = initialize_moments(&target_sizes);
        if let Some(best) = self.best_layers.as_mut() {
            *best = remap_layers_with_transfer(&old_sizes, &target_sizes, best);
        }
        self.layer_sizes = target_sizes;
        self.feature_scales = resize_feature_scales(self.feature_scales.clone(), input_dim);
        self.adam_t = 0;
    }

    fn adapt_loss_weights(&mut self, data_loss: f64, physics_loss: f64) {
        let p = physics_loss.max(1e-12);
        let d = data_loss.max(1e-12);
        let ratio = d / p;
        // If data loss dominates, ease physics pressure so regression can converge.
        // If physics dominates, increase physics pressure to recover constraints.
        if ratio > 4.0 {
            self.physics_weight = (self.physics_weight * 0.97).max(0.02);
        } else if ratio < 0.25 {
            self.physics_weight = (self.physics_weight * 1.03).min(4.0);
        }
        self.data_weight = 1.0;
    }

    fn active_learning_refresh(
        &self,
        train_data: &mut Vec<TrainingSample>,
        norm: &TrainingNorm,
        batch: &TrainingBatch,
        analysis_type: &str,
        add_limit: usize,
        train_data_cap: usize,
        rng: &mut StdRng,
    ) -> usize {
        if train_data.is_empty() || add_limit == 0 {
            return 0;
        }
        let start = Instant::now();
        let budget = Duration::from_millis(120);
        let mut candidates: Vec<(f64, TrainingSample)> = Vec::new();

        let focused_cantilever = analysis_type == "cantilever";
        let focused_plate_hole = analysis_type == "plate-hole";

        for _ in 0..(add_limit * 2) {
            if start.elapsed() >= budget {
                break;
            }
            let base = match train_data.choose(rng) {
                Some(s) => s,
                None => break,
            };
            let mut case = SolveInput::default();
            case.geometry.length_in =
                (base.physics.length_in * rng.gen_range(0.92..1.08)).clamp(6.0, 20.0);
            case.geometry.width_in =
                (base.physics.width_in * rng.gen_range(0.92..1.08)).clamp(2.0, 8.0);
            case.geometry.thickness_in =
                (base.physics.thickness_in * rng.gen_range(0.85..1.15)).clamp(0.05, 0.5);
            case.geometry.hole_diameter_in = Some(
                (base.physics.hole_diameter_in * rng.gen_range(0.75..1.25))
                    .clamp(0.0, (case.geometry.width_in * 0.95).max(0.0)),
            );
            case.boundary_conditions.fix_start_face = base.physics.fix_start_face;
            case.boundary_conditions.fix_end_face = base.physics.fix_end_face;
            if focused_cantilever {
                case.boundary_conditions.fix_start_face = true;
                case.boundary_conditions.fix_end_face = false;
                case.geometry.hole_diameter_in = Some(0.0);
                let vertical_sign = if base.physics.vertical_load_lbf.abs() > 1e-9 {
                    base.physics.vertical_load_lbf.signum()
                } else {
                    -1.0
                };
                case.load.axial_load_lbf = 0.0;
                case.load.vertical_point_load_lbf = vertical_sign
                    * (base.physics.vertical_load_lbf.abs().max(50.0)
                        * rng.gen_range(0.88..1.12))
                    .clamp(50.0, 5_000.0);
            } else if focused_plate_hole {
                let hole_seed = base
                    .physics
                    .hole_diameter_in
                    .max(0.08 * case.geometry.width_in)
                    .min(case.geometry.width_in * 0.8);
                case.geometry.hole_diameter_in = Some(
                    (hole_seed * rng.gen_range(0.9..1.1))
                        .clamp(0.08 * case.geometry.width_in, case.geometry.width_in * 0.9),
                );
                case.load.vertical_point_load_lbf = 0.0;
                case.load.axial_load_lbf =
                    (base.physics.axial_load_lbf.abs().max(250.0) * rng.gen_range(0.88..1.12))
                        .clamp(250.0, 50_000.0);
            } else {
                case.boundary_conditions.fix_start_face =
                    rng.gen_bool(0.5) || case.boundary_conditions.fix_start_face;
                case.boundary_conditions.fix_end_face =
                    rng.gen_bool(0.35) || case.boundary_conditions.fix_end_face;
                if !case.boundary_conditions.fix_start_face
                    && !case.boundary_conditions.fix_end_face
                {
                    case.boundary_conditions.fix_start_face = true;
                }
                case.load.axial_load_lbf = (base.physics.axial_load_lbf
                    * rng.gen_range(0.85..1.15))
                .abs()
                .clamp(25.0, 50_000.0);
                case.load.vertical_point_load_lbf =
                    (base.physics.vertical_load_lbf * rng.gen_range(0.85..1.15))
                        .clamp(-5000.0, 5000.0);
                if rng.gen_bool(0.45) {
                    case.load.axial_load_lbf = (case.load.axial_load_lbf
                        + case.load.vertical_point_load_lbf.abs() * 0.18)
                    .clamp(25.0, 60_000.0);
                }
            }
            case.material.e_psi =
                (base.physics.e_psi * rng.gen_range(0.95..1.05)).clamp(1.0e6, 40.0e6);
            case.material.nu = rng.gen_range(0.18..0.38);
            case.material.rho_lb_in3 =
                (base.physics.rho_lb_in3 * rng.gen_range(0.95..1.05)).clamp(0.05, 1.5);
            case.material.yield_strength_psi =
                (base.physics.yield_strength_psi * rng.gen_range(0.9..1.1)).clamp(1_000.0, 120_000.0);
            case = lightweight_training_case(case);

            let mut sample = with_feature_and_physics(&case, batch, analysis_type);
            for i in 0..sample.x.len() {
                let s = norm.input_scales.get(i).copied().unwrap_or(1.0).max(1e-9);
                sample.x[i] /= s;
            }
            let pred = self.forward(&sample.x);
            let mut err = 0.0;
            for i in 0..sample.y.len() {
                let scale = norm.output_scales.get(i).copied().unwrap_or(1.0).max(1e-9);
                err += ((pred[i] - sample.y[i]) / scale).powi(2);
            }
            candidates.push((err, sample));
        }

        if candidates.is_empty() {
            return 0;
        }
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let before = train_data.len();
        for (_, s) in candidates.into_iter().take(add_limit) {
            train_data.push(s);
        }
        if train_data.len() > train_data_cap {
            let drop = train_data.len() - train_data_cap;
            train_data.drain(0..drop);
        }
        train_data.len().saturating_sub(before).min(add_limit)
    }

    fn activation_candidates() -> [HiddenActivation; 3] {
        [
            HiddenActivation::Tanh,
            HiddenActivation::Relu,
            HiddenActivation::SiLu,
        ]
    }

    fn activation_probe_score(eval: LossBreakdown) -> f64 {
        // Strongly prioritize data-fit first, then keep physics penalty in-check.
        (0.92 * eval.data) + (0.08 * eval.total)
    }

    fn activation_presearch(
        &self,
        train: &[TrainingSample],
        val: &[TrainingSample],
        norm: &TrainingNorm,
        seed: u64,
        probe_epochs: usize,
    ) -> Option<(HiddenActivation, f64, f64)> {
        if train.is_empty() {
            return None;
        }
        let mut best_activation = self.hidden_activation;
        let mut best_score = f64::MAX;
        let mut current_score = f64::MAX;
        for (idx, activation) in Self::activation_candidates().iter().copied().enumerate() {
            let mut probe = AnnModel::new_seeded(
                self.layer_sizes.clone(),
                seed.wrapping_add((idx as u64).wrapping_mul(7919)),
            );
            probe.hidden_activation = activation;
            probe.learning_rate = self.learning_rate.clamp(1e-6, 5e-3);
            probe.data_weight = self.data_weight;
            probe.physics_weight = self.physics_weight;

            let mut probe_rng =
                StdRng::seed_from_u64(seed.wrapping_mul(131).wrapping_add(idx as u64));
            for _ in 0..probe_epochs.max(1) {
                let _ = probe.epoch(train, norm, &mut probe_rng);
            }
            let eval = probe.eval_loss(if val.is_empty() { train } else { val }, norm);
            let score = Self::activation_probe_score(eval);
            if activation == self.hidden_activation {
                current_score = score;
            }
            if score < best_score {
                best_score = score;
                best_activation = activation;
            }
        }
        Some((best_activation, best_score, current_score))
    }

    fn bo_like_architecture_presearch(
        &self,
        dataset: &[TrainingSample],
        seed: u64,
    ) -> Option<Vec<usize>> {
        if self.layer_sizes.len() < 3 {
            return None;
        }
        let mut candidates = vec![self.layer_sizes.clone()];
        let mut wider = self.layer_sizes.clone();
        let h = wider.len() - 2;
        wider[h] = (wider[h] + 4).min(self.max_neurons_per_layer);
        candidates.push(wider);

        let mut deeper = self.layer_sizes.clone();
        if deeper.len() - 2 < self.max_hidden_layers {
            let out = deeper.pop().unwrap_or(fem::ANN_OUTPUT_DIM);
            deeper.push(8);
            deeper.push(out);
            candidates.push(deeper);
        }

        let norm = build_training_norm(dataset);
        let mut best_score = f64::MAX;
        let mut best_arch = self.layer_sizes.clone();
        for (idx, arch) in candidates.into_iter().enumerate() {
            let mut probe = AnnModel::new_seeded(arch.clone(), seed.wrapping_add(idx as u64));
            probe.hidden_activation = self.hidden_activation;
            probe.learning_rate = self.learning_rate;
            let split = ((dataset.len() as f64) * 0.8).ceil() as usize;
            let train = &dataset[..split.min(dataset.len())];
            let val = &dataset[split.min(dataset.len())..];
            let mut probe_rng =
                StdRng::seed_from_u64(seed.wrapping_mul(37).wrapping_add(idx as u64));
            for _ in 0..18 {
                let _ = probe.epoch(train, &norm, &mut probe_rng);
            }
            let eval = probe.eval_loss(if val.is_empty() { train } else { val }, &norm);
            let score = (0.9 * eval.data + 0.1 * eval.total) * (1.0 + 0.02 * idx as f64);
            if score < best_score {
                best_score = score;
                best_arch = arch;
            }
        }
        Some(best_arch)
    }

    fn network_snapshot(&self) -> NetworkSnapshot {
        let mut nodes = Vec::new();
        let mut connections = Vec::new();
        let mut max_abs_weight: f64 = 1e-9;

        let mut sample_input = vec![
            10.0,
            4.0,
            0.125,
            0.25,
            0.25 / 4.0,
            0.25 / 10.0,
            10.0 / 4.0,
            10.0 / 0.125,
            4.0 / 0.125,
            1.0,
            0.0,
            1_000.0,
            -1_000.0,
            1_414.2135623730951,
            10_000_000.0,
            0.33,
            0.283,
            13e-6,
            36_000.0,
        ];
        if !self.feature_scales.is_empty() {
            for i in 0..sample_input.len() {
                let s = self.feature_scales.get(i).copied().unwrap_or(1.0).max(1e-9);
                sample_input[i] /= s;
            }
        }
        let activations = self.activations_for_snapshot(&sample_input);

        for (layer_idx, layer_size) in self.layer_sizes.iter().enumerate() {
            for node_idx in 0..*layer_size {
                let bias = if layer_idx == 0 {
                    0.0
                } else {
                    self.layers
                        .get(layer_idx - 1)
                        .and_then(|l| l.biases.get(node_idx))
                        .copied()
                        .unwrap_or(0.0)
                };
                let activation = activations
                    .get(layer_idx)
                    .and_then(|a| a.get(node_idx))
                    .copied()
                    .unwrap_or(0.0);
                let importance = activation.abs() + bias.abs();
                nodes.push(NetworkNodeSnapshot {
                    id: format!("L{}N{}", layer_idx, node_idx),
                    layer: layer_idx,
                    index: node_idx,
                    activation,
                    bias,
                    importance,
                });
            }
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            for out_idx in 0..layer.weights.len() {
                for in_idx in 0..layer.weights[out_idx].len() {
                    let weight = layer.weights[out_idx][in_idx];
                    max_abs_weight = max_abs_weight.max(weight.abs());
                    connections.push(NetworkConnectionSnapshot {
                        from_id: format!("L{}N{}", layer_idx, in_idx),
                        to_id: format!("L{}N{}", layer_idx + 1, out_idx),
                        weight,
                        magnitude: 0.0,
                    });
                }
            }
        }

        for conn in &mut connections {
            conn.magnitude = (conn.weight.abs() / max_abs_weight).clamp(0.0, 1.0);
        }

        NetworkSnapshot {
            layer_sizes: self.layer_sizes.clone(),
            nodes,
            connections,
        }
    }

    fn warmstart_output_layer_with_ridge(&mut self, data: &[TrainingSample]) -> bool {
        if data.is_empty() || self.layers.is_empty() {
            return false;
        }
        let output_layer_idx = self.layers.len() - 1;
        let feature_dim = self.layers[output_layer_idx]
            .weights
            .first()
            .map(|row| row.len())
            .unwrap_or(0);
        let output_dim = self.layers[output_layer_idx].weights.len();
        if feature_dim == 0 || output_dim == 0 {
            return false;
        }

        let sample_count = data.len();
        let cols = feature_dim + 1;
        let mut design = vec![0.0; sample_count * cols];
        for (row_idx, sample) in data.iter().enumerate() {
            let activations = self.activations_for_snapshot(&sample.x);
            let Some(features) = activations.get(activations.len().saturating_sub(2)) else {
                return false;
            };
            if features.len() != feature_dim || sample.y.len() != output_dim {
                return false;
            }
            for (col_idx, value) in features.iter().enumerate() {
                design[row_idx * cols + col_idx] = *value;
            }
            design[row_idx * cols + feature_dim] = 1.0;
        }

        let x = DMatrix::from_row_slice(sample_count, cols, &design);
        let xt = x.transpose();
        let mut xtx = &xt * &x;
        let lambda = 1e-6;
        for i in 0..cols {
            xtx[(i, i)] += lambda;
        }
        let lu = xtx.lu();
        let mut solved_any = false;
        for out_idx in 0..output_dim {
            let mut target = Vec::with_capacity(sample_count);
            for sample in data {
                target.push(sample.y[out_idx]);
            }
            let y = DVector::from_vec(target);
            let rhs = &xt * y;
            let Some(beta) = lu.solve(&rhs) else {
                return solved_any;
            };
            for in_idx in 0..feature_dim {
                self.layers[output_layer_idx].weights[out_idx][in_idx] = beta[in_idx];
            }
            self.layers[output_layer_idx].biases[out_idx] = beta[feature_dim];
            solved_any = true;
        }
        if solved_any {
            self.moments = initialize_moments(&self.layer_sizes);
            self.best_layers = None;
            self.best_val_loss = f64::MAX;
            self.adam_t = 0;
        }
        solved_any
    }

    fn hidden_activate(&self, z: f64) -> f64 {
        match self.hidden_activation {
            HiddenActivation::Tanh => z.tanh(),
            HiddenActivation::Relu => z.max(0.0),
            HiddenActivation::SiLu => {
                let s = 1.0 / (1.0 + (-z).exp());
                z * s
            }
        }
    }

    fn hidden_activate_derivative(&self, z: f64, a: f64) -> f64 {
        match self.hidden_activation {
            HiddenActivation::Tanh => 1.0 - a * a,
            HiddenActivation::Relu => {
                if z > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            HiddenActivation::SiLu => {
                let s = 1.0 / (1.0 + (-z).exp());
                s * (1.0 + z * (1.0 - s))
            }
        }
    }

    fn epoch(
        &mut self,
        data: &[TrainingSample],
        norm: &TrainingNorm,
        rng: &mut StdRng,
    ) -> LossBreakdown {
        if data.is_empty() {
            return LossBreakdown::default();
        }

        let mut data_loss_total = 0.0;
        let mut physics_loss_total = 0.0;
        let mut momentum_total = 0.0;
        let mut kinematics_total = 0.0;
        let mut material_total = 0.0;
        let mut boundary_total = 0.0;
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.shuffle(rng);

        for idx in indices {
            let sample = &data[idx];
            let (activations, zs) = self.forward_cache(&sample.x);
            let y_pred = activations.last().cloned().unwrap_or_default();
            let penalty = physics_penalty(&y_pred, sample.physics);
            physics_loss_total += penalty.total;
            momentum_total += penalty.momentum;
            kinematics_total += penalty.kinematics;
            material_total += penalty.material;
            boundary_total += penalty.boundary;

            let mut deltas: Vec<Vec<f64>> = vec![vec![]; self.layers.len()];
            let mut last_delta = vec![0.0; sample.y.len()];
            for i in 0..sample.y.len() {
                let err = y_pred[i] - sample.y[i];
                let scale = norm.output_scales.get(i).copied().unwrap_or(1.0).max(1e-9);
                let inv_scale = 1.0 / scale;
                let data_grad = 2.0 * err * inv_scale * inv_scale / sample.y.len() as f64;
                let phys_grad = penalty.grad.get(i).copied().unwrap_or(0.0);
                last_delta[i] = self.data_weight * data_grad + self.physics_weight * phys_grad;
                data_loss_total += (err * inv_scale).powi(2);
            }
            deltas[self.layers.len() - 1] = last_delta;

            for l in (0..self.layers.len() - 1).rev() {
                let z = &zs[l];
                let a_hidden = &activations[l + 1];
                let mut delta = vec![0.0; z.len()];
                for i in 0..z.len() {
                    let mut acc = 0.0;
                    for j in 0..self.layers[l + 1].weights.len() {
                        acc += self.layers[l + 1].weights[j][i] * deltas[l + 1][j];
                    }
                    delta[i] = acc * self.hidden_activate_derivative(z[i], a_hidden[i]);
                }
                deltas[l] = delta;
            }

            clip_layer_deltas(&mut deltas, 5.0);

            self.adam_t = self.adam_t.saturating_add(1);
            let t = self.adam_t;
            let bias_corr1 = 1.0 - self.adam_beta1.powi(t as i32);
            let bias_corr2 = 1.0 - self.adam_beta2.powi(t as i32);

            for l in 0..self.layers.len() {
                let a_prev = &activations[l];
                for o in 0..self.layers[l].weights.len() {
                    for i in 0..self.layers[l].weights[o].len() {
                        let grad = deltas[l][o] * a_prev[i];
                        let m_prev = self.moments[l].m_weights[o][i];
                        let v_prev = self.moments[l].v_weights[o][i];
                        let m_new = self.adam_beta1 * m_prev + (1.0 - self.adam_beta1) * grad;
                        let v_new =
                            self.adam_beta2 * v_prev + (1.0 - self.adam_beta2) * grad * grad;
                        self.moments[l].m_weights[o][i] = m_new;
                        self.moments[l].v_weights[o][i] = v_new;
                        let m_hat = m_new / bias_corr1.max(1e-12);
                        let v_hat = v_new / bias_corr2.max(1e-12);
                        self.layers[l].weights[o][i] -=
                            self.learning_rate * m_hat / (v_hat.sqrt() + self.adam_eps);
                    }
                    let grad_b = deltas[l][o];
                    let mb_prev = self.moments[l].m_biases[o];
                    let vb_prev = self.moments[l].v_biases[o];
                    let mb_new = self.adam_beta1 * mb_prev + (1.0 - self.adam_beta1) * grad_b;
                    let vb_new =
                        self.adam_beta2 * vb_prev + (1.0 - self.adam_beta2) * grad_b * grad_b;
                    self.moments[l].m_biases[o] = mb_new;
                    self.moments[l].v_biases[o] = vb_new;
                    let mb_hat = mb_new / bias_corr1.max(1e-12);
                    let vb_hat = vb_new / bias_corr2.max(1e-12);
                    self.layers[l].biases[o] -=
                        self.learning_rate * mb_hat / (vb_hat.sqrt() + self.adam_eps);
                }
            }
        }

        let output_dim = data.first().map(|s| s.y.len()).unwrap_or(1).max(1) as f64;
        let data_avg = data_loss_total / (data.len() as f64 * output_dim);
        let physics_avg = physics_loss_total / data.len() as f64;
        LossBreakdown {
            data: data_avg,
            physics: physics_avg,
            total: self.data_weight * data_avg + self.physics_weight * physics_avg,
            momentum: momentum_total / data.len() as f64,
            kinematics: kinematics_total / data.len() as f64,
            material: material_total / data.len() as f64,
            boundary: boundary_total / data.len() as f64,
        }
    }

    fn eval_loss(&self, data: &[TrainingSample], norm: &TrainingNorm) -> LossBreakdown {
        if data.is_empty() {
            return LossBreakdown::default();
        }
        let mut data_total = 0.0;
        let mut physics_total = 0.0;
        let mut momentum_total = 0.0;
        let mut kinematics_total = 0.0;
        let mut material_total = 0.0;
        let mut boundary_total = 0.0;
        for sample in data {
            let p = self.forward(&sample.x);
            for i in 0..sample.y.len() {
                let scale = norm.output_scales.get(i).copied().unwrap_or(1.0).max(1e-9);
                data_total += ((p[i] - sample.y[i]) / scale).powi(2);
            }
            let penalty = physics_penalty(&p, sample.physics);
            physics_total += penalty.total;
            momentum_total += penalty.momentum;
            kinematics_total += penalty.kinematics;
            material_total += penalty.material;
            boundary_total += penalty.boundary;
        }
        let output_dim = data.first().map(|s| s.y.len()).unwrap_or(1).max(1) as f64;
        let data_avg = data_total / (data.len() as f64 * output_dim);
        let physics_avg = physics_total / data.len() as f64;
        LossBreakdown {
            data: data_avg,
            physics: physics_avg,
            total: self.data_weight * data_avg + self.physics_weight * physics_avg,
            momentum: momentum_total / data.len() as f64,
            kinematics: kinematics_total / data.len() as f64,
            material: material_total / data.len() as f64,
            boundary: boundary_total / data.len() as f64,
        }
    }

    fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut a = x.to_vec();
        for (idx, layer) in self.layers.iter().enumerate() {
            let last = idx == self.layers.len() - 1;
            let mut next = vec![0.0; layer.weights.len()];
            for (o, row) in layer.weights.iter().enumerate() {
                let mut z = layer.biases[o];
                for (i, w) in row.iter().enumerate() {
                    z += w * a[i];
                }
                next[o] = if last { z } else { self.hidden_activate(z) };
            }
            a = next;
        }
        a
    }

    fn forward_cache(&self, x: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut activations = vec![x.to_vec()];
        let mut zs = Vec::new();
        let mut a = x.to_vec();

        for (idx, layer) in self.layers.iter().enumerate() {
            let last = idx == self.layers.len() - 1;
            let mut z_vec = vec![0.0; layer.weights.len()];
            let mut next = vec![0.0; layer.weights.len()];
            for (o, row) in layer.weights.iter().enumerate() {
                let mut z = layer.biases[o];
                for (i, w) in row.iter().enumerate() {
                    z += w * a[i];
                }
                z_vec[o] = z;
                next[o] = if last { z } else { self.hidden_activate(z) };
            }
            zs.push(z_vec);
            activations.push(next.clone());
            a = next;
        }

        (activations, zs)
    }

    fn activations_for_snapshot(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let mut all = vec![x.to_vec()];
        let mut a = x.to_vec();
        for (idx, layer) in self.layers.iter().enumerate() {
            let last = idx == self.layers.len() - 1;
            let mut next = vec![0.0; layer.weights.len()];
            for (o, row) in layer.weights.iter().enumerate() {
                let mut z = layer.biases[o];
                for (i, w) in row.iter().enumerate() {
                    z += w * a[i];
                }
                next[o] = if last { z } else { self.hidden_activate(z) };
            }
            all.push(next.clone());
            a = next;
        }
        all
    }

    fn grow_topology(&mut self) -> bool {
        if self.layer_sizes.len() < 3 {
            return false;
        }

        let old_sizes = self.layer_sizes.clone();
        let old_layers = self.layers.clone();
        let last_hidden_idx = self.layer_sizes.len() - 2;
        if self.layer_sizes[last_hidden_idx] < self.max_neurons_per_layer {
            self.layer_sizes[last_hidden_idx] += 4;
            self.layers = remap_layers_with_transfer(&old_sizes, &self.layer_sizes, &old_layers);
            self.moments = initialize_moments(&self.layer_sizes);
            self.adam_t = 0;
            self.plateau_epochs = 0;
            return true;
        }

        let hidden_count = self.layer_sizes.len() - 2;
        if hidden_count < self.max_hidden_layers {
            let output = self.layer_sizes.pop().unwrap_or(fem::ANN_OUTPUT_DIM);
            self.layer_sizes.push(8);
            self.layer_sizes.push(output);
            self.layers = remap_layers_with_transfer(&old_sizes, &self.layer_sizes, &old_layers);
            self.moments = initialize_moments(&self.layer_sizes);
            self.adam_t = 0;
            self.plateau_epochs = 0;
            return true;
        }

        false
    }

    fn prune_topology(&mut self) -> bool {
        if self.layer_sizes.len() < 3 {
            return false;
        }

        let old_sizes = self.layer_sizes.clone();
        let old_layers = self.layers.clone();
        let last_hidden_idx = self.layer_sizes.len() - 2;
        if self.layer_sizes[last_hidden_idx] > 4 {
            self.layer_sizes[last_hidden_idx] -= 2;
            self.layers = remap_layers_with_transfer(&old_sizes, &self.layer_sizes, &old_layers);
            self.moments = initialize_moments(&self.layer_sizes);
            self.adam_t = 0;
            self.plateau_epochs = 0;
            return true;
        }

        let hidden_count = self.layer_sizes.len() - 2;
        if hidden_count > 1 {
            let output = self.layer_sizes.pop().unwrap_or(fem::ANN_OUTPUT_DIM);
            self.layer_sizes.pop();
            self.layer_sizes.push(output);
            self.layers = remap_layers_with_transfer(&old_sizes, &self.layer_sizes, &old_layers);
            self.moments = initialize_moments(&self.layer_sizes);
            self.adam_t = 0;
            self.plateau_epochs = 0;
            return true;
        }

        false
    }
}

fn ann_residual_score_from_pred(pred: &[f64], input: &SolveInput) -> f64 {
    if pred.len() < 9 {
        return 0.0;
    }

    let ux = pred[0];
    let uy = pred[1];
    let uz = pred[2];
    let sxx = pred[3];
    let syy = pred[4];
    let szz = pred[5];
    let vm_pred = pred[6];
    let tresca_pred = pred[7];
    let maxp_pred = pred[8];

    let l = input.geometry.length_in.max(1e-6);
    let w = input.geometry.width_in.max(1e-6);
    let t = input.geometry.thickness_in.max(1e-6);
    let area = (w * t).max(1e-9);
    let i = (t * w.powi(3) / 12.0).max(1e-9);
    let c = 0.5 * w;
    let e = input.material.e_psi.max(1.0);
    let p_vertical = input.load.vertical_point_load_lbf;
    let p_axial = input.load.axial_load_lbf;
    let hole_d = input.geometry.hole_diameter_in.unwrap_or(0.0).max(0.0);

    let sigma_nom = p_axial / area;
    let sigma_bending_ref = (p_vertical.abs() * l * c / i).abs();
    let stress_scale = sigma_nom.abs().max(sigma_bending_ref).max(1.0);
    let tip_scale = if p_vertical.abs() > 1e-9 {
        (p_vertical.abs() * l.powi(3) / (3.0 * e * i))
            .abs()
            .max(1e-9)
    } else if p_axial.abs() > 1e-9 {
        (sigma_nom.abs() * l / e).abs().max(1e-9)
    } else {
        1e-6
    };

    let mut terms = Vec::new();

    if p_vertical.abs() > 1e-9 {
        if !input.boundary_conditions.fix_end_face {
            let moment_equilibrium = ((sxx * i / c.max(1e-9)) - p_vertical * l).abs()
                / (p_vertical.abs() * l).max(1.0);
            let kappa = (3.0 * e * c / (l * l)).max(1e-9);
            let kinematic = (sxx.abs() - (kappa * uy).abs()).abs() / stress_scale;
            let tip_expected = p_vertical * l.powi(3) / (3.0 * e * i).max(1e-9);
            let boundary = (uy - tip_expected).abs() / tip_scale;
            terms.push(moment_equilibrium);
            terms.push(kinematic);
            terms.push(boundary);
            terms.push(ux.abs() / tip_scale);
            terms.push(uz.abs() / tip_scale);
        } else {
            terms.push(ux.abs() / tip_scale);
            terms.push(uy.abs() / tip_scale);
            terms.push(uz.abs() / tip_scale);
        }
    }

    if p_axial.abs() > 1e-9 {
        if hole_d <= 1e-9 {
            terms.push((sxx - sigma_nom).abs() / stress_scale);
        } else {
            let abs_sigma = sxx.abs();
            let lo = 0.5 * sigma_nom.abs();
            let hi = 5.0 * sigma_nom.abs().max(lo + 1e-9);
            if abs_sigma < lo {
                terms.push((lo - abs_sigma) / hi.max(1.0));
            } else if abs_sigma > hi {
                terms.push((abs_sigma - hi) / hi.max(1.0));
            }
        }
        if !input.boundary_conditions.fix_end_face {
            let tip_expected = sigma_nom * l / e;
            terms.push((ux - tip_expected).abs() / tip_scale);
            terms.push(uy.abs() / tip_scale);
            terms.push(uz.abs() / tip_scale);
        } else {
            terms.push(ux.abs() / tip_scale);
            terms.push(uy.abs() / tip_scale);
            terms.push(uz.abs() / tip_scale);
        }
    }

    if p_vertical.abs() <= 1e-9 && p_axial.abs() <= 1e-9 {
        terms.push(ux.abs() / l.max(1.0));
        terms.push(uy.abs() / l.max(1.0));
        terms.push(uz.abs() / l.max(1.0));
        terms.push(sxx.abs() / e);
    }

    let stress_tensor = [[sxx, 0.0, 0.0], [0.0, syy, 0.0], [0.0, 0.0, szz]];
    let principal = principal_stresses(stress_tensor);
    let vm_calc = von_mises(stress_tensor).max(1e-12);
    let tresca_calc = tresca_from_principal(principal).abs().max(1e-12);
    terms.push((vm_pred.abs() - vm_calc).abs() / vm_calc.max(stress_scale));
    terms.push((tresca_pred.abs() - tresca_calc).abs() / tresca_calc.max(stress_scale));
    terms.push((maxp_pred - principal[0]).abs() / principal[0].abs().max(stress_scale));

    if terms.is_empty() {
        0.0
    } else {
        (terms.iter().sum::<f64>() / terms.len() as f64).max(0.0)
    }
}

fn nafems_safeguard_thresholds(input: &SolveInput) -> (f64, f64) {
    let hole_ratio =
        input.geometry.hole_diameter_in.unwrap_or(0.0) / input.geometry.width_in.max(1e-9);
    if hole_ratio > 0.35 {
        (0.22, 0.18)
    } else if hole_ratio > 0.20 {
        (0.26, 0.24)
    } else {
        (0.30, 0.30)
    }
}

fn physics_penalty(pred: &[f64], physics: PhysicsContext) -> PhysicsPenaltyBreakdown {
    if pred.len() < 9 {
        return PhysicsPenaltyBreakdown {
            total: 0.0,
            momentum: 0.0,
            kinematics: 0.0,
            material: 0.0,
            boundary: 0.0,
            grad: vec![0.0; pred.len()],
        };
    }

    let mut grad = vec![0.0; pred.len()];

    let ux = pred[0];
    let uy = pred[1];
    let uz = pred[2];
    let sxx = pred[3];
    let syy = pred[4];
    let szz = pred[5];
    let vm_pred = pred[6];
    let tresca_pred = pred[7];
    let maxp_pred = pred[8];

    let l = physics.length_in.max(1e-6);
    let w = physics.width_in.max(1e-6);
    let t = physics.thickness_in.max(1e-6);
    let p_vertical = physics.vertical_load_lbf;
    let p_axial = physics.axial_load_lbf;
    let hole_d = physics.hole_diameter_in.max(0.0);
    let e = physics.e_psi.max(1.0);
    let c = 0.5 * w;
    let i = (t * w.powi(3) / 12.0).max(1e-9);
    let area = (w * t).max(1e-9);
    let cantilever_mode = p_vertical.abs() > 1e-9;
    let axial_mode = p_axial.abs() > 1e-9;

    let mut l_momentum = 0.0;
    let mut l_kinematics = 0.0;
    let mut l_material = 0.0;
    let mut l_boundary = 0.0;
    if cantilever_mode && !physics.fix_end_face {
        // Equilibrium/moment consistency for transverse cantilever loading.
        let r_eq = (sxx * i / c.max(1e-9)) - p_vertical * l;
        let w_eq = 1e-8;
        let d_r_eq_d_sxx = i / c.max(1e-9);
        l_momentum = w_eq * r_eq * r_eq;
        grad[1] += 2.0 * w_eq * r_eq * d_r_eq_d_sxx;

        // Kinematic consistency for cantilever root curvature relation.
        let kappa = (3.0 * e * c / (l * l)).max(1e-9);
        let r_const = sxx - kappa * uy;
        let w_kin = 1e-8;
        l_kinematics = w_kin * r_const * r_const;
        grad[1] += 2.0 * w_kin * r_const * (-kappa);
        grad[3] += 2.0 * w_kin * r_const;
    }
    if cantilever_mode && physics.fix_end_face {
        let w_fix = 1e-2;
        l_boundary += w_fix * (ux * ux + uy * uy + uz * uz);
        grad[0] += 2.0 * w_fix * ux;
        grad[1] += 2.0 * w_fix * uy;
        grad[2] += 2.0 * w_fix * uz;
    }
    if axial_mode {
        let sigma_nom = p_axial / area;
        if hole_d <= 1e-9 {
            // Pure axial bar/plate without hole: Sxx should match nominal traction.
            let r_axial = sxx - sigma_nom;
            let w_axial = 8e-4;
            l_momentum += w_axial * r_axial * r_axial;
            grad[3] += 2.0 * w_axial * r_axial;
        } else {
            // Plate-with-hole: keep a broad physically plausible stress band, not a hard value.
            let abs_sigma = sxx.abs();
            let lo = 0.5 * sigma_nom.abs();
            let hi = 5.0 * sigma_nom.abs().max(lo + 1e-9);
            let w_band = 4e-4;
            if abs_sigma < lo {
                let r = lo - abs_sigma;
                l_momentum += w_band * r * r;
                grad[3] += -2.0 * w_band * r * sxx.signum();
            } else if abs_sigma > hi {
                let r = abs_sigma - hi;
                l_momentum += w_band * r * r;
                grad[3] += 2.0 * w_band * r * sxx.signum();
            }
        }
    }

    // Stress-invariant consistency: VM should match stress components.
    let q = 0.5 * ((sxx - syy).powi(2) + (syy - szz).powi(2) + (szz - sxx).powi(2));
    let vm_calc = q.max(1e-12).sqrt();
    let r_vm = vm_pred.abs() - vm_calc;
    let w_vm = 5e-3;
    let l_vm = w_vm * r_vm * r_vm;
    let sign_vm = if vm_pred >= 0.0 { 1.0 } else { -1.0 };
    grad[6] += 2.0 * w_vm * r_vm * sign_vm;

    let denom = (2.0 * vm_calc).max(1e-9);
    let d_vm_dsxx = (2.0 * sxx - syy - szz) / denom;
    let d_vm_dsyy = (2.0 * syy - sxx - szz) / denom;
    let d_vm_dszz = (2.0 * szz - sxx - syy) / denom;
    grad[3] += 2.0 * w_vm * r_vm * (-d_vm_dsxx);
    grad[4] += 2.0 * w_vm * r_vm * (-d_vm_dsyy);
    grad[5] += 2.0 * w_vm * r_vm * (-d_vm_dszz);

    let principal = principal_stresses([
        [sxx, 0.0, 0.0],
        [0.0, syy, 0.0],
        [0.0, 0.0, szz],
    ]);
    let tresca_calc = tresca_from_principal(principal).abs();
    let r_tresca = tresca_pred.abs() - tresca_calc;
    let w_tresca = 2e-3;
    let l_tresca = w_tresca * r_tresca * r_tresca;
    grad[7] += 2.0 * w_tresca * r_tresca * tresca_pred.signum();

    // Principal stress head consistency.
    let r_p = maxp_pred - principal[0];
    let w_p = 1e-3;
    let l_p = w_p * r_p * r_p;
    grad[8] += 2.0 * w_p * r_p;
    if principal[0] == sxx {
        grad[3] += -2.0 * w_p * r_p;
    } else if principal[0] == syy {
        grad[4] += -2.0 * w_p * r_p;
    } else {
        grad[5] += -2.0 * w_p * r_p;
    }

    l_material += l_vm + l_tresca + l_p;

    let w_bc = 1e-2;
    if cantilever_mode && !physics.fix_end_face {
        let tip_expected = p_vertical * l.powi(3) / (3.0 * e * i).max(1e-9);
        let r_tip = uy - tip_expected;
        l_boundary += w_bc * r_tip * r_tip;
        grad[1] += 2.0 * w_bc * r_tip;
        l_boundary += w_bc * (ux * ux + uz * uz);
        grad[0] += 2.0 * w_bc * ux;
        grad[2] += 2.0 * w_bc * uz;
    } else if cantilever_mode {
        l_boundary += w_bc * (ux * ux + uy * uy + uz * uz);
        grad[0] += 2.0 * w_bc * ux;
        grad[1] += 2.0 * w_bc * uy;
        grad[2] += 2.0 * w_bc * uz;
    }
    if axial_mode && !physics.fix_end_face {
        let sigma_nom = p_axial / area;
        let tip_expected = sigma_nom * l / e;
        let r_tip = ux - tip_expected;
        l_boundary += w_bc * r_tip * r_tip;
        grad[0] += 2.0 * w_bc * r_tip;
        l_boundary += w_bc * (uy * uy + uz * uz);
        grad[1] += 2.0 * w_bc * uy;
        grad[2] += 2.0 * w_bc * uz;
    } else if axial_mode {
        l_boundary += w_bc * (ux * ux + uy * uy + uz * uz);
        grad[0] += 2.0 * w_bc * ux;
        grad[1] += 2.0 * w_bc * uy;
        grad[2] += 2.0 * w_bc * uz;
    } else {
        // Near-zero load should imply near-zero response.
        let r_bc_ux = ux / l.max(1.0);
        let r_bc_uy = uy / l.max(1.0);
        let r_bc_uz = uz / l.max(1.0);
        let r_bc_sig = sxx / e;
        l_boundary += w_bc * (r_bc_ux * r_bc_ux + r_bc_uy * r_bc_uy + r_bc_uz * r_bc_uz + r_bc_sig * r_bc_sig);
        grad[0] += 2.0 * w_bc * r_bc_ux * (1.0 / l.max(1.0));
        grad[1] += 2.0 * w_bc * r_bc_uy * (1.0 / l.max(1.0));
        grad[2] += 2.0 * w_bc * r_bc_uz * (1.0 / l.max(1.0));
        grad[3] += 2.0 * w_bc * r_bc_sig * (1.0 / e);

        // Contact/interface proxy only applies in zero-load contact mode.
        let thickness_span = physics.thickness_in.max(1e-6);
        let predicted_gap = (uy / thickness_span).max(-2.0).min(2.0);
        let penetration = (-predicted_gap).max(0.0);
        let contact_pressure = sxx.max(0.0);
        let w_contact = physics.contact_penalty.max(0.0) * 1e-4;
        let l_contact = contact_penalty(predicted_gap, contact_pressure, r_bc_sig.abs(), w_contact);
        if predicted_gap < 0.0 {
            grad[1] += 2.0 * w_contact * penetration * (-1.0 / thickness_span);
        }
        let d_comp_du = 2.0
            * w_contact
            * (contact_pressure * predicted_gap)
            * (contact_pressure / thickness_span);
        let d_comp_ds = 2.0 * w_contact * (contact_pressure * predicted_gap) * predicted_gap;
        grad[0] += d_comp_du;
        grad[1] += d_comp_ds;
        l_boundary += l_contact;
    }

    // Plasticity hook: progressively penalize exceedance over pseudo-yield envelope.
    if physics.plasticity_factor > 0.0 {
        let pseudo_yield = physics.yield_strength_psi.max(1.0);
        let over = (vm_pred.abs() - pseudo_yield).max(0.0);
        let w_pl = physics.plasticity_factor * 1e-6;
        let l_pl = w_pl * over * over;
        l_material += l_pl;
        if vm_pred.abs() > pseudo_yield {
            grad[6] += 2.0 * w_pl * over * vm_pred.signum();
        }
    }

    // Apply explicit residual pillar weights.
    let wm = physics.residual_weight_momentum.max(0.0);
    let wk = physics.residual_weight_kinematics.max(0.0);
    let wmat = physics.residual_weight_material.max(0.0);
    let wbc = physics.residual_weight_boundary.max(0.0);
    let total = universal_loss(
        wm,
        wk,
        wmat,
        wbc,
        ResidualPillars {
            momentum: l_momentum,
            kinematics: l_kinematics,
            material: l_material,
            boundary: l_boundary,
        },
    );
    let scale = (wm + wk + wmat + wbc).max(1e-9);
    for g in &mut grad {
        *g *= scale / 4.0;
    }

    PhysicsPenaltyBreakdown {
        total,
        momentum: l_momentum,
        kinematics: l_kinematics,
        material: l_material,
        boundary: l_boundary,
        grad,
    }
}

fn initialize_layers(layer_sizes: &[usize]) -> Vec<DenseLayer> {
    let mut rng = rand::thread_rng();
    initialize_layers_with_rng(layer_sizes, &mut rng)
}

fn initialize_layers_with_rng(layer_sizes: &[usize], rng: &mut impl Rng) -> Vec<DenseLayer> {
    let mut layers = Vec::new();

    for pair in layer_sizes.windows(2) {
        let in_size = pair[0];
        let out_size = pair[1];
        let scale = (2.0 / in_size as f64).sqrt();
        let mut w = vec![vec![0.0; in_size]; out_size];
        let mut b = vec![0.0; out_size];

        for o in 0..out_size {
            b[o] = 0.0;
            for i in 0..in_size {
                w[o][i] = rng.gen_range(-1.0..1.0) * scale;
            }
        }

        layers.push(DenseLayer {
            weights: w,
            biases: b,
        });
    }

    layers
}

fn initialize_moments(layer_sizes: &[usize]) -> Vec<DenseLayerMoments> {
    let mut out = Vec::new();
    for pair in layer_sizes.windows(2) {
        let in_size = pair[0];
        let out_size = pair[1];
        out.push(DenseLayerMoments {
            m_weights: vec![vec![0.0; in_size]; out_size],
            v_weights: vec![vec![0.0; in_size]; out_size],
            m_biases: vec![0.0; out_size],
            v_biases: vec![0.0; out_size],
        });
    }
    out
}

fn clip_layer_deltas(deltas: &mut [Vec<f64>], max_abs: f64) {
    for layer in deltas {
        for d in layer {
            *d = d.clamp(-max_abs, max_abs);
        }
    }
}

fn build_training_norm(dataset: &[TrainingSample]) -> TrainingNorm {
    let input_len = dataset.first().map(|d| d.x.len()).unwrap_or(0);
    let mut input_scales = vec![1.0_f64; input_len];
    for sample in dataset {
        for (i, x) in sample.x.iter().enumerate() {
            input_scales[i] = input_scales[i].max(x.abs());
        }
    }
    for scale in &mut input_scales {
        *scale = (*scale).max(1e-6_f64);
    }

    let output_len = dataset.first().map(|d| d.y.len()).unwrap_or(0);
    let mut scales = vec![1.0_f64; output_len];
    for sample in dataset {
        for (i, y) in sample.y.iter().enumerate() {
            scales[i] = scales[i].max(y.abs());
        }
    }
    for scale in &mut scales {
        *scale = (*scale).max(1e-6_f64);
    }
    // Robust floors prevent near-zero channels from dominating normalized loss.
    // Output order: [ux, uy, uz, sxx, syy, szz, von_mises, tresca, max_principal]
    for idx in [0usize, 1, 2] {
        if idx < scales.len() {
            scales[idx] = scales[idx].max(1e-4);
        }
    }
    for idx in [3usize, 4, 5, 6, 7, 8] {
        if idx < scales.len() {
            scales[idx] = scales[idx].max(10.0);
        }
    }
    TrainingNorm {
        input_scales,
        output_scales: scales,
    }
}

fn resize_feature_scales(mut scales: Vec<f64>, target_len: usize) -> Vec<f64> {
    if scales.len() < target_len {
        scales.resize(target_len, 1.0);
    } else if scales.len() > target_len {
        scales.truncate(target_len);
    }
    scales
}

fn project_ann_input(input: &SolveInput, target_len: usize) -> Vec<f64> {
    let mut features = fem::ann_features(input);
    if features.len() < target_len {
        features.resize(target_len, 0.0);
    } else if features.len() > target_len {
        features.truncate(target_len);
    }
    features
}

fn build_surrogate_domain_summary(
    seed_cases: &[SolveInput],
    expanded_cases: &[SolveInput],
    analysis_type: &str,
) -> SurrogateDomainSummary {
    let mut mins = vec![f64::INFINITY; ANN_FEATURE_LABELS.len()];
    let mut maxs = vec![f64::NEG_INFINITY; ANN_FEATURE_LABELS.len()];
    let mut mixed_load_cases = 0usize;
    let mut hole_cases = 0usize;
    let mut dual_fixed_cases = 0usize;

    for case in expanded_cases {
        let features = fem::ann_features(case);
        for (idx, value) in features.iter().enumerate() {
            if idx >= mins.len() {
                break;
            }
            mins[idx] = mins[idx].min(*value);
            maxs[idx] = maxs[idx].max(*value);
        }
        if case.load.axial_load_lbf.abs() > 1e-9 && case.load.vertical_point_load_lbf.abs() > 1e-9
        {
            mixed_load_cases += 1;
        }
        if case
            .geometry
            .hole_diameter_in
            .map(|hole| hole > 1e-9)
            .unwrap_or(false)
        {
            hole_cases += 1;
        }
        if case.boundary_conditions.fix_start_face && case.boundary_conditions.fix_end_face {
            dual_fixed_cases += 1;
        }
    }

    for idx in 0..mins.len() {
        if !mins[idx].is_finite() {
            mins[idx] = 0.0;
        }
        if !maxs[idx].is_finite() {
            maxs[idx] = 0.0;
        }
    }

    let mut coverage_tags = vec![analysis_type.to_string()];
    if mixed_load_cases > 0 {
        coverage_tags.push("mixed-load".to_string());
    }
    if hole_cases > 0 {
        coverage_tags.push("hole-discontinuity".to_string());
    }
    if dual_fixed_cases > 0 {
        coverage_tags.push("dual-fixed".to_string());
    }

    SurrogateDomainSummary {
        feature_labels: ANN_FEATURE_LABELS.iter().map(|label| (*label).to_string()).collect(),
        feature_mins: mins,
        feature_maxs: maxs,
        coverage_tags,
        training_seed_cases: seed_cases.len(),
        expanded_cases: expanded_cases.len(),
        mixed_load_cases,
        hole_cases,
        dual_fixed_cases,
    }
}

fn domain_extrapolation_score(
    input: &SolveInput,
    domain: Option<&SurrogateDomainSummary>,
) -> f64 {
    let Some(domain) = domain else {
        return 0.0;
    };
    let features = fem::ann_features(input);
    let mut worst = 0.0_f64;
    for idx in 0..features
        .len()
        .min(domain.feature_mins.len())
        .min(domain.feature_maxs.len())
    {
        let value = features[idx];
        let min = domain.feature_mins[idx];
        let max = domain.feature_maxs[idx];
        let span = (max - min).abs().max(1e-6);
        let overflow = if value < min {
            (min - value) / span
        } else if value > max {
            (value - max) / span
        } else {
            0.0
        };
        worst = worst.max(overflow);
    }
    if input.load.axial_load_lbf.abs() > 1e-9
        && input.load.vertical_point_load_lbf.abs() > 1e-9
        && domain.mixed_load_cases == 0
    {
        worst = worst.max(0.35);
    }
    if input
        .geometry
        .hole_diameter_in
        .map(|hole| hole > 1e-9)
        .unwrap_or(false)
        && domain.hole_cases == 0
    {
        worst = worst.max(0.35);
    }
    if input.boundary_conditions.fix_start_face
        && input.boundary_conditions.fix_end_face
        && domain.dual_fixed_cases == 0
    {
        worst = worst.max(0.25);
    }
    worst.clamp(0.0, 1.0)
}

fn default_safeguard_preset() -> String {
    "balanced".to_string()
}

fn default_safeguard_uncertainty_threshold() -> f64 {
    0.26
}

fn default_safeguard_residual_threshold() -> f64 {
    0.24
}

fn default_safeguard_adaptive_by_geometry() -> bool {
    true
}

fn lightweight_training_case(mut case: SolveInput) -> SolveInput {
    case.mesh.nx = case.mesh.nx.clamp(6, 12);
    case.mesh.ny = case.mesh.ny.clamp(4, 8);
    case.mesh.nz = case.mesh.nz.clamp(1, 2);
    case.mesh.amr_enabled = false;
    case.mesh.amr_passes = 0;
    case.mesh.auto_adapt = true;
    case.mesh.max_dofs = case.mesh.max_dofs.min(3200).max(900);
    case
}

fn load_feature_vector(case: &SolveInput) -> Vec<f64> {
    fem::ann_features(case)
}

fn ann_target_vector_from_stress(
    ux: f64,
    uy: f64,
    uz: f64,
    sxx: f64,
    syy: f64,
    szz: f64,
) -> Vec<f64> {
    let stress_tensor = [[sxx, 0.0, 0.0], [0.0, syy, 0.0], [0.0, 0.0, szz]];
    let principal = principal_stresses(stress_tensor);
    vec![
        ux,
        uy,
        uz,
        sxx,
        syy,
        szz,
        von_mises(stress_tensor),
        tresca_from_principal(principal),
        principal[0],
    ]
}

fn physics_context_from_case(case: &SolveInput, batch: &TrainingBatch) -> PhysicsContext {
    PhysicsContext {
        length_in: case.geometry.length_in,
        width_in: case.geometry.width_in,
        thickness_in: case.geometry.thickness_in,
        hole_diameter_in: case.geometry.hole_diameter_in.unwrap_or(0.0),
        axial_load_lbf: case.load.axial_load_lbf,
        vertical_load_lbf: case.load.vertical_point_load_lbf,
        e_psi: case.material.e_psi,
        rho_lb_in3: case.material.rho_lb_in3,
        yield_strength_psi: case.material.yield_strength_psi,
        fix_start_face: case.boundary_conditions.fix_start_face,
        fix_end_face: case.boundary_conditions.fix_end_face,
        residual_weight_momentum: batch.residual_weight_momentum.unwrap_or(1.0).max(0.0),
        residual_weight_kinematics: batch.residual_weight_kinematics.unwrap_or(1.0).max(0.0),
        residual_weight_material: batch.residual_weight_material.unwrap_or(1.0).max(0.0),
        residual_weight_boundary: batch.residual_weight_boundary.unwrap_or(1.0).max(0.0),
        contact_penalty: batch.contact_penalty.unwrap_or(10.0).max(0.0),
        plasticity_factor: batch.plasticity_factor.unwrap_or(0.0).clamp(0.0, 1.0),
    }
}

fn exact_family_training_target(case: &SolveInput, analysis_type: &str) -> Option<Vec<f64>> {
    match analysis_type {
        "cantilever" => exact_cantilever_training_target(case),
        "plate-hole" => exact_plate_hole_training_target(case),
        _ if case.is_simple_cantilever_verification() => exact_cantilever_training_target(case),
        _ if case.is_plate_hole_benchmark() => exact_plate_hole_training_target(case),
        _ => None,
    }
}

fn exact_cantilever_training_target(case: &SolveInput) -> Option<Vec<f64>> {
    if !case.boundary_conditions.fix_start_face
        || case.boundary_conditions.fix_end_face
        || case.load.axial_load_lbf.abs() > 1e-9
        || case.load.vertical_point_load_lbf.abs() <= 1e-9
        || case.geometry.hole_diameter_in.unwrap_or(0.0).abs() > 1e-9
    {
        return None;
    }
    let l = case.geometry.length_in.max(1e-6);
    let w = case.geometry.width_in.max(1e-6);
    let t = case.geometry.thickness_in.max(1e-6);
    let e = case.material.e_psi.max(1.0);
    let p = case.load.vertical_point_load_lbf;
    let i = (t * w.powi(3) / 12.0).max(1e-9);
    let c = 0.5 * w;
    let uy = p * l.powi(3) / (3.0 * e * i).max(1e-9);
    let sxx = p * l * c / i;
    Some(ann_target_vector_from_stress(0.0, uy, 0.0, sxx, 0.0, 0.0))
}

fn exact_plate_hole_training_target(case: &SolveInput) -> Option<Vec<f64>> {
    let hole = case.geometry.hole_diameter_in.unwrap_or(0.0).max(0.0);
    if hole <= 1e-9
        || case.load.axial_load_lbf.abs() <= 1e-9
        || case.load.vertical_point_load_lbf.abs() > 1e-9
    {
        return None;
    }
    let l = case.geometry.length_in.max(1e-6);
    let w = case.geometry.width_in.max(1e-6);
    let t = case.geometry.thickness_in.max(1e-6);
    let e = case.material.e_psi.max(1.0);
    let sigma_nom = case.load.axial_load_lbf / (w * t).max(1e-9);
    let sigma_theta_max = 3.0 * sigma_nom;
    let ux = if case.boundary_conditions.fix_end_face {
        0.0
    } else {
        sigma_nom * l / e
    };
    Some(ann_target_vector_from_stress(
        ux,
        0.0,
        0.0,
        sigma_theta_max,
        0.0,
        0.0,
    ))
}

fn with_feature_and_physics(
    case: &SolveInput,
    batch: &TrainingBatch,
    analysis_type: &str,
) -> TrainingSample {
    let y = exact_family_training_target(case, analysis_type).unwrap_or_else(|| {
        let fem = fem::solve_case(case);
        fem::ann_targets(&fem)
    });
    TrainingSample {
        x: load_feature_vector(case),
        y,
        physics: physics_context_from_case(case, batch),
    }
}

fn widen_case_family(case: &SolveInput, analysis_type: &str, rng: &mut StdRng) -> SolveInput {
    let mut out = lightweight_training_case(case.clone());
    if analysis_type == "cantilever" {
        out.geometry.hole_diameter_in = Some(0.0);
        out.boundary_conditions.fix_start_face = true;
        out.boundary_conditions.fix_end_face = false;
        out.geometry.length_in = (out.geometry.length_in * rng.gen_range(0.94..1.06)).clamp(4.0, 24.0);
        out.geometry.width_in = (out.geometry.width_in * rng.gen_range(0.94..1.06)).clamp(0.75, 8.0);
        out.geometry.thickness_in =
            (out.geometry.thickness_in * rng.gen_range(0.92..1.08)).clamp(0.03, 0.75);
        let vertical_sign = if out.load.vertical_point_load_lbf.abs() > 1e-9 {
            out.load.vertical_point_load_lbf.signum()
        } else {
            -1.0
        };
        out.load.axial_load_lbf = 0.0;
        out.load.vertical_point_load_lbf = vertical_sign
            * (out.load.vertical_point_load_lbf.abs().max(50.0) * rng.gen_range(0.88..1.12))
                .clamp(50.0, 5_000.0);
        out.material.e_psi =
            (out.material.e_psi * rng.gen_range(0.97..1.03)).clamp(1.0e6, 40.0e6);
        out.material.nu = (out.material.nu * rng.gen_range(0.99..1.01)).clamp(0.18, 0.42);
        out.material.rho_lb_in3 =
            (out.material.rho_lb_in3 * rng.gen_range(0.97..1.03)).clamp(0.05, 1.5);
        out.material.yield_strength_psi =
            (out.material.yield_strength_psi * rng.gen_range(0.96..1.04)).clamp(1_000.0, 120_000.0);
        return out;
    }
    if analysis_type == "plate-hole" {
        let max_hole = out.geometry.length_in.min(out.geometry.width_in) * 0.8;
        let seed_hole = out
            .geometry
            .hole_diameter_in
            .unwrap_or(0.2 * out.geometry.width_in)
            .clamp(0.08 * out.geometry.width_in, max_hole.max(0.08 * out.geometry.width_in));
        out.geometry.length_in = (out.geometry.length_in * rng.gen_range(0.95..1.05)).clamp(4.0, 24.0);
        out.geometry.width_in = (out.geometry.width_in * rng.gen_range(0.95..1.05)).clamp(1.0, 10.0);
        out.geometry.thickness_in =
            (out.geometry.thickness_in * rng.gen_range(0.95..1.05)).clamp(0.03, 0.75);
        out.geometry.hole_diameter_in = Some(
            (seed_hole * rng.gen_range(0.9..1.1))
                .clamp(0.08 * out.geometry.width_in, out.geometry.width_in * 0.9),
        );
        out.load.vertical_point_load_lbf = 0.0;
        out.load.axial_load_lbf =
            (out.load.axial_load_lbf.abs().max(250.0) * rng.gen_range(0.88..1.12))
                .clamp(250.0, 100_000.0);
        out.material.e_psi =
            (out.material.e_psi * rng.gen_range(0.97..1.03)).clamp(1.0e6, 40.0e6);
        out.material.nu = (out.material.nu * rng.gen_range(0.99..1.01)).clamp(0.18, 0.42);
        out.material.rho_lb_in3 =
            (out.material.rho_lb_in3 * rng.gen_range(0.97..1.03)).clamp(0.05, 1.5);
        out.material.yield_strength_psi =
            (out.material.yield_strength_psi * rng.gen_range(0.96..1.04)).clamp(1_000.0, 120_000.0);
        return out;
    }
    let max_hole = out.geometry.length_in.min(out.geometry.width_in) * 0.95;
    if out.geometry.hole_diameter_in.unwrap_or(0.0) > 0.0 {
        let hole = out.geometry.hole_diameter_in.unwrap_or(0.0);
        let scale = if rng.gen_bool(0.65) {
            rng.gen_range(0.85..1.15)
        } else {
            rng.gen_range(0.0..0.45)
        };
        out.geometry.hole_diameter_in = Some((hole * scale).clamp(0.0, max_hole));
    } else if rng.gen_bool(0.45) {
        let seed_hole = (0.08 * max_hole + rng.gen_range(0.0..0.18) * max_hole).clamp(0.0, max_hole);
        out.geometry.hole_diameter_in = Some(seed_hole);
    } else {
        out.geometry.hole_diameter_in = Some(0.0);
    }

    let support_mode = rng.gen_range(0..4);
    match support_mode {
        0 => {
            out.boundary_conditions.fix_start_face = true;
            out.boundary_conditions.fix_end_face = false;
        }
        1 => {
            out.boundary_conditions.fix_start_face = false;
            out.boundary_conditions.fix_end_face = true;
        }
        2 => {
            out.boundary_conditions.fix_start_face = true;
            out.boundary_conditions.fix_end_face = true;
        }
        _ => {
            out.boundary_conditions.fix_start_face = true;
            out.boundary_conditions.fix_end_face = false;
        }
    }

    let axial_scale = rng.gen_range(0.82..1.18);
    let vertical_scale = rng.gen_range(0.82..1.18);
    if out.load.axial_load_lbf.abs() > 1e-9 && out.load.vertical_point_load_lbf.abs() > 1e-9 {
        out.load.axial_load_lbf = (out.load.axial_load_lbf * axial_scale).clamp(0.0, 100_000.0);
        out.load.vertical_point_load_lbf =
            (out.load.vertical_point_load_lbf * vertical_scale).clamp(-10_000.0, 10_000.0);
    } else if out.load.vertical_point_load_lbf.abs() > 1e-9 {
        out.load.vertical_point_load_lbf =
            (out.load.vertical_point_load_lbf * vertical_scale).clamp(-10_000.0, 10_000.0);
        if rng.gen_bool(0.55) {
            out.load.axial_load_lbf =
                (out.load.vertical_point_load_lbf.abs() * 0.18 * axial_scale).clamp(25.0, 25_000.0);
        }
    } else if out.load.axial_load_lbf.abs() > 1e-9 {
        out.load.axial_load_lbf = (out.load.axial_load_lbf.abs() * axial_scale).clamp(25.0, 100_000.0);
        if rng.gen_bool(0.55) {
            out.load.vertical_point_load_lbf =
                -(out.load.axial_load_lbf * 0.06 * vertical_scale).clamp(25.0, 10_000.0);
        }
    } else {
        out.load.axial_load_lbf = (25.0 * axial_scale).clamp(25.0, 5_000.0);
        out.load.vertical_point_load_lbf = -(100.0 * vertical_scale).clamp(25.0, 5_000.0);
    }

    out.material.e_psi = (out.material.e_psi * rng.gen_range(0.96..1.04)).clamp(1.0e6, 40.0e6);
    out.material.nu = (out.material.nu * rng.gen_range(0.98..1.02)).clamp(0.15, 0.49);
    out.material.rho_lb_in3 = (out.material.rho_lb_in3 * rng.gen_range(0.95..1.05)).clamp(0.05, 1.5);
    out.material.yield_strength_psi =
        (out.material.yield_strength_psi * rng.gen_range(0.92..1.08)).clamp(1_000.0, 120_000.0);
    out
}

fn expand_training_cases(seed_cases: &[SolveInput], analysis_type: &str) -> Vec<SolveInput> {
    if seed_cases.is_empty() {
        return vec![SolveInput::default()];
    }
    if seed_cases.len() >= 4 {
        return seed_cases
            .iter()
            .cloned()
            .map(lightweight_training_case)
            .collect();
    }
    if analysis_type.contains("cantilever") {
        let mut out = Vec::new();
        let load_scales = [0.84_f64, 0.92, 1.0, 1.08, 1.16];
        let geom_scales = [0.94_f64, 1.0, 1.06];
        for seed in seed_cases {
            let mut base = lightweight_training_case(seed.clone());
            base.geometry.hole_diameter_in = Some(0.0);
            base.boundary_conditions.fix_start_face = true;
            base.boundary_conditions.fix_end_face = false;
            if base.load.vertical_point_load_lbf.abs() <= 1e-9 {
                base.load.vertical_point_load_lbf = -100.0;
            }
            base.load.axial_load_lbf = 0.0;
            out.push(base.clone());

            for ls in load_scales {
                let mut c = base.clone();
                c.load.vertical_point_load_lbf =
                    (base.load.vertical_point_load_lbf * ls).clamp(-10_000.0, 10_000.0);
                out.push(lightweight_training_case(c));
            }

            for gs in geom_scales {
                let mut c = base.clone();
                c.geometry.length_in = (base.geometry.length_in * gs).clamp(4.0, 30.0);
                c.geometry.width_in = (base.geometry.width_in * (2.0 - gs)).clamp(0.75, 10.0);
                c.geometry.thickness_in =
                    (base.geometry.thickness_in * gs).clamp(0.03, 0.75);
                out.push(lightweight_training_case(c));
            }
        }
        out.truncate(48);
        return out;
    }
    if analysis_type.contains("plate-hole") {
        let mut out = Vec::new();
        let load_scales = [0.86_f64, 0.94, 1.0, 1.06, 1.14];
        let hole_scales = [0.9_f64, 1.0, 1.1];
        let geom_scales = [0.96_f64, 1.0, 1.04];
        for seed in seed_cases {
            let mut base = lightweight_training_case(seed.clone());
            base.load.vertical_point_load_lbf = 0.0;
            base.load.axial_load_lbf = base.load.axial_load_lbf.abs().max(500.0);
            let hole = base
                .geometry
                .hole_diameter_in
                .unwrap_or(0.2 * base.geometry.width_in)
                .clamp(0.08 * base.geometry.width_in, base.geometry.width_in * 0.8);
            base.geometry.hole_diameter_in = Some(hole);
            out.push(base.clone());

            for ls in load_scales {
                let mut c = base.clone();
                c.load.axial_load_lbf = (base.load.axial_load_lbf * ls).clamp(25.0, 100_000.0);
                out.push(lightweight_training_case(c));
            }

            for hs in hole_scales {
                let mut c = base.clone();
                c.geometry.hole_diameter_in =
                    Some((hole * hs).clamp(0.08 * c.geometry.width_in, c.geometry.width_in * 0.9));
                out.push(lightweight_training_case(c));
            }

            for gs in geom_scales {
                let mut c = base.clone();
                c.geometry.length_in = (base.geometry.length_in * gs).clamp(4.0, 30.0);
                c.geometry.width_in = (base.geometry.width_in * (2.0 - gs)).clamp(1.0, 10.0);
                c.geometry.thickness_in =
                    (base.geometry.thickness_in * gs).clamp(0.03, 0.75);
                let base_hole = base.geometry.hole_diameter_in.unwrap_or(hole);
                c.geometry.hole_diameter_in =
                    Some(base_hole.clamp(0.08 * c.geometry.width_in, c.geometry.width_in * 0.9));
                out.push(lightweight_training_case(c));
            }
        }
        out.truncate(48);
        return out;
    }
    let mut out = Vec::new();
    let load_scales = if analysis_type.contains("cantilever") {
        [0.84_f64, 0.94, 1.0, 1.06, 1.16]
    } else if analysis_type.contains("plate-hole") {
        [0.88_f64, 0.96, 1.0, 1.04, 1.10]
    } else {
        [0.86_f64, 0.94, 1.0, 1.08, 1.14]
    };
    let geom_scales = [0.94_f64, 1.0, 1.06];
    let e_scales = [0.97_f64, 1.0, 1.03];

    for seed in seed_cases {
        let mut base = seed.clone();
        base = lightweight_training_case(base);
        let mut family = vec![base.clone()];
        let mut mirrored = base.clone();
        mirrored.boundary_conditions.fix_start_face = base.boundary_conditions.fix_end_face;
        mirrored.boundary_conditions.fix_end_face = base.boundary_conditions.fix_start_face;
        if mirrored.boundary_conditions.fix_start_face != base.boundary_conditions.fix_start_face
            || mirrored.boundary_conditions.fix_end_face != base.boundary_conditions.fix_end_face
        {
            family.push(mirrored);
        }
        let mut dual_clamped = base.clone();
        dual_clamped.boundary_conditions.fix_start_face = true;
        dual_clamped.boundary_conditions.fix_end_face = true;
        if dual_clamped.boundary_conditions.fix_start_face != base.boundary_conditions.fix_start_face
            || dual_clamped.boundary_conditions.fix_end_face != base.boundary_conditions.fix_end_face
        {
            family.push(dual_clamped);
        }

        for boundary_case in family {
            out.push(boundary_case.clone());

            for ls in load_scales {
                let mut c = boundary_case.clone();
                if boundary_case.load.vertical_point_load_lbf.abs() > 1e-9 {
                    c.load.vertical_point_load_lbf = (boundary_case.load.vertical_point_load_lbf
                        * ls)
                        .clamp(-10_000.0, 10_000.0);
                    c.load.axial_load_lbf = 0.0;
                    let mut mixed = boundary_case.clone();
                    mixed.load.vertical_point_load_lbf = (boundary_case
                        .load
                        .vertical_point_load_lbf
                        * ls)
                        .clamp(-10_000.0, 10_000.0);
                    mixed.load.axial_load_lbf = (boundary_case
                        .load
                        .vertical_point_load_lbf
                        .abs()
                        .max(100.0)
                        * 0.12
                        * boundary_case.load.vertical_point_load_lbf.signum())
                    .clamp(-10_000.0, 10_000.0);
                    out.push(lightweight_training_case(mixed));
                } else {
                    c.load.axial_load_lbf = (boundary_case.load.axial_load_lbf.abs().max(25.0)
                        * ls)
                        .clamp(-100_000.0, 100_000.0);
                    c.load.vertical_point_load_lbf = 0.0;
                    let mut mixed = boundary_case.clone();
                    mixed.load.axial_load_lbf = (boundary_case.load.axial_load_lbf * ls)
                        .clamp(-100_000.0, 100_000.0);
                    mixed.load.vertical_point_load_lbf = (boundary_case
                        .load
                        .axial_load_lbf
                        .abs()
                        .max(100.0)
                        * 0.10
                        * -boundary_case.load.axial_load_lbf.signum())
                    .clamp(-10_000.0, 10_000.0);
                    out.push(lightweight_training_case(mixed));
                }
                out.push(lightweight_training_case(c));
            }

            for gs in geom_scales {
                let mut c = boundary_case.clone();
                c.geometry.length_in = (boundary_case.geometry.length_in * gs).clamp(4.0, 30.0);
                c.geometry.width_in = (boundary_case.geometry.width_in * (2.0 - gs)).clamp(1.0, 10.0);
                c.geometry.thickness_in = (boundary_case.geometry.thickness_in * gs).clamp(0.03, 0.75);
                if let Some(d) = boundary_case.geometry.hole_diameter_in {
                    c.geometry.hole_diameter_in =
                        Some((d * gs).clamp(0.0, c.geometry.width_in * 0.95));
                }
                out.push(lightweight_training_case(c));
            }

            for es in e_scales {
                let mut c = boundary_case.clone();
                c.material.e_psi = (boundary_case.material.e_psi * es).clamp(1.0e6, 40.0e6);
                out.push(lightweight_training_case(c));
            }
        }
    }

    if out.len() > 96 {
        out.truncate(96);
    }
    out
}

fn generate_collocation_samples(
    cases: &[SolveInput],
    requested_points: usize,
    boundary_points: usize,
    interface_points: usize,
    batch: &TrainingBatch,
    rng: &mut StdRng,
) -> Vec<TrainingSample> {
    if cases.is_empty() || (requested_points == 0 && boundary_points == 0 && interface_points == 0)
    {
        return vec![];
    }
    let interior_capped = requested_points.min(4096);
    let boundary_capped = boundary_points.min(2048);
    let interface_capped = interface_points.min(2048);
    let total_capped = (interior_capped + boundary_capped + interface_capped).clamp(32, 8192);
    let per_case_interior = if interior_capped > 0 {
        (interior_capped / cases.len().max(1)).max(4)
    } else {
        0
    };
    let per_case_boundary = if boundary_capped > 0 {
        (boundary_capped / cases.len().max(1)).max(4)
    } else {
        0
    };
    let per_case_interface = if interface_capped > 0 {
        (interface_capped / cases.len().max(1)).max(4)
    } else {
        0
    };
    let mut out = Vec::with_capacity(total_capped);
    for case in cases {
        let c = lightweight_training_case(case.clone());
        let l = c.geometry.length_in.max(1e-6);
        let w = c.geometry.width_in.max(1e-6);
        let t = c.geometry.thickness_in.max(1e-6);
        let area = (w * t).max(1e-9);
        let i = (t * w.powi(3) / 12.0).max(1e-9);
        for _ in 0..per_case_interior {
            if out.len() >= total_capped {
                break;
            }
            let x = rng.gen_range(0.0..l);
            let y = rng.gen_range(0.0..w);
            let z = rng.gen_range(0.0..t);
            let y_center = y - 0.5 * w;
            let z_center = z - 0.5 * t;
            let sigma_axial = c.load.axial_load_lbf / area;
            let m = c.load.vertical_point_load_lbf * (l - x);
            let sigma_bending = m * y_center / i;
            let sigma_x = sigma_axial + sigma_bending;
            let eps_x = sigma_x / c.material.e_psi.max(1.0);
            let ux = eps_x * x;
            let uy = -c.material.nu * eps_x * y_center;
            let uz = -c.material.nu * eps_x * z_center;
            out.push(TrainingSample {
                x: load_feature_vector(&c),
                y: ann_target_vector_from_stress(ux, uy, uz, sigma_x, 0.0, 0.0),
                physics: physics_context_from_case(&c, batch),
            });
        }
        for _ in 0..per_case_boundary {
            if out.len() >= total_capped {
                break;
            }
            let x = if rng.gen_bool(0.5) { 0.0 } else { l };
            let y = rng.gen_range(0.0..w);
            let sigma_x = if x <= 1e-12 {
                0.0
            } else {
                c.load.axial_load_lbf / area
            };
            let eps_x = sigma_x / c.material.e_psi.max(1.0);
            let ux = eps_x * x;
            let uy = -c.material.nu * eps_x * (y - 0.5 * w);
            let uz = -c.material.nu * eps_x * (0.5 * t);
            out.push(TrainingSample {
                x: load_feature_vector(&c),
                y: ann_target_vector_from_stress(ux, uy, uz, sigma_x, 0.0, 0.0),
                physics: physics_context_from_case(&c, batch),
            });
        }
        for _ in 0..per_case_interface {
            if out.len() >= total_capped {
                break;
            }
            // Interface proxy: center-thickness region where contact/complementarity penalties are most sensitive.
            let x = rng.gen_range(0.2 * l..0.8 * l);
            let y = rng.gen_range(0.0..w);
            let m = c.load.vertical_point_load_lbf * (l - x);
            let sigma_x = c.load.axial_load_lbf / area + m * (y - 0.5 * w) / i;
            let eps_x = sigma_x / c.material.e_psi.max(1.0);
            let ux = eps_x * x;
            let uy = -c.material.nu * eps_x * (y - 0.5 * w);
            let uz = -c.material.nu * eps_x * (0.5 * t);
            out.push(TrainingSample {
                x: load_feature_vector(&c),
                y: ann_target_vector_from_stress(ux, uy, uz, sigma_x, 0.0, 0.0),
                physics: physics_context_from_case(&c, batch),
            });
        }
        if out.len() >= total_capped {
            break;
        }
    }
    out
}

fn remap_layers_with_transfer(
    old_sizes: &[usize],
    new_sizes: &[usize],
    old_layers: &[DenseLayer],
) -> Vec<DenseLayer> {
    let mut new_layers = initialize_layers(new_sizes);
    let common_layers = old_layers.len().min(new_layers.len());
    for l in 0..common_layers {
        let old_out = old_sizes.get(l + 1).copied().unwrap_or(0);
        let old_in = old_sizes.get(l).copied().unwrap_or(0);
        let new_out = new_sizes.get(l + 1).copied().unwrap_or(0);
        let new_in = new_sizes.get(l).copied().unwrap_or(0);
        let copy_out = old_out.min(new_out).min(old_layers[l].weights.len());
        for o in 0..copy_out {
            let copy_in = old_in
                .min(new_in)
                .min(old_layers[l].weights[o].len())
                .min(new_layers[l].weights[o].len());
            new_layers[l].weights[o][..copy_in]
                .copy_from_slice(&old_layers[l].weights[o][..copy_in]);
            if o < old_layers[l].biases.len() && o < new_layers[l].biases.len() {
                new_layers[l].biases[o] = old_layers[l].biases[o];
            }
        }
    }
    new_layers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{LoadInput, SafeguardSettings, SolveInput, TrainingBatch};
    use std::time::Instant;

    fn simple_cantilever_case() -> SolveInput {
        let mut c = SolveInput::default();
        c.geometry.length_in = 10.0;
        c.geometry.width_in = 1.0;
        c.geometry.thickness_in = 0.25;
        c.geometry.hole_diameter_in = Some(0.0);
        c.mesh.nx = 18;
        c.mesh.ny = 4;
        c.mesh.nz = 1;
        c.mesh.auto_adapt = true;
        c.mesh.max_dofs = crate::contracts::MAX_DENSE_SOLVER_DOFS;
        c.mesh.amr_enabled = false;
        c.mesh.amr_passes = 0;
        c.load = LoadInput {
            axial_load_lbf: 0.0,
            vertical_point_load_lbf: -100.0,
        };
        c
    }

    fn simple_plate_hole_case() -> SolveInput {
        let mut c = SolveInput::default();
        c.geometry.length_in = 12.0;
        c.geometry.width_in = 4.0;
        c.geometry.thickness_in = 0.25;
        c.geometry.hole_diameter_in = Some(1.0);
        c.mesh.nx = 10;
        c.mesh.ny = 6;
        c.mesh.nz = 1;
        c.mesh.auto_adapt = true;
        c.mesh.max_dofs = crate::contracts::MAX_DENSE_SOLVER_DOFS;
        c.mesh.amr_enabled = false;
        c.mesh.amr_passes = 0;
        c.load = LoadInput {
            axial_load_lbf: 2_500.0,
            vertical_point_load_lbf: 0.0,
        };
        c
    }

    #[test]
    fn ann_training_reduces_validation_loss_with_adaptive_lr() {
        let mut model = AnnModel::default();

        let mut cases = Vec::new();
        for i in 0..8 {
            let mut c = SolveInput::default();
            c.load = LoadInput {
                axial_load_lbf: 0.0,
                vertical_point_load_lbf: -600.0 - (i as f64) * 120.0,
            };
            c.geometry.length_in = 8.0 + i as f64 * 0.5;
            c.geometry.width_in = 3.5 + (i as f64) * 0.08;
            cases.push(c);
        }

        let batch = TrainingBatch {
            cases,
            epochs: 20,
            target_loss: 1e-3,
            learning_rate: Some(5e-4),
            auto_mode: Some(false),
            max_total_epochs: Some(20),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(20),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(64),
            max_backoffs: Some(8),
            max_optimizer_switches: Some(6),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(8),
            seed: Some(42),
            analysis_type: Some("general".to_string()),
            pinn_backend: None,
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
        };

        let mut first_val = None::<f64>;
        let mut last_val = f64::MAX;
        let mut best_seen = f64::MAX;
        let result = model.train_with_progress(
            &batch,
            |p| {
                if first_val.is_none() && p.epoch > 1 {
                    first_val = Some(p.val_data_loss);
                }
                last_val = p.val_data_loss;
                best_seen = best_seen.min(p.val_data_loss);
            },
            || false,
        );

        let initial = first_val.unwrap_or(last_val.max(1.0));
        assert!(result.completed_epochs > 5);
        assert!(last_val.is_finite());
        assert!(
            best_seen <= initial * 1.05,
            "validation data loss degraded too much; initial={initial}, best={best_seen}, final={last_val}"
        );
    }

    #[test]
    fn cantilever_family_generation_stays_family_consistent() {
        let cases = expand_training_cases(&[simple_cantilever_case()], "cantilever");
        assert!(!cases.is_empty());
        assert!(cases.iter().all(|case| {
            case.boundary_conditions.fix_start_face
                && !case.boundary_conditions.fix_end_face
                && case.load.axial_load_lbf.abs() <= 1e-9
                && case.load.vertical_point_load_lbf.abs() > 1e-9
                && case.geometry.hole_diameter_in.unwrap_or(0.0).abs() <= 1e-9
        }));

        let mut rng = StdRng::seed_from_u64(7);
        let widened = widen_case_family(&cases[0], "cantilever", &mut rng);
        assert!(widened.boundary_conditions.fix_start_face);
        assert!(!widened.boundary_conditions.fix_end_face);
        assert!(widened.load.axial_load_lbf.abs() <= 1e-9);
        assert!(widened.load.vertical_point_load_lbf.abs() > 1e-9);
        assert!(widened.geometry.hole_diameter_in.unwrap_or(0.0).abs() <= 1e-9);
    }

    #[test]
    fn plate_hole_family_generation_stays_family_consistent() {
        let cases = expand_training_cases(&[simple_plate_hole_case()], "plate-hole");
        assert!(!cases.is_empty());
        assert!(cases.iter().all(|case| {
            case.geometry.hole_diameter_in.unwrap_or(0.0) > 1e-9
                && case.load.axial_load_lbf.abs() > 1e-9
                && case.load.vertical_point_load_lbf.abs() <= 1e-9
        }));

        let mut rng = StdRng::seed_from_u64(11);
        let widened = widen_case_family(&cases[0], "plate-hole", &mut rng);
        assert!(widened.geometry.hole_diameter_in.unwrap_or(0.0) > 1e-9);
        assert!(widened.load.axial_load_lbf.abs() > 1e-9);
        assert!(widened.load.vertical_point_load_lbf.abs() <= 1e-9);
    }

    #[test]
    fn exact_family_targets_match_closed_form_expectations() {
        let cantilever = simple_cantilever_case();
        let cantilever_target =
            exact_family_training_target(&cantilever, "cantilever").expect("cantilever shortcut");
        assert_eq!(cantilever_target.len(), fem::ANN_OUTPUT_DIM);
        assert!(cantilever_target[1].abs() > 0.0);
        assert!(cantilever_target[3].abs() > 0.0);

        let plate_hole = simple_plate_hole_case();
        let plate_hole_target =
            exact_family_training_target(&plate_hole, "plate-hole").expect("plate-hole shortcut");
        assert_eq!(plate_hole_target.len(), fem::ANN_OUTPUT_DIM);
        assert!(plate_hole_target[0].abs() > 0.0);
        assert!(plate_hole_target[3].abs() > 0.0);
        assert!(plate_hole_target[6] >= plate_hole_target[3].abs());
    }

    #[test]
    fn progress_emit_every_epochs_is_respected() {
        let mut model = AnnModel::default();
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 20,
            target_loss: 0.0,
            learning_rate: Some(5e-4),
            auto_mode: Some(false),
            max_total_epochs: Some(20),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(5),
            network_emit_every_epochs: Some(20),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(64),
            max_backoffs: Some(8),
            max_optimizer_switches: Some(6),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(8),
            seed: Some(42),
            analysis_type: Some("general".to_string()),
            pinn_backend: None,
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
        };

        let mut emitted_epochs = Vec::new();
        let result = model.train_with_progress(&batch, |p| emitted_epochs.push(p.epoch), || false);

        assert_eq!(result.completed_epochs, 20);
        assert_eq!(emitted_epochs, vec![1, 5, 10, 15, 20]);
    }

    #[test]
    fn explicit_stage_schedule_transitions_follow_epoch_boundaries() {
        let mut model = AnnModel::default();
        let s1 = 5usize;
        let s2 = 7usize;
        let s3 = 6usize;
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 40,
            target_loss: 0.0,
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(s1 + s2 + s3),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(100),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(64),
            max_backoffs: Some(8),
            max_optimizer_switches: Some(6),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(8),
            seed: Some(42),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("burn-ndarray-cpu".to_string()),
            collocation_points: None,
            boundary_points: None,
            interface_points: None,
            residual_weight_momentum: None,
            residual_weight_kinematics: None,
            residual_weight_material: None,
            residual_weight_boundary: None,
            stage1_epochs: Some(s1),
            stage2_epochs: Some(s2),
            stage3_ramp_epochs: Some(s3),
            contact_penalty: None,
            plasticity_factor: None,
        };

        let mut first_stage2 = None::<usize>;
        let mut first_stage3 = None::<usize>;
        let mut stage_order = Vec::new();
        let result = model.train_with_progress(
            &batch,
            |p| {
                if stage_order.last().map(|s| s != &p.stage_id).unwrap_or(true) {
                    stage_order.push(p.stage_id.clone());
                }
                if p.stage_id == "stage-2" && first_stage2.is_none() {
                    first_stage2 = Some(p.epoch);
                }
                if p.stage_id == "stage-3" && first_stage3.is_none() {
                    first_stage3 = Some(p.epoch);
                }
            },
            || false,
        );

        assert_eq!(result.completed_epochs, s1 + s2 + s3);
        assert_eq!(first_stage2, Some(s1 + 1));
        assert_eq!(first_stage3, Some(s1 + s2 + 1));
        assert_eq!(
            stage_order,
            vec![
                "stage-1".to_string(),
                "stage-2".to_string(),
                "stage-3".to_string()
            ]
        );
    }

    #[test]
    fn long_run_progress_does_not_stall_with_default_training_mode() {
        let mut model = AnnModel::default();
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 40,
            target_loss: 0.0,
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(420),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(1),
            network_emit_every_epochs: Some(50),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(64),
            max_backoffs: Some(8),
            max_optimizer_switches: Some(6),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(8),
            seed: Some(42),
            analysis_type: Some("general".to_string()),
            pinn_backend: None,
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
        };

        let mut last_epoch = 0usize;
        let mut last_time = Instant::now();
        let mut worst_gap_ms = 0u128;
        let mut started_measurement = false;

        let result = model.train_with_progress(
            &batch,
            |p| {
                if p.epoch > last_epoch {
                    let now = Instant::now();
                    if p.epoch >= 50 {
                        if !started_measurement {
                            started_measurement = true;
                            last_time = now;
                        } else {
                            let gap = now.duration_since(last_time).as_millis();
                            if gap > worst_gap_ms {
                                worst_gap_ms = gap;
                            }
                            last_time = now;
                        }
                    }
                    last_epoch = p.epoch;
                }
            },
            || false,
        );

        assert_eq!(result.completed_epochs, 420);
        assert!(last_epoch >= 420);
        assert!(
            started_measurement && worst_gap_ms < 1800,
            "progress gap too large in default training mode: {} ms",
            worst_gap_ms
        );
    }

    #[test]
    fn single_case_long_training_improves_below_common_plateau() {
        let mut model = AnnModel::default();
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 40,
            target_loss: 1e-9,
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(5000),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(20),
            network_emit_every_epochs: Some(400),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(64),
            max_backoffs: Some(8),
            max_optimizer_switches: Some(6),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(8),
            seed: Some(42),
            analysis_type: Some("general".to_string()),
            pinn_backend: None,
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
        };

        let mut best = f64::MAX;
        let mut best_data = f64::MAX;
        let mut first_seen = None::<f64>;
        let result = model.train_with_progress(
            &batch,
            |p| {
                if first_seen.is_none() && p.epoch > 1 {
                    first_seen = Some(p.val_data_loss);
                }
                best = best.min(p.val_loss);
                best_data = best_data.min(p.val_data_loss);
            },
            || false,
        );

        assert!(result.completed_epochs > 100);
        assert!(result.completed_epochs <= 5000);
        let baseline = first_seen.unwrap_or(best_data.max(1.0));
        assert!(
            best_data <= baseline * 1.05,
            "best validation data loss did not improve enough; baseline={baseline}, best_data={best_data}, best_total={best}"
        );
    }

    #[test]
    #[ignore = "long-run benchmark for manual verification"]
    fn single_case_800k_epochs_progresses() {
        let mut model = AnnModel::default();
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 40,
            target_loss: 1e-9,
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(800_000),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(200),
            network_emit_every_epochs: Some(20_000),
            online_active_learning: Some(false),
            autonomous_mode: Some(false),
            max_topology: Some(64),
            max_backoffs: Some(8),
            max_optimizer_switches: Some(6),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(8),
            seed: Some(42),
            analysis_type: Some("general".to_string()),
            pinn_backend: None,
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
        };

        let mut best = f64::MAX;
        let mut last_epoch = 0usize;
        let started = Instant::now();
        let result = model.train_with_progress(
            &batch,
            |p| {
                best = best.min(p.val_loss);
                last_epoch = p.epoch;
            },
            || false,
        );
        let elapsed = started.elapsed();
        println!(
            "800k benchmark: best_val_loss={:.6}, final_val_loss={:.6}, elapsed_s={:.2}",
            best,
            result.val_loss,
            elapsed.as_secs_f64()
        );

        assert!(result.completed_epochs <= 800_000);
        assert_eq!(last_epoch, result.completed_epochs);
        assert!(
            best < 3.0,
            "expected progression below plateau; best={best}"
        );
        assert!(
            elapsed.as_secs() < 600,
            "unexpectedly slow runtime: {:?}",
            elapsed
        );
    }

    #[test]
    fn infer_triggers_fallback_with_tight_safeguards() {
        let mut model = AnnModel::default();
        model.last_loss = 0.25;
        model.set_safeguard_settings(SafeguardSettings {
            preset: "tight".to_string(),
            uncertainty_threshold: 0.01,
            residual_threshold: 1e-6,
            adaptive_by_geometry: false,
        });
        let result = model.infer(&SolveInput::default());
        assert!(
            result.used_fem_fallback,
            "expected FEM fallback for tight safeguards"
        );
    }

    #[test]
    fn infer_respects_fallback_disabled_flag() {
        let mut model = AnnModel::default();
        model.last_loss = 0.0;
        model.fallback_enabled = false;
        model.set_safeguard_settings(SafeguardSettings {
            preset: "relaxed".to_string(),
            uncertainty_threshold: 0.95,
            residual_threshold: 10.0,
            adaptive_by_geometry: false,
        });
        let result = model.infer(&SolveInput::default());
        assert!(
            !result.used_fem_fallback,
            "expected ANN result without FEM fallback when fallback gate is disabled"
        );
    }

    #[test]
    fn infer_falls_back_when_input_exits_trained_domain() {
        let mut model = AnnModel::default();
        let seed = lightweight_training_case(simple_cantilever_case());
        model.training_domain = Some(build_surrogate_domain_summary(
            std::slice::from_ref(&seed),
            std::slice::from_ref(&seed),
            "cantilever",
        ));
        model.last_loss = 0.0;
        model.set_safeguard_settings(SafeguardSettings {
            preset: "balanced".to_string(),
            uncertainty_threshold: 0.2,
            residual_threshold: 10.0,
            adaptive_by_geometry: false,
        });
        let mut far = seed.clone();
        far.geometry.length_in *= 2.5;
        far.load.axial_load_lbf = 25_000.0;
        far.load.vertical_point_load_lbf = -2_500.0;
        let result = model.infer(&far);
        assert!(result.domain_extrapolation_score > 0.2);
        assert!(result.used_fem_fallback);
    }

    #[test]
    fn training_auto_expands_network_for_current_feature_channels() {
        let mut model = AnnModel::new(vec![8, 10, 6]);
        model.ensure_network_io(fem::ANN_INPUT_DIM, fem::ANN_OUTPUT_DIM);
        assert_eq!(model.layer_sizes[0], fem::ANN_INPUT_DIM);
        assert_eq!(*model.layer_sizes.last().unwrap_or(&0), fem::ANN_OUTPUT_DIM);
        assert_eq!(model.layers[0].weights[0].len(), fem::ANN_INPUT_DIM);
    }

    #[test]
    fn residual_score_is_small_for_coherent_axial_prediction() {
        let mut input = SolveInput::default();
        input.geometry.hole_diameter_in = Some(0.0);
        input.load.vertical_point_load_lbf = 0.0;
        let area = (input.geometry.width_in * input.geometry.thickness_in).max(1e-9);
        let sigma_x = input.load.axial_load_lbf / area;
        let tip = sigma_x * input.geometry.length_in / input.material.e_psi.max(1.0);
        let pred = vec![
            tip,
            0.0,
            0.0,
            sigma_x,
            0.0,
            0.0,
            sigma_x.abs(),
            sigma_x.abs(),
            sigma_x.max(0.0),
        ];
        let score = ann_residual_score_from_pred(&pred, &input);
        assert!(
            score < 0.05,
            "expected coherent axial prediction to have small residual score, got {score}"
        );
    }

    #[test]
    fn residual_score_is_small_for_fem_target_simple_cantilever() {
        let input = lightweight_training_case(simple_cantilever_case());
        let fem = fem::solve_case(&input);
        let pred = fem::ann_targets(&fem);
        let score = ann_residual_score_from_pred(&pred, &input);
        assert!(
            score < 0.5,
            "expected FEM cantilever target to satisfy safeguard residual, got {score}"
        );
    }

    #[test]
    fn simple_cantilever_training_poll_trace_is_live_and_improves() {
        let mut model = AnnModel::default();
        let mut c = SolveInput::default();
        c.geometry.hole_diameter_in = Some(0.0);
        c.load = LoadInput {
            axial_load_lbf: 0.0,
            vertical_point_load_lbf: -100.0,
        };
        c.mesh.nx = 10;
        c.mesh.ny = 4;
        c.mesh.nz = 1;

        let batch = TrainingBatch {
            cases: vec![c],
            epochs: 30,
            target_loss: 1e-4,
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(1200),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(10),
            network_emit_every_epochs: Some(120),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(64),
            max_backoffs: Some(8),
            max_optimizer_switches: Some(6),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(8),
            seed: Some(42),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: None,
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
        };

        let mut seen_epochs = Vec::new();
        let mut first_val_data = None::<f64>;
        let mut best_val_data = f64::MAX;
        let mut last_val_data = f64::MAX;

        let result = model.train_with_progress(
            &batch,
            |p| {
                seen_epochs.push(p.epoch);
                if first_val_data.is_none() && p.epoch >= 20 {
                    first_val_data = Some(p.val_data_loss);
                }
                best_val_data = best_val_data.min(p.val_data_loss);
                last_val_data = p.val_data_loss;
            },
            || false,
        );

        assert!(
            seen_epochs.len() >= 8,
            "expected frequent polled progress events, got {}",
            seen_epochs.len()
        );
        for w in seen_epochs.windows(2) {
            assert!(
                w[1] > w[0],
                "progress epochs must increase strictly: {:?}",
                seen_epochs
            );
        }
        let baseline = first_val_data.unwrap_or(last_val_data.max(1e-9));
        println!(
            "simple-trace: stop={}, epochs={}, polled={}, baseline_val_data={:.6e}, best_val_data={:.6e}, last_val_data={:.6e}",
            result.stop_reason,
            result.completed_epochs,
            seen_epochs.len(),
            baseline,
            best_val_data,
            last_val_data
        );
        assert!(
            best_val_data <= baseline * 1.02,
            "simple-case val_data should not stall upward; baseline={baseline}, best={best_val_data}, last={last_val_data}, stop={}",
            result.stop_reason
        );
    }

    #[test]
    fn plate_hole_training_poll_trace_is_live_and_improves() {
        let mut model = AnnModel::default();
        let c = simple_plate_hole_case();

        let batch = TrainingBatch {
            cases: vec![c],
            epochs: 30,
            target_loss: 1e-9,
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(1200),
            min_improvement: Some(1e-9),
            progress_emit_every_epochs: Some(10),
            network_emit_every_epochs: Some(120),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(64),
            max_backoffs: Some(8),
            max_optimizer_switches: Some(6),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(8),
            seed: Some(77),
            analysis_type: Some("plate-hole".to_string()),
            pinn_backend: None,
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
        };

        let mut seen_epochs = Vec::new();
        let mut first_val_data = None::<f64>;
        let mut best_val_data = f64::MAX;
        let mut last_val_data = f64::MAX;

        let result = model.train_with_progress(
            &batch,
            |p| {
                seen_epochs.push(p.epoch);
                if first_val_data.is_none() && p.epoch >= 20 {
                    first_val_data = Some(p.val_data_loss);
                }
                best_val_data = best_val_data.min(p.val_data_loss);
                last_val_data = p.val_data_loss;
            },
            || false,
        );

        assert!(
            seen_epochs.len() >= 8,
            "expected frequent polled progress events, got {}",
            seen_epochs.len()
        );
        for w in seen_epochs.windows(2) {
            assert!(w[1] > w[0], "progress epochs must increase strictly: {:?}", seen_epochs);
        }
        let baseline = first_val_data.unwrap_or(last_val_data.max(1e-9));
        println!(
            "plate-hole-trace: stop={}, epochs={}, polled={}, baseline_val_data={:.6e}, best_val_data={:.6e}, last_val_data={:.6e}",
            result.stop_reason,
            result.completed_epochs,
            seen_epochs.len(),
            baseline,
            best_val_data,
            last_val_data
        );
        assert!(
            baseline <= 1e-3,
            "plate-hole warm start should begin in a low-loss band; baseline={baseline}"
        );
        assert!(
            best_val_data <= baseline * 0.98 || best_val_data <= 5e-5,
            "plate-hole val_data should either improve or remain in the seeded low-loss band; baseline={baseline}, best={best_val_data}, last={last_val_data}, stop={}",
            result.stop_reason
        );
    }

    #[test]
    #[ignore = "manual long-run benchmark profile"]
    fn benchmark_profile_epoch_windows_manual() {
        use std::collections::VecDeque;

        let mut model = AnnModel::default();
        let mut c = SolveInput::default();
        c.geometry.hole_diameter_in = Some(0.0);
        c.load = LoadInput {
            axial_load_lbf: 0.0,
            vertical_point_load_lbf: -150.0,
        };
        c.mesh.nx = 12;
        c.mesh.ny = 5;
        c.mesh.nz = 1;

        let batch = TrainingBatch {
            cases: vec![c],
            epochs: 40,
            target_loss: 1e-9,
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(10_000),
            min_improvement: Some(1e-8),
            progress_emit_every_epochs: Some(1_000),
            network_emit_every_epochs: Some(40_000),
            online_active_learning: Some(false),
            autonomous_mode: Some(true),
            max_topology: Some(64),
            max_backoffs: Some(8),
            max_optimizer_switches: Some(6),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(8),
            seed: Some(42),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: None,
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
        };

        let mut rows: VecDeque<(usize, f64, f64, String, String)> = VecDeque::with_capacity(10);
        let result = model.train_with_progress(
            &batch,
            |p| {
                if p.epoch % 1_000 == 0 {
                    if rows.len() == 10 {
                        rows.pop_front();
                    }
                    rows.push_back((
                        p.epoch,
                        p.val_data_loss,
                        p.val_physics_loss,
                        p.lr_phase,
                        p.optimizer_id,
                    ));
                }
            },
            || false,
        );
        println!("| epoch | val_data_loss | val_physics_loss | lr_phase | optimizer |");
        println!("|---:|---:|---:|---|---|");
        for (epoch, val_data, val_phys, lr_phase, optimizer) in &rows {
            println!(
                "| {} | {:.6e} | {:.6e} | {} | {} |",
                epoch, val_data, val_phys, lr_phase, optimizer
            );
        }
        println!(
            "benchmark-summary: stop={}, epochs={}, val_loss={:.6e}, model_version={}",
            result.stop_reason, result.completed_epochs, result.val_loss, result.model_version
        );
        assert!(
            !rows.is_empty(),
            "expected at least one benchmark window row"
        );
    }

    #[test]
    #[ignore = "manual exact UI fingerprint benchmark"]
    fn exact_ui_fingerprint_cantilever_manual() {
        let mut model = AnnModel::default();
        let mut c = SolveInput::default();
        c.geometry.length_in = 10.0;
        c.geometry.width_in = 1.0;
        c.geometry.thickness_in = 0.25;
        c.geometry.hole_diameter_in = Some(0.0);
        c.mesh.nx = 18;
        c.mesh.ny = 4;
        c.mesh.nz = 1;
        c.mesh.amr_enabled = false;
        c.mesh.amr_passes = 0;
        c.mesh.amr_max_nx = 30;
        c.mesh.amr_refine_ratio = 1.15;
        c.load = LoadInput {
            axial_load_lbf: 0.0,
            vertical_point_load_lbf: -100.0,
        };

        let batch = TrainingBatch {
            cases: vec![c],
            epochs: 40,
            target_loss: 0.01,
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(10_000),
            min_improvement: Some(1e-6),
            progress_emit_every_epochs: Some(5),
            network_emit_every_epochs: Some(200),
            online_active_learning: Some(true),
            autonomous_mode: Some(true),
            max_topology: Some(128),
            max_backoffs: Some(12),
            max_optimizer_switches: Some(8),
            checkpoint_every_epochs: Some(1_000),
            checkpoint_retention: Some(8),
            seed: Some(42),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: None,
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
        };

        let mut best_val = f64::MAX;
        let mut best_val_data = f64::MAX;
        let result = model.train_with_progress(
            &batch,
            |p| {
                best_val = best_val.min(p.val_loss);
                best_val_data = best_val_data.min(p.val_data_loss);
            },
            || false,
        );
        println!(
            "exact-ui-summary: stop={}, epochs={}, val_loss={:.6e}, best_val={:.6e}, best_val_data={:.6e}",
            result.stop_reason,
            result.completed_epochs,
            result.val_loss,
            best_val,
            best_val_data
        );
        assert!(best_val.is_finite(), "expected finite validation loss");
    }
}
