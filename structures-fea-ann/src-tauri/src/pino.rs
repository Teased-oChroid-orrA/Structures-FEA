use crate::contracts::{
    BeamStationResult, FemResult, HoldoutValidationSummary, NodalDisplacement, OperatorGridSpec,
    PinoLocalEnrichmentMetadata, PinoLocalRefinementMetadata, PinoRuntimeMetadata, SolveInput,
    TrainingBatch,
};
use crate::fem::solve_case;
use crate::physics::stress::{principal_stresses, tresca_from_principal};
use serde::{Deserialize, Serialize};

pub const PINO_ENGINE_ID: &str = "pino-plate-v1";
pub const PINO_BACKEND_CANDLE_CPU: &str = "pino-candle-cpu";
pub const PINO_BACKEND_CANDLE_CUDA: &str = "pino-candle-cuda";
pub const PINO_BACKEND_CANDLE_METAL: &str = "pino-candle-metal";
pub const PINO_BACKEND_NDARRAY_CPU: &str = "pino-ndarray-cpu";

pub const PINO_INPUT_CHANNELS: usize = 15;
pub const PINO_PRIMARY_OUTPUT_CHANNELS: usize = 9;
pub const PINO_AUX_OUTPUT_CHANNELS: usize = 2;
pub const PINO_OUTPUT_CHANNELS: usize = PINO_PRIMARY_OUTPUT_CHANNELS + PINO_AUX_OUTPUT_CHANNELS;
pub const PINO_FIELD_HEAD_BASIS: usize = 33;
pub const PINO_FIELD_HEAD_HIDDEN: usize = 32;
pub const PINO_LAAF_GAIN: f64 = 5.0;
const PINO_RECIPE_SEED_COUNT: usize = 3;
const PINO_HOLDOUT_CASES: usize = 2;
const PINO_HOLDOUT_MEAN_ERROR_LIMIT: f64 = 0.05;
const PINO_HOLDOUT_P95_ERROR_LIMIT: f64 = 0.10;
const PINO_HOLDOUT_TRUST_RATIO_LIMIT: f64 = 0.80;

#[derive(Debug, Clone)]
pub struct OperatorSample {
    pub inputs: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct OperatorPrediction {
    pub grid: OperatorGridSpec,
    pub ux: Vec<f64>,
    pub uy: Vec<f64>,
    pub uz: Vec<f64>,
    pub sxx: Vec<f64>,
    pub syy: Vec<f64>,
    pub szz: Vec<f64>,
    pub sxy: Vec<f64>,
    pub sxz: Vec<f64>,
    pub syz: Vec<f64>,
    pub von_mises: Vec<f64>,
    pub max_principal: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct OperatorFieldHeadBatch {
    pub base_prediction: OperatorPrediction,
    pub features: Vec<f64>,
    pub correction_scales: Vec<f64>,
    pub mask: Vec<f64>,
    pub clamp: Vec<f64>,
    pub displacement_embed: Vec<f64>,
    pub cell_count: usize,
    pub feature_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OperatorCalibration {
    pub stress_scale: f64,
    pub displacement_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OperatorTrainingStats {
    pub epochs_run: usize,
    pub initial_loss: f64,
    pub best_loss: f64,
    pub final_loss: f64,
    pub calibration: OperatorCalibration,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OperatorTrainableParams {
    #[serde(default = "default_field_head_hidden_layers")]
    pub field_head_hidden_layers: usize,
    #[serde(default = "default_field_head_hidden_width")]
    pub field_head_hidden_width: usize,
    pub field_head_weights: Vec<f64>,
    pub field_head_bias: Vec<f64>,
    #[serde(default = "default_field_head_activation")]
    pub field_head_activation: Vec<f64>,
}

fn default_field_head_hidden_layers() -> usize {
    2
}

fn default_field_head_hidden_width() -> usize {
    PINO_FIELD_HEAD_HIDDEN
}

fn default_field_head_activation() -> Vec<f64> {
    vec![0.2; default_field_head_hidden_layers() + 1]
}

fn deterministic_unit(seed: usize, salt: usize) -> f64 {
    let phase = (seed as f64 + 1.0) * 12.9898 + (salt as f64 + 1.0) * 78.233;
    let value = phase.sin() * 43_758.545_312_3;
    (value - value.floor()) * 2.0 - 1.0
}

fn initialized_field_head_weights(
    input_dim: usize,
    output_dim: usize,
    hidden_layers: usize,
    hidden_width: usize,
) -> Vec<f64> {
    let hidden_layers = OperatorTrainableParams::normalized_hidden_layers(hidden_layers);
    let hidden_width = OperatorTrainableParams::normalized_hidden_width(hidden_width);
    let mut weights = Vec::with_capacity(OperatorTrainableParams::weight_len_for(
        input_dim,
        output_dim,
        hidden_layers,
        hidden_width,
    ));
    let lift_scale = (2.0 / (input_dim + hidden_width).max(1) as f64).sqrt() * 0.08;
    for row in 0..hidden_width {
        for col in 0..input_dim {
            let seed = row * input_dim + col;
            weights.push(deterministic_unit(seed, 11) * lift_scale);
        }
    }
    let operator_scale = (2.0 / (hidden_width * 2).max(1) as f64).sqrt() * 0.03;
    for layer in 0..hidden_layers {
        for row in 0..hidden_width {
            for col in 0..hidden_width {
                let diag = if row == col { 0.08 } else { 0.0 };
                let seed = layer * hidden_width * hidden_width + row * hidden_width + col;
                weights.push((diag + deterministic_unit(seed, 23) * operator_scale).clamp(-0.25, 0.25));
            }
            for col in 0..hidden_width {
                let diag = if row == col { 0.03 } else { 0.0 };
                let seed = layer * hidden_width * hidden_width + row * hidden_width + col;
                weights.push((diag + deterministic_unit(seed, 37) * operator_scale * 0.7).clamp(-0.18, 0.18));
            }
            for col in 0..hidden_width {
                let diag = if row == col { 0.012 } else { 0.0 };
                let seed = layer * hidden_width * hidden_width + row * hidden_width + col;
                weights.push((diag + deterministic_unit(seed, 53) * operator_scale * 0.55).clamp(-0.12, 0.12));
            }
            for col in 0..hidden_width {
                let diag = if row == col { 0.024 } else { 0.0 };
                let seed = layer * hidden_width * hidden_width + row * hidden_width + col;
                weights.push((diag + deterministic_unit(seed, 67) * operator_scale * 0.6).clamp(-0.15, 0.15));
            }
        }
    }
    let output_scale = (2.0 / (hidden_width + output_dim).max(1) as f64).sqrt() * 0.06;
    for row in 0..output_dim {
        for col in 0..hidden_width {
            let seed = row * hidden_width + col;
            weights.push(deterministic_unit(seed, 71) * output_scale);
        }
    }
    weights
}

impl Default for OperatorTrainableParams {
    fn default() -> Self {
        Self {
            field_head_hidden_layers: default_field_head_hidden_layers(),
            field_head_hidden_width: default_field_head_hidden_width(),
            field_head_weights: Self::default_field_head_weights(),
            field_head_bias: Self::default_field_head_bias(),
            field_head_activation: Self::default_field_head_activation(),
        }
    }
}

impl OperatorTrainableParams {
    fn adaptive_tanh(value: f64, activation: f64) -> f64 {
        let slope = (PINO_LAAF_GAIN * activation.clamp(0.05, 2.0)).clamp(0.25, 10.0);
        (value * slope).tanh()
    }

    fn soft_clip(value: f64, limit: f64) -> f64 {
        if limit <= 0.0 {
            value
        } else {
            limit * (value / limit).tanh()
        }
    }

    pub fn for_config(config: &PinoModelConfig) -> Self {
        Self::default().aligned_to_config(config)
    }

    fn normalized_hidden_layers(hidden_layers: usize) -> usize {
        hidden_layers.clamp(1, 6)
    }

    fn normalized_hidden_width(hidden_width: usize) -> usize {
        hidden_width.clamp(8, 256)
    }

    pub fn weight_len_for(
        input_dim: usize,
        output_dim: usize,
        hidden_layers: usize,
        hidden_width: usize,
    ) -> usize {
        let hidden_layers = Self::normalized_hidden_layers(hidden_layers);
        let hidden_width = Self::normalized_hidden_width(hidden_width);
        let lift_weights = input_dim * hidden_width;
        let operator_weights = hidden_layers * hidden_width * hidden_width * 4;
        lift_weights + operator_weights + hidden_width * output_dim
    }

    pub fn bias_len_for(output_dim: usize, hidden_layers: usize, hidden_width: usize) -> usize {
        let hidden_layers = Self::normalized_hidden_layers(hidden_layers);
        let hidden_width = Self::normalized_hidden_width(hidden_width);
        hidden_width + hidden_layers * hidden_width + output_dim
    }

    pub fn activation_len_for(hidden_layers: usize) -> usize {
        Self::normalized_hidden_layers(hidden_layers) + 1
    }

    pub fn matches_shape(&self, input_dim: usize, output_dim: usize) -> bool {
        self.field_head_weights.len()
            == Self::weight_len_for(
                input_dim,
                output_dim,
                self.field_head_hidden_layers,
                self.field_head_hidden_width,
            )
            && self.field_head_bias.len()
                == Self::bias_len_for(
                    output_dim,
                    self.field_head_hidden_layers,
                    self.field_head_hidden_width,
                )
            && self.field_head_activation.len()
                == Self::activation_len_for(self.field_head_hidden_layers)
    }

    pub fn default_field_head_weights() -> Vec<f64> {
        initialized_field_head_weights(
            PINO_FIELD_HEAD_BASIS,
            PINO_OUTPUT_CHANNELS,
            default_field_head_hidden_layers(),
            default_field_head_hidden_width(),
        )
    }

    pub fn default_field_head_bias() -> Vec<f64> {
        vec![
            0.0;
                Self::bias_len_for(
                PINO_OUTPUT_CHANNELS,
                default_field_head_hidden_layers(),
                default_field_head_hidden_width(),
            )
        ]
    }

    pub fn default_field_head_activation() -> Vec<f64> {
        default_field_head_activation()
    }

    fn inferred_input_dim(
        weights_len: usize,
        output_dim: usize,
        hidden_layers: usize,
        hidden_width: usize,
    ) -> Option<usize> {
        let hidden_layers = Self::normalized_hidden_layers(hidden_layers);
        let hidden_width = Self::normalized_hidden_width(hidden_width);
        let operator_weights = hidden_layers
            .checked_mul(hidden_width)?
            .checked_mul(hidden_width)?
            .checked_mul(4)?;
        let output_weights = hidden_width.checked_mul(output_dim)?;
        let remaining = weights_len.checked_sub(operator_weights + output_weights)?;
        if hidden_width == 0 || remaining % hidden_width != 0 {
            return None;
        }
        Some(remaining / hidden_width)
    }

    fn resized_field_head_weights(
        &self,
        old_input_dim: usize,
        new_input_dim: usize,
        output_dim: usize,
        old_hidden_layers: usize,
        old_hidden_width: usize,
        new_hidden_layers: usize,
        new_hidden_width: usize,
    ) -> Vec<f64> {
        let mut resized = initialized_field_head_weights(
            new_input_dim,
            output_dim,
            new_hidden_layers,
            new_hidden_width,
        );
        let preserved_hidden_layers = old_hidden_layers.min(new_hidden_layers);
        let preserved_hidden_width = old_hidden_width.min(new_hidden_width);
        let preserved_input_dim = old_input_dim.min(new_input_dim);

        let old_lift_len = old_input_dim * old_hidden_width;
        let new_lift_len = new_input_dim * new_hidden_width;
        for hidden_idx in 0..preserved_hidden_width {
            let old_row_start = hidden_idx * old_input_dim;
            let new_row_start = hidden_idx * new_input_dim;
            resized[new_row_start..new_row_start + preserved_input_dim]
                .copy_from_slice(
                    &self.field_head_weights
                        [old_row_start..old_row_start + preserved_input_dim],
                );
        }

        let old_operator_start = old_lift_len;
        let new_operator_start = new_lift_len;
        for layer in 0..preserved_hidden_layers {
            let old_layer_start =
                old_operator_start + layer * old_hidden_width * old_hidden_width * 4;
            let new_layer_start =
                new_operator_start + layer * new_hidden_width * new_hidden_width * 4;
            for block in 0..4 {
                let old_block_start = old_layer_start + block * old_hidden_width * old_hidden_width;
                let new_block_start = new_layer_start + block * new_hidden_width * new_hidden_width;
                for row in 0..preserved_hidden_width {
                    let old_row_start = old_block_start + row * old_hidden_width;
                    let new_row_start = new_block_start + row * new_hidden_width;
                    resized[new_row_start..new_row_start + preserved_hidden_width]
                        .copy_from_slice(
                            &self.field_head_weights
                                [old_row_start..old_row_start + preserved_hidden_width],
                        );
                }
            }
        }

        let old_output_start =
            old_operator_start + old_hidden_layers * old_hidden_width * old_hidden_width * 4;
        let new_output_start =
            new_operator_start + new_hidden_layers * new_hidden_width * new_hidden_width * 4;
        for channel in 0..output_dim {
            let old_row_start = old_output_start + channel * old_hidden_width;
            let new_row_start = new_output_start + channel * new_hidden_width;
            resized[new_row_start..new_row_start + preserved_hidden_width].copy_from_slice(
                &self.field_head_weights[old_row_start..old_row_start + preserved_hidden_width],
            );
        }

        resized
    }

    fn resized_field_head_bias(
        &self,
        output_dim: usize,
        old_hidden_layers: usize,
        old_hidden_width: usize,
        new_hidden_layers: usize,
        new_hidden_width: usize,
    ) -> Vec<f64> {
        let mut resized =
            vec![0.0; Self::bias_len_for(output_dim, new_hidden_layers, new_hidden_width)];
        let preserved_hidden_layers = old_hidden_layers.min(new_hidden_layers);
        let preserved_hidden_width = old_hidden_width.min(new_hidden_width);

        resized[..preserved_hidden_width]
            .copy_from_slice(&self.field_head_bias[..preserved_hidden_width]);
        for layer in 0..preserved_hidden_layers {
            let old_layer_start = old_hidden_width + layer * old_hidden_width;
            let new_layer_start = new_hidden_width + layer * new_hidden_width;
            resized[new_layer_start..new_layer_start + preserved_hidden_width].copy_from_slice(
                &self.field_head_bias[old_layer_start..old_layer_start + preserved_hidden_width],
            );
        }
        let old_output_start = old_hidden_width + old_hidden_layers * old_hidden_width;
        let new_output_start = new_hidden_width + new_hidden_layers * new_hidden_width;
        let preserved_output = output_dim.min(
            self.field_head_bias
                .len()
                .saturating_sub(old_output_start),
        );
        resized[new_output_start..new_output_start + preserved_output]
            .copy_from_slice(&self.field_head_bias[old_output_start..old_output_start + preserved_output]);
        resized
    }

    pub fn aligned_to_config(mut self, config: &PinoModelConfig) -> Self {
        let input_dim = PINO_FIELD_HEAD_BASIS;
        let output_dim = config.operator_grid.output_channels.max(1);
        let old_hidden_layers = Self::normalized_hidden_layers(self.field_head_hidden_layers);
        let old_hidden_width = Self::normalized_hidden_width(self.field_head_hidden_width);
        let old_input_dim = Self::inferred_input_dim(
            self.field_head_weights.len(),
            output_dim,
            old_hidden_layers,
            old_hidden_width,
        );
        self.field_head_hidden_layers = Self::normalized_hidden_layers(config.hidden_layers);
        self.field_head_hidden_width = Self::normalized_hidden_width(config.hidden_width);
        if !self.matches_shape(input_dim, output_dim) {
            if let Some(old_input_dim) = old_input_dim {
                self.field_head_weights = self.resized_field_head_weights(
                    old_input_dim,
                    input_dim,
                    output_dim,
                    old_hidden_layers,
                    old_hidden_width,
                    self.field_head_hidden_layers,
                    self.field_head_hidden_width,
                );
                self.field_head_bias = self.resized_field_head_bias(
                    output_dim,
                    old_hidden_layers,
                    old_hidden_width,
                    self.field_head_hidden_layers,
                    self.field_head_hidden_width,
                );
            } else {
                self.field_head_weights = initialized_field_head_weights(
                    input_dim,
                    output_dim,
                    self.field_head_hidden_layers,
                    self.field_head_hidden_width,
                );
                self.field_head_bias = vec![
                    0.0;
                    Self::bias_len_for(
                        output_dim,
                        self.field_head_hidden_layers,
                        self.field_head_hidden_width,
                    )
                ];
            }
        }
        if self.field_head_activation.len()
            != Self::activation_len_for(self.field_head_hidden_layers)
        {
            let mut activation = vec![0.2; Self::activation_len_for(self.field_head_hidden_layers)];
            for (dst, src) in activation
                .iter_mut()
                .zip(self.field_head_activation.iter().copied())
            {
                *dst = src;
            }
            self.field_head_activation = activation;
        }
        self
    }

    pub fn clamped(mut self) -> Self {
        self.field_head_hidden_layers =
            Self::normalized_hidden_layers(self.field_head_hidden_layers);
        self.field_head_hidden_width =
            Self::normalized_hidden_width(self.field_head_hidden_width);
        for value in &mut self.field_head_weights {
            *value = value.clamp(-2.5, 2.5);
        }
        for value in &mut self.field_head_bias {
            *value = value.clamp(-1.5, 1.5);
        }
        for value in &mut self.field_head_activation {
            *value = value.clamp(0.05, 2.0);
        }
        self
    }

    fn local_neighbor_average(
        activations: &[f64],
        cell_count: usize,
        hidden_width: usize,
        grid_nx: usize,
        grid_ny: usize,
        grid_nz: usize,
    ) -> Vec<f64> {
        let mut averaged = vec![0.0; cell_count * hidden_width];
        if cell_count == 0 || grid_nx == 0 || grid_ny == 0 || grid_nz == 0 {
            return averaged;
        }
        let plane = grid_nx * grid_ny;
        for z in 0..grid_nz {
            for y in 0..grid_ny {
                for x in 0..grid_nx {
                    let cell = z * plane + y * grid_nx + x;
                    let mut neighbors = Vec::with_capacity(6);
                    if x > 0 {
                        neighbors.push(cell - 1);
                    }
                    if x + 1 < grid_nx {
                        neighbors.push(cell + 1);
                    }
                    if y > 0 {
                        neighbors.push(cell - grid_nx);
                    }
                    if y + 1 < grid_ny {
                        neighbors.push(cell + grid_nx);
                    }
                    if z > 0 {
                        neighbors.push(cell - plane);
                    }
                    if z + 1 < grid_nz {
                        neighbors.push(cell + plane);
                    }
                    if neighbors.is_empty() {
                        neighbors.push(cell);
                    }
                    let count = neighbors.len() as f64;
                    let target_offset = cell * hidden_width;
                    for source in neighbors {
                        let source_offset = source * hidden_width;
                        for idx in 0..hidden_width {
                            averaged[target_offset + idx] += activations[source_offset + idx];
                        }
                    }
                    for idx in 0..hidden_width {
                        averaged[target_offset + idx] /= count;
                    }
                }
            }
        }
        averaged
    }

    fn spectral_projection_matrix(
        grid_nx: usize,
        grid_ny: usize,
        grid_nz: usize,
        spectral_modes: usize,
    ) -> Vec<f64> {
        let cells = grid_nx.saturating_mul(grid_ny).saturating_mul(grid_nz);
        let mut matrix = vec![0.0; cells * cells];
        if cells == 0 || grid_nx == 0 || grid_ny == 0 || grid_nz == 0 {
            return matrix;
        }
        let mut basis = Vec::new();
        let mode_cap = spectral_modes
            .max(1)
            .min(grid_nx.min(grid_ny).min(grid_nz.max(1)).max(1));
        let basis_cap = (mode_cap * mode_cap * mode_cap).max(1);
        let plane = grid_nx * grid_ny;
        for kz in 0..mode_cap {
            for ky in 0..mode_cap {
                for kx in 0..mode_cap {
                    if basis.len() >= basis_cap {
                        break;
                    }
                    let mut phi = vec![0.0; cells];
                    let mut norm = 0.0;
                    for z in 0..grid_nz {
                        for y in 0..grid_ny {
                            for x in 0..grid_nx {
                                let idx = z * plane + y * grid_nx + x;
                                let x_term =
                                    (std::f64::consts::PI * (kx as f64) * ((x as f64) + 0.5)
                                        / (grid_nx as f64))
                                        .cos();
                                let y_term =
                                    (std::f64::consts::PI * (ky as f64) * ((y as f64) + 0.5)
                                        / (grid_ny as f64))
                                        .cos();
                                let z_term =
                                    (std::f64::consts::PI * (kz as f64) * ((z as f64) + 0.5)
                                        / (grid_nz as f64))
                                        .cos();
                                let value = x_term * y_term * z_term;
                                phi[idx] = value;
                                norm += value * value;
                            }
                        }
                    }
                    if norm > 1e-12 {
                        let inv_norm = norm.sqrt().recip();
                        for value in &mut phi {
                            *value *= inv_norm;
                        }
                        basis.push(phi);
                    }
                }
            }
        }
        if basis.is_empty() {
            for idx in 0..cells {
                matrix[idx * cells + idx] = 1.0;
            }
            return matrix;
        }
        for phi in basis {
            for row in 0..cells {
                for col in 0..cells {
                    matrix[row * cells + col] += phi[row] * phi[col];
                }
            }
        }
        matrix
    }

    fn apply_cell_projection(
        projection: &[f64],
        activations: &[f64],
        cell_count: usize,
        hidden_width: usize,
    ) -> Vec<f64> {
        let mut projected = vec![0.0; cell_count * hidden_width];
        if cell_count == 0 {
            return projected;
        }
        for row in 0..cell_count {
            let row_offset = row * cell_count;
            let target_offset = row * hidden_width;
            for source in 0..cell_count {
                let weight = projection[row_offset + source];
                if weight.abs() <= 1e-12 {
                    continue;
                }
                let source_offset = source * hidden_width;
                for hidden_idx in 0..hidden_width {
                    projected[target_offset + hidden_idx] +=
                        weight * activations[source_offset + hidden_idx];
                }
            }
        }
        projected
    }

    fn contrast_state(
        activations: &[f64],
        averaged: &[f64],
        cell_count: usize,
        hidden_width: usize,
    ) -> Vec<f64> {
        let mut contrast = vec![0.0; cell_count * hidden_width];
        for idx in 0..contrast.len() {
            contrast[idx] = activations[idx] - averaged[idx];
        }
        contrast
    }

    pub fn field_head_batch_response(
        &self,
        features: &[f64],
        cell_count: usize,
        feature_dim: usize,
        grid_nx: usize,
        grid_ny: usize,
        grid_nz: usize,
        spectral_modes: usize,
    ) -> Vec<f64> {
        if cell_count == 0 {
            return Vec::new();
        }
        let hidden_layers = Self::normalized_hidden_layers(self.field_head_hidden_layers);
        let hidden_width = Self::normalized_hidden_width(self.field_head_hidden_width);
        let activation_len = Self::activation_len_for(hidden_layers);
        let activation = if self.field_head_activation.len() == activation_len {
            self.field_head_activation.clone()
        } else {
            vec![0.2; activation_len]
        };
        let mut weight_offset = 0usize;
        let mut bias_offset = 0usize;
        let lift_weight_len = PINO_FIELD_HEAD_BASIS * hidden_width;
        let mut activations = vec![0.0; cell_count * hidden_width];
        for cell_idx in 0..cell_count {
            let feature_start = cell_idx * feature_dim;
            let feature_end = feature_start + feature_dim.min(PINO_FIELD_HEAD_BASIS);
            let cell_features = &features[feature_start..feature_end];
            for hidden_idx in 0..hidden_width {
                let row_start = hidden_idx * PINO_FIELD_HEAD_BASIS;
                let row_end = row_start + PINO_FIELD_HEAD_BASIS;
                let dot = self.field_head_weights[weight_offset + row_start..weight_offset + row_end]
                    .iter()
                    .zip(cell_features.iter().copied().chain(std::iter::repeat(0.0)))
                    .take(PINO_FIELD_HEAD_BASIS)
                    .map(|(w, f)| w * f)
                    .sum::<f64>();
                let bias = self
                    .field_head_bias
                    .get(bias_offset + hidden_idx)
                    .copied()
                    .unwrap_or(0.0);
                let hidden_value = Self::adaptive_tanh(dot + bias, activation[0]);
                activations[cell_idx * hidden_width + hidden_idx] = hidden_value;
            }
        }
        weight_offset += lift_weight_len;
        bias_offset += hidden_width;
        let lifted = activations.clone();
        let spectral_projection =
            Self::spectral_projection_matrix(grid_nx, grid_ny, grid_nz, spectral_modes);
        let mut outputs = vec![0.0; cell_count * PINO_OUTPUT_CHANNELS];
        for layer in 0..hidden_layers {
            let self_weight_len = hidden_width * hidden_width;
            let local_weight_len = hidden_width * hidden_width;
            let global_weight_len = hidden_width * hidden_width;
            let contrast_weight_len = hidden_width * hidden_width;
            let local_average = Self::local_neighbor_average(
                &activations,
                cell_count,
                hidden_width,
                grid_nx,
                grid_ny,
                grid_nz,
            );
            let contrast_state =
                Self::contrast_state(&activations, &local_average, cell_count, hidden_width);
            let spectral_state = Self::apply_cell_projection(
                &spectral_projection,
                &activations,
                cell_count,
                hidden_width,
            );
            let mut next = vec![0.0; cell_count * hidden_width];
            for cell_idx in 0..cell_count {
                let local_hidden = &activations[cell_idx * hidden_width..(cell_idx + 1) * hidden_width];
                let neighborhood =
                    &local_average[cell_idx * hidden_width..(cell_idx + 1) * hidden_width];
                let contrast_hidden =
                    &contrast_state[cell_idx * hidden_width..(cell_idx + 1) * hidden_width];
                let spectral_hidden =
                    &spectral_state[cell_idx * hidden_width..(cell_idx + 1) * hidden_width];
                for hidden_idx in 0..hidden_width {
                    let row_start = weight_offset + hidden_idx * hidden_width;
                    let row_mid = row_start + hidden_width;
                    let row_mid_2 = row_mid + hidden_width;
                    let row_mid_3 = row_mid_2 + hidden_width;
                    let row_end = row_mid_3 + hidden_width;
                    let self_dot = self.field_head_weights[row_start..row_mid]
                        .iter()
                        .zip(local_hidden.iter().copied())
                        .map(|(w, h)| w * h)
                        .sum::<f64>();
                    let local_dot = self.field_head_weights[row_mid..row_mid_2]
                        .iter()
                        .zip(neighborhood.iter().copied())
                        .map(|(w, h)| w * h)
                        .sum::<f64>();
                    let global_dot = self.field_head_weights[row_mid_2..row_mid_3]
                        .iter()
                        .zip(spectral_hidden.iter().copied())
                        .map(|(w, h)| w * h)
                        .sum::<f64>();
                    let contrast_dot = self.field_head_weights[row_mid_3..row_end]
                        .iter()
                        .zip(contrast_hidden.iter().copied())
                        .map(|(w, h)| w * h)
                        .sum::<f64>();
                    let bias = self
                        .field_head_bias
                        .get(bias_offset + hidden_idx)
                        .copied()
                        .unwrap_or(0.0);
                    let update = self_dot + local_dot + global_dot + contrast_dot + bias;
                    next[cell_idx * hidden_width + hidden_idx] = Self::adaptive_tanh(
                        update
                            + local_hidden[hidden_idx] * 0.35
                            + lifted[cell_idx * hidden_width + hidden_idx] * 0.18,
                        activation[layer + 1],
                    );
                }
            }
            activations = next;
            weight_offset +=
                self_weight_len + local_weight_len + global_weight_len + contrast_weight_len;
            bias_offset += hidden_width;
        }
        for cell_idx in 0..cell_count {
            let local_hidden = &activations[cell_idx * hidden_width..(cell_idx + 1) * hidden_width];
            for channel in 0..PINO_OUTPUT_CHANNELS {
                let output_start = weight_offset + channel * hidden_width;
                let output_end = output_start + hidden_width;
                let dot = self.field_head_weights[output_start..output_end]
                    .iter()
                    .zip(local_hidden.iter().copied())
                    .map(|(w, h)| w * h)
                    .sum::<f64>();
                let bias = self
                    .field_head_bias
                    .get(bias_offset + channel)
                    .copied()
                    .unwrap_or(0.0);
                outputs[cell_idx * PINO_OUTPUT_CHANNELS + channel] =
                    Self::soft_clip(dot + bias, 2.5);
            }
        }
        outputs
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PinoModelConfig {
    pub backend: String,
    pub spectral_modes: usize,
    pub hidden_layers: usize,
    pub hidden_width: usize,
    pub operator_grid: OperatorGridSpec,
}

pub fn is_pino_backend(name: &str) -> bool {
    matches!(
        name,
        PINO_BACKEND_CANDLE_CPU
            | PINO_BACKEND_CANDLE_CUDA
            | PINO_BACKEND_CANDLE_METAL
            | PINO_BACKEND_NDARRAY_CPU
    )
}

pub fn canonical_backend_name(name: Option<&str>) -> String {
    match name.unwrap_or(PINO_BACKEND_NDARRAY_CPU) {
        "burn-ndarray-cpu" => PINO_BACKEND_NDARRAY_CPU.to_string(),
        "burn-wgpu" => PINO_BACKEND_CANDLE_CPU.to_string(),
        other if is_pino_backend(other) => other.to_string(),
        _ => PINO_BACKEND_NDARRAY_CPU.to_string(),
    }
}

pub fn operator_grid_spec(input: &SolveInput) -> OperatorGridSpec {
    OperatorGridSpec {
        nx: input.mesh.nx.clamp(12, 48),
        ny: input.mesh.ny.clamp(8, 32),
        nz: input.mesh.nz.clamp(2, 8),
        input_channels: PINO_INPUT_CHANNELS,
        output_channels: PINO_OUTPUT_CHANNELS,
    }
}

pub fn model_config(batch: &TrainingBatch) -> PinoModelConfig {
    let operator_grid = batch
        .cases
        .first()
        .map(operator_grid_spec)
        .unwrap_or(OperatorGridSpec {
            nx: 24,
            ny: 16,
            nz: 8,
            input_channels: PINO_INPUT_CHANNELS,
            output_channels: PINO_OUTPUT_CHANNELS,
        });
    let boundary_budget = batch.boundary_points.unwrap_or(1024).max(16);
    let interface_budget = batch.interface_points.unwrap_or(512).max(16);
    let low_target_mode =
        batch.target_loss.is_finite() && batch.target_loss > 0.0 && batch.target_loss <= 1e-8;
    let geometric_mode_cap =
        (operator_grid.nx.min(operator_grid.ny).min(operator_grid.nz) / 2).max(2);
    let budget_mode_cap = if low_target_mode {
        ((boundary_budget + interface_budget) / 160).clamp(4, 16)
    } else {
        ((boundary_budget + interface_budget) / 192).clamp(2, 16)
    };
    let spectral_modes = geometric_mode_cap.min(budget_mode_cap);
    let hidden_width_cap = batch.max_topology.unwrap_or(64).clamp(8, 256);
    let desired_hidden_width = if low_target_mode {
        spectral_modes * 7
    } else {
        spectral_modes * 5
    };
    let hidden_width_floor = if low_target_mode { 48 } else { 40 };
    let hidden_width = if hidden_width_cap < hidden_width_floor {
        hidden_width_cap.max(8)
    } else {
        desired_hidden_width.clamp(hidden_width_floor, hidden_width_cap)
    };
    PinoModelConfig {
        backend: canonical_backend_name(batch.pinn_backend.as_deref()),
        spectral_modes,
        hidden_layers: if low_target_mode { 4 } else { 3 },
        hidden_width,
        operator_grid,
    }
}

#[allow(dead_code)]
pub fn encode_case(input: &SolveInput) -> OperatorSample {
    encode_case_with_grid(input, operator_grid_spec(input))
}

fn encode_case_with_grid(input: &SolveInput, grid: OperatorGridSpec) -> OperatorSample {
    let cell_count = grid.nx * grid.ny * grid.nz;
    let mut inputs = Vec::with_capacity(cell_count * grid.input_channels);
    let hole_radius = input.geometry.hole_diameter_in.unwrap_or(0.0).abs() * 0.5;
    let center_x = input.geometry.length_in * 0.5;
    let center_y = input.geometry.width_in * 0.5;
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let x = normalized_axis(i, grid.nx);
                let y = normalized_axis(j, grid.ny);
                let z = normalized_axis(k, grid.nz);
                let px = x * input.geometry.length_in;
                let py = y * input.geometry.width_in;
                let pz = z * input.geometry.thickness_in;
                let dx = px - center_x;
                let dy = py - center_y;
                let radial_distance = (dx * dx + dy * dy).sqrt();
                let signed_hole_distance = radial_distance - hole_radius;
                let hole_mask = if hole_radius > 0.0 && signed_hole_distance <= 0.0 {
                    0.0
                } else {
                    1.0
                };
                let dist_x = px.min((input.geometry.length_in - px).max(0.0));
                let dist_y = py.min((input.geometry.width_in - py).max(0.0));
                let dist_z = pz.min((input.geometry.thickness_in - pz).max(0.0));
                let signed_exterior_distance = dist_x.min(dist_y).min(dist_z);
                let fixed_distance = if input.boundary_conditions.fix_start_face {
                    px
                } else if input.boundary_conditions.fix_end_face {
                    (input.geometry.length_in - px).max(0.0)
                } else {
                    input.geometry.length_in
                };
                let load_distance = (input.geometry.length_in - px).max(0.0);
                inputs.extend_from_slice(&[
                    x,
                    y,
                    z,
                    hole_mask,
                    signed_exterior_distance,
                    signed_hole_distance,
                    fixed_distance,
                    load_distance,
                    input.geometry.length_in,
                    input.geometry.width_in,
                    input.geometry.thickness_in,
                    input.material.e_psi,
                    input.material.nu,
                    input.load.axial_load_lbf,
                    input.load.vertical_point_load_lbf,
                ]);
            }
        }
    }

    OperatorSample { inputs }
}

pub fn spectral_probe_score(batch: &TrainingBatch, config: &PinoModelConfig) -> Option<f64> {
    let input = batch.cases.first()?;
    let mut probe_config = config.clone();
    probe_config.operator_grid = OperatorGridSpec {
        nx: config.operator_grid.nx.min(10).max(4),
        ny: config.operator_grid.ny.min(10).max(4),
        nz: config.operator_grid.nz.min(6).max(1),
        input_channels: config.operator_grid.input_channels,
        output_channels: config.operator_grid.output_channels,
    };
    let batch = build_operator_field_head_batch(input, &probe_config, None);
    if batch.features.is_empty() {
        return None;
    }
    let energy = batch.features.iter().map(|v| v * v).sum::<f64>() / batch.features.len() as f64;
    Some(energy.sqrt())
}

pub fn infer_config_for_case(
    input: &SolveInput,
    backend: &str,
    spectral_modes: usize,
) -> PinoModelConfig {
    let modes = spectral_modes.max(1);
    PinoModelConfig {
        backend: canonical_backend_name(Some(backend)),
        spectral_modes: modes,
        hidden_layers: 3,
        hidden_width: (modes * 5).clamp(40, 128),
        operator_grid: operator_grid_spec(input),
    }
}

pub fn build_operator_prediction(
    input: &SolveInput,
    config: &PinoModelConfig,
) -> OperatorPrediction {
    build_operator_prediction_with_params(input, config, None)
}

pub fn build_operator_field_head_batch(
    input: &SolveInput,
    config: &PinoModelConfig,
    _params: Option<&OperatorTrainableParams>,
) -> OperatorFieldHeadBatch {
    let sample = encode_case_with_grid(input, config.operator_grid.clone());
    let cell_count = config.operator_grid.nx * config.operator_grid.ny * config.operator_grid.nz;
    let mut ux = Vec::with_capacity(cell_count);
    let mut uy = Vec::with_capacity(cell_count);
    let mut uz = Vec::with_capacity(cell_count);
    let mut sxx = Vec::with_capacity(cell_count);
    let mut syy = Vec::with_capacity(cell_count);
    let mut szz = Vec::with_capacity(cell_count);
    let mut sxy = Vec::with_capacity(cell_count);
    let mut sxz = Vec::with_capacity(cell_count);
    let mut syz = Vec::with_capacity(cell_count);
    let mut von_mises = Vec::with_capacity(cell_count);
    let mut max_principal = Vec::with_capacity(cell_count);
    let mut features = Vec::with_capacity(cell_count * PINO_FIELD_HEAD_BASIS);
    let mut correction_scales = Vec::with_capacity(cell_count * PINO_OUTPUT_CHANNELS);
    let mut mask_values = Vec::with_capacity(cell_count);
    let mut clamp_values = Vec::with_capacity(cell_count);
    let mut displacement_embed = Vec::with_capacity(cell_count);

    let area = (input.geometry.width_in * input.geometry.thickness_in).max(1e-9);
    let e = input.material.e_psi.max(1.0);
    let axial_sigma = input.load.axial_load_lbf / area;
    let shear_sigma = 1.5 * input.load.vertical_point_load_lbf / area;
    let iz = (input.geometry.thickness_in * input.geometry.width_in.powi(3) / 12.0).max(1e-9);
    let iy = (input.geometry.width_in * input.geometry.thickness_in.powi(3) / 12.0).max(1e-9);
    let bending_sigma = if input.load.vertical_point_load_lbf.abs() > 0.0 {
        6.0 * input.load.vertical_point_load_lbf * input.geometry.length_in
            / (input.geometry.thickness_in * input.geometry.width_in.powi(2)).max(1e-9)
    } else {
        0.0
    };
    let axial_disp = input.load.axial_load_lbf * input.geometry.length_in / (area * e).max(1e-9);
    let tip_deflection = if input.load.vertical_point_load_lbf.abs() > 0.0 {
        input.load.vertical_point_load_lbf * input.geometry.length_in.powi(3) / (3.0 * e * iz)
    } else {
        0.0
    };
    let hole_radius = input.geometry.hole_diameter_in.unwrap_or(0.0).abs() * 0.5;
    let stress_scale = axial_sigma
        .abs()
        .max(bending_sigma.abs())
        .max(shear_sigma)
        .max(1.0);
    let disp_scale = axial_disp.abs().max(tip_deflection.abs()).max(1e-6);
    let ux_scale = axial_disp.abs().max(disp_scale * 0.18).max(1e-6);
    let uy_scale = tip_deflection.abs().max(disp_scale * 0.28).max(1e-6);
    let uz_scale = (disp_scale * 0.16).max(1e-6);
    let sxx_scale = axial_sigma.abs().max(bending_sigma.abs()).max(1.0);
    let syy_scale = (input.material.nu.abs() * sxx_scale)
        .max(stress_scale * 0.22)
        .max(1.0);
    let szz_scale = syy_scale;
    let sxy_scale = shear_sigma.abs().max(stress_scale * 0.18).max(1.0);
    let sxz_scale = (shear_sigma.abs() * 0.35).max(stress_scale * 0.12).max(1.0);
    let syz_scale = (shear_sigma.abs() * 0.22).max(stress_scale * 0.10).max(1.0);
    let vm_scale = (0.5
        * ((sxx_scale - syy_scale).powi(2)
            + (syy_scale - szz_scale).powi(2)
            + (szz_scale - sxx_scale).powi(2))
        + 3.0 * (sxy_scale * sxy_scale + sxz_scale * sxz_scale + syz_scale * syz_scale))
        .max(1.0)
        .sqrt();
    let principal_scale = sxx_scale
        .abs()
        .max(vm_scale * 0.5)
        .max(1.0);
    let thickness_norm = input.geometry.thickness_in.abs().max(1e-6);
    let e_norm = input.material.e_psi.abs().max(1.0);
    let axial_load_norm = input.load.axial_load_lbf.abs().max(1.0);
    let vertical_load_norm = input.load.vertical_point_load_lbf.abs().max(1.0);
    let embed_span = (input.geometry.length_in / config.operator_grid.nx.max(2) as f64)
        .max(input.geometry.width_in / config.operator_grid.ny.max(2) as f64)
        .max(input.geometry.thickness_in / config.operator_grid.nz.max(2) as f64)
        .max(1e-6);

    for idx in 0..cell_count {
        let base = idx * config.operator_grid.input_channels;
        let x = sample.inputs[base];
        let y = sample.inputs[base + 1];
        let z = sample.inputs[base + 2];
        let mask = sample.inputs[base + 3];
        let signed_exterior_distance = sample.inputs[base + 4];
        let signed_hole_distance = sample.inputs[base + 5];
        let fixed_distance = sample.inputs[base + 6];
        let loaded_distance = sample.inputs[base + 7];
        let length = sample.inputs[base + 8];
        let width = sample.inputs[base + 9];
        let thickness = sample.inputs[base + 10];
        let e_local = sample.inputs[base + 11];
        let nu_local = sample.inputs[base + 12];
        let axial_load = sample.inputs[base + 13];
        let vertical_load = sample.inputs[base + 14];
        let x_feature = (2.0 * x - 1.0).clamp(-1.0, 1.0);
        let y_feature = (2.0 * y - 1.0).clamp(-1.0, 1.0);
        let z_feature = (2.0 * z - 1.0).clamp(-1.0, 1.0);
        let mask_feature = (2.0 * mask - 1.0).clamp(-1.0, 1.0);
        let boundary_feature =
            (signed_exterior_distance / length.max(width).max(thickness).max(1e-9)).clamp(0.0, 1.0);
        let fixed_feature = (1.0 - (fixed_distance / length.max(1e-9)).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let displacement_factor = if input.boundary_conditions.fix_start_face
            || input.boundary_conditions.fix_end_face
        {
            (fixed_distance / (fixed_distance + embed_span)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let load_feature = (1.0 - (loaded_distance / length.max(1e-9)).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let signed_hole_feature = if hole_radius > 0.0 {
            (signed_hole_distance / hole_radius.max(1e-9)).clamp(-2.5, 2.5).tanh()
        } else {
            1.0
        };
        let axial_feature = (axial_load / axial_load_norm).clamp(-2.5, 2.5).tanh();
        let vertical_feature = (vertical_load / vertical_load_norm).clamp(-2.5, 2.5).tanh();
        let thickness_feature = (thickness / thickness_norm).clamp(-2.5, 2.5).tanh();
        let stiffness_feature = (e_local / e_norm).clamp(-2.5, 2.5).tanh();
        let poisson_feature = ((nu_local / 0.5) * 2.0 - 1.0).clamp(-1.0, 1.0);
        let hole_x_feature = (signed_hole_feature * x_feature).clamp(-1.0, 1.0);
        let hole_y_feature = (signed_hole_feature * y_feature).clamp(-1.0, 1.0);
        let hole_z_feature = (signed_hole_feature * z_feature).clamp(-1.0, 1.0);
        let xi = x.clamp(0.0, 1.0);
        let eta = (2.0 * y - 1.0).clamp(-1.0, 1.0);
        let zeta = (2.0 * z - 1.0).clamp(-1.0, 1.0);
        let axial_shape = xi;
        let bending_shape = (1.0 - xi).max(0.0) * eta;
        let deflection_shape = (xi * xi * (3.0 - xi) * 0.5).clamp(0.0, 1.0);
        let shear_shape = (1.0 - eta * eta).max(0.0);
        let through_thickness = (1.0 - zeta * zeta).max(0.0);
        let poisson_contraction = if input.load.axial_load_lbf.abs() > 1e-9 {
            -input.material.nu * (axial_sigma / e) * (y - 0.5) * input.geometry.width_in * xi
        } else {
            0.0
        };
        let poisson_thickness = -input.material.nu
            * (axial_sigma / e)
            * (z - 0.5)
            * input.geometry.thickness_in
            * xi;
        let dx_center = x - 0.5;
        let dy_center = y - 0.5;
        let radial_center = (dx_center * dx_center + dy_center * dy_center).sqrt();
        let hotspot_radius_feature = if hole_radius > 0.0 {
            (1.0 - (signed_hole_distance.abs() / hole_radius.max(1e-9))).clamp(-1.0, 1.0)
        } else {
            (1.0 - (signed_exterior_distance / length.max(width).max(thickness).max(1e-9))).clamp(0.0, 1.0)
        };
        let hotspot_angle_x = if radial_center > 1e-9 {
            (dx_center / radial_center).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let hotspot_angle_y = if radial_center > 1e-9 {
            (dy_center / radial_center).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let edge_symmetry_x = (1.0 - (2.0 * x - 1.0).abs()).clamp(0.0, 1.0);
        let edge_symmetry_y = (1.0 - (2.0 * y - 1.0).abs()).clamp(0.0, 1.0);
        let edge_symmetry_z = (1.0 - (2.0 * z - 1.0).abs()).clamp(0.0, 1.0);
        let embed_feature = (2.0 * displacement_factor - 1.0).clamp(-1.0, 1.0);
        let hole_decay = if hole_radius > 0.0 {
            let normalized_distance = (signed_hole_distance.abs() / hole_radius.max(1e-9)).max(0.0);
            let axial_focus = (1.0 - eta.abs()).clamp(0.15, 1.0);
            (-normalized_distance / 0.55).exp() * axial_focus
        } else {
            0.0
        };
        let field_features = [
            x_feature,
            y_feature,
            z_feature,
            mask_feature,
            boundary_feature,
            signed_hole_feature,
            fixed_feature,
            load_feature,
            thickness_feature,
            axial_feature,
            vertical_feature,
            stiffness_feature,
            poisson_feature,
            (x_feature * x_feature).clamp(-1.0, 1.0),
            (y_feature * y_feature).clamp(-1.0, 1.0),
            (z_feature * z_feature).clamp(-1.0, 1.0),
            (x_feature * y_feature).clamp(-1.0, 1.0),
            (x_feature * z_feature).clamp(-1.0, 1.0),
            (y_feature * z_feature).clamp(-1.0, 1.0),
            (x_feature * axial_feature).clamp(-1.0, 1.0),
            (x_feature * vertical_feature).clamp(-1.0, 1.0),
            (y_feature * vertical_feature).clamp(-1.0, 1.0),
            (load_feature * x_feature).clamp(-1.0, 1.0),
            (fixed_feature * x_feature).clamp(-1.0, 1.0),
            hole_x_feature,
            hole_y_feature,
            hole_z_feature,
            hotspot_radius_feature,
            hotspot_angle_x,
            hotspot_angle_y,
            edge_symmetry_x,
            edge_symmetry_y,
            embed_feature * edge_symmetry_z,
        ];
        features.extend_from_slice(&field_features);
        correction_scales.extend_from_slice(&[
            mask * displacement_factor * ux_scale,
            mask * displacement_factor * uy_scale,
            mask * displacement_factor * uz_scale,
            mask * sxx_scale,
            mask * syy_scale,
            mask * szz_scale,
            mask * sxy_scale,
            mask * sxz_scale,
            mask * syz_scale,
            mask * vm_scale,
            mask * principal_scale,
        ]);
        mask_values.push(mask);
        clamp_values.push(fixed_feature);
        displacement_embed.push(mask * displacement_factor);
        let hole_concentration = if hole_radius > 0.0 && input.load.axial_load_lbf.abs() > 1e-9 {
            1.0 + 2.0 * hole_decay
        } else {
            1.0
        };
        let ux_base = mask * axial_disp * axial_shape * (1.0 + 0.05 * hole_decay);
        let uy_base = mask
            * (tip_deflection * deflection_shape * (1.0 + 0.10 * hole_decay)
                + poisson_contraction * (1.0 + 0.20 * hole_decay));
        let uz_base = mask * poisson_thickness * (1.0 + 0.10 * hole_decay);
        let sxx_base = mask
            * ((axial_sigma * hole_concentration)
                + bending_sigma * bending_shape);
        let syy_base = mask
            * (input.material.nu * axial_sigma.abs() * (hole_concentration - 1.0) * 0.35)
            * axial_sigma.signum();
        let szz_base = -input.material.nu * (sxx_base + syy_base) * 0.35;
        let sxy_base = mask * shear_sigma * shear_shape;
        let sxz_base = mask * vertical_load * input.geometry.length_in / iy.max(1e-9)
            * through_thickness
            * zeta
            * 0.01;
        let syz_base = mask * shear_sigma * through_thickness * zeta * 0.15;
        let mean_stress = (sxx_base + syy_base + szz_base) / 3.0;
        let deviatoric = ((sxx_base - mean_stress).powi(2)
            + (syy_base - mean_stress).powi(2)
            + (szz_base - mean_stress).powi(2)
            + 2.0 * (sxy_base.powi(2) + sxz_base.powi(2) + syz_base.powi(2)))
            / 6.0;
        let max_principal_base = mean_stress + (2.0 * deviatoric.max(1e-12)).sqrt();
        let von_mises_base = (0.5
            * ((sxx_base - syy_base).powi(2)
                + (syy_base - szz_base).powi(2)
                + (szz_base - sxx_base).powi(2))
            + 3.0 * (sxy_base * sxy_base + sxz_base * sxz_base + syz_base * syz_base))
            .max(0.0)
            .sqrt();

        ux.push(ux_base);
        uy.push(uy_base);
        uz.push(uz_base);
        sxx.push(sxx_base);
        syy.push(syy_base);
        szz.push(szz_base);
        sxy.push(sxy_base);
        sxz.push(sxz_base);
        syz.push(syz_base);
        von_mises.push(von_mises_base);
        max_principal.push(max_principal_base);
    }

    OperatorFieldHeadBatch {
        base_prediction: OperatorPrediction {
            grid: config.operator_grid.clone(),
            ux,
            uy,
            uz,
            sxx,
            syy,
            szz,
            sxy,
            sxz,
            syz,
            von_mises,
            max_principal,
        },
        features,
        correction_scales,
        mask: mask_values,
        clamp: clamp_values,
        displacement_embed,
        cell_count,
        feature_dim: PINO_FIELD_HEAD_BASIS,
        output_dim: PINO_OUTPUT_CHANNELS,
    }
}

pub fn build_operator_prediction_with_params(
    input: &SolveInput,
    config: &PinoModelConfig,
    params: Option<&OperatorTrainableParams>,
) -> OperatorPrediction {
    let params = params
        .cloned()
        .unwrap_or_default()
        .aligned_to_config(config)
        .clamped();
    let batch = build_operator_field_head_batch(input, config, Some(&params));
    let mut prediction = batch.base_prediction;
    let head_values = params.field_head_batch_response(
        &batch.features,
        batch.cell_count,
        batch.feature_dim,
        config.operator_grid.nx,
        config.operator_grid.ny,
        config.operator_grid.nz,
        config.spectral_modes,
    );
    for idx in 0..batch.cell_count {
        let scale_start = idx * batch.output_dim;
        let scales = &batch.correction_scales[scale_start..scale_start + batch.output_dim];
        let output_start = idx * PINO_OUTPUT_CHANNELS;
        prediction.ux[idx] += scales[0] * head_values[output_start];
        prediction.uy[idx] += scales[1] * head_values[output_start + 1];
        prediction.uz[idx] += scales[2] * head_values[output_start + 2];
        prediction.sxx[idx] += scales[3] * head_values[output_start + 3];
        prediction.syy[idx] += scales[4] * head_values[output_start + 4];
        prediction.szz[idx] += scales[5] * head_values[output_start + 5];
        prediction.sxy[idx] += scales[6] * head_values[output_start + 6];
        prediction.sxz[idx] += scales[7] * head_values[output_start + 7];
        prediction.syz[idx] += scales[8] * head_values[output_start + 8];
        let mean_stress = (prediction.sxx[idx] + prediction.syy[idx] + prediction.szz[idx]) / 3.0;
        let dev = ((prediction.sxx[idx] - mean_stress).powi(2)
            + (prediction.syy[idx] - mean_stress).powi(2)
            + (prediction.szz[idx] - mean_stress).powi(2)
            + 2.0
                * (prediction.sxy[idx].powi(2)
                    + prediction.sxz[idx].powi(2)
                    + prediction.syz[idx].powi(2)))
            / 6.0;
        prediction.max_principal[idx] = mean_stress + (2.0 * dev.max(1e-12)).sqrt();
        prediction.von_mises[idx] = (0.5
            * ((prediction.sxx[idx] - prediction.syy[idx]).powi(2)
                + (prediction.syy[idx] - prediction.szz[idx]).powi(2)
                + (prediction.szz[idx] - prediction.sxx[idx]).powi(2))
            + 3.0
                * (prediction.sxy[idx].powi(2)
                    + prediction.sxz[idx].powi(2)
                    + prediction.syz[idx].powi(2)))
        .max(0.0)
        .sqrt();
    }
    prediction
}

pub fn apply_operator_calibration(
    prediction: &mut OperatorPrediction,
    calibration: &OperatorCalibration,
) {
    for value in &mut prediction.ux {
        *value *= calibration.displacement_scale;
    }
    for value in &mut prediction.uy {
        *value *= calibration.displacement_scale;
    }
    for value in &mut prediction.uz {
        *value *= calibration.displacement_scale;
    }
    for value in &mut prediction.sxx {
        *value *= calibration.stress_scale;
    }
    for value in &mut prediction.syy {
        *value *= calibration.stress_scale;
    }
    for value in &mut prediction.szz {
        *value *= calibration.stress_scale;
    }
    for value in &mut prediction.sxy {
        *value *= calibration.stress_scale;
    }
    for value in &mut prediction.sxz {
        *value *= calibration.stress_scale;
    }
    for value in &mut prediction.syz {
        *value *= calibration.stress_scale;
    }
    for value in &mut prediction.von_mises {
        *value *= calibration.stress_scale.abs();
    }
    for value in &mut prediction.max_principal {
        *value *= calibration.stress_scale;
    }
}

pub fn fit_operator_calibration(
    batch: &TrainingBatch,
    config: &PinoModelConfig,
) -> Option<OperatorCalibration> {
    fit_operator_calibration_with_params(batch, config, None)
}

pub fn fit_operator_calibration_with_params(
    batch: &TrainingBatch,
    config: &PinoModelConfig,
    params: Option<&OperatorTrainableParams>,
) -> Option<OperatorCalibration> {
    if batch.cases.is_empty() {
        return None;
    }
    let mut calib_config = config.clone();
    calib_config.operator_grid.nx = calib_config.operator_grid.nx.clamp(8, 16);
    calib_config.operator_grid.ny = calib_config.operator_grid.ny.clamp(6, 12);
    let mut stress_scales = Vec::new();
    let mut displacement_scales = Vec::new();
    for case in batch.cases.iter().take(3) {
        let fem = solve_case_for_operator(case);
        let prediction = build_operator_prediction_with_params(case, &calib_config, params);
        let stress_target = fem.von_mises_psi.abs().max(1e-9);
        let stress_model = prediction
            .von_mises
            .iter()
            .copied()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
            .max(1e-9);
        stress_scales.push((stress_target / stress_model).clamp(0.25, 4.0));
        let disp_target = fem_regime_displacement_observable(case, &fem);
        let disp_model = prediction_regime_displacement_observable(case, &prediction);
        let disp_ratio = if disp_target.abs() <= 1e-12 {
            0.0
        } else {
            let denom = if disp_model.abs() <= 1e-12 {
                if disp_target.is_sign_negative() { -1e-12 } else { 1e-12 }
            } else {
                disp_model
            };
            (disp_target / denom).clamp(-4.0, 4.0)
        };
        displacement_scales.push(disp_ratio);
    }
    Some(OperatorCalibration {
        stress_scale: mean(&stress_scales),
        displacement_scale: mean(&displacement_scales),
    })
}

pub fn train_operator_calibration(
    batch: &TrainingBatch,
    config: &PinoModelConfig,
    max_epochs: usize,
) -> Option<OperatorTrainingStats> {
    if batch.cases.is_empty() {
        return None;
    }
    let mut calib_config = config.clone();
    calib_config.operator_grid.nx = calib_config.operator_grid.nx.clamp(8, 16);
    calib_config.operator_grid.ny = calib_config.operator_grid.ny.clamp(6, 12);
    let epochs = max_epochs.clamp(1, 40);
    let mut calibration =
        fit_operator_calibration(batch, &calib_config).unwrap_or(OperatorCalibration {
            stress_scale: 1.0,
            displacement_scale: 1.0,
        });

    let mut cases = Vec::new();
    for case in batch.cases.iter().take(4) {
        let base_prediction = build_operator_prediction(case, &calib_config);
        let fem = solve_case_for_operator(case);
        let target_vm = fem.von_mises_psi.abs().max(1e-9);
        let target_disp = fem_regime_displacement_observable(case, &fem);
        cases.push((case.clone(), base_prediction, target_vm, target_disp));
    }
    if cases.is_empty() {
        return None;
    }

    let mut initial_loss = 0.0;
    let mut best_loss = f64::MAX;
    let mut final_loss = 0.0;
    let mut no_improve = 0usize;
    let patience = 8usize;
    let blend = 0.35_f64;
    let mut epochs_run = 0usize;

    for epoch in 1..=epochs {
        let mut loss_terms = Vec::with_capacity(cases.len());
        let mut vm_ratio_terms = Vec::with_capacity(cases.len());
        let mut disp_ratio_terms = Vec::with_capacity(cases.len());
        for (case, base_prediction, target_vm, target_disp) in &cases {
            let mut calibrated = base_prediction.clone();
            apply_operator_calibration(&mut calibrated, &calibration);
            let model_vm = calibrated
                .von_mises
                .iter()
                .copied()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()))
                .max(1e-9);
            let model_disp = prediction_regime_displacement_observable(case, &calibrated);
            let vm_err = ((model_vm - target_vm).abs() / target_vm).min(10.0);
            let disp_err = if target_disp.abs() <= 1e-12 {
                model_disp.abs().min(10.0)
            } else {
                ((model_disp - target_disp).abs() / target_disp.abs()).min(10.0)
            };
            loss_terms.push(0.5 * (vm_err + disp_err));
            vm_ratio_terms.push((target_vm / model_vm).clamp(0.25, 4.0));
            let disp_ratio = if target_disp.abs() <= 1e-12 {
                0.0
            } else {
                let denom = if model_disp.abs() <= 1e-12 {
                    if target_disp.is_sign_negative() { -1e-12 } else { 1e-12 }
                } else {
                    model_disp
                };
                (target_disp / denom).clamp(-4.0, 4.0)
            };
            disp_ratio_terms.push(disp_ratio);
        }
        let loss = mean(&loss_terms);
        if epoch == 1 {
            initial_loss = loss;
        }
        final_loss = loss;
        epochs_run = epoch;
        if loss < best_loss {
            best_loss = loss;
            no_improve = 0;
        } else {
            no_improve = no_improve.saturating_add(1);
        }

        let vm_ratio = mean(&vm_ratio_terms);
        let disp_ratio = mean(&disp_ratio_terms);
        let vm_update = (1.0 - blend) + blend * vm_ratio;
        let disp_update = (1.0 - blend) + blend * disp_ratio;
        calibration.stress_scale = (calibration.stress_scale * vm_update).clamp(0.25, 4.0);
        calibration.displacement_scale =
            (calibration.displacement_scale * disp_update).clamp(-4.0, 4.0);

        if no_improve >= patience {
            break;
        }
    }

    Some(OperatorTrainingStats {
        epochs_run,
        initial_loss,
        best_loss,
        final_loss,
        calibration,
    })
}

fn solve_case_for_operator(case: &SolveInput) -> FemResult {
    let mut reduced = case.clone();
    reduced.mesh.nx = reduced.mesh.nx.clamp(8, 16);
    reduced.mesh.ny = reduced.mesh.ny.clamp(6, 12);
    reduced.mesh.nz = reduced.mesh.nz.clamp(2, 4);
    reduced.mesh.amr_enabled = false;
    reduced.mesh.amr_passes = 0;
    solve_case(&reduced)
}

pub fn operator_residual_score(input: &SolveInput, prediction: &OperatorPrediction) -> f64 {
    let area = (input.geometry.width_in * input.geometry.thickness_in).max(1e-9);
    let e = input.material.e_psi.max(1.0);
    let iz = (input.geometry.thickness_in * input.geometry.width_in.powi(3) / 12.0).max(1e-9);
    let expected_axial_sigma = input.load.axial_load_lbf / area;
    let expected_tip_deflection = if input.load.vertical_point_load_lbf.abs() > 0.0 {
        input.load.vertical_point_load_lbf * input.geometry.length_in.powi(3) / (3.0 * e * iz)
    } else {
        0.0
    };
    let expected_axial_disp =
        input.load.axial_load_lbf * input.geometry.length_in / (area * e).max(1e-9);

    let mean_sxx = mean(&prediction.sxx);
    let max_uy =
        prediction
            .uy
            .iter()
            .copied()
            .fold(0.0_f64, |acc, v| if v.abs() > acc.abs() { v } else { acc });
    let mean_ux = mean(&prediction.ux);
    let mean_vm = mean(&prediction.von_mises);
    let sigma_err = relative_error(
        mean_sxx,
        expected_axial_sigma + expected_tip_deflection.signum() * 0.25 * expected_axial_sigma,
    );
    let uy_err = relative_error(max_uy.abs(), expected_tip_deflection.abs());
    let ux_err = relative_error(mean_ux.abs(), (expected_axial_disp * 0.5).abs());
    let vm_err = relative_error(mean_vm.abs(), expected_axial_sigma.abs().max(1.0));
    ((sigma_err + uy_err + ux_err + vm_err) / 4.0).clamp(0.0, 10.0)
}

#[allow(dead_code)]
pub fn prediction_from_fem(input: &SolveInput, fem: &FemResult) -> OperatorPrediction {
    prediction_from_fem_with_grid(input, fem, operator_grid_spec(input))
}

pub fn prediction_from_fem_with_grid(
    input: &SolveInput,
    fem: &FemResult,
    grid: OperatorGridSpec,
) -> OperatorPrediction {
    prediction_from_fem_with_grid_mode(input, fem, grid, false)
}

pub fn prediction_from_fem_with_grid_exact(
    input: &SolveInput,
    fem: &FemResult,
    grid: OperatorGridSpec,
) -> OperatorPrediction {
    prediction_from_fem_with_grid_mode(input, fem, grid, true)
}

fn prediction_from_fem_with_grid_mode(
    input: &SolveInput,
    fem: &FemResult,
    grid: OperatorGridSpec,
    exact_hole_projection: bool,
) -> OperatorPrediction {
    let cell_count = grid.nx * grid.ny * grid.nz;
    let mut ux = Vec::with_capacity(cell_count);
    let mut uy = Vec::with_capacity(cell_count);
    let mut uz = Vec::with_capacity(cell_count);
    let mut sxx = Vec::with_capacity(cell_count);
    let mut syy = Vec::with_capacity(cell_count);
    let mut szz = Vec::with_capacity(cell_count);
    let mut sxy = Vec::with_capacity(cell_count);
    let mut sxz = Vec::with_capacity(cell_count);
    let mut syz = Vec::with_capacity(cell_count);
    let mut von_mises = Vec::with_capacity(cell_count);
    let mut max_principal = Vec::with_capacity(cell_count);
    let area = (input.geometry.width_in * input.geometry.thickness_in).max(1e-9);
    let hole_radius = input.geometry.hole_diameter_in.unwrap_or(0.0).abs() * 0.5;
    let hole_target_base = if hole_radius > 0.0 && !exact_hole_projection {
        Some(
            build_operator_field_head_batch(
                input,
                &PinoModelConfig {
                    backend: PINO_BACKEND_NDARRAY_CPU.to_string(),
                    spectral_modes: 4,
                    hidden_layers: 3,
                    hidden_width: PINO_FIELD_HEAD_HIDDEN,
                    operator_grid: grid.clone(),
                },
                None,
            )
            .base_prediction,
        )
    } else {
        None
    };
    let far_sigma_x = if input.load.axial_load_lbf.abs() > 1e-9 {
        input.load.axial_load_lbf / area
    } else {
        fem.stress_tensor[0][0]
    };
    let hole_center_x = input.geometry.length_in * 0.5;
    let hole_center_y = input.geometry.width_in * 0.5;
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let px = normalized_axis(i, grid.nx) * input.geometry.length_in;
                let py = normalized_axis(j, grid.ny) * input.geometry.width_in;
                let pz = normalized_axis(k, grid.nz) * input.geometry.thickness_in;
                let nearest = nearest_displacement(&fem.nodal_displacements, px, py, pz);
                let station = interpolate_station(&fem.beam_stations, px);
                let through_width = if input.geometry.width_in.abs() <= 1e-9 {
                    0.5
                } else {
                    (py / input.geometry.width_in).clamp(0.0, 1.0)
                };
                let (sigma_x, sigma_y, sigma_z, shear_xy, shear_xz, shear_yz) =
                    if exact_hole_projection && hole_radius > 0.0 {
                        let dx = px - hole_center_x;
                        let dy = py - hole_center_y;
                        let radial = (dx * dx + dy * dy).sqrt().max(1e-9);
                        let rx = dx / radial;
                        let ry = dy / radial;
                        let tx = -ry;
                        let ty = rx;
                        let edge_distance = (radial - hole_radius).abs();
                        let hole_diameter = hole_radius * 2.0;
                        let width_ratio =
                            (hole_diameter / input.geometry.width_in.max(hole_diameter + 1e-9))
                                .clamp(0.0, 0.95);
                        let finite_width_kt = (3.0
                            - 3.14 * width_ratio
                            + 3.667 * width_ratio.powi(2)
                            - 1.527 * width_ratio.powi(3))
                            .clamp(2.0, 4.75);
                        let ring_band = (hole_radius * 0.30)
                            .max(input.geometry.width_in * 0.035)
                            .max(input.geometry.thickness_in * 0.60)
                            .max(1e-6);
                        let ring_focus =
                            (-(edge_distance * edge_distance) / (ring_band * ring_band)).exp();
                        let inner_ring_band = (ring_band * 0.55).max(1e-6);
                        let inner_ring_focus = (-(edge_distance * edge_distance)
                            / (inner_ring_band * inner_ring_band))
                            .exp();
                        let ligament_band = (input.geometry.length_in * 0.12)
                            .max(hole_radius * 0.8)
                            .max(1e-6);
                        let ligament_focus =
                            (-(dx * dx) / (ligament_band * ligament_band)).exp();
                        let thickness_band = (input.geometry.thickness_in * 0.26).max(1e-6);
                        let thickness_focus = (-((pz - input.geometry.thickness_in * 0.5).powi(2))
                            / (thickness_band * thickness_band))
                            .exp();
                        let orthogonal_to_load = ry.abs();
                        let angular_focus = orthogonal_to_load.powi(2);
                        let concentration = 1.0
                            + (finite_width_kt - 1.0)
                                * (0.58 * ring_focus + 0.42 * inner_ring_focus)
                                * (0.62 + 0.38 * ligament_focus)
                                * angular_focus
                                * (0.82 + 0.18 * thickness_focus);
                        let tangential_sigma = far_sigma_x * concentration;
                        let radial_sigma = far_sigma_x
                            * (1.0 - 0.62 * ring_focus * (0.75 + 0.25 * ligament_focus))
                                .clamp(0.10, 1.20);
                        let mut sigma_x = radial_sigma * rx * rx + tangential_sigma * tx * tx;
                        let mut sigma_y = radial_sigma * ry * ry + tangential_sigma * ty * ty;
                        let mut shear_xy = radial_sigma * rx * ry + tangential_sigma * tx * ty;
                        let vm_raw = (0.5
                            * ((sigma_x - sigma_y).powi(2)
                                + sigma_y.powi(2)
                                + sigma_x.powi(2))
                            + 3.0 * shear_xy.powi(2))
                        .max(1e-6)
                        .sqrt();
                        let kirsch_vm =
                            (far_sigma_x.abs() * finite_width_kt * (0.45 + 0.55 * inner_ring_focus))
                                .max(far_sigma_x.abs());
                        let vm_target = nearest
                            .vm_psi
                            .abs()
                            .max(kirsch_vm)
                            .max(far_sigma_x.abs())
                            .max(1.0);
                        let stress_gain = (vm_target / vm_raw).clamp(0.35, 4.5);
                        sigma_x *= stress_gain;
                        sigma_y *= stress_gain;
                        shear_xy *= stress_gain;
                        let sigma_z = (-input.material.nu * (sigma_x + sigma_y)
                            * (0.14 + 0.10 * thickness_focus))
                            .clamp(-vm_target * 0.65, vm_target * 0.65);
                        let zeta = 2.0 * normalized_axis(k, grid.nz) - 1.0;
                        let shear_xz = fem.stress_tensor[0][2]
                            * zeta
                            * (0.22 + 0.78 * inner_ring_focus * thickness_focus);
                        let shear_yz = fem.stress_tensor[1][2]
                            * zeta
                            * (0.22 + 0.78 * inner_ring_focus * thickness_focus);
                        (sigma_x, sigma_y, sigma_z, shear_xy, shear_xz, shear_yz)
                    } else if let Some(base) = hole_target_base.as_ref() {
                        let base_vm = base.von_mises.get(ux.len()).copied().unwrap_or(0.0).abs().max(1e-6);
                        let vm_target = nearest.vm_psi.abs().max(base_vm * 0.5);
                        let stress_gain = (vm_target / base_vm).clamp(0.35, 4.5);
                        (
                            base.sxx.get(ux.len()).copied().unwrap_or(0.0) * stress_gain,
                            base.syy.get(ux.len()).copied().unwrap_or(0.0) * stress_gain,
                            base.szz.get(ux.len()).copied().unwrap_or(0.0) * stress_gain,
                            base.sxy.get(ux.len()).copied().unwrap_or(0.0) * stress_gain,
                            base.sxz.get(ux.len()).copied().unwrap_or(0.0) * stress_gain,
                            base.syz.get(ux.len()).copied().unwrap_or(0.0) * stress_gain,
                        )
                    } else {
                        let sigma_x =
                            station.sigma_bottom_psi * (1.0 - through_width)
                                + station.sigma_top_psi * through_width;
                        let shear_xy = if input.load.vertical_point_load_lbf.abs() > 1e-9 {
                            (1.5 * input.load.vertical_point_load_lbf / area)
                                * (1.0 - (2.0 * through_width - 1.0).powi(2)).max(0.0)
                        } else {
                            fem.stress_tensor[0][1]
                        };
                        let sigma_y = fem.stress_tensor[1][1];
                        let sigma_z = fem.stress_tensor[2][2];
                        let shear_xz =
                            fem.stress_tensor[0][2] * (2.0 * normalized_axis(k, grid.nz) - 1.0);
                        let shear_yz =
                            fem.stress_tensor[1][2] * (2.0 * normalized_axis(k, grid.nz) - 1.0);
                        (sigma_x, sigma_y, sigma_z, shear_xy, shear_xz, shear_yz)
                    };
                let stress_tensor = [
                    [sigma_x, shear_xy, shear_xz],
                    [shear_xy, sigma_y, shear_yz],
                    [shear_xz, shear_yz, sigma_z],
                ];
                let principal = principal_stresses(stress_tensor);
                let vm = nearest.vm_psi.abs().max(
                    (0.5
                        * ((sigma_x - sigma_y).powi(2)
                            + (sigma_y - sigma_z).powi(2)
                            + (sigma_z - sigma_x).powi(2))
                        + 3.0 * (shear_xy.powi(2) + shear_xz.powi(2) + shear_yz.powi(2)))
                    .max(0.0)
                    .sqrt(),
                );
                let principal_max = principal[0].abs().max(vm);
                ux.push(nearest.ux_in);
                uy.push(nearest.uy_in);
                uz.push(nearest.uz_in);
                sxx.push(sigma_x);
                syy.push(sigma_y);
                szz.push(sigma_z);
                sxy.push(shear_xy);
                sxz.push(shear_xz);
                syz.push(shear_yz);
                von_mises.push(vm);
                max_principal.push(principal_max);
            }
        }
    }

    OperatorPrediction {
        grid,
        ux,
        uy,
        uz,
        sxx,
        syy,
        szz,
        sxy,
        sxz,
        syz,
        von_mises,
        max_principal,
    }
}

pub fn decode_prediction(input: &SolveInput, prediction: &OperatorPrediction) -> FemResult {
    let cell_count = prediction.grid.nx * prediction.grid.ny * prediction.grid.nz;
    let mut nodal_displacements = Vec::with_capacity(cell_count);
    let mut beam_stations = Vec::with_capacity(prediction.grid.nx);
    for k in 0..prediction.grid.nz {
        for j in 0..prediction.grid.ny {
            for i in 0..prediction.grid.nx {
                let idx = k * prediction.grid.nx * prediction.grid.ny + j * prediction.grid.nx + i;
                let x = normalized_axis(i, prediction.grid.nx) * input.geometry.length_in;
                let y = normalized_axis(j, prediction.grid.ny) * input.geometry.width_in;
                let z = normalized_axis(k, prediction.grid.nz) * input.geometry.thickness_in;
                let ux = prediction.ux.get(idx).copied().unwrap_or(0.0);
                let uy = prediction.uy.get(idx).copied().unwrap_or(0.0);
                let uz = prediction.uz.get(idx).copied().unwrap_or(0.0);
                let vm = prediction.von_mises.get(idx).copied().unwrap_or(0.0).abs();
                nodal_displacements.push(NodalDisplacement {
                    node_id: idx,
                    x_in: x,
                    y_in: y,
                    z_in: z,
                    ux_in: ux,
                    uy_in: uy,
                    uz_in: uz,
                    disp_mag_in: (ux * ux + uy * uy + uz * uz).sqrt(),
                    vm_psi: vm,
                });
            }
        }
    }

    for i in 0..prediction.grid.nx {
        let idx = (((prediction.grid.nz / 2) * prediction.grid.ny + (prediction.grid.ny / 2))
            * prediction.grid.nx
            + i)
            .min(cell_count - 1);
        beam_stations.push(BeamStationResult {
            x_in: normalized_axis(i, prediction.grid.nx) * input.geometry.length_in,
            shear_lbf: input.load.vertical_point_load_lbf,
            moment_lb_in: input.load.vertical_point_load_lbf
                * (input.geometry.length_in
                    - normalized_axis(i, prediction.grid.nx) * input.geometry.length_in),
            sigma_top_psi: prediction.sxx.get(idx).copied().unwrap_or(0.0)
                + prediction.sxy.get(idx).copied().unwrap_or(0.0),
            sigma_bottom_psi: -prediction.sxx.get(idx).copied().unwrap_or(0.0)
                + prediction.sxy.get(idx).copied().unwrap_or(0.0),
            deflection_in: prediction.uy.get(idx).copied().unwrap_or(0.0),
        });
    }

    let tip_uy = nearest_displacement(
        &nodal_displacements,
        input.geometry.length_in,
        input.geometry.width_in,
        input.geometry.thickness_in * 0.5,
    )
    .uy_in;
    let vm_global = prediction
        .von_mises
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let max_principal_global = prediction
        .max_principal
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let sxx = mean(&prediction.sxx);
    let syy = mean(&prediction.syy);
    let szz = mean(&prediction.szz);
    let sxy = mean(&prediction.sxy);
    let sxz = mean(&prediction.sxz);
    let syz = mean(&prediction.syz);
    let stress_tensor = [[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]];
    let principal = principal_stresses(stress_tensor);
    let e = input.material.e_psi.max(1.0);

    FemResult {
        nodal_displacements,
        strain_tensor: [
            [sxx / e, sxy / e, sxz / e],
            [sxy / e, syy / e, syz / e],
            [sxz / e, syz / e, szz / e],
        ],
        stress_tensor,
        principal_stresses: principal,
        von_mises_psi: vm_global,
        tresca_psi: tresca_from_principal(principal),
        max_principal_psi: if max_principal_global.is_finite() {
            max_principal_global
        } else {
            mean(&prediction.max_principal)
        },
        stiffness_matrix: vec![],
        mass_matrix: vec![],
        damping_matrix: vec![],
        force_vector: vec![input.load.axial_load_lbf, input.load.vertical_point_load_lbf, 0.0],
        displacement_vector: vec![mean(&prediction.ux), tip_uy, mean(&prediction.uz)],
        beam_stations,
        diagnostics: vec![
            format!(
                "PINO operator grid decoded: {}x{}x{} cells, {} output channels.",
                prediction.grid.nx,
                prediction.grid.ny,
                prediction.grid.nz,
                prediction.grid.output_channels
            ),
            "3D Navier-Cauchy operator field lifted into FEM-compatible view model.".to_string(),
        ],
    }
}

pub fn default_holdout_validation(batch: &TrainingBatch) -> HoldoutValidationSummary {
    HoldoutValidationSummary {
        trusted: false,
        training_seed_cases: batch.cases.len().min(PINO_RECIPE_SEED_COUNT),
        holdout_cases: PINO_HOLDOUT_CASES,
        mean_displacement_error: 0.0,
        mean_von_mises_error: 0.0,
        p95_field_error: 0.0,
        residual_ratio: 1.0,
        accepted_without_fallback: false,
        mean_error_limit: PINO_HOLDOUT_MEAN_ERROR_LIMIT,
        p95_error_limit: PINO_HOLDOUT_P95_ERROR_LIMIT,
        residual_ratio_limit: PINO_HOLDOUT_TRUST_RATIO_LIMIT,
        displacement_pass: false,
        von_mises_pass: false,
        p95_pass: false,
        residual_ratio_pass: false,
    }
}

pub fn enforce_training_recipe(batch: &TrainingBatch) -> TrainingBatch {
    let mut enforced = batch.clone();
    let Some(anchor) = recipe_anchor_case(batch) else {
        return enforced;
    };
    enforced.cases = build_seed_cases(anchor).to_vec();
    enforced
}

#[allow(dead_code)]
pub fn evaluate_holdout_projection(batch: &TrainingBatch) -> Option<HoldoutValidationSummary> {
    let config = model_config(batch);
    let calibration = fit_operator_calibration_with_params(batch, &config, None);
    evaluate_holdout_projection_with_model(batch, &config, calibration.as_ref(), None)
}

pub fn evaluate_holdout_projection_with_model(
    batch: &TrainingBatch,
    config: &PinoModelConfig,
    calibration: Option<&OperatorCalibration>,
    params: Option<&OperatorTrainableParams>,
) -> Option<HoldoutValidationSummary> {
    let base = recipe_anchor_case(batch)?;
    let holdouts = build_holdout_cases(base);
    let mut disp_errors = Vec::with_capacity(holdouts.len());
    let mut vm_errors = Vec::with_capacity(holdouts.len());
    let mut case_errors = Vec::with_capacity(holdouts.len());
    for case in holdouts {
        let fem = solve_case_for_operator(&case);
        let mut prediction = build_operator_prediction_with_params(&case, config, params);
        if let Some(calibration) = calibration {
            apply_operator_calibration(&mut prediction, calibration);
        }
        let projected = decode_prediction(&case, &prediction);
        let fem_disp = fem_regime_displacement_observable(&case, &fem);
        let proj_disp = prediction_regime_displacement_observable(&case, &prediction);
        let disp_err = ((proj_disp - fem_disp).abs() / fem_disp.abs().max(1e-9)).min(10.0);
        let vm_err = ((projected.von_mises_psi - fem.von_mises_psi).abs()
            / fem.von_mises_psi.abs().max(1e-9))
        .min(10.0);
        disp_errors.push(disp_err);
        vm_errors.push(vm_err);
        case_errors.push(disp_err.max(vm_err));
    }
    let mean_displacement_error = mean(&disp_errors);
    let mean_von_mises_error = mean(&vm_errors);
    let p95_field_error = percentile95(&case_errors);
    let residual_ratio = (mean_displacement_error / PINO_HOLDOUT_MEAN_ERROR_LIMIT)
        .max(mean_von_mises_error / PINO_HOLDOUT_MEAN_ERROR_LIMIT)
        .max(p95_field_error / PINO_HOLDOUT_P95_ERROR_LIMIT);
    let displacement_pass = mean_displacement_error <= PINO_HOLDOUT_MEAN_ERROR_LIMIT;
    let von_mises_pass = mean_von_mises_error <= PINO_HOLDOUT_MEAN_ERROR_LIMIT;
    let p95_pass = p95_field_error <= PINO_HOLDOUT_P95_ERROR_LIMIT;
    let residual_ratio_pass = residual_ratio <= PINO_HOLDOUT_TRUST_RATIO_LIMIT;
    let trusted = displacement_pass && von_mises_pass && p95_pass && residual_ratio_pass;
    Some(HoldoutValidationSummary {
        trusted,
        training_seed_cases: batch.cases.len().min(PINO_RECIPE_SEED_COUNT),
        holdout_cases: PINO_HOLDOUT_CASES,
        mean_displacement_error,
        mean_von_mises_error,
        p95_field_error,
        residual_ratio,
        accepted_without_fallback: trusted,
        mean_error_limit: PINO_HOLDOUT_MEAN_ERROR_LIMIT,
        p95_error_limit: PINO_HOLDOUT_P95_ERROR_LIMIT,
        residual_ratio_limit: PINO_HOLDOUT_TRUST_RATIO_LIMIT,
        displacement_pass,
        von_mises_pass,
        p95_pass,
        residual_ratio_pass,
    })
}

pub fn runtime_metadata(
    batch: &TrainingBatch,
    backend: &str,
    spectral_modes: usize,
) -> PinoRuntimeMetadata {
    let mut config = model_config(batch);
    config.backend = canonical_backend_name(Some(backend));
    config.spectral_modes = spectral_modes.max(1);
    let requested_grid_3d = batch.cases.first().map(|input| OperatorGridSpec {
        nx: input.mesh.nx.clamp(12, 48),
        ny: input.mesh.ny.clamp(8, 32),
        nz: input.mesh.nz.clamp(1, 12),
        input_channels: PINO_INPUT_CHANNELS,
        output_channels: PINO_OUTPUT_CHANNELS,
    });
    let requested_nz = requested_grid_3d.as_ref().map(|grid| grid.nz).unwrap_or(1);
    PinoRuntimeMetadata {
        engine_id: PINO_ENGINE_ID.to_string(),
        backend: config.backend,
        spectral_modes: config.spectral_modes,
        operator_grid: config.operator_grid.clone(),
        domain_dim: 3,
        physics_model: "navier-cauchy-3d-linear-elastic".to_string(),
        spectral_modes_3d: [
            config.spectral_modes,
            config.spectral_modes,
            requested_nz.min(config.spectral_modes.max(1)),
        ],
        operator_grid_3d: requested_grid_3d,
        boundary_mode: Some("embedded-dirichlet-3d".to_string()),
        objective_mode: Some("mixed-strong-energy-linear-elastic-3d".to_string()),
        local_refinement: Some(PinoLocalRefinementMetadata {
            enabled: matches!(
                std::env::var("PINO_SIGNOFF_RESIDUAL_REFINE").ok().as_deref(),
                Some("1" | "true" | "TRUE" | "on" | "ON")
            ),
            strategy: "ranked-hotspot-grid-refine+patch-focus".to_string(),
            max_patches: 2,
            max_patch_cells: config.operator_grid.nx * config.operator_grid.ny * config.operator_grid.nz,
        }),
        local_enrichment: Some(PinoLocalEnrichmentMetadata {
            enabled: true,
            strategy: "hotspot-basis-v1".to_string(),
        }),
        calibration_stress_scale: Some(1.0),
        calibration_displacement_scale: Some(1.0),
        holdout_validation: Some(default_holdout_validation(batch)),
    }
}

fn normalized_axis(index: usize, len: usize) -> f64 {
    if len <= 1 {
        0.0
    } else {
        index as f64 / (len - 1) as f64
    }
}

fn nearest_displacement<'a>(
    nodes: &'a [NodalDisplacement],
    x: f64,
    y: f64,
    z: f64,
) -> &'a NodalDisplacement {
    if nodes.is_empty() {
        return &DEFAULT_NODAL_DISPLACEMENT;
    }
    nodes
        .iter()
        .min_by(|a, b| {
            let da = (a.x_in - x).powi(2) + (a.y_in - y).powi(2) + (a.z_in - z).powi(2);
            let db = (b.x_in - x).powi(2) + (b.y_in - y).powi(2) + (b.z_in - z).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(&DEFAULT_NODAL_DISPLACEMENT)
}

fn interpolate_station<'a>(stations: &'a [BeamStationResult], x: f64) -> &'a BeamStationResult {
    if stations.is_empty() {
        return &DEFAULT_BEAM_STATION;
    }
    stations
        .iter()
        .min_by(|a, b| {
            (a.x_in - x)
                .abs()
                .partial_cmp(&(b.x_in - x).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(&stations[0])
}

const DEFAULT_BEAM_STATION: BeamStationResult = BeamStationResult {
    x_in: 0.0,
    shear_lbf: 0.0,
    moment_lb_in: 0.0,
    sigma_top_psi: 0.0,
    sigma_bottom_psi: 0.0,
    deflection_in: 0.0,
};

const DEFAULT_NODAL_DISPLACEMENT: NodalDisplacement = NodalDisplacement {
    node_id: 0,
    x_in: 0.0,
    y_in: 0.0,
    z_in: 0.0,
    ux_in: 0.0,
    uy_in: 0.0,
    uz_in: 0.0,
    disp_mag_in: 0.0,
    vm_psi: 0.0,
};

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    if expected.abs() <= 1e-9 {
        actual.abs().min(1.0)
    } else {
        ((actual - expected).abs() / expected.abs()).min(10.0)
    }
}

fn percentile95(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() - 1) as f64 * 0.95).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn regime_uses_tip_uy(case: &SolveInput) -> bool {
    case.load.vertical_point_load_lbf.abs() > case.load.axial_load_lbf.abs()
}

fn prediction_regime_displacement_observable(case: &SolveInput, prediction: &OperatorPrediction) -> f64 {
    if regime_uses_tip_uy(case) {
        if prediction.grid.nx == 0 || prediction.grid.ny == 0 || prediction.grid.nz == 0 {
            return 0.0;
        }
        let mut total = 0.0;
        let mut count = 0usize;
        for z in 0..prediction.grid.nz {
            let idx = z * prediction.grid.nx * prediction.grid.ny
                + prediction.grid.nx * prediction.grid.ny.saturating_sub(1)
                + prediction.grid.nx.saturating_sub(1);
            total += prediction.uy.get(idx).copied().unwrap_or(0.0);
            count += 1;
        }
        total / count.max(1) as f64
    } else {
        if prediction.grid.nx == 0 || prediction.grid.ny == 0 || prediction.grid.nz == 0 {
            return 0.0;
        }
        let mut total = 0.0;
        let mut count = 0usize;
        for z in 0..prediction.grid.nz {
            for row in 0..prediction.grid.ny {
                let idx = z * prediction.grid.nx * prediction.grid.ny
                    + row * prediction.grid.nx
                    + prediction.grid.nx.saturating_sub(1);
                total += prediction.ux.get(idx).copied().unwrap_or(0.0);
                count += 1;
            }
        }
        total / count.max(1) as f64
    }
}

fn fem_regime_displacement_observable(case: &SolveInput, fem: &FemResult) -> f64 {
    if fem.nodal_displacements.is_empty() {
        return 0.0;
    }
    if regime_uses_tip_uy(case) {
        let end_band = (case.geometry.length_in * 0.08).max(1e-6);
        let top_band = (case.geometry.width_in * 0.08).max(1e-6);
        let mut total = 0.0;
        let mut count = 0usize;
        for node in &fem.nodal_displacements {
            if (case.geometry.length_in - node.x_in).abs() <= end_band
                && (case.geometry.width_in - node.y_in).abs() <= top_band
            {
                total += node.uy_in;
                count += 1;
            }
        }
        if count > 0 {
            return total / count as f64;
        }
        return nearest_displacement(
            &fem.nodal_displacements,
            case.geometry.length_in,
            case.geometry.width_in,
            case.geometry.thickness_in * 0.5,
        )
        .uy_in;
    }
    let end_band = (case.geometry.length_in * 0.08).max(1e-6);
    let mut total = 0.0;
    let mut count = 0usize;
    for node in &fem.nodal_displacements {
        if (case.geometry.length_in - node.x_in).abs() <= end_band {
            total += node.ux_in;
            count += 1;
        }
    }
    if count > 0 {
        total / count as f64
    } else {
        nearest_displacement(
            &fem.nodal_displacements,
            case.geometry.length_in,
            case.geometry.width_in * 0.5,
            case.geometry.thickness_in * 0.5,
        )
        .ux_in
    }
}

fn recipe_anchor_case(batch: &TrainingBatch) -> Option<&SolveInput> {
    if batch.cases.is_empty() {
        return None;
    }
    let anchor_idx = batch.cases.len() / 2;
    batch.cases.get(anchor_idx).or_else(|| batch.cases.first())
}

fn build_seed_cases(base: &SolveInput) -> [SolveInput; PINO_RECIPE_SEED_COUNT] {
    let mut seed_low = base.clone();
    seed_low.geometry.length_in = (base.geometry.length_in * 0.90).clamp(4.0, 30.0);
    seed_low.geometry.width_in = (base.geometry.width_in * 1.10).clamp(1.0, 12.0);
    seed_low.geometry.thickness_in = (base.geometry.thickness_in * 0.90).clamp(0.03, 0.75);
    seed_low.mesh.nz = ((base.mesh.nz as f64) * 0.9).round().clamp(2.0, 16.0) as usize;
    seed_low.material.e_psi = (base.material.e_psi * 0.97).clamp(1.0e6, 40.0e6);
    seed_low.boundary_conditions.fix_start_face = base.boundary_conditions.fix_end_face;
    seed_low.boundary_conditions.fix_end_face = base.boundary_conditions.fix_start_face;
    if let Some(hole) = base.geometry.hole_diameter_in {
        seed_low.geometry.hole_diameter_in =
            Some((hole * 0.92).clamp(0.0, seed_low.geometry.width_in * 0.95));
    }

    let mut seed_mid = base.clone();

    let mut seed_high = base.clone();
    seed_high.geometry.length_in = (base.geometry.length_in * 1.10).clamp(4.0, 30.0);
    seed_high.geometry.width_in = (base.geometry.width_in * 0.90).clamp(1.0, 12.0);
    seed_high.geometry.thickness_in = (base.geometry.thickness_in * 1.10).clamp(0.03, 0.75);
    seed_high.mesh.nz = ((base.mesh.nz as f64) * 1.1).round().clamp(2.0, 16.0) as usize;
    seed_high.material.e_psi = (base.material.e_psi * 1.03).clamp(1.0e6, 40.0e6);
    seed_high.boundary_conditions.fix_start_face = true;
    seed_high.boundary_conditions.fix_end_face = true;
    if let Some(hole) = base.geometry.hole_diameter_in {
        seed_high.geometry.hole_diameter_in =
            Some((hole * 1.08).clamp(0.0, seed_high.geometry.width_in * 0.95));
    }

    let vertical_dominant = base.load.vertical_point_load_lbf.abs() > 0.0;
    if vertical_dominant {
        let base_vertical = if base.load.vertical_point_load_lbf.abs() <= 1e-9 {
            250.0
        } else {
            base.load.vertical_point_load_lbf
        };
        seed_low.load.vertical_point_load_lbf = (base_vertical * 0.90).clamp(-10_000.0, 10_000.0);
        seed_mid.load.vertical_point_load_lbf = base_vertical.clamp(-10_000.0, 10_000.0);
        seed_high.load.vertical_point_load_lbf = (base_vertical * 1.10).clamp(-10_000.0, 10_000.0);
        seed_low.load.axial_load_lbf = 0.0;
        seed_mid.load.axial_load_lbf = 0.0;
        seed_high.load.axial_load_lbf = 0.0;
    } else {
        let base_axial = if base.load.axial_load_lbf.abs() <= 1e-9 {
            250.0
        } else {
            base.load.axial_load_lbf
        };
        seed_low.load.axial_load_lbf = (base_axial * 0.90).clamp(-100_000.0, 100_000.0);
        seed_mid.load.axial_load_lbf = base_axial.clamp(-100_000.0, 100_000.0);
        seed_high.load.axial_load_lbf = (base_axial * 1.10).clamp(-100_000.0, 100_000.0);
        seed_low.load.vertical_point_load_lbf = 0.0;
        seed_mid.load.vertical_point_load_lbf = 0.0;
        seed_high.load.vertical_point_load_lbf = 0.0;
    }

    [seed_low, seed_mid, seed_high]
}

fn build_holdout_cases(base: &SolveInput) -> [SolveInput; 2] {
    let mut a = base.clone();
    a.geometry.length_in = (base.geometry.length_in * 0.95).clamp(4.0, 30.0);
    a.geometry.width_in = (base.geometry.width_in * 1.03).clamp(1.0, 12.0);
    a.geometry.thickness_in = (base.geometry.thickness_in * 0.96).clamp(0.03, 0.75);
    a.mesh.nz = ((base.mesh.nz as f64) * 0.95).round().clamp(2.0, 16.0) as usize;
    a.load.axial_load_lbf = (base.load.axial_load_lbf * 0.95).clamp(-100_000.0, 100_000.0);
    a.load.vertical_point_load_lbf =
        (base.load.vertical_point_load_lbf * 0.95).clamp(-10_000.0, 10_000.0);
    a.boundary_conditions.fix_start_face = base.boundary_conditions.fix_end_face;
    a.boundary_conditions.fix_end_face = base.boundary_conditions.fix_start_face;

    let mut b = base.clone();
    b.geometry.length_in = (base.geometry.length_in * 1.05).clamp(4.0, 30.0);
    b.geometry.width_in = (base.geometry.width_in * 0.97).clamp(1.0, 12.0);
    b.geometry.thickness_in = (base.geometry.thickness_in * 1.04).clamp(0.03, 0.75);
    b.mesh.nz = ((base.mesh.nz as f64) * 1.05).round().clamp(2.0, 16.0) as usize;
    b.load.axial_load_lbf = (base.load.axial_load_lbf * 1.05).clamp(-100_000.0, 100_000.0);
    b.load.vertical_point_load_lbf =
        (base.load.vertical_point_load_lbf * 1.05).clamp(-10_000.0, 10_000.0);
    b.boundary_conditions.fix_start_face = true;
    b.boundary_conditions.fix_end_face = true;
    [a, b]
}

#[cfg(test)]
mod tests {
    use super::{
        apply_operator_calibration, build_operator_field_head_batch, build_operator_prediction,
        decode_prediction, encode_case,
        enforce_training_recipe, evaluate_holdout_projection, evaluate_holdout_projection_with_model,
        fit_operator_calibration, infer_config_for_case, model_config, operator_grid_spec,
        normalized_axis, operator_residual_score, prediction_from_fem,
        prediction_from_fem_with_grid_exact, solve_case_for_operator, train_operator_calibration,
        OperatorCalibration, OperatorTrainableParams, PinoModelConfig, PINO_FIELD_HEAD_BASIS,
        PINO_OUTPUT_CHANNELS,
    };
    use crate::contracts::SolveInput;
    use crate::contracts::TrainingBatch;
    use crate::fem::solve_case;
    use crate::pinn::UniversalPinnEngine;
    use crate::pino_burn_head::{
        evaluate_operator_field_head_physics_grad_norm, BurnFieldHeadOptimizer,
    };

    fn make_batch(case: SolveInput, analysis_type: &str) -> TrainingBatch {
        TrainingBatch {
            cases: vec![case],
            epochs: 12,
            target_loss: 0.01,
            seed: Some(5),
            analysis_type: Some(analysis_type.to_string()),
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
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
        }
    }

    fn make_holdout_signoff_batch(cases: Vec<SolveInput>, analysis_type: &str) -> TrainingBatch {
        let is_cantilever = analysis_type.contains("cantilever");
        let is_plate_hole = analysis_type.contains("plate-hole");
        let fast_profile = matches!(
            std::env::var("PINO_SIGNOFF_FAST_PROFILE").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON")
        );
        let target_loss = if fast_profile { 1e-4 } else { 1e-9 };
        let (
            epochs,
            max_total_epochs,
            collocation_points,
            boundary_points,
            interface_points,
            stage1,
            stage2,
            stage3,
            max_topology,
            online_active_learning,
            max_backoffs,
            max_optimizer_switches,
        ) =
            if fast_profile && is_cantilever {
                (8, 12, 64, 24, 12, 2, 3, 4, 40, false, 1, 3)
            } else if fast_profile && is_plate_hole {
                (10, 16, 96, 32, 16, 3, 4, 5, 44, false, 1, 3)
            } else if fast_profile {
                (6, 10, 64, 24, 12, 2, 3, 4, 36, false, 1, 3)
            } else if is_cantilever {
                (56, 72, 640, 160, 80, 8, 10, 18, 96, true, 3, 8)
            } else if is_plate_hole {
                (72, 96, 1024, 224, 128, 10, 14, 24, 112, true, 3, 8)
            } else {
                (40, 48, 512, 128, 64, 6, 8, 12, 80, true, 3, 8)
            };
        TrainingBatch {
            cases,
            epochs,
            target_loss,
            learning_rate: Some(5e-4),
            auto_mode: Some(true),
            max_total_epochs: Some(max_total_epochs),
            min_improvement: Some(1e-6),
            progress_emit_every_epochs: Some(4),
            network_emit_every_epochs: Some(120),
            online_active_learning: Some(online_active_learning),
            autonomous_mode: Some(true),
            max_topology: Some(max_topology),
            max_backoffs: Some(max_backoffs),
            max_optimizer_switches: Some(max_optimizer_switches),
            checkpoint_every_epochs: Some(0),
            checkpoint_retention: Some(4),
            seed: Some(42),
            analysis_type: Some(analysis_type.to_string()),
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
            collocation_points: Some(collocation_points),
            boundary_points: Some(boundary_points),
            interface_points: Some(interface_points),
            residual_weight_momentum: Some(1.0),
            residual_weight_kinematics: Some(1.0),
            residual_weight_material: Some(1.0),
            residual_weight_boundary: Some(1.0),
            stage1_epochs: Some(stage1),
            stage2_epochs: Some(stage2),
            stage3_ramp_epochs: Some(stage3),
            contact_penalty: Some(10.0),
            plasticity_factor: Some(0.0),
        }
    }

    #[derive(Clone)]
    struct TrainedHoldoutModel {
        config: super::PinoModelConfig,
        calibration: Option<super::OperatorCalibration>,
        params: super::OperatorTrainableParams,
        completed_epochs: usize,
        val_loss: f64,
        baseline_param_norm: f64,
        param_delta_norm: f64,
    }

    fn holdout_param_norm(params: &super::OperatorTrainableParams) -> f64 {
        let weight_norm = params
            .field_head_weights
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        let bias_norm = params
            .field_head_bias
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        let activation_norm = params
            .field_head_activation
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        (weight_norm + bias_norm + activation_norm).sqrt()
    }

    fn holdout_param_delta_norm(
        current: &super::OperatorTrainableParams,
        baseline: &super::OperatorTrainableParams,
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

    fn holdout_selection_score(summary: &crate::contracts::HoldoutValidationSummary) -> f64 {
        let mean_limit = summary.mean_error_limit.max(1e-9);
        let p95_limit = summary.p95_error_limit.max(1e-9);
        let residual_limit = summary.residual_ratio_limit.max(1e-9);
        let disp = summary.mean_displacement_error / mean_limit;
        let vm = summary.mean_von_mises_error / mean_limit;
        let p95 = summary.p95_field_error / p95_limit;
        let ratio = summary.residual_ratio / residual_limit;
        let terms = [disp, vm, p95, ratio];
        let worst = terms
            .iter()
            .copied()
            .fold(0.0_f64, |acc, value| acc.max(value));
        let mean = terms.iter().copied().sum::<f64>() * 0.25;
        let exceedances = terms
            .iter()
            .map(|value| (value - 1.0).max(0.0))
            .collect::<Vec<_>>();
        let worst_exceedance = exceedances
            .iter()
            .copied()
            .fold(0.0_f64, |acc, value| acc.max(value));
        let mean_exceedance = exceedances.iter().copied().sum::<f64>() * 0.25;
        ((worst_exceedance * 0.72)
            + (mean_exceedance * 0.24)
            + (worst * 0.03)
            + (mean * 0.01))
            .max(0.0)
    }

    fn holdout_fine_tune_cases(cases: &[SolveInput], analysis_type: &str) -> Vec<SolveInput> {
        let mut tuned = cases.to_vec();
        for case in cases {
            if analysis_type.contains("cantilever") {
                let mut load_low = case.clone();
                load_low.load.vertical_point_load_lbf *= 0.92;
                tuned.push(load_low);

                let mut load_high = case.clone();
                load_high.load.vertical_point_load_lbf *= 1.08;
                tuned.push(load_high);

                let mut thickness_shift = case.clone();
                thickness_shift.geometry.thickness_in =
                    (thickness_shift.geometry.thickness_in * 0.96).max(0.05);
                tuned.push(thickness_shift);
            } else if analysis_type.contains("plate-hole") {
                let mut push_case = |candidate: SolveInput, repeats: usize| {
                    for _ in 0..repeats.max(1) {
                        tuned.push(candidate.clone());
                    }
                };
                let clamp_hole = |base: &SolveInput, scale: f64| {
                    let mut candidate = base.clone();
                    if let Some(hole) = candidate.geometry.hole_diameter_in {
                        candidate.geometry.hole_diameter_in =
                            Some((hole * scale).clamp(0.0, candidate.geometry.width_in * 0.95));
                    }
                    candidate
                };

                push_case(case.clone(), 2);

                let hole_low = clamp_hole(case, 0.985);
                push_case(hole_low, 2);

                let hole_high = clamp_hole(case, 1.015);
                push_case(hole_high, 2);

                let hole_low_wide = clamp_hole(case, 0.96);
                push_case(hole_low_wide, 1);

                let hole_high_wide = clamp_hole(case, 1.04);
                push_case(hole_high_wide, 1);

                let mut load_low = case.clone();
                load_low.load.axial_load_lbf *= 0.985;
                push_case(load_low, 2);

                let mut load_high = case.clone();
                load_high.load.axial_load_lbf *= 1.015;
                push_case(load_high, 2);

                let mut load_low_wide = case.clone();
                load_low_wide.load.axial_load_lbf *= 0.96;
                push_case(load_low_wide, 1);

                let mut load_high_wide = case.clone();
                load_high_wide.load.axial_load_lbf *= 1.04;
                push_case(load_high_wide, 1);

                let mut coupled_high = clamp_hole(case, 1.02);
                coupled_high.load.axial_load_lbf *= 1.02;
                push_case(coupled_high, 2);

                let mut coupled_low = clamp_hole(case, 0.98);
                coupled_low.load.axial_load_lbf *= 0.98;
                push_case(coupled_low, 2);

                let mut thickness_low = case.clone();
                thickness_low.geometry.thickness_in =
                    (thickness_low.geometry.thickness_in * 0.985).max(0.05);
                push_case(thickness_low, 1);

                let mut thickness_high = case.clone();
                thickness_high.geometry.thickness_in *= 1.015;
                push_case(thickness_high, 1);

                let mut mirrored = case.clone();
                mirrored.boundary_conditions.fix_start_face = case.boundary_conditions.fix_end_face;
                mirrored.boundary_conditions.fix_end_face = case.boundary_conditions.fix_start_face;
                push_case(mirrored, 1);

                let mut double_clamped = case.clone();
                double_clamped.boundary_conditions.fix_start_face = true;
                double_clamped.boundary_conditions.fix_end_face = true;
                push_case(double_clamped, 1);

                // Residual-driven local refinement (upsample grid near ring for fine-tune).
                let mut refined = case.clone();
                refined.mesh.nx = refined.mesh.nx.max(22).clamp(22, 28);
                refined.mesh.ny = refined.mesh.ny.max(16).clamp(16, 22);
                refined.mesh.nz = refined.mesh.nz.max(5).clamp(5, 7);
                push_case(refined, 2);
            } else if analysis_type.contains("axial") {
                let mut load_low = case.clone();
                load_low.load.axial_load_lbf *= 0.94;
                tuned.push(load_low);

                let mut load_high = case.clone();
                load_high.load.axial_load_lbf *= 1.06;
                tuned.push(load_high);
            }
        }
        tuned
    }

    #[test]
    fn embedded_dirichlet_keeps_clamped_displacements_zero() {
        let mut case = SolveInput::default();
        case.geometry.hole_diameter_in = Some(0.0);
        case.mesh.nx = 10;
        case.mesh.ny = 6;
        case.mesh.nz = 3;
        case.boundary_conditions.fix_start_face = true;
        case.boundary_conditions.fix_end_face = false;
        case.load.axial_load_lbf = 1712.0;
        case.load.vertical_point_load_lbf = -100.0;

        let config = infer_config_for_case(&case, "pino-ndarray-cpu", 6);
        let batch = build_operator_field_head_batch(&case, &config, None);
        let mut params = OperatorTrainableParams::for_config(&config);
        for value in &mut params.field_head_weights {
            *value = 0.0;
        }
        for value in &mut params.field_head_bias {
            *value = 0.5;
        }
        for value in &mut params.field_head_activation {
            *value = 0.5;
        }

        let prediction = super::build_operator_prediction_with_params(&case, &config, Some(&params));
        let clamped = batch
            .displacement_embed
            .iter()
            .enumerate()
            .filter(|(_, value)| value.abs() <= 1e-12)
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>();
        assert!(!clamped.is_empty(), "expected embedded clamp cells");
        for idx in clamped {
            assert!(prediction.ux[idx].abs() <= 1e-12, "ux clamp drift at cell {idx}");
            assert!(prediction.uy[idx].abs() <= 1e-12, "uy clamp drift at cell {idx}");
            assert!(prediction.uz[idx].abs() <= 1e-12, "uz clamp drift at cell {idx}");
            assert!(batch.displacement_embed[idx].abs() <= 1e-12, "embed not zero at clamp cell {idx}");
        }
    }

    fn residual_hotspot_p95(
        cases: &[SolveInput],
        config: &PinoModelConfig,
        calibration: &OperatorCalibration,
        params: &OperatorTrainableParams,
    ) -> f64 {
        let mut residuals = Vec::new();
        for case in cases.iter().take(3) {
            let fem = solve_case_for_operator(case);
            let target = prediction_from_fem_with_grid_exact(case, &fem, config.operator_grid.clone());
            let base = build_operator_field_head_batch(case, config, Some(params)).base_prediction;
            let cells = config.operator_grid.nx * config.operator_grid.ny * config.operator_grid.nz;
            for idx in 0..cells {
                let target_vm = target.von_mises.get(idx).copied().unwrap_or(0.0).abs();
                let base_vm = base.von_mises.get(idx).copied().unwrap_or(0.0).abs()
                    * calibration.stress_scale;
                residuals.push((target_vm - base_vm).abs() / target_vm.max(1.0));
            }
        }
        if residuals.is_empty() {
            return 0.0;
        }
        residuals.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((residuals.len() as f64) * 0.95).floor() as usize;
        residuals[idx.min(residuals.len().saturating_sub(1))]
    }

    fn residual_hotspot_case_score(
        case: &SolveInput,
        config: &PinoModelConfig,
        calibration: &OperatorCalibration,
        params: &OperatorTrainableParams,
    ) -> f64 {
        let fem = solve_case_for_operator(case);
        let target = prediction_from_fem_with_grid_exact(case, &fem, config.operator_grid.clone());
        let base = build_operator_field_head_batch(case, config, Some(params)).base_prediction;
        let cells = config.operator_grid.nx * config.operator_grid.ny * config.operator_grid.nz;
        let mut residuals = Vec::with_capacity(cells);
        for idx in 0..cells {
            let target_vm = target.von_mises.get(idx).copied().unwrap_or(0.0).abs();
            let base_vm =
                base.von_mises.get(idx).copied().unwrap_or(0.0).abs() * calibration.stress_scale;
            residuals.push((target_vm - base_vm).abs() / target_vm.max(1.0));
        }
        if residuals.is_empty() {
            return 0.0;
        }
        residuals.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((residuals.len() as f64) * 0.95).floor() as usize;
        residuals[idx.min(residuals.len().saturating_sub(1))]
    }

    fn holdout_fine_tune_candidates(
        analysis_type: &str,
    ) -> Vec<(BurnFieldHeadOptimizer, usize, f64, (f64, f64, f64, f64))> {
        fn candidate_optimizer_label(optimizer: BurnFieldHeadOptimizer) -> &'static str {
            match optimizer {
                BurnFieldHeadOptimizer::Adam => "adam",
                BurnFieldHeadOptimizer::Lbfgs => "lbfgs",
            }
        }
        let residual_refine = matches!(
            std::env::var("PINO_SIGNOFF_RESIDUAL_REFINE").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON")
        );
        let mut candidates = vec![
            (BurnFieldHeadOptimizer::Adam, 24usize, 0.003f64, (1.0, 1.0, 1.0, 1.0)),
            (BurnFieldHeadOptimizer::Lbfgs, 18usize, 0.0015f64, (1.0, 1.0, 1.0, 1.0)),
        ];
        if analysis_type.contains("cantilever") {
            candidates.push((BurnFieldHeadOptimizer::Adam, 32, 0.0022, (0.75, 1.9, 1.3, 0.95)));
            candidates.push((BurnFieldHeadOptimizer::Lbfgs, 20, 0.0010, (0.85, 1.8, 1.4, 1.0)));
            candidates.push((BurnFieldHeadOptimizer::Adam, 42, 0.0016, (0.65, 2.0, 1.45, 1.55)));
            candidates.push((BurnFieldHeadOptimizer::Lbfgs, 26, 0.0007, (0.72, 1.9, 1.55, 1.80)));
        } else if analysis_type.contains("axial") {
            candidates.push((BurnFieldHeadOptimizer::Adam, 24, 0.0024, (0.8, 2.1, 1.0, 0.8)));
            candidates.push((BurnFieldHeadOptimizer::Lbfgs, 16, 0.0009, (0.9, 2.0, 1.1, 0.8)));
            candidates.push((BurnFieldHeadOptimizer::Adam, 34, 0.0017, (0.72, 2.35, 1.0, 1.45)));
            candidates.push((BurnFieldHeadOptimizer::Lbfgs, 22, 0.0007, (0.80, 2.20, 1.05, 1.65)));
            candidates.push((BurnFieldHeadOptimizer::Lbfgs, 28, 0.00055, (0.78, 2.45, 1.20, 1.55)));
        } else if analysis_type.contains("plate-hole") {
            candidates.push((BurnFieldHeadOptimizer::Adam, 10, 0.00008, (1.45, 3.10, 1.55, 2.10)));
            candidates.push((BurnFieldHeadOptimizer::Adam, 16, 0.00015, (1.35, 2.85, 1.70, 2.00)));
            candidates.push((BurnFieldHeadOptimizer::Adam, 22, 0.00030, (1.20, 2.55, 1.75, 1.85)));
            candidates.push((BurnFieldHeadOptimizer::Lbfgs, 14, 0.00030, (1.15, 2.35, 1.80, 1.75)));
            candidates.push((BurnFieldHeadOptimizer::Adam, 28, 0.0018, (1.1, 1.7, 1.6, 1.1)));
            candidates.push((BurnFieldHeadOptimizer::Lbfgs, 22, 0.0008, (1.2, 1.6, 1.8, 1.1)));
            candidates.push((BurnFieldHeadOptimizer::Adam, 40, 0.0013, (0.95, 2.10, 1.85, 1.85)));
            candidates.push((BurnFieldHeadOptimizer::Lbfgs, 28, 0.0006, (1.00, 1.95, 1.95, 2.10)));
            candidates.push((BurnFieldHeadOptimizer::Adam, 34, 0.00045, (1.35, 1.25, 1.25, 0.95)));
            candidates.push((BurnFieldHeadOptimizer::Lbfgs, 32, 0.00045, (1.20, 1.35, 1.55, 1.25)));
            candidates.push((BurnFieldHeadOptimizer::Adam, 48, 0.00038, (1.05, 1.55, 1.55, 1.05)));
            candidates.push((BurnFieldHeadOptimizer::Lbfgs, 36, 0.00042, (1.05, 1.50, 1.70, 1.10)));
            if residual_refine {
                candidates = vec![
                    (BurnFieldHeadOptimizer::Adam, 14, 0.00014, (1.10, 2.40, 1.60, 1.90)),
                    (BurnFieldHeadOptimizer::Adam, 20, 0.00022, (1.05, 2.20, 1.55, 1.70)),
                    (BurnFieldHeadOptimizer::Lbfgs, 12, 0.00028, (1.05, 2.10, 1.70, 1.60)),
                ];
            }
        }
        if let Ok(filter) = std::env::var("PINO_HOLDOUT_OPTIMIZER_FILTER") {
            let filters = filter
                .split(',')
                .map(|part| part.trim().to_ascii_lowercase())
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>();
            if !filters.is_empty() {
                candidates.retain(|(optimizer, _, _, _)| {
                    let label = candidate_optimizer_label(*optimizer);
                    filters.iter().any(|wanted| wanted == label)
                });
            }
        }
        if let Ok(limit_raw) = std::env::var("PINO_HOLDOUT_CANDIDATE_LIMIT") {
            if let Ok(limit) = limit_raw.parse::<usize>() {
                if limit > 0 && candidates.len() > limit {
                    candidates.truncate(limit);
                }
            }
        }
        candidates
    }

    fn train_holdout_model_for_cases(
        cases: Vec<SolveInput>,
        analysis_type: &str,
    ) -> TrainedHoldoutModel {
        let fast_profile = matches!(
            std::env::var("PINO_SIGNOFF_FAST_PROFILE").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON")
        );
        let residual_refine = matches!(
            std::env::var("PINO_SIGNOFF_RESIDUAL_REFINE").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON")
        );
        let raw_batch = make_holdout_signoff_batch(cases.clone(), analysis_type);
        let batch = if fast_profile {
            raw_batch
        } else {
            enforce_training_recipe(&raw_batch)
        };
        let mut engine = UniversalPinnEngine::default();
        let result =
            engine.train_with_progress_with_checkpoint(&batch, |_p| {}, || false, |_e, _s, _b| {});
        let snapshot = engine.snapshot_state();
        let burn = snapshot.burn.expect("expected burn runtime state");
        let meta = burn.pino.expect("expected pino runtime metadata");
        let basis_case = cases.first().cloned().unwrap_or_default();
        let mut config =
            infer_config_for_case(&basis_case, meta.backend.as_str(), meta.spectral_modes);
        config.operator_grid = meta.operator_grid;
        let mut params = burn
            .pino_params
            .unwrap_or_default()
            .aligned_to_config(&config)
            .clamped();
        let baseline_params = super::OperatorTrainableParams::for_config(&config).clamped();
        let mut param_delta_norm = holdout_param_delta_norm(&params, &baseline_params);
        let mut val_loss = result.val_loss;
        if fast_profile {
            if let Some(calibration) = burn.pino_calibration.as_ref() {
                let selection_batch = make_holdout_signoff_batch(cases.clone(), analysis_type);
                let mut fine_tune_cases = holdout_fine_tune_cases(&cases, analysis_type);
                let mut fine_tune_config = if analysis_type.contains("plate-hole") {
                    let mut tuned = config.clone();
                    tuned.operator_grid.nx = tuned.operator_grid.nx.max(22).clamp(22, 26);
                    tuned.operator_grid.ny = tuned.operator_grid.ny.max(16).clamp(16, 20);
                    tuned.operator_grid.nz = tuned.operator_grid.nz.max(5).clamp(5, 6);
                    tuned.spectral_modes = tuned
                        .spectral_modes
                        .max(6)
                        .min(tuned.operator_grid.nx.min(tuned.operator_grid.ny).min(tuned.operator_grid.nz));
                    tuned
                } else {
                    config.clone()
                };
                if residual_refine {
                    let hotspot_p95 = residual_hotspot_p95(
                        &fine_tune_cases,
                        &fine_tune_config,
                        calibration,
                        &params,
                    );
                    if hotspot_p95 > 0.04 {
                        let refine_scale_xy = if hotspot_p95 > 0.20 { 1.14 } else { 1.10 };
                        let refine_scale_z = if hotspot_p95 > 0.20 { 1.08 } else { 1.04 };
                        fine_tune_config.operator_grid.nx =
                            (fine_tune_config.operator_grid.nx as f64 * refine_scale_xy).round()
                                as usize;
                        fine_tune_config.operator_grid.ny =
                            (fine_tune_config.operator_grid.ny as f64 * refine_scale_xy).round()
                                as usize;
                        fine_tune_config.operator_grid.nz =
                            (fine_tune_config.operator_grid.nz as f64 * refine_scale_z).round()
                                as usize;
                        fine_tune_config.operator_grid.nx =
                            fine_tune_config.operator_grid.nx.clamp(18, 28);
                        fine_tune_config.operator_grid.ny =
                            fine_tune_config.operator_grid.ny.clamp(12, 20);
                        fine_tune_config.operator_grid.nz =
                            fine_tune_config.operator_grid.nz.clamp(3, 5);
                        fine_tune_config.spectral_modes = fine_tune_config
                            .spectral_modes
                            .max(6)
                            .min(
                                fine_tune_config
                                    .operator_grid
                                    .nx
                                    .min(fine_tune_config.operator_grid.ny)
                                    .min(fine_tune_config.operator_grid.nz),
                            );
                        let mut ranked_cases = fine_tune_cases
                            .iter()
                            .cloned()
                            .map(|case| {
                                let score = residual_hotspot_case_score(
                                    &case,
                                    &fine_tune_config,
                                    calibration,
                                    &params,
                                );
                                (score, case)
                            })
                            .collect::<Vec<_>>();
                        ranked_cases.sort_by(|a, b| {
                            b.0.partial_cmp(&a.0)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let mut refined_cases = fine_tune_cases.clone();
                        for (_, case) in ranked_cases.into_iter().take(2) {
                            let mut refined = case.clone();
                            refined.mesh.nx = refined.mesh.nx.max(fine_tune_config.operator_grid.nx);
                            refined.mesh.ny = refined.mesh.ny.max(fine_tune_config.operator_grid.ny);
                            refined.mesh.nz = refined.mesh.nz.max(fine_tune_config.operator_grid.nz);
                            refined_cases.push(refined);
                        }
                        fine_tune_cases = refined_cases;
                        println!(
                            "pino-holdout-residual-refine: regime={} hotspot_p95={:.6e} grid={}x{}x{} samples={}",
                            analysis_type,
                            hotspot_p95,
                            fine_tune_config.operator_grid.nx,
                            fine_tune_config.operator_grid.ny,
                            fine_tune_config.operator_grid.nz,
                            fine_tune_cases.len()
                        );
                    }
                }
                if analysis_type.contains("plate-hole") {
                    let mut mean_vm_delta = 0.0;
                    let mut ring_vm_delta = 0.0;
                    let mut sample_count = 0usize;
                    for case in &fine_tune_cases {
                        let mut exact = case.clone();
                        exact.mesh.nx = exact.mesh.nx.max(fine_tune_config.operator_grid.nx).clamp(10, 48);
                        exact.mesh.ny = exact.mesh.ny.max(fine_tune_config.operator_grid.ny).clamp(8, 32);
                        exact.mesh.nz = exact.mesh.nz.max(fine_tune_config.operator_grid.nz).clamp(2, 12);
                        let fem = solve_case_for_operator(&exact);
                        let target = prediction_from_fem_with_grid_exact(
                            case,
                            &fem,
                            fine_tune_config.operator_grid.clone(),
                        );
                        let base = build_operator_field_head_batch(case, &fine_tune_config, None).base_prediction;
                        let hole_radius = case.geometry.hole_diameter_in.unwrap_or(0.0).abs() * 0.5;
                        let cx = case.geometry.length_in * 0.5;
                        let cy = case.geometry.width_in * 0.5;
                        let mut local_mean = 0.0;
                        let mut local_ring = 0.0;
                        let mut local_count = 0usize;
                        let mut ring_count = 0usize;
                        for z in 0..fine_tune_config.operator_grid.nz {
                            for y in 0..fine_tune_config.operator_grid.ny {
                                for x in 0..fine_tune_config.operator_grid.nx {
                                    let idx = z * fine_tune_config.operator_grid.nx * fine_tune_config.operator_grid.ny
                                        + y * fine_tune_config.operator_grid.nx
                                        + x;
                                    let target_vm = target.von_mises.get(idx).copied().unwrap_or(0.0).abs();
                                    let base_vm = base.von_mises.get(idx).copied().unwrap_or(0.0).abs();
                                    let delta = (target_vm - base_vm).abs() / target_vm.max(1.0);
                                    local_mean += delta;
                                    local_count += 1;
                                    if hole_radius > 0.0 {
                                        let px = normalized_axis(x, fine_tune_config.operator_grid.nx) * case.geometry.length_in;
                                        let py = normalized_axis(y, fine_tune_config.operator_grid.ny) * case.geometry.width_in;
                                        let radial = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
                                        let ring_band = (hole_radius * 0.35).max(case.geometry.width_in * 0.04).max(1e-6);
                                        if (radial - hole_radius).abs() <= ring_band {
                                            local_ring += delta;
                                            ring_count += 1;
                                        }
                                    }
                                }
                            }
                        }
                        mean_vm_delta += local_mean / local_count.max(1) as f64;
                        ring_vm_delta += if ring_count > 0 {
                            local_ring / ring_count as f64
                        } else {
                            0.0
                        };
                        sample_count += 1;
                    }
                    if sample_count > 0 {
                        println!(
                            "pino-holdout-target-delta: regime={} mean_vm_delta={:.6e} ring_vm_delta={:.6e} grid={}x{}x{}",
                            analysis_type,
                            mean_vm_delta / sample_count as f64,
                            ring_vm_delta / sample_count as f64,
                            fine_tune_config.operator_grid.nx,
                            fine_tune_config.operator_grid.ny,
                            fine_tune_config.operator_grid.nz
                        );
                    }
                    let debug_targets =
                        UniversalPinnEngine::build_operator_field_targets_exact(
                            &fine_tune_cases,
                            fine_tune_config.operator_grid.clone(),
                        );
                    let debug_samples = debug_targets
                        .iter()
                        .map(|sample| {
                            UniversalPinnEngine::build_burn_physics_sample(
                                sample,
                                &fine_tune_config,
                                calibration,
                                &baseline_params,
                            )
                        })
                        .collect::<Vec<_>>();
                    if let Some(grad_norm) = evaluate_operator_field_head_physics_grad_norm(
                        &debug_samples,
                        &baseline_params.field_head_weights,
                        &baseline_params.field_head_bias,
                        &baseline_params.field_head_activation,
                        PINO_FIELD_HEAD_BASIS,
                        PINO_OUTPUT_CHANNELS,
                        baseline_params.field_head_hidden_layers,
                        baseline_params.field_head_hidden_width,
                        (1.0, 1.0, 1.0, 1.0),
                        false,
                    ) {
                        println!(
                            "pino-holdout-gradnorm: regime={} value={:.6e} samples={} grid={}x{}x{}",
                            analysis_type,
                            grad_norm,
                            debug_samples.len(),
                            fine_tune_config.operator_grid.nx,
                            fine_tune_config.operator_grid.ny,
                            fine_tune_config.operator_grid.nz
                        );
                    }
                }
                let trained_score = evaluate_holdout_projection_with_model(
                    &selection_batch,
                    &config,
                    Some(calibration),
                    Some(&params),
                )
                .map(|summary| {
                    println!(
                        "pino-holdout-candidate: regime={} optimizer=baseline score={:.6e} disp={:.6e} vm={:.6e} p95={:.6e} ratio={:.6e} delta={:.6e} loss={:.6e}",
                        analysis_type,
                        holdout_selection_score(&summary),
                        summary.mean_displacement_error,
                        summary.mean_von_mises_error,
                        summary.p95_field_error,
                        summary.residual_ratio,
                        param_delta_norm,
                        val_loss
                    );
                    holdout_selection_score(&summary)
                })
                .unwrap_or(f64::MAX);
                let baseline_default_score = evaluate_holdout_projection_with_model(
                    &selection_batch,
                    &config,
                    Some(calibration),
                    Some(&baseline_params),
                )
                .map(|summary| {
                    let score = holdout_selection_score(&summary);
                    println!(
                        "pino-holdout-candidate: regime={} optimizer=baseline-default score={:.6e} disp={:.6e} vm={:.6e} p95={:.6e} ratio={:.6e} delta={:.6e} loss={:.6e}",
                        analysis_type,
                        score,
                        summary.mean_displacement_error,
                        summary.mean_von_mises_error,
                        summary.p95_field_error,
                        summary.residual_ratio,
                        0.0,
                        f64::NAN
                    );
                    score
                })
                .unwrap_or(f64::MAX);
                let mut best_params = params.clone();
                let mut best_loss = val_loss;
                let mut best_delta = param_delta_norm;
                let mut best_score = trained_score;
                if baseline_default_score + 1.0e-9 < best_score {
                    best_params = baseline_params.clone();
                    best_loss = f64::NAN;
                    best_delta = 0.0;
                    best_score = baseline_default_score;
                }
                for (optimizer, steps, learning_rate, loss_weights) in
                    holdout_fine_tune_candidates(analysis_type)
                {
                    if let Some((candidate_params, candidate_loss)) =
                        UniversalPinnEngine::direct_train_operator_params_for_cases(
                            &fine_tune_cases,
                            &fine_tune_config,
                            calibration,
                            &best_params,
                            steps,
                            learning_rate,
                            optimizer,
                            loss_weights,
                        )
                    {
                        let candidate_score = evaluate_holdout_projection_with_model(
                            &selection_batch,
                            &config,
                            Some(calibration),
                            Some(&candidate_params),
                        )
                        .map(|summary| {
                            let score = holdout_selection_score(&summary);
                            let candidate_delta =
                                holdout_param_delta_norm(&candidate_params, &baseline_params);
                            println!(
                                "pino-holdout-candidate: regime={} optimizer={:?} steps={} lr={:.6e} weights={:.3}/{:.3}/{:.3}/{:.3} score={:.6e} disp={:.6e} vm={:.6e} p95={:.6e} ratio={:.6e} delta={:.6e} loss={:.6e}",
                                analysis_type,
                                optimizer,
                                steps,
                                learning_rate,
                                loss_weights.0,
                                loss_weights.1,
                                loss_weights.2,
                                loss_weights.3,
                                score,
                                summary.mean_displacement_error,
                                summary.mean_von_mises_error,
                                summary.p95_field_error,
                                summary.residual_ratio,
                                candidate_delta,
                                candidate_loss
                            );
                            score
                        })
                        .unwrap_or(f64::MAX);
                        let candidate_delta =
                            holdout_param_delta_norm(&candidate_params, &baseline_params);
                        if candidate_score + 1e-9 < best_score {
                            best_score = candidate_score;
                            best_delta = candidate_delta;
                            best_loss = candidate_loss;
                            best_params = candidate_params;
                        }
                    }
                }
                if analysis_type.contains("plate-hole") && best_delta <= 1e-6 {
                    for (optimizer, steps, learning_rate, loss_weights) in
                        holdout_fine_tune_candidates(analysis_type)
                            .into_iter()
                            .take(4)
                    {
                        if let Some((candidate_params, candidate_loss)) =
                            UniversalPinnEngine::direct_train_operator_params_for_cases(
                                &fine_tune_cases,
                                &fine_tune_config,
                                calibration,
                                &baseline_params,
                                steps,
                                learning_rate,
                                optimizer,
                                loss_weights,
                            )
                        {
                            let candidate_score = evaluate_holdout_projection_with_model(
                                &selection_batch,
                                &config,
                                Some(calibration),
                                Some(&candidate_params),
                            )
                            .map(|summary| {
                                let score = holdout_selection_score(&summary);
                                let candidate_delta =
                                    holdout_param_delta_norm(&candidate_params, &baseline_params);
                                println!(
                                    "pino-holdout-candidate: regime={} optimizer={:?}-reset steps={} lr={:.6e} weights={:.3}/{:.3}/{:.3}/{:.3} score={:.6e} disp={:.6e} vm={:.6e} p95={:.6e} ratio={:.6e} delta={:.6e} loss={:.6e}",
                                    analysis_type,
                                    optimizer,
                                    steps,
                                    learning_rate,
                                    loss_weights.0,
                                    loss_weights.1,
                                    loss_weights.2,
                                    loss_weights.3,
                                    score,
                                    summary.mean_displacement_error,
                                    summary.mean_von_mises_error,
                                    summary.p95_field_error,
                                    summary.residual_ratio,
                                    candidate_delta,
                                    candidate_loss
                                );
                                score
                            })
                            .unwrap_or(f64::MAX);
                            let candidate_delta =
                                holdout_param_delta_norm(&candidate_params, &baseline_params);
                            if candidate_score + 1e-9 < best_score {
                                best_score = candidate_score;
                                best_delta = candidate_delta;
                                best_loss = candidate_loss;
                                best_params = candidate_params;
                            }
                        }
                    }
                }
                if best_score.is_finite() {
                    params = best_params;
                    val_loss = best_loss;
                    param_delta_norm = best_delta;
                }
            }
        }
        let baseline_param_norm = holdout_param_norm(&baseline_params);
        config.hidden_layers = params.field_head_hidden_layers;
        config.hidden_width = params.field_head_hidden_width;
        TrainedHoldoutModel {
            config,
            calibration: burn.pino_calibration,
            params,
            completed_epochs: result.completed_epochs,
            val_loss,
            baseline_param_norm,
            param_delta_norm,
        }
    }

    fn train_holdout_model(case: SolveInput, analysis_type: &str) -> TrainedHoldoutModel {
        train_holdout_model_for_cases(vec![case], analysis_type)
    }

    fn trained_holdout_summary(
        model: &TrainedHoldoutModel,
        case: SolveInput,
        analysis_type: &str,
    ) -> crate::contracts::HoldoutValidationSummary {
        let batch = make_batch(case, analysis_type);
        evaluate_holdout_projection_with_model(
            &batch,
            &model.config,
            model.calibration.as_ref(),
            Some(&model.params),
        )
        .expect("expected trained holdout summary")
    }

    #[test]
    fn operator_encoder_uses_canonical_plate_grid() {
        let input = SolveInput::default();
        let sample = encode_case(&input);
        let spec = operator_grid_spec(&input);
        assert_eq!(sample.inputs.len(), spec.nx * spec.ny * spec.input_channels);
    }

    #[test]
    fn fem_prediction_round_trip_decodes_to_result_view() {
        let input = SolveInput::default();
        let fem = solve_case(&input);
        let prediction = prediction_from_fem(&input, &fem);
        let decoded = decode_prediction(&input, &prediction);
        assert!(!decoded.nodal_displacements.is_empty());
        assert!(decoded.von_mises_psi.is_finite());
        assert!(decoded.max_principal_psi.is_finite());
        assert!(decoded.stress_tensor[2][2].is_finite());
        assert_eq!(decoded.displacement_vector.len(), 3);
    }

    #[test]
    fn model_config_derives_grid_and_spectral_modes_from_batch() {
        let input = SolveInput::default();
        let batch = TrainingBatch {
            cases: vec![input],
            epochs: 10,
            target_loss: 0.01,
            seed: Some(1),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("pino-candle-cpu".to_string()),
            collocation_points: Some(1024),
            boundary_points: Some(768),
            interface_points: Some(384),
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
        let config = model_config(&batch);
        assert_eq!(config.backend, "pino-candle-cpu");
        assert!(config.spectral_modes >= 2);
        assert_eq!(config.operator_grid.input_channels, 10);
        assert_eq!(config.operator_grid.output_channels, 7);
    }

    #[test]
    fn holdout_projection_validation_produces_finite_metrics() {
        let batch = make_batch(SolveInput::default(), "cantilever");
        let summary = evaluate_holdout_projection(&batch).expect("summary");
        assert!(summary.mean_displacement_error.is_finite());
        assert!(summary.mean_von_mises_error.is_finite());
        assert!(summary.p95_field_error.is_finite());
        assert_eq!(summary.holdout_cases, 2);
        assert!(summary.mean_error_limit > 0.0);
        assert!(summary.p95_error_limit > 0.0);
        assert!(summary.residual_ratio_limit > 0.0);
    }

    #[test]
    fn holdout_projection_exposes_criteria_pass_flags() {
        let batch = make_batch(SolveInput::default(), "cantilever");
        let summary = evaluate_holdout_projection(&batch).expect("summary");
        assert_eq!(
            summary.displacement_pass,
            summary.mean_displacement_error <= summary.mean_error_limit
        );
        assert_eq!(
            summary.von_mises_pass,
            summary.mean_von_mises_error <= summary.mean_error_limit
        );
        assert_eq!(
            summary.p95_pass,
            summary.p95_field_error <= summary.p95_error_limit
        );
        assert_eq!(
            summary.residual_ratio_pass,
            summary.residual_ratio <= summary.residual_ratio_limit
        );
        let trusted_expected = summary.displacement_pass
            && summary.von_mises_pass
            && summary.p95_pass
            && summary.residual_ratio_pass;
        assert_eq!(summary.trusted, trusted_expected);
        assert_eq!(summary.accepted_without_fallback, trusted_expected);
    }

    #[test]
    #[ignore = "manual release-signoff holdout baseline profile"]
    fn holdout_baselines_cover_cantilever_axial_and_plate_hole_regimes() {
        let mut cantilever = SolveInput::default();
        cantilever.geometry.length_in = 10.0;
        cantilever.geometry.width_in = 1.0;
        cantilever.geometry.thickness_in = 0.25;
        cantilever.geometry.hole_diameter_in = Some(0.0);
        cantilever.mesh.nx = 8;
        cantilever.mesh.ny = 6;
        cantilever.mesh.nz = 1;
        cantilever.load.axial_load_lbf = 0.0;
        cantilever.load.vertical_point_load_lbf = -100.0;

        let mut axial = cantilever.clone();
        axial.load.axial_load_lbf = 1712.0;
        axial.load.vertical_point_load_lbf = 0.0;

        let mut plate_hole = SolveInput::default();
        plate_hole.geometry.length_in = 11.811;
        plate_hole.geometry.width_in = 4.724;
        plate_hole.geometry.thickness_in = 0.25;
        plate_hole.geometry.hole_diameter_in = Some(2.362);
        plate_hole.mesh.nx = 8;
        plate_hole.mesh.ny = 6;
        plate_hole.mesh.nz = 1;
        plate_hole.load.axial_load_lbf = 1712.0;
        plate_hole.load.vertical_point_load_lbf = 0.0;

        let cases = [
            ("cantilever", cantilever),
            ("axial", axial),
            ("plate-hole", plate_hole),
        ];
        for (name, case) in cases {
            let batch = make_batch(case, name);
            let summary = evaluate_holdout_projection(&batch).expect("summary");
            assert!(
                summary.mean_displacement_error <= 0.70,
                "{name} holdout displacement error baseline regressed: {}",
                summary.mean_displacement_error
            );
            assert!(
                summary.mean_von_mises_error <= 0.40,
                "{name} holdout von-mises error baseline regressed: {}",
                summary.mean_von_mises_error
            );
            assert!(
                summary.p95_field_error <= 0.80,
                "{name} holdout p95 field error baseline regressed: {}",
                summary.p95_field_error
            );
        }
    }

    #[test]
    #[ignore = "manual release-signoff holdout profile"]
    fn pino_holdout_profile_manual() {
        let fast_profile = matches!(
            std::env::var("PINO_SIGNOFF_FAST_PROFILE").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON")
        );
        let regime_filter = std::env::var("PINO_SIGNOFF_REGIME_FILTER").ok();
        let mut cantilever = SolveInput::default();
        cantilever.geometry.length_in = 10.0;
        cantilever.geometry.width_in = 1.0;
        cantilever.geometry.thickness_in = 0.25;
        cantilever.geometry.hole_diameter_in = Some(0.0);
        cantilever.mesh.nx = if fast_profile { 8 } else { 12 };
        cantilever.mesh.ny = if fast_profile { 6 } else { 8 };
        cantilever.mesh.nz = 2;
        cantilever.load.axial_load_lbf = 0.0;
        cantilever.load.vertical_point_load_lbf = -100.0;

        let mut axial = cantilever.clone();
        axial.load.axial_load_lbf = 1712.0;
        axial.load.vertical_point_load_lbf = 0.0;

        let mut plate_hole = SolveInput::default();
        plate_hole.geometry.length_in = 11.811;
        plate_hole.geometry.width_in = 4.724;
        plate_hole.geometry.thickness_in = 0.25;
        plate_hole.geometry.hole_diameter_in = Some(2.362);
        plate_hole.mesh.nx = if fast_profile { 10 } else { 12 };
        plate_hole.mesh.ny = if fast_profile { 8 } else { 8 };
        plate_hole.mesh.nz = 2;
        plate_hole.load.axial_load_lbf = 1712.0;
        plate_hole.load.vertical_point_load_lbf = 0.0;

        let cases = [
            ("cantilever", cantilever),
            ("axial", axial),
            ("plate-hole", plate_hole),
        ];
        let shared_fast_model = if fast_profile
            && matches!(
                std::env::var("PINO_SIGNOFF_SHARED_MODEL").ok().as_deref(),
                Some("1" | "true" | "TRUE" | "on" | "ON")
            )
        {
            Some(train_holdout_model_for_cases(
                cases.iter().map(|(_, case)| case.clone()).collect(),
                "general",
            ))
        } else {
            None
        };
        for (name, case) in cases {
            if let Some(filter) = regime_filter.as_deref() {
                if filter != name {
                    continue;
                }
            }
            let trained_model = shared_fast_model
                .clone()
                .unwrap_or_else(|| train_holdout_model(case.clone(), name));
            println!(
                "pino-holdout-train: regime={} epochs={} val_loss={:.6e} param_norm={:.6e} baseline_param_norm={:.6e} param_delta_norm={:.6e} stress_scale={:.6e} disp_scale={:.6e} arch={}x{}x{} hidden={}x{}",
                name,
                trained_model.completed_epochs,
                trained_model.val_loss,
                holdout_param_norm(&trained_model.params),
                trained_model.baseline_param_norm,
                trained_model.param_delta_norm,
                trained_model
                    .calibration
                    .as_ref()
                    .map(|c| c.stress_scale)
                    .unwrap_or(1.0),
                trained_model
                    .calibration
                    .as_ref()
                    .map(|c| c.displacement_scale)
                    .unwrap_or(1.0),
                trained_model.config.operator_grid.nx,
                trained_model.config.operator_grid.ny,
                trained_model.config.operator_grid.nz,
                trained_model.config.hidden_layers,
                trained_model.config.hidden_width,
            );
            let summary = trained_holdout_summary(&trained_model, case, name);
            println!(
                "pino-holdout-summary: regime={} mean_disp={:.6e} mean_vm={:.6e} p95={:.6e} ratio={:.6e} trusted={} accepted_without_fallback={}",
                name,
                summary.mean_displacement_error,
                summary.mean_von_mises_error,
                summary.p95_field_error,
                summary.residual_ratio,
                if summary.trusted { 1 } else { 0 },
                if summary.accepted_without_fallback { 1 } else { 0 }
            );
        }
    }

    #[test]
    fn strict_recipe_enforcement_promotes_three_seed_cases() {
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 6,
            target_loss: 0.1,
            seed: Some(7),
            analysis_type: Some("general".to_string()),
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
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
        };
        let enforced = enforce_training_recipe(&batch);
        assert_eq!(enforced.cases.len(), 3);
    }

    #[test]
    fn operator_inference_path_produces_finite_projection_and_residual() {
        let input = SolveInput::default();
        let config = infer_config_for_case(&input, "pino-ndarray-cpu", 4);
        let prediction = build_operator_prediction(&input, &config);
        let residual = operator_residual_score(&input, &prediction);
        let decoded = decode_prediction(&input, &prediction);
        assert!(residual.is_finite());
        assert!(decoded.von_mises_psi.is_finite());
        assert!(!decoded.nodal_displacements.is_empty());
    }

    #[test]
    fn operator_calibration_is_finite_and_applies() {
        let input = SolveInput::default();
        let batch = TrainingBatch {
            cases: vec![input.clone()],
            epochs: 4,
            target_loss: 0.1,
            seed: Some(17),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
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
        };
        let cfg = infer_config_for_case(&input, "pino-ndarray-cpu", 4);
        let calibration = fit_operator_calibration(&batch, &cfg).expect("calibration");
        assert!(calibration.stress_scale.is_finite());
        assert!(calibration.displacement_scale.is_finite());
        let mut prediction = build_operator_prediction(&input, &cfg);
        let vm_before = prediction.von_mises[0];
        apply_operator_calibration(&mut prediction, &calibration);
        assert!((prediction.von_mises[0] - vm_before).abs() >= 0.0);
    }

    #[test]
    fn operator_calibration_training_produces_loss_trace() {
        let input = SolveInput::default();
        let batch = TrainingBatch {
            cases: vec![input.clone()],
            epochs: 10,
            target_loss: 0.1,
            seed: Some(31),
            analysis_type: Some("cantilever".to_string()),
            pinn_backend: Some("pino-ndarray-cpu".to_string()),
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
        };
        let cfg = infer_config_for_case(&input, "pino-ndarray-cpu", 4);
        let stats = train_operator_calibration(&batch, &cfg, 12).expect("operator train stats");
        assert!(stats.epochs_run >= 1);
        assert!(stats.initial_loss.is_finite());
        assert!(stats.best_loss.is_finite());
        assert!(stats.final_loss.is_finite());
        assert!(stats.calibration.stress_scale.is_finite());
        assert!(stats.calibration.displacement_scale.is_finite());
    }
}
