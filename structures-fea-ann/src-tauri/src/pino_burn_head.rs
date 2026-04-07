use crate::pino::{
    PINO_AUX_OUTPUT_CHANNELS, PINO_DISPLACEMENT_OUTPUT_CHANNELS, PINO_LAAF_GAIN,
    PINO_OUTPUT_CHANNELS, PINO_PRIMARY_OUTPUT_CHANNELS,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "pino-ndarray-cpu")]
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "pino-ndarray-cpu")]
use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
#[cfg(feature = "pino-ndarray-cpu")]
use burn::tensor::{Data, Tensor};
#[cfg(feature = "pino-ndarray-cpu")]
use burn_autodiff::{Autodiff, grads::Gradients};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurnFieldHeadOptimizer {
    Adam,
    Lbfgs,
}

#[derive(Debug, Clone)]
pub struct BurnPhysicsSample {
    pub features: Vec<f64>,
    pub target_fields: Vec<f64>,
    pub base_fields: Vec<f64>,
    pub correction_scales: Vec<f64>,
    pub output_dim: usize,
    pub characteristic_disp_scale: f64,
    pub characteristic_stress_scale: f64,
    pub observable_target: Vec<f64>,
    pub observable_scale: Vec<f64>,
    pub observable_weight: Vec<f64>,
    pub observable_projection: Vec<f64>,
    pub observable_fifth_uses_vm: bool,
    pub benchmark_vm_active: Vec<f64>,
    pub benchmark_sxx_peak: Vec<f64>,
    pub stress_focus: Vec<f64>,
    pub ring_loss_mask: Vec<f64>,
    pub mask: Vec<f64>,
    pub clamp: Vec<f64>,
    pub displacement_embed: Vec<f64>,
    pub traction_mask: Vec<f64>,
    pub traction_normal: Vec<f64>,
    pub traction_target: Vec<f64>,
    pub grid_nx: usize,
    pub grid_ny: usize,
    pub grid_nz: usize,
    pub spectral_modes: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub e_modulus: f64,
    pub poisson: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BurnPhysicsLossBreakdown {
    pub total: f64,
    pub data: f64,
    pub displacement_fit: f64,
    pub stress_fit: f64,
    pub observable: f64,
    pub auxiliary: f64,
    pub auxiliary_data: f64,
    pub invariant: f64,
    pub equilibrium: f64,
    pub constitutive: f64,
    pub constitutive_normal: f64,
    pub constitutive_shear: f64,
    pub weak_energy: f64,
    pub boundary: f64,
}

#[derive(Debug, Clone)]
pub struct BurnPhysicsTrainingOutcome {
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
    pub activation: Vec<f64>,
    pub breakdown: BurnPhysicsLossBreakdown,
}

const PINO_OBSERVABLE_COUNT: usize = 5;
static ISOLATED_EXACT_CANTILEVER_OVERRIDE: AtomicBool = AtomicBool::new(false);
static BENCHMARK_CANTILEVER_OVERRIDE: AtomicBool = AtomicBool::new(false);

pub fn set_isolated_exact_cantilever_override(enabled: bool) {
    ISOLATED_EXACT_CANTILEVER_OVERRIDE.store(enabled, Ordering::Relaxed);
}

pub fn set_benchmark_cantilever_override(enabled: bool) {
    BENCHMARK_CANTILEVER_OVERRIDE.store(enabled, Ordering::Relaxed);
}

fn isolated_exact_cantilever_surface(exact_surface: bool) -> bool {
    exact_surface
        && (ISOLATED_EXACT_CANTILEVER_OVERRIDE.load(Ordering::Relaxed)
            || env_flag("PINO_HEADLESS_ISOLATED_EXACT_CANTILEVER", false)
            || matches!(
                std::env::var("PINO_HEADLESS_BENCHMARK_ID").ok().as_deref(),
                Some("benchmark_cantilever_2d")
            ))
}

fn benchmark_cantilever_characteristic_training(exact_surface: bool) -> bool {
    exact_surface
        && (BENCHMARK_CANTILEVER_OVERRIDE.load(Ordering::Relaxed)
            || matches!(
                std::env::var("PINO_HEADLESS_BENCHMARK_ID").ok().as_deref(),
                Some("benchmark_cantilever_2d")
            ))
}

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

#[cfg(feature = "pino-ndarray-cpu")]
type HeadBackend = Autodiff<burn_ndarray::NdArray<f32>>;
#[cfg(feature = "pino-ndarray-cpu")]
type HeadInnerBackend = burn_ndarray::NdArray<f32>;

#[cfg(feature = "pino-ndarray-cpu")]
#[derive(Module, Debug)]
struct BurnFieldHead<B: burn::tensor::backend::Backend> {
    lift_weight: Param<Tensor<B, 2>>,
    lift_bias: Param<Tensor<B, 1>>,
    lift_activation_scale: Param<Tensor<B, 1>>,
    operator_self_weights: Vec<Param<Tensor<B, 2>>>,
    operator_local_weights: Vec<Param<Tensor<B, 2>>>,
    operator_global_weights: Vec<Param<Tensor<B, 2>>>,
    operator_contrast_weights: Vec<Param<Tensor<B, 2>>>,
    operator_biases: Vec<Param<Tensor<B, 1>>>,
    operator_activation_scales: Vec<Param<Tensor<B, 1>>>,
    output_weight: Param<Tensor<B, 2>>,
    output_bias: Param<Tensor<B, 1>>,
    hidden_layers: usize,
    hidden_width: usize,
}

#[cfg(feature = "pino-ndarray-cpu")]
struct PhysicsSampleTensors {
    features: Tensor<HeadBackend, 2>,
    base_fields: Tensor<HeadBackend, 2>,
    correction_scales: Tensor<HeadBackend, 2>,
    output_dim: usize,
    primary_target_fields: Tensor<HeadBackend, 2>,
    auxiliary_target_fields: Tensor<HeadBackend, 2>,
    auxiliary_base_fields: Tensor<HeadBackend, 2>,
    auxiliary_scale_fields: Tensor<HeadBackend, 2>,
    neighbor_average: Tensor<HeadBackend, 2>,
    spectral_projection: Tensor<HeadBackend, 2>,
    mask_cells: Tensor<HeadBackend, 2>,
    clamp_cells: Tensor<HeadBackend, 2>,
    traction_normals: Tensor<HeadBackend, 2>,
    traction_targets: Tensor<HeadBackend, 2>,
    primary_mask_fields: Tensor<HeadBackend, 2>,
    auxiliary_mask_fields: Tensor<HeadBackend, 2>,
    traction_mask_fields: Tensor<HeadBackend, 2>,
    observable_target: Tensor<HeadBackend, 2>,
    observable_scale: Tensor<HeadBackend, 2>,
    observable_weight: Tensor<HeadBackend, 2>,
    observable_projection: Tensor<HeadBackend, 2>,
    observable_fifth_uses_vm: bool,
    benchmark_vm_active_cells: Tensor<HeadBackend, 2>,
    benchmark_sxx_peak_cells: Tensor<HeadBackend, 2>,
    stress_focus_cells: Tensor<HeadBackend, 2>,
    stress_focus_primary_fields: Tensor<HeadBackend, 2>,
    stress_focus_auxiliary_fields: Tensor<HeadBackend, 2>,
    ring_loss_mask: Tensor<HeadBackend, 2>,
    coarse_primary_target: Tensor<HeadBackend, 2>,
    coarse_auxiliary_target: Tensor<HeadBackend, 2>,
    diff_x: Tensor<HeadBackend, 2>,
    diff_y: Tensor<HeadBackend, 2>,
    diff_z: Tensor<HeadBackend, 2>,
    grid_nx: usize,
    grid_ny: usize,
    grid_nz: usize,
    e_modulus: f32,
    poisson: f32,
    disp_scale: f32,
    stress_scale: f32,
    equilibrium_scale: f32,
    primary_channel_scale: Tensor<HeadBackend, 2>,
    auxiliary_channel_scale: Tensor<HeadBackend, 2>,
    boundary_scale: f32,
}

#[cfg(feature = "pino-ndarray-cpu")]
struct PhysicsLossTensors {
    total: Tensor<HeadBackend, 1>,
    data: Tensor<HeadBackend, 1>,
    displacement_fit: Tensor<HeadBackend, 1>,
    stress_fit: Tensor<HeadBackend, 1>,
    observable: Tensor<HeadBackend, 1>,
    auxiliary: Tensor<HeadBackend, 1>,
    auxiliary_data: Tensor<HeadBackend, 1>,
    invariant: Tensor<HeadBackend, 1>,
    equilibrium: Tensor<HeadBackend, 1>,
    constitutive: Tensor<HeadBackend, 1>,
    constitutive_normal: Tensor<HeadBackend, 1>,
    constitutive_shear: Tensor<HeadBackend, 1>,
    weak_energy: Tensor<HeadBackend, 1>,
    boundary: Tensor<HeadBackend, 1>,
}

#[cfg(feature = "pino-ndarray-cpu")]
struct DenseGradientCollector<'a> {
    grads: &'a mut GradientsParams,
    values: Vec<f64>,
}

#[cfg(feature = "pino-ndarray-cpu")]
impl<'a> DenseGradientCollector<'a> {
    fn new(grads: &'a mut GradientsParams, capacity: usize) -> Self {
        Self {
            grads,
            values: Vec::with_capacity(capacity.max(1)),
        }
    }

    fn finish(self) -> Vec<f64> {
        self.values
    }
}

#[cfg(feature = "pino-ndarray-cpu")]
impl ModuleVisitor<HeadBackend> for DenseGradientCollector<'_> {
    fn visit_float<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<HeadBackend, D>) {
        let len = tensor.clone().into_data().convert::<f64>().value.len();
        if let Some(grad) = self.grads.remove::<HeadInnerBackend, D>(id) {
            self.values.extend(grad.into_data().convert::<f64>().value);
        } else {
            self.values.extend(std::iter::repeat_n(0.0, len));
        }
    }
}

#[cfg(feature = "pino-ndarray-cpu")]
#[derive(Debug)]
struct DenseLbfgsState {
    history_size: usize,
    s_history: Vec<Vec<f64>>,
    y_history: Vec<Vec<f64>>,
    rho_history: Vec<f64>,
    q_buffer: Vec<f64>,
    direction: Vec<f64>,
    candidate: Vec<f64>,
    alpha_buffer: Vec<f64>,
}

#[cfg(feature = "pino-ndarray-cpu")]
impl DenseLbfgsState {
    fn new(dim: usize, history_size: usize) -> Self {
        Self {
            history_size: history_size.max(1),
            s_history: Vec::with_capacity(history_size.max(1)),
            y_history: Vec::with_capacity(history_size.max(1)),
            rho_history: Vec::with_capacity(history_size.max(1)),
            q_buffer: vec![0.0; dim],
            direction: vec![0.0; dim],
            candidate: vec![0.0; dim],
            alpha_buffer: vec![0.0; history_size.max(1)],
        }
    }

    fn ensure_dim(&mut self, dim: usize) {
        if self.q_buffer.len() != dim {
            self.q_buffer.resize(dim, 0.0);
            self.direction.resize(dim, 0.0);
            self.candidate.resize(dim, 0.0);
            self.reset_history();
        }
    }

    fn reset_history(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
        self.rho_history.clear();
        self.alpha_buffer.fill(0.0);
    }

    fn compute_direction(&mut self, gradient: &[f64]) -> &[f64] {
        self.ensure_dim(gradient.len());
        self.q_buffer.clone_from_slice(gradient);
        let history_len = self.s_history.len();
        if self.alpha_buffer.len() < history_len {
            self.alpha_buffer.resize(history_len, 0.0);
        }
        for idx in (0..history_len).rev() {
            let alpha = self.rho_history[idx] * dot(&self.s_history[idx], &self.q_buffer);
            self.alpha_buffer[idx] = alpha;
            axpy(&mut self.q_buffer, -alpha, &self.y_history[idx]);
        }
        let mut gamma = 1.0;
        if let Some((s_last, y_last)) = self.s_history.last().zip(self.y_history.last()) {
            let yy = dot(y_last, y_last);
            if yy > 1e-18 {
                gamma = (dot(s_last, y_last) / yy).clamp(1e-4, 1e4);
            }
        }
        for (dst, src) in self.direction.iter_mut().zip(self.q_buffer.iter().copied()) {
            *dst = gamma * src;
        }
        for idx in 0..history_len {
            let beta = self.rho_history[idx] * dot(&self.y_history[idx], &self.direction);
            axpy(
                &mut self.direction,
                self.alpha_buffer[idx] - beta,
                &self.s_history[idx],
            );
        }
        for value in &mut self.direction {
            *value = -*value;
        }
        if dot(gradient, &self.direction) >= -1e-12 {
            for (dst, src) in self.direction.iter_mut().zip(gradient.iter().copied()) {
                *dst = -src;
            }
        }
        &self.direction
    }

    fn candidate_from(&mut self, param: &[f64], direction: &[f64], step: f64) -> &[f64] {
        self.ensure_dim(param.len());
        for ((dst, base), dir) in self
            .candidate
            .iter_mut()
            .zip(param.iter().copied())
            .zip(direction.iter().copied())
        {
            *dst = base + step * dir;
        }
        &self.candidate
    }

    fn push_history(&mut self, step_delta: &[f64], grad_delta: &[f64]) {
        let sy = dot(step_delta, grad_delta);
        if !sy.is_finite() || sy <= 1e-12 {
            return;
        }
        if self.s_history.len() == self.history_size {
            self.s_history.remove(0);
            self.y_history.remove(0);
            self.rho_history.remove(0);
        }
        self.s_history.push(step_delta.to_vec());
        self.y_history.push(grad_delta.to_vec());
        self.rho_history.push(1.0 / sy);
    }
}

#[cfg(feature = "pino-ndarray-cpu")]
impl<B: burn::tensor::backend::Backend<FloatElem = f32>> BurnFieldHead<B> {
    fn adaptive_tanh(tensor: Tensor<B, 2>, activation: Tensor<B, 1>) -> Tensor<B, 2> {
        let slope = activation.clamp(0.05, 2.0) * (PINO_LAAF_GAIN as f32);
        (tensor * slope.unsqueeze()).tanh()
    }

    fn soft_clip(tensor: Tensor<B, 2>, limit: f32) -> Tensor<B, 2> {
        (tensor.clone() / limit).tanh() * limit
    }

    fn new_from_state(
        device: &B::Device,
        input_dim: usize,
        output_dim: usize,
        hidden_layers: usize,
        hidden_width: usize,
        weights: &[f64],
        bias: &[f64],
        activation: &[f64],
    ) -> Self {
        let hidden_layers = hidden_layers.clamp(1, 6);
        let hidden_width = hidden_width.clamp(8, 256);
        let mut weight_offset = 0usize;
        let mut bias_offset = 0usize;
        let mut activation_offset = 0usize;
        let take_f32 = |values: &[f64], start: usize, len: usize| -> Vec<f32> {
            values
                .iter()
                .skip(start)
                .take(len)
                .map(|value| *value as f32)
                .chain(std::iter::repeat_n(0.0, len))
                .take(len)
                .collect::<Vec<_>>()
        };
        let lift_weight_values = take_f32(weights, weight_offset, input_dim * hidden_width);
        weight_offset += input_dim * hidden_width;
        let lift_bias_values = take_f32(bias, bias_offset, hidden_width);
        bias_offset += hidden_width;
        let lift_activation_values = take_f32(activation, activation_offset, 1);
        activation_offset += 1;

        let mut operator_self_weights = Vec::with_capacity(hidden_layers);
        let mut operator_local_weights = Vec::with_capacity(hidden_layers);
        let mut operator_global_weights = Vec::with_capacity(hidden_layers);
        let mut operator_contrast_weights = Vec::with_capacity(hidden_layers);
        let mut operator_biases = Vec::with_capacity(hidden_layers);
        let mut operator_activation_scales = Vec::with_capacity(hidden_layers);
        for _ in 0..hidden_layers {
            operator_self_weights.push(Param::from_data(
                Data::new(
                    take_f32(weights, weight_offset, hidden_width * hidden_width),
                    [hidden_width, hidden_width].into(),
                ),
                device,
            ));
            weight_offset += hidden_width * hidden_width;
            operator_local_weights.push(Param::from_data(
                Data::new(
                    take_f32(weights, weight_offset, hidden_width * hidden_width),
                    [hidden_width, hidden_width].into(),
                ),
                device,
            ));
            weight_offset += hidden_width * hidden_width;
            operator_global_weights.push(Param::from_data(
                Data::new(
                    take_f32(weights, weight_offset, hidden_width * hidden_width),
                    [hidden_width, hidden_width].into(),
                ),
                device,
            ));
            weight_offset += hidden_width * hidden_width;
            operator_contrast_weights.push(Param::from_data(
                Data::new(
                    take_f32(weights, weight_offset, hidden_width * hidden_width),
                    [hidden_width, hidden_width].into(),
                ),
                device,
            ));
            weight_offset += hidden_width * hidden_width;
            operator_biases.push(Param::from_data(
                Data::new(take_f32(bias, bias_offset, hidden_width), [hidden_width].into()),
                device,
            ));
            bias_offset += hidden_width;
            operator_activation_scales.push(Param::from_data(
                Data::new(take_f32(activation, activation_offset, 1), [1].into()),
                device,
            ));
            activation_offset += 1;
        }
        let output_weight_values = take_f32(weights, weight_offset, hidden_width * output_dim);
        let output_bias_values = take_f32(bias, bias_offset, output_dim);
        Self {
            lift_weight: Param::from_data(
                Data::new(lift_weight_values, [hidden_width, input_dim].into()),
                device,
            ),
            lift_bias: Param::from_data(Data::new(lift_bias_values, [hidden_width].into()), device),
            lift_activation_scale: Param::from_data(
                Data::new(lift_activation_values, [1].into()),
                device,
            ),
            operator_self_weights,
            operator_local_weights,
            operator_global_weights,
            operator_contrast_weights,
            operator_biases,
            operator_activation_scales,
            output_weight: Param::from_data(
                Data::new(output_weight_values, [output_dim, hidden_width].into()),
                device,
            ),
            output_bias: Param::from_data(Data::new(output_bias_values, [output_dim].into()), device),
            hidden_layers,
            hidden_width,
        }
    }

    fn encode(
        &self,
        input: Tensor<B, 2>,
        neighbor_average: Tensor<B, 2>,
        spectral_projection: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let mut current = Self::adaptive_tanh(
            input.matmul(self.lift_weight.val().swap_dims(0, 1)) + self.lift_bias.val().unsqueeze(),
            self.lift_activation_scale.val(),
        );
        let lifted = current.clone();
        for layer in 0..self.hidden_layers {
            let local = neighbor_average.clone().matmul(current.clone());
            let contrast = current.clone() - local.clone();
            let spectral = spectral_projection.clone().matmul(current.clone());
            let update = current
                .clone()
                .matmul(self.operator_self_weights[layer].val().swap_dims(0, 1))
                + local.matmul(self.operator_local_weights[layer].val().swap_dims(0, 1))
                + spectral.matmul(self.operator_global_weights[layer].val().swap_dims(0, 1))
                + contrast.matmul(self.operator_contrast_weights[layer].val().swap_dims(0, 1))
                + self.operator_biases[layer].val().unsqueeze();
            current = Self::adaptive_tanh(
                update + current * 0.35 + lifted.clone() * 0.18,
                self.operator_activation_scales[layer].val(),
            );
        }
        current
    }

    fn forward(
        &self,
        input: Tensor<B, 2>,
        neighbor_average: Tensor<B, 2>,
        spectral_projection: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let current = self.encode(input, neighbor_average, spectral_projection);
        Self::soft_clip(
            current.matmul(self.output_weight.val().swap_dims(0, 1))
                + self.output_bias.val().unsqueeze(),
            2.5,
        )
    }
}

#[cfg(feature = "pino-ndarray-cpu")]
fn derivative_axis_matrix(size: usize, spacing: f64) -> Vec<f32> {
    let mut matrix = vec![0.0f32; size * size];
    if size <= 1 {
        return matrix;
    }
    let h = spacing.max(1e-6) as f32;
    for col in 0..size {
        if col == 0 {
            matrix[col] = -1.0 / h;
            matrix[size + col] = 1.0 / h;
        } else if col + 1 == size {
            matrix[(size - 2) * size + col] = -1.0 / h;
            matrix[(size - 1) * size + col] = 1.0 / h;
        } else {
            matrix[(col - 1) * size + col] = -0.5 / h;
            matrix[(col + 1) * size + col] = 0.5 / h;
        }
    }
    matrix
}

#[cfg(feature = "pino-ndarray-cpu")]
fn derivative_operator_matrix(nx: usize, ny: usize, nz: usize, axis: usize, spacing: f64) -> Vec<f32> {
    let cells = nx.saturating_mul(ny).saturating_mul(nz);
    let mut matrix = vec![0.0f32; cells * cells];
    if cells == 0 {
        return matrix;
    }
    let axis_size = match axis {
        0 => nx,
        1 => ny,
        _ => nz,
    };
    let axis_matrix = derivative_axis_matrix(axis_size, spacing);
    let plane = nx * ny;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let cell = z * plane + y * nx + x;
                match axis {
                    0 => {
                        for src_x in 0..nx {
                            let source = z * plane + y * nx + src_x;
                            matrix[cell * cells + source] = axis_matrix[x * nx + src_x];
                        }
                    }
                    1 => {
                        for src_y in 0..ny {
                            let source = z * plane + src_y * nx + x;
                            matrix[cell * cells + source] = axis_matrix[y * ny + src_y];
                        }
                    }
                    _ => {
                        for src_z in 0..nz {
                            let source = src_z * plane + y * nx + x;
                            matrix[cell * cells + source] = axis_matrix[z * nz + src_z];
                        }
                    }
                }
            }
        }
    }
    matrix
}

#[cfg(feature = "pino-ndarray-cpu")]
fn neighbor_average_matrix(nx: usize, ny: usize, nz: usize) -> Vec<f32> {
    let cells = nx.saturating_mul(ny).saturating_mul(nz);
    let mut matrix = vec![0.0f32; cells * cells];
    if cells == 0 {
        return matrix;
    }
    let plane = nx * ny;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let cell = z * plane + y * nx + x;
                let mut neighbors = Vec::with_capacity(6);
                if x > 0 {
                    neighbors.push(cell - 1);
                }
                if x + 1 < nx {
                    neighbors.push(cell + 1);
                }
                if y > 0 {
                    neighbors.push(cell - nx);
                }
                if y + 1 < ny {
                    neighbors.push(cell + nx);
                }
                if z > 0 {
                    neighbors.push(cell - plane);
                }
                if z + 1 < nz {
                    neighbors.push(cell + plane);
                }
                if neighbors.is_empty() {
                    neighbors.push(cell);
                }
                let scale = 1.0f32 / (neighbors.len() as f32);
                for neighbor in neighbors {
                    matrix[cell * cells + neighbor] = scale;
                }
            }
        }
    }
    matrix
}

#[cfg(feature = "pino-ndarray-cpu")]
fn spectral_projection_matrix(nx: usize, ny: usize, nz: usize, spectral_modes: usize) -> Vec<f32> {
    let cells = nx.saturating_mul(ny).saturating_mul(nz);
    let mut matrix = vec![0.0f32; cells * cells];
    if cells == 0 || nx == 0 || ny == 0 || nz == 0 {
        return matrix;
    }
    let mode_cap = spectral_modes.max(1).min(nx.min(ny).min(nz).max(1));
    let mut basis: Vec<Vec<f32>> = Vec::new();
    let plane = nx * ny;
    for kz in 0..mode_cap {
        for ky in 0..mode_cap {
            for kx in 0..mode_cap {
                let mut phi = vec![0.0f32; cells];
                let mut norm = 0.0f64;
                for z in 0..nz {
                    for y in 0..ny {
                        for x in 0..nx {
                            let idx = z * plane + y * nx + x;
                            let x_term = (std::f64::consts::PI * (kx as f64) * ((x as f64) + 0.5)
                                / (nx as f64))
                                .cos();
                            let y_term = (std::f64::consts::PI * (ky as f64) * ((y as f64) + 0.5)
                                / (ny as f64))
                                .cos();
                            let z_term = (std::f64::consts::PI * (kz as f64) * ((z as f64) + 0.5)
                                / (nz as f64))
                                .cos();
                            let value = (x_term * y_term * z_term) as f32;
                            phi[idx] = value;
                            norm += (value as f64) * (value as f64);
                        }
                    }
                }
                if norm > 1e-12 {
                    let inv_norm = norm.sqrt().recip() as f32;
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
            let row_offset = row * cells;
            for col in 0..cells {
                matrix[row_offset + col] += phi[row] * phi[col];
            }
        }
    }
    matrix
}

#[cfg(feature = "pino-ndarray-cpu")]
fn tensor_scalar(value: f32, device: &<HeadBackend as burn::tensor::backend::Backend>::Device) -> Tensor<HeadBackend, 1> {
    Tensor::<HeadBackend, 1>::from_data(Data::new(vec![value], [1].into()), device)
}

#[cfg(feature = "pino-ndarray-cpu")]
fn robust_mean_square(tensor: Tensor<HeadBackend, 2>) -> Tensor<HeadBackend, 1> {
    let squared = tensor.clone() * tensor;
    ((squared + 1.0).sqrt() - 1.0).mean()
}

#[cfg(feature = "pino-ndarray-cpu")]
fn tensor_to_scalar(tensor: Tensor<HeadBackend, 1>) -> Option<f64> {
    tensor.into_data().convert::<f64>().value.first().copied()
}

#[cfg(feature = "pino-ndarray-cpu")]
fn smooth_normalized_tail(loss: Tensor<HeadBackend, 1>, scale: f32) -> Tensor<HeadBackend, 1> {
    let normalized = loss / scale.max(1.0e-6);
    ((normalized.clone() * normalized) + 1.0).sqrt() - 1.0
}

#[cfg(feature = "pino-ndarray-cpu")]
fn prepare_physics_samples(samples: &[BurnPhysicsSample]) -> Option<Vec<PhysicsSampleTensors>> {
    if samples.is_empty() {
        return None;
    }
    let device = Default::default();
    let mut prepared = Vec::with_capacity(samples.len());
    let mut grid_tensor_cache: HashMap<
        (usize, usize, usize, usize, u64, u64, u64),
        (
            Tensor<HeadBackend, 2>,
            Tensor<HeadBackend, 2>,
            Tensor<HeadBackend, 2>,
            Tensor<HeadBackend, 2>,
            Tensor<HeadBackend, 2>,
        ),
    > = HashMap::new();
    for sample in samples {
        let output_dim = sample.output_dim.clamp(1, PINO_OUTPUT_CHANNELS);
        let cells = sample
            .grid_nx
            .checked_mul(sample.grid_ny)?
            .checked_mul(sample.grid_nz)?;
        if cells == 0
            || sample.features.is_empty()
            || sample.features.len() % cells != 0
            || sample.target_fields.len() != cells * PINO_OUTPUT_CHANNELS
            || sample.base_fields.len() != cells * PINO_OUTPUT_CHANNELS
            || sample.correction_scales.len() != cells * PINO_OUTPUT_CHANNELS
            || sample.observable_target.len() != PINO_OBSERVABLE_COUNT
            || sample.observable_scale.len() != PINO_OBSERVABLE_COUNT
            || sample.observable_weight.len() != PINO_OBSERVABLE_COUNT
            || sample.observable_projection.len() != cells * PINO_OBSERVABLE_COUNT
            || sample.benchmark_vm_active.len() != cells
            || sample.benchmark_sxx_peak.len() != cells
            || sample.stress_focus.len() != cells
            || sample.ring_loss_mask.len() != cells
            || sample.mask.len() != cells
            || sample.clamp.len() != cells
            || sample.displacement_embed.len() != cells
            || sample.traction_mask.len() != cells
            || sample.traction_normal.len() != cells * 3
            || sample.traction_target.len() != cells * 3
        {
            return None;
        }
        let feature_dim = sample.features.len() / cells;
        let features = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample.features.iter().map(|value| *value as f32).collect::<Vec<_>>(),
                [cells, feature_dim].into(),
            ),
            &device,
        );
        let target_fields = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample
                    .target_fields
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>(),
                [cells, PINO_OUTPUT_CHANNELS].into(),
            ),
            &device,
        );
        let mut model_base_fields = Vec::with_capacity(cells * output_dim);
        let mut model_correction_scales = Vec::with_capacity(cells * output_dim);
        for idx in 0..cells {
            let start = idx * PINO_OUTPUT_CHANNELS;
            let end = start + output_dim;
            model_base_fields.extend(
                sample.base_fields[start..end]
                    .iter()
                    .map(|value| *value as f32),
            );
            model_correction_scales.extend(
                sample.correction_scales[start..end]
                    .iter()
                    .map(|value| *value as f32),
            );
        }
        let base_fields = Tensor::<HeadBackend, 2>::from_data(
            Data::new(model_base_fields, [cells, output_dim].into()),
            &device,
        );
        let correction_scales = Tensor::<HeadBackend, 2>::from_data(
            Data::new(model_correction_scales, [cells, output_dim].into()),
            &device,
        );
        let cache_key = (
            sample.grid_nx,
            sample.grid_ny,
            sample.grid_nz,
            sample.spectral_modes,
            sample.dx.to_bits(),
            sample.dy.to_bits(),
            sample.dz.to_bits(),
        );
        let (
            neighbor_average,
            spectral_projection,
            diff_x,
            diff_y,
            diff_z,
        ) = {
            let cached = grid_tensor_cache.entry(cache_key).or_insert_with(|| {
                (
                    Tensor::<HeadBackend, 2>::from_data(
                        Data::new(
                            neighbor_average_matrix(sample.grid_nx, sample.grid_ny, sample.grid_nz),
                            [cells, cells].into(),
                        ),
                        &device,
                    ),
                    Tensor::<HeadBackend, 2>::from_data(
                        Data::new(
                            spectral_projection_matrix(
                                sample.grid_nx,
                                sample.grid_ny,
                                sample.grid_nz,
                                sample.spectral_modes,
                            ),
                            [cells, cells].into(),
                        ),
                        &device,
                    ),
                    Tensor::<HeadBackend, 2>::from_data(
                        Data::new(
                            derivative_operator_matrix(
                                sample.grid_nx,
                                sample.grid_ny,
                                sample.grid_nz,
                                0,
                                sample.dx,
                            ),
                            [cells, cells].into(),
                        ),
                        &device,
                    ),
                    Tensor::<HeadBackend, 2>::from_data(
                        Data::new(
                            derivative_operator_matrix(
                                sample.grid_nx,
                                sample.grid_ny,
                                sample.grid_nz,
                                1,
                                sample.dy,
                            ),
                            [cells, cells].into(),
                        ),
                        &device,
                    ),
                    Tensor::<HeadBackend, 2>::from_data(
                        Data::new(
                            derivative_operator_matrix(
                                sample.grid_nx,
                                sample.grid_ny,
                                sample.grid_nz,
                                2,
                                sample.dz,
                            ),
                            [cells, cells].into(),
                        ),
                        &device,
                    ),
                )
            });
            (
                cached.0.clone(),
                cached.1.clone(),
                cached.2.clone(),
                cached.3.clone(),
                cached.4.clone(),
            )
        };
        let mask_cells = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample.mask.iter().map(|value| *value as f32).collect::<Vec<_>>(),
                [cells, 1].into(),
            ),
            &device,
        );
        let clamp_cells = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample.clamp.iter().map(|value| *value as f32).collect::<Vec<_>>(),
                [cells, 1].into(),
            ),
            &device,
        );
        let traction_cells = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample
                    .traction_mask
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>(),
                [cells, 1].into(),
            ),
            &device,
        );
        let traction_normals = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample
                    .traction_normal
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>(),
                [cells, 3].into(),
            ),
            &device,
        );
        let traction_targets = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample
                    .traction_target
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>(),
                [cells, 3].into(),
            ),
            &device,
        );
        let primary_target_fields = target_fields.clone().slice([0..cells, 0..PINO_PRIMARY_OUTPUT_CHANNELS]);
        let auxiliary_target_fields = target_fields
            .clone()
            .slice([0..cells, PINO_PRIMARY_OUTPUT_CHANNELS..PINO_OUTPUT_CHANNELS]);
        let auxiliary_base_fields = if output_dim > PINO_PRIMARY_OUTPUT_CHANNELS {
            base_fields
                .clone()
                .slice([0..cells, PINO_PRIMARY_OUTPUT_CHANNELS..output_dim])
        } else {
            Tensor::<HeadBackend, 2>::zeros([cells, PINO_AUX_OUTPUT_CHANNELS], &device)
        };
        let auxiliary_scale_fields = if output_dim > PINO_PRIMARY_OUTPUT_CHANNELS {
            correction_scales
                .clone()
                .slice([0..cells, PINO_PRIMARY_OUTPUT_CHANNELS..output_dim])
        } else {
            Tensor::<HeadBackend, 2>::ones([cells, PINO_AUX_OUTPUT_CHANNELS], &device)
        };
        let primary_mask_fields = mask_cells
            .clone()
            .matmul(Tensor::<HeadBackend, 2>::ones([1, PINO_PRIMARY_OUTPUT_CHANNELS], &device));
        let auxiliary_mask_fields = mask_cells
            .clone()
            .matmul(Tensor::<HeadBackend, 2>::ones([1, PINO_AUX_OUTPUT_CHANNELS], &device));
        let traction_mask_fields = traction_cells
            .clone()
            .matmul(Tensor::<HeadBackend, 2>::ones([1, 3], &device));
        let stress_focus_cells = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample
                    .stress_focus
                    .iter()
                    .map(|value| value.max(1.0) as f32)
                    .collect::<Vec<_>>(),
                [cells, 1].into(),
            ),
            &device,
        );
        let ring_loss_mask = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample
                    .ring_loss_mask
                    .iter()
                    .map(|value| value.max(0.0) as f32)
                    .collect::<Vec<_>>(),
                [cells, 1].into(),
            ),
            &device,
        );
        let stress_focus_primary_fields = stress_focus_cells
            .clone()
            .matmul(Tensor::<HeadBackend, 2>::ones([1, PINO_PRIMARY_OUTPUT_CHANNELS - 3], &device));
        let stress_focus_auxiliary_fields = stress_focus_cells
            .clone()
            .matmul(Tensor::<HeadBackend, 2>::ones([1, PINO_AUX_OUTPUT_CHANNELS], &device));
        let observable_target = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample
                    .observable_target
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>(),
                [1, PINO_OBSERVABLE_COUNT].into(),
            ),
            &device,
        );
        let observable_scale = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample
                    .observable_scale
                    .iter()
                    .map(|value| (*value).max(1e-6) as f32)
                    .collect::<Vec<_>>(),
                [1, PINO_OBSERVABLE_COUNT].into(),
            ),
            &device,
        );
        let observable_weight = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample
                    .observable_weight
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>(),
                [1, PINO_OBSERVABLE_COUNT].into(),
            ),
            &device,
        );
        let observable_projection = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample
                    .observable_projection
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>(),
                [PINO_OBSERVABLE_COUNT, cells].into(),
            ),
            &device,
        );
        let benchmark_vm_active_cells = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample
                    .benchmark_vm_active
                    .iter()
                    .map(|value| value.max(0.0) as f32)
                    .collect::<Vec<_>>(),
                [cells, 1].into(),
            ),
            &device,
        );
        let benchmark_sxx_peak_cells = Tensor::<HeadBackend, 2>::from_data(
            Data::new(
                sample
                    .benchmark_sxx_peak
                    .iter()
                    .map(|value| value.max(0.0) as f32)
                    .collect::<Vec<_>>(),
                [cells, 1].into(),
            ),
            &device,
        );
        let mut disp_scale = 1.0e-6_f64;
        let mut stress_scale = 1.0_f64;
        let mut channel_scale = [1.0_f64; PINO_OUTPUT_CHANNELS];
        for channel in 0..PINO_OUTPUT_CHANNELS {
            let mut mean_abs = 0.0;
            let mut corr_abs = 0.0;
            for idx in 0..cells {
                mean_abs += sample.target_fields[idx * PINO_OUTPUT_CHANNELS + channel].abs();
                corr_abs += sample.correction_scales[idx * PINO_OUTPUT_CHANNELS + channel].abs();
            }
            mean_abs /= cells as f64;
            corr_abs /= cells as f64;
            let representable_floor = (corr_abs * 0.35).max(1e-6);
            channel_scale[channel] = mean_abs.max(representable_floor);
            if channel < 3 {
                disp_scale = disp_scale.max(channel_scale[channel]);
            } else {
                stress_scale = stress_scale.max(channel_scale[channel]);
            }
        }
        let clamp_count = sample
            .clamp
            .iter()
            .copied()
            .filter(|value| *value > 0.5)
            .count();
        let traction_count = sample
            .traction_mask
            .iter()
            .copied()
            .filter(|value| *value > 0.5)
            .count();
        let boundary_scale = if clamp_count + traction_count == 0 {
            1.0
        } else {
            ((cells as f64) / ((clamp_count + traction_count) as f64))
                .sqrt()
                .clamp(1.0, 8.0)
        };
        let equilibrium_scale =
            (stress_scale / sample.dx.max(sample.dy).max(sample.dz).max(1e-6)).max(1.0);
        let coarse_primary_target = (neighbor_average.clone().matmul(primary_target_fields.clone())
            + spectral_projection.clone().matmul(primary_target_fields.clone()))
            * 0.5;
        let coarse_auxiliary_target =
            (neighbor_average.clone().matmul(auxiliary_target_fields.clone())
                + spectral_projection
                    .clone()
                    .matmul(auxiliary_target_fields.clone()))
                * 0.5;
        prepared.push(PhysicsSampleTensors {
            features,
            base_fields,
            correction_scales,
            output_dim,
            primary_target_fields,
            auxiliary_target_fields,
            auxiliary_base_fields,
            auxiliary_scale_fields,
            neighbor_average,
            spectral_projection,
            mask_cells,
            clamp_cells,
            traction_normals,
            traction_targets,
            primary_mask_fields,
            auxiliary_mask_fields,
            traction_mask_fields,
            observable_target,
            observable_scale,
            observable_weight,
            observable_projection,
            observable_fifth_uses_vm: sample.observable_fifth_uses_vm,
            benchmark_vm_active_cells,
            benchmark_sxx_peak_cells,
            stress_focus_cells,
            stress_focus_primary_fields,
            stress_focus_auxiliary_fields,
            ring_loss_mask,
            coarse_primary_target,
            coarse_auxiliary_target,
            diff_x,
            diff_y,
            diff_z,
            grid_nx: sample.grid_nx,
            grid_ny: sample.grid_ny,
            grid_nz: sample.grid_nz,
            e_modulus: sample.e_modulus.max(1.0) as f32,
            poisson: sample.poisson.clamp(0.0, 0.499) as f32,
            disp_scale: disp_scale as f32,
            stress_scale: stress_scale as f32,
            equilibrium_scale: equilibrium_scale as f32,
            primary_channel_scale: Tensor::<HeadBackend, 2>::from_data(
                Data::new(
                    vec![
                        channel_scale[0] as f32,
                        channel_scale[1] as f32,
                        channel_scale[2] as f32,
                        channel_scale[3] as f32,
                        channel_scale[4] as f32,
                        channel_scale[5] as f32,
                        channel_scale[6] as f32,
                        channel_scale[7] as f32,
                        channel_scale[8] as f32,
                    ],
                    [1, PINO_PRIMARY_OUTPUT_CHANNELS].into(),
                ),
                &device,
            ),
            auxiliary_channel_scale: Tensor::<HeadBackend, 2>::from_data(
                Data::new(
                    vec![channel_scale[9] as f32, channel_scale[10] as f32],
                    [1, PINO_AUX_OUTPUT_CHANNELS].into(),
                ),
                &device,
            ),
            boundary_scale: boundary_scale as f32,
        });
    }
    Some(prepared)
}

#[cfg(feature = "pino-ndarray-cpu")]
fn apply_characteristic_training_scales(
    prepared: &mut [PhysicsSampleTensors],
    samples: &[BurnPhysicsSample],
) {
    for (prepared_sample, raw_sample) in prepared.iter_mut().zip(samples.iter()) {
        let characteristic_disp = raw_sample.characteristic_disp_scale.max(0.0);
        let characteristic_stress = raw_sample.characteristic_stress_scale.max(0.0);
        if characteristic_disp <= 0.0 && characteristic_stress <= 0.0 {
            continue;
        }

        let device = prepared_sample.primary_channel_scale.device();
        let mut primary_scale = prepared_sample
            .primary_channel_scale
            .clone()
            .into_data()
            .value;
        let mut observable_scale = prepared_sample
            .observable_scale
            .clone()
            .into_data()
            .value;

        if characteristic_disp > 0.0 {
            for value in primary_scale.iter_mut().take(3) {
                *value = value.max(characteristic_disp as f32);
            }
            for value in observable_scale.iter_mut().take(2) {
                *value = value.max(characteristic_disp as f32);
            }
            prepared_sample.disp_scale =
                (prepared_sample.disp_scale as f64).max(characteristic_disp) as f32;
        }

        if characteristic_stress > 0.0 {
            for value in primary_scale.iter_mut().skip(3) {
                *value = value.max(characteristic_stress as f32);
            }
            for value in observable_scale.iter_mut().skip(2) {
                *value = value.max(characteristic_stress as f32);
            }
            let mut auxiliary_scale = prepared_sample
                .auxiliary_channel_scale
                .clone()
                .into_data()
                .value;
            for value in auxiliary_scale.iter_mut() {
                *value = value.max(characteristic_stress as f32);
            }
            prepared_sample.auxiliary_channel_scale = Tensor::<HeadBackend, 2>::from_data(
                Data::new(auxiliary_scale, [1, PINO_AUX_OUTPUT_CHANNELS].into()),
                &device,
            );
            prepared_sample.stress_scale =
                (prepared_sample.stress_scale as f64).max(characteristic_stress) as f32;
            let characteristic_equilibrium = characteristic_stress
                / raw_sample
                    .dx
                    .max(raw_sample.dy)
                    .max(raw_sample.dz)
                    .max(1e-6);
            prepared_sample.equilibrium_scale = (prepared_sample.equilibrium_scale as f64)
                .max(characteristic_equilibrium)
                .max(1.0) as f32;
        }

        prepared_sample.primary_channel_scale = Tensor::<HeadBackend, 2>::from_data(
            Data::new(primary_scale, [1, PINO_PRIMARY_OUTPUT_CHANNELS].into()),
            &device,
        );
        prepared_sample.observable_scale = Tensor::<HeadBackend, 2>::from_data(
            Data::new(observable_scale, [1, PINO_OBSERVABLE_COUNT].into()),
            &device,
        );
    }
}

#[cfg(feature = "pino-ndarray-cpu")]
fn prepare_physics_samples_for_objective(
    samples: &[BurnPhysicsSample],
    exact_surface: bool,
    characteristic_train_scaling: bool,
) -> Option<Vec<PhysicsSampleTensors>> {
    let mut prepared = prepare_physics_samples(samples)?;
    if characteristic_train_scaling || benchmark_cantilever_characteristic_training(exact_surface) {
        apply_characteristic_training_scales(&mut prepared, samples);
    }
    Some(prepared)
}

#[cfg(feature = "pino-ndarray-cpu")]
fn physics_loss(
    model: &BurnFieldHead<HeadBackend>,
    samples: &[PhysicsSampleTensors],
    loss_weights: (f64, f64, f64, f64),
    exact_surface: bool,
) -> Option<PhysicsLossTensors> {
    let device = samples.first()?.features.device();
    let isolated_exact_cantilever = isolated_exact_cantilever_surface(exact_surface);
    let benchmark_cantilever = BENCHMARK_CANTILEVER_OVERRIDE.load(Ordering::Relaxed)
        || matches!(
            std::env::var("PINO_HEADLESS_BENCHMARK_ID").ok().as_deref(),
            Some("benchmark_cantilever_2d")
        );
    let displacement_primary_exact = isolated_exact_cantilever && exact_surface;
    let mut data_acc = tensor_scalar(0.0, &device);
    let mut displacement_acc = tensor_scalar(0.0, &device);
    let mut stress_acc = tensor_scalar(0.0, &device);
    let mut observable_acc = tensor_scalar(0.0, &device);
    let mut auxiliary_acc = tensor_scalar(0.0, &device);
    let mut auxiliary_data_acc = tensor_scalar(0.0, &device);
    let mut invariant_acc = tensor_scalar(0.0, &device);
    let mut eq_acc = tensor_scalar(0.0, &device);
    let mut constitutive_acc = tensor_scalar(0.0, &device);
    let mut constitutive_normal_acc = tensor_scalar(0.0, &device);
    let mut constitutive_shear_acc = tensor_scalar(0.0, &device);
    let mut weak_energy_acc = tensor_scalar(0.0, &device);
    let mut boundary_acc = tensor_scalar(0.0, &device);
    let mut any_benchmark_displacement_primary = false;
    let auxiliary_weight = if exact_surface {
        0.0f32
    } else if loss_weights.0 <= 0.24 && loss_weights.2 <= 0.30 {
        0.03f32
    } else if loss_weights.0 <= 0.60 && loss_weights.2 <= 0.60 {
        0.06f32
    } else {
        0.10f32
    };
    let coarse_weight = if exact_surface {
        0.0f32
    } else if loss_weights.0 <= 0.24 && loss_weights.2 <= 0.30 {
        0.38f32
    } else if loss_weights.0 <= 0.60 && loss_weights.2 <= 0.60 {
        0.22f32
    } else {
        0.10f32
    };
    for sample in samples {
        let cells = sample.grid_nx * sample.grid_ny * sample.grid_nz;
        let displacement_only_head = sample.output_dim == PINO_DISPLACEMENT_OUTPUT_CHANNELS;
        let benchmark_displacement_primary = benchmark_cantilever && displacement_only_head;
        any_benchmark_displacement_primary |= benchmark_displacement_primary;
        let raw_predictions = model.forward(
            sample.features.clone(),
            sample.neighbor_average.clone(),
            sample.spectral_projection.clone(),
        );
        let predicted_fields =
            sample.base_fields.clone() + raw_predictions.clone() * sample.correction_scales.clone();
        let primary_predicted = if displacement_only_head {
            predicted_fields.clone()
        } else {
            predicted_fields
                .clone()
                .slice([0..cells, 0..PINO_PRIMARY_OUTPUT_CHANNELS])
        };
        let displacement_data_loss = robust_mean_square(
            ((primary_predicted.clone().slice([0..cells, 0..3])
                - sample.primary_target_fields.clone().slice([0..cells, 0..3]))
                / sample.primary_channel_scale.clone().slice([0..1, 0..3]))
                * sample.primary_mask_fields.clone().slice([0..cells, 0..3]),
        );

        let ux = predicted_fields.clone().slice([0..cells, 0..1]);
        let uy = predicted_fields.clone().slice([0..cells, 1..2]);
        let uz = predicted_fields.clone().slice([0..cells, 2..3]);
        let grad_x = sample.diff_x.clone().matmul(primary_predicted.clone());
        let grad_y = sample.diff_y.clone().matmul(primary_predicted.clone());
        let grad_z = sample.diff_z.clone().matmul(primary_predicted.clone());
        let dux_dx = grad_x.clone().slice([0..cells, 0..1]);
        let dux_dy = grad_y.clone().slice([0..cells, 0..1]);
        let dux_dz = grad_z.clone().slice([0..cells, 0..1]);
        let duy_dx = grad_x.clone().slice([0..cells, 1..2]);
        let duy_dy = grad_y.clone().slice([0..cells, 1..2]);
        let duy_dz = grad_z.clone().slice([0..cells, 1..2]);
        let duz_dx = grad_x.clone().slice([0..cells, 2..3]);
        let duz_dy = grad_y.clone().slice([0..cells, 2..3]);
        let duz_dz = grad_z.clone().slice([0..cells, 2..3]);

        let exx = dux_dx;
        let eyy = duy_dy;
        let ezz = duz_dz;
        let gxy = dux_dy.clone() + duy_dx.clone();
        let gxz = dux_dz.clone() + duz_dx.clone();
        let gyz = duy_dz.clone() + duz_dy.clone();
        let lambda = sample.e_modulus * sample.poisson
            / ((1.0 + sample.poisson) * (1.0 - 2.0 * sample.poisson)).max(1e-4);
        let shear = sample.e_modulus / (2.0 * (1.0 + sample.poisson)).max(1e-4);
        let plane_stress_2d = sample.grid_nz <= 1;
        let trace = exx.clone() + eyy.clone() + ezz.clone();
        let plane_stress_c = sample.e_modulus / (1.0 - sample.poisson * sample.poisson).max(1e-4);
        let zero_stress_column = || {
            Tensor::<HeadBackend, 2>::from_data(
                Data::new(vec![0.0; cells], [cells, 1].into()),
                &device,
            )
        };
        let expected_sxx = if plane_stress_2d {
            (exx.clone() + eyy.clone() * sample.poisson) * plane_stress_c
        } else {
            exx.clone() * (2.0 * shear) + trace.clone() * lambda
        };
        let expected_syy = if plane_stress_2d {
            (eyy.clone() + exx.clone() * sample.poisson) * plane_stress_c
        } else {
            eyy.clone() * (2.0 * shear) + trace.clone() * lambda
        };
        let expected_szz = if plane_stress_2d {
            zero_stress_column()
        } else {
            ezz.clone() * (2.0 * shear) + trace.clone() * lambda
        };
        let expected_sxy = gxy * shear;
        let expected_sxz = if plane_stress_2d {
            zero_stress_column()
        } else {
            gxz * shear
        };
        let expected_syz = if plane_stress_2d {
            zero_stress_column()
        } else {
            gyz * shear
        };
        let primary_data_loss = if displacement_only_head || displacement_primary_exact {
            displacement_data_loss.clone()
        } else {
            robust_mean_square(
                ((predicted_fields
                    .clone()
                    .slice([0..cells, 0..PINO_PRIMARY_OUTPUT_CHANNELS])
                    - sample.primary_target_fields.clone())
                    / sample.primary_channel_scale.clone())
                    * sample.primary_mask_fields.clone(),
            )
        };
        let derived_stress_predicted = Tensor::cat(
            vec![
                expected_sxx.clone(),
                expected_syy.clone(),
                expected_szz.clone(),
                expected_sxy.clone(),
                expected_sxz.clone(),
                expected_syz.clone(),
            ],
            1,
        );
        let legacy_stress_data_loss = robust_mean_square(
            (((if displacement_only_head || displacement_primary_exact {
                derived_stress_predicted.clone()
            } else {
                primary_predicted
                    .clone()
                    .slice([0..cells, 3..PINO_PRIMARY_OUTPUT_CHANNELS])
            })
                - sample
                    .primary_target_fields
                    .clone()
                    .slice([0..cells, 3..PINO_PRIMARY_OUTPUT_CHANNELS]))
                / sample
                    .primary_channel_scale
                    .clone()
                    .slice([0..1, 3..PINO_PRIMARY_OUTPUT_CHANNELS]))
                * sample
                    .primary_mask_fields
                    .clone()
                    .slice([0..cells, 3..PINO_PRIMARY_OUTPUT_CHANNELS])
                * sample.stress_focus_primary_fields.clone(),
        );
        let stress_fields_for_metric = if displacement_only_head {
            derived_stress_predicted.clone()
        } else {
            primary_predicted
                .clone()
                .slice([0..cells, 3..PINO_PRIMARY_OUTPUT_CHANNELS])
        };
        let stress_grad_x = sample.diff_x.clone().matmul(stress_fields_for_metric.clone());
        let stress_grad_y = sample.diff_y.clone().matmul(stress_fields_for_metric.clone());
        let stress_grad_z = sample.diff_z.clone().matmul(stress_fields_for_metric.clone());
        let metric_sxx = stress_fields_for_metric.clone().slice([0..cells, 0..1]);
        let metric_syy = stress_fields_for_metric.clone().slice([0..cells, 1..2]);
        let metric_szz = stress_fields_for_metric.clone().slice([0..cells, 2..3]);
        let metric_sxy = stress_fields_for_metric.clone().slice([0..cells, 3..4]);
        let metric_sxz = stress_fields_for_metric.clone().slice([0..cells, 4..5]);
        let metric_syz = stress_fields_for_metric.clone().slice([0..cells, 5..6]);
        let sxx = metric_sxx.clone();
        let syy = metric_syy.clone();
        let szz = metric_szz.clone();
        let sxy = metric_sxy.clone();
        let sxz = metric_sxz.clone();
        let syz = metric_syz.clone();
        let dsxx_dx = stress_grad_x.clone().slice([0..cells, 0..1]);
        let dsxy_dy = stress_grad_y.clone().slice([0..cells, 3..4]);
        let dsxz_dz = stress_grad_z.clone().slice([0..cells, 4..5]);
        let dsxy_dx = stress_grad_x.clone().slice([0..cells, 3..4]);
        let dsyy_dy = stress_grad_y.clone().slice([0..cells, 1..2]);
        let dsyz_dz = stress_grad_z.clone().slice([0..cells, 5..6]);
        let dsxz_dx = stress_grad_x.clone().slice([0..cells, 4..5]);
        let dsyz_dy = stress_grad_y.clone().slice([0..cells, 5..6]);
        let dszz_dz = stress_grad_z.clone().slice([0..cells, 2..3]);
        let equilibrium_norm = sample.equilibrium_scale;
        let equilibrium_x_loss = robust_mean_square(
            ((dsxx_dx + dsxy_dy + dsxz_dz) / equilibrium_norm) * sample.mask_cells.clone(),
        );
        let equilibrium_y_loss = robust_mean_square(
            ((dsxy_dx + dsyy_dy + dsyz_dz) / equilibrium_norm) * sample.mask_cells.clone(),
        );
        let equilibrium_z_loss = robust_mean_square(
            ((dsxz_dx + dsyz_dy + dszz_dz) / equilibrium_norm) * sample.mask_cells.clone(),
        );
        let equilibrium_loss = if plane_stress_2d {
            equilibrium_x_loss + equilibrium_y_loss
        } else {
            equilibrium_x_loss + equilibrium_y_loss + equilibrium_z_loss
        };
        let constitutive_normal_loss = robust_mean_square(
            ((sxx.clone() - expected_sxx.clone()) / sample.stress_scale)
                * sample.mask_cells.clone()
                * sample.stress_focus_cells.clone(),
        ) + robust_mean_square(
            ((syy.clone() - expected_syy.clone()) / sample.stress_scale)
                * sample.mask_cells.clone()
                * sample.stress_focus_cells.clone(),
        ) + robust_mean_square(
            ((szz.clone() - expected_szz.clone()) / sample.stress_scale)
                * sample.mask_cells.clone()
                * sample.stress_focus_cells.clone(),
        );
        let constitutive_shear_loss = robust_mean_square(
            ((sxy.clone() - expected_sxy.clone()) / sample.stress_scale)
                * sample.mask_cells.clone()
                * sample.stress_focus_cells.clone(),
        ) + robust_mean_square(
            ((sxz.clone() - expected_sxz.clone()) / sample.stress_scale)
                * sample.mask_cells.clone()
                * sample.stress_focus_cells.clone(),
        ) + robust_mean_square(
            ((syz.clone() - expected_syz.clone()) / sample.stress_scale)
                * sample.mask_cells.clone()
                * sample.stress_focus_cells.clone(),
        );
        let expected_energy_density = if plane_stress_2d {
            ((expected_sxx.clone() * exx.clone())
                + (expected_syy.clone() * eyy.clone())
                + (expected_sxy.clone() * (dux_dy.clone() + duy_dx.clone())))
                * 0.5
        } else {
            ((expected_sxx.clone() * exx.clone())
                + (expected_syy.clone() * eyy.clone())
                + (expected_szz.clone() * ezz.clone())
                + (expected_sxy.clone() * (dux_dy.clone() + duy_dx.clone()))
                + (expected_sxz.clone() * (dux_dz.clone() + duz_dx.clone()))
                + (expected_syz.clone() * (duy_dz.clone() + duz_dy.clone())))
                * 0.5
        };
        let predicted_energy_density = if plane_stress_2d {
            ((sxx.clone() * exx.clone())
                + (syy.clone() * eyy.clone())
                + (sxy.clone() * (dux_dy.clone() + duy_dx.clone())))
                * 0.5
        } else {
            ((sxx.clone() * exx.clone())
                + (syy.clone() * eyy.clone())
                + (szz.clone() * ezz.clone())
                + (sxy.clone() * (dux_dy.clone() + duy_dx.clone()))
                + (sxz.clone() * (dux_dz.clone() + duz_dx.clone()))
                + (syz.clone() * (duy_dz.clone() + duz_dy.clone())))
                * 0.5
        };
        let energy_scale = (sample.stress_scale * sample.disp_scale).max(1.0e-6);
        let weak_energy_loss = robust_mean_square(
            ((predicted_energy_density - expected_energy_density) / energy_scale)
                * sample.mask_cells.clone()
                * sample.stress_focus_cells.clone(),
        );

        let derived_principal = if plane_stress_2d {
            let mean_stress = (metric_sxx.clone() + metric_syy.clone()) * 0.5;
            let radius = ((((metric_sxx.clone() - metric_syy.clone()) * 0.5).powf_scalar(2.0)
                + metric_sxy.clone().powf_scalar(2.0))
                + 1.0e-9)
                .sqrt();
            mean_stress + radius
        } else {
            let mean_stress =
                (metric_sxx.clone() + metric_syy.clone() + metric_szz.clone()) / 3.0;
            let p2 = ((metric_sxx.clone() - mean_stress.clone()).powf_scalar(2.0)
                + (metric_syy.clone() - mean_stress.clone()).powf_scalar(2.0)
                + (metric_szz.clone() - mean_stress.clone()).powf_scalar(2.0)
                + (metric_sxy.clone().powf_scalar(2.0)
                    + metric_sxz.clone().powf_scalar(2.0)
                    + metric_syz.clone().powf_scalar(2.0))
                    * 2.0)
                / 6.0;
            mean_stress + (p2 * 2.0 + 1.0e-9).sqrt()
        };
        let derived_vm = if plane_stress_2d {
            ((metric_sxx.clone().powf_scalar(2.0)
                - metric_sxx.clone() * metric_syy.clone()
                + metric_syy.clone().powf_scalar(2.0)
                + metric_sxy.clone().powf_scalar(2.0) * 3.0
                + 1.0e-9))
                .sqrt()
        } else {
            ((((metric_sxx.clone() - metric_syy.clone()).powf_scalar(2.0)
                + (metric_syy.clone() - metric_szz.clone()).powf_scalar(2.0)
                + (metric_szz.clone() - metric_sxx.clone()).powf_scalar(2.0))
                * 0.5
                + (metric_sxy.clone().powf_scalar(2.0)
                    + metric_sxz.clone().powf_scalar(2.0)
                    + metric_syz.clone().powf_scalar(2.0))
                    * 3.0
                + 1.0e-9))
                .sqrt()
        };
        let derived_auxiliary = Tensor::cat(
            vec![
                derived_vm.clone().reshape([cells, 1]),
                derived_principal.clone().reshape([cells, 1]),
            ],
            1,
        );
        let ring_band_loss = robust_mean_square(
            (derived_auxiliary.clone().slice([0..cells, 0..1])
                - sample.auxiliary_target_fields.clone().slice([0..cells, 0..1]))
                * sample.ring_loss_mask.clone()
                * sample.mask_cells.clone(),
        );
        let observable_predicted = if benchmark_displacement_primary {
            let disp_mag =
                ((ux.clone().powf_scalar(2.0)
                    + uy.clone().powf_scalar(2.0)
                    + uz.clone().powf_scalar(2.0))
                    + 1.0e-9)
                    .sqrt();
            Tensor::cat(
                vec![
                    sample
                        .observable_projection
                        .clone()
                        .slice([0..1, 0..cells])
                        .matmul(predicted_fields.clone().slice([0..cells, 1..2])),
                    sample
                        .observable_projection
                        .clone()
                        .slice([1..2, 0..cells])
                        .matmul(disp_mag),
                    sample
                        .observable_projection
                        .clone()
                        .slice([2..3, 0..cells])
                        .matmul(derived_auxiliary.clone().slice([0..cells, 0..1])),
                    sample
                        .observable_projection
                        .clone()
                        .slice([3..4, 0..cells])
                        .matmul(derived_auxiliary.clone().slice([0..cells, 1..2])),
                    sample
                        .observable_projection
                        .clone()
                        .slice([4..5, 0..cells])
                        .matmul(metric_sxx.clone().abs()),
                ],
                1,
            )
        } else {
            Tensor::cat(
                vec![
                    sample
                        .observable_projection
                        .clone()
                        .slice([0..1, 0..cells])
                        .matmul(predicted_fields.clone().slice([0..cells, 1..2])),
                    sample
                        .observable_projection
                        .clone()
                        .slice([1..2, 0..cells])
                        .matmul(predicted_fields.clone().slice([0..cells, 0..1])),
                    sample
                        .observable_projection
                        .clone()
                        .slice([2..3, 0..cells])
                        .matmul(derived_auxiliary.clone().slice([0..cells, 0..1])),
                    sample
                        .observable_projection
                        .clone()
                        .slice([3..4, 0..cells])
                        .matmul(derived_auxiliary.clone().slice([0..cells, 1..2])),
                    if sample.observable_fifth_uses_vm {
                        sample
                            .observable_projection
                            .clone()
                            .slice([4..5, 0..cells])
                            .matmul(derived_auxiliary.clone().slice([0..cells, 0..1]))
                    } else {
                        sample
                            .observable_projection
                            .clone()
                            .slice([4..5, 0..cells])
                            .matmul(metric_sxx.clone().abs())
                    },
                ],
                1,
            )
        };
        let observable_loss = robust_mean_square(
            ((observable_predicted - sample.observable_target.clone())
                / sample.observable_scale.clone())
                * sample.observable_weight.clone(),
        );
        let auxiliary_data_loss = robust_mean_square(
            ((derived_auxiliary.clone() - sample.auxiliary_target_fields.clone())
                / sample.auxiliary_channel_scale.clone())
                * sample.auxiliary_mask_fields.clone()
                * sample.stress_focus_auxiliary_fields.clone(),
        );
        let benchmark_sigma_xx_loss = robust_mean_square(
            ((expected_sxx.clone()
                - sample
                    .primary_target_fields
                    .clone()
                    .slice([0..cells, 3..4]))
                / sample.primary_channel_scale.clone().slice([0..1, 3..4]))
                * sample.benchmark_sxx_peak_cells.clone(),
        );
        let benchmark_vm_loss = robust_mean_square(
            ((derived_auxiliary.clone().slice([0..cells, 0..1])
                - sample.auxiliary_target_fields.clone().slice([0..cells, 0..1]))
                / sample.auxiliary_channel_scale.clone().slice([0..1, 0..1]))
                * sample.benchmark_vm_active_cells.clone(),
        );
        let auxiliary_consistency_loss = if displacement_only_head {
            tensor_scalar(0.0, &device)
        } else {
            let raw_auxiliary_predictions = raw_predictions
                .clone()
                .slice([0..cells, PINO_PRIMARY_OUTPUT_CHANNELS..PINO_OUTPUT_CHANNELS]);
            let auxiliary_consistency_target =
                (derived_auxiliary.clone() - sample.auxiliary_base_fields.clone())
                    / (sample.auxiliary_scale_fields.clone() + 1.0e-6);
            robust_mean_square(
                (raw_auxiliary_predictions - auxiliary_consistency_target)
                    * sample.auxiliary_mask_fields.clone()
                    * sample.stress_focus_auxiliary_fields.clone(),
            )
        };

        let coarse_primary_pred = (sample.neighbor_average.clone().matmul(primary_predicted.clone())
            + sample.spectral_projection.clone().matmul(primary_predicted.clone()))
            * 0.5;
        let coarse_aux_pred = (sample.neighbor_average.clone().matmul(derived_auxiliary.clone())
            + sample.spectral_projection.clone().matmul(derived_auxiliary.clone()))
            * 0.5;
        let coarse_aux_loss = robust_mean_square(
            ((coarse_aux_pred - sample.coarse_auxiliary_target.clone())
                / sample.auxiliary_channel_scale.clone())
                * sample.auxiliary_mask_fields.clone()
                * sample.stress_focus_auxiliary_fields.clone(),
        );
        let coarse_displacement_loss = robust_mean_square(
            ((coarse_primary_pred.clone().slice([0..cells, 0..3])
                - sample.coarse_primary_target.clone().slice([0..cells, 0..3]))
                / sample.primary_channel_scale.clone().slice([0..1, 0..3]))
                * sample.primary_mask_fields.clone().slice([0..cells, 0..3]),
        );
        let coarse_primary_loss = if displacement_only_head {
            coarse_displacement_loss.clone()
        } else {
            robust_mean_square(
                ((coarse_primary_pred.clone() - sample.coarse_primary_target.clone())
                    / sample.primary_channel_scale.clone())
                    * sample.primary_mask_fields.clone(),
            )
        };
        let coarse_stress_loss = if displacement_only_head {
            tensor_scalar(0.0, &device)
        } else {
            let coarse_metric_stress = coarse_primary_pred
                .clone()
                .slice([0..cells, 3..PINO_PRIMARY_OUTPUT_CHANNELS]);
            robust_mean_square(
                ((coarse_metric_stress
                    - sample
                        .coarse_primary_target
                        .clone()
                        .slice([0..cells, 3..PINO_PRIMARY_OUTPUT_CHANNELS]))
                    / sample
                        .primary_channel_scale
                        .clone()
                        .slice([0..1, 3..PINO_PRIMARY_OUTPUT_CHANNELS]))
                    * sample
                        .primary_mask_fields
                        .clone()
                        .slice([0..cells, 3..PINO_PRIMARY_OUTPUT_CHANNELS])
                    * sample.stress_focus_primary_fields.clone(),
            )
        };

        let clamp_boundary_loss = (robust_mean_square((ux / sample.disp_scale) * sample.clamp_cells.clone())
            + robust_mean_square((uy / sample.disp_scale) * sample.clamp_cells.clone())
            + robust_mean_square((uz / sample.disp_scale) * sample.clamp_cells.clone()))
            * sample.boundary_scale;
        let traction_x = metric_sxx.clone() * sample.traction_normals.clone().slice([0..cells, 0..1])
            + metric_sxy.clone() * sample.traction_normals.clone().slice([0..cells, 1..2])
            + metric_sxz.clone() * sample.traction_normals.clone().slice([0..cells, 2..3]);
        let traction_y = metric_sxy.clone() * sample.traction_normals.clone().slice([0..cells, 0..1])
            + metric_syy.clone() * sample.traction_normals.clone().slice([0..cells, 1..2])
            + metric_syz.clone() * sample.traction_normals.clone().slice([0..cells, 2..3]);
        let traction_z = if plane_stress_2d {
            zero_stress_column()
        } else {
            metric_sxz.clone() * sample.traction_normals.clone().slice([0..cells, 0..1])
                + metric_syz.clone() * sample.traction_normals.clone().slice([0..cells, 1..2])
                + metric_szz.clone() * sample.traction_normals.clone().slice([0..cells, 2..3])
        };
        let traction_pred = Tensor::cat(vec![traction_x, traction_y, traction_z], 1);
        let traction_boundary_loss = robust_mean_square(
            ((traction_pred - sample.traction_targets.clone()) / sample.stress_scale)
                * sample.traction_mask_fields.clone(),
        ) * sample.boundary_scale;
        let boundary_loss = clamp_boundary_loss + traction_boundary_loss;
        let displacement_total = displacement_data_loss + coarse_displacement_loss * coarse_weight;
        let ring_loss_weight = if exact_surface {
            env_f64("PINO_EXACT_RING_LOSS_WEIGHT", 0.85).clamp(0.0, 4.0)
        } else {
            env_f64("PINO_RING_LOSS_WEIGHT", 0.38).clamp(0.0, 4.0)
        };
        let stress_total = if benchmark_displacement_primary {
            benchmark_sigma_xx_loss + benchmark_vm_loss * 1.25 + ring_band_loss * ring_loss_weight
        } else {
            legacy_stress_data_loss
                + coarse_stress_loss * coarse_weight
                + ring_band_loss * ring_loss_weight
        };
        let auxiliary_data_total = if benchmark_displacement_primary {
            tensor_scalar(0.0, &device)
        } else {
            auxiliary_data_loss + coarse_aux_loss * coarse_weight
        };
        data_acc = data_acc + primary_data_loss + coarse_primary_loss * coarse_weight + observable_loss.clone();
        displacement_acc = displacement_acc + displacement_total;
        stress_acc = stress_acc + stress_total;
        observable_acc = observable_acc + observable_loss;
        auxiliary_acc = auxiliary_acc
            + auxiliary_data_total.clone()
            + auxiliary_consistency_loss.clone() * if exact_surface { 0.0 } else { 0.18 };
        auxiliary_data_acc = auxiliary_data_acc + auxiliary_data_total;
        invariant_acc = invariant_acc + auxiliary_consistency_loss.clone();
        eq_acc = eq_acc + equilibrium_loss;
        constitutive_acc = constitutive_acc
            + constitutive_normal_loss.clone()
            + constitutive_shear_loss.clone()
            + weak_energy_loss.clone() * if exact_surface { 0.12 } else { 0.45 }
            + auxiliary_consistency_loss.clone() * if exact_surface { 0.0 } else { 0.35 };
        constitutive_normal_acc = constitutive_normal_acc + constitutive_normal_loss;
        constitutive_shear_acc = constitutive_shear_acc + constitutive_shear_loss;
        weak_energy_acc = weak_energy_acc + weak_energy_loss;
        boundary_acc = boundary_acc + boundary_loss;
    }

    let sample_count = samples.len() as f32;
    let data = data_acc / sample_count;
    let displacement_fit = displacement_acc / sample_count;
    let stress_fit = stress_acc / sample_count;
    let observable = observable_acc / sample_count;
    let auxiliary = auxiliary_acc / sample_count;
    let auxiliary_data = auxiliary_data_acc / sample_count;
    let invariant = invariant_acc / sample_count;
    let equilibrium = eq_acc / sample_count;
    let constitutive = constitutive_acc / sample_count;
    let constitutive_normal = constitutive_normal_acc / sample_count;
    let constitutive_shear = constitutive_shear_acc / sample_count;
    let weak_energy = weak_energy_acc / sample_count;
    let boundary = boundary_acc / sample_count;
    let total = if any_benchmark_displacement_primary {
        let base_data = loss_weights.1 as f32;
        let base_boundary = loss_weights.3 as f32;
        let displacement_rel = smooth_normalized_tail(displacement_fit.clone(), 0.05);
        let stress_rel = smooth_normalized_tail(stress_fit.clone(), 1.0);
        let observable_rel = smooth_normalized_tail(observable.clone(), 0.20);
        let data_rel = smooth_normalized_tail(data.clone(), 0.40);
        let boundary_rel = smooth_normalized_tail(boundary.clone(), 0.50);
        let equilibrium_rel = smooth_normalized_tail(equilibrium.clone(), 250.0);
        if exact_surface {
            let stress_rel = smooth_normalized_tail(stress_fit.clone(), 0.50);
            let observable_rel = smooth_normalized_tail(observable.clone(), 1.00);
            displacement_rel * (base_data.max(1.0) * 2.20)
                + stress_rel * (base_data.max(1.0) * 2.40)
                + observable_rel * (base_data.max(1.0) * 0.20)
                + data_rel * (base_data.max(1.0) * 0.05)
                + boundary_rel * (base_boundary.max(0.25) * 0.02)
                + equilibrium_rel * 0.005
        } else {
            displacement_rel * (base_data.max(1.0) * 1.70)
                + observable_rel * (base_data.max(1.0) * 1.45)
                + stress_rel * (base_data.max(1.0) * 1.30)
                + data_rel * (base_data.max(1.0) * 0.35)
                + boundary_rel * (base_boundary.max(0.25) * 0.25)
                + equilibrium_rel * 0.08
        }
    } else if isolated_exact_cantilever {
        let base_data = loss_weights.1 as f32;
        let base_eq = loss_weights.0 as f32;
        let base_constitutive = loss_weights.2 as f32;
        let base_boundary = loss_weights.3 as f32;
        let floor_proxy = (tensor_to_scalar(stress_fit.clone()).unwrap_or(0.0).abs() / 120.0)
            + (tensor_to_scalar(constitutive_normal.clone()).unwrap_or(0.0).abs() / 32.0)
            + (tensor_to_scalar(constitutive_shear.clone()).unwrap_or(0.0).abs() / 18.0)
            + (tensor_to_scalar(weak_energy.clone()).unwrap_or(0.0).abs() / 80.0)
            + (tensor_to_scalar(observable.clone()).unwrap_or(0.0).abs() / 2.0)
            + (tensor_to_scalar(equilibrium.clone()).unwrap_or(0.0).abs() / 0.5)
            + (tensor_to_scalar(displacement_fit.clone()).unwrap_or(0.0).abs() / 0.25)
            + (tensor_to_scalar(boundary.clone()).unwrap_or(0.0).abs() / 0.05);
        let relative_focus = ((floor_proxy / 20.0).clamp(0.10, 1.0)) as f32;
        let absolute_focus = (1.15 - relative_focus * 0.70).clamp(0.35, 1.10);
        let late_floor_focus = ((1.5 - floor_proxy).clamp(0.0, 1.5) / 1.5) as f32;
        let stress_scale = (24.0 - 10.0 * late_floor_focus).max(10.0);
        let constitutive_normal_scale = (6.0 - 2.5 * late_floor_focus).max(2.5);
        let data_rel = smooth_normalized_tail(data.clone(), 1.0);
        let observable_rel = smooth_normalized_tail(observable.clone(), 1.0);
        let displacement_rel = smooth_normalized_tail(displacement_fit.clone(), 0.25);
        let stress_rel = smooth_normalized_tail(stress_fit.clone(), stress_scale);
        let equilibrium_rel = smooth_normalized_tail(equilibrium.clone(), 0.10);
        let constitutive_normal_rel =
            smooth_normalized_tail(constitutive_normal.clone(), constitutive_normal_scale);
        let constitutive_shear_rel = smooth_normalized_tail(constitutive_shear.clone(), 1.6);
        let weak_energy_rel = smooth_normalized_tail(weak_energy.clone(), 0.10);
        let boundary_rel = smooth_normalized_tail(boundary.clone(), 0.05);
        let stress_tail = smooth_normalized_tail(stress_fit.clone(), stress_scale);
        let equilibrium_tail = equilibrium_rel.clone();
        let constitutive_normal_tail =
            smooth_normalized_tail(constitutive_normal.clone(), constitutive_normal_scale);
        let constitutive_shear_tail = constitutive_shear_rel.clone();
        let weak_energy_tail = weak_energy_rel.clone();
        data_rel * (base_data * 0.30 * relative_focus)
            + observable_rel * (base_data.max(1.0) * 1.15 * relative_focus)
            + displacement_rel * (base_data.max(1.0) * 1.35 * relative_focus)
            + stress_rel * (base_data.max(1.0) * (1.10 + 0.18 * late_floor_focus) * relative_focus)
            + equilibrium_rel * (base_eq.max(0.25) * 0.95 * relative_focus)
            + constitutive_normal_rel
                * (base_constitutive.max(0.25) * (1.70 + 0.24 * late_floor_focus) * relative_focus)
            + constitutive_shear_rel
                * (base_constitutive.max(0.25) * 1.40 * relative_focus)
            + weak_energy_rel * (1.95 * relative_focus)
            + boundary_rel * (base_boundary.max(0.35) * 0.60 * relative_focus)
            + stress_tail * (base_data.max(1.0) * (0.16 + 0.08 * late_floor_focus) * absolute_focus)
            + equilibrium_tail * (base_eq.max(0.25) * 0.12 * absolute_focus)
            + constitutive_normal_tail
                * (base_constitutive.max(0.25) * (0.22 + 0.10 * late_floor_focus) * absolute_focus)
            + constitutive_shear_tail
                * (base_constitutive.max(0.25) * 0.18 * absolute_focus)
            + weak_energy_tail * (0.16 * absolute_focus)
    } else if exact_surface {
        let base_data = loss_weights.1 as f32;
        let base_boundary = loss_weights.3 as f32;
        data.clone() * base_data
            + observable.clone() * (base_data * 1.4)
            + equilibrium.clone() * loss_weights.0 as f32
            + constitutive.clone() * loss_weights.2 as f32
            + boundary.clone() * base_boundary
            + stress_fit.clone() * 0.10
            + displacement_fit.clone() * (base_data.max(1.0) * 0.30)
            + stress_fit.clone() * (base_data.max(1.0) * 0.50)
            + observable.clone() * (base_data.max(1.0) * 0.25)
            + auxiliary_data.clone() * (base_data.max(1.0) * 0.08)
            + boundary.clone() * (base_boundary.max(0.25) * 0.20)
    } else {
        data.clone() * loss_weights.1 as f32
            + observable.clone() * (loss_weights.1 as f32 * 1.4)
            + auxiliary.clone() * (loss_weights.1 as f32 * auxiliary_weight)
            + equilibrium.clone() * loss_weights.0 as f32
            + constitutive.clone() * loss_weights.2 as f32
            + boundary.clone() * loss_weights.3 as f32
            + stress_fit.clone() * 0.10
    };
    Some(PhysicsLossTensors {
        total,
        data,
        displacement_fit,
        stress_fit,
        observable,
        auxiliary,
        auxiliary_data,
        invariant,
        equilibrium,
        constitutive,
        constitutive_normal,
        constitutive_shear,
        weak_energy,
        boundary,
    })
}

#[cfg(feature = "pino-ndarray-cpu")]
fn physics_breakdown_to_scalar(losses: PhysicsLossTensors) -> BurnPhysicsLossBreakdown {
    BurnPhysicsLossBreakdown {
        total: tensor_to_scalar(losses.total).unwrap_or(f64::INFINITY),
        data: tensor_to_scalar(losses.data).unwrap_or(f64::INFINITY),
        displacement_fit: tensor_to_scalar(losses.displacement_fit).unwrap_or(f64::INFINITY),
        stress_fit: tensor_to_scalar(losses.stress_fit).unwrap_or(f64::INFINITY),
        observable: tensor_to_scalar(losses.observable).unwrap_or(f64::INFINITY),
        auxiliary: tensor_to_scalar(losses.auxiliary).unwrap_or(f64::INFINITY),
        auxiliary_data: tensor_to_scalar(losses.auxiliary_data).unwrap_or(f64::INFINITY),
        invariant: tensor_to_scalar(losses.invariant).unwrap_or(f64::INFINITY),
        equilibrium: tensor_to_scalar(losses.equilibrium).unwrap_or(f64::INFINITY),
        constitutive: tensor_to_scalar(losses.constitutive).unwrap_or(f64::INFINITY),
        constitutive_normal: tensor_to_scalar(losses.constitutive_normal).unwrap_or(f64::INFINITY),
        constitutive_shear: tensor_to_scalar(losses.constitutive_shear).unwrap_or(f64::INFINITY),
        weak_energy: tensor_to_scalar(losses.weak_energy).unwrap_or(f64::INFINITY),
        boundary: tensor_to_scalar(losses.boundary).unwrap_or(f64::INFINITY),
    }
}

#[cfg(feature = "pino-ndarray-cpu")]
fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum::<f64>()
}

#[cfg(feature = "pino-ndarray-cpu")]
fn axpy(dst: &mut [f64], alpha: f64, src: &[f64]) {
    for (dst, src) in dst.iter_mut().zip(src.iter().copied()) {
        *dst += alpha * src;
    }
}

#[cfg(feature = "pino-ndarray-cpu")]
fn split_flat_params<'a>(
    flat: &'a [f64],
    weights_len: usize,
    bias_len: usize,
) -> Option<(&'a [f64], &'a [f64], &'a [f64])> {
    if flat.len() < weights_len + bias_len {
        return None;
    }
    Some((
        &flat[..weights_len],
        &flat[weights_len..weights_len + bias_len],
        &flat[weights_len + bias_len..],
    ))
}

#[cfg(feature = "pino-ndarray-cpu")]
fn dense_gradients_from_model(
    model: &BurnFieldHead<HeadBackend>,
    grads: &mut GradientsParams,
    capacity: usize,
) -> Vec<f64> {
    let mut visitor = DenseGradientCollector::new(grads, capacity);
    model.visit(&mut visitor);
    visitor.finish()
}

#[cfg(feature = "pino-ndarray-cpu")]
fn capture_model_state(model: &BurnFieldHead<HeadBackend>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let lift_weight_data = model.lift_weight.val().into_data().convert::<f64>();
    let output_weight_data = model.output_weight.val().into_data().convert::<f64>();
    let lift_bias_data = model.lift_bias.val().into_data().convert::<f64>();
    let output_bias_data = model.output_bias.val().into_data().convert::<f64>();
    let mut activation = model
        .lift_activation_scale
        .val()
        .into_data()
        .convert::<f64>()
        .value;
    let mut weights = lift_weight_data.value;
    for layer in 0..model.hidden_layers {
        weights.extend(
            model.operator_self_weights[layer]
                .val()
                .into_data()
                .convert::<f64>()
                .value,
        );
        weights.extend(
            model.operator_local_weights[layer]
                .val()
                .into_data()
                .convert::<f64>()
                .value,
        );
        weights.extend(
            model.operator_global_weights[layer]
                .val()
                .into_data()
                .convert::<f64>()
                .value,
        );
        weights.extend(
            model.operator_contrast_weights[layer]
                .val()
                .into_data()
                .convert::<f64>()
                .value,
        );
        activation.extend(
            model.operator_activation_scales[layer]
                .val()
                .into_data()
                .convert::<f64>()
                .value,
        );
    }
    weights.extend(output_weight_data.value);
    let mut bias = lift_bias_data.value;
    for bias_param in &model.operator_biases {
        bias.extend(bias_param.val().into_data().convert::<f64>().value);
    }
    bias.extend(output_bias_data.value);
    (weights, bias, activation)
}

#[cfg(feature = "pino-ndarray-cpu")]
fn ridge_refit_output_layer(
    prepared: &[PhysicsSampleTensors],
    init_weights: &[f64],
    init_bias: &[f64],
    init_activation: &[f64],
    input_dim: usize,
    output_dim: usize,
    hidden_layers: usize,
    hidden_width: usize,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let device = prepared.first()?.features.device();
    let mut model = BurnFieldHead::<HeadBackend>::new_from_state(
        &device,
        input_dim,
        output_dim,
        hidden_layers,
        hidden_width,
        init_weights,
        init_bias,
        init_activation,
    );
    let mut row_count = 0usize;
    for sample in prepared {
        row_count = row_count.saturating_add(sample.grid_nx * sample.grid_ny * sample.grid_nz);
    }
    if row_count == 0 {
        return None;
    }
    let cols = hidden_width + 1;
    let mut design = vec![0.0f64; row_count * cols];
    let mut targets = vec![0.0f64; row_count * output_dim];
    let mut row_offset = 0usize;
    for sample in prepared {
        let cells = sample.grid_nx * sample.grid_ny * sample.grid_nz;
        let hidden = model.encode(
            sample.features.clone(),
            sample.neighbor_average.clone(),
            sample.spectral_projection.clone(),
        );
        let hidden_vec = hidden.into_data().convert::<f64>().value;
        let primary = sample
            .primary_target_fields
            .clone()
            .into_data()
            .convert::<f64>()
            .value;
        let auxiliary = sample
            .auxiliary_target_fields
            .clone()
            .into_data()
            .convert::<f64>()
            .value;
        for row in 0..cells {
            let global_row = row_offset + row;
            for col in 0..hidden_width {
                design[global_row * cols + col] = hidden_vec[row * hidden_width + col];
            }
            design[global_row * cols + hidden_width] = 1.0;
            if output_dim == PINO_DISPLACEMENT_OUTPUT_CHANNELS {
                for col in 0..PINO_DISPLACEMENT_OUTPUT_CHANNELS {
                    targets[global_row * output_dim + col] =
                        primary[row * PINO_PRIMARY_OUTPUT_CHANNELS + col];
                }
            } else {
                for col in 0..PINO_PRIMARY_OUTPUT_CHANNELS {
                    targets[global_row * output_dim + col] =
                        primary[row * PINO_PRIMARY_OUTPUT_CHANNELS + col];
                }
                for col in 0..PINO_AUX_OUTPUT_CHANNELS {
                    targets[global_row * output_dim + PINO_PRIMARY_OUTPUT_CHANNELS + col] =
                        auxiliary[row * PINO_AUX_OUTPUT_CHANNELS + col];
                }
            }
        }
        row_offset += cells;
    }
    let x = DMatrix::from_row_slice(row_count, cols, &design);
    let xt = x.transpose();
    let mut xtx = &xt * &x;
    let lambda = 1e-6;
    for idx in 0..cols {
        xtx[(idx, idx)] += lambda;
    }
    let lu = xtx.lu();
    let mut output_weight = vec![0.0f64; output_dim * hidden_width];
    let mut output_bias = vec![0.0f64; output_dim];
    for out_idx in 0..output_dim {
        let mut target = Vec::with_capacity(row_count);
        for row in 0..row_count {
            target.push(targets[row * output_dim + out_idx]);
        }
        let y = DVector::from_vec(target);
        let rhs = &xt * y;
        let beta = lu.solve(&rhs)?;
        for in_idx in 0..hidden_width {
            output_weight[out_idx * hidden_width + in_idx] = beta[in_idx];
        }
        output_bias[out_idx] = beta[hidden_width];
    }
    model.output_weight = Param::from_data(
        Data::new(
            output_weight.iter().map(|value| *value as f32).collect::<Vec<_>>(),
            [output_dim, hidden_width].into(),
        ),
        &device,
    );
    model.output_bias = Param::from_data(
        Data::new(
            output_bias.iter().map(|value| *value as f32).collect::<Vec<_>>(),
            [output_dim].into(),
        ),
        &device,
    );
    Some(capture_model_state(&model))
}

fn evaluate_physics_loss_dense(
    prepared: &[PhysicsSampleTensors],
    flat_params: &[f64],
    weights_len: usize,
    bias_len: usize,
    input_dim: usize,
    output_dim: usize,
    hidden_layers: usize,
    hidden_width: usize,
    loss_weights: (f64, f64, f64, f64),
    exact_surface: bool,
) -> Option<(BurnPhysicsLossBreakdown, Vec<f64>)> {
    let (weights, bias, activation) = split_flat_params(flat_params, weights_len, bias_len)?;
    let device = prepared.first()?.features.device();
    let model = BurnFieldHead::<HeadBackend>::new_from_state(
        &device,
        input_dim,
        output_dim,
        hidden_layers,
        hidden_width,
        weights,
        bias,
        activation,
    );
    let losses = physics_loss(&model, prepared, loss_weights, exact_surface)?;
    let breakdown = BurnPhysicsLossBreakdown {
        total: tensor_to_scalar(losses.total.clone()).unwrap_or(f64::INFINITY),
        data: tensor_to_scalar(losses.data.clone()).unwrap_or(f64::INFINITY),
        displacement_fit: tensor_to_scalar(losses.displacement_fit.clone()).unwrap_or(f64::INFINITY),
        stress_fit: tensor_to_scalar(losses.stress_fit.clone()).unwrap_or(f64::INFINITY),
        observable: tensor_to_scalar(losses.observable.clone()).unwrap_or(f64::INFINITY),
        auxiliary: tensor_to_scalar(losses.auxiliary.clone()).unwrap_or(f64::INFINITY),
        auxiliary_data: tensor_to_scalar(losses.auxiliary_data.clone()).unwrap_or(f64::INFINITY),
        invariant: tensor_to_scalar(losses.invariant.clone()).unwrap_or(f64::INFINITY),
        equilibrium: tensor_to_scalar(losses.equilibrium.clone()).unwrap_or(f64::INFINITY),
        constitutive: tensor_to_scalar(losses.constitutive.clone()).unwrap_or(f64::INFINITY),
        constitutive_normal: tensor_to_scalar(losses.constitutive_normal.clone()).unwrap_or(f64::INFINITY),
        constitutive_shear: tensor_to_scalar(losses.constitutive_shear.clone()).unwrap_or(f64::INFINITY),
        weak_energy: tensor_to_scalar(losses.weak_energy.clone()).unwrap_or(f64::INFINITY),
        boundary: tensor_to_scalar(losses.boundary.clone()).unwrap_or(f64::INFINITY),
    };
    let grads = losses.total.backward();
    let mut grads = GradientsParams::from_grads(grads, &model);
    let gradient = dense_gradients_from_model(&model, &mut grads, flat_params.len());
    Some((breakdown, gradient))
}

#[cfg(feature = "pino-ndarray-cpu")]
fn train_operator_field_head_physics_inner(
    samples: &[BurnPhysicsSample],
    init_weights: &[f64],
    init_bias: &[f64],
    init_activation: &[f64],
    input_dim: usize,
    output_dim: usize,
    hidden_layers: usize,
    hidden_width: usize,
    steps: usize,
    learning_rate: f64,
    loss_weights: (f64, f64, f64, f64),
    exact_surface: bool,
    characteristic_train_scaling: bool,
) -> Option<BurnPhysicsTrainingOutcome> {
    let prepared =
        prepare_physics_samples_for_objective(samples, exact_surface, characteristic_train_scaling)?;
    let device = prepared.first()?.features.device();
    let mut model = BurnFieldHead::<HeadBackend>::new_from_state(
        &device,
        input_dim,
        output_dim,
        hidden_layers,
        hidden_width,
        init_weights,
        init_bias,
        init_activation,
    );
    let mut optimizer = AdamConfig::new().init();
    let train_steps = steps.clamp(1, 48);
    let lr = learning_rate.clamp(1e-4, 5e-2);
    let mut last = BurnPhysicsLossBreakdown::default();
    let mut saw_nonfinite = false;
    let mut best_weights = init_weights.to_vec();
    let mut best_bias = init_bias.to_vec();
    let mut best_activation = init_activation.to_vec();
    let mut best_breakdown = BurnPhysicsLossBreakdown {
        total: f64::INFINITY,
        ..BurnPhysicsLossBreakdown::default()
    };

    for _ in 0..train_steps {
        let losses = physics_loss(&model, &prepared, loss_weights, exact_surface)?;
        last = physics_breakdown_to_scalar(PhysicsLossTensors {
            total: losses.total.clone(),
            data: losses.data.clone(),
            displacement_fit: losses.displacement_fit.clone(),
            stress_fit: losses.stress_fit.clone(),
            observable: losses.observable.clone(),
            auxiliary: losses.auxiliary.clone(),
            auxiliary_data: losses.auxiliary_data.clone(),
            invariant: losses.invariant.clone(),
            equilibrium: losses.equilibrium.clone(),
            constitutive: losses.constitutive.clone(),
            constitutive_normal: losses.constitutive_normal.clone(),
            constitutive_shear: losses.constitutive_shear.clone(),
            weak_energy: losses.weak_energy.clone(),
            boundary: losses.boundary.clone(),
        });
        if !last.total.is_finite() {
            saw_nonfinite = true;
            break;
        }
        if last.total < best_breakdown.total {
            let lift_weight_data = model.lift_weight.val().into_data().convert::<f64>();
            let output_weight_data = model.output_weight.val().into_data().convert::<f64>();
            let lift_bias_data = model.lift_bias.val().into_data().convert::<f64>();
            let output_bias_data = model.output_bias.val().into_data().convert::<f64>();
            let mut activation = model
                .lift_activation_scale
                .val()
                .into_data()
                .convert::<f64>()
                .value;
            let mut weights = lift_weight_data.value;
            for layer in 0..model.hidden_layers {
                weights.extend(
                    model.operator_self_weights[layer]
                        .val()
                        .into_data()
                        .convert::<f64>()
                        .value,
                );
                weights.extend(
                    model.operator_local_weights[layer]
                        .val()
                        .into_data()
                        .convert::<f64>()
                        .value,
                );
                weights.extend(
                    model.operator_global_weights[layer]
                        .val()
                        .into_data()
                        .convert::<f64>()
                        .value,
                );
                weights.extend(
                    model.operator_contrast_weights[layer]
                        .val()
                        .into_data()
                        .convert::<f64>()
                        .value,
                );
                activation.extend(
                    model.operator_activation_scales[layer]
                        .val()
                        .into_data()
                        .convert::<f64>()
                        .value,
                );
            }
            weights.extend(output_weight_data.value);
            let mut bias = lift_bias_data.value;
            for bias_param in &model.operator_biases {
                bias.extend(bias_param.val().into_data().convert::<f64>().value);
            }
            bias.extend(output_bias_data.value);
            best_weights = weights;
            best_bias = bias;
            best_activation = activation;
            best_breakdown = last;
        }
        let grads: Gradients = losses.total.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(lr, model, grads);
    }
    if saw_nonfinite && best_breakdown.total.is_finite() {
        return Some(BurnPhysicsTrainingOutcome {
            weights: best_weights,
            bias: best_bias,
            activation: best_activation,
            breakdown: best_breakdown,
        });
    }
    let lift_weight_data = model.lift_weight.val().into_data().convert::<f64>();
    let output_weight_data = model.output_weight.val().into_data().convert::<f64>();
    let lift_bias_data = model.lift_bias.val().into_data().convert::<f64>();
    let output_bias_data = model.output_bias.val().into_data().convert::<f64>();
    let mut activation = model
        .lift_activation_scale
        .val()
        .into_data()
        .convert::<f64>()
        .value;
    let mut weights = lift_weight_data.value;
    for layer in 0..model.hidden_layers {
        weights.extend(
            model.operator_self_weights[layer]
                .val()
                .into_data()
                .convert::<f64>()
                .value,
        );
        weights.extend(
            model.operator_local_weights[layer]
                .val()
                .into_data()
                .convert::<f64>()
                .value,
        );
        weights.extend(
            model.operator_global_weights[layer]
                .val()
                .into_data()
                .convert::<f64>()
                .value,
        );
        weights.extend(
            model.operator_contrast_weights[layer]
                .val()
                .into_data()
                .convert::<f64>()
                .value,
        );
        activation.extend(
            model.operator_activation_scales[layer]
                .val()
                .into_data()
                .convert::<f64>()
                .value,
        );
    }
    weights.extend(output_weight_data.value);
    let mut bias = lift_bias_data.value;
    for bias_param in &model.operator_biases {
        bias.extend(bias_param.val().into_data().convert::<f64>().value);
    }
    bias.extend(output_bias_data.value);
    Some(BurnPhysicsTrainingOutcome {
        weights,
        bias,
        activation,
        breakdown: last,
    })
}

#[cfg(feature = "pino-ndarray-cpu")]
fn train_operator_field_head_physics_lbfgs(
    samples: &[BurnPhysicsSample],
    init_weights: &[f64],
    init_bias: &[f64],
    init_activation: &[f64],
    input_dim: usize,
    output_dim: usize,
    hidden_layers: usize,
    hidden_width: usize,
    steps: usize,
    learning_rate: f64,
    loss_weights: (f64, f64, f64, f64),
    exact_surface: bool,
    characteristic_train_scaling: bool,
) -> Option<BurnPhysicsTrainingOutcome> {
    let prepared =
        prepare_physics_samples_for_objective(samples, exact_surface, characteristic_train_scaling)?;
    let weights_len = init_weights.len();
    let bias_len = init_bias.len();
    let mut current = Vec::with_capacity(weights_len + bias_len + init_activation.len());
    current.extend_from_slice(init_weights);
    current.extend_from_slice(init_bias);
    current.extend_from_slice(init_activation);
    let (mut current_breakdown, mut current_grad) = evaluate_physics_loss_dense(
        &prepared,
        &current,
        weights_len,
        bias_len,
        input_dim,
        output_dim,
        hidden_layers,
        hidden_width,
        loss_weights,
        exact_surface,
    )?;
    let mut best = current.clone();
    let mut best_breakdown = current_breakdown;
    let mut state = DenseLbfgsState::new(current.len(), 8);
    let max_iters = steps.clamp(1, 48);
    let init_step = (learning_rate * 192.0).clamp(0.05, 1.0);

    for _ in 0..max_iters {
        let grad_norm = current_grad.iter().map(|value| value * value).sum::<f64>().sqrt();
        if !grad_norm.is_finite() || grad_norm <= 1e-7 {
            break;
        }
        let direction = state.compute_direction(&current_grad).to_vec();
        let grad_dot_dir = dot(&current_grad, &direction);
        if !grad_dot_dir.is_finite() || grad_dot_dir >= -1e-12 {
            break;
        }
        let mut accepted: Option<(f64, BurnPhysicsLossBreakdown, Vec<f64>, Vec<f64>)> = None;
        let mut step = init_step;
        for _ in 0..12 {
            let candidate = state.candidate_from(&current, &direction, step).to_vec();
            if let Some((candidate_breakdown, candidate_grad)) = evaluate_physics_loss_dense(
                &prepared,
                &candidate,
                weights_len,
                bias_len,
                input_dim,
                output_dim,
                hidden_layers,
                hidden_width,
                loss_weights,
                exact_surface,
            ) {
                let armijo_rhs = current_breakdown.total + 1e-4 * step * grad_dot_dir;
                if candidate_breakdown.total.is_finite()
                    && candidate_breakdown.total <= armijo_rhs
                {
                    accepted = Some((step, candidate_breakdown, candidate_grad, candidate));
                    break;
                }
            }
            step *= 0.5;
            if step < 1e-6 {
                break;
            }
        }
        let Some((_step, next_breakdown, next_grad, next_param)) = accepted else {
            break;
        };
        let step_delta = next_param
            .iter()
            .zip(current.iter())
            .map(|(next, current)| next - current)
            .collect::<Vec<_>>();
        let grad_delta = next_grad
            .iter()
            .zip(current_grad.iter())
            .map(|(next, current)| next - current)
            .collect::<Vec<_>>();
        state.push_history(&step_delta, &grad_delta);
        current = next_param;
        current_grad = next_grad;
        current_breakdown = next_breakdown;
        if current_breakdown.total < best_breakdown.total {
            best = current.clone();
            best_breakdown = current_breakdown;
        }
    }

    let (weights, bias, activation) = split_flat_params(&best, weights_len, bias_len)?;
    Some(BurnPhysicsTrainingOutcome {
        weights: weights.to_vec(),
        bias: bias.to_vec(),
        activation: activation.to_vec(),
        breakdown: best_breakdown,
    })
}

#[cfg(feature = "pino-ndarray-cpu")]
pub fn train_operator_field_head_physics(
    samples: &[BurnPhysicsSample],
    init_weights: &[f64],
    init_bias: &[f64],
    init_activation: &[f64],
    input_dim: usize,
    output_dim: usize,
    hidden_layers: usize,
    hidden_width: usize,
    steps: usize,
    learning_rate: f64,
    optimizer: BurnFieldHeadOptimizer,
    loss_weights: (f64, f64, f64, f64),
    exact_surface: bool,
    characteristic_train_scaling: bool,
) -> Option<BurnPhysicsTrainingOutcome> {
    let mut outcome = match optimizer {
        BurnFieldHeadOptimizer::Adam => train_operator_field_head_physics_inner(
            samples,
            init_weights,
            init_bias,
            init_activation,
            input_dim,
            output_dim,
            hidden_layers,
            hidden_width,
            steps,
            learning_rate,
            loss_weights,
            exact_surface,
            characteristic_train_scaling,
        ),
        BurnFieldHeadOptimizer::Lbfgs => train_operator_field_head_physics_lbfgs(
            samples,
            init_weights,
            init_bias,
            init_activation,
            input_dim,
            output_dim,
            hidden_layers,
            hidden_width,
            steps,
            learning_rate,
            loss_weights,
            exact_surface,
            characteristic_train_scaling,
        ),
    }?;
    if isolated_exact_cantilever_surface(exact_surface) {
        let prepared =
            prepare_physics_samples_for_objective(samples, exact_surface, characteristic_train_scaling)?;
        if let Some((ridge_weights, ridge_bias, ridge_activation)) = ridge_refit_output_layer(
            &prepared,
            &outcome.weights,
            &outcome.bias,
            &outcome.activation,
            input_dim,
            output_dim,
            hidden_layers,
            hidden_width,
        ) {
            let mut flat = Vec::with_capacity(
                ridge_weights.len() + ridge_bias.len() + ridge_activation.len(),
            );
            flat.extend_from_slice(&ridge_weights);
            flat.extend_from_slice(&ridge_bias);
            flat.extend_from_slice(&ridge_activation);
            if let Some((ridge_breakdown, _)) = evaluate_physics_loss_dense(
                &prepared,
                &flat,
                ridge_weights.len(),
                ridge_bias.len(),
                input_dim,
                output_dim,
                hidden_layers,
                hidden_width,
                loss_weights,
                exact_surface,
            ) {
                let accept_margin = (outcome.breakdown.total.abs() * 0.0005).max(1e-12);
                if ridge_breakdown.total + accept_margin < outcome.breakdown.total {
                    outcome = BurnPhysicsTrainingOutcome {
                        weights: ridge_weights,
                        bias: ridge_bias,
                        activation: ridge_activation,
                        breakdown: ridge_breakdown,
                    };
                }
            }
        }
    }
    Some(outcome)
}

#[cfg(feature = "pino-ndarray-cpu")]
pub fn evaluate_operator_field_head_physics(
    samples: &[BurnPhysicsSample],
    weights: &[f64],
    bias: &[f64],
    activation: &[f64],
    input_dim: usize,
    output_dim: usize,
    hidden_layers: usize,
    hidden_width: usize,
    loss_weights: (f64, f64, f64, f64),
    exact_surface: bool,
) -> Option<BurnPhysicsLossBreakdown> {
    let prepared = prepare_physics_samples_for_objective(samples, exact_surface, false)?;
    let device = prepared.first()?.features.device();
    let model = BurnFieldHead::<HeadBackend>::new_from_state(
        &device,
        input_dim,
        output_dim,
        hidden_layers,
        hidden_width,
        weights,
        bias,
        activation,
    );
    Some(physics_breakdown_to_scalar(physics_loss(
        &model,
        &prepared,
        loss_weights,
        exact_surface,
    )?))
}

#[cfg(feature = "pino-ndarray-cpu")]
#[allow(dead_code)]
pub fn evaluate_operator_field_head_physics_grad_norm(
    samples: &[BurnPhysicsSample],
    weights: &[f64],
    bias: &[f64],
    activation: &[f64],
    input_dim: usize,
    output_dim: usize,
    hidden_layers: usize,
    hidden_width: usize,
    loss_weights: (f64, f64, f64, f64),
    exact_surface: bool,
) -> Option<f64> {
    let prepared = prepare_physics_samples_for_objective(samples, exact_surface, false)?;
    let weights_len = weights.len();
    let bias_len = bias.len();
    let mut flat = Vec::with_capacity(weights_len + bias_len + activation.len());
    flat.extend_from_slice(weights);
    flat.extend_from_slice(bias);
    flat.extend_from_slice(activation);
    let (_breakdown, grad) = evaluate_physics_loss_dense(
        &prepared,
        &flat,
        weights_len,
        bias_len,
        input_dim,
        output_dim,
        hidden_layers,
        hidden_width,
        loss_weights,
        exact_surface,
    )?;
    let norm = grad.iter().map(|value| value * value).sum::<f64>().sqrt();
    Some(norm)
}

#[cfg(not(feature = "pino-ndarray-cpu"))]
pub fn train_operator_field_head_physics(
    _samples: &[BurnPhysicsSample],
    _init_weights: &[f64],
    _init_bias: &[f64],
    _init_activation: &[f64],
    _input_dim: usize,
    _output_dim: usize,
    _hidden_layers: usize,
    _hidden_width: usize,
    _steps: usize,
    _learning_rate: f64,
    _optimizer: BurnFieldHeadOptimizer,
    _loss_weights: (f64, f64, f64, f64),
    _exact_surface: bool,
    _characteristic_train_scaling: bool,
) -> Option<BurnPhysicsTrainingOutcome> {
    None
}

#[cfg(not(feature = "pino-ndarray-cpu"))]
pub fn evaluate_operator_field_head_physics(
    _samples: &[BurnPhysicsSample],
    _weights: &[f64],
    _bias: &[f64],
    _activation: &[f64],
    _input_dim: usize,
    _output_dim: usize,
    _hidden_layers: usize,
    _hidden_width: usize,
    _loss_weights: (f64, f64, f64, f64),
    _exact_surface: bool,
) -> Option<BurnPhysicsLossBreakdown> {
    None
}

#[cfg(not(feature = "pino-ndarray-cpu"))]
#[allow(dead_code)]
pub fn evaluate_operator_field_head_physics_grad_norm(
    _samples: &[BurnPhysicsSample],
    _weights: &[f64],
    _bias: &[f64],
    _activation: &[f64],
    _input_dim: usize,
    _output_dim: usize,
    _hidden_layers: usize,
    _hidden_width: usize,
    _loss_weights: (f64, f64, f64, f64),
    _exact_surface: bool,
) -> Option<f64> {
    None
}

#[cfg(all(test, feature = "pino-ndarray-cpu"))]
mod tests {
    use super::*;
    use crate::pino::{
        OperatorTrainableParams, PINO_FIELD_HEAD_BASIS, PINO_OUTPUT_CHANNELS,
    };

    fn simple_sample() -> BurnPhysicsSample {
        let grid_nx = 4;
        let grid_ny = 3;
        let grid_nz = 2;
        let cell_count = grid_nx * grid_ny * grid_nz;
        let mut features = Vec::with_capacity(cell_count * PINO_FIELD_HEAD_BASIS);
        let mut target_fields = Vec::with_capacity(cell_count * PINO_OUTPUT_CHANNELS);
        let mut mask = Vec::with_capacity(cell_count);
        let mut clamp = Vec::with_capacity(cell_count);
        let mut traction_mask = Vec::with_capacity(cell_count);
        let mut traction_normal = Vec::with_capacity(cell_count * 3);
        let mut traction_target = Vec::with_capacity(cell_count * 3);

        for z in 0..grid_nz {
            for y in 0..grid_ny {
                for x in 0..grid_nx {
                    let xn = x as f64 / (grid_nx.saturating_sub(1).max(1) as f64);
                    let yn = y as f64 / (grid_ny.saturating_sub(1).max(1) as f64);
                    let zn = z as f64 / (grid_nz.saturating_sub(1).max(1) as f64);
                    let x_feature = 2.0 * xn - 1.0;
                    let y_feature = 2.0 * yn - 1.0;
                    let z_feature = 2.0 * zn - 1.0;
                    let dx_center = xn - 0.5;
                    let dy_center = yn - 0.5;
                    let radial_center = (dx_center * dx_center + dy_center * dy_center).sqrt();
                    let hotspot_radius_feature = (1.0 - radial_center * 2.0).clamp(-1.0, 1.0);
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
                    let edge_symmetry_x = (1.0 - x_feature.abs()).clamp(0.0, 1.0);
                    let edge_symmetry_y = (1.0 - y_feature.abs()).clamp(0.0, 1.0);
                    let embed_feature = if x == 0 { -1.0 } else { 1.0 };
                    features.extend_from_slice(&[
                        x_feature,
                        y_feature,
                        z_feature,
                        1.0,
                        0.2,
                        1.0,
                        if x == 0 { 1.0 } else { 0.0 },
                        if x + 1 == grid_nx { 1.0 } else { 0.0 },
                        1.0,
                        0.75,
                        0.25,
                        0.7,
                        -0.1,
                        0.35,
                        -0.15,
                        x_feature.powi(2),
                        y_feature.powi(2),
                        z_feature.powi(2),
                        x_feature * y_feature,
                        x_feature * z_feature,
                        y_feature * z_feature,
                        x_feature * 0.35,
                        y_feature * -0.15,
                        z_feature,
                        x_feature * 0.1,
                        y_feature * 0.2,
                        z_feature * -0.3,
                        hotspot_radius_feature,
                        hotspot_angle_x,
                        hotspot_angle_y,
                        edge_symmetry_x,
                        edge_symmetry_y,
                        embed_feature,
                    ]);
                    let ux = 0.0025 * xn;
                    let uy = -0.0015 * yn;
                    let uz = 0.0007 * zn;
                    let sxx = 125.0 * xn;
                    let syy = -40.0 * yn;
                    let szz = 12.0 * zn;
                    let sxy = 15.0 * (xn + yn) * 0.5;
                    let sxz = 4.0 * (xn + zn) * 0.5;
                    let syz = -3.5 * (yn + zn) * 0.5;
                    let vm = (0.5
                        * ((sxx - syy).powi(2) + (syy - szz).powi(2) + (szz - sxx).powi(2))
                        + 3.0 * (sxy * sxy + sxz * sxz + syz * syz))
                        .max(0.0)
                        .sqrt();
                    let p1 = sxx.max(syy).max(szz);
                    target_fields.extend_from_slice(&[
                        ux, uy, uz, sxx, syy, szz, sxy, sxz, syz, vm, p1,
                    ]);
                    mask.push(1.0);
                    clamp.push(if x == 0 { 1.0 } else { 0.0 });
                    let is_loaded = x + 1 == grid_nx;
                    let normal = if is_loaded {
                        [1.0, 0.0, 0.0]
                    } else if x == 0 {
                        [-1.0, 0.0, 0.0]
                    } else if y == 0 {
                        [0.0, -1.0, 0.0]
                    } else if y + 1 == grid_ny {
                        [0.0, 1.0, 0.0]
                    } else if z == 0 {
                        [0.0, 0.0, -1.0]
                    } else {
                        [0.0, 0.0, 1.0]
                    };
                    traction_mask.push(if x == 0 { 0.0 } else { 1.0 });
                    traction_normal.extend_from_slice(&normal);
                    traction_target.extend_from_slice(&[
                        if is_loaded { 125.0 } else { 0.0 },
                        0.0,
                        0.0,
                    ]);
                }
            }
        }

        let mut tip_uy_projection = vec![0.0; cell_count];
        let mut end_ux_projection = vec![0.0; cell_count];
        let mut stress_projection = vec![0.0; cell_count];
        for z in 0..grid_nz {
            for y in 0..grid_ny {
                for x in 0..grid_nx {
                    let idx = z * grid_nx * grid_ny + y * grid_nx + x;
                    let xn = x as f64 / (grid_nx.saturating_sub(1).max(1) as f64);
                    let yn = y as f64 / (grid_ny.saturating_sub(1).max(1) as f64);
                    let zn = z as f64 / (grid_nz.saturating_sub(1).max(1) as f64);
                    tip_uy_projection[idx] =
                        xn.powi(8) * yn.powi(8) * (1.0 - (2.0 * zn - 1.0).abs()).max(0.0);
                    end_ux_projection[idx] = xn.powi(8);
                    stress_projection[idx] =
                        (1.0 - xn).powi(6) * (yn.powi(4) + (1.0 - yn).powi(4)).max(1e-6);
                }
            }
        }
        let normalize = |weights: &mut [f64]| {
            let sum = weights.iter().copied().sum::<f64>().max(1e-9);
            for value in weights.iter_mut() {
                *value /= sum;
            }
        };
        normalize(&mut tip_uy_projection);
        normalize(&mut end_ux_projection);
        normalize(&mut stress_projection);
        let weighted = |values: &[f64], weights: &[f64]| {
            values
                .iter()
                .copied()
                .zip(weights.iter().copied())
                .map(|(value, weight)| value * weight)
                .sum::<f64>()
        };
        let ux_values = target_fields
            .chunks_exact(PINO_OUTPUT_CHANNELS)
            .map(|values| values[0])
            .collect::<Vec<_>>();
        let uy_values = target_fields
            .chunks_exact(PINO_OUTPUT_CHANNELS)
            .map(|values| values[1])
            .collect::<Vec<_>>();
        let vm_values = target_fields
            .chunks_exact(PINO_OUTPUT_CHANNELS)
            .map(|values| values[9])
            .collect::<Vec<_>>();
        let principal_values = target_fields
            .chunks_exact(PINO_OUTPUT_CHANNELS)
            .map(|values| values[10])
            .collect::<Vec<_>>();
        let sxx_values = target_fields
            .chunks_exact(PINO_OUTPUT_CHANNELS)
            .map(|values| values[3])
            .collect::<Vec<_>>();
        let displacement_embed = clamp
            .iter()
            .map(|value| if *value > 0.5 { 0.0 } else { 1.0 })
            .collect::<Vec<_>>();

        BurnPhysicsSample {
            features,
            target_fields,
            base_fields: vec![0.0; cell_count * PINO_OUTPUT_CHANNELS],
            correction_scales: vec![1.0; cell_count * PINO_OUTPUT_CHANNELS],
            output_dim: PINO_OUTPUT_CHANNELS,
            characteristic_disp_scale: 0.0,
            characteristic_stress_scale: 0.0,
            observable_target: vec![
                weighted(&uy_values, &tip_uy_projection),
                weighted(&ux_values, &end_ux_projection),
                weighted(&vm_values, &stress_projection),
                weighted(&principal_values, &stress_projection),
                weighted(&sxx_values, &stress_projection),
            ],
            observable_scale: vec![1.0; PINO_OBSERVABLE_COUNT],
            observable_weight: vec![1.0, 1.0, 1.2, 1.0, 1.0],
            observable_projection: [
                tip_uy_projection,
                end_ux_projection,
                stress_projection.clone(),
                stress_projection,
                vec![1.0 / cell_count.max(1) as f64; cell_count],
            ]
            .concat(),
            observable_fifth_uses_vm: false,
            benchmark_vm_active: vec![1.0; cell_count],
            benchmark_sxx_peak: vec![1.0; cell_count],
            stress_focus: vec![1.0; cell_count],
            ring_loss_mask: vec![0.0; cell_count],
            mask,
            clamp,
            displacement_embed,
            traction_mask,
            traction_normal,
            traction_target,
            grid_nx,
            grid_ny,
            grid_nz,
            spectral_modes: 4,
            dx: 0.25,
            dy: 0.5,
            dz: 0.5,
            e_modulus: 30_000_000.0,
            poisson: 0.29,
        }
    }

    #[test]
    fn lbfgs_branch_reduces_physics_loss_without_fallback() {
        let sample = simple_sample();
        let params = OperatorTrainableParams::default();
        let init = evaluate_operator_field_head_physics(
            &[sample.clone()],
            &params.field_head_weights,
            &params.field_head_bias,
            &params.field_head_activation,
            PINO_FIELD_HEAD_BASIS,
            PINO_OUTPUT_CHANNELS,
            params.field_head_hidden_layers,
            params.field_head_hidden_width,
            (1.0, 1.0, 0.15, 1.0),
            false,
        )
        .expect("initial physics loss");
        let trained = train_operator_field_head_physics(
            &[sample.clone()],
            &params.field_head_weights,
            &params.field_head_bias,
            &params.field_head_activation,
            PINO_FIELD_HEAD_BASIS,
            PINO_OUTPUT_CHANNELS,
            params.field_head_hidden_layers,
            params.field_head_hidden_width,
            10,
            0.002,
            BurnFieldHeadOptimizer::Lbfgs,
            (1.0, 1.0, 0.15, 1.0),
            false,
            false,
        )
        .expect("lbfgs outcome");
        assert!(trained.breakdown.total.is_finite());
        let final_loss = evaluate_operator_field_head_physics(
            &[sample],
            &trained.weights,
            &trained.bias,
            &trained.activation,
            PINO_FIELD_HEAD_BASIS,
            PINO_OUTPUT_CHANNELS,
            params.field_head_hidden_layers,
            params.field_head_hidden_width,
            (1.0, 1.0, 0.15, 1.0),
            false,
        )
        .expect("final physics loss");
        assert!(
            final_loss.total <= init.total,
            "expected lbfgs to reduce loss: init={} final={}",
            init.total,
            final_loss.total
        );
    }

    #[test]
    fn burn_forward_matches_runtime_operator_response() {
        let sample = simple_sample();
        let params = OperatorTrainableParams::default();
        let prepared = prepare_physics_samples(&[sample.clone()]).expect("prepared sample");
        let device = prepared[0].features.device();
        let model = BurnFieldHead::<HeadBackend>::new_from_state(
            &device,
            PINO_FIELD_HEAD_BASIS,
            PINO_OUTPUT_CHANNELS,
            params.field_head_hidden_layers,
            params.field_head_hidden_width,
            &params.field_head_weights,
            &params.field_head_bias,
            &params.field_head_activation,
        );
        let burn = model
            .forward(
                prepared[0].features.clone(),
                prepared[0].neighbor_average.clone(),
                prepared[0].spectral_projection.clone(),
            )
            .into_data()
            .convert::<f64>()
            .value;
        let runtime = params.field_head_batch_response(
            &sample.features,
            sample.grid_nx * sample.grid_ny * sample.grid_nz,
            PINO_FIELD_HEAD_BASIS,
            sample.grid_nx,
            sample.grid_ny,
            sample.grid_nz,
            sample.spectral_modes,
            sample.output_dim,
        );
        let max_diff = burn
            .iter()
            .zip(runtime.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff <= 5.0e-4,
            "burn/runtime operator mismatch too large: {max_diff}"
        );
    }
}
