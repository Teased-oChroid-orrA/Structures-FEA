use rand::Rng;

use crate::contracts::{
    AnnResult, ModelStatus, NetworkConnectionSnapshot, NetworkNodeSnapshot, NetworkSnapshot,
    SolveInput, TrainResult, TrainingBatch, TrainingProgressEvent,
};
use crate::fem;

#[derive(Clone)]
struct DenseLayer {
    weights: Vec<Vec<f64>>, // out x in
    biases: Vec<f64>,
}

#[derive(Clone)]
pub struct AnnModel {
    layer_sizes: Vec<usize>,
    layers: Vec<DenseLayer>,
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
}

#[derive(Clone, Copy)]
struct PhysicsContext {
    length_in: f64,
    width_in: f64,
    thickness_in: f64,
    vertical_load_lbf: f64,
    e_psi: f64,
}

#[derive(Clone)]
struct TrainingSample {
    x: Vec<f64>,
    y: Vec<f64>,
    physics: PhysicsContext,
}

impl Default for AnnModel {
    fn default() -> Self {
        Self::new(vec![8, 12, 12, 6])
    }
}

impl AnnModel {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        let layers = initialize_layers(&layer_sizes);
        Self {
            layer_sizes,
            layers,
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
        }
    }

    pub fn train_with_progress<F>(&mut self, batch: &TrainingBatch, mut on_epoch: F) -> TrainResult
    where
        F: FnMut(TrainingProgressEvent),
    {
        if let Some(lr) = batch.learning_rate {
            self.learning_rate = lr.clamp(1e-6, 1e-1);
        }

        let samples = if batch.cases.is_empty() {
            vec![SolveInput::default()]
        } else {
            batch.cases.clone()
        };

        let mut dataset = Vec::new();
        for case in &samples {
            let fem = fem::solve_case(case);
            dataset.push(TrainingSample {
                x: fem::ann_features(case),
                y: fem::ann_targets(&fem),
                physics: PhysicsContext {
                    length_in: case.geometry.length_in,
                    width_in: case.geometry.width_in,
                    thickness_in: case.geometry.thickness_in,
                    vertical_load_lbf: case.load.vertical_point_load_lbf,
                    e_psi: case.material.e_psi,
                },
            });
        }

        self.train_samples += dataset.len();

        let split_idx = ((dataset.len() as f64) * 0.8).ceil() as usize;
        let (train_data, val_data) = dataset.split_at(split_idx.min(dataset.len()));
        let cycle_epochs = batch.epochs.clamp(1, 200);
        let auto_mode = batch.auto_mode.unwrap_or(true);
        let max_total_epochs = batch
            .max_total_epochs
            .unwrap_or(cycle_epochs * 20)
            .clamp(1, 100_000_000);
        let emit_stride = (max_total_epochs / 2000).clamp(1, 100);
        let min_improvement = batch.min_improvement.unwrap_or(1e-7).max(1e-12);
        let target_loss = batch.target_loss.max(0.0);

        let mut loss = 0.0;
        let mut val_loss = f64::MAX;
        let mut completed_epochs = 0usize;
        let mut reached_target = false;
        let mut prev_cycle_best = f64::MAX;

        while completed_epochs < max_total_epochs {
            let cycle_start_best = val_loss;

            for _ in 0..cycle_epochs {
                if completed_epochs >= max_total_epochs {
                    break;
                }

                loss = self.epoch(train_data);
                val_loss = self.eval_loss(if val_data.is_empty() { train_data } else { val_data });
                completed_epochs += 1;

                if val_loss + 1e-10 < self.best_val_loss {
                    self.best_val_loss = val_loss;
                    self.best_layers = Some(self.layers.clone());
                    self.plateau_epochs = 0;
                } else {
                    self.plateau_epochs += 1;
                }

                if val_loss > self.best_val_loss * 1.5 {
                    if let Some(saved) = self.best_layers.clone() {
                        self.layers = saved;
                    }
                    self.learning_rate *= 0.5;
                }

                if completed_epochs == 1
                    || completed_epochs % emit_stride == 0
                    || completed_epochs >= max_total_epochs
                    || (auto_mode && val_loss <= target_loss)
                {
                    on_epoch(TrainingProgressEvent {
                        epoch: completed_epochs,
                        total_epochs: max_total_epochs,
                        loss,
                        val_loss,
                        learning_rate: self.learning_rate,
                        architecture: self.layer_sizes.clone(),
                        progress_ratio: completed_epochs as f64 / max_total_epochs as f64,
                        network: self.network_snapshot(),
                    });
                }

                if auto_mode && val_loss <= target_loss {
                    reached_target = true;
                    break;
                }
            }

            if !auto_mode {
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

            // Topology and LR adaptation between cycles.
            if self.plateau_epochs >= 3 && val_loss > target_loss {
                let _ = self.grow_topology();
                self.model_version += 1;
            } else if val_loss < (target_loss * 0.35).max(1e-10) {
                if self.prune_topology() {
                    self.model_version += 1;
                }
            }
        }

        let mut notes = vec![];
        let mut grew = false;
        let mut pruned = false;

        if self.plateau_epochs >= 3 && val_loss > target_loss {
            grew = self.grow_topology();
            if grew {
                notes.push("Adaptive growth triggered due to plateau and residual threshold.".to_string());
                self.model_version += 1;
            }
        }

        if val_loss < (target_loss * 0.35).max(1e-10) {
            pruned = self.prune_topology();
            if pruned {
                notes.push("Adaptive pruning triggered: reduced hidden complexity after convergence.".to_string());
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

        TrainResult {
            model_version: self.model_version,
            loss,
            val_loss,
            architecture: self.layer_sizes.clone(),
            learning_rate: self.learning_rate,
            grew,
            pruned,
            completed_epochs,
            reached_target,
            stop_reason: if reached_target {
                "target-loss-reached".to_string()
            } else if completed_epochs >= max_total_epochs {
                "max-epochs-reached".to_string()
            } else {
                "manual-stop".to_string()
            },
            notes,
        }
    }

    pub fn infer(&self, input: &SolveInput) -> AnnResult {
        let x = fem::ann_features(input);
        let pred = self.forward(&x);
        let mut fem_like = fem::fem_from_ann_prediction(input, &pred);

        let confidence = (1.0 / (1.0 + self.last_loss.abs())).clamp(0.0, 1.0);
        let uncertainty = (1.0 - confidence).clamp(0.0, 1.0);

        let mut used_fallback = false;
        let mut diagnostics = vec![format!(
            "ANN inference with model v{} and architecture {:?}",
            self.model_version, self.layer_sizes
        )];

        if self.fallback_enabled && uncertainty > 0.35 {
            fem_like = fem::solve_case(input);
            used_fallback = true;
            diagnostics.push("Uncertainty gate exceeded threshold; FEM fallback used.".to_string());
        }

        AnnResult {
            fem_like,
            confidence,
            uncertainty,
            model_version: self.model_version,
            used_fem_fallback: used_fallback,
            diagnostics,
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
        }
    }

    fn network_snapshot(&self) -> NetworkSnapshot {
        let mut nodes = Vec::new();
        let mut connections = Vec::new();
        let mut max_abs_weight: f64 = 1e-9;

        let sample_input = vec![10.0, 4.0, 0.125, 0.0, -1000.0, 10_000_000.0, 0.33, 13e-6];
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

    fn epoch(&mut self, data: &[TrainingSample]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        for sample in data {
            let (activations, zs) = self.forward_cache(&sample.x);
            let y_pred = activations.last().cloned().unwrap_or_default();
            let (physics_loss, physics_grad) = physics_penalty(&y_pred, sample.physics);

            let mut deltas: Vec<Vec<f64>> = vec![vec![]; self.layers.len()];
            let mut last_delta = vec![0.0; sample.y.len()];
            for i in 0..sample.y.len() {
                let err = y_pred[i] - sample.y[i];
                let data_grad = 2.0 * err / sample.y.len() as f64;
                let phys_grad = physics_grad.get(i).copied().unwrap_or(0.0);
                last_delta[i] = data_grad + phys_grad;
                total_loss += err * err;
            }
            total_loss += physics_loss;
            deltas[self.layers.len() - 1] = last_delta;

            for l in (0..self.layers.len() - 1).rev() {
                let z = &zs[l];
                let mut delta = vec![0.0; z.len()];
                for i in 0..z.len() {
                    let mut acc = 0.0;
                    for j in 0..self.layers[l + 1].weights.len() {
                        acc += self.layers[l + 1].weights[j][i] * deltas[l + 1][j];
                    }
                    delta[i] = acc * (1.0 - z[i].tanh().powi(2));
                }
                deltas[l] = delta;
            }

            for l in 0..self.layers.len() {
                let a_prev = &activations[l];
                for o in 0..self.layers[l].weights.len() {
                    for i in 0..self.layers[l].weights[o].len() {
                        self.layers[l].weights[o][i] -= self.learning_rate * deltas[l][o] * a_prev[i];
                    }
                    self.layers[l].biases[o] -= self.learning_rate * deltas[l][o];
                }
            }
        }

        total_loss / data.len() as f64
    }

    fn eval_loss(&self, data: &[TrainingSample]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        for sample in data {
            let p = self.forward(&sample.x);
            for i in 0..sample.y.len() {
                total += (p[i] - sample.y[i]).powi(2);
            }
            let (physics_loss, _) = physics_penalty(&p, sample.physics);
            total += physics_loss;
        }
        total / data.len() as f64
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
                next[o] = if last { z } else { z.tanh() };
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
                next[o] = if last { z } else { z.tanh() };
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
                next[o] = if last { z } else { z.tanh() };
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

        let last_hidden_idx = self.layer_sizes.len() - 2;
        if self.layer_sizes[last_hidden_idx] < self.max_neurons_per_layer {
            self.layer_sizes[last_hidden_idx] += 4;
            self.layers = initialize_layers(&self.layer_sizes);
            self.plateau_epochs = 0;
            return true;
        }

        let hidden_count = self.layer_sizes.len() - 2;
        if hidden_count < self.max_hidden_layers {
            let output = self.layer_sizes.pop().unwrap_or(6);
            self.layer_sizes.push(8);
            self.layer_sizes.push(output);
            self.layers = initialize_layers(&self.layer_sizes);
            self.plateau_epochs = 0;
            return true;
        }

        false
    }

    fn prune_topology(&mut self) -> bool {
        if self.layer_sizes.len() < 3 {
            return false;
        }

        let last_hidden_idx = self.layer_sizes.len() - 2;
        if self.layer_sizes[last_hidden_idx] > 4 {
            self.layer_sizes[last_hidden_idx] -= 2;
            self.layers = initialize_layers(&self.layer_sizes);
            self.plateau_epochs = 0;
            return true;
        }

        let hidden_count = self.layer_sizes.len() - 2;
        if hidden_count > 1 {
            let output = self.layer_sizes.pop().unwrap_or(6);
            self.layer_sizes.pop();
            self.layer_sizes.push(output);
            self.layers = initialize_layers(&self.layer_sizes);
            self.plateau_epochs = 0;
            return true;
        }

        false
    }
}

fn physics_penalty(pred: &[f64], physics: PhysicsContext) -> (f64, Vec<f64>) {
    if pred.len() < 6 {
        return (0.0, vec![0.0; pred.len()]);
    }

    let mut grad = vec![0.0; pred.len()];

    let tip = pred[0];
    let sxx = pred[1];
    let syy = pred[2];
    let szz = pred[3];
    let vm_pred = pred[4];
    let maxp_pred = pred[5];

    let l = physics.length_in.max(1e-6);
    let w = physics.width_in.max(1e-6);
    let t = physics.thickness_in.max(1e-6);
    let p = physics.vertical_load_lbf;
    let e = physics.e_psi.max(1.0);
    let c = 0.5 * w;
    let i = (t * w.powi(3) / 12.0).max(1e-9);

    // Equilibrium/moment consistency: sigma*c/I = P*L.
    let r_eq = (sxx * i / c.max(1e-9)) - p * l;
    let w_eq = 1e-8;
    let d_r_eq_d_sxx = i / c.max(1e-9);
    let l_eq = w_eq * r_eq * r_eq;
    grad[1] += 2.0 * w_eq * r_eq * d_r_eq_d_sxx;

    // Constitutive/kinematic consistency for cantilever root curvature relation.
    let kappa = (3.0 * e * c / (l * l)).max(1e-9);
    let r_const = sxx - kappa * tip;
    let w_const = 1e-8;
    let l_const = w_const * r_const * r_const;
    grad[0] += 2.0 * w_const * r_const * (-kappa);
    grad[1] += 2.0 * w_const * r_const;

    // Stress-invariant consistency: VM should match stress components.
    let q = 0.5 * ((sxx - syy).powi(2) + (syy - szz).powi(2) + (szz - sxx).powi(2));
    let vm_calc = q.max(1e-12).sqrt();
    let r_vm = vm_pred.abs() - vm_calc;
    let w_vm = 5e-3;
    let l_vm = w_vm * r_vm * r_vm;
    let sign_vm = if vm_pred >= 0.0 { 1.0 } else { -1.0 };
    grad[4] += 2.0 * w_vm * r_vm * sign_vm;

    let denom = (2.0 * vm_calc).max(1e-9);
    let d_vm_dsxx = (2.0 * sxx - syy - szz) / denom;
    let d_vm_dsyy = (2.0 * syy - sxx - szz) / denom;
    let d_vm_dszz = (2.0 * szz - sxx - syy) / denom;
    grad[1] += 2.0 * w_vm * r_vm * (-d_vm_dsxx);
    grad[2] += 2.0 * w_vm * r_vm * (-d_vm_dsyy);
    grad[3] += 2.0 * w_vm * r_vm * (-d_vm_dszz);

    // Principal stress head consistency.
    let mut principal_max = sxx;
    let mut principal_idx = 1usize;
    if syy > principal_max {
        principal_max = syy;
        principal_idx = 2;
    }
    if szz > principal_max {
        principal_max = szz;
        principal_idx = 3;
    }
    let r_p = maxp_pred - principal_max;
    let w_p = 1e-3;
    let l_p = w_p * r_p * r_p;
    grad[5] += 2.0 * w_p * r_p;
    grad[principal_idx] += -2.0 * w_p * r_p;

    // Boundary/load consistency: near-zero load should imply near-zero response.
    let zero_gate = 1.0 / (1.0 + p.abs());
    let r_bc_tip = zero_gate * tip;
    let r_bc_sig = zero_gate * sxx / e;
    let w_bc = 1e-2;
    let l_bc = w_bc * (r_bc_tip * r_bc_tip + r_bc_sig * r_bc_sig);
    grad[0] += 2.0 * w_bc * r_bc_tip * zero_gate;
    grad[1] += 2.0 * w_bc * r_bc_sig * (zero_gate / e);

    (l_eq + l_const + l_vm + l_p + l_bc, grad)
}

fn initialize_layers(layer_sizes: &[usize]) -> Vec<DenseLayer> {
    let mut rng = rand::thread_rng();
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

        layers.push(DenseLayer { weights: w, biases: b });
    }

    layers
}
