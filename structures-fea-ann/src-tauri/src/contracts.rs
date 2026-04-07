use serde::{Deserialize, Serialize};

pub const SUPPORTED_ELEMENT_TYPE: &str = "hex8";
pub const SUPPORTED_UNIT_SYSTEM: &str = "inch-lbf-second";
pub const MAX_DENSE_SOLVER_DOFS: usize = 3_200;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometryInput {
    pub length_in: f64,
    pub width_in: f64,
    pub thickness_in: f64,
    #[serde(default)]
    pub hole_diameter_in: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeshControls {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub element_type: String,
    #[serde(default = "default_true")]
    pub auto_adapt: bool,
    #[serde(default = "default_max_dofs")]
    pub max_dofs: usize,
    #[serde(default = "default_true")]
    pub amr_enabled: bool,
    #[serde(default = "default_amr_passes")]
    pub amr_passes: usize,
    #[serde(default = "default_amr_max_nx")]
    pub amr_max_nx: usize,
    #[serde(default = "default_amr_refine_ratio")]
    pub amr_refine_ratio: f64,
}

fn default_true() -> bool {
    true
}

fn default_max_dofs() -> usize {
    MAX_DENSE_SOLVER_DOFS
}

fn default_amr_passes() -> usize {
    2
}

fn default_amr_max_nx() -> usize {
    28
}

fn default_amr_refine_ratio() -> f64 {
    1.2
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundaryConditionInput {
    pub fix_start_face: bool,
    pub fix_end_face: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadInput {
    pub axial_load_lbf: f64,
    pub vertical_point_load_lbf: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Material {
    pub e_psi: f64,
    pub nu: f64,
    pub rho_lb_in3: f64,
    pub alpha_per_f: f64,
    pub yield_strength_psi: f64,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            e_psi: 29_000_000.0,
            nu: 0.3,
            rho_lb_in3: 0.283,
            alpha_per_f: 6.5e-6,
            yield_strength_psi: 36_000.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SolveInput {
    pub geometry: GeometryInput,
    pub mesh: MeshControls,
    pub material: Material,
    pub boundary_conditions: BoundaryConditionInput,
    pub load: LoadInput,
    pub unit_system: String,
    pub delta_t_f: Option<f64>,
}

impl Default for SolveInput {
    fn default() -> Self {
        Self {
            geometry: GeometryInput {
                length_in: 11.811,
                width_in: 4.724,
                thickness_in: 0.25,
                hole_diameter_in: Some(2.362),
            },
            mesh: MeshControls {
                nx: 28,
                ny: 14,
                nz: 1,
                element_type: "hex8".to_string(),
                auto_adapt: true,
                max_dofs: MAX_DENSE_SOLVER_DOFS,
                amr_enabled: true,
                amr_passes: 3,
                amr_max_nx: 40,
                amr_refine_ratio: 1.15,
            },
            material: Material::default(),
            boundary_conditions: BoundaryConditionInput {
                fix_start_face: true,
                fix_end_face: false,
            },
            load: LoadInput {
                axial_load_lbf: 1_712.0,
                vertical_point_load_lbf: 0.0,
            },
            unit_system: "inch-lbf-second".to_string(),
            delta_t_f: Some(0.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodalDisplacement {
    pub node_id: usize,
    pub x_in: f64,
    pub y_in: f64,
    pub z_in: f64,
    pub ux_in: f64,
    pub uy_in: f64,
    pub uz_in: f64,
    pub disp_mag_in: f64,
    pub vm_psi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FemResult {
    pub nodal_displacements: Vec<NodalDisplacement>,
    pub strain_tensor: [[f64; 3]; 3],
    pub stress_tensor: [[f64; 3]; 3],
    pub principal_stresses: [f64; 3],
    pub von_mises_psi: f64,
    pub tresca_psi: f64,
    pub max_principal_psi: f64,
    pub stiffness_matrix: Vec<Vec<f64>>,
    pub mass_matrix: Vec<Vec<f64>>,
    pub damping_matrix: Vec<Vec<f64>>,
    pub force_vector: Vec<f64>,
    pub displacement_vector: Vec<f64>,
    pub beam_stations: Vec<BeamStationResult>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BeamStationResult {
    pub x_in: f64,
    pub shear_lbf: f64,
    pub moment_lb_in: f64,
    pub sigma_top_psi: f64,
    pub sigma_bottom_psi: f64,
    pub deflection_in: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThermalInput {
    pub solve_input: SolveInput,
    pub delta_t_f: f64,
    pub restrained_x: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThermalResult {
    pub thermal_strain_x: f64,
    pub thermal_stress_psi: f64,
    pub combined_stress_tensor: [[f64; 3]; 3],
    pub principal_stresses: [f64; 3],
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DynamicInput {
    pub solve_input: SolveInput,
    pub time_step_s: f64,
    pub end_time_s: f64,
    pub damping_ratio: f64,
    pub pulse_duration_s: f64,
    pub pulse_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DynamicResult {
    pub time_s: Vec<f64>,
    pub displacement_in: Vec<f64>,
    pub velocity_in_s: Vec<f64>,
    pub acceleration_in_s2: Vec<f64>,
    pub stable: bool,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FailureInput {
    pub stress_tensor: [[f64; 3]; 3],
    pub yield_strength_psi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FailureResult {
    pub von_mises_psi: f64,
    pub tresca_psi: f64,
    pub max_principal_psi: f64,
    pub safety_factor_vm: f64,
    pub safety_factor_tresca: f64,
    pub safety_factor_principal: f64,
    pub failed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingBatch {
    pub cases: Vec<SolveInput>,
    pub epochs: usize,
    pub target_loss: f64,
    #[serde(default)]
    pub training_mode: Option<String>,
    #[serde(default)]
    pub benchmark_id: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub analysis_type: Option<String>,
    #[serde(default)]
    pub pinn_backend: Option<String>,
    #[serde(default)]
    pub collocation_points: Option<usize>,
    #[serde(default)]
    pub boundary_points: Option<usize>,
    #[serde(default)]
    pub interface_points: Option<usize>,
    #[serde(default)]
    pub residual_weight_momentum: Option<f64>,
    #[serde(default)]
    pub residual_weight_kinematics: Option<f64>,
    #[serde(default)]
    pub residual_weight_material: Option<f64>,
    #[serde(default)]
    pub residual_weight_boundary: Option<f64>,
    #[serde(default)]
    pub stage1_epochs: Option<usize>,
    #[serde(default)]
    pub stage2_epochs: Option<usize>,
    #[serde(default)]
    pub stage3_ramp_epochs: Option<usize>,
    #[serde(default)]
    pub contact_penalty: Option<f64>,
    #[serde(default)]
    pub plasticity_factor: Option<f64>,
    pub learning_rate: Option<f64>,
    pub auto_mode: Option<bool>,
    pub max_total_epochs: Option<usize>,
    pub min_improvement: Option<f64>,
    pub progress_emit_every_epochs: Option<usize>,
    pub network_emit_every_epochs: Option<usize>,
    pub online_active_learning: Option<bool>,
    pub autonomous_mode: Option<bool>,
    pub max_topology: Option<usize>,
    pub max_backoffs: Option<usize>,
    pub max_optimizer_switches: Option<usize>,
    pub checkpoint_every_epochs: Option<usize>,
    pub checkpoint_retention: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OperatorGridSpec {
    pub nx: usize,
    pub ny: usize,
    #[serde(default = "default_operator_grid_nz")]
    pub nz: usize,
    pub input_channels: usize,
    pub output_channels: usize,
}

fn default_operator_grid_nz() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HoldoutValidationSummary {
    pub trusted: bool,
    pub training_seed_cases: usize,
    pub holdout_cases: usize,
    pub mean_displacement_error: f64,
    pub mean_von_mises_error: f64,
    pub p95_field_error: f64,
    pub residual_ratio: f64,
    pub accepted_without_fallback: bool,
    #[serde(default = "default_holdout_mean_error_limit")]
    pub mean_error_limit: f64,
    #[serde(default = "default_holdout_p95_error_limit")]
    pub p95_error_limit: f64,
    #[serde(default = "default_holdout_residual_ratio_limit")]
    pub residual_ratio_limit: f64,
    #[serde(default)]
    pub displacement_pass: bool,
    #[serde(default)]
    pub von_mises_pass: bool,
    #[serde(default)]
    pub p95_pass: bool,
    #[serde(default)]
    pub residual_ratio_pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SurrogateDomainSummary {
    pub feature_labels: Vec<String>,
    pub feature_mins: Vec<f64>,
    pub feature_maxs: Vec<f64>,
    #[serde(default)]
    pub coverage_tags: Vec<String>,
    pub training_seed_cases: usize,
    pub expanded_cases: usize,
    #[serde(default)]
    pub mixed_load_cases: usize,
    #[serde(default)]
    pub hole_cases: usize,
    #[serde(default)]
    pub dual_fixed_cases: usize,
}

fn default_holdout_mean_error_limit() -> f64 {
    0.05
}

fn default_holdout_p95_error_limit() -> f64 {
    0.10
}

fn default_holdout_residual_ratio_limit() -> f64 {
    0.80
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PinoLocalRefinementMetadata {
    pub enabled: bool,
    pub strategy: String,
    pub max_patches: usize,
    pub max_patch_cells: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PinoLocalEnrichmentMetadata {
    pub enabled: bool,
    pub strategy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PinoRuntimeMetadata {
    pub engine_id: String,
    pub backend: String,
    pub spectral_modes: usize,
    pub operator_grid: OperatorGridSpec,
    #[serde(default = "default_pino_domain_dim")]
    pub domain_dim: usize,
    #[serde(default = "default_pino_physics_model")]
    pub physics_model: String,
    #[serde(default)]
    pub spectral_modes_3d: [usize; 3],
    #[serde(default)]
    pub operator_grid_3d: Option<OperatorGridSpec>,
    #[serde(default)]
    pub boundary_mode: Option<String>,
    #[serde(default)]
    pub objective_mode: Option<String>,
    #[serde(default)]
    pub local_refinement: Option<PinoLocalRefinementMetadata>,
    #[serde(default)]
    pub local_enrichment: Option<PinoLocalEnrichmentMetadata>,
    #[serde(default)]
    pub calibration_stress_scale: Option<f64>,
    #[serde(default)]
    pub calibration_displacement_scale: Option<f64>,
    pub holdout_validation: Option<HoldoutValidationSummary>,
}

fn default_pino_domain_dim() -> usize {
    2
}

fn default_pino_physics_model() -> String {
    "plate-operator-2d-linear-elastic".to_string()
}

impl GeometryInput {
    fn validate(&self) -> Result<(), String> {
        if !self.length_in.is_finite() || self.length_in <= 0.0 {
            return Err("Geometry length must be a positive finite value.".to_string());
        }
        if !self.width_in.is_finite() || self.width_in <= 0.0 {
            return Err("Geometry width must be a positive finite value.".to_string());
        }
        if !self.thickness_in.is_finite() || self.thickness_in <= 0.0 {
            return Err("Geometry thickness must be a positive finite value.".to_string());
        }
        if let Some(hole) = self.hole_diameter_in {
            if !hole.is_finite() {
                return Err("Hole diameter must be finite when provided.".to_string());
            }
            if hole < 0.0 {
                return Err("Hole diameter cannot be negative.".to_string());
            }
            if hole > self.length_in.min(self.width_in) {
                return Err(format!(
                    "Hole diameter {hole} exceeds the in-plane dimensions of the plate."
                ));
            }
        }
        Ok(())
    }
}

impl MeshControls {
    pub fn estimated_dofs(&self) -> usize {
        (self.nx + 1) * (self.ny + 1) * (self.nz + 1) * 3
    }

    fn validate(&self) -> Result<(), String> {
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err("Mesh subdivision counts must be positive.".to_string());
        }
        if self.element_type != SUPPORTED_ELEMENT_TYPE {
            return Err(format!(
                "Unsupported element type '{}'. Only '{}' is currently implemented.",
                self.element_type, SUPPORTED_ELEMENT_TYPE
            ));
        }
        if self.max_dofs == 0 {
            return Err("maxDofs must be a positive integer.".to_string());
        }
        if self.max_dofs > MAX_DENSE_SOLVER_DOFS {
            return Err(format!(
                "maxDofs {} exceeds the current dense solver ceiling of {} DOFs.",
                self.max_dofs, MAX_DENSE_SOLVER_DOFS
            ));
        }
        let estimated_dofs = self.estimated_dofs();
        if !self.auto_adapt && estimated_dofs > self.max_dofs {
            return Err(format!(
                "Requested mesh estimates {} DOFs, which exceeds maxDofs {} with autoAdapt disabled.",
                estimated_dofs, self.max_dofs
            ));
        }
        if !self.amr_refine_ratio.is_finite() || self.amr_refine_ratio < 1.0 {
            return Err("AMR refine ratio must be finite and at least 1.0.".to_string());
        }
        Ok(())
    }
}

impl Material {
    fn validate(&self) -> Result<(), String> {
        if !self.e_psi.is_finite() || self.e_psi <= 0.0 {
            return Err("Young's modulus must be a positive finite value.".to_string());
        }
        if !self.nu.is_finite() || self.nu <= -1.0 || self.nu >= 0.5 {
            return Err("Poisson ratio must be finite and in (-1, 0.5).".to_string());
        }
        if !self.rho_lb_in3.is_finite() || self.rho_lb_in3 <= 0.0 {
            return Err("Density must be a positive finite value.".to_string());
        }
        if !self.alpha_per_f.is_finite() {
            return Err("Thermal expansion coefficient must be finite.".to_string());
        }
        if !self.yield_strength_psi.is_finite() || self.yield_strength_psi <= 0.0 {
            return Err("Yield strength must be a positive finite value.".to_string());
        }
        Ok(())
    }
}

impl SolveInput {
    pub fn analysis_mode_label(&self) -> &'static str {
        if self.is_plate_hole_benchmark() {
            "analytical-benchmark"
        } else {
            "general-fem"
        }
    }

    pub fn is_plate_hole_benchmark(&self) -> bool {
        self.geometry
            .hole_diameter_in
            .map(|d| d > 0.0)
            .unwrap_or(false)
            && self.load.axial_load_lbf.abs() > 0.0
            && self.load.vertical_point_load_lbf.abs() < 1e-9
    }

    pub fn is_simple_cantilever_verification(&self) -> bool {
        self.boundary_conditions.fix_start_face
            && !self.boundary_conditions.fix_end_face
            && self.load.axial_load_lbf.abs() < 1e-9
            && self.load.vertical_point_load_lbf.abs() > 1e-9
            && self
                .geometry
                .hole_diameter_in
                .map(|d| d.abs() <= 1e-9)
                .unwrap_or(true)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.geometry.validate()?;
        self.mesh.validate()?;
        self.material.validate()?;
        if !self.boundary_conditions.fix_start_face && !self.boundary_conditions.fix_end_face {
            return Err(
                "At least one boundary face must be fixed to avoid a singular system.".to_string(),
            );
        }
        if self.unit_system != SUPPORTED_UNIT_SYSTEM {
            return Err(format!(
                "Unsupported unit system '{}'. Only '{}' is currently implemented.",
                self.unit_system, SUPPORTED_UNIT_SYSTEM
            ));
        }
        if !self.load.axial_load_lbf.is_finite() || !self.load.vertical_point_load_lbf.is_finite() {
            return Err("Loads must be finite values.".to_string());
        }
        if let Some(delta_t_f) = self.delta_t_f {
            if !delta_t_f.is_finite() {
                return Err("deltaTF must be finite when provided.".to_string());
            }
        }
        Ok(())
    }
}

impl TrainingBatch {
    pub fn validate(&self) -> Result<(), String> {
        if self.cases.is_empty() {
            return Err("Training batch must contain at least one case.".to_string());
        }
        if self.epochs == 0 {
            return Err("Training batch epochs must be positive.".to_string());
        }
        if !self.target_loss.is_finite() || self.target_loss < 0.0 {
            return Err("Training target loss must be a finite, non-negative value.".to_string());
        }
        for (idx, case) in self.cases.iter().enumerate() {
            case.validate()
                .map_err(|err| format!("Training case {idx} is invalid: {err}"))?;
        }

        if let Some(analysis_type) = &self.analysis_type {
            match analysis_type.as_str() {
                "general" | "cantilever" | "plate-hole" => {}
                other => {
                    return Err(format!(
                        "Unsupported analysis type '{other}'. Expected general, cantilever, or plate-hole."
                    ))
                }
            }
        }

        if let Some(backend) = &self.pinn_backend {
            match backend.as_str() {
                "pino-ndarray-cpu"
                | "pino-candle-cpu"
                | "pino-candle-cuda"
                | "pino-candle-metal"
                | "burn-ndarray-cpu"
                | "burn-wgpu" => {}
                other => {
                    return Err(format!(
                        "Unsupported PINO backend '{other}'."
                    ))
                }
            }
        }

        if let Some(training_mode) = &self.training_mode {
            match training_mode.as_str() {
                "legacy-mixed-exact" | "benchmark" | "production-generalized" => {}
                other => {
                    return Err(format!(
                        "Unsupported training mode '{other}'. Expected legacy-mixed-exact, benchmark, or production-generalized."
                    ))
                }
            }
        }

        if matches!(self.benchmark_id.as_deref(), Some("")) {
            return Err("benchmarkId must be non-empty when provided.".to_string());
        }
        if matches!(self.training_mode.as_deref(), Some("benchmark")) && self.benchmark_id.is_none() {
            return Err("benchmarkId is required when trainingMode is benchmark.".to_string());
        }

        for (name, value) in [
            ("learningRate", self.learning_rate),
            ("minImprovement", self.min_improvement),
            ("contactPenalty", self.contact_penalty),
            ("plasticityFactor", self.plasticity_factor),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || value < 0.0 {
                    return Err(format!("{name} must be a finite, non-negative value."));
                }
            }
        }

        for (name, value) in [
            ("collocationPoints", self.collocation_points),
            ("boundaryPoints", self.boundary_points),
            ("interfacePoints", self.interface_points),
            ("stage1Epochs", self.stage1_epochs),
            ("stage2Epochs", self.stage2_epochs),
            ("stage3RampEpochs", self.stage3_ramp_epochs),
            ("maxTopology", self.max_topology),
            ("maxBackoffs", self.max_backoffs),
            ("maxOptimizerSwitches", self.max_optimizer_switches),
            ("checkpointRetention", self.checkpoint_retention),
            ("maxTotalEpochs", self.max_total_epochs),
            ("progressEmitEveryEpochs", self.progress_emit_every_epochs),
            ("networkEmitEveryEpochs", self.network_emit_every_epochs),
            ("checkpointEveryEpochs", self.checkpoint_every_epochs),
        ] {
            if let Some(value) = value {
                if value == 0 && name != "checkpointEveryEpochs" {
                    return Err(format!("{name} must be positive when provided."));
                }
            }
        }

        if matches!(self.checkpoint_retention, Some(0)) {
            return Err("checkpointRetention must be positive when provided.".to_string());
        }
        if matches!(self.max_topology, Some(v) if v < 2) {
            return Err("maxTopology must be at least 2 when provided.".to_string());
        }
        if matches!(self.max_backoffs, Some(0)) || matches!(self.max_optimizer_switches, Some(0)) {
            return Err("Backoff and optimizer-switch limits must be positive when provided.".to_string());
        }
        if matches!(self.progress_emit_every_epochs, Some(0))
            || matches!(self.network_emit_every_epochs, Some(0))
        {
            return Err("Progress and network emit cadences must be positive when provided.".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_solve_input_stays_within_dense_ceiling() {
        let input = SolveInput::default();
        assert_eq!(input.mesh.max_dofs, MAX_DENSE_SOLVER_DOFS);
        assert!(input.validate().is_ok());
    }

    #[test]
    fn oversize_dense_mesh_is_rejected_by_validation() {
        let mut input = SolveInput::default();
        input.mesh.max_dofs = MAX_DENSE_SOLVER_DOFS + 1;
        assert!(input.validate().is_err());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkNodeSnapshot {
    pub id: String,
    pub layer: usize,
    pub index: usize,
    pub activation: f64,
    pub bias: f64,
    pub importance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkConnectionSnapshot {
    pub from_id: String,
    pub to_id: String,
    pub weight: f64,
    pub magnitude: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkSnapshot {
    pub layer_sizes: Vec<usize>,
    pub nodes: Vec<NetworkNodeSnapshot>,
    pub connections: Vec<NetworkConnectionSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingProgressEvent {
    pub epoch: usize,
    pub total_epochs: usize,
    pub loss: f64,
    pub val_loss: f64,
    #[serde(default)]
    pub data_loss: f64,
    #[serde(default)]
    pub physics_loss: f64,
    #[serde(default)]
    pub val_data_loss: f64,
    #[serde(default)]
    pub val_physics_loss: f64,
    #[serde(default)]
    pub momentum_residual: f64,
    #[serde(default)]
    pub kinematic_residual: f64,
    #[serde(default)]
    pub material_residual: f64,
    #[serde(default)]
    pub boundary_residual: f64,
    #[serde(default)]
    pub displacement_fit: f64,
    #[serde(default)]
    pub stress_fit: f64,
    #[serde(default)]
    pub invariant_residual: f64,
    #[serde(default)]
    pub constitutive_normal_residual: f64,
    #[serde(default)]
    pub constitutive_shear_residual: f64,
    #[serde(default)]
    pub val_displacement_fit: f64,
    #[serde(default)]
    pub val_stress_fit: f64,
    #[serde(default)]
    pub val_invariant_residual: f64,
    #[serde(default)]
    pub val_constitutive_normal_residual: f64,
    #[serde(default)]
    pub val_constitutive_shear_residual: f64,
    #[serde(default)]
    pub hybrid_mode: String,
    #[serde(default)]
    pub stage_id: String,
    #[serde(default)]
    pub optimizer_id: String,
    #[serde(default)]
    pub lr_phase: String,
    #[serde(default)]
    pub target_band_low: f64,
    #[serde(default)]
    pub target_band_high: f64,
    #[serde(default)]
    pub trend_slope: f64,
    #[serde(default)]
    pub trend_variance: f64,
    #[serde(default)]
    pub watchdog_trigger_count: usize,
    #[serde(default)]
    pub collocation_samples_added: usize,
    #[serde(default)]
    pub train_data_size: usize,
    #[serde(default)]
    pub train_data_cap: usize,
    #[serde(default)]
    pub residual_weight_momentum: f64,
    #[serde(default)]
    pub residual_weight_kinematics: f64,
    #[serde(default)]
    pub residual_weight_material: f64,
    #[serde(default)]
    pub residual_weight_boundary: f64,
    pub learning_rate: f64,
    pub architecture: Vec<usize>,
    pub progress_ratio: f64,
    #[serde(default)]
    pub training_mode: String,
    #[serde(default)]
    pub benchmark_id: Option<String>,
    #[serde(default)]
    pub gate_status: String,
    #[serde(default)]
    pub certified_best_metric: f64,
    #[serde(default)]
    pub dominant_blocker: Option<String>,
    #[serde(default)]
    pub stalled_reason: Option<String>,
    pub network: NetworkSnapshot,
    #[serde(default)]
    pub pino: Option<PinoRuntimeMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkCertification {
    pub status: String,
    pub summary: String,
    pub suggested_target_loss: f64,
    #[serde(default)]
    pub tip_displacement_relative_error: Option<f64>,
    #[serde(default)]
    pub max_displacement_relative_error: Option<f64>,
    #[serde(default)]
    pub mean_von_mises_relative_error: Option<f64>,
    #[serde(default)]
    pub max_sigma_xx_relative_error: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingTickEvent {
    pub epoch: usize,
    pub total_epochs: usize,
    pub loss: f64,
    pub val_loss: f64,
    pub learning_rate: f64,
    pub architecture: Vec<usize>,
    pub progress_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingRunStatus {
    pub running: bool,
    pub stop_requested: bool,
    pub completed: bool,
    pub last_result: Option<TrainResult>,
    pub last_error: Option<String>,
    pub diagnostics: TrainingDiagnostics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDiagnostics {
    pub best_val_loss: f64,
    pub epochs_since_improvement: usize,
    pub lr_schedule_phase: String,
    pub current_learning_rate: f64,
    pub data_weight: f64,
    pub physics_weight: f64,
    #[serde(default)]
    pub residual_weight_momentum: f64,
    #[serde(default)]
    pub residual_weight_kinematics: f64,
    #[serde(default)]
    pub residual_weight_material: f64,
    #[serde(default)]
    pub residual_weight_boundary: f64,
    pub active_learning_rounds: usize,
    pub active_learning_samples_added: usize,
    pub safeguard_triggers: usize,
    pub curriculum_backoffs: usize,
    pub optimizer_switches: usize,
    pub checkpoint_rollbacks: usize,
    pub target_floor_estimate: f64,
    pub trend_stop_reason: String,
    pub active_stage: String,
    pub active_optimizer: String,
    pub bo_presearch_used: bool,
    pub bo_selected_architecture: Vec<usize>,
    #[serde(default)]
    pub momentum_residual: f64,
    #[serde(default)]
    pub kinematic_residual: f64,
    #[serde(default)]
    pub material_residual: f64,
    #[serde(default)]
    pub boundary_residual: f64,
    #[serde(default)]
    pub displacement_fit: f64,
    #[serde(default)]
    pub stress_fit: f64,
    #[serde(default)]
    pub invariant_residual: f64,
    #[serde(default)]
    pub constitutive_normal_residual: f64,
    #[serde(default)]
    pub constitutive_shear_residual: f64,
    #[serde(default)]
    pub val_displacement_fit: f64,
    #[serde(default)]
    pub val_stress_fit: f64,
    #[serde(default)]
    pub val_invariant_residual: f64,
    #[serde(default)]
    pub val_constitutive_normal_residual: f64,
    #[serde(default)]
    pub val_constitutive_shear_residual: f64,
    #[serde(default)]
    pub hybrid_mode: String,
    #[serde(default)]
    pub collocation_points: usize,
    #[serde(default)]
    pub boundary_points: usize,
    #[serde(default)]
    pub interface_points: usize,
    #[serde(default)]
    pub collocation_samples_added: usize,
    #[serde(default)]
    pub train_data_size: usize,
    #[serde(default)]
    pub train_data_cap: usize,
    #[serde(default)]
    pub training_mode: String,
    #[serde(default)]
    pub benchmark_id: Option<String>,
    #[serde(default)]
    pub gate_status: String,
    #[serde(default)]
    pub certified_best_metric: f64,
    #[serde(default)]
    pub reproducibility_spread: Option<f64>,
    #[serde(default)]
    pub dominant_blocker: Option<String>,
    #[serde(default)]
    pub stalled_reason: Option<String>,
    #[serde(default)]
    pub benchmark_certification: Option<BenchmarkCertification>,
    #[serde(default)]
    pub run_budget_used: usize,
    #[serde(default)]
    pub run_budget_total: usize,
    #[serde(default)]
    pub recent_events: Vec<String>,
    #[serde(default)]
    pub pino: Option<PinoRuntimeMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingCheckpointInfo {
    pub id: String,
    pub tag: String,
    pub path: String,
    pub created_epoch: usize,
    pub model_version: u64,
    pub best_val_loss: f64,
    pub is_best: bool,
    pub created_at_unix_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointSaveInput {
    pub tag: Option<String>,
    pub mark_best: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointRetentionPolicy {
    pub keep_last: usize,
    pub keep_best: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResumeTrainingResult {
    pub checkpoint: TrainingCheckpointInfo,
    pub model_status: ModelStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PurgeCheckpointsResult {
    pub removed: usize,
    pub kept: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainResult {
    pub model_version: u64,
    pub loss: f64,
    pub val_loss: f64,
    pub architecture: Vec<usize>,
    pub learning_rate: f64,
    pub grew: bool,
    pub pruned: bool,
    pub completed_epochs: usize,
    pub reached_target: bool,
    pub reached_target_loss: bool,
    pub reached_autonomous_convergence: bool,
    pub stop_reason: String,
    pub notes: Vec<String>,
    #[serde(default)]
    pub training_mode: Option<String>,
    #[serde(default)]
    pub benchmark_id: Option<String>,
    #[serde(default)]
    pub gate_status: Option<String>,
    #[serde(default)]
    pub certified_best_metric: Option<f64>,
    #[serde(default)]
    pub reproducibility_spread: Option<f64>,
    #[serde(default)]
    pub dominant_blocker: Option<String>,
    #[serde(default)]
    pub stalled_reason: Option<String>,
    #[serde(default)]
    pub benchmark_certification: Option<BenchmarkCertification>,
    #[serde(default)]
    pub pino: Option<PinoRuntimeMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnnResult {
    pub fem_like: FemResult,
    pub confidence: f64,
    pub uncertainty: f64,
    pub model_version: u64,
    pub used_fem_fallback: bool,
    pub fallback_reason: Option<String>,
    #[serde(default)]
    pub domain_extrapolation_score: f64,
    pub residual_score: f64,
    pub uncertainty_threshold: f64,
    pub residual_threshold: f64,
    pub diagnostics: Vec<String>,
    #[serde(default)]
    pub surrogate_domain: Option<SurrogateDomainSummary>,
    #[serde(default)]
    pub pino: Option<PinoRuntimeMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    pub model_version: u64,
    pub architecture: Vec<usize>,
    pub learning_rate: f64,
    pub last_loss: f64,
    pub train_samples: usize,
    pub audit_frequency: usize,
    pub fallback_enabled: bool,
    pub safeguard_settings: SafeguardSettings,
    #[serde(default)]
    pub surrogate_domain: Option<SurrogateDomainSummary>,
    #[serde(default)]
    pub pino: Option<PinoRuntimeMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SafeguardSettings {
    pub preset: String,
    pub uncertainty_threshold: f64,
    pub residual_threshold: f64,
    pub adaptive_by_geometry: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeFingerprint {
    pub app_version: String,
    pub build_profile: String,
    pub target_os: String,
    pub target_arch: String,
    pub debug_build: bool,
    pub git_commit: String,
    pub build_time_utc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingBenchmarkManifest {
    pub id: String,
    pub title: String,
    pub description: String,
    pub training_mode: String,
    pub analysis_type: String,
    pub gate_name: String,
    pub gate_target_loss: f64,
    pub recommended_learning_rate: f64,
    pub max_runtime_seconds: usize,
    pub recommended_epochs: usize,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReportInput {
    pub path: String,
    pub format: String,
    pub solve_input: SolveInput,
    pub fem_result: Option<FemResult>,
    pub ann_result: Option<AnnResult>,
    pub dynamic_result: Option<DynamicResult>,
    pub thermal_result: Option<ThermalResult>,
    pub failure_result: Option<FailureResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    pub path: String,
    pub bytes_written: usize,
    pub format: String,
}
