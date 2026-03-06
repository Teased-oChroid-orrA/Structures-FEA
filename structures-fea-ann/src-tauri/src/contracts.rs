use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometryInput {
    pub length_in: f64,
    pub width_in: f64,
    pub thickness_in: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeshControls {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub element_type: String,
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
            e_psi: 10_000_000.0,
            nu: 0.33,
            rho_lb_in3: 0.0975,
            alpha_per_f: 13.0e-6,
            yield_strength_psi: 40_000.0,
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
                length_in: 10.0,
                width_in: 4.0,
                thickness_in: 0.125,
            },
            mesh: MeshControls {
                nx: 10,
                ny: 4,
                nz: 1,
                element_type: "hex8".to_string(),
            },
            material: Material::default(),
            boundary_conditions: BoundaryConditionInput {
                fix_start_face: true,
                fix_end_face: false,
            },
            load: LoadInput {
                axial_load_lbf: 0.0,
                vertical_point_load_lbf: -1000.0,
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
    pub learning_rate: Option<f64>,
    pub auto_mode: Option<bool>,
    pub max_total_epochs: Option<usize>,
    pub min_improvement: Option<f64>,
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
    pub learning_rate: f64,
    pub architecture: Vec<usize>,
    pub progress_ratio: f64,
    pub network: NetworkSnapshot,
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
    pub stop_reason: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnnResult {
    pub fem_like: FemResult,
    pub confidence: f64,
    pub uncertainty: f64,
    pub model_version: u64,
    pub used_fem_fallback: bool,
    pub diagnostics: Vec<String>,
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
