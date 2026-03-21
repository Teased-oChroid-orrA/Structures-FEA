use crate::contracts::{ThermalInput, ThermalResult};

use super::stress::principal_stresses;

pub fn run(input: &ThermalInput, base_stress_tensor: [[f64; 3]; 3]) -> ThermalResult {
    let m = &input.solve_input.material;
    let eps_th = m.alpha_per_f * input.delta_t_f;
    let sigma_th = if input.restrained_x {
        m.e_psi * eps_th
    } else {
        0.0
    };

    let mut combined = base_stress_tensor;
    combined[0][0] += sigma_th;

    ThermalResult {
        thermal_strain_x: eps_th,
        thermal_stress_psi: sigma_th,
        combined_stress_tensor: combined,
        principal_stresses: principal_stresses(combined),
        diagnostics: vec![
            format!("Thermal strain eps_th = {:.6e}", eps_th),
            format!("Thermal stress sigma_th = {:.3} psi", sigma_th),
            "Thermal model assumes isotropic alpha and uniform deltaT.".to_string(),
        ],
    }
}
