use crate::contracts::{
    BenchmarkCertification, SolveInput, TrainingBatch, TrainingBenchmarkManifest,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct BenchmarkCertificationInput {
    pub displacement_fit: f64,
    pub stress_fit: f64,
    pub observable: f64,
    pub equilibrium: f64,
    pub constitutive_normal: f64,
    pub constitutive_shear: f64,
    pub weak_energy: f64,
    pub boundary: f64,
    pub tip_displacement_relative_error: Option<f64>,
    pub max_displacement_relative_error: Option<f64>,
    pub mean_von_mises_relative_error: Option<f64>,
    pub max_sigma_xx_relative_error: Option<f64>,
}

pub fn list_training_benchmarks() -> Vec<TrainingBenchmarkManifest> {
    vec![
        TrainingBenchmarkManifest {
            id: "benchmark_bar_1d".to_string(),
            title: "1D Bar".to_string(),
            description: "Displacement-primary axial bar sanity benchmark with exact closed-form response.".to_string(),
            training_mode: "benchmark".to_string(),
            analysis_type: "general".to_string(),
            gate_name: "Gate 1".to_string(),
            gate_target_loss: 1e-3,
            recommended_learning_rate: 5e-4,
            max_runtime_seconds: 45,
            recommended_epochs: 256,
            active: true,
        },
        TrainingBenchmarkManifest {
            id: "benchmark_cantilever_2d".to_string(),
            title: "2D Cantilever".to_string(),
            description: "Isolated cantilever benchmark for exact displacement-primary convergence before harder families.".to_string(),
            training_mode: "benchmark".to_string(),
            analysis_type: "cantilever".to_string(),
            gate_name: "Gate 3".to_string(),
            gate_target_loss: 1e-4,
            recommended_learning_rate: 5e-4,
            max_runtime_seconds: 180,
            recommended_epochs: 1024,
            active: true,
        },
        TrainingBenchmarkManifest {
            id: "benchmark_patch_test_2d".to_string(),
            title: "2D Patch Test".to_string(),
            description: "Linear elasticity patch test used to verify exact reproduction and residual consistency.".to_string(),
            training_mode: "benchmark".to_string(),
            analysis_type: "general".to_string(),
            gate_name: "Gate 4".to_string(),
            gate_target_loss: 1e-4,
            recommended_learning_rate: 4e-4,
            max_runtime_seconds: 180,
            recommended_epochs: 1024,
            active: true,
        },
        TrainingBenchmarkManifest {
            id: "benchmark_plate_hole_2d".to_string(),
            title: "Plate With Hole".to_string(),
            description: "Plate-with-hole benchmark promoted only after simpler benchmark gates are stable.".to_string(),
            training_mode: "benchmark".to_string(),
            analysis_type: "plate-hole".to_string(),
            gate_name: "Gate 4".to_string(),
            gate_target_loss: 1e-2,
            recommended_learning_rate: 3.5e-4,
            max_runtime_seconds: 240,
            recommended_epochs: 1536,
            active: true,
        },
    ]
}

pub fn get_training_benchmark(id: &str) -> Option<TrainingBenchmarkManifest> {
    list_training_benchmarks()
        .into_iter()
        .find(|benchmark| benchmark.id == id)
}

pub fn certify_training_benchmark(
    id: &str,
    certified_best_metric: f64,
    breakdown: Option<BenchmarkCertificationInput>,
) -> Option<BenchmarkCertification> {
    let benchmark = get_training_benchmark(id)?;
    if !certified_best_metric.is_finite() || certified_best_metric <= 0.0 {
        return Some(BenchmarkCertification {
            status: "insufficient".to_string(),
            summary: format!(
                "{} did not produce a finite benchmark metric, so certification is unavailable.",
                benchmark.title
            ),
            suggested_target_loss: benchmark.gate_target_loss,
            tip_displacement_relative_error: None,
            max_displacement_relative_error: None,
            mean_von_mises_relative_error: None,
            max_sigma_xx_relative_error: None,
        });
    }
    if certified_best_metric <= benchmark.gate_target_loss {
        return Some(BenchmarkCertification {
            status: "gate-pass".to_string(),
            summary: format!(
                "{} cleared the formal benchmark gate at {:.3e}.",
                benchmark.title, benchmark.gate_target_loss
            ),
            suggested_target_loss: benchmark.gate_target_loss,
            tip_displacement_relative_error: breakdown.and_then(|b| b.tip_displacement_relative_error),
            max_displacement_relative_error: breakdown.and_then(|b| b.max_displacement_relative_error),
            mean_von_mises_relative_error: breakdown.and_then(|b| b.mean_von_mises_relative_error),
            max_sigma_xx_relative_error: breakdown.and_then(|b| b.max_sigma_xx_relative_error),
        });
    }
    match (id, breakdown) {
        ("benchmark_cantilever_2d", Some(b)) => {
            let development_target = 2.5e-2;
            let physical_band_pass = b
                .tip_displacement_relative_error
                .map(|value| value <= 5.0e-2)
                .unwrap_or(false)
                && b.max_displacement_relative_error
                    .map(|value| value <= 5.0e-2)
                    .unwrap_or(false)
                && b.mean_von_mises_relative_error
                    .map(|value| value <= 1.0e-1)
                    .unwrap_or(false)
                && b.max_sigma_xx_relative_error
                    .map(|value| value <= 1.0e-1)
                    .unwrap_or(false);
            let development_pass = certified_best_metric <= development_target
                && b.displacement_fit <= 5.0e-2
                && b.equilibrium <= 5.0e-2
                && b.boundary <= 1.0e-3
                && b.observable <= 4.0e-1
                && b.constitutive_normal <= 5.0e-1
                && b.constitutive_shear <= 1.0e-1
                && b.weak_energy <= 1.0e-3
                && b.stress_fit <= 2.5
                && physical_band_pass;
            if development_pass {
                Some(BenchmarkCertification {
                    status: "development-pass".to_string(),
                    summary: format!(
                        "{} missed the formal {:.1e} gate, but the current floor {:.3e} stays inside the benchmark development band. Physical errors: tip {:.3e}, max-disp {:.3e}, mean-vm {:.3e}, max-sxx {:.3e}. Treat this as a practical benchmark floor, not engineering certification in physical units.",
                        benchmark.title,
                        benchmark.gate_target_loss,
                        certified_best_metric,
                        b.tip_displacement_relative_error.unwrap_or(f64::INFINITY),
                        b.max_displacement_relative_error.unwrap_or(f64::INFINITY),
                        b.mean_von_mises_relative_error.unwrap_or(f64::INFINITY),
                        b.max_sigma_xx_relative_error.unwrap_or(f64::INFINITY),
                    ),
                    suggested_target_loss: development_target,
                    tip_displacement_relative_error: b.tip_displacement_relative_error,
                    max_displacement_relative_error: b.max_displacement_relative_error,
                    mean_von_mises_relative_error: b.mean_von_mises_relative_error,
                    max_sigma_xx_relative_error: b.max_sigma_xx_relative_error,
                })
            } else {
                Some(BenchmarkCertification {
                    status: "insufficient".to_string(),
                    summary: format!(
                        "{} is still outside the benchmark development band at {:.3e}; physical errors tip {:.3e}, max-disp {:.3e}, mean-vm {:.3e}, max-sxx {:.3e}; residual blockers stress {:.3e}, cn {:.3e}, obs {:.3e}, eq {:.3e}.",
                        benchmark.title,
                        certified_best_metric,
                        b.tip_displacement_relative_error.unwrap_or(f64::INFINITY),
                        b.max_displacement_relative_error.unwrap_or(f64::INFINITY),
                        b.mean_von_mises_relative_error.unwrap_or(f64::INFINITY),
                        b.max_sigma_xx_relative_error.unwrap_or(f64::INFINITY),
                        b.stress_fit,
                        b.constitutive_normal,
                        b.observable,
                        b.equilibrium,
                    ),
                    suggested_target_loss: benchmark.gate_target_loss,
                    tip_displacement_relative_error: b.tip_displacement_relative_error,
                    max_displacement_relative_error: b.max_displacement_relative_error,
                    mean_von_mises_relative_error: b.mean_von_mises_relative_error,
                    max_sigma_xx_relative_error: b.max_sigma_xx_relative_error,
                })
            }
        }
        _ => Some(BenchmarkCertification {
            status: "insufficient".to_string(),
            summary: format!(
                "{} has not cleared its formal benchmark gate of {:.3e}.",
                benchmark.title, benchmark.gate_target_loss
            ),
            suggested_target_loss: benchmark.gate_target_loss,
            tip_displacement_relative_error: breakdown.and_then(|b| b.tip_displacement_relative_error),
            max_displacement_relative_error: breakdown.and_then(|b| b.max_displacement_relative_error),
            mean_von_mises_relative_error: breakdown.and_then(|b| b.mean_von_mises_relative_error),
            max_sigma_xx_relative_error: breakdown.and_then(|b| b.max_sigma_xx_relative_error),
        }),
    }
}

fn benchmark_seed_cases(id: &str) -> Option<Vec<SolveInput>> {
    let mut case = SolveInput::default();
    case.mesh.auto_adapt = false;
    case.mesh.amr_enabled = false;
    case.mesh.max_dofs = crate::contracts::MAX_DENSE_SOLVER_DOFS.min(2_400);
    match id {
        "benchmark_bar_1d" => {
            case.geometry.length_in = 12.0;
            case.geometry.width_in = 1.0;
            case.geometry.thickness_in = 0.25;
            case.geometry.hole_diameter_in = Some(0.0);
            case.mesh.nx = 10;
            case.mesh.ny = 2;
            case.mesh.nz = 1;
            case.load.axial_load_lbf = 250.0;
            case.load.vertical_point_load_lbf = 0.0;
            case.boundary_conditions.fix_start_face = true;
            case.boundary_conditions.fix_end_face = false;
            Some(vec![case])
        }
        "benchmark_cantilever_2d" => {
            case.geometry.length_in = 10.0;
            case.geometry.width_in = 1.0;
            case.geometry.thickness_in = 0.25;
            case.geometry.hole_diameter_in = Some(0.0);
            case.mesh.nx = 16;
            case.mesh.ny = 12;
            case.mesh.nz = 1;
            case.load.axial_load_lbf = 0.0;
            case.load.vertical_point_load_lbf = -100.0;
            case.boundary_conditions.fix_start_face = true;
            case.boundary_conditions.fix_end_face = false;
            Some(vec![case])
        }
        "benchmark_patch_test_2d" => {
            case.geometry.length_in = 8.0;
            case.geometry.width_in = 4.0;
            case.geometry.thickness_in = 0.25;
            case.geometry.hole_diameter_in = Some(0.0);
            case.mesh.nx = 8;
            case.mesh.ny = 4;
            case.mesh.nz = 1;
            case.load.axial_load_lbf = 400.0;
            case.load.vertical_point_load_lbf = 0.0;
            case.boundary_conditions.fix_start_face = true;
            case.boundary_conditions.fix_end_face = false;
            Some(vec![case])
        }
        "benchmark_plate_hole_2d" => {
            case.geometry.length_in = 11.811;
            case.geometry.width_in = 4.724;
            case.geometry.thickness_in = 0.25;
            case.geometry.hole_diameter_in = Some(2.362);
            case.mesh.nx = 20;
            case.mesh.ny = 10;
            case.mesh.nz = 1;
            case.load.axial_load_lbf = 500.0;
            case.load.vertical_point_load_lbf = 0.0;
            case.boundary_conditions.fix_start_face = true;
            case.boundary_conditions.fix_end_face = false;
            Some(vec![case])
        }
        _ => None,
    }
}

pub fn apply_training_benchmark(mut batch: TrainingBatch) -> Result<TrainingBatch, String> {
    let Some(benchmark_id) = batch.benchmark_id.clone() else {
        return Ok(batch);
    };
    let benchmark = get_training_benchmark(&benchmark_id).ok_or_else(|| {
        format!(
            "Unknown benchmarkId '{benchmark_id}'. Call listTrainingBenchmarks for supported benchmark profiles."
        )
    })?;
    let cases = benchmark_seed_cases(&benchmark_id).ok_or_else(|| {
        format!("Benchmark '{benchmark_id}' does not yet have canonical seed cases.")
    })?;
    batch.training_mode = Some("benchmark".to_string());
    batch.analysis_type = Some(benchmark.analysis_type);
    batch.cases = cases;
    if !batch.target_loss.is_finite() || batch.target_loss <= 0.0 {
        batch.target_loss = benchmark.gate_target_loss;
    }
    if batch.learning_rate.is_none() {
        batch.learning_rate = Some(benchmark.recommended_learning_rate);
    }
    if batch.max_total_epochs.is_none() {
        batch.max_total_epochs = Some(benchmark.recommended_epochs);
    }
    batch.auto_mode = Some(true);
    batch.autonomous_mode = Some(true);
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_training_benchmark_normalizes_cantilever_profile() {
        let batch = TrainingBatch {
            cases: vec![SolveInput::default()],
            epochs: 12,
            target_loss: 0.0,
            training_mode: Some("benchmark".to_string()),
            benchmark_id: Some("benchmark_cantilever_2d".to_string()),
            seed: Some(42),
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

        let normalized = apply_training_benchmark(batch).expect("benchmark normalization");
        assert_eq!(normalized.training_mode.as_deref(), Some("benchmark"));
        assert_eq!(normalized.analysis_type.as_deref(), Some("cantilever"));
        assert_eq!(normalized.benchmark_id.as_deref(), Some("benchmark_cantilever_2d"));
        assert_eq!(normalized.target_loss, 1e-4);
        assert_eq!(normalized.learning_rate, Some(5e-4));
        assert_eq!(normalized.max_total_epochs, Some(1024));
        assert_eq!(normalized.cases.len(), 1);
        assert_eq!(normalized.cases[0].load.vertical_point_load_lbf, -100.0);
        assert_eq!(normalized.cases[0].load.axial_load_lbf, 0.0);
        assert_eq!(normalized.cases[0].mesh.nx, 16);
        assert_eq!(normalized.cases[0].mesh.ny, 12);
    }

    #[test]
    fn cantilever_certification_marks_development_pass_floor() {
        let certification = certify_training_benchmark(
            "benchmark_cantilever_2d",
            2.000981e-2,
            Some(BenchmarkCertificationInput {
                displacement_fit: 2.529541e-2,
                stress_fit: 1.896285,
                observable: 3.344716e-1,
                equilibrium: 2.916663e-2,
                constitutive_normal: 3.372455e-1,
                constitutive_shear: 4.863068e-2,
                weak_energy: 9.742255e-5,
                boundary: 2.549258e-5,
                tip_displacement_relative_error: Some(2.0e-2),
                max_displacement_relative_error: Some(3.0e-2),
                mean_von_mises_relative_error: Some(8.0e-2),
                max_sigma_xx_relative_error: Some(9.0e-2),
            }),
        )
        .expect("cantilever certification");
        assert_eq!(certification.status, "development-pass");
        assert_eq!(certification.suggested_target_loss, 2.5e-2);
    }
}
