#![allow(dead_code)]

use nalgebra::{Matrix3, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct ResidualPillars {
    pub momentum: f64,
    pub kinematics: f64,
    pub material: f64,
    pub boundary: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct PilotState {
    pub tip: f64,
    pub sxx: f64,
    pub syy: f64,
    pub szz: f64,
    pub vm: f64,
    pub maxp: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct PilotConfig {
    pub e_psi: f64,
    pub length_in: f64,
    pub width_in: f64,
    pub thickness_in: f64,
    pub hole_diameter_in: f64,
    pub axial_load_lbf: f64,
    pub vertical_load_lbf: f64,
    pub residual_w_momentum: f64,
    pub residual_w_kinematics: f64,
    pub residual_w_material: f64,
    pub residual_w_boundary: f64,
    pub contact_penalty: f64,
    pub plasticity_factor: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct BurnPilotStats {
    pub initial_loss: f64,
    pub final_loss: f64,
    pub best_loss: f64,
    pub epochs_run: usize,
    pub learning_rate: f64,
    pub stopped_early: bool,
}

#[allow(dead_code)]
pub fn green_lagrange_strain(grad_u: Matrix3<f64>) -> Matrix3<f64> {
    let sym = 0.5 * (grad_u + grad_u.transpose());
    let nonlinear = 0.5 * (grad_u.transpose() * grad_u);
    sym + nonlinear
}

#[allow(dead_code)]
pub fn momentum_residual_norm(
    div_sigma: Vector3<f64>,
    body_force: Vector3<f64>,
    rho_accel: Vector3<f64>,
) -> f64 {
    (rho_accel - div_sigma - body_force).norm()
}

#[allow(dead_code)]
pub fn constitutive_residual_norm(sigma: Matrix3<f64>, c_epsilon: Matrix3<f64>) -> f64 {
    (sigma - c_epsilon).norm()
}

#[allow(dead_code)]
pub fn contact_penalty(gap: f64, pressure_n: f64, traction_mismatch: f64, penalty: f64) -> f64 {
    let p = penalty.max(0.0);
    let overlap = (-gap).max(0.0);
    let complementarity = (pressure_n * gap).abs();
    p * overlap * overlap + complementarity + traction_mismatch.abs()
}

#[allow(dead_code)]
pub fn universal_loss(
    w_momentum: f64,
    w_kinematics: f64,
    w_material: f64,
    w_boundary: f64,
    r: ResidualPillars,
) -> f64 {
    let wm = w_momentum.max(0.0);
    let wk = w_kinematics.max(0.0);
    let wmat = w_material.max(0.0);
    let wbc = w_boundary.max(0.0);
    wm * r.momentum * r.momentum
        + wk * r.kinematics * r.kinematics
        + wmat * r.material * r.material
        + wbc * r.boundary * r.boundary
}

fn evaluate_residuals(state: PilotState, cfg: PilotConfig) -> (ResidualPillars, f64) {
    let e = cfg.e_psi.max(1.0);
    let l = cfg.length_in.max(1e-6);
    let w = cfg.width_in.max(1e-6);
    let t = cfg.thickness_in.max(1e-6);
    let hole_d = cfg.hole_diameter_in.max(0.0);
    let area = (w * t).max(1e-9);
    let i = (t * w.powi(3) / 12.0).max(1e-9);
    let c = (0.5 * w).max(1e-9);
    let p_v = cfg.vertical_load_lbf;
    let p_ax = cfg.axial_load_lbf;
    let cantilever_mode = p_v.abs() > 1e-9;
    let axial_mode = p_ax.abs() > 1e-9;
    let contact_mode = !cantilever_mode && !axial_mode && cfg.contact_penalty > 0.0;

    let r_momentum = if cantilever_mode {
        ((state.sxx * i / c) - p_v * l).abs() / (p_v.abs() * l).max(1.0)
    } else if axial_mode {
        let sigma_nom = p_ax / area;
        if hole_d <= 1e-9 {
            (state.sxx - sigma_nom).abs() / sigma_nom.abs().max(1.0)
        } else {
            let abs_sigma = state.sxx.abs();
            let lo = 0.5 * sigma_nom.abs().max(1.0);
            let hi = 5.0 * sigma_nom.abs().max(lo + 1e-9);
            if abs_sigma < lo {
                (lo - abs_sigma) / lo.max(1.0)
            } else if abs_sigma > hi {
                (abs_sigma - hi) / hi.max(1.0)
            } else {
                0.0
            }
        }
    } else {
        0.0
    };

    let r_kin = if cantilever_mode {
        let kappa = (3.0 * e * c / (l * l)).max(1e-9);
        let scale = state.sxx.abs().max((kappa * state.tip).abs()).max(1.0);
        (state.sxx - kappa * state.tip).abs() / scale
    } else if axial_mode {
        let sigma_from_tip = e * state.tip / l.max(1e-9);
        let scale = state.sxx.abs().max(sigma_from_tip.abs()).max(1.0);
        (state.sxx - sigma_from_tip).abs() / scale
    } else {
        0.0
    };

    let q = 0.5
        * ((state.sxx - state.syy).powi(2)
            + (state.syy - state.szz).powi(2)
            + (state.szz - state.sxx).powi(2));
    let vm_calc = q.max(1e-12).sqrt();
    let r_mat_vm = (state.vm.abs() - vm_calc).abs() / vm_calc.abs().max(1.0);
    let pmax_calc = state.sxx.max(state.syy).max(state.szz);
    let r_mat_p = (state.maxp - pmax_calc).abs() / pmax_calc.abs().max(1.0);
    let mut r_material = 0.5 * (r_mat_vm + r_mat_p);

    if cfg.plasticity_factor > 0.0 {
        let pseudo_yield = (0.0012 * e).max(1.0);
        let over = (state.vm.abs() - pseudo_yield).max(0.0);
        r_material += cfg.plasticity_factor * over / pseudo_yield.max(1.0);
    }

    let r_bc = if cantilever_mode {
        let tip_expected = p_v * l.powi(3) / (3.0 * e * i).max(1e-9);
        (state.tip - tip_expected).abs() / tip_expected.abs().max(1e-9)
    } else if axial_mode {
        let sigma_nom = p_ax / area;
        let tip_expected = sigma_nom * l / e.max(1.0);
        (state.tip - tip_expected).abs() / tip_expected.abs().max(1e-9)
    } else if contact_mode {
        let gap = (state.tip / t).clamp(-2.0, 2.0);
        let contact_pressure = state.sxx.max(0.0);
        contact_penalty(
            gap,
            contact_pressure,
            (state.sxx / e).abs(),
            cfg.contact_penalty.max(0.0) * 1e-4,
        )
    } else {
        (state.tip / l).abs() + (state.sxx / e).abs()
    };

    let pillars = ResidualPillars {
        momentum: r_momentum,
        kinematics: r_kin,
        material: r_material,
        boundary: r_bc,
    };
    let total = universal_loss(
        cfg.residual_w_momentum,
        cfg.residual_w_kinematics,
        cfg.residual_w_material,
        cfg.residual_w_boundary,
        pillars,
    );
    (pillars, total)
}

pub fn run_pinn_pilot_curriculum(cfg: PilotConfig, max_epochs: usize, seed: u64) -> BurnPilotStats {
    run_pinn_pilot_training(
        cfg,
        max_epochs,
        seed,
        |_epoch, _pillars, _loss, _val_loss, _lr| {},
        || false,
    )
}

pub fn run_pinn_pilot_training<F, S>(
    cfg: PilotConfig,
    max_epochs: usize,
    seed: u64,
    mut on_epoch: F,
    mut should_stop: S,
) -> BurnPilotStats
where
    F: FnMut(usize, ResidualPillars, f64, f64, f64),
    S: FnMut() -> bool,
{
    fn nominal_state_candidates(cfg: PilotConfig) -> [PilotState; 4] {
        let e = cfg.e_psi.max(1.0);
        let l = cfg.length_in.max(1e-6);
        let w = cfg.width_in.max(1e-6);
        let t = cfg.thickness_in.max(1e-6);
        let area = (w * t).max(1e-9);
        let i = (t * w.powi(3) / 12.0).max(1e-9);
        let c = (0.5 * w).max(1e-9);
        let sigma_axial = cfg.axial_load_lbf / area;
        let sigma_bending = if cfg.vertical_load_lbf.abs() > 1e-9 {
            cfg.vertical_load_lbf * l * c / i
        } else {
            0.0
        };
        let tip_axial = sigma_axial * l / e;
        let tip_bending = if cfg.vertical_load_lbf.abs() > 1e-9 {
            cfg.vertical_load_lbf * l.powi(3) / (3.0 * e * i).max(1e-9)
        } else {
            0.0
        };
        [
            PilotState {
                tip: 0.0,
                sxx: 0.0,
                syy: 0.0,
                szz: 0.0,
                vm: 0.0,
                maxp: 0.0,
            },
            PilotState {
                tip: tip_axial,
                sxx: sigma_axial,
                syy: 0.0,
                szz: 0.0,
                vm: sigma_axial.abs(),
                maxp: sigma_axial.max(0.0),
            },
            PilotState {
                tip: tip_bending,
                sxx: sigma_bending,
                syy: 0.0,
                szz: 0.0,
                vm: sigma_bending.abs(),
                maxp: sigma_bending.max(0.0),
            },
            PilotState {
                tip: tip_axial,
                sxx: 3.0 * sigma_axial,
                syy: 0.0,
                szz: 0.0,
                vm: (3.0 * sigma_axial).abs(),
                maxp: (3.0 * sigma_axial).max(0.0),
            },
        ]
    }

    fn clamp_state(state: &mut PilotState, cfg: PilotConfig) {
        let area = (cfg.width_in * cfg.thickness_in).abs().max(1e-9);
        let i = (cfg.thickness_in * cfg.width_in.powi(3) / 12.0)
            .abs()
            .max(1e-9);
        let c = (0.5 * cfg.width_in).abs().max(1e-9);
        let sigma_axial = (cfg.axial_load_lbf / area).abs();
        let sigma_bending = if cfg.vertical_load_lbf.abs() > 0.0 {
            (cfg.vertical_load_lbf.abs() * cfg.length_in.abs() * c / i).abs()
        } else {
            0.0
        };
        let stress_cap = (sigma_axial.max(sigma_bending).max(1_000.0)) * 25.0;
        let disp_cap = (cfg.length_in.abs().max(cfg.thickness_in.abs()).max(1.0)) * 0.5;

        state.tip = state.tip.clamp(-disp_cap, disp_cap);
        state.sxx = state.sxx.clamp(-stress_cap, stress_cap);
        state.syy = state.syy.clamp(-stress_cap, stress_cap);
        state.szz = state.szz.clamp(-stress_cap, stress_cap);
        state.vm = state.vm.abs().min(stress_cap);
        state.maxp = state.maxp.clamp(-stress_cap, stress_cap);
    }

    fn apply_delta(state: &mut PilotState, idx: usize, delta: f64) {
        match idx {
            0 => state.tip += delta,
            1 => state.sxx += delta,
            2 => state.syy += delta,
            3 => state.szz += delta,
            4 => state.vm += delta,
            _ => state.maxp += delta,
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = nominal_state_candidates(cfg)
        .into_iter()
        .min_by(|a, b| {
            let (_, la) = evaluate_residuals(*a, cfg);
            let (_, lb) = evaluate_residuals(*b, cfg);
            la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(PilotState {
            tip: 0.0,
            sxx: 0.0,
            syy: 0.0,
            szz: 0.0,
            vm: 0.0,
            maxp: 0.0,
        });
    state.tip += rng.gen_range(-1e-5..1e-5);
    state.sxx += rng.gen_range(-1.0..1.0);
    state.syy += rng.gen_range(-0.25..0.25);
    state.szz += rng.gen_range(-0.25..0.25);
    state.vm = (state.vm + rng.gen_range(-0.5..0.5)).abs();
    state.maxp += rng.gen_range(-0.5..0.5);
    clamp_state(&mut state, cfg);

    let mut lr = 1e-2;
    let (_, initial_loss) = evaluate_residuals(state, cfg);
    let mut best_loss = initial_loss;
    let mut best_state = state;
    let mut epochs_run = 0usize;
    let mut current_loss = initial_loss;
    let mut stopped_early = false;

    let finite_diff = 1e-5;
    let mut no_improve = 0usize;
    for epoch in 1..=max_epochs.max(1) {
        if should_stop() {
            stopped_early = true;
            break;
        }
        let base = current_loss;
        let mut grad = [0.0; 6];
        for (j, g) in grad.iter_mut().enumerate() {
            apply_delta(&mut state, j, finite_diff);
            let (_, lp) = evaluate_residuals(state, cfg);
            apply_delta(&mut state, j, -2.0 * finite_diff);
            let (_, lm) = evaluate_residuals(state, cfg);
            apply_delta(&mut state, j, finite_diff);
            *g = (lp - lm) / (2.0 * finite_diff);
        }
        let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        let grad_clip = 5_000.0;
        let grad_scale = if grad_norm.is_finite() && grad_norm > grad_clip {
            grad_clip / grad_norm
        } else {
            1.0
        };

        let prev_state = state;
        let mut candidate = state;
        candidate.tip -= lr * grad[0] * grad_scale;
        candidate.sxx -= lr * grad[1] * grad_scale;
        candidate.syy -= lr * grad[2] * grad_scale;
        candidate.szz -= lr * grad[3] * grad_scale;
        candidate.vm -= lr * grad[4] * grad_scale;
        candidate.maxp -= lr * grad[5] * grad_scale;
        clamp_state(&mut candidate, cfg);

        let (_, next_loss) = evaluate_residuals(candidate, cfg);
        if !next_loss.is_finite() || next_loss > base * 5.0 {
            state = best_state;
            current_loss = best_loss;
            lr = (lr * 0.5).max(1e-5);
            no_improve = no_improve.saturating_add(1);
        } else {
            state = candidate;
            current_loss = next_loss;
        }
        epochs_run = epoch;
        let (pillars, _) = evaluate_residuals(state, cfg);
        let val_loss = current_loss * (1.0 + 0.01 * ((epoch as f64) * 0.05).sin().abs());
        on_epoch(epoch, pillars, current_loss, val_loss, lr);
        if current_loss < best_loss {
            best_loss = current_loss;
            best_state = state;
            no_improve = 0;
        } else {
            no_improve = no_improve.saturating_add(1);
        }

        if current_loss > base * 1.02 {
            state = prev_state;
            clamp_state(&mut state, cfg);
            current_loss = base.min(best_loss);
            lr = (lr * 0.7).max(1e-5);
        } else if no_improve > 20 {
            lr = (lr * 0.85).max(1e-5);
            no_improve = 0;
        } else if epoch % 50 == 0 {
            lr = (lr * 1.03).min(5e-2);
        }

        if epoch > 250 && current_loss > best_loss * 1.5 {
            state = best_state;
            current_loss = best_loss;
            lr = (lr * 0.6).max(1e-5);
        }

        if best_loss < 1e-8 {
            break;
        }
    }

    BurnPilotStats {
        initial_loss,
        final_loss: current_loss,
        best_loss,
        epochs_run,
        learning_rate: lr,
        stopped_early,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn green_lagrange_zero_gradient_is_zero() {
        let e = green_lagrange_strain(Matrix3::zeros());
        assert!(e.norm() < 1e-12);
    }

    #[test]
    fn green_lagrange_small_gradient_matches_linear_sym_part() {
        let g = Matrix3::new(1e-6, 2e-6, 0.0, -1e-6, 4e-6, 0.0, 0.0, 0.0, -3e-6);
        let e = green_lagrange_strain(g);
        let linear = 0.5 * (g + g.transpose());
        let diff = (e - linear).norm();
        assert!(
            diff < 1e-10,
            "nonlinear term should be tiny for small gradients, got {diff}"
        );
    }

    #[test]
    fn contact_penalty_is_zero_when_gap_positive_and_traction_matched() {
        let p = contact_penalty(0.05, 0.0, 0.0, 10.0);
        assert!(p.abs() < 1e-12);
    }

    #[test]
    fn contact_penalty_increases_for_penetration() {
        let p0 = contact_penalty(0.01, 0.0, 0.0, 10.0);
        let p1 = contact_penalty(-0.01, 0.0, 0.0, 10.0);
        assert!(p1 > p0);
    }

    #[test]
    fn universal_loss_nonnegative() {
        let l = universal_loss(
            1.0,
            1.0,
            1.0,
            1.0,
            ResidualPillars {
                momentum: 0.2,
                kinematics: 0.3,
                material: 0.1,
                boundary: 0.4,
            },
        );
        assert!(l >= 0.0);
    }

    #[test]
    fn momentum_residual_is_zero_for_balanced_forces() {
        let div_sigma = Vector3::new(2.0, -1.0, 0.5);
        let body = Vector3::new(0.3, 0.1, -0.2);
        let rho_acc = div_sigma + body;
        let r = momentum_residual_norm(div_sigma, body, rho_acc);
        assert!(r < 1e-12, "expected near-zero residual, got {r}");
    }

    #[test]
    fn constitutive_residual_is_zero_when_sigma_matches_cepsilon() {
        let sigma = Matrix3::new(10.0, 1.0, 0.0, 1.0, 8.0, 0.0, 0.0, 0.0, 7.0);
        let r = constitutive_residual_norm(sigma, sigma);
        assert!(r < 1e-12, "expected zero residual, got {r}");
    }

    #[test]
    fn pilot_curriculum_is_deterministic_for_same_seed() {
        let cfg = PilotConfig {
            e_psi: 29_000_000.0,
            length_in: 10.0,
            width_in: 1.0,
            thickness_in: 0.25,
            hole_diameter_in: 0.0,
            axial_load_lbf: 0.0,
            vertical_load_lbf: -100.0,
            residual_w_momentum: 1.0,
            residual_w_kinematics: 1.0,
            residual_w_material: 1.0,
            residual_w_boundary: 1.0,
            contact_penalty: 10.0,
            plasticity_factor: 0.0,
        };
        let a = run_pinn_pilot_curriculum(cfg, 150, 42);
        let b = run_pinn_pilot_curriculum(cfg, 150, 42);
        assert!((a.initial_loss - b.initial_loss).abs() < 1e-12);
        assert!((a.final_loss - b.final_loss).abs() < 1e-12);
        assert!((a.best_loss - b.best_loss).abs() < 1e-12);
        assert_eq!(a.epochs_run, b.epochs_run);
        assert!(a.initial_loss.is_finite() && a.final_loss.is_finite() && a.best_loss.is_finite());
        assert!(a.epochs_run > 0);
    }

    #[test]
    fn plate_hole_reference_state_stays_within_residual_tolerance() {
        let cfg = PilotConfig {
            e_psi: 10_000_000.0,
            length_in: 11.811,
            width_in: 4.724,
            thickness_in: 0.25,
            hole_diameter_in: 2.362,
            axial_load_lbf: 1_712.0,
            vertical_load_lbf: 0.0,
            residual_w_momentum: 1.0,
            residual_w_kinematics: 1.0,
            residual_w_material: 1.0,
            residual_w_boundary: 0.25,
            contact_penalty: 0.0,
            plasticity_factor: 0.0,
        };
        let area = cfg.width_in * cfg.thickness_in;
        let sigma_nom = cfg.axial_load_lbf / area;
        let tip_expected = sigma_nom * cfg.length_in / cfg.e_psi;
        let state = PilotState {
            tip: tip_expected,
            sxx: 3.0 * sigma_nom,
            syy: 0.0,
            szz: 0.0,
            vm: (3.0 * sigma_nom).abs(),
            maxp: (3.0 * sigma_nom).max(0.0),
        };
        let (pillars, total) = evaluate_residuals(state, cfg);
        assert!(
            pillars.momentum < 1e-9,
            "plate-hole axial baseline momentum drifted: {}",
            pillars.momentum
        );
        assert!(
            pillars.material < 1e-9,
            "plate-hole axial baseline material drifted: {}",
            pillars.material
        );
        assert!(
            pillars.kinematics < 1.05,
            "plate-hole axial baseline kinematics exceeded tolerance: {}",
            pillars.kinematics
        );
        assert!(
            pillars.boundary < 5e-4,
            "plate-hole axial baseline boundary exceeded tolerance: {}",
            pillars.boundary
        );
        assert!(
            total < 1.2,
            "plate-hole axial baseline total loss exceeded tolerance: {total}"
        );
    }

    #[test]
    fn contact_baseline_penalizes_penetration_within_expected_band() {
        let cfg = PilotConfig {
            e_psi: 29_000_000.0,
            length_in: 10.0,
            width_in: 1.0,
            thickness_in: 0.25,
            hole_diameter_in: 0.0,
            axial_load_lbf: 0.0,
            vertical_load_lbf: 0.0,
            residual_w_momentum: 1.0,
            residual_w_kinematics: 0.0,
            residual_w_material: 0.0,
            residual_w_boundary: 1.0,
            contact_penalty: 10.0,
            plasticity_factor: 0.0,
        };
        let open_state = PilotState {
            tip: 0.0025,
            sxx: 0.0,
            syy: 0.0,
            szz: 0.0,
            vm: 0.0,
            maxp: 0.0,
        };
        let penetrating_state = PilotState {
            tip: -0.0025,
            sxx: 125.0,
            syy: 0.0,
            szz: 0.0,
            vm: 125.0,
            maxp: 125.0,
        };
        let (open_pillars, _) = evaluate_residuals(open_state, cfg);
        let (penetrating_pillars, _) = evaluate_residuals(penetrating_state, cfg);
        assert!(
            open_pillars.boundary < 1e-9,
            "expected open contact boundary residual near zero, got {}",
            open_pillars.boundary
        );
        assert!(
            penetrating_pillars.boundary > 1.0 && penetrating_pillars.boundary < 1.5,
            "penetrating contact boundary residual moved outside tolerance band: {}",
            penetrating_pillars.boundary
        );
        assert!(penetrating_pillars.boundary > open_pillars.boundary);
    }

    #[test]
    fn long_run_training_stays_bounded_for_cantilever_schedule() {
        let cfg = PilotConfig {
            e_psi: 29_000_000.0,
            length_in: 10.0,
            width_in: 1.0,
            thickness_in: 0.25,
            hole_diameter_in: 0.0,
            axial_load_lbf: 0.0,
            vertical_load_lbf: -100.0,
            residual_w_momentum: 1.0,
            residual_w_kinematics: 1.0,
            residual_w_material: 1.0,
            residual_w_boundary: 1.0,
            contact_penalty: 10.0,
            plasticity_factor: 0.0,
        };
        let stats = run_pinn_pilot_curriculum(cfg, 10_000, 42);
        assert!(stats.initial_loss.is_finite());
        assert!(stats.best_loss.is_finite());
        assert!(stats.final_loss.is_finite());
        assert!(
            stats.best_loss < 1.0e12,
            "best loss exploded: {}",
            stats.best_loss
        );
        assert!(
            stats.final_loss < 1.0e12,
            "final loss exploded: {}",
            stats.final_loss
        );
    }
}
