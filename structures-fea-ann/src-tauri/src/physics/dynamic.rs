use crate::contracts::{DynamicInput, DynamicResult};

const G_IN_PER_S2: f64 = 386.088_582_677_165_35;

pub fn run(input: &DynamicInput) -> Result<DynamicResult, String> {
    validate(input)?;

    let s = &input.solve_input;
    let area = s.geometry.width_in * s.geometry.thickness_in;
    let volume = area * s.geometry.length_in;
    let weight = s.material.rho_lb_in3 * volume;
    let mass = (weight / G_IN_PER_S2).max(1e-9);
    let k = (s.material.e_psi * area / s.geometry.length_in.max(1e-6)).max(1e-6);
    let c_crit = 2.0 * (k * mass).sqrt();
    let c = input.damping_ratio.max(0.0) * c_crit;

    let dt = input.time_step_s.max(1e-6);
    let total_steps = (input.end_time_s / dt).ceil().max(1.0) as usize;

    let beta = 0.25;
    let gamma = 0.5;

    let mut t = vec![0.0; total_steps + 1];
    let mut u = vec![0.0; total_steps + 1];
    let mut v = vec![0.0; total_steps + 1];
    let mut a = vec![0.0; total_steps + 1];

    let effective_k = k + gamma / (beta * dt) * c + mass / (beta * dt * dt);
    let pulse_force_lbf = forcing_lbf(input);
    let load_source = if input.solve_input.load.vertical_point_load_lbf.abs() > 1e-9 {
        "vertical-point-load"
    } else {
        "axial-load"
    };

    for i in 0..total_steps {
        t[i + 1] = (i as f64 + 1.0) * dt;
        let f_next = forcing(t[i + 1], pulse_force_lbf, input);

        let rhs = f_next
            + mass
                * (u[i] / (beta * dt * dt)
                    + v[i] / (beta * dt)
                    + (1.0 / (2.0 * beta) - 1.0) * a[i])
            + c * (gamma * u[i] / (beta * dt)
                + (gamma / beta - 1.0) * v[i]
                + dt * (gamma / (2.0 * beta) - 1.0) * a[i]);

        u[i + 1] = rhs / effective_k;
        a[i + 1] = (u[i + 1] - u[i]) / (beta * dt * dt)
            - v[i] / (beta * dt)
            - (1.0 / (2.0 * beta) - 1.0) * a[i];
        v[i + 1] = v[i] + dt * ((1.0 - gamma) * a[i] + gamma * a[i + 1]);
    }

    Ok(DynamicResult {
        time_s: t,
        displacement_in: u,
        velocity_in_s: v,
        acceleration_in_s2: a,
        stable: effective_k.is_finite() && mass.is_finite() && k.is_finite(),
        diagnostics: vec![
            "Implicit Newmark integration (beta=0.25, gamma=0.5).".to_string(),
            format!("Load source: {load_source}, pulse force = {:.3} lbf.", pulse_force_lbf),
            format!(
                "Equivalent k={:.3}, effective mass={:.6}, damping={:.3}",
                k, mass, c
            ),
        ],
    })
}

fn validate(input: &DynamicInput) -> Result<(), String> {
    input.solve_input.validate()?;
    if !input.time_step_s.is_finite() || input.time_step_s <= 0.0 {
        return Err("timeStepS must be a positive finite value.".to_string());
    }
    if !input.end_time_s.is_finite() || input.end_time_s <= 0.0 {
        return Err("endTimeS must be a positive finite value.".to_string());
    }
    if !input.damping_ratio.is_finite() || input.damping_ratio < 0.0 {
        return Err("dampingRatio must be finite and non-negative.".to_string());
    }
    if !input.pulse_duration_s.is_finite() || input.pulse_duration_s < 0.0 {
        return Err("pulseDurationS must be finite and non-negative.".to_string());
    }
    if !input.pulse_scale.is_finite() {
        return Err("pulseScale must be finite.".to_string());
    }
    if input.solve_input.load.axial_load_lbf.abs() < 1e-12
        && input.solve_input.load.vertical_point_load_lbf.abs() < 1e-12
    {
        return Err("Dynamic analysis requires a non-zero axial or vertical pulse load.".to_string());
    }
    Ok(())
}

fn forcing_lbf(input: &DynamicInput) -> f64 {
    let load = if input.solve_input.load.vertical_point_load_lbf.abs() > 1e-12 {
        input.solve_input.load.vertical_point_load_lbf
    } else {
        input.solve_input.load.axial_load_lbf
    };
    load * input.pulse_scale.max(0.0)
}

fn forcing(t: f64, pulse_force_lbf: f64, input: &DynamicInput) -> f64 {
    if t <= input.pulse_duration_s {
        pulse_force_lbf
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::SolveInput;

    #[test]
    fn uses_vertical_load_for_pulse_when_present() {
        let mut input = DynamicInput {
            solve_input: SolveInput::default(),
            time_step_s: 0.001,
            end_time_s: 0.01,
            damping_ratio: 0.02,
            pulse_duration_s: 0.003,
            pulse_scale: 1.0,
        };
        input.solve_input.load.axial_load_lbf = 0.0;
        input.solve_input.load.vertical_point_load_lbf = 100.0;
        let result = run(&input).expect("dynamic solve");
        assert!(result.displacement_in.iter().any(|v| *v != 0.0));
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.contains("vertical-point-load")));
    }

    #[test]
    fn rejects_non_positive_time_step() {
        let input = DynamicInput {
            solve_input: SolveInput::default(),
            time_step_s: 0.0,
            end_time_s: 0.01,
            damping_ratio: 0.02,
            pulse_duration_s: 0.003,
            pulse_scale: 1.0,
        };
        assert!(run(&input).is_err());
    }
}
