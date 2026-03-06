use crate::contracts::{DynamicInput, DynamicResult};

pub fn run(input: &DynamicInput) -> DynamicResult {
    let s = &input.solve_input;
    let area = s.geometry.width_in * s.geometry.thickness_in;
    let volume = area * s.geometry.length_in;
    let mass = (s.material.rho_lb_in3 * volume).max(1e-6);
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

    for i in 0..total_steps {
        t[i + 1] = (i as f64 + 1.0) * dt;
        let f_next = forcing(t[i + 1], input);

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

    DynamicResult {
        time_s: t,
        displacement_in: u,
        velocity_in_s: v,
        acceleration_in_s2: a,
        stable: effective_k.is_finite(),
        diagnostics: vec![
            "Implicit Newmark integration (beta=0.25, gamma=0.5).".to_string(),
            format!("Equivalent k={:.3}, m={:.6}, c={:.3}", k, mass, c),
        ],
    }
}

fn forcing(t: f64, input: &DynamicInput) -> f64 {
    if t <= input.pulse_duration_s.max(0.0) {
        input.solve_input.load.axial_load_lbf * input.pulse_scale.max(0.0)
    } else {
        0.0
    }
}
