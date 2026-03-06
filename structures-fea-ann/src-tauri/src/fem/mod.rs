use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use crate::contracts::{BeamStationResult, FemResult, NodalDisplacement, SolveInput};
use crate::physics::stress::{principal_stresses, tresca_from_principal, von_mises};

type Mat6 = SMatrix<f64, 6, 6>;
type MatB = SMatrix<f64, 6, 24>;
type MatK = SMatrix<f64, 24, 24>;
type Vec6 = SVector<f64, 6>;

#[derive(Clone)]
struct HexElement {
    nodes: [usize; 8],
}

pub fn solve_case(input: &SolveInput) -> FemResult {
    let mesh = &input.mesh;

    let mut nx = mesh.nx.max(1);
    let mut ny = mesh.ny.max(1);
    let mut nz = mesh.nz.max(1);

    let mut diagnostics = Vec::new();
    if mesh.auto_adapt {
        let target_dofs = mesh.max_dofs.max(300);
        let mut est_dofs = (nx + 1) * (ny + 1) * (nz + 1) * 3;
        if est_dofs > target_dofs {
            let scale = ((target_dofs as f64) / (est_dofs as f64)).cbrt().clamp(0.2, 1.0);
            nx = ((nx as f64) * scale).round().max(1.0) as usize;
            ny = ((ny as f64) * scale).round().max(1.0) as usize;
            nz = ((nz as f64) * scale).round().max(1.0) as usize;
            est_dofs = (nx + 1) * (ny + 1) * (nz + 1) * 3;
            diagnostics.push(format!(
                "Auto mesh adapt enabled: reduced mesh to nx={}, ny={}, nz={} to target <= {} DOFs (actual {}).",
                nx, ny, nz, target_dofs, est_dofs
            ));
        } else {
            diagnostics.push(format!(
                "Auto mesh adapt enabled: requested mesh retained (estimated DOFs {}).",
                est_dofs
            ));
        }
    }

    let mut amr_notes = Vec::new();
    if mesh.amr_enabled && mesh.amr_passes > 0 {
        let max_nx = mesh.amr_max_nx.max(nx);
        let trigger = mesh.amr_refine_ratio.max(1.0);
        for pass in 0..mesh.amr_passes {
            let probe = solve_case_fixed(input, nx, ny, nz, Vec::new());
            let (ratio, max_g, mean_g) = stress_gradient_ratio(&probe.beam_stations);
            if ratio < trigger {
                amr_notes.push(format!(
                    "AMR pass {}: convergence reached (gradient ratio {:.3} < trigger {:.3}).",
                    pass + 1,
                    ratio,
                    trigger
                ));
                break;
            }
            let next_nx = ((nx as f64) * 1.35).ceil() as usize;
            let refined_nx = next_nx.max(nx + 1).min(max_nx);
            if refined_nx <= nx {
                amr_notes.push(format!(
                    "AMR pass {}: reached AMR max NX {} (ratio {:.3}, max grad {:.3e}, mean grad {:.3e}).",
                    pass + 1,
                    max_nx,
                    ratio,
                    max_g,
                    mean_g
                ));
                break;
            }
            amr_notes.push(format!(
                "AMR pass {}: stress-gradient ratio {:.3} (max {:.3e}, mean {:.3e}) -> refining NX {} -> {}.",
                pass + 1,
                ratio,
                max_g,
                mean_g,
                nx,
                refined_nx
            ));
            nx = refined_nx;
        }
    }

    let mut result = solve_case_fixed(input, nx, ny, nz, diagnostics);
    result.diagnostics.extend(amr_notes);
    result
}

fn solve_case_fixed(
    input: &SolveInput,
    nx: usize,
    ny: usize,
    nz: usize,
    mut diagnostics: Vec<String>,
) -> FemResult {
    let g = &input.geometry;
    let m = &input.material;

    let hx = g.length_in / nx as f64;
    let hy = g.width_in / ny as f64;
    let hz = g.thickness_in / nz as f64;

    let nodes = generate_nodes(g.length_in, g.width_in, g.thickness_in, nx, ny, nz);
    let elements = generate_elements(nx, ny, nz);
    let ndof = nodes.len() * 3;

    let mut k_global = DMatrix::<f64>::zeros(ndof, ndof);
    let mut m_global = DMatrix::<f64>::zeros(ndof, ndof);
    let mut f_global = DVector::<f64>::zeros(ndof);

    let d = elasticity_matrix(m.e_psi, m.nu);

    for elem in &elements {
        let (ke, me) = element_matrices(&d, m.rho_lb_in3, hx, hy, hz);
        assemble_element(elem, &ke, &mut k_global);
        assemble_element(elem, &me, &mut m_global);
    }

    let load_node = nearest_node(&nodes, g.length_in, g.width_in, g.thickness_in * 0.5);
    let load_dof = load_node * 3 + 1;
    f_global[load_dof] += input.load.vertical_point_load_lbf;

    let alpha = 1e-6;
    let beta = 1e-5;
    let mut c_global = m_global.clone() * alpha + k_global.clone() * beta;

    let fixed_dofs = fixed_dofs_left_face(&nodes, g.length_in, input.boundary_conditions.fix_start_face);
    for dof in fixed_dofs {
        for j in 0..ndof {
            k_global[(dof, j)] = 0.0;
            k_global[(j, dof)] = 0.0;
            m_global[(dof, j)] = 0.0;
            m_global[(j, dof)] = 0.0;
            c_global[(dof, j)] = 0.0;
            c_global[(j, dof)] = 0.0;
        }
        k_global[(dof, dof)] = 1.0;
        m_global[(dof, dof)] = 1.0;
        c_global[(dof, dof)] = 1.0;
        f_global[dof] = 0.0;
    }

    let u_global = match k_global.clone().lu().solve(&f_global) {
        Some(u) => u,
        None => DVector::<f64>::zeros(ndof),
    };

    let (nodal_vm, nodal_sigma_x, nodal_counts) = recover_nodal_stress(&elements, &nodes, &u_global, &d, hx, hy, hz);

    let mut nodal_displacements = Vec::with_capacity(nodes.len());
    for (i, p) in nodes.iter().enumerate() {
        let ux = u_global[i * 3];
        let uy = u_global[i * 3 + 1];
        let uz = u_global[i * 3 + 2];
        let disp_mag = (ux * ux + uy * uy + uz * uz).sqrt();
        let vm = if nodal_counts[i] > 0.0 {
            nodal_vm[i] / nodal_counts[i]
        } else {
            0.0
        };
        nodal_displacements.push(NodalDisplacement {
            node_id: i,
            x_in: p[0],
            y_in: p[1],
            z_in: p[2],
            ux_in: ux,
            uy_in: uy,
            uz_in: uz,
            disp_mag_in: disp_mag,
            vm_psi: vm,
        });
    }

    let top_root_sigma = root_top_sigma(&nodes, &nodal_sigma_x, &nodal_counts, g.width_in);
    let stress_tensor = [[top_root_sigma, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let strain_tensor = [[top_root_sigma / m.e_psi.max(1.0), 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let principal = principal_stresses(stress_tensor);

    let vm_global = nodal_displacements
        .iter()
        .map(|n| n.vm_psi)
        .fold(0.0_f64, |acc, v| acc.max(v));

    let beam_stations = build_beam_stations(&nodes, &nodal_displacements, &nodal_sigma_x, &nodal_counts, nx);

    let mut result = FemResult {
        nodal_displacements,
        strain_tensor,
        stress_tensor,
        principal_stresses: principal,
        von_mises_psi: vm_global,
        tresca_psi: tresca_from_principal(principal),
        max_principal_psi: principal[0],
        stiffness_matrix: compact_matrix(&k_global, 32),
        mass_matrix: compact_matrix(&m_global, 32),
        damping_matrix: compact_matrix(&c_global, 32),
        force_vector: f_global.iter().take(64).copied().collect(),
        displacement_vector: vec![0.0, u_global[load_dof]],
        beam_stations,
        diagnostics: Vec::new(),
    };

    diagnostics.push(format!(
        "3D HEX assembly complete: nodes={}, elements={}, dof={}",
        nodes.len(),
        elements.len(),
        ndof
    ));
    diagnostics.push(format!("Point load applied at node {} (dof {}).", load_node, load_dof));
    diagnostics.push("BC elimination applied on left fixed face DOFs.".to_string());
    diagnostics.push("Stress recovered at Gauss points and averaged to nodes for post-processing.".to_string());
    result.diagnostics = diagnostics;
    result
}

fn stress_gradient_ratio(stations: &[BeamStationResult]) -> (f64, f64, f64) {
    if stations.len() < 2 {
        return (1.0, 0.0, 0.0);
    }
    let mut grads = Vec::with_capacity(stations.len() - 1);
    for i in 0..stations.len() - 1 {
        let dx = (stations[i + 1].x_in - stations[i].x_in).abs().max(1e-9);
        let g = ((stations[i + 1].sigma_top_psi - stations[i].sigma_top_psi).abs()) / dx;
        grads.push(g);
    }
    let mean = grads.iter().sum::<f64>() / grads.len() as f64;
    let max = grads.iter().copied().fold(0.0_f64, f64::max);
    if mean <= 1e-12 {
        return (1.0, max, mean);
    }
    (max / mean, max, mean)
}

pub fn ann_features(input: &SolveInput) -> Vec<f64> {
    vec![
        input.geometry.length_in,
        input.geometry.width_in,
        input.geometry.thickness_in,
        input.load.axial_load_lbf,
        input.load.vertical_point_load_lbf,
        input.material.e_psi,
        input.material.nu,
        input.material.alpha_per_f,
    ]
}

pub fn ann_targets(fem: &FemResult) -> Vec<f64> {
    vec![
        fem.displacement_vector[1],
        fem.stress_tensor[0][0],
        fem.stress_tensor[1][1],
        fem.stress_tensor[2][2],
        fem.von_mises_psi,
        fem.max_principal_psi,
    ]
}

pub fn fem_from_ann_prediction(input: &SolveInput, pred: &[f64]) -> FemResult {
    let mut fem = solve_case(input);
    let sigma_x = pred[1];
    fem.stress_tensor = [[sigma_x, 0.0, 0.0], [0.0, pred[2], 0.0], [0.0, 0.0, pred[3]]];
    fem.von_mises_psi = pred[4].abs();
    fem.max_principal_psi = pred[5];
    fem.displacement_vector = vec![0.0, pred[0]];
    fem.diagnostics
        .push("ANN/PINN surrogate mapped onto full FEM mesh state for visualization and auditing.".to_string());
    fem
}

fn generate_nodes(l: f64, w: f64, t: f64, nx: usize, ny: usize, nz: usize) -> Vec<[f64; 3]> {
    let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push([
                    l * i as f64 / nx as f64,
                    w * j as f64 / ny as f64,
                    t * k as f64 / nz as f64,
                ]);
            }
        }
    }
    nodes
}

fn generate_elements(nx: usize, ny: usize, nz: usize) -> Vec<HexElement> {
    let mut elems = Vec::with_capacity(nx * ny * nz);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let n0 = node_index(i, j, k, nx, ny);
                let n1 = node_index(i + 1, j, k, nx, ny);
                let n2 = node_index(i + 1, j + 1, k, nx, ny);
                let n3 = node_index(i, j + 1, k, nx, ny);
                let n4 = node_index(i, j, k + 1, nx, ny);
                let n5 = node_index(i + 1, j, k + 1, nx, ny);
                let n6 = node_index(i + 1, j + 1, k + 1, nx, ny);
                let n7 = node_index(i, j + 1, k + 1, nx, ny);
                elems.push(HexElement {
                    nodes: [n0, n1, n2, n3, n4, n5, n6, n7],
                });
            }
        }
    }
    elems
}

fn node_index(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    k * (ny + 1) * (nx + 1) + j * (nx + 1) + i
}

fn elasticity_matrix(e: f64, nu: f64) -> Mat6 {
    let c = e / ((1.0 + nu) * (1.0 - 2.0 * nu)).max(1e-9);
    let l = nu * c;
    let mu = e / (2.0 * (1.0 + nu)).max(1e-9);
    Mat6::from_row_slice(&[
        l + 2.0 * mu,
        l,
        l,
        0.0,
        0.0,
        0.0,
        l,
        l + 2.0 * mu,
        l,
        0.0,
        0.0,
        0.0,
        l,
        l,
        l + 2.0 * mu,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        mu,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        mu,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        mu,
    ])
}

fn element_matrices(d: &Mat6, rho: f64, hx: f64, hy: f64, hz: f64) -> (MatK, MatK) {
    let mut ke = MatK::zeros();
    let mut me = MatK::zeros();
    let g = 1.0 / 3.0_f64.sqrt();
    let gps = [-g, g];
    let det_j = (hx * hy * hz) / 8.0;

    for &xi in &gps {
        for &eta in &gps {
            for &zeta in &gps {
                let (n, grads) = shape_fns_and_grads(xi, eta, zeta, hx, hy, hz);
                let b = build_b_matrix(&grads);
                ke += b.transpose() * (*d) * b * det_j;

                for a in 0..8 {
                    for bidx in 0..8 {
                        let mab = rho * n[a] * n[bidx] * det_j;
                        for d in 0..3 {
                            me[(a * 3 + d, bidx * 3 + d)] += mab;
                        }
                    }
                }
            }
        }
    }

    (ke, me)
}

fn shape_fns_and_grads(xi: f64, eta: f64, zeta: f64, hx: f64, hy: f64, hz: f64) -> ([f64; 8], [[f64; 3]; 8]) {
    let signs = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ];

    let mut n = [0.0; 8];
    let mut grads = [[0.0; 3]; 8];

    for (i, (sx, sy, sz)) in signs.iter().enumerate() {
        n[i] = 0.125 * (1.0 + sx * xi) * (1.0 + sy * eta) * (1.0 + sz * zeta);
        let dxi = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta);
        let deta = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta);
        let dzeta = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta);
        grads[i] = [dxi * 2.0 / hx, deta * 2.0 / hy, dzeta * 2.0 / hz];
    }

    (n, grads)
}

fn build_b_matrix(grads: &[[f64; 3]; 8]) -> MatB {
    let mut b = MatB::zeros();
    for (a, g) in grads.iter().enumerate() {
        let c = a * 3;
        b[(0, c)] = g[0];
        b[(1, c + 1)] = g[1];
        b[(2, c + 2)] = g[2];
        b[(3, c)] = g[1];
        b[(3, c + 1)] = g[0];
        b[(4, c + 1)] = g[2];
        b[(4, c + 2)] = g[1];
        b[(5, c)] = g[2];
        b[(5, c + 2)] = g[0];
    }
    b
}

fn assemble_element(elem: &HexElement, ke: &MatK, global: &mut DMatrix<f64>) {
    for a in 0..8 {
        for b in 0..8 {
            for ia in 0..3 {
                for ib in 0..3 {
                    let row = elem.nodes[a] * 3 + ia;
                    let col = elem.nodes[b] * 3 + ib;
                    global[(row, col)] += ke[(a * 3 + ia, b * 3 + ib)];
                }
            }
        }
    }
}

fn fixed_dofs_left_face(nodes: &[[f64; 3]], length: f64, fix_start: bool) -> Vec<usize> {
    if !fix_start {
        return Vec::new();
    }
    let tol = (length.abs() * 1e-9).max(1e-9);
    let mut dofs = Vec::new();
    for (i, p) in nodes.iter().enumerate() {
        if p[0].abs() <= tol {
            dofs.push(i * 3);
            dofs.push(i * 3 + 1);
            dofs.push(i * 3 + 2);
        }
    }
    dofs
}

fn nearest_node(nodes: &[[f64; 3]], x: f64, y: f64, z: f64) -> usize {
    let mut idx = 0usize;
    let mut best = f64::MAX;
    for (i, p) in nodes.iter().enumerate() {
        let d = (p[0] - x).powi(2) + (p[1] - y).powi(2) + (p[2] - z).powi(2);
        if d < best {
            best = d;
            idx = i;
        }
    }
    idx
}

fn recover_nodal_stress(
    elements: &[HexElement],
    nodes: &[[f64; 3]],
    u: &DVector<f64>,
    d: &Mat6,
    hx: f64,
    hy: f64,
    hz: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut nodal_vm = vec![0.0; nodes.len()];
    let mut nodal_sigma_x = vec![0.0; nodes.len()];
    let mut counts = vec![0.0; nodes.len()];
    let g = 1.0 / 3.0_f64.sqrt();
    let gps = [-g, g];

    for elem in elements {
        let mut ue = SVector::<f64, 24>::zeros();
        for a in 0..8 {
            let nid = elem.nodes[a];
            ue[a * 3] = u[nid * 3];
            ue[a * 3 + 1] = u[nid * 3 + 1];
            ue[a * 3 + 2] = u[nid * 3 + 2];
        }

        let mut sigma_acc: f64 = 0.0;
        let mut vm_acc: f64 = 0.0;
        let mut samples: f64 = 0.0;
        for &xi in &gps {
            for &eta in &gps {
                for &zeta in &gps {
                    let (_, grads) = shape_fns_and_grads(xi, eta, zeta, hx, hy, hz);
                    let b = build_b_matrix(&grads);
                    let strain: Vec6 = b * ue;
                    let stress: Vec6 = (*d) * strain;
                    let tensor = [
                        [stress[0], stress[3], stress[5]],
                        [stress[3], stress[1], stress[4]],
                        [stress[5], stress[4], stress[2]],
                    ];
                    vm_acc += von_mises(tensor);
                    sigma_acc += stress[0];
                    samples += 1.0;
                }
            }
        }

        let vm_elem = vm_acc / samples.max(1.0);
        let sigma_elem = sigma_acc / samples.max(1.0);

        for &nid in &elem.nodes {
            nodal_vm[nid] += vm_elem;
            nodal_sigma_x[nid] += sigma_elem;
            counts[nid] += 1.0;
        }
    }

    (nodal_vm, nodal_sigma_x, counts)
}

fn root_top_sigma(nodes: &[[f64; 3]], sigma_x: &[f64], counts: &[f64], width: f64) -> f64 {
    let mut best_sigma = 0.0;
    let mut best_y = f64::MIN;
    for (i, p) in nodes.iter().enumerate() {
        if p[0].abs() < 1e-8 && p[1] >= best_y {
            best_y = p[1];
            best_sigma = if counts[i] > 0.0 { sigma_x[i] / counts[i] } else { 0.0 };
        }
    }
    if best_y <= f64::MIN / 2.0 {
        return 0.0;
    }
    if width.abs() < 1e-9 {
        return 0.0;
    }
    best_sigma
}

fn build_beam_stations(
    nodes: &[[f64; 3]],
    nodal_disp: &[NodalDisplacement],
    sigma_x: &[f64],
    counts: &[f64],
    nx: usize,
) -> Vec<BeamStationResult> {
    let mut stations = Vec::new();
    let n = nx.max(2);
    let max_x = nodes.iter().map(|p| p[0]).fold(0.0_f64, f64::max).max(1e-9);

    for i in 0..=n {
        let x_target = max_x * i as f64 / n as f64;
        let mut best = 0usize;
        let mut score = f64::MAX;
        for (nid, p) in nodes.iter().enumerate() {
            let s = (p[0] - x_target).abs() + (max_x - p[1]).abs() * 0.2;
            if s < score {
                score = s;
                best = nid;
            }
        }
        let sx = if counts[best] > 0.0 {
            sigma_x[best] / counts[best]
        } else {
            0.0
        };
        stations.push(BeamStationResult {
            x_in: nodes[best][0],
            shear_lbf: 0.0,
            moment_lb_in: 0.0,
            sigma_top_psi: sx,
            sigma_bottom_psi: -sx,
            deflection_in: nodal_disp[best].uy_in,
        });
    }

    stations.sort_by(|a, b| a.x_in.total_cmp(&b.x_in));
    stations
}

fn compact_matrix(m: &DMatrix<f64>, n: usize) -> Vec<Vec<f64>> {
    let r = m.nrows().min(n);
    let c = m.ncols().min(n);
    let mut out = vec![vec![0.0; c]; r];
    for i in 0..r {
        for j in 0..c {
            out[i][j] = m[(i, j)];
        }
    }
    out
}
