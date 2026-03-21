use nalgebra::{Matrix3, SymmetricEigen};

pub fn von_mises(stress: [[f64; 3]; 3]) -> f64 {
    let sx = stress[0][0];
    let sy = stress[1][1];
    let sz = stress[2][2];
    let txy = stress[0][1];
    let tyz = stress[1][2];
    let txz = stress[0][2];
    (0.5 * ((sx - sy).powi(2) + (sy - sz).powi(2) + (sz - sx).powi(2))
        + 3.0 * (txy.powi(2) + tyz.powi(2) + txz.powi(2)))
    .sqrt()
}

pub fn principal_stresses(stress: [[f64; 3]; 3]) -> [f64; 3] {
    let m = Matrix3::new(
        stress[0][0],
        stress[0][1],
        stress[0][2],
        stress[1][0],
        stress[1][1],
        stress[1][2],
        stress[2][0],
        stress[2][1],
        stress[2][2],
    );
    let eig = SymmetricEigen::new(m);
    let mut vals = [eig.eigenvalues[0], eig.eigenvalues[1], eig.eigenvalues[2]];
    vals.sort_by(|a, b| b.total_cmp(a));
    vals
}

pub fn tresca_from_principal(principal: [f64; 3]) -> f64 {
    (principal[0] - principal[2]).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn principal_sort_descending() {
        let p = principal_stresses([[2000.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, -10.0]]);
        assert!(p[0] >= p[1] && p[1] >= p[2]);
    }

    #[test]
    fn von_mises_uniaxial() {
        let vm = von_mises([[2000.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        assert!((vm - 2000.0).abs() < 1e-6);
    }
}
