use crate::contracts::{FailureInput, FailureResult};

use super::stress::{principal_stresses, tresca_from_principal, von_mises};

pub fn evaluate(input: &FailureInput) -> FailureResult {
    let principal = principal_stresses(input.stress_tensor);
    let vm = von_mises(input.stress_tensor);
    let tresca = tresca_from_principal(principal);
    let max_principal = principal[0];
    let sy = input.yield_strength_psi.max(1.0);

    let sf_vm = sy / vm.max(1e-9);
    let sf_tresca = sy / tresca.max(1e-9);
    let sf_principal = sy / max_principal.abs().max(1e-9);

    FailureResult {
        von_mises_psi: vm,
        tresca_psi: tresca,
        max_principal_psi: max_principal,
        safety_factor_vm: sf_vm,
        safety_factor_tresca: sf_tresca,
        safety_factor_principal: sf_principal,
        failed: sf_vm < 1.0 || sf_tresca < 1.0 || sf_principal < 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_flags_low_margin() {
        let result = evaluate(&FailureInput {
            stress_tensor: [[50_000.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            yield_strength_psi: 40_000.0,
        });
        assert!(result.failed);
    }
}
