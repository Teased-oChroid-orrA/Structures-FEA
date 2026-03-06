# Validation Pack

## Baseline static case
- Geometry: L=10 in, W=4 in, t=0.125 in
- Load: P=1000 lbf
- Area: A=0.5 in^2
- Applied nominal stress: sigma = P/A = 2000 psi

## Equilibrium / constitutive checks
- Equilibrium (static): div(sigma)+b = 0
- Constitutive (isotropic linear): sigma = C:(epsilon-epsilon_th)

## Expected numeric anchors (default Al 6061-T6 constants)
- epsilon_x = sigma_x/E = 2.0e3 / 1.0e7 = 2.0e-4
- u(L) = epsilon_x * L = 0.002 in

## Thermal check
- epsilon_th = alpha * deltaT
- restrained_x => sigma_th = E * epsilon_th

## Failure checks
- Uniaxial anchor: Von Mises = |sigma_x|
- Principal ordering descending

## Dynamic checks
- Newmark beta=0.25 gamma=0.5
- Finite response and no NaN/Inf across history
