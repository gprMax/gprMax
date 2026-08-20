# GSSI-like 1.5 GHz antenna energy-pattern convergence

## Purpose

This calculation reproduces the lossless half-space configuration used by
Warren and Giannopoulos (2017) and tests the reported convergence of
finite-radius, time-integrated GPR antenna patterns with increasing distance.
It also compares the finite-radius patterns with an independently calculated
broadband layered-medium equivalent-current NTFF pattern.

Reference: C. Warren and A. Giannopoulos, "Characterisation of a ground
penetrating radar antenna in lossless homogeneous and lossy heterogeneous
environments," *Signal Processing*, 132, 221--226, 2017,
doi:10.1016/j.sigpro.2016.04.010.

## Model and measures

- Toolbox model: `antenna_like_GSSI_1500`.
- Background: lossless half-space with relative permittivity 5 and relative
  permeability 1.
- Observation radii: 0.10--0.58 m in 0.02 m steps (25 radii).
- Angular interval: 3 degrees, omitting points exactly in the material
  interface.
- Time window: 8 ns.
- Spatial discretisation: 2 mm. The paper used 1 mm; the coarser grid was
  required to retain the paper's full 0.58 m radius on one 24 GB GPU.

The finite-radius field-energy measure is

    Psi_E(r, theta) = dt sum_n E_theta(r, theta, n)^2

or its magnetic-field counterpart. The comparison uses pattern shape: every
radius is divided by its own angular maximum.

The asymptotic reference integrates the layered NTFF spectrum from 0.1 to
4.0 GHz,

    Psi_NTFF(theta) proportional to integral |E(theta, f)|^2 df.

For the H-plane magnetic comparison, the electric NTFF spectrum is converted
using the wave impedance of the medium before integration. Parseval checks on
the finite-radius records show that the selected band contains 99.71% of the
aggregate E-plane energy and 99.90% of the aggregate H-plane energy.

## Results

| plane | RMS shape difference at 0.10 m | RMS at 0.58 m | maximum at 0.58 m |
|---|---:|---:|---:|
| E | 0.0934 | 0.0138 | 0.0480 |
| H | 0.1894 | 0.0344 | 0.0860 |

The distance trend reproduces the paper's physical result: near the antenna,
the pattern includes reactive and non-asymptotic structure; with increasing
radius, the curves collapse toward a stable ground-directed pattern. The
H-plane converges more slowly than the E-plane in this run.

This is not a point-by-point numerical comparison with digitised data from the
published figures. It is a reproduction of the same configuration and energy
measure, supplemented by a new layered-NTFF asymptotic reference.

## Files

- `gssi_energy_E_paper_style.png`: all 25 E-plane radii with a common energy
  reference, following the original presentation.
- `gssi_energy_H_paper_style.png`: corresponding H-plane plot.
- `gssi_energy_convergence.png`: selected radii, layered-NTFF reference, and
  quantitative convergence curves.
- `summary.json`: all numerical convergence and spectral-coverage values.
- `gssi_energy_E.npz`, `gssi_energy_H.npz`: processed finite-radius patterns.
- `gssi_energy_E.h5`, `gssi_energy_H.h5`: raw simulations and NTFF outputs.
