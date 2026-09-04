# Eigenmode solver rerun results

Follow-up: [targeted validation of the remaining dispersion fixes](remaining_dispersion_fixes.md) records the five reruns and updated measurements after `eb664546`. The 57-case results below describe the earlier baseline.

Status: **COMPLETE**. 57/57 maintained FDTD case runs completed. Standalone FDFD solves cover 24 frequency/geometry pairs and 30 modal rows.

The final CST patch far-field simulation was completed by the user and verified from its fresh HDF5 output. Its exact simulation command and wall time were not recorded. The plotter had still selected the older `patch_recentered_closed_ntff.h5`; its default now matches the current model basename and it also accepts `--gprmax-output`. The comparison plots and numerical summary were regenerated from `patch_antenna_recentered_closed_ntff_backed_pml.h5`. The earlier setup and output-permission failures remain in the attempt history.

The verified file contains 25,964 timesteps and 11 frequencies with 65,160 directions each. All eight inspected far-field datasets are finite, and all 11 modal coefficients and 11 power-wave flags are valid. The file SHA-256 and dataset shapes/types are recorded in `summary.json`.

The runtime environment below describes the automated reruns. Only properties verified from the user-produced HDF5 are attributed to the manual simulation; its unrecorded details are left unknown.

The solver baseline is commit `03a84afd467aeb3ce372077e3d15d3cdfc163b00` on `codex/fdfd-dispersion-validation`, including the numerical-dispersion compensation and virtual-waveguide documentation. The run date is 2026-09-04 in Asia/Shanghai. Follow-up changes supply missing timesteps in three test fixtures, prevent cropping in the tutorial horn plot, and remove an undefined PML-profile name in the CST patch input. These changes do not alter the production solver.

## Scope and interpretation

| Family | Physical FDTD runs | Evidence |
| --- | ---: | --- |
| Maintained regression matrix | 17 | Physical trend gates; grid-spacing diagnostics |
| Six curated examples | 7 | Successful runs and regenerated plots; complete-S study has two drives |
| Partial-cutoff validation | 1 | Analytical magnitude/phase gates |
| CST horn and patch comparisons | 3 | Inter-code differences, without acceptance gates |
| CPU multiport, copper validation and PEC/copper comparison | 5 | Two multiport drives, one copper validation, two comparison runs |
| Internal-PML sweep and two fixtures | 24 | Twenty-two sweep runs and two fixtures; finite modal data checks |

An additional ad-hoc dense horn check completed during this work. At the user request, its S-parameter results are kept only as ignored local artifacts and are not retained in this report or added to the maintained horn case. The maintained horn input uses its original five frequency bins. This extra check is separate from the 57 maintained runs.

The direct FDFD validator is separate from these FDTD runs. It uses the standalone solver defaults without an FDTD timestep or longitudinal spacing; its continuum-index checks remain useful but do not themselves exercise compensation. Pytest executes additional compact test simulations and is counted separately.

Ignored `legacy/` inputs and equivalent `.in` representations of the six Python tutorials are excluded. The dipole-only internal-PML fixture does not use the eigenmode solver. Automated reruns generated fresh simulation output without `--reuse`; the PML driver used `--force`.

A successful command or an ungated comparison is recorded as completed, not as proof of physical accuracy. Multiport residuals, tutorial/CST plots, and PML trends are reported with their limitations. All original numerical thresholds are retained.

## Runtime

Python 3.14.4, NumPy 2.4.3, SciPy 1.17.1, pytest 9.0.3; Windows 11. CUDA device 0 is an NVIDIA GeForce RTX 4070 Laptop GPU (8 GiB), using single-precision FDTD and MSVC 14.44 for CUDA host compilation. FDFD preprocessing runs on the host.

The five CPU analytical/comparison FDTD runs use double precision and four threads. The PML CLI models retain single-precision defaults and use four OpenMP threads from the environment; the 14-thread host banner reports available physical cores, not the configured solver count. The non-GPU pytest process limits OpenMP/BLAS to two threads, except tests that explicitly choose their own count. GPU study tests request four OpenMP and two BLAS threads. Per-command runtimes below are process wall times, except PML rows, whose timers come from gprMax; two-drive workflows share one recorded time. Concurrent runs make these timings unsuitable as a controlled performance comparison.

## Numerical acceptance

| Workflow | Status | Unchanged criterion |
| --- | --- | --- |
| Standalone FDFD analytical indices | passed | Maximum relative neff error: 0.1% for parallel plate, slab, rectangular guide; 1.5% for cylindrical TE11 pair. |
| Copper-wall propagation | passed | Maximum S11 < -20 dB; FDFD attenuation relative L2 error < 1%; FDTD attenuation relative L2 error < 2%. |
| Eigenmode regression physical trends | passed | Straight guides: max \|S21(dB)\| < 0.75 dB, max S11 < -20 dB. Bends: strictly increasing mean S21 with radius and >=2 dB small-to-large improvement. Lossy guide: >3 dB lower mean S21. Broadband profile: lower mean absolute S21(dB) than single-frequency profile. Grid-spacing metrics are reported without a convergence threshold. |
| Partial-cutoff analytical TE10 | passed | Maximum S21 magnitude error <=0.45 dB and circular phase error <=3 degrees; coefficient and power masks checked by the existing plotter. |

### Copper-wall guide

| Metric | Before compensation | Rerun |
| --- | ---: | ---: |
| Surface fit relative L2 error | 0.026023% | 0.026023% |
| FDFD attenuation relative L2 error | 0.146829% | 0.681438% |
| FDTD attenuation relative L2 error | 0.759861% | 0.759867% |
| Maximum driven-port S11 | -65.000459 dB | -101.089343 dB |

The FDFD attenuation difference from continuum perturbation theory increases and remains below the original 1% gate. The source reflection decreases, while the FDTD attenuation error is nearly unchanged. These are different comparisons and should not be combined into a single improvement claim.

[Copper comparison figure](../impedance_surface/results/copper_wall_waveguide/copper_wall_waveguide.png) and [full numerical summary](../impedance_surface/results/copper_wall_waveguide/summary.json).

### Multiport de-embedding

The full incident-matrix solve has maximum linear-magnitude error 0.0020366 and phase error 1.226380 degrees against continuum TE10 propagation. Its maximum measured network residual is 3.69535e-16, versus 0.0398982 for diagonal normalization. The prescribed analytical matrix is recovered with error 4.61489e-16. These values verify the algebra and report the finite-grid guide discrepancy; no new acceptance threshold was added.

[Multiport figure](../eigenmode_multiport_deembedding/rectangular_waveguide_deembedding.png) and [full summary](../eigenmode_multiport_deembedding/summary.json).

### Internal-PML experiment

All 24 runs contain 61 distinct, finite modal frequencies each (1464 rows), with coefficient and power-wave validity true. The two retained PML fixtures differ in complex S11 by at most 8.33958e-06. This is a consistency check, without an added threshold.

| Fill | PML cells | Worst raw S11 (dB) | Worst excess relative to 50 cells (dB) |
| --- | ---: | ---: | ---: |
| empty | 0 | 0.004 | 0.042 |
| empty | 4 | -26.954 | -28.964 |
| empty | 6 | -31.615 | -42.710 |
| empty | 8 | -32.516 | -51.583 |
| empty | 10 | -32.589 | -54.529 |
| empty | 12 | -32.544 | -57.036 |
| empty | 16 | -32.412 | -61.868 |
| empty | 20 | -32.322 | -65.884 |
| empty | 30 | -32.355 | -85.852 |
| empty | 40 | -32.351 | -111.245 |
| empty | 50 | -32.351 | reference |
| lossy_debye | 0 | -16.171 | -16.169 |
| lossy_debye | 4 | -57.211 | -76.112 |
| lossy_debye | 6 | -57.690 | -95.754 |
| lossy_debye | 8 | -57.748 | -124.908 |
| lossy_debye | 10 | -57.749 | -130.548 |
| lossy_debye | 12 | -57.749 | -132.157 |
| lossy_debye | 16 | -57.748 | -134.609 |
| lossy_debye | 20 | -57.748 | -133.000 |
| lossy_debye | 30 | -57.748 | -133.221 |
| lossy_debye | 40 | -57.747 | -131.212 |
| lossy_debye | 50 | -57.748 | reference |

[Full PML report](../../experimental/internal_pml_slab/length_sweep_results/pml_length_report.md). The long-slab subtraction removes a shared finite-time/source residual and is a diagnostic, not an independent analytical oracle.

### Regression measurements

```text
PASS: straight_waveguide/3d/cylindrical_waveguide: mean S21=-0.057 dB, max S11=-34.490 dB
PASS: straight_waveguide/3d/rectangular_waveguide: mean S21=0.000 dB, max S11=-49.994 dB
PASS: straight_waveguide/2d_tm/dielectric_waveguide: mean S21=-0.002 dB, max S11=-41.610 dB
PASS: straight_waveguide/2d_te/dielectric_waveguide: mean S21=-0.000 dB, max S11=-39.079 dB
PASS: dx_0p05mm: S21 range=-0.000 to 0.000 dB, max absolute=0.000 dB, half peak-to-peak=0.000 dB
PASS: dx_0p10mm: S21 range=-0.000 to 0.000 dB, max absolute=0.000 dB, half peak-to-peak=0.000 dB
PASS: dx_0p20mm: S21 range=-0.000 to 0.000 dB, max absolute=0.000 dB, half peak-to-peak=0.000 dB
PASS: 2d_te curved bends: mean S21 small=-2.482, medium=-0.239, large=-0.109 dB; improvement=2.373 dB
PASS: 2d_tm curved bends: mean S21 small=-5.489, medium=-0.799, large=-0.375 dB; improvement=5.115 dB
PASS: loss comparison: lossy=-6.127 dB, non-lossy=-0.002 dB
PASS: source-profile comparison: broadband mean |S21|=0.0021 dB, single-profile=0.0788 dB
```

### Partial-cutoff measurements

```text
24.25 GHz generalized amplitudes: S11=-35.217 dB, gprMax S21=-13.074 dB, analytical S21=-13.120 dB
24.36 GHz generalized amplitudes: S11=-35.972 dB, gprMax S21=-11.959 dB, analytical S21=-12.147 dB
24.46 GHz generalized amplitudes: S11=-31.493 dB, gprMax S21=-11.201 dB, analytical S21=-11.083 dB
24.57 GHz generalized amplitudes: S11=-27.982 dB, gprMax S21=-9.547 dB, analytical S21=-9.900 dB
24.67 GHz generalized amplitudes: S11=-40.384 dB, gprMax S21=-8.668 dB, analytical S21=-8.550 dB
24.78 GHz generalized amplitudes: S11=-24.226 dB, gprMax S21=-6.631 dB, analytical S21=-6.933 dB
24.89 GHz generalized amplitudes: S11=-17.242 dB, gprMax S21=-4.681 dB, analytical S21=-4.788 dB
Maximum S21 theory error: 0.353 dB magnitude, 2.145 degrees phase
Wrote testing/validation/rectangular_waveguide_partial_cutoff/rectangular_waveguide_partial_cutoff_s11_s21.png
```

The single-precision grid-spacing residuals are:

| Spacing (mm) | Maximum absolute S21 (dB) | RMS S21 about zero (dB) |
| ---: | ---: | ---: |
| 0.20 | 0.000122700 | 0.000057628 |
| 0.10 | 0.000139783 | 0.000070214 |
| 0.05 | 0.000166704 | 0.000070139 |

These small residuals do not decrease monotonically with grid spacing. The dashed second-order guide is a visual reference, not a fit or demonstrated quadratic convergence.

The ungated phased-array example peaks at 35, 32, 28, 25 and 23 degrees for nominal frequencies 8, 9, 10, 11 and 12 GHz. This documents beam squint and is not an acceptance test.

### CST patch comparison

The rerun minimum S11 is -12.2972 dB at 2.430000 GHz. CST reference data are unchanged. These are differences between numerical solvers; no correctness gate is applied.

[Patch S11 figure](../../other_codes/cst/patch_antenna/S-parameter/patch_s11.png), [patch far-field comparison](../../other_codes/cst/patch_antenna/farfield/patch_farfield_comparison_2p45ghz.png), and [horn far-field comparison](../../other_codes/cst/horn_antenna/horn_antenna_farfield_polar_comparison.png).

The verified manual patch far-field output was compared at 2.449999872 GHz (requested 2.45 GHz). The principal-plane measurements are ungated comparisons with the unchanged CST references.

| Cut | gprMax peak (dBi) | Peak theta (degrees) | Peak difference vs CST FEM (dB) | Front-hemisphere RMSE vs CST FEM (dB) |
| --- | ---: | ---: | ---: | ---: |
| phi_0_deg | 5.869401 | 0.0 | 0.134401 | 0.304189 |
| phi_90_deg | 6.179636 | 18.0 | 0.283636 | 0.260533 |

The comparison clips directivity at -40 dBi for the error statistics. [Full far-field comparison summary](../../other_codes/cst/patch_antenna/farfield/patch_farfield_comparison_2p45ghz.json).

## Automated tests

| Selection | Passed | Skipped | Deselected | Failures | Errors | Seconds | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 37 selected files, not gpu | 682 | 0 | 30 | 3 | 0 | 571.905 | failed |
| GPU source / virtual-waveguide tests | 10 | 10 | 44 | 0 | 0 | 77.355 | passed |
| GPU eigenmode / port studies | 5 | 5 | 27 | 0 | 0 | 66.344 | passed |
| Corrected mock-grid module rerun | 9 | 0 | 0 | 0 | 0 | 0.098 | passed |

The initial non-GPU selection reported 682 passes and three failures caused by mock grids missing `dt`. The three test fixtures were given `dt=1e-12`; no production code, assertion, or threshold changed. The complete affected module then passed 9/9 tests, resolving all three initial failures. Across the original run and this targeted rerun, 685 unique non-GPU tests pass. The entire selection was not rerun after this fixture correction. The initial failures remain recorded in the table and JSON.

Together, the evidence records 700 unique passing tests: 685 non-GPU and 15 CUDA. 15 OpenCL tests are skipped because `pyopencl` is unavailable.

Recorded skip reasons:

- OpenCL hardware is unavailable: No module named 'pyopencl'

## Visual verification

The visual-review log covers 14 regenerated regression and tutorial plots. The standalone FDFD, two multiport, copper-validation and PEC/copper comparison figures were also inspected. Axes, labels and modal interpretation were checked. A cropped tick in the tutorial horn 3D figure was fixed with a tight image bounding box and regenerated from the fresh simulation output. No FDTD rerun was needed for that display-only fix.

Additional fresh figures inspected with no unresolved visual issues:

- `testing/validation/rectangular_waveguide_partial_cutoff/rectangular_waveguide_partial_cutoff_s11_s21.png`.
- `testing/other_codes/cst/horn_antenna/horn_antenna_farfield_polar_comparison.png`.
- `testing/other_codes/cst/patch_antenna/S-parameter/patch_s11.png`.

The patch far-field figures regenerated from the verified manual output were also visually inspected; the detailed review and output provenance are retained in `summary.json`.

## Physical case ledger

Paths and working directories are repository-relative. `python` means the activated gprMax interpreter. Exact argument arrays, logs, test selections, metrics and all post-processing commands are retained in [summary.json](summary.json).

| Case/workflow | FDTD runs | Backend / precision | Seconds | Status |
| --- | ---: | --- | ---: | --- |
| bending_waveguide/2d_te/large_bend | 1 | CUDA / single | 39.359 | completed |
| bending_waveguide/2d_te/medium_bend | 1 | CUDA / single | 39.712 | completed |
| bending_waveguide/2d_te/small_bend | 1 | CUDA / single | 35.662 | completed |
| bending_waveguide/2d_tm/large_bend | 1 | CUDA / single | 37.272 | completed |
| bending_waveguide/2d_tm/medium_bend | 1 | CUDA / single | 31.183 | completed |
| bending_waveguide/2d_tm/small_bend | 1 | CUDA / single | 27.048 | completed |
| broadband_vs_single_frequency/broadband | 1 | CUDA / single | 34.313 | completed |
| broadband_vs_single_frequency/single_frequency | 1 | CUDA / single | 10.632 | completed |
| grid_spacing/rectangular_waveguide/dx_0p05mm | 1 | CUDA / single | 68.823 | completed |
| grid_spacing/rectangular_waveguide/dx_0p10mm | 1 | CUDA / single | 25.334 | completed |
| grid_spacing/rectangular_waveguide/dx_0p20mm | 1 | CUDA / single | 17.666 | completed |
| loss_comparison/lossy | 1 | CUDA / single | 23.274 | completed |
| loss_comparison/nonlossy | 1 | CUDA / single | 28.732 | completed |
| straight_waveguide/2d_te/dielectric_waveguide | 1 | CUDA / single | 37.247 | completed |
| straight_waveguide/2d_tm/dielectric_waveguide | 1 | CUDA / single | 26.619 | completed |
| straight_waveguide/3d/cylindrical_waveguide | 1 | CUDA / single | 37.901 | completed |
| straight_waveguide/3d/rectangular_waveguide | 1 | CUDA / single | 18.289 | completed |
| example_1_straight_waveguide | 1 | CUDA / single | 16.698 | completed |
| example_2_curved_waveguide | 1 | CUDA / single | 8.409 | completed |
| example_3_antenna_and_farfield | 1 | CUDA / single | 42.174 | completed |
| example_4_complete_s_matrix | 2 | CUDA / single | 23.276 | completed |
| example_5_phased_array | 1 | CUDA / single | 42.656 | completed |
| example_6_near_cutoff | 1 | CUDA / single | 56.280 | completed |
| partial-cutoff | 1 | CUDA / single | 75.500 | completed |
| cst-horn | 1 | CUDA / single | 147.802 | completed |
| cst-patch-sparameters | 1 | CUDA / single | 235.073 | completed |
| cst-patch-farfield | 1 | unrecorded / unrecorded | unrecorded | completed |
| multiport | 2 | cpu / double | 2.702 | completed |
| copper | 1 | cpu / double | 147.019 | completed |
| pec_copper_example | 2 | cpu / double | 20.425 | completed |
| pml/empty_pec_control | 1 | CPU / single | 9.422 | completed |
| pml/empty_pml_04_cells | 1 | CPU / single | 9.469 | completed |
| pml/empty_pml_06_cells | 1 | CPU / single | 9.524 | completed |
| pml/empty_pml_08_cells | 1 | CPU / single | 9.241 | completed |
| pml/empty_pml_10_cells | 1 | CPU / single | 9.717 | completed |
| pml/empty_pml_12_cells | 1 | CPU / single | 11.459 | completed |
| pml/empty_pml_16_cells | 1 | CPU / single | 11.036 | completed |
| pml/empty_pml_20_cells | 1 | CPU / single | 11.099 | completed |
| pml/empty_pml_30_cells | 1 | CPU / single | 10.811 | completed |
| pml/empty_pml_40_cells | 1 | CPU / single | 10.717 | completed |
| pml/empty_pml_50_cells | 1 | CPU / single | 11.770 | completed |
| pml/lossy_debye_pec_control | 1 | CPU / single | 14.336 | completed |
| pml/lossy_debye_pml_04_cells | 1 | CPU / single | 14.436 | completed |
| pml/lossy_debye_pml_06_cells | 1 | CPU / single | 13.926 | completed |
| pml/lossy_debye_pml_08_cells | 1 | CPU / single | 13.804 | completed |
| pml/lossy_debye_pml_10_cells | 1 | CPU / single | 13.827 | completed |
| pml/lossy_debye_pml_12_cells | 1 | CPU / single | 11.263 | completed |
| pml/lossy_debye_pml_16_cells | 1 | CPU / single | 15.610 | completed |
| pml/lossy_debye_pml_20_cells | 1 | CPU / single | 14.792 | completed |
| pml/lossy_debye_pml_30_cells | 1 | CPU / single | 11.676 | completed |
| pml/lossy_debye_pml_40_cells | 1 | CPU / single | 11.450 | completed |
| pml/lossy_debye_pml_50_cells | 1 | CPU / single | 11.373 | completed |
| pml/pml-rectangular_waveguide_external_pml | 1 | CPU / single | 4.604 | completed |
| pml/pml-rectangular_waveguide_internal_pml | 1 | CPU / single | 4.797 | completed |

Retries retain the original failure or skipped dependency and the final attempt in `summary.json`. The 57-run coverage count refers to distinct physical cases, not failed/repeated attempts. The ledger reports the latest attempt.

| Retried job | Initial status | Latest status |
| --- | --- | --- |
| cst-patch-farfield | failed | completed |
| cst-patch-farfield-plot | skipped-dependency | completed |

`cst-patch-farfield` attempt 1 (input_setup): `ValueError: Internal PML slab 'internal_pml_1' refers to unknown profile 'port_load'.`.

`cst-patch-farfield` attempt 2 (environment_output_permission): `PermissionError: [Errno 13] Unable to synchronously create file (unable to open file: name = 'testing/other_codes/cst/patch_antenna/farfield/patch_recentered_closed_ntff.h5', errno = 13, error message = 'Permission denied', flags = 13, o_flags = 302)`.

An intermediate patch-farfield attempt completed field stepping but could not overwrite the existing HDF5 file because its filesystem permissions denied write access. This is an output-permission failure, not a numerical acceptance failure; it is recorded separately from the original input-profile error.

The retained patch-farfield input referenced an undefined `port_load` profile. Removing only that trailing token from its `#pml_slab` line selects the existing default HORIPML/CFS profile. The 29-cell slab geometry, antenna/source/NTFF coordinates, and 20 ns record remain the same. The corrected input was rerun; the initial setup failure is retained separately.

An elevated preflight also failed because its working directory was incorrect. It did not launch the solver and is not counted as a simulation.

## Reproduction commands

Run the following repository-relative commands in the stated working directory. Commands are recorded executions unless explicitly labeled as a reproduction command; the manual simulation command was not captured. PML per-case inputs are generated by the sweep command; generated raw inputs are not tracked.

### CPU analytical and comparison workflows

Working directory: `.`.

```text
python -m testing.validation.validate_fdfd_eigenmodes
```

```text
python -m testing.validation.eigenmode_multiport_deembedding.validate_rectangular_waveguide
```

```text
python -m testing.validation.impedance_surface.validate_copper_wall_waveguide --threads 4
```
Working directory: `examples/features/impedance_surface/rectangular_waveguide_comparison`.

```text
python run_comparison.py --threads 4
```

```text
python plot_results.py
```

### GPU workflows and post-processing

Working directory: `.`.

```text
python -m gprMax testing/regression/eigenmode_sources/bending_waveguide/2d_te/large_bend/large_bend.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/bending_waveguide/2d_te/medium_bend/medium_bend.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/bending_waveguide/2d_te/small_bend/small_bend.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/bending_waveguide/2d_tm/large_bend/large_bend.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/bending_waveguide/2d_tm/medium_bend/medium_bend.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/bending_waveguide/2d_tm/small_bend/small_bend.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/broadband_vs_single_frequency/broadband/broadband.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/broadband_vs_single_frequency/single_frequency/single_frequency.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/grid_spacing/rectangular_waveguide/dx_0p05mm/rectangular_dx_0p05mm.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/grid_spacing/rectangular_waveguide/dx_0p10mm/rectangular_dx_0p10mm.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/grid_spacing/rectangular_waveguide/dx_0p20mm/rectangular_dx_0p20mm.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/loss_comparison/lossy/lossy.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/loss_comparison/nonlossy/nonlossy.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/straight_waveguide/2d_te/dielectric_waveguide/dielectric_waveguide.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/straight_waveguide/2d_tm/dielectric_waveguide/dielectric_waveguide.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/straight_waveguide/3d/cylindrical_waveguide/cylindrical_waveguide.in -gpu 0 --hide-progress-bars
```

```text
python -m gprMax testing/regression/eigenmode_sources/straight_waveguide/3d/rectangular_waveguide/rectangular_waveguide.in -gpu 0 --hide-progress-bars
```

```text
python testing/regression/eigenmode_sources/validate_sparameters.py testing/regression/eigenmode_sources
```

```text
python testing/regression/eigenmode_sources/plot_snapshots.py testing/regression/eigenmode_sources/bending_waveguide/2d_te/large_bend testing/regression/eigenmode_sources/bending_waveguide/2d_te/medium_bend testing/regression/eigenmode_sources/bending_waveguide/2d_te/small_bend testing/regression/eigenmode_sources/bending_waveguide/2d_tm/large_bend testing/regression/eigenmode_sources/bending_waveguide/2d_tm/medium_bend testing/regression/eigenmode_sources/bending_waveguide/2d_tm/small_bend testing/regression/eigenmode_sources/broadband_vs_single_frequency/broadband testing/regression/eigenmode_sources/broadband_vs_single_frequency/single_frequency testing/regression/eigenmode_sources/grid_spacing/rectangular_waveguide/dx_0p05mm testing/regression/eigenmode_sources/grid_spacing/rectangular_waveguide/dx_0p10mm testing/regression/eigenmode_sources/grid_spacing/rectangular_waveguide/dx_0p20mm testing/regression/eigenmode_sources/loss_comparison/lossy testing/regression/eigenmode_sources/loss_comparison/nonlossy testing/regression/eigenmode_sources/straight_waveguide/2d_te/dielectric_waveguide testing/regression/eigenmode_sources/straight_waveguide/2d_tm/dielectric_waveguide testing/regression/eigenmode_sources/straight_waveguide/3d/cylindrical_waveguide testing/regression/eigenmode_sources/straight_waveguide/3d/rectangular_waveguide
```

```text
python testing/regression/eigenmode_sources/plot_sparameters.py testing/regression/eigenmode_sources
```

```text
python testing/regression/eigenmode_sources/bending_waveguide/plot_bend_comparison.py testing/regression/eigenmode_sources/bending_waveguide
```

```text
python testing/regression/eigenmode_sources/broadband_vs_single_frequency/plot_source_comparison.py testing/regression/eigenmode_sources/broadband_vs_single_frequency
```

```text
python testing/regression/eigenmode_sources/grid_spacing/plot_grid_spacing.py testing/regression/eigenmode_sources/grid_spacing
```

```text
python testing/regression/eigenmode_sources/loss_comparison/plot_loss_comparison.py testing/regression/eigenmode_sources/loss_comparison
```

```text
python examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide.py --gpu 0
```

```text
python examples/features/eigenmode_ports/example_1_straight_waveguide/plot_results.py
```

```text
python examples/features/eigenmode_ports/example_2_curved_waveguide/curved_waveguide.py --gpu 0
```

```text
python examples/features/eigenmode_ports/example_2_curved_waveguide/plot_results.py
```

```text
python examples/features/eigenmode_ports/example_3_antenna_and_farfield/horn_antenna.py --gpu 0
```

```text
python examples/features/eigenmode_ports/example_3_antenna_and_farfield/plot_results.py
```

```text
python examples/features/eigenmode_ports/example_4_complete_s_matrix/complete_s_matrix.py --gpu 0
```

```text
python examples/features/eigenmode_ports/example_4_complete_s_matrix/plot_results.py
```

```text
python examples/features/eigenmode_ports/example_5_phased_array/phased_array.py --gpu 0
```

```text
python examples/features/eigenmode_ports/example_5_phased_array/plot_results.py
```

```text
python examples/features/eigenmode_ports/example_6_near_cutoff/near_cutoff.py --gpu 0
```

```text
python examples/features/eigenmode_ports/example_6_near_cutoff/plot_results.py
```

```text
python -m gprMax testing/validation/rectangular_waveguide_partial_cutoff/rectangular_waveguide_partial_cutoff.in -gpu 0 --hide-progress-bars
```

```text
python testing/validation/rectangular_waveguide_partial_cutoff/plot_partial_cutoff.py
```

```text
python -m gprMax testing/other_codes/cst/horn_antenna/horn_antenna.in -gpu 0 --hide-progress-bars
```

```text
python testing/other_codes/cst/horn_antenna/plot_horn_results.py
```

```text
python -m gprMax testing/other_codes/cst/patch_antenna/S-parameter/patch_antenna.in -outputfile testing/other_codes/cst/patch_antenna/S-parameter/patch_antenna -gpu 0 --hide-progress-bars
```

```text
python testing/other_codes/cst/patch_antenna/S-parameter/plot_patch_sparameters.py
```

Reproduction command (the exact manual invocation was not recorded):

```text
python -m gprMax testing/other_codes/cst/patch_antenna/farfield/patch_antenna_recentered_closed_ntff_backed_pml.in -gpu 0
```

```text
python testing/other_codes/cst/patch_antenna/farfield/plot_patch_farfield_comparison.py
```

### Internal-PML sweep and fixtures

Working directory: `.`.

```text
python -u testing/experimental/internal_pml_slab/run_waveguide_length_sweep.py --force --output-dir .pytest_cache/eigenmode-rerun-20260904/pml-length
```

```text
python -m gprMax testing/experimental/internal_pml_slab/rectangular_waveguide_external_pml.in -outputfile .pytest_cache/eigenmode-rerun-20260904/pml-rectangular_waveguide_external_pml
```

```text
python -m gprMax testing/experimental/internal_pml_slab/rectangular_waveguide_internal_pml.in -outputfile .pytest_cache/eigenmode-rerun-20260904/pml-rectangular_waveguide_internal_pml
```

### Automated tests

Working directory: `.`.

```text
python -m pytest -m 'not gpu' -q --basetemp .pytest_cache/eigenmode-rerun-20260904/temp-nongpu --junitxml .pytest_cache/eigenmode-rerun-20260904/nongpu.junit.xml --durations=20 tests/cmds_multiuse/test_eigenmode_commands.py tests/cmds_multiuse/test_eigenmode_source_2d.py tests/cmds_multiuse/test_eigenmode_subgrid.py tests/cmds_multiuse/test_virtual_waveguide_commands.py tests/fdfd_eigenmode_solver/test_fdfd_1d_mode_solver.py tests/fdfd_eigenmode_solver/test_fdfd_2d_mode_solver.py tests/fdfd_eigenmode_solver/test_fdfd_validation.py tests/fdfd_eigenmode_solver/test_numerical_dispersion.py tests/fdfd_eigenmode_solver/test_surface_impedance_fdfd_operator.py tests/fdfd_eigenmode_solver/test_surface_impedance_operator.py tests/fdfd_eigenmode_solver/test_virtual_waveguide_device.py tests/impedance_surfaces/test_copper_wall_waveguide.py tests/ntff/test_hash_command.py tests/ntff/test_port_power.py tests/ntff/test_solver_integration.py tests/ntff/test_validation_guards.py tests/outputs/test_fields_outputs_unit.py tests/outputs/test_sar.py tests/ports/test_port_study.py tests/test_eigenmode_anchor_policy.py tests/test_eigenmode_auto_guard_failure.py tests/test_eigenmode_config.py tests/test_eigenmode_grid_spacing_plot.py tests/test_eigenmode_numerical_dispersion.py tests/test_eigenmode_plotting.py tests/test_eigenmode_ports.py tests/test_eigenmode_source_broadband.py tests/test_eigenmode_source_upstream_compat.py tests/test_eigenmode_sparameter_plot.py tests/test_eigenmode_study.py tests/test_geometry_fixed_rejects_stateful_sources.py tests/test_virtual_waveguide_coupling.py tests/test_virtual_waveguide_integration.py tests/toolboxes/test_plot_port.py tests/updates/test_mpi_transmission_line_timing.py tests/updates/test_solver.py tests/updates/test_updates_base.py
```

```text
python -m pytest tests/fdfd_eigenmode_solver/test_virtual_waveguide_device.py tests/test_eigenmode_ports.py tests/test_virtual_waveguide_coupling.py tests/test_virtual_waveguide_integration.py -m gpu -q --gpu-device=0 --basetemp=.pytest_cache/eigenmode-rerun-20260904/gpu-pytest-tmp --junitxml=.pytest_cache/eigenmode-rerun-20260904/gpu-pytest.xml --tb=short
```

```text
python -m pytest -m gpu -q --gpu-device=0 --opencl-device=0 --basetemp .pytest_cache/eigenmode-rerun-20260904/temp-gpu-studies --junitxml .pytest_cache/eigenmode-rerun-20260904/gpu-studies.junit.xml --durations=20 tests/ports/test_port_study.py tests/test_eigenmode_study.py
```

```text
python -m pytest -m 'not gpu' -q --basetemp .pytest_cache/eigenmode-rerun-20260904/temp-fixture-rerun --junitxml .pytest_cache/eigenmode-rerun-20260904/fixture-rerun.junit.xml tests/test_eigenmode_source_upstream_compat.py
```

## Retained and local outputs

Tracked compact CSV/JSON/PNG results remain in their established validation, PML and CST directories. The horn case retains its far-field comparison only; the ad-hoc dense S-parameter input, plotting helper and outputs remain ignored locally. Tutorial/regression HDF5, CSV, field plots, snapshots, temporary PML inputs, compiler caches, and detailed command logs remain ignored local working data. Their paths are recorded for audit of this checkout, but they are not independent reference data and are not added to Git. The final report retains completion and numerical evidence even where raw files are ignored.

No output from an unfinished workflow is presented as newly validated evidence.
