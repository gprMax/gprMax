# Remaining FDFD dispersion fixes: targeted validation

Completed on 2026-09-04 against the changes following `eb66454691ad3ba96ae1819493045154e30e9c42` on `update_fdfd_dispersion`. This validates static conductivity compensation in dispersive materials and interpolation of broadband operator indices before the inverse spatial-difference mapping. Exact bulk ADE pole matching remains a documented follow-up.

The earlier [57-case report](report.md) describes the previous baseline. Only the five cases below were rerun for this change. Commands, source/output SHA-256 hashes, measured values, and HDF5 dimensions are recorded in [remaining_dispersion_fixes.json](remaining_dispersion_fixes.json).

## Tests

- **305 passed**, 30 GPU-marked tests deselected, in 179.62 seconds in the focused source, monitor, FDFD, anchor-policy, cached-study, subgrid, and virtual-waveguide suite.
- **52 passed** in the material-focused run, including 31 tests also present in the focused suite. Coverage includes Debye, Lorentz, Drude, and inclusive materials; serial/MPI extraction; omitted timestep; and PEC masking.
- Analytic broadband regressions exercise signed and complex operator indices, off-anchor TEM propagation, exact anchors and endpoint extrapolation, one-anchor and legacy compatibility, cached mode selection, physical-cutoff branches, spatial-stop-band rejection/filtering, and conditioned generalized monitor coefficients.
- The signed/complex single-frequency staggering regressions from `eb664546` remain passing.

The focused command, from the repository root in the local `gprMax` conda environment, was:

```powershell
python -m pytest -m 'not gpu' -q --basetemp .pytest_cache/remaining-dispersion-suite tests/test_eigenmode_numerical_dispersion.py tests/test_eigenmode_source_broadband.py tests/test_eigenmode_ports.py tests/test_eigenmode_anchor_policy.py tests/test_eigenmode_auto_guard_failure.py tests/test_eigenmode_source_upstream_compat.py tests/test_eigenmode_study.py tests/ports/test_port_study.py tests/fdfd_eigenmode_solver tests/cmds_multiuse/test_eigenmode_source_2d.py tests/cmds_multiuse/test_eigenmode_subgrid.py tests/test_virtual_waveguide_integration.py
```

## FDTD reruns and numerical gates

All five simulations exited successfully on CUDA device 0, an NVIDIA GeForce RTX 4070 Laptop GPU. The runtime used CUDA 13.3, driver 596.49, MSVC toolset 14.44, and the `gprMax` conda environment. Each model was run with `python -m gprMax <input> -gpu 0 --hide-progress-bars`; the exact input paths are in the JSON record. Wall times include process setup, modal solves, and output writing.

| Case | Wall time (s) | Measured result |
| --- | ---: | --- |
| Broadband multi-anchor | 26.56 | Mean absolute S21 magnitude in dB: 0.002062 |
| Single-frequency profile | 8.08 | Mean absolute S21 magnitude in dB: 0.078834 |
| Lossy waveguide | 13.76 | Mean S21: -6.126876 dB |
| Nonlossy waveguide | 13.63 | Mean S21: -0.001965 dB |
| Partial-cutoff rectangular waveguide | 49.84 | Maximum magnitude error 0.353233 dB; maximum circular phase error 2.145409 degrees |

The broadband profile remains closer to 0 dB transmission than the single-frequency profile. The lossy/nonlossy separation exceeds the existing 3 dB requirement. Both comparisons retain 126 valid fundamental-mode power bins per case. The partial-cutoff errors remain below the existing 0.45 dB and 3 degree limits; every generalized coefficient is valid, power validity is restricted to frequencies above cutoff, and the existing settled-S11 gate passes.

The checks were performed with the maintained `validate_sparameters.py` on each comparison directory and `plot_partial_cutoff.py` on the partial-cutoff case. S-parameter plots, comparison plots, field snapshots, and the partial-cutoff plot were regenerated from the fresh outputs. Existing ignore rules for raw outputs and regression plots were retained.

All ten source/receiver port groups in the five fresh HDF5 outputs contain finite per-bin `beta` with `Units="rad/m"` and the new `anchor_operator_neff` dataset. The beta axes are `(frequency, mode)`; operator-anchor axes are `(anchor, mode)`. Existing physical-index datasets retain their meaning.

The untracked CST patch-antenna S-parameter CSV was left untouched during these targeted reruns and was subsequently removed at the user's explicit request during PR validation. Horn results and unrelated simulation cases were not regenerated for this targeted comparison.

## Diagnostic and light-speed consistency follow-up

Single-frequency I/Q selection now records and reports the actual reasons: complex modal profile, drive phase/delay, and complex longitudinal staggering. Negative real propagation remains on the signed real-only path. Monitor preparation and compatibility finalization now use the simulation electromagnetic constants consistently with source staggering and the FDFD solvers.

After this cleanup, **149 source, monitor, and numerical-dispersion tests passed**. The checks include individual/combined I/Q reasons and a configured light speed differing from the module-level constant, covering real-only and I/Q source injection, device envelope assembly, and normal/legacy monitor decomposition. The five GPU measurements and their source hashes above precede this cleanup; the follow-up does not change propagation with the default production constants.

## Full PR validation

On 2026-09-04, the complete pytest suite was run without marker exclusions in the local `gprMax2` environment (Python 3.13.15), with CUDA device 0 and OpenCL enabled. The command was:

```powershell
python -u -m pytest -q --gpu-device 0 --basetemp <runtime>/pytest-tmp --junitxml <runtime>/full.xml --durations=25
```

The full run completed in 1213.45 seconds with **6,508 passed, 24 skipped, and one MATLAB startup failure**. MATLAB R2025b could not start inside the sandbox (`System Error: File system inconsistency`), before executing any test assertions. The same unchanged integration test passed outside the sandbox with a fresh host temporary directory: **1 passed in 35.15 seconds**. The final result across the full run and that environment-only rerun is **6,509 passed, 24 skipped, with no unresolved failures**.

CUDA and OpenCL tests executed successfully. The XML contains 121 passing test IDs containing `cuda` and 117 containing `opencl`; these name-based counts can overlap when a test module covers both backends. The suite reported 284 warnings, including PyCUDA notices that compilation succeeded with compiler output.

The skips comprise two optional CAD/visualisation modules (pythonocc-core and PyVista), one mpi4py-fft test, three tests requiring mpiexec, three requiring MPI-enabled h5py, seven optional geometry-import tests (NIfTI, NRRD, MetaImage, and VTK), six existing unimplemented tests, and two Apple Metal tests. No CUDA or OpenCL hardware tests were skipped.

Before the complete run, an output-path test was corrected to use an isolated temporary path rather than assume the repository has no `out` directory. Three related filesystem side-effect tests now use `tmp_path`; all 30 tests in that module passed. An initial Python 3.14 attempt also exposed an outdated local plane-wave Cython binary. Validation therefore used the current Python 3.13 extensions; all 25 compiled extensions were checked against their source timestamps, and the previously failing plane-wave comparison passed. No production code was changed to accommodate either environment issue.

The final source hashes, full-run counts, successful MATLAB rerun, backend name counts, skip reasons, and slowest tests are recorded in [pr_validation.json](pr_validation.json). The eigenmode documentation examples were also checked for Python syntax/API binding, and the expanded study section parsed without reStructuredText warnings; a full Sphinx build was not run because Sphinx is unavailable in the local environment.
