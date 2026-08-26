# MATLAB Antenna Toolbox comparison suite

These directories contain reproducible free-space antenna comparisons between
gprMax FDTD/KSIR and MATLAB Antenna Toolbox Method-of-Moments models. They are
numerical inter-code comparisons and research examples, not validation against
an exact reference solution or small pytest regressions. The finest models
require a CUDA GPU, MATLAB with Antenna Toolbox, and minutes of runtime.

The current complementary cases are:

| Directory | Capability exercised |
|---|---|
| `antenna_dipole_fs` | Thin wire, closed-form pattern, source-edge current, one-port S11 |
| `antenna_monopole_fs` | Finite PEC ground plate, edge diffraction, back radiation |
| `antenna_bowtie_fs` | Planar triangular PEC rasterisation, x-directed feed |
| `antenna_dipole_array_fs` | Coherent dual feeds, mutual coupling, active impedance, array factor |
| `antenna_patch_fs` | Dielectric substrate, finite ground, probe feeds, mesh convergence |
| `antenna_reflector_grounded` | Strip dipole above an infinite PEC plane, five-face layered NTFF |
| `rcs_comparison` | Plane-wave scattering, absolute monostatic RCS, PEC plates |

Every modern case follows the same evidence chain:

1. build and run the gprMax model;
2. persist receiver and KSIR data in the normal gprMax HDF5 output;
3. reopen that HDF5 file and check dataset shapes, coordinates, and values;
4. run the independent MATLAB MoM model on a shared physical geometry;
5. create plots and machine-readable quantitative metrics; and
6. write a fine `vtkhdf` geometry for inspection in ParaView.

The patterns are globally normalised and test angular shape, not realised
gain. Port comparisons are more sensitive because FDTD Yee-edge feeds and
staircased conductors cannot be exactly identical to MoM delta-gap ports and
continuous surfaces. Each case README states its equivalence assumptions,
independent spectral resolution, current definition, and current numerical
result.

Neither solver is treated as ground truth. Differences quantify the combined
effects of the numerical methods, meshes, conductor representations, and feed
models. Analytical reference cases used for correctness validation are kept
separately in ``testing/validation``.

Large solver `.h5` and `.vtkhdf` files are reproducible working products and
should remain outside a source-only pull request. Compact plots, tables, and
metrics may be retained as reviewable evidence. MATLAB `.mat` files containing
the independent reference curves are deliberately retained: this allows users
without MATLAB or Antenna Toolbox to rerun the Python comparisons. Each case
README identifies which generated products form its portable reference set.
