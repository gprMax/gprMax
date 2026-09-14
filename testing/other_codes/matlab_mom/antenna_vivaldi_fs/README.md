# Free-space PEC Vivaldi: MATLAB MoM versus gprMax

This is an **independent numerical-code comparison**, not an analytical
validation or a claim that the two feed models are identical. It reproduces
the standard MATLAB `vivaldi` antenna in air. There is no dielectric board,
microstrip transition, coaxial connector, finite-conductivity metal, or ground
half-space. The finite sheet surrounding the slot is the antenna itself.

## Geometry and excitation

The MATLAB script explicitly sets the catalogue parameters:

| Parameter | Value |
|---|---:|
| PEC sheet length × width | 300 × 125 mm |
| Exponential taper length | 243 mm |
| Aperture width | 105 mm |
| Opening rate | 25 |
| Slot width | 0.5 mm |
| Circular cavity diameter | 24 mm |
| Cavity-to-taper spacing | 23 mm |
| Feed x, relative to sheet centre | −104.5 mm |
| Reference impedance | 50 ohm |

`vivaldi_antenna_matlab.m` exports the actual polygon returned by
`pcbStack(vivaldi(...))` to `results/vivaldi_geometry.json`. It contains the
exterior and cavity loops in metres. The gprMax script reads this portable
outline; it does not require MATLAB or reconstruct the taper from a separate
formula. The outline is translated by `(220.5, 112.25, 72)` mm into a
`440 × 226 × 144` mm air domain, with no rotation or rescaling.

MATLAB uses its delta-gap MoM feed. The exported polygon contains a small
metal feed bridge. The FDTD rasteriser excludes the interior of that bridge
from the sheet and places one y-directed, 50-ohm `VoltageSource` across the
0.5 mm gap. The feed node is `(116, 112, 72)` mm; the physical Ey centre is
`(116, 112.25, 72)` mm. This is a finite Yee-edge excitation, not an exact
implementation of the MoM delta gap. In particular, the longitudinal extent
of a driven edge's associated cell changes when dx is refined.

The CAD polygon is sampled at xy **face centres**. Contiguous occupied faces
are grouped into ordinary zero-thickness `Plate` objects. Thus all four
tangential electric edges of each retained face are PEC, as in normal
gprMax plate construction. No volumetric PEC voxels, material averaging, or
new solver routines are used. The curved boundaries are staircased. The feed
guard checks that the driven Ey edge remains non-PEC and that both adjacent
conductor banks are present.

| Mesh | dx, dy, dz (mm) | Cells |
|---|---|---:|
| coarse | 2, 0.5, 2 | 7,159,680 |
| fine | 1, 0.5, 1 | 28,638,720 |
| finer | 0.5, 0.5, 1 | 57,277,440 |

These are directional mesh-sensitivity checks, **not isotropic convergence**:
dy is fixed so that the feed slot remains one Ey edge wide. The physical
antenna dimensions and source centre remain fixed. Each run uses a 20 ns
Gaussian-driven transient, 12-cell domain PMLs and a closed equivalent-current
NTFF surface. The stored feed-voltage tail is checked in post-processing.

## Run and inspect

From the repository root, with gprMax built in the active environment:

```bash
case=testing/other_codes/matlab_mom/antenna_vivaldi_fs
python "$case/vivaldi_antenna_gprmax.py" --mesh coarse --gpu 0
python "$case/vivaldi_antenna_gprmax.py" --mesh fine --gpu 0
python "$case/vivaldi_antenna_gprmax.py" --mesh finer --gpu 0
python "$case/inspect_vivaldi_geometry.py" --mesh finer
python "$case/plot_vivaldi_comparison.py"
```

Omit `--gpu` for CPU execution; use `--precision double` to request double
precision. These switches select real solver paths, not alternative
post-processing. To generate only geometry, add `--geometry-only` and choose
a separate `--results-dir` so it cannot replace an existing run's metadata.
Open `results/vivaldi_<mesh>_geometry.vtkhdf` in ParaView and threshold the
cell-data `Material` array to PEC (ID 0). The inspection script reads this
actual geometry output, checks every PEC edge against the expected masks,
and plots it together with the MATLAB outline and the feed gap.

The retained MAT, JSON and CSV references mean MATLAB is optional. To regenerate
the reference using MATLAB with Antenna Toolbox:

```matlab
addpath('testing/other_codes/matlab_mom/antenna_vivaldi_fs');
vivaldi_antenna_matlab;              % 8 mm maximum surface-mesh edge
vivaldi_antenna_matlab(0.006, '/tmp/vivaldi_matlab_6mm');
```

Use a suitable local output directory in the second command on Windows.
The reference script was executed with MATLAB R2024b, Antenna Toolbox 24.2.
Its MAT file contains numeric geometry, mesh and reference results, rather
than a serialized antenna object requiring that toolbox to inspect.

To replot the retained comparison without running either solver or retaining
large HDF5 files:

```bash
python "$case/plot_vivaldi_comparison.py" --no-export
```

For a subset of runs, use e.g. `--meshes coarse fine`. Compact results and
reference MAT files belong in the repository; raw `.h5` and `.vtkhdf` files
are reproducible, ignored working products.

## Quantities and conventions

The frequency sweep uses 21 points between 1 and 2 GHz. Native voltage-port
`Zin` and complex `S11` are read from the HDF5 file with their validity masks;
no legacy receiver-contour calculation or custom gap correction is introduced.
Complex interpolation onto the MATLAB grid is confined to valid adjacent bins.
The independent spectral spacing is approximately 50 MHz for the 20 ns run.

At 1, 1.5 and 2 GHz the full sphere is sampled every 5 degrees, and both
principal cuts every 2 degrees. In both full-circle plots, angle zero points
towards the aperture (+x):

- XY / E plane: direction `(cos(a), sin(a), 0)`;
- XZ / H plane: direction `(cos(a), 0, sin(a))`.

The comparison checks these directions against the saved NTFF theta/phi
coordinates. It uses total radiation intensity (both Etheta and Ephi).
Absolute directivity is `D = 4πU/Prad`, gain is `G = 4πU/Pacc`, and realised
gain is `4πU/Pinc`; gprMax integrates radiated power using its own full-sphere
quadrature, not a normalisation fitted to the MATLAB curves. For MATLAB's
lossless PEC model in air, gain equals directivity, and realised gain includes
the `1 − |S11|²` mismatch factor. gprMax's calculated radiation efficiency
provides a separate numerical power-balance diagnostic and is not forced to 1.

Plot clipping below −25 dBi is for display only. Pattern RMS differences use
reference samples above each MATLAB cut's peak minus 20 dB; they are absolute
dBi differences, not optimally rescaled pattern-shape errors. Impedance uses
complex relative L2 difference across the band, and S11 uses absolute complex
differences to avoid singular relative errors near a match.

`results/vivaldi_comparison_metrics.json` records measured differences and run
times. Read those with the mesh and feed limitations above; a successful run
or a near-unity radiation efficiency does not demonstrate that S11 has
converged to the independent MoM result. This is not an automated pass/fail
gate for the physical agreement of the solvers.

## Measured results and remaining disagreement

The retained September 2026 runs used MATLAB R2024b and real CUDA execution
on NVIDIA TITAN RTX devices, in single precision. CPU geometry-only generation
was also executed; no CPU time-domain, OpenCL, Metal or MPI comparison is
claimed for this antenna. The 8 mm MATLAB mesh contains 2,746 triangles;
the independent 6 mm mesh reference is retained in `results/matlab_mesh6mm`.

| gprMax mesh | Maximum absolute complex S11 difference | Complex Zin relative L2 difference | Run wall time |
|---|---:|---:|---:|
| coarse | 0.539 | 59.5% | 88.9 s |
| fine | 0.308 | 40.2% | 232.7 s |
| finer | 0.305 | 39.4% | 492.9 s |

**The input impedance has not converged to the MATLAB result.** Refining dx
from 1 to 0.5 mm produces little further improvement. These curves are retained
to expose this discrepancy, not presented as a passed impedance validation.
The one-cell slot, finite feed and unresolved refinement in y/z need further
investigation before attributing the difference to one particular cause.

Peak directivity is closer, but this does not establish agreement everywhere:

| Frequency | MATLAB | gprMax finer | E-plane RMS difference | H-plane RMS difference |
|---|---:|---:|---:|---:|
| 1 GHz | 3.166 dBi | 3.039 dBi | 0.539 dB | 1.967 dB |
| 1.5 GHz | 5.707 dBi | 5.720 dBi | 0.535 dB | 1.926 dB |
| 2 GHz | 7.188 dBi | 7.577 dBi | 0.473 dB | 2.257 dB |

The RMS region is defined above; secondary lobes and null positions are
more sensitive than the main beam. Calculated radiation efficiency for the
finer mesh is between 0.999812 and 1.000348, within 0.035% of the lossless
value. This checks the numerical power balance, not agreement with MATLAB's
different discretisation and feed. Values slightly greater than one are
retained, not clipped. Gain and realised gain are exported with the patterns;
realised gain remains sensitive to the unresolved impedance difference.

The 8-to-6 mm MATLAB mesh check changes complex Zin by 1.94% in relative L2,
complex S11 by at most 0.0108, and peak directivity by at most 0.034 dB.
Doubling the coarse FDTD transient from 20 to 40 ns changes complex S11 by at
most 0.000168 and Zin by at most 0.079 ohm. Neither check explains the main
inter-code discrepancy. The 20 ns feed-voltage tail is below 9e-5 of its peak
on all three retained grids. Run times include setup, solving, far-field
post-processing and output; they are not an equal-work CPU/GPU benchmark.

`check_vivaldi_sensitivity.py` reproduces these sensitivity metrics. For the
time-window check, first run:

```bash
python "$case/vivaldi_antenna_gprmax.py" --mesh coarse --gpu 0 --time-ns 40 --results-dir /tmp/vivaldi_40ns
python "$case/check_vivaldi_sensitivity.py" --time-dir /tmp/vivaldi_40ns
```

The lightweight tests in `tests/ntff/test_vivaldi_comparison_model.py` check
polygon membership, cavity preservation, symmetry, the feed gap, retained
reference arrays and input units. They do not execute either field solver.
`inspect_vivaldi_geometry.py` separately checked every PEC edge in the actual
three CUDA geometry exports and in the CPU geometry-only export.

## References

- MathWorks, [Vivaldi antenna catalogue object](https://www.mathworks.com/help/antenna/ref/vivaldi.html): geometry and parameter definitions.
- MathWorks, [Feed model](https://www.mathworks.com/help/antenna/ug/feed-model.html): Antenna Toolbox delta-gap excitation.

The comparison scripts use public MATLAB and gprMax interfaces; no solver
implementation is copied between the codes.
