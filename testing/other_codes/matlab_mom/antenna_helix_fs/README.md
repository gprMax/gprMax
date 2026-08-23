# Three-turn helix: gprMax FDTD/KSIR versus MATLAB MoM

This case compares a three-turn, ground-plane-backed axial-mode helix in
gprMax with the `helix` Method-of-Moments model in MATLAB Antenna Toolbox.
It compares quantities that are not exercised by the linearly polarised
dipole and patch examples: circular-polarisation handedness, axial ratio,
front-to-back ratio, and axial beamwidth, as well as directivity, gain,
realised gain, and efficiency.

The models use a 22 mm helix radius, 35 mm turn spacing, three turns, a 2 mm
feed stub, and a 75 mm-radius circular ground plane. They are evaluated at
2.2 GHz, where the helix circumference is approximately one wavelength and
the antenna operates in its circularly polarised axial mode. The 1.56 GHz
resonance of the default MATLAB object is not its best circular-polarisation
point.

The curved gprMax conductor is a connected chain of 72 arbitrary-axis PEC
cylinders. Fifteen-degree chords have less than 0.2 mm sag and, after their
endpoints are rounded to the 1 mm grid, the centreline length differs from the
analytical helix by less than 0.5 percent. This avoids a Manhattan staircase,
which would spuriously lengthen the conductor by almost 50 percent. The
cylinder radius is 1 mm. MATLAB uses its documented equivalent strip width
relation `w = 4r`, hence a 4 mm strip. The circular zero-thickness ground plane
is represented by 72 triangular PEC sectors rather than a square plate.

A one-cell voltage source drives a one-cell PEC feed stub from the ground
plane. The gprMax script can use either the original 150 Ohm resistive source
or a zero-resistance hard source that prescribes the delta-gap voltage and
derives terminal current from the surrounding Ampere loop. Both use a 150 Ohm
travelling-wave reference. The full 2 degree sphere has 16,380 paired
directions. gprMax stores the complex `Etheta` and `Ephi` fields and all
antenna metrics in its HDF5 output; the comparison script reads that file
after the blocking simulation completes.

The circular components are calculated as

```text
E+ = (Etheta + j Ephi) / sqrt(2)
E- = (Etheta - j Ephi) / sqrt(2).
```

The comparison explicitly maps these components to MATLAB RHCP/LHCP because
opposite phasor time conventions exchange the two labels. Axial ratio is then
calculated from the circular-component amplitudes and independently checked
against MATLAB's native `axialRatio` result.

## Run

From the repository root:

```bash
MPI4PY_RC_INITIALIZE=0 conda run --no-capture-output -n gprMax-devel python testing/other_codes/matlab_mom/antenna_helix_fs/helix_antenna_gprmax.py --gpu 0
matlab -batch "run('testing/other_codes/matlab_mom/antenna_helix_fs/helix_antenna_matlab.m')"
MPLCONFIGDIR=/tmp/matplotlib-gprmax conda run --no-capture-output -n gprMax-devel python testing/other_codes/matlab_mom/antenna_helix_fs/plot_helix_comparison.py
```

Run and plot the ideal hard-source variant with:

```bash
MPI4PY_RC_INITIALIZE=0 conda run --no-capture-output -n gprMax-devel python testing/other_codes/matlab_mom/antenna_helix_fs/helix_antenna_gprmax.py --source-mode hard --gpu 0
MPLCONFIGDIR=/tmp/matplotlib-gprmax conda run --no-capture-output -n gprMax-devel python testing/other_codes/matlab_mom/antenna_helix_fs/plot_helix_comparison.py --source-mode hard
```

Omit `--gpu 0` to use the Cython CPU solver. Use `--geometry-only` to create
`results/helix_antenna_geometry.vtkhdf` for inspection in ParaView without
running the time-domain solver.

The principal outputs are:

- `results/helix_antenna_gprmax.h5`
- `results/helix_antenna_geometry.vtkhdf`
- `results/helix_antenna_comparison.png`
- `results/helix_antenna_gprmax_3d_realized_gain.png`
- `results/helix_antenna_comparison_metrics.json`
- `results/helix_antenna_gprmax_hard.h5`
- `results/helix_antenna_comparison_hard.png`
- `results/helix_antenna_comparison_hard_metrics.json`

## Current result

At 2.2 GHz, gprMax and MATLAB predict peak directivities of 10.1468 and
10.2852 dBi, respectively. Their +z axial ratios are 2.0219 and 2.0424 dB,
and their half-power beamwidths are 54.64 and 53.79 degrees. Front-to-back
ratios are 11.88 and 12.36 dB. The directivity RMS difference over directions
within 25 dB of each pattern maximum is 0.498 dB.

Both conductors are lossless, so the calculated radiation efficiencies are
100.0084 percent for gprMax and 100 percent for MATLAB; the small gprMax excess
is numerical integration error. The 150 Ohm mismatch efficiencies are 89.96
and 95.71 percent. Consequently, peak realised gains are 9.6876 and 10.0946
dBi, a larger difference than the 0.1385 dB directivity difference. The
corresponding input impedances are approximately `83.1 - j30.2 Ohm` and
`114.6 - j42.8 Ohm`, showing that most of the realised-gain difference is a
feed-discretisation effect rather than a radiation-pattern error.

The CUDA field update took 67.4 seconds on an NVIDIA TITAN RTX; the complete
run, including geometry construction, HDF5 writing, and full-sphere KSIR
post-processing, took 2 minutes 4 seconds.

### Ideal hard-source result

With the zero-resistance voltage source, gprMax predicts
`85.9 - j22.5 Ohm`, a -10.85 dB S11 magnitude against 150 Ohms, and a 9.668
dBi peak realised gain. The real impedance moves 2.8 Ohms towards MATLAB, but
the reactive discrepancy grows; this feed change therefore does not remove
the dominant FDTD/MoM geometry and conductor-discretisation differences. The
radiation pattern remains excellent: peak directivity differs by 0.146 dB,
axial ratio by 0.003 dB, and half-power beamwidth by 0.92 degrees.

The hard-port accepted-power balance gives 97.74 percent apparent radiation
efficiency for this otherwise lossless model, compared with 100.01 percent
for the resistive-source result. This 2.3 percent residual is a useful limit
of the local one-cell Ampere-loop power estimate in this feed geometry; the
directivity and polarisation comparisons do not depend on that port-power
normalisation. The hard-source implementation is therefore valuable for
MoM-like voltage/current impedance experiments, but it should not be assumed
to improve every absolute gain result automatically.

The MATLAB geometry and strip/cylinder equivalence are described in the
[MathWorks helix documentation](https://www.mathworks.com/help/antenna/ref/helix.html).
