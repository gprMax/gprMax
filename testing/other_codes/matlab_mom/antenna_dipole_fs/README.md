# Half-wave dipole: gprMax FDTD/KSIR versus MATLAB MoM

This case compares a centre-fed thin-wire dipole modelled by gprMax with a
`dipoleCylindrical` Method-of-Moments model from MATLAB Antenna Toolbox and
the closed-form infinitesimal-radius half-wave radiation pattern.

The gprMax antenna consists of two 75 mm PEC Yee-edge arms separated by a
one-cell, 1 mm voltage-source gap. Its outer length is therefore 151 mm. A
conventional PEC wire represented by forcing the axial electric field to zero
on a three-dimensional Cartesian FDTD grid has an effective radius of
approximately 0.23 cell widths. The MATLAB cylinder consequently uses a
0.23 mm radius for the 1 mm gprMax grid. This convention is discussed, for
example, by [Taniguchi et al.](https://doi.org/10.1541/ieejpes.128.263).

Both models use a 73 Ohm reference impedance. The gprMax source is a one-cell
resistive voltage source, so its incident voltage is one half of the Thevenin
generator waveform. Two independent port calculations are made from the
persisted receiver histories after the blocking simulation returns:

- incident/reflected voltage separation using the known Thevenin waveform;
- input impedance from the source-edge voltage and the Ampere-contour current,
  followed by `S11 = (Zin - 73) / (Zin + 73)`.

For a z-directed source edge, gprMax's native current output is

```text
Iz = dx [Hx(i,j-1,k) - Hx(i,j,k)]
   + dy [Hy(i,j,k) - Hy(i-1,j,k)].
```

This is the discrete line integral of the four circulating magnetic-field
samples and includes the total current enclosed by the Yee dual face. Native
`Ix`, `Iy`, and `Iz` receiver outputs are currently available with the CPU
solver only. The comparison therefore saves the same four H samples at three
receiver positions and reconstructs exactly this expression, allowing the
identical method to run with CUDA, OpenCL, and Metal. The adjacent electric
voltages are averaged onto the magnetic-field half-time step before taking
`Zin = V / I`.

The engineering-convention FFT uses a 100 ns record, giving 10 MHz independent
resolution, with eight-times zero padding for a 1.25 MHz sampled frequency
grid.

A closed KSIR surface records the 0.95 GHz field. In addition to the x-z
elevation and x-y azimuth cuts, a common 2 degree full-sphere grid is used to
compare absolute directivity, gain, realised gain, radiation intensity, and
radiation, mismatch, and total efficiencies. The gprMax values are read from
the persisted HDF5 output. The expected elevation field of an ideal half-wave
dipole is proportional to

```text
cos((pi / 2) cos(theta)) / sin(theta),
```

and the azimuth pattern is omnidirectional. The original cut comparison
normalises the patterns to test their shape. The full-sphere metric comparison
retains the absolute normalisation and uses the same 73 Ohm reference impedance
in both solvers.

The one-cell gprMax feed gap is not identical to the delta-gap MoM port. The
effective-radius equivalence is also an approximation, so S11 agreement is a
stronger and more grid-sensitive test than the far-field pattern.

## Run

From the repository root:

```bash
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_dipole_fs/dipole_antenna_gprmax.py --gpu 0
matlab -batch "run('testing/other_codes/matlab_mom/antenna_dipole_fs/dipole_antenna_matlab.m')"
MPLCONFIGDIR=/tmp/matplotlib-gprmax conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_dipole_fs/plot_dipole_comparison.py
MPLCONFIGDIR=/tmp/matplotlib-gprmax conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_dipole_fs/plot_dipole_metric_comparison.py
```

Omit `--gpu 0` to use the Cython CPU solver. Use `--geometry-only` to generate
only `results/dipole_antenna_geometry.vtkhdf`, or `--postprocess-only` to
regenerate the gprMax CSV files from the existing HDF5 output.

The generated comparison products are:

- `results/dipole_pattern_comparison.png`
- `results/dipole_s11_comparison.png`
- `results/dipole_impedance_comparison.png`
- `results/dipole_comparison_metrics.json`
- `results/dipole_antenna_metric_comparison.png`
- `results/dipole_antenna_metric_comparison.json`
- `results/dipole_antenna_geometry.vtkhdf`

The older `testing/models_basic/antenna_wire_dipole_fs` input remains unchanged
as a compact regression model; this directory is the reproducible cross-solver
comparison and example case.

## Current result

The CUDA gprMax model contains 15.625 million cells and 51,927 time steps. On
an NVIDIA TITAN RTX, the solve took 3 minutes 29 seconds and used approximately
828 MB host plus 1.1 GB device memory.

The interpolated 73 Ohm S11 minima are 0.9420 GHz for gprMax and 0.9482 GHz for
MATLAB, a 6.17 MHz difference that is smaller than the 10 MHz independent
resolution of the time record. Their -10 dB bandwidths are 90.62 and
95.67 MHz, respectively. The depths of both very narrow minima are close to
-43 dB but should not be over-interpreted at this time-window resolution.

The independent gprMax magnetic-contour calculation gives an interpolated S11
minimum of 0.9416 GHz and a -10 dB bandwidth of 90.56 MHz. Its complex S11
differs from the source-wave calculation by only 0.0056 RMS across the complete
0.55--1.35 GHz band. At 0.9400 GHz, the contour gives an input impedance of
`71.57 - j1.54 Ohm`, compared with `70.01 - j7.79 Ohm` from MATLAB. Across the
complete band the gprMax/MATLAB RMS differences are 1.82 Ohm in resistance and
16.88 Ohm in reactance.

The normalised patterns agree extremely well. In the x-z plane, gprMax differs
from MATLAB by 0.021 dB RMS and 0.067 dB maximum away from the nulls; it differs
from the ideal half-wave expression by 0.020 dB RMS and 0.033 dB maximum. The
gprMax x-y azimuth ripple is only 0.000025 dB.

The absolute full-sphere comparison at 0.95 GHz gives peak directivities of
2.1451 dBi from gprMax and 2.1075 dBi from MATLAB, a difference of 0.0376 dB.
Their directivity patterns differ by 0.0252 dB RMS above the -30 dB floor, or
0.0191 dB RMS after normalisation. Peak realised gain differs by 0.0237 dB.
The gprMax radiation efficiency is 100.0075 percent (the small excess is
numerical integration error), compared with 100 percent for the lossless
MATLAB model. The corresponding 73 Ohm mismatch efficiencies are 99.6513 and
99.9793 percent.
