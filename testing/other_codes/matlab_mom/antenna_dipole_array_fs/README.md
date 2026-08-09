# Two-element dipole array: gprMax FDTD/KSIR versus MATLAB MoM

This case tests coherent multiple-source excitation, mutual coupling, active
port impedance, and KSIR array patterns. It compares two equal-phase
z-directed thin-wire dipoles in gprMax with a MATLAB Antenna Toolbox
`linearArray` of two `dipoleCylindrical` elements.

Each element has a 75 mm outer length: two 37 mm PEC Yee-edge arms separated
by a one-cell, 1 mm voltage-source gap. Their centres are 80 mm apart along x.
The MATLAB wires use a 0.23 mm radius, corresponding to the established
approximately 0.23-cell effective radius of a 1 mm-grid axial FDTD wire. Both
50 Ohm voltage sources use the same waveform with zero relative phase.

The impedance reported here is the **active impedance per port** under that
simultaneous excitation, not the conventional single-port S11 obtained while
terminating the other element in 50 Ohms. MATLAB's `impedance(linearArray, f)`
returns one active impedance for each driven element. The gprMax calculation
independently reconstructs voltage and the Ampere-contour current at both
source edges:

```text
Iz = dx [Hx(i,j-1,k) - Hx(i,j,k)]
   + dy [Hy(i,j,k) - Hy(i-1,j,k)].
```

The active reflection coefficient is
`Gamma_active = (Zactive - 50) / (Zactive + 50)`. As a separate consistency
check, gprMax also calculates the reflected voltage at each simultaneously
driven Thevenin source. Both persisted port histories and the complete KSIR
result are read back from the HDF5 file only after the blocking simulation
returns.

The 60 ns engineering-convention record gives 16.67 MHz independent frequency
resolution; eight-times zero padding only interpolates between these samples.
The closed KSIR box records the 1.9 GHz array field. The x-z cut contains both
the element factor and the equal-phase array factor along the array axis; the
y-z transverse cut contains the element pattern without the x-axis phase
progression.

## Run

From the repository root:

```bash
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_dipole_array_fs/dipole_array_gprmax.py --gpu 0
matlab -batch "run('testing/other_codes/matlab_mom/antenna_dipole_array_fs/dipole_array_matlab.m')"
MPLCONFIGDIR=/tmp/matplotlib-gprmax conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_dipole_array_fs/plot_dipole_array_comparison.py
```

Omit `--gpu 0` for Cython CPU execution. Use `--geometry-only` to write only
the fine ParaView model, or `--postprocess-only` to regenerate the gprMax CSVs
from the existing HDF5 file.

The principal products are:

- `results/dipole_array_geometry.vtkhdf`
- `results/dipole_array_pattern_comparison.png`
- `results/dipole_array_active_gamma_comparison.png`
- `results/dipole_array_active_impedance_comparison.png`
- `results/dipole_array_comparison_metrics.json`

## Current result

The CUDA model has 15.625 million cells and 31,157 iterations. On an NVIDIA
TITAN RTX, its solve took 2 minutes 5.7 seconds and used approximately 843 MB
host plus 1.1 GB device memory. The fine geometry contains exactly 148 PEC arm
edges and two distinct source edges. Neither source was overwritten by PEC.

The 1.9 GHz patterns agree closely even through the array-factor structure. In
the x-z array-axis plane the gprMax/MATLAB difference is 0.056 dB RMS and
0.198 dB maximum away from the nulls. In the y-z transverse plane it is
0.040 dB RMS and 0.101 dB maximum.

The interpolated active-reflection minima are 1.9271 GHz and -20.34 dB for the
gprMax `V/I` result, 1.9279 GHz and -20.32 dB from gprMax voltage waves, and
1.9571 GHz and -21.21 dB from MATLAB. Across 1.65--2.15 GHz the active
resistance and reactance RMS differences are 3.06 and 12.83 Ohm. The two
independent gprMax active-reflection estimates differ by 0.0075 complex RMS.

The most sensitive implementation check is port symmetry. The RMS difference
between the two independently sampled gprMax active impedances is only
3.9 micro-Ohm (MATLAB: 0.018 micro-Ohm), demonstrating equal excitation and
geometrical placement without averaging the two ports into agreement.
