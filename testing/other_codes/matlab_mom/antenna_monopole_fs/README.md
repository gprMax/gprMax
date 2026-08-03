# Finite-ground monopole: gprMax FDTD/KSIR versus MATLAB MoM

This case compares a centre-fed quarter-wave monopole over a finite square PEC
ground plane in gprMax with a MATLAB Antenna Toolbox
`monopoleCylindrical` Method-of-Moments model. Two gprMax feeds are retained as
separate controls: the original one-cell voltage gap and the Hyun equivalent
coaxial magnetic frill attached to an improved thin wire. MATLAB uses its
native delta-gap feed, so the frill comparison tests the complete antenna
response rather than asserting identical local feed models.

Both models use a 79 mm ground-to-tip height and a 160 mm square PEC plate. The
voltage-fed gprMax antenna consists of a one-cell, 1 mm source gap followed by
a 78 mm PEC Yee edge. The frill-fed antenna is a 79 mm thin wire with its
physical radius set to 0.23 mm. MATLAB uses the same 0.23 mm wire radius,
matching the established 0.23-cell effective radius of an axial PEC wire on
the 1 mm FDTD grid. Both gprMax ports use a 36.5 Ohm reference impedance.

This is deliberately a stronger KSIR test than the free-space dipole. The
closed integration surface encloses a large zero-thickness PEC plate, its
induced electric surface currents, and its diffracting edges. The comparison
includes the complete x-z elevation plane, including radiation behind the
finite plate, and the x-y ground-plane cut.

The 36.5 Ohm port is analysed in two independent ways: separation of incident
and reflected source voltages, and `Zin = V / I` using the discrete Ampere
current around the driven z edge. Because native `Iz` receiver output is
currently CPU-only, the model saves the four required magnetic-field samples
and reconstructs the identical expression on CUDA:

```text
Iz = dx [Hx(i,j-1,k) - Hx(i,j,k)]
   + dy [Hy(i,j,k) - Hy(i-1,j,k)].
```

Adjacent electric voltages are averaged onto the magnetic-field half-time
step. The engineering-convention FFT uses a 100 ns record, providing 10 MHz
independent resolution, with eight-times zero padding. MATLAB is evaluated at
the independent 10 MHz samples rather than at every interpolated zero-padding
sample, avoiding redundant finite-plate MoM solutions.

## Run

From the repository root:

```bash
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_monopole_fs/monopole_antenna_gprmax.py --gpu 0
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_monopole_fs/monopole_antenna_gprmax.py --feed frill --gpu 0
matlab -batch "run('testing/other_codes/matlab_mom/antenna_monopole_fs/monopole_antenna_matlab.m')"
MPLCONFIGDIR=/tmp/matplotlib-gprmax conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_monopole_fs/plot_monopole_comparison.py
```

Omit `--gpu 0` to use the Cython CPU solver. Use `--geometry-only` to write the
fine Yee-edge ParaView model without solving, or `--postprocess-only` to
regenerate the selected feed's CSV files from an existing HDF5 output. The
frill port spectra are read from the automatic `/frills/frill1` HDF5 output;
the plotting script interpolates these complex spectra onto the MATLAB sample
frequencies when the stable FDTD timesteps give slightly different FFT bins.

The generated products include:

- `results/monopole_antenna_geometry.vtkhdf`
- `results/monopole_antenna_frill_geometry.vtkhdf`
- `results/monopole_pattern_comparison.png`
- `results/monopole_s11_comparison.png`
- `results/monopole_impedance_comparison.png`
- `results/monopole_comparison_metrics.json`

## Current result

The CUDA model contains 15.625 million cells and 51,927 iterations. On an
NVIDIA TITAN RTX, the solve took 3 minutes 30 seconds and used approximately
1.2 GB host plus 1.1 GB device memory. The fine ParaView geometry is 169 MB.
Its material counts contain exactly 51,520 PEC plate edges and 78 PEC monopole
edges, plus the single driven source edge.

KSIR agrees extremely well with MATLAB despite the large enclosed PEC plate.
The complete x-z elevation cuts differ by 0.040 dB RMS and 0.077 dB maximum
away from the nulls. The x-y cuts differ by 0.0004 dB RMS.

The frill-fed KSIR pattern differs from MATLAB by 0.0397 dB RMS in the x-z
plane and 0.00043 dB RMS in the x-y plane. More importantly for isolating the
feed, the frill and voltage-gap gprMax patterns differ by only 0.00076 dB RMS
in x-z and 0.00011 dB RMS in x-y.

The interpolated 36.5 Ohm S11 minima are 0.9063 GHz from the gprMax
magnetic-contour impedance, 0.9064 GHz from gprMax source-wave separation, and
0.9117 GHz from MATLAB. The 5.4 MHz gprMax/MATLAB offset is smaller than the
10 MHz independent resolution. Their -10 dB bandwidths are 44.93, 45.00, and
47.61 MHz, respectively.

The magnetic-frill S11 minimum is 0.9057 GHz and its interpolated -10 dB
bandwidth is 46.11 MHz. Over 0.85--0.95 GHz its input-impedance differences
from MATLAB are 0.54 Ohm RMS in resistance and 2.85 Ohm RMS in reactance. The
small remaining differences are consistent with comparing the explicit
equivalent coax aperture against MATLAB's ideal delta gap.

At the 0.90124 GHz output sample, gprMax gives
`Zin = 22.46 + j0.01 Ohm`, compared with `22.01 - j2.48 Ohm` from MATLAB. Over
0.85--0.95 GHz, the RMS differences are 0.48 Ohm in resistance and 2.43 Ohm in
reactance. The two independent gprMax complex S11 calculations differ by only
0.0029 RMS over the full output band.
