# Triangular bow tie: gprMax FDTD/KSIR versus MATLAB MoM

This case compares a planar PEC bow-tie antenna in gprMax with MATLAB Antenna
Toolbox's `bowtieTriangular` Method-of-Moments model. The two triangles have a
101 mm total outer length, 100 mm outer width, and 89.4299 degree flare angle.
The gprMax triangles lie in the x-y plane and are separated by one x-directed,
1 mm voltage-source edge. Both models use a 50 Ohm port.

This replaces the electrical feed geometry in the older bow-tie script in this
directory without altering that historical file. A fine VTK edge audit found
that a triangle's cell-centre rasterisation begins the right wing one cell
beyond its acute apex. The new model therefore adds the single PEC edge from
the source's right node to the first rasterised wing edge. Without that
connector the right wing is electrically disconnected even though the drawing
looks almost continuous. The generated material inventory provides an
additional check: it contains 10,335 PEC edges, one source edge, and no source
edge overwritten by PEC.

The source-edge current is reconstructed from the same four magnetic samples
used by the native gprMax `Ix` output:

```text
Ix = dy [Hy(i,j,k-1) - Hy(i,j,k)]
   + dz [Hz(i,j,k) - Hz(i,j-1,k)].
```

This makes the active input impedance available on CUDA, OpenCL, and Metal,
where native current receiver outputs are not yet exposed. The adjacent
electric voltages are centred on the magnetic half time step. S11 is then
calculated independently from `Zin = V/I` and from separation of the total and
incident Thevenin-source voltages.

The 80 ns engineering-convention transform has 12.5 MHz independent frequency
resolution and is zero padded by eight. MATLAB is evaluated only at the
independent samples. A closed KSIR surface encloses the complete PEC bow tie and
records the 0.82 GHz far field in the x-z elevation and x-y antenna-plane cuts.

## Run

From the repository root:

```bash
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_bowtie_fs/bowtie_antenna_gprmax.py --gpu 0
matlab -batch "run('testing/other_codes/matlab_mom/antenna_bowtie_fs/bowtie_antenna_matlab.m')"
MPLCONFIGDIR=/tmp/matplotlib-gprmax conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_bowtie_fs/plot_bowtie_comparison.py
```

Omit `--gpu 0` for the CPU solver. Use `--geometry-only` to generate only the
fine ParaView geometry, or `--postprocess-only` to reconstruct all gprMax CSV
products from the persisted HDF5 output.

The principal products are:

- `results/bowtie_antenna_geometry.vtkhdf`
- `results/bowtie_pattern_comparison.png`
- `results/bowtie_s11_comparison.png`
- `results/bowtie_impedance_comparison.png`
- `results/bowtie_comparison_metrics.json`

## Current result

The CUDA model has four million cells and 41,542 iterations. On an NVIDIA
TITAN RTX, its solve took 57.9 seconds and used approximately 463 MB host plus
427 MB device memory.

At 0.82 GHz, the normalised KSIR/MATLAB difference is 0.014 dB RMS and
0.059 dB maximum in the x-z elevation plane. In the x-y antenna plane it is
0.335 dB RMS and 1.954 dB maximum away from the deep nulls.

The interpolated S11 minima are 0.7996 GHz and -12.89 dB for the gprMax
magnetic-contour impedance, 0.7999 GHz and -12.91 dB for gprMax source-wave
separation, and 0.8248 GHz and -13.91 dB for MATLAB. The 25 MHz resonance
offset is about two independent frequency bins and is consistent with the
one-cell FDTD feed gap and edge-rasterised metal differing from MATLAB's MoM
feed and continuous triangular surfaces. The gprMax and MATLAB resistance RMS
difference over 0.45--1.20 GHz is 1.44 Ohm; the reactance difference is
8.90 Ohm. The two independent gprMax complex S11 estimates differ by only
0.0024 RMS.
