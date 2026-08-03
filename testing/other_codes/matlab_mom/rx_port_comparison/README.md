# Production `RxPort` comparison with MATLAB antennas

This comparison uses the production gprMax `/ports/feed` HDF5 datasets; it
does not reconstruct current from surrounding magnetic-field receivers. Four
single-edge feeds are covered: a thin-wire dipole, triangular bow-tie,
finite-ground monopole, and substrate-backed patch. The bow-tie, monopole, and
patch preserve their MATLAB dimensions. The dipole uses a documented 2 mm grid
and a 150 mm outer span rather than the 151 mm MATLAB model. Existing MATLAB
Antenna Toolbox MoM results are interpolated onto the native gprMax FFT bins.

Run all four CPU models and create the plots with:

```bash
FI_PROVIDER=shm python testing/other_codes/matlab_mom/rx_port_comparison/compare_rx_port_matlab_antennas.py
```

Use a CUDA device by index with, for example:

```bash
FI_PROVIDER=shm python testing/other_codes/matlab_mom/rx_port_comparison/compare_rx_port_matlab_antennas.py --gpu 0
```

To regenerate CSV, JSON, and PNG files from existing HDF5 results:

```bash
FI_PROVIDER=shm python testing/other_codes/matlab_mom/rx_port_comparison/compare_rx_port_matlab_antennas.py --postprocess-only
```

The 6 ns time window gives 166.7 MHz independent frequency resolution. The
monopole uses 8 ns and the narrower-band patch uses 15 ns. The plots include
the uncorrected source-plane S11 as an audit curve, but all reported gprMax
terminal impedance is derived from the corrected `S11` output.

The independent CUDA invariance suite is kept separately under
`testing/backend_consistency/rx_port`; it is not part of the MATLAB
comparison.
