# Dielectric-sphere RCS validation

A z-polarised discrete plane wave propagating along +x illuminates a 16 mm-radius, lossless dielectric sphere with relative permittivity 4. A closed equivalent-current NTFF surface returns monostatic backscatter at theta=90 deg, phi=180 deg. The comparison uses the exact homogeneous-sphere Mie series evaluated independently of gprMax.

Overall validation status: **PASS**.

- Grid spacing: 0.5 mm (32 cells per radius)
- Frequency range: 0.75--9 GHz
- RMS RCS error: 0.2706 dB
- Maximum absolute RCS error: 0.7223 dB

The error includes the voxelised representation of the curved material interface as well as FDTD and transformation errors. Narrow dielectric resonances are especially sensitive to small geometry and phase shifts.

## Outputs

- [Backscatter comparison](dielectric_sphere_backscatter_rcs.png)
- `dielectric_sphere_backscatter_rcs.csv`
- `summary.json`
