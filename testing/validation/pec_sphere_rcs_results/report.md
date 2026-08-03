# PEC-sphere RCS validation

A z-polarised discrete plane wave propagating along +x illuminates a
16 mm-radius PEC sphere. A closed equivalent-current NTFF surface
returns monostatic backscatter at theta=90 deg, phi=180 deg. The
comparison uses an independently evaluated PEC Mie series.

Overall validation status: **PASS**.

- Grid spacing: 0.5 mm (32 cells per radius)
- Frequency range: 0.75--9 GHz
- Electrical-size range: 0.2515--3.018
- RMS RCS error: 0.4416 dB
- Maximum absolute RCS error: 0.9537 dB

The largest dB errors normally occur at sharp Mie nulls, where a small
frequency shift in the staircased FDTD sphere produces a large ratio.

## Outputs

- [Backscatter comparison](pec_sphere_backscatter_rcs.png)
- `pec_sphere_backscatter_rcs.csv`
- `summary.json`
