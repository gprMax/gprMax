# Internal PML waveguide length sweep

Band: 30-45 GHz; cell size: 0.2 mm; longest-slab reference: 50 cells.

`Worst excess` is the maximum over the band of |S11(length) - S11(reference)|. The reference row is omitted because its self-difference is exactly zero.

## empty

| Cells | Length (mm) | Worst raw S11 (dB) | Worst excess (dB) |
|---:|---:|---:|---:|
| 0 | 0.0 | 0.00 | 0.04 |
| 4 | 0.8 | -26.95 | -28.96 |
| 6 | 1.2 | -31.62 | -42.71 |
| 8 | 1.6 | -32.52 | -51.58 |
| 10 | 2.0 | -32.59 | -54.53 |
| 12 | 2.4 | -32.54 | -57.04 |
| 16 | 3.2 | -32.41 | -61.87 |
| 20 | 4.0 | -32.32 | -65.88 |
| 30 | 6.0 | -32.36 | -85.85 |
| 40 | 8.0 | -32.35 | -111.24 |
| 50 | 10.0 | -32.35 | reference |

## lossy_debye

| Cells | Length (mm) | Worst raw S11 (dB) | Worst excess (dB) |
|---:|---:|---:|---:|
| 0 | 0.0 | -16.17 | -16.17 |
| 4 | 0.8 | -57.21 | -76.11 |
| 6 | 1.2 | -57.69 | -95.75 |
| 8 | 1.6 | -57.75 | -124.91 |
| 10 | 2.0 | -57.75 | -130.55 |
| 12 | 2.4 | -57.75 | -132.16 |
| 16 | 3.2 | -57.75 | -134.61 |
| 20 | 4.0 | -57.75 | -133.00 |
| 30 | 6.0 | -57.75 | -133.22 |
| 40 | 8.0 | -57.75 | -131.21 |
| 50 | 10.0 | -57.75 | reference |
