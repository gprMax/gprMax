# Internal PML waveguide length sweep

Band: 30-45 GHz; cell size: 0.2 mm; longest-slab reference: 50 cells.

`Worst excess` is the maximum over the band of |S11(length) - S11(reference)|. The reference row is omitted because its self-difference is exactly zero.

## empty

| Cells | Length (mm) | Worst raw S11 (dB) | Worst excess (dB) |
|---:|---:|---:|---:|
| 0 | 0.0 | 0.00 | 0.04 |
| 4 | 0.8 | -26.99 | -28.96 |
| 6 | 1.2 | -31.68 | -42.71 |
| 8 | 1.6 | -32.59 | -51.58 |
| 10 | 2.0 | -32.66 | -54.55 |
| 12 | 2.4 | -32.62 | -57.05 |
| 16 | 3.2 | -32.48 | -61.90 |
| 20 | 4.0 | -32.39 | -65.86 |
| 30 | 6.0 | -32.43 | -85.86 |
| 40 | 8.0 | -32.42 | -110.41 |
| 50 | 10.0 | -32.42 | reference |

## lossy_debye

| Cells | Length (mm) | Worst raw S11 (dB) | Worst excess (dB) |
|---:|---:|---:|---:|
| 0 | 0.0 | -16.16 | -16.17 |
| 4 | 0.8 | -57.66 | -76.16 |
| 6 | 1.2 | -58.13 | -95.73 |
| 8 | 1.6 | -58.20 | -124.95 |
| 10 | 2.0 | -58.20 | -127.70 |
| 12 | 2.4 | -58.20 | -130.17 |
| 16 | 3.2 | -58.20 | -131.78 |
| 20 | 4.0 | -58.20 | -131.93 |
| 30 | 6.0 | -58.20 | -131.46 |
| 40 | 8.0 | -58.20 | -132.16 |
| 50 | 10.0 | -58.20 | reference |
