# FDFD eigenmode effective-index validation

Production 1D and 2D FDFD eigenmode results are compared with
independent analytical dispersion relations at multiple frequencies.
The cylindrical-guide comparison reports both numerically solved members
of the degenerate TE11 pair.

Overall validation status: **PASS**.

| Case | Frequencies | Modes | RMS error | Maximum error | Limit | Status |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| 1D PEC parallel-plate guide: TM1 | 6 | 1 | 0.0010% | 0.0021% | 0.1% | PASS |
| 1D dielectric slab: even TE0 | 6 | 1 | 0.0049% | 0.0101% | 0.1% | PASS |
| 2D PEC rectangular guide: TE10 | 6 | 1 | 0.0160% | 0.0329% | 0.1% | PASS |
| 2D PEC cylindrical guide: TE11 pair | 6 | 2 | 0.5265% | 1.0431% | 1.5% | PASS |

The 1D dielectric reference solves the even symmetric-slab
transcendental equation. PEC parallel-plate and rectangular-guide
references use their closed-form cutoff dispersion. The cylindrical
TE11 reference uses the first zero of the derivative of J1.

## Outputs

- [Effective-index comparison](neff_comparison.png)
- `neff_comparison.csv`
- `summary.json`
