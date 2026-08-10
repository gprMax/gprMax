# Internal PML waveguide termination experiment

`run_waveguide_length_sweep.py` measures modal S11 for a PEC rectangular
waveguide terminated by a PEC-backed internal PML slab. The PML entrance and
source plane remain fixed while the slab grows, so the result isolates slab
length rather than changing the source-to-load distance.

The sweep contains an empty-guide worst case and a conductive, one-pole Debye
fill. The latter exercises the same frequency-dependent material model in the
FDFD eigenmode solve, the source/receiver interpolation, the FDTD ADE, and the
PML region.

Run from the repository root with the gprMax conda environment:

```powershell
conda run -n gprMax python testing/experimental/internal_pml_slab/run_waveguide_length_sweep.py
```

The default lengths are 0 (a same-plane PEC control), 4, 6, 8, 10, 12, 16,
20, 30, 40, and 50 cells. The longest slab is the reference used to remove the
common finite-time/source residual from the complex S11. Use `--force` to
rerun existing cases and `--help` for material, length, and output options.

For placement in another model:

- Put the zero-stretch PML entrance in a straight, longitudinally invariant
  section of the guide.
- Make the slab transverse bounds coincide with the complete guided aperture.
- Keep the same materials, including dispersive poles and conductivity,
  extruded through the slab.
- Leave automatic PEC enclosure enabled for a PEC guide. It creates the four
  transverse walls and the maximum-stretch backing plate; coincident existing
  PEC walls are accepted.
- Do not place sources, receivers, discontinuities, or geometry transitions in
  the slab.

Compact results are under `length_sweep_results/`. The full per-case inputs,
logs, HDF5 files, and S-parameter CSVs are generated on demand and are not
stored here.
