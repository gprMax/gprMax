# Running the SIBC tests

For routine development and the standard CPU CI selection:

```sh
python -m pytest tests/impedance_surfaces tests/test_virtual_waveguide_impedance.py -m "not slow" --durations=20
```

For the complete validation, including every parameter combination:

```sh
python -m pytest tests/impedance_surfaces tests/test_virtual_waveguide_impedance.py --durations=20
```

The `slow` selection retains the full copper-waveguide validation, the
Lorentz-substrate microstrip pulse comparison, and the additional combinations
of three 2D test matrices. Routine runs retain 24 of 96 rotated virtual-field
cases, 16 of 36 boundary/modal-build cases, and 12 of 36 independent 3D
comparisons. These selections cover every axis mapping and polarization,
both propagation directions, both precisions, active/passive behavior, and
all three surface laws across the appropriate matrices. Unfiltered runs
still execute every case. No physical observation windows are shortened.

The lossy/dispersive virtual-guide comparisons in
`tests/test_virtual_waveguide_impedance.py` also keep their extra rotated
MRIPML cases under `slow`; routine cases retain all four host laws in both
precisions, plus reduced TE/TM and active-source comparisons.

The 2D, virtual-guide, perfect-conductor-contact, and coupled bulk/PML physics tests use an
opt-in fixture to suppress unused fit PNG rendering. Material fitting and
all physical assertions remain active. Dedicated fit-plot lifecycle tests
continue to render and inspect real images.

Keep the inexpensive dense-equation, negative-control, and long-step-count
stability checks in routine runs. A large iteration count alone does not
make a small-grid test expensive.
