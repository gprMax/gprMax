============================
Surface-impedance validation
============================

This package contains the independent analytical comparison for the
surface-impedance boundary implementation. The driver writes numerical CSV
data, a PNG comparison, and a machine-readable ``summary.json``. Solver HDF5
files are retained only below an ignored ``_cache`` directory for optional
``--reuse`` analysis.

The validation is ``validate_copper_wall_waveguide.py``, which models
copper-preset TE10 propagation.

The copper excitation covers 120--150 GHz with 31 one-GHz DFT points. Its
source port uses all 31 frequencies plus four guards for 35 anchors, while the
passive ports use 11 guarded anchors. This preserves exact source ``neff``
validation and modal injection without repeating dense FDFD solves at both
propagation monitors. The 80--180 GHz fit explicitly uses
``fit_order='auto'`` and selects three poles. The case uses a 0.1 mm cubic
grid, a 210 mm domain, source/passive planes at 90, 105, and 145 mm, and a
500 ps record. The record ends 97.415 ps before the conservative earliest
wall return. The retained result includes bulk numerical-dispersion
compensation. Its impedance-fit, FDFD-attenuation, and FDTD-attenuation errors
are 0.026023%, 0.681438%, and 0.759867%; maximum :math:`S_{11}` is
-101.0893 dB. The four-thread rerun on 2026-09-04 took 147.019 s including
analysis and plot generation.

The figure uses four panels: actual and fitted :math:`Z_s`,
driven-port :math:`S_{11}`, FDFD effective-index attenuation against
perturbation theory, and FDTD two-plane :math:`S_{21}` attenuation against
perturbation theory. The acceptance gates are -20 dB maximum reflection and
1%/2% FDFD/FDTD attenuation error.

Run from the repository root::

    python -m testing.validation.impedance_surface.validate_copper_wall_waveguide --threads 4

By default, outputs are written under:

.. code-block:: text

    results/copper_wall_waveguide/
        copper_wall_waveguide.png
        summary.json

The directory also contains its corresponding CSV table. Pass
``--output-dir`` to select a different location, and ``--reuse`` to regenerate
CSV, PNG, and JSON results from an existing compatible local cache without
repeating the FDTD run.
