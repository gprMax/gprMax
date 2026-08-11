CST comparisons
===============

This directory contains comparisons between gprMax and CST Studio Suite for a
microstrip-fed rectangular patch antenna reconstructed from the supplied ACIS
``patch.sab`` model.

Current coverage
----------------

The ``S-parameter`` directory contains the same-geometry S11
comparison between gprMax, the CST time-domain finite-integration-technique
(FIT) solver, and the CST frequency-domain finite-element-method (FEM) solver.
The lossless substrate has relative permittivity 4.4 and the conductors are
PEC. The retained comparison spans 1.6--3.2 GHz.

Run from the repository root with::

    conda run -n gprMax python -m gprMax testing/other_codes/cst/patch_antenna/S-parameter/patch_antenna.in -outputfile testing/other_codes/cst/patch_antenna/S-parameter/patch_antenna
    conda run -n gprMax python testing/other_codes/cst/patch_antenna/S-parameter/plot_patch_sparameters.py

The second command regenerates ``patch_s11.png`` and
``patch_sparameter_summary.json`` from the gprMax CSV and CST Touchstone
exports.

The ``farfield`` directory compares the 2.45 GHz directivity from the gprMax
closed near-to-far-field transform with CST FIT time-domain and CST FEM
frequency-domain results using adaptive mesh refinement. It includes
Cartesian front-to-back cuts for ``phi=0`` and ``phi=90`` and full polar-plane
cuts for ``phi=0/180`` and ``phi=90/270``. The JSON summary records the
main-beam angle and directivity of every trace without adding markers or
annotations to the figures.

Run from the repository root with::

    conda run -n gprMax python -m gprMax testing/other_codes/cst/patch_antenna/farfield/patch_antenna_recentered_closed_ntff_backed_pml.in -outputfile testing/other_codes/cst/patch_antenna/farfield/patch_recentered_closed_ntff
    conda run -n gprMax python testing/other_codes/cst/patch_antenna/farfield/plot_patch_farfield_comparison.py

The large gprMax HDF5 output and raw CST far-field exports are intentionally
ignored. The input model, comparison script, plots, tabulated cuts, and summary
metrics are retained so the reviewed result remains available without adding
large solver outputs to the repository.
