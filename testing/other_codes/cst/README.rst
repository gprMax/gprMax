CST comparisons
===============

This directory contains comparisons between gprMax and CST Studio Suite for a
microstrip-fed rectangular patch antenna reconstructed from the supplied ACIS
``patch.sab`` model.

Current coverage
----------------

The ``S-parameter`` directory contains the current same-geometry S11
comparison between gprMax, the CST time-domain finite-integration-technique
(FIT) solver, and the CST frequency-domain finite-element-method (FEM) solver.
The lossless substrate has relative permittivity 4.4 and the conductors are
PEC. The retained comparison spans 1.6--3.2 GHz.

Run from the repository root with::

    conda run -n gprMax python -m gprMax testing/other_codes/cst/S-parameter/patch_antenna.in -outputfile testing/other_codes/cst/S-parameter/patch_antenna
    conda run -n gprMax python testing/other_codes/cst/S-parameter/plot_patch_sparameters.py

The second command regenerates ``patch_s11.png`` and
``patch_sparameter_summary.json`` from the gprMax CSV and CST Touchstone
exports.

Other comparisons, including far-field radiation patterns, are still under
test and are intentionally excluded from version control until their modelling
and numerical accuracy have been established.
