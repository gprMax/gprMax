Horn antenna comparison
=======================

This directory compares gprMax with the CST Studio Suite time-domain
finite-integration-technique (FIT) solver and frequency-domain
finite-element-method (FEM) solver with adaptive mesh refinement for a
rectangular pyramidal horn reconstructed from ``horn.sab``.

Model and outputs
-----------------

The horn has a 24 x 10 mm inner throat, a 30 mm straight waveguide section, a
228 mm flare, and a 110 x 80 mm inner mouth. ``horn_antenna.in`` represents the
1 mm PEC wall on a 1 mm FDTD grid. A dominant-mode eigenmode port and virtual
waveguide excite the horn without extending the physical guide to the domain
boundary.

The excitation band spans 8--12 GHz. The port uses automatic modal-anchor
candidates that cover that band and the significant spectrum of the automatic
excitation waveform. Propagation, reference selection, and fallback decisions
are made per port and mode. This validation intentionally retains and presents
only the far-field comparison; its gprMax and CST S-parameter results are not
included. A closed near-to-far-field surface samples the 8, 9, 10, 11, and
12 GHz patterns in the complete x-z and y-z principal planes at one-degree
resolution.

Run from the repository root with::

    conda run -n gprMax python -m gprMax testing/other_codes/cst/horn_antenna/horn_antenna.in
    conda run -n gprMax python testing/other_codes/cst/horn_antenna/plot_horn_results.py

The plotting command creates ``horn_antenna_farfield_polar_comparison.png``.
All three solver traces use the shared CST-comparison style: gprMax is blue and
solid, CST FIT is yellow and dashed, and CST FEM is pink and dashed with the
FEM trace drawn on top.

The ``#virtual_waveguide`` command in ``horn_antenna.in`` moves the modal source
into a 21-cell auxiliary guide with 12 termination cells and 6 clear cells. The
horn model itself is the retained integration case for that virtual termination.

Repository contents
-------------------

The input model, ACIS geometry, plotting script, compact CST principal-plane
table, and final far-field comparison figure are retained. Large gprMax
HDF5/geometry outputs, generated S-parameter output, full-sphere CST far-field
exports, and regenerable eigenmode diagnostics are intentionally ignored.
Consequently, a fresh checkout can regenerate the comparison figure after
running the gprMax model; the ten roughly 10 MB CST exports are not needed.

If the original full-sphere exports are available locally, refresh or audit the
retained table with::

    conda run -n gprMax python testing/other_codes/cst/horn_antenna/plot_horn_results.py --refresh-cst-cuts
    conda run -n gprMax python testing/other_codes/cst/horn_antenna/plot_horn_results.py --audit-cst-cuts

The refresh command deterministically extracts the x-z and y-z cuts into
``horn_farfield_principal_planes_cst.csv``. The audit command requires all ten
raw files and verifies that every retained value is an exact round trip.
