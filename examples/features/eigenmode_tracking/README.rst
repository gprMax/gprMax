Eigenmode tracking examples
==========================

All eleven galleries were regenerated on 24 October 2026. The pipeline
separates longitudinal TE/TM content, Ku-Kv and Ku+Kv before symmetry
refinement, then transports any remaining unresolved blocks. User polarization
mixing requires a certified C4v E pair with significant, well-conditioned
rank-two electric moments. Gallery manifests retain source hashes and validity
checks. The finite copper windows retain artificial-edge confinement warnings;
these numerical modes still require aperture-convergence assessment.

The first nine undriven geometry-only cases cover every PEC/copper/dielectric and
rectangular/square/cylindrical combination. Each script automatically tracks
at least sixteen channels and writes native modal fields, real/imaginary
propagation curves, and full electric and magnetic field galleries.

* `Example 1: PEC rectangular <example_1_C2v_rectangular_PEC/README.rst>`_
* `Example 2: PEC square <example_2_C4v_square_PEC/README.rst>`_
* `Example 3: PEC cylindrical <example_3_C4v_cylindrical_PEC/README.rst>`_
* `Example 4: Copper rectangular <example_4_C2v_rectangular_copper/README.rst>`_
* `Example 5: Copper square <example_5_C4v_square_copper/README.rst>`_
* `Example 6: Copper cylindrical <example_6_C4v_cylindrical_copper/README.rst>`_
* `Example 7: Dielectric rectangular <example_7_C2v_rectangular_dielectric/README.rst>`_
* `Example 8: Dielectric square <example_8_C4v_square_dielectric/README.rst>`_
* `Example 9: Dielectric cylindrical <example_9_C4v_cylindrical_dielectric/README.rst>`_

Run from the repository root::

   python examples/features/eigenmode_tracking/example_1_C2v_rectangular_PEC/tracked_modes.py

The default spacing is 0.25 mm. ``--spacing-mm``, ``--modes``, and
``--verification full`` control refinement, mode count, and additional checks.
``--output`` selects a fresh result directory. The recorded galleries are
separate from fresh run outputs. Source hashes and run settings are retained
with each gallery. Hashes describe original run-file bytes before Git
normalizes text line endings. Dielectric-core apertures have distant artificial PEC rims;
these examples are not analytical validation. Cylindrical symmetry labels
refer to the sampled Cartesian problem.

Crossing examples
-----------------

* `Example 10: two polarized branches cross in frequency <example_10_anisotropic_frequency_crossing/README.rst>`_.
* `Example 11: persistent doublets cross as axial permittivity changes <example_11_C4v_square_persistent_pair_crossing/README.rst>`_.

These focused crossing cases use two and eight channels. Example 11 combines
geometry-only port snapshots with an undriven material-parameter FDFD sweep
and an independent analytical comparison.

Numbered folders include descriptive geometry/material names. Recorded
metadata identifies the executed sources, including the tracking implementation.
