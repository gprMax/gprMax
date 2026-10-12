Eigenmode tracking examples
==========================

The first nine undriven geometry-only cases cover every PEC/copper/dielectric and
rectangular/square/cylindrical combination. Each script automatically tracks
at least sixteen channels and writes native modal fields, real/imaginary
propagation curves, and full electric and magnetic field galleries.

* `Example 1: PEC rectangular <example_1/README.rst>`_
* `Example 2: PEC square <example_2/README.rst>`_
* `Example 3: PEC cylindrical <example_3/README.rst>`_
* `Example 4: Copper rectangular <example_4/README.rst>`_
* `Example 5: Copper square <example_5/README.rst>`_
* `Example 6: Copper cylindrical <example_6/README.rst>`_
* `Example 7: Dielectric rectangular <example_7/README.rst>`_
* `Example 8: Dielectric square <example_8/README.rst>`_
* `Example 9: Dielectric cylindrical <example_9/README.rst>`_

Run from the repository root::

   python examples/features/eigenmode_tracking/example_1/tracked_modes.py

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

* `Example 10: two polarized branches cross in frequency <example_10/README.rst>`_.
* `Example 11: persistent doublets cross as axial permittivity changes <example_11/README.rst>`_.

These focused crossing cases use two and eight channels. Example 11 combines
geometry-only port snapshots with an undriven material-parameter FDFD sweep
and an independent analytical comparison.
