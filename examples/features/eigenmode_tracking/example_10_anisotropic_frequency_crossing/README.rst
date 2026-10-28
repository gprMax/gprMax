Example 10: Two polarized branches crossing in frequency
========================================================

A 16 × 8 mm PEC bore filled with diag(2.25, 1, 1) has two orthogonally
polarized fundamental branches. Their raw eigenvalue ranks exchange over
12.8–16.1 GHz. The geometry-only run exports both branches and their E/H
profiles, with no excitation or time stepping.

Run from the repository root::

   python examples/features/eigenmode_tracking/example_10_anisotropic_frequency_crossing/tracked_modes.py

Use ``--spacing-mm`` for the native geometry mesh and ``--output`` for a
fresh directory. Recorded plots are separate from fresh outputs.

.. image:: gallery/frequency_crossing_dispersion.png
   :width: 100%
   :alt: frequency crossing dispersion.png

.. image:: gallery/frequency_crossing_E_modes.png
   :width: 100%
   :alt: frequency crossing E modes.png

.. image:: gallery/frequency_crossing_H_modes.png
   :width: 100%
   :alt: frequency crossing H modes.png

Recorded plots were regenerated on 24 October 2026. See ``gallery/manifest.json`` for source hashes, run settings, numerical validity and confinement diagnostics.
