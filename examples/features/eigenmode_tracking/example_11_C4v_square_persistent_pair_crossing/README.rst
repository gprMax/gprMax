Example 11: Persistent doublets crossing in a square guide
==========================================================

A square PEC bore contains diag(2, 2, epsilon_z). Geometry-only snapshots
show eight native port modes before and after a material-parameter crossing.
An undriven FDFD sweep at 20 GHz then follows TE20/02 and TM12/21 doublets
through the exact numerical coincidence, where all four input modes are
deliberately mixed. The plots compare tracked profiles and propagation
constants with analytical solutions. This is an axial-permittivity sweep,
not a dispersive frequency sweep.

Run from the repository root::

   python examples/features/eigenmode_tracking/example_11_C4v_square_persistent_pair_crossing/tracked_modes.py

Use ``--spacing-mm`` for the native geometry mesh and ``--output`` for a
fresh directory. Recorded plots are separate from fresh outputs.

.. image:: gallery/propagation_constants.png
   :width: 100%
   :alt: propagation constants.png

.. image:: gallery/modal_profiles.png
   :width: 100%
   :alt: modal profiles.png

.. image:: gallery/before_crossing_E_modes.png
   :width: 100%
   :alt: before crossing E modes.png

.. image:: gallery/after_crossing_E_modes.png
   :width: 100%
   :alt: after crossing E modes.png

.. image:: gallery/before_crossing_H_modes.png
   :width: 100%
   :alt: Magnetic fields before the crossing.

.. image:: gallery/after_crossing_H_modes.png
   :width: 100%
   :alt: Magnetic fields after the crossing.

Recorded plots were regenerated on 24 October 2026. See ``gallery/manifest.json`` for source hashes, run settings, numerical validity and confinement diagnostics.
