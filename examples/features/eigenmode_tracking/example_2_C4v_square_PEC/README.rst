Example 2: PEC square tracked modes
===================================

Run from the repository root::

   python examples/features/eigenmode_tracking/example_2_C4v_square_PEC/tracked_modes.py

The script performs a geometry-only run, tracks at least sixteen modes, and
writes native E/H fields, electric and magnetic galleries, and real/imaginary
propagation curves. It automatically detects C4v on the square Yee mesh. No excitation
or time stepping is used. The default spacing is 0.25 mm. Use ``--spacing-mm``
for mesh refinement, ``--modes`` for the requested channel count, or
``--output`` for a fresh result directory. Internal partners can complete a
persistent degenerate group.

The bore is vacuum-filled and bounded by PEC walls.

Recorded geometry-only gallery
------------------------------

.. image:: gallery/pec_square_dispersion.png
   :width: 100%
   :alt: Tracked modal dispersion.

.. image:: gallery/pec_square_E_modes.png
   :width: 100%
   :alt: Electric fields for every tracked channel.

.. image:: gallery/pec_square_H_modes.png
   :width: 100%
   :alt: Magnetic fields for every tracked channel.

Recorded plots were regenerated on 24 October 2026. See ``gallery/manifest.json`` for source hashes, run settings, numerical validity and confinement diagnostics.
