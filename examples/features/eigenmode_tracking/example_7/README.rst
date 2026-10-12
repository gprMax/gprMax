Example 7: DIELECTRIC rectangular tracked modes
===============================================

Run from the repository root::

   python examples/features/eigenmode_tracking/example_7/tracked_modes.py

The script performs a geometry-only run, tracks at least sixteen modes, and
writes native E/H fields, electric and magnetic galleries, and real/imaginary
propagation curves. It automatically detects C2v. No excitation
or time stepping is used. The default spacing is 0.25 mm. Use ``--spacing-mm``
for mesh refinement, ``--modes`` for the requested channel count, or
``--output`` for a fresh result directory. Internal partners can complete a
persistent degenerate group.

The dielectric core has εr = 4 and is surrounded by air. Its finite port
aperture has an artificial PEC rim; inspect convergence when enlarging it.

Recorded geometry-only gallery
------------------------------

.. image:: gallery/dielectric_rectangular_dispersion.png
   :width: 100%
   :alt: Tracked modal dispersion.

.. image:: gallery/dielectric_rectangular_E_modes.png
   :width: 100%
   :alt: Electric fields for every tracked channel.

.. image:: gallery/dielectric_rectangular_H_modes.png
   :width: 100%
   :alt: Magnetic fields for every tracked channel.
