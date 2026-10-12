Automatic tracking gallery
==========================

Run from the repository root in an environment with this checkout installed:

.. code-block:: console

    python examples/features/eigenmode_ports/example_10_tracked_mode_gallery/tracked_modes.py

The default run inspects 16 tracked channels in each of six guides: metallic
and dielectric rectangular, square, and circular cross-sections. It uses
``tracking='auto'`` without manual groups or polarizations. Symmetry selection
is automatic. Internal candidate modes complete degenerate groups even when
their partners lie beyond the 16 displayed channels.

All cases use five frequency anchors from 40 to 42 GHz and a 0.5 mm cubic
mesh. Rectangles are 20 by 14 mm; squares are 20 by 20 mm; circles have a
10 mm radius. Metallic guides have PEC walls and air-filled bores. Dielectric
guides are lossless rods with relative permittivity 4 in air, inside a 40 mm
square computational window. Their modal aperture has an artificial PEC rim;
increase the window and refine the mesh to check convergence to an isolated
rod. Circular interfaces are voxelized. These plots describe the supplied
discrete guides.

The plotted ``neff`` is the Yee propagation index, including numerical
dispersion; ``operator_neff`` is also saved in the HDF5 bank. At this mesh,
some dielectric propagation indices exceed the continuum core index of 2.
Full verification can flag mesh sensitivity, especially at voxelized circular
walls. Refine the mesh to assess index accuracy independently of tracking.
The finite dielectric window can also leave exterior-aperture verification
unresolved; a valid tracked bank does not establish aperture convergence.

Only geometry and modal solves run. There is no excitation, waveform, virtual
guide, time stepping, or S-parameter calculation. The one-iteration time
window satisfies scene construction and is never advanced.

Outputs are written to ``results/`` beside the scripts:

* ``CASE.modes.h5``: tracked complex fields on native Yee coordinates,
  effective indices, validity masks, tracking and symmetry diagnostics.
* ``CASE_E_modes.png`` and ``CASE_H_modes.png``: all 16 profiles at 41 GHz.
  Colour shows each mode's magnitude normalized to its own peak; arrows show
  transverse fields at a common E/H phase; red outlines mark the guide.
* ``CASE_dispersion.png``: real and imaginary effective indices at every
  anchor. Degenerate curves can overlap.
* ``manifest.json``: settings, script hashes, completion status, residuals,
  symmetry decisions, and output filenames for every completed case.

Mode numbers are tracked numerical channels, not analytic TE/TM labels.
Use ``--profiles`` to also inspect each channel's fields at every anchor.
Use ``--verification full`` for additional verification solves, ``--modes``
for a larger bank, and ``--spacing-mm`` for mesh refinement. Choose a fresh
``--output`` directory for each run; existing manifests are protected.

.. code-block:: console

    python examples/features/eigenmode_ports/example_10_tracked_mode_gallery/tracked_modes.py --case metallic_square --verification full --profiles --output results/square_full
    python examples/features/eigenmode_ports/example_10_tracked_mode_gallery/plot_results.py results/square_full/metallic_square.modes.h5
