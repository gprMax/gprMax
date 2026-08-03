=========================
Eigenmode source examples
=========================

These models demonstrate the FDFD eigenmode solver and its use as a source in
the FDTD grid. Start with ``dielectric_slab_2d_tm.in`` or its adjacent Python
API equivalent, ``dielectric_slab_2d_tm.py``.

Before running a full simulation, build the geometry and inspect the solved
mode:

.. code-block:: console

    python -m gprMax examples/features/eigenmode_sources/dielectric_slab_2d_tm.in --geometry-only

Geometry-only mode still builds the material grid and solves the eigenmode,
but skips FDTD time stepping. It automatically produces the diagnostic modal
plot. Check its polarisation, symmetry, confinement, mode order, and behaviour
at material or conducting boundaries.

The supplied models are:

``dielectric_slab_2d_tm.in`` and ``dielectric_slab_2d_tm.py``
    Equivalent hash-command and Python API versions of a fundamental TM mode
    in a straight dielectric slab.

``dielectric_bend_2d_te.in``
    A TE mode travelling through a 90-degree dielectric bend.

``rectangular_waveguide_3d.in``
    A full-vector mode in an air-filled rectangular PEC waveguide.

``microstrip_3d.in``
    A lossy full-vector microstrip mode over an FR4 substrate.

``broadband_dielectric_channel_3d.in``
    A broadband source whose complex mode is interpolated from seven FDFD
    anchor frequencies.

The larger coordinate-direction, PEC/PMC, dimensionality, and broadband
comparison matrix is maintained under
``testing/regression/eigenmode_sources``. Generated snapshots and modal plots
are outputs and should not be committed.
