=========================
Eigenmode source examples
=========================

The three maintained user examples form a short tutorial and are reproduced
and explained in ``docs/source/eigenmode.rst``:

``dielectric_slab_2d_tm.in``
    A straight dielectric guide with source port 1 and receiver port 2. It
    produces almost 0 dB fundamental-mode S21, very low S11, and negligible
    transmission into the second monitored mode.

``curved_dielectric_waveguide_2d_tm.in``
    The same modal workflow applied to a tight cylindrical-sector bend. Its
    lower fundamental-mode S21, larger reflection, and non-zero mode-2 S21
    demonstrate bend loss and higher-order-mode conversion.

``dielectric_rod_antenna_3d.in``
    A tapered dielectric-rod antenna fed through the omitted face of a
    five-face Huygens surface. It produces S11, directivity, gain, realized
    gain, and efficiency without counting the feed termination as radiation.

Inspect a modal field before time stepping:

.. code-block:: console

    python -m gprMax examples/features/eigenmode_sources/dielectric_slab_2d_tm.in --geometry-only

Run the 21-point straight-guide sweep and plot all valid two-mode S11/S21
traces plus the six requested field times:

.. code-block:: console

    python -m gprMax examples/features/eigenmode_sources/dielectric_slab_2d_tm.in -outputfile examples/features/eigenmode_sources/dielectric_slab_2d_tm
    python examples/features/eigenmode_sources/plot_dielectric_slab_2d_tm.py

Run the curved-guide sweep and plot its multimode S-parameters plus eight
field times ending at 2.5 ns:

.. code-block:: console

    python -m gprMax examples/features/eigenmode_sources/curved_dielectric_waveguide_2d_tm.in -outputfile examples/features/eigenmode_sources/curved_dielectric_waveguide_2d_tm
    python examples/features/eigenmode_sources/plot_curved_dielectric_waveguide_2d_tm.py

Run the nine-point antenna sweep and plot S11 and its far fields:

.. code-block:: console

    python -m gprMax examples/features/eigenmode_sources/dielectric_rod_antenna_3d.in -outputfile examples/features/eigenmode_sources/dielectric_rod_antenna_3d
    python examples/features/eigenmode_sources/plot_dielectric_rod_antenna_3d.py

Each plot script writes beside its input. The straight and curved scripts
write both an S-parameter plot and a common-colour-scale ``Ez`` snapshot
sequence, which makes the pulse propagation and bend conversion visible.
``plot_dielectric_rod_antenna_3d.py`` writes S11, peak far-field levels across
the nine-frequency sweep, and the 7 GHz principal-plane directivity, gain, and
realized gain. Generated CSV, HDF5, VTK-HDF, modal-profile, snapshot, and plot
files are ignored by Git; rerun the examples to recreate them locally.

The larger straight/bent, lossy, broadband, rectangular, and cylindrical
validation matrix lives under ``testing/regression/eigenmode_sources`` rather
than in this introductory examples directory.
