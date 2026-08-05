=======================
Eigenmode port examples
=======================

These three numbered examples form the beginner tutorial in
``docs/source/eigenmode_port.rst``:

``example_1_straight_waveguide``
    Start here. Inspect the two physical guided modes, calculate multimode S11
    and S21, and learn how artificial PEC-boundary modes can appear.

``example_2_curved_waveguide``
    Repeat the workflow for a tight bend and observe reflection and conversion
    from the launched mode into other monitored modes.

``example_3_antenna_and_farfield``
    Feed a pyramidal horn through a rectangular-waveguide eigenmode port and
    calculate S11, a 3D directivity pattern, and E-/H-plane directivity, gain,
    and realized gain.

Run every command below from the repository root.

Example 1
=========

.. code-block:: console

    python -m gprMax examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide.in --geometry-only
    python -m gprMax examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide.in -outputfile examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide
    python examples/features/eigenmode_ports/example_1_straight_waveguide/plot_results.py

Example 2
=========

.. code-block:: console

    python -m gprMax examples/features/eigenmode_ports/example_2_curved_waveguide/curved_waveguide.in --geometry-only
    python -m gprMax examples/features/eigenmode_ports/example_2_curved_waveguide/curved_waveguide.in -outputfile examples/features/eigenmode_ports/example_2_curved_waveguide/curved_waveguide
    python examples/features/eigenmode_ports/example_2_curved_waveguide/plot_results.py

Example 3
=========

The 3D horn is more expensive. Eigenmode sources currently require the CPU
solver, so do not add ``-gpu`` to this example command.

.. code-block:: console

    python -m gprMax examples/features/eigenmode_ports/example_3_antenna_and_farfield/horn_antenna.in --geometry-only
    python -m gprMax examples/features/eigenmode_ports/example_3_antenna_and_farfield/horn_antenna.in -outputfile examples/features/eigenmode_ports/example_3_antenna_and_farfield/horn_antenna
    python examples/features/eigenmode_ports/example_3_antenna_and_farfield/plot_results.py

Generated CSV, HDF5, VTK-HDF, modal-field, snapshot, and result-plot files are
ignored by Git and can be recreated by rerunning the examples. The larger
validation matrix remains under ``testing/regression/eigenmode_sources``.
