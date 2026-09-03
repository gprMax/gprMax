=======================
Eigenmode port examples
=======================

These six numbered Python examples form the tutorial in
``docs/source/eigenmode_port.rst``:

``example_1_straight_waveguide``
    Start here. Inspect the two physical guided modes, calculate multimode S11
    and S21, and learn how artificial PEC-boundary modes can appear.

``example_2_curved_waveguide``
    Repeat the workflow for a tight bend and observe reflection and conversion
    from the launched mode into other monitored modes.

``example_3_antenna_and_farfield``
    Feed a pyramidal horn through a rectangular-waveguide eigenmode port and
    virtual waveguide. Calculate S11 and antenna patterns using a closed NTFF
    surface.

``example_4_complete_s_matrix``
    Drive the dominant quasi-TEM mode at each end of a gapped microstrip in one
    study, assemble its complete 2 by 2 S matrix without rebuilding, and
    compare the magnitude and phase of ``S21`` and ``S12`` for reciprocity.

``example_5_phased_array``
    Drive four waveguide antenna elements with a progressive phase and plot
    both driven-port active S-parameters and a dense xy-plane far-field cut
    that demonstrates beam squint.

``example_6_near_cutoff``
    Resolve TE10 immediately above and below cutoff with dense, branch-aware
    anchor points and distinguish coefficient validity from power-wave
    validity.

Run every command below from the repository root. Each model exposes a
``build_scene()`` function and uses ``gprMax.run`` directly. Output defaults
to the script directory so the no-argument plotter can find it. Pass
``--geometry-only`` to inspect modal fields and the pulse, ``--gpu N`` for a
CUDA device, or ``--output PATH`` to change the output stem. Example 4 defines
its study cases in Python and accepts ``--restart N``.

Eigenmode outputs use ``reference_basis_valid``,
``power_basis_valid``, ``coefficient_valid``, ``power_wave_valid``, and the
corresponding S/active-S/study masks to distinguish reference eligibility,
conditioned modal coefficients, and physical power waves.

Example 1
=========

.. code-block:: console

    python examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide.py --geometry-only
    python examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide.py
    python examples/features/eigenmode_ports/example_1_straight_waveguide/plot_results.py

Example 2
=========

.. code-block:: console

    python examples/features/eigenmode_ports/example_2_curved_waveguide/curved_waveguide.py --geometry-only
    python examples/features/eigenmode_ports/example_2_curved_waveguide/curved_waveguide.py
    python examples/features/eigenmode_ports/example_2_curved_waveguide/plot_results.py

Example 3
=========

The 3D horn is more expensive than Examples 1 and 2.

.. code-block:: console

    python examples/features/eigenmode_ports/example_3_antenna_and_farfield/horn_antenna.py --geometry-only
    python examples/features/eigenmode_ports/example_3_antenna_and_farfield/horn_antenna.py
    python examples/features/eigenmode_ports/example_3_antenna_and_farfield/plot_results.py

Example 4
=========

.. code-block:: console

    python examples/features/eigenmode_ports/example_4_complete_s_matrix/complete_s_matrix.py
    python examples/features/eigenmode_ports/example_4_complete_s_matrix/plot_results.py

Example 5
=========

.. code-block:: console

    python examples/features/eigenmode_ports/example_5_phased_array/phased_array.py --geometry-only
    python examples/features/eigenmode_ports/example_5_phased_array/phased_array.py
    python examples/features/eigenmode_ports/example_5_phased_array/plot_results.py

Example 6
=========

.. code-block:: console

    python examples/features/eigenmode_ports/example_6_near_cutoff/near_cutoff.py --geometry-only
    python examples/features/eigenmode_ports/example_6_near_cutoff/near_cutoff.py
    python examples/features/eigenmode_ports/example_6_near_cutoff/plot_results.py

Generated CSV, HDF5, VTK-HDF, modal-field, snapshot, and result-plot files are
ignored by Git and can be recreated by rerunning the examples. The larger
validation matrix remains under ``testing/regression/eigenmode_sources``.
