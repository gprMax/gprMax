=======================
Eigenmode port examples
=======================

These eight numbered Python examples form the tutorial in
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

``example_7_degenerate_te11``
    Assign global y/x electric polarizations to the degenerate circular TE11
    pair at every anchor. Switch the launched polarization using only the
    excitation's mode number, with a virtual-guide source and receiving port.

``example_8_auto_degenerate_te11``
    Opt into automatic tracking, omit the manual degenerate group, and inspect
    automatic TE11 pair detection plus default deterministic x/y labels at two
    ports of a straight guide. Compare co- and cross-polarized S11 and S21.

Example 8 uses ``verification="fast"`` and ``plot_fields=True`` for quick
inspection. Its geometry-only run shows dispersion and modal fields before
running a driven simulation. Separate geometry-only galleries and crossing
examples are under ``examples/features/eigenmode_tracking``.

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

Example 7
=========

Mode 1 is vertical (global y); mode 2 is horizontal (global x). Each run
has a separate output stem. The folder also contains an equivalent hash model.
Both ports enable the standard modal profile pictures with ``plot_fields=True``;
the folder README includes the generated profiles for both tracked polarizations.

.. code-block:: console

    python examples/features/eigenmode_ports/example_7_degenerate_te11/circular_te11.py --geometry-only
    python examples/features/eigenmode_ports/example_7_degenerate_te11/circular_te11.py --mode 1
    python examples/features/eigenmode_ports/example_7_degenerate_te11/plot_results.py --mode 1
    python examples/features/eigenmode_ports/example_7_degenerate_te11/circular_te11.py --mode 2
    python examples/features/eigenmode_ports/example_7_degenerate_te11/plot_results.py --mode 2

Example 8
=========

The tracker discovers the circular TE11 pair without a ``degenerate`` setting.
The omitted ``mode_polarizations`` setting defaults the pair to global x/y
directions. Supplying the mapping remains useful for swapping or rotating them.
If ``modes`` is changed to ``(1,)``, the partner is still solved internally for
subspace transport but is not added as a public monitor channel. Two ports on
the straight guide expose co- and cross-polarized S11 and S21.

The air bore is carved after the PEC volume. With ``averaging="y"``, PEC wins
at electric-field samples shared with the remaining wall. With
``averaging="n"``, air wins at the samples the bore writes, including shared
wall samples it touches. Both settings work because the mode solver reads
the same final assignments as FDTD, but they can change the effective bore
size and cutoff. Explicit frequency anchors resolve the modal impedance
between 20 and 24 GHz; degeneracy detection and polarization remain automatic.
The output and virtual-feed terminations use 16-cell PMLs.

.. code-block:: console

    python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/auto_degenerate_te11.py --geometry-only
    python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/auto_degenerate_te11.py --mode 1
    python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/plot_results.py --mode 1
    python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/auto_degenerate_te11.py --mode 2
    python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/plot_results.py --mode 2

Geometry-only mode tracking
===========================

Mode-inspection examples live separately under
``examples/features/eigenmode_tracking``. Its nine numbered cases cover
rectangular, square, and cylindrical PEC/copper guides and open dielectric
cores, with automatic tracking and native E/H galleries. These are
geometry-only runs with no excitation or time stepping.
