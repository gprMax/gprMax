=======================
Eigenmode port examples
=======================

These nine numbered Python examples form the tutorial in
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

``example_9_auto_mode_crossing``
    Track the orthogonally polarized modes of one anisotropic guide through a
    true propagation-constant crossing where raw eigensolver order changes,
    with S11 and S21 measured between two ports.

Automatic mode tracking is experimental and requires further testing. Use it
at your discretion; legacy tracking remains the default. Examples 8 and 9 use
``verification="fast"`` for quick visual
inspection. Both set ``plot_fields=True``, so running either model writes a
modal-profile figure with the tracked dispersion curves above the fields.
Review these plots before using the tracked profiles in a time-domain model.

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

The free-space bore uses ``averaging="y"`` to preserve the tangential electric
PEC samples shared with wall voxels. ``averaging="n"`` represents a different
Yee boundary and is also supported: the modal solver reads the final electric
PEC samples used by FDTD. Explicit frequency anchors resolve the modal impedance
between 20 and 24 GHz; degeneracy detection and polarization remain automatic.
The output and virtual-feed terminations use 16-cell PMLs.

.. code-block:: console

    python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/auto_degenerate_te11.py --geometry-only
    python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/auto_degenerate_te11.py --mode 1
    python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/plot_results.py --mode 1
    python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/auto_degenerate_te11.py --mode 2
    python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/plot_results.py --mode 2

Example 9
=========

The single PEC guide is filled with a diagonal dielectric tensor. Its
orthogonally polarized fundamental branches have different cutoff terms and
slopes, so their phase-index curves cross. The field rows show whether each
public label retains its polarization as raw eigenvalue order changes. Matching
automatic ports at each end expose both modal reflection and transmission.

The anisotropic fill is built before the four PEC wall boxes so it cannot
overwrite their tangential electric boundary samples. Frequency anchors are
closer together near the lower band's cutoff, where modal impedance changes
rapidly. Both terminations use 16-cell PMLs; port separation remains 52 mm.

.. code-block:: console

    python examples/features/eigenmode_ports/example_9_auto_mode_crossing/auto_mode_crossing.py --geometry-only
    python examples/features/eigenmode_ports/example_9_auto_mode_crossing/auto_mode_crossing.py --mode 1
    python examples/features/eigenmode_ports/example_9_auto_mode_crossing/plot_results.py --mode 1
    python examples/features/eigenmode_ports/example_9_auto_mode_crossing/auto_mode_crossing.py --mode 2
    python examples/features/eigenmode_ports/example_9_auto_mode_crossing/plot_results.py --mode 2

Reflection accuracy
===================

Double-precision CPU runs check all 41 frequencies from 20--24 GHz in Example 8
and all 55 frequencies from 13.1--15.8 GHz in Example 9. The worst co-polarized
S11 over each band's sampled frequencies is:

.. list-table:: Straight-guide reflection after preserving the PEC walls
   :header-rows: 1

   * - Example
     - Launched mode 1
     - Launched mode 2
   * - 8: circular TE11
     - -62.9 dB
     - -62.9 dB
   * - 9: anisotropic crossing
     - -81.1 dB
     - -62.4 dB

Co-polarized S21 stays within 0.00011 dB of 0 dB in these runs. The regression
in ``tests/test_eigenmode_tracking_examples.py`` checks S11 below -60 dB,
S21 within 0.005 dB, retained degeneracy/crossing identities, and matching PEC
samples in the modal and time-domain geometry.

The earlier roughly -30 dB reflection floor came primarily from a mismatch
between overwritten wall samples and the voxel-derived modal PEC mask; manual
tracking reproduced it. The solver now samples the final electric Yee masks
and reconstructs the adjacent magnetic fields consistently. Example 7 with a
non-averaged bore reaches -61.1 dB worst-case S11 for both polarizations with
either tracking method. Once the modal and FDTD boundaries agree,
interpolation between sparse modal anchors and finite PML reflection become
visible. Confident tracking does not by itself guarantee an accurate
interpolated impedance. Check convergence in anchor spacing, PML thickness,
and recording duration when smaller reflections matter. These checks concern
the represented voxel guide, not convergence to an ideal smooth circular wall.

Generated CSV, HDF5, VTK-HDF, modal-field, snapshot, and result-plot files are
ignored by Git and can be recreated by rerunning the examples. The larger
validation matrix remains under ``testing/regression/eigenmode_sources``.
