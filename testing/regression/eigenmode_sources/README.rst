==================================
Eigenmode-source regression matrix
==================================

This directory contains regression and diagnostic models for the FDFD
eigenmode solver and its TF/SF injection into the FDTD grid. These cases are
deliberately broader and more repetitive than the curated user examples in
``examples/features/eigenmode_sources``.

Directory layout
================

``cases/two_dimensional``
    TMz and TEz PEC/PMC waveguides, a dielectric slab, and dielectric bends.
    These cases exercise the one-dimensional cross-section solvers, invariant
    coordinates, Yee staggering, and both electric and magnetic boundary
    constraints.

``cases/directionality``
    Six propagation directions for dielectric ridges, microstrip, rectangular
    PEC and PMC waveguides, and a cylindrical PEC waveguide. A matching
    ``+``/``-`` pair should have the same effective index and mirrored modal
    and propagated fields.

``cases/broadband``
    Matched single-frequency and seven-anchor-frequency cases in 2D TM, 2D TE,
    and 3D. These compare a control mode with complex modal-field interpolation
    over a broadband excitation.

The final ``y`` in every regression ``#eigenmode_source`` command intentionally
retains modal-field plots during normal simulations. User models normally
omit this option: a plot is then produced automatically in geometry-only mode
and suppressed during an ordinary FDTD run.

What to inspect
===============

Before interpreting receiver or snapshot data, check the modal plot for:

* the requested polarisation and mode order;
* symmetry and propagation direction;
* confinement and evanescent decay outside dielectric guides;
* tangential-field behaviour at PEC and PMC walls; and
* consistency of effective index between matching directions.

The FDTD snapshots should then show a clean wave travelling in the requested
direction. Strong decay in the microstrip cases is expected because their FR4
model is lossy.

Running the suite
=================

From the repository root, with the gprMax environment active, run:

.. code-block:: console

   python testing/regression/eigenmode_sources/run_all.py

The runner discovers every ``.in`` file beneath ``cases``, executes it, and
uses ``plot_snapshots.py`` to create combined linear and global-normalised dB
``|E|`` figures. It also invokes the broadband power-comparison helper.

Useful options are:

.. code-block:: text

   --dry-run       Print commands without running them.
   --skip-runs     Regenerate plots from existing output only.
   --skip-plots    Run the gprMax models without post-processing.
   --root PATH     Run only the cases below PATH.
   --gprmax-arg X  Pass X to every gprMax run; repeat as needed.

For example, to list only the 2D cases that would run:

.. code-block:: console

   python testing/regression/eigenmode_sources/run_all.py \
       --root testing/regression/eigenmode_sources/cases/two_dimensional \
       --dry-run --skip-plots

Generated ``.h5``, ``.vtkhdf``, snapshot directories, and ``.png`` plots are
ignored locally and must not be committed. The version-controlled suite
contains inputs and post-processing code only.
