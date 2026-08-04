==================================
Eigenmode-source regression matrix
==================================

This directory contains regression and diagnostic models for the FDFD
eigenmode solver and its TF/SF injection into the FDTD grid. These cases are
deliberately broader and more repetitive than the curated user examples in
``examples/features/eigenmode_sources``.

Directory layout
================

``cases/straight_waveguide``
    Broadband-driven straight-guide checks: 2D TM and TE dielectric slabs,
    plus 3D rectangular and cylindrical PEC waveguides. The cylindrical guide
    deliberately uses one 55 GHz modal solve because its first two modes are a
    degenerate pair; the DFT still covers 45--65 GHz. The expected
    fundamental-mode S21 is approximately 0 dB and S11 is very small.

``cases/bending_waveguide``
    Broadband 2D TM and TE 90-degree curved dielectric bends made from
    annular cylindrical sectors. Each polarisation has 15, 30, and 100 mm
    centreline-radius cases. The larger radius is expected to improve
    fundamental-mode S21.

``cases/loss_comparison``
    Matched broadband non-lossy and lossy 2D TM slab runs. The lossy guide
    should have lower S21.

``cases/broadband_vs_single_frequency``
    The same broadband waveform injected using either multi-frequency modal
    anchors or only the 3 GHz modal profile.

``legacy``
    Previous development runs moved out of the maintained matrix. This
    directory is intentionally ignored by Git and is not discovered by the
    default runner.

The source mode token is ``excitation_mode[,mode_count]`` and the following
integer is its explicit port index. A receiver supplies one mode count followed
by its explicit port index. The three values before the final ``y`` define the
direct-DFT start, stop, and point count. The final ``y`` intentionally retains
modal-field plots during normal regression runs.

What to inspect
===============

Before interpreting receiver or snapshot data, check the modal plot for:

* the requested polarisation and mode order;
* symmetry and propagation direction;
* confinement and evanescent decay outside dielectric guides;
* tangential-field behaviour at PEC and PMC walls; and
* consistency of effective index between matching directions.

The FDTD snapshots should show clean straight-guide propagation, progressively
gentler curved-bend propagation, and the expected extra attenuation in the
lossy comparison.

Running the suite
=================

From the repository root, with the gprMax environment active, run:

.. code-block:: console

   python testing/regression/eigenmode_sources/run_all.py

The runner discovers every ``.in`` file beneath ``cases``, executes it, and
checks the requested physical trends before plotting. Straight guides require
fundamental S21 within 0.75 dB of 0 dB and S11 below -20 dB; curved-bend mean
fundamental S21 must improve monotonically with radius and by at least 2 dB
from the 15 mm to the 100 mm case;
the lossy guide must transmit at least 3 dB less than the non-lossy guide; and
multi-anchor broadband injection must stay closer to 0 dB than the
single-profile result. The runner then uses
``plot_snapshots.py`` to create combined linear and global-normalised dB
``|E|`` figures. It creates a magnitude/phase plot for every S-parameter CSV
and invokes the bend-radius, lossy, and source-profile comparison helpers.

Useful options are:

.. code-block:: text

   --dry-run       Print commands without running them.
   --skip-runs     Regenerate plots from existing output only.
   --skip-plots    Run the gprMax models without post-processing.
   --root PATH     Run only the cases below PATH.
   --gprmax-arg X  Pass X to every gprMax run; repeat as needed.

For example, to list only the straight-guide cases that would run:

.. code-block:: console

   python testing/regression/eigenmode_sources/run_all.py \
       --root testing/regression/eigenmode_sources/cases/straight_waveguide \
       --dry-run --skip-plots

Generated ``.h5``, ``.csv``, ``.vtkhdf``, snapshot directories, and ``.png``
plots are ignored locally and must not be committed. The version-controlled
suite contains inputs and post-processing code only.
