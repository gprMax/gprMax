==================================
Eigenmode-source regression matrix
==================================

This directory contains regression and diagnostic models for the FDFD
eigenmode solver and its TF/SF injection into the FDTD grid. These cases are
deliberately broader and more repetitive than the curated user examples in
``examples/features/eigenmode_ports``.

Directory layout
================

``straight_waveguide``
    Broadband-driven straight-guide checks: 2D TM and TE dielectric slabs,
    plus 3D rectangular and cylindrical PEC waveguides. The cylindrical guide
    deliberately requests automatic anchors for a degenerate modal pair; a
    severe tracking mismatch should warn and fall back to one band-centre
    anchor. The common DFT still covers 45--65 GHz. The expected
    fundamental-mode S21 is approximately 0 dB and S11 is very small.

``bending_waveguide``
    Broadband 2D TM and TE 90-degree curved dielectric bends made from
    annular cylindrical sectors. Each polarisation has 15, 30, and 100 mm
    centreline-radius cases. The larger radius is expected to improve
    fundamental-mode S21.

``loss_comparison``
    Matched broadband non-lossy and lossy 2D TM slab runs. The lossy guide
    should have lower S21.

``broadband_vs_single_frequency``
    The same broadband waveform injected using either multi-frequency modal
    anchors or only the 3 GHz modal profile.

``grid_spacing``
    A 3D rectangular PEC waveguide repeated at 0.20, 0.10, and 0.05 mm cubic
    spacing. The physical geometry, 1 mm PML thickness, port positions,
    frequency range, and time window remain fixed. The comparison helper plots
    fundamental-mode S21 and max-absolute, half-peak-to-peak, and RMS
    fluctuation metrics versus spacing, with a second-order convergence guide.

``legacy``
    Previous development runs moved out of the maintained matrix. This
    directory is intentionally ignored by Git and is not discovered by the
    default runner.

The TE10 partial-cutoff model compares directly with an analytical solution,
so it belongs to the separate
``testing/validation/rectangular_waveguide_partial_cutoff`` validation rather
than this behavioural regression matrix.

Each model defines its DFT range once with ``#eigenmode_band``. Every
``#eigenmode_port`` then supplies a unique port number, plane, direction,
comma-separated monitored modes, and independent explicit or ``auto`` modal
anchors. One ``#eigenmode_excitation`` selects the active port and mode and,
for these regressions, generates the band-adapted automatic waveform. A final
``y`` on a port intentionally retains modal-field plots during normal runs.

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

The 0.75--7 GHz 2D cases deliberately use an 8 ns time window. Their
band-adapted excitation has about 4 ns of temporal support, so a 4 ns run
ends while the transmitted tail is still travelling to port 2. That receiver
truncation can produce an apparently well-matched guide with a false S21 near
-30 dB. Keep enough time after the excitation for the complete packet to
cross the receiving port and enter the PML.

Running the suite
=================

From the repository root, with the gprMax environment active, run:

.. code-block:: console

   python testing/regression/eigenmode_sources/run_all.py

The runner discovers every ``.in`` file beneath this directory except those
under ``legacy``, executes it, and
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
       --root testing/regression/eigenmode_sources/straight_waveguide \
       --dry-run --skip-plots

Generated ``.h5``, ``.csv``, ``.vtkhdf``, snapshot directories, and ``.png``
plots are ignored locally and must not be committed. The version-controlled
suite contains inputs and post-processing code only.
