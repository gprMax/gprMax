Eigensource Test Runs
=====================

This folder contains small 2D and 3D regression-style runs for checking
eigenmode source solving and injection in different propagation directions.

The tests are not intended as high-accuracy production examples. They are short
runs designed to make modal purity, propagation direction, and PEC/PMC
Yee-grid staggering problems visible in field snapshots.

Test Groups
-----------

``2d_tm`` and ``2d_te``
    Six one-dimensional cross-section tests:

    - ``2d_tm/pec_waveguide`` solves the one-dimensional ``E_z`` scalar mode
      and injects the active ``E_z/H_x/H_y`` TMz field system.
    - ``2d_te/pec_waveguide`` solves the one-dimensional ``H_z`` scalar mode
      and injects the active ``E_x/E_y/H_z`` TEz field system.
    - ``2d_tm/dielectric_slab`` exercises staggered dielectric sampling. Its
      source aperture includes 25 mm of free space on both sides of the slab
      core, allowing the evanescent modal tails to decay before the aperture
      boundary and transverse PML.
    - ``2d_te/pmc_waveguide`` exercises the magnetic-field constraints at two
      PMC walls.
    - ``2d_tm/dielectric_bend`` and ``2d_te/dielectric_bend`` inject a guided
      slab mode into a straight input, a quarter-annulus 90 degree bend, and a
      straight output. The time sequence makes the guided front and radiation
      leaking from the bend visible for both polarizations.

    All use ``inf`` for the invariant source coordinates. The generated
    ``*_TM_fields.png`` or ``*_TE_fields.png`` image contains line plots of
    all three active modal fields. Each case now contains eight ``xy``
    timestamps over its longer propagation distance. The plotting script
    writes both a linear ``*_Eabs.png`` sequence and a global-normalized
    ``*_Eabs_dB.png`` sequence; the latter makes weak bend radiation easier
    to see.

``dielectric_ridge``
    Six dielectric-only tests:

    - ``ridge_x_plus`` and ``ridge_x_minus``
    - ``ridge_y_plus`` and ``ridge_y_minus``
    - ``ridge_z_plus`` and ``ridge_z_minus``

    Each case uses a dielectric ridge-like cross-section made from two
    dielectric rectangles. These runs are the baseline direction tests. They
    should inject cleanly for all ``+`` and ``-`` directions because there is no
    PEC boundary to stress the Yee-component constraint handling.

``microstrip``
    Six microstrip tests:

    - ``microstrip_x_plus`` and ``microstrip_x_minus``
    - ``microstrip_y_plus`` and ``microstrip_y_minus``
    - ``microstrip_z_plus`` and ``microstrip_z_minus``

    Each case uses a lossy FR4 substrate, a bottom PEC ground plane, and a top
    PEC strip. These tests check eigenmode solving and injection for a practical
    lossy guided structure in all six propagation directions.

``rectangular_waveguide``
    Six air-filled rectangular PEC waveguide tests:

    - ``waveguide_x_plus`` and ``waveguide_x_minus``
    - ``waveguide_y_plus`` and ``waveguide_y_minus``
    - ``waveguide_z_plus`` and ``waveguide_z_minus``

    Each case uses only PEC walls around an air-filled 6 mm by 4 mm inner
    waveguide aperture. The eigenmode source plane is limited to that inner
    aperture and does not include the PEC wall cells. The 50 GHz source is
    above the TE10 cutoff of the 6 mm broad wall.

``pmc_rectangular_waveguide``
    Six air-filled rectangular PMC waveguide tests:

    - ``pmc_waveguide_x_plus`` and ``pmc_waveguide_x_minus``
    - ``pmc_waveguide_y_plus`` and ``pmc_waveguide_y_minus``
    - ``pmc_waveguide_z_plus`` and ``pmc_waveguide_z_minus``

    These cases use the same 6 mm by 4 mm aperture as the PEC rectangular
    waveguide tests, but all four walls are PMC. The source plane is limited to
    the air-filled aperture. These runs exercise PMC-constrained FDFD modes and
    modal field injection into the FDTD Yee grid without relying on a
    dielectric-loaded interface that may contain nearby spurious modes.

``cylindrical_waveguide``
    Six air-filled cylindrical PEC waveguide tests:

    - ``cylindrical_waveguide_x_plus`` and ``cylindrical_waveguide_x_minus``
    - ``cylindrical_waveguide_y_plus`` and ``cylindrical_waveguide_y_minus``
    - ``cylindrical_waveguide_z_plus`` and ``cylindrical_waveguide_z_minus``

    Each case uses a 3.2 mm radius PEC cylinder and a concentric 3 mm radius
    air cylinder along the propagation axis. The rectangular eigenmode source
    plane extends one grid cell beyond the outer PEC cylinder's bounding
    square, so these cases check circular PEC constraints and the cylinder
    geometry primitive in the FDFD solve and source injection path. The 50 GHz
    source is above the dominant TE11 cutoff of the air bore radius.

Direction Convention
--------------------

The suffix gives the source normal and propagation direction:

``x_plus`` / ``x_minus``
    Source plane is normal to x and injects in ``+x`` or ``-x``.

``y_plus`` / ``y_minus``
    Source plane is normal to y and injects in ``+y`` or ``-y``.

``z_plus`` / ``z_minus``
    Source plane is normal to z and injects in ``+z`` or ``-z``.

The simulation domain is shortened in the source normal direction to keep the
runs quick. The two transverse dimensions are kept unchanged so the source
cross-section remains comparable between axes.

Expected Behaviour
------------------

For each ``+/-`` pair:

- The solved effective index should match to numerical precision.
- The solved modal field should be the same field mirrored across the
  transverse top/bottom direction.
- The injected wave should propagate in the requested direction.
- The transverse modal profile should stay clean as it propagates.

The ``dielectric_ridge`` group should pass these checks without involving
perfect-conductor logic. The ``pmc_rectangular_waveguide`` group additionally
checks that PMC constraints are consistent between:

- the FDFD eigenmode solver;
- the gprMax Yee magnetic-component material IDs;
- the time-domain TF/SF source update.

For the ``microstrip`` group, strong attenuation in the field-against-time
plots is expected and is not a failed source injection by itself. These cases
use FR4 with non-zero conductivity at 20 GHz, and FR4 is highly lossy at this
frequency. The decaying received fields therefore reflect the material loss in
the guided structure.

Useful Outputs
--------------

Each run writes:

``*_eigenmode_*_Eu_Ev.png`` and ``*_eigenmode_*_Hu_Hv.png``
    Field plots of a solved 3D-model FDFD mode.

``*_eigenmode_*_TM_fields.png`` and ``*_eigenmode_*_TE_fields.png``
    Yee-staggered line profiles of all three active fields in a solved 2D
    model mode. Solid lines show the real profile and dashed lines show its
    magnitude.

``*_snaps/*.h5``
    Time snapshots from planes containing the propagation axis.

``*_center_snapshots_Eabs.png``
    Combined ``|E|`` snapshot plots generated by the plotting helper.

Plotting Helper
---------------

Use ``plot_direction_snapshots.py`` to regenerate combined ``|E|`` images from
snapshot files:

.. code-block:: powershell

   python eigensource_test_run\plot_direction_snapshots.py `
       eigensource_test_run\pmc_rectangular_waveguide\pmc_waveguide_x_plus `
       eigensource_test_run\pmc_rectangular_waveguide\pmc_waveguide_x_minus

The script expects each case directory to contain a snapshot directory named
``<case_name>_snaps``.

Run All Cases
-------------

Use the run-all helper to execute every ``.in`` file below
``eigensource_test_run`` and then regenerate all combined ``|E|`` snapshot
plots with ``plot_direction_snapshots.py``.

Windows PowerShell:

.. code-block:: powershell

   .\eigensource_test_run\run_all_eigensource_tests.ps1

Windows Command Prompt:

.. code-block:: bat

   eigensource_test_run\run_all_eigensource_tests.bat

macOS/Linux:

.. code-block:: bash

   sh eigensource_test_run/run_all_eigensource_tests.sh

Run the commands from the repository root with the ``gprMax`` environment
active.

Useful options:

.. code-block:: text

   --dry-run       Print commands without running them.
   --skip-runs     Only regenerate plots from existing snapshot files.
   --skip-plots    Only run the gprMax input files.
   --gprmax-arg X  Pass an extra argument to every gprMax run. Repeat as needed.
