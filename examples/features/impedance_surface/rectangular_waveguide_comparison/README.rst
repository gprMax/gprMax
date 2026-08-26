===================================================
PEC versus copper rectangular-waveguide eigenmodes
===================================================

This example builds the same short rectangular TE10 waveguide twice. The
first guide uses ideal PEC walls. The second assigns a fitted copper
``SurfaceImpedance`` ID directly to the same four ordinary ``Box`` objects.

It demonstrates three finite-conductivity effects:

* the impedance-aware FDFD solve returns a complex effective index;
* :math:`-k_0\operatorname{Im}(n_\mathrm{eff})` gives positive conductor
  attenuation under gprMax's :math:`e^{-j\beta x}` convention;
* tangential :math:`E_z` is zero at a PEC side wall but small and non-zero at
  the copper wall.

Run
===

From the repository root, using the local ``gprMax`` conda environment:

.. code-block:: console

    cd examples/features/impedance_surface/rectangular_waveguide_comparison
    conda run -n gprMax python run_comparison.py --threads 4
    conda run -n gprMax python plot_results.py

To build only the geometry and write the FDFD modal-field figures:

.. code-block:: console

    conda run -n gprMax python run_comparison.py --geometry-only --threads 4

The full run writes ``pec_rectangular_waveguide.h5`` and
``copper_rectangular_waveguide.h5``. Both runs also write independently scaled
vector-field figures named ``*_eigenmode_fields.png``. ``plot_results.py``
writes ``rectangular_waveguide_eigenmode_comparison.png`` and prints the
140 GHz effective indices, copper attenuation, and peak wall-to-centre field
ratios.

Material syntax
===============

The Python model uses the breaking fit API. The fit range is mandatory for a
metal preset, while ``fit_order='auto'`` selects the smallest order meeting
the requested tolerance. A geometry-only run always writes the fit diagnostic;
``plot_fit=False`` suppresses that diagnostic during a full FDTD run, while
``plot_fit=True`` writes it for both run types:

.. code-block:: python

    scene.add(gprMax.SurfaceImpedance(
        id='copper_wall',
        preset='copper',
        fit_frequency_range=(80e9, 200e9),
        fit_order='auto',
        fit_tolerance=2e-3,
        plot_fit=False,
    ))
    scene.add(gprMax.Box(
        p1=lower,
        p2=upper,
        material_id='copper_wall',
        averaging='n',
    ))

The equivalent hash definitions are:

.. code-block:: text

    #surface_impedance: copper_wall preset copper 80e9 200e9 auto 2e-3 n
    #box: x0 y0 z0 x1 y1 z1 copper_wall n

Dependencies and caveats
========================

The model requires NumPy, h5py, Matplotlib, and the normal gprMax runtime; the
bundled conda environment supplies them. Surface-impedance volumes currently
run only in the 3-D CPU solver, so the script deliberately requests CPU double
precision and provides no GPU option.

A surface-impedance ID represents the boundary of a volumetric geometry. It
can be assigned anywhere an ordinary volume material ID is accepted, as the
four ``Box`` walls demonstrate, but assigning it to a zero-thickness sheet
geometry is invalid and raises an input error.

The copper preset is the thick, smooth, non-magnetic 293 K good-conductor
model over the explicitly fitted RF band. It is not an optical, thin-film,
roughness, plating, alloy, or temperature-dependent copper model.

Impedance walls cannot intersect a PML. The walls therefore end one retained
cell before each x PML, and the 100 ps record ends before an end reflection can
return to a receiver. This is a local wall/eigenmode demonstration, not a
matched waveguide-termination example.

The two modal-field PNGs choose their own colour/vector scales. Use the
normalized receiver trace in the combined comparison for the quantitative
non-zero tangential-field demonstration.
