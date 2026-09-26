===================================================
PEC versus copper rectangular-waveguide eigenmodes
===================================================

This example builds the same short rectangular TE10 waveguide twice. The
first guide uses ideal PEC walls. The second assigns a fitted copper
``SurfaceImpedance`` ID directly to the same four ordinary ``Box`` objects.

Both port windows extend one cell into the opaque walls beyond the air
aperture. This keeps the copper surfaces inside the port's PEC rim, so their
impedance and loss remain in the modal solve. A window ending exactly on a
wall would replace that wall's modal boundary condition with PEC, whether or
not a virtual waveguide is attached.

It demonstrates three finite-conductivity effects:

* the impedance-aware FDFD solve returns a complex effective index;
* :math:`-k_0\operatorname{Im}(n_\mathrm{eff})` gives positive conductor
  attenuation under gprMax's :math:`e^{-j\beta x}` convention;
* tangential :math:`E_z` is zero at a PEC side wall but small and non-zero at
  the copper wall.

Run
===

From the repository root with gprMax installed in the active environment:

.. code-block:: console

    python examples/features/impedance_surface/rectangular_waveguide_comparison/run_comparison.py --threads 4
    python examples/features/impedance_surface/rectangular_waveguide_comparison/plot_results.py

To build only the geometry and write the FDFD modal-field figures:

.. code-block:: console

    python examples/features/impedance_surface/rectangular_waveguide_comparison/run_comparison.py --geometry-only --threads 4

The full run writes ``pec_rectangular_waveguide.h5`` and
``copper_rectangular_waveguide.h5``. Both runs also write independently scaled
vector-field figures named ``*_eigenmode_fields.png``. ``plot_results.py``
writes ``rectangular_waveguide_eigenmode_comparison.png`` and prints the
140 GHz effective indices, copper attenuation, and peak wall-to-centre field
ratios.

Material syntax
===============

The fit range is mandatory for a
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
        plot_fit=True,
    ))
    scene.add(gprMax.Box(
        p1=lower,
        p2=upper,
        material_id='copper_wall',
        averaging='n',
    ))

The equivalent hash definitions are:

.. code-block:: text

    #surface_impedance: copper_wall preset copper 80e9 200e9 auto 2e-3 y
    #box: x0 y0 z0 x1 y1 z1 copper_wall n

Dependencies and caveats
========================

The model requires NumPy, h5py, Matplotlib, and the normal gprMax runtime; the
gprMax environment supplies them. Surface-impedance volumes support 3-D and
2-D TE/TM CPU main grids. This example uses 3-D CPU double precision and
provides no GPU option.

A surface-impedance ID represents the boundary of a volumetric geometry. It
can be assigned anywhere an ordinary volume material ID is accepted, as the
four ``Box`` walls demonstrate, but assigning it to a zero-thickness sheet
geometry is invalid and raises an input error.

The copper preset is the thick, smooth, non-magnetic 293 K good-conductor
model over the explicitly fitted RF band. It is not an optical, thin-film,
roughness, plating, alloy, or temperature-dependent copper model.

This example ends its walls one retained cell before each x PML, and the
100 ps record ends before an end reflection can return to a receiver. It
demonstrates the local wall law. Uniform impedance walls can extend through
longitudinal PML or a virtual guide under the host-material and coverage
requirements in the surface-impedance guide; those matched terminations are
demonstrated separately by ``../virtual_waveguide_2d.py``.

The two modal-field PNGs choose their own colour/vector scales. Use the
normalized receiver trace in the combined comparison for the quantitative
non-zero tangential-field demonstration.
