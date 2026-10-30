.. _rebar-mesh:

********************************
Two-layer reinforcement meshes
********************************

Reinforced-concrete models often contain a regular grid of steel bars near the
top surface. :class:`gprMax.RebarMesh` builds such a grid directly from the
slab extents, bar spacing, diameter, cover, and layer separation. The command
is available through the Python API and as ``#rebar_mesh`` in an input file.

Geometry
========

The reinforced volume is the axis-aligned box between ``p1`` and ``p2``. Let
the lower and upper corners of that box be

.. math::

    \mathbf{l} = (\min(x_1,x_2), \min(y_1,y_2), \min(z_1,z_2)), \qquad
    \mathbf{u} = (\max(x_1,x_2), \max(y_1,y_2), \max(z_1,z_2)).

For bar diameter :math:`d`, radius :math:`r=d/2`, clear cover :math:`c`, and
layer separation :math:`s`, the two bar-centre heights are

.. math::

    z_\mathrm{top} = u_z - c - r, \qquad
    z_\mathrm{bottom} = z_\mathrm{top} - s.

The upper layer runs along x at :math:`z_\mathrm{top}`. Its bar centres are
placed along y at

.. math::

    y_j = l_y + c + r + j\,\Delta y, \qquad j=0,1,\ldots,
    \quad y_j \leq u_y-c-r.

Each upper bar runs from
:math:`l_x+c` to :math:`u_x-c`.

The lower layer is rotated by 90 degrees. Its bar centres are placed along x
at

.. math::

    x_i = l_x + c + r + i\,\Delta x, \qquad i=0,1,\ldots,
    \quad x_i \leq u_x-c-r.

Each lower bar runs from
:math:`l_y+c` to :math:`u_y-c`.

``spacing`` may be a scalar used for both directions or an ``(x, y)`` pair.
The x component controls the positions of the lower, y-directed bars; the y
component controls the positions of the upper, x-directed bars.

Cover and layer placement
=========================

``cover`` is the clear distance from the top face and from the side faces to
the nearest bar surface. The top bar surface is therefore one cover distance
below the upper z face. Bar ends stop one cover distance inside the side
faces.

``layer_spacing`` is measured between bar centrelines. A value of zero places
both layers at the same height, so the orthogonal bars intersect. A positive
value must be at least one bar diameter, and the lower layer must remain
inside the reinforced volume. The mesh does not model the bottom cover
separately; choose ``layer_spacing`` so that the lower layer sits at the
required depth.

Spacing must also be at least one bar diameter. The x and y spans must each fit
at least one bar centre, which requires a span of twice the cover plus one
diameter. Bars are not clipped to a smaller region, merged, or trimmed around
openings.

Materials and tags
==================

Use ``material_id`` for an isotropic material or ``material_ids`` for
uniaxial x, y, and z material properties. The one or three identifiers follow
the same rules as ``#cylinder``. Dielectric smoothing is available for
isotropic materials and is disabled automatically for directional materials.

Every generated bar is a separate call to the compiled cylinder builder. The
bars therefore support the usual geometry tags, material averaging, MPI
partitioning, and later geometry overwrites. Define the host concrete volume
before the mesh so that the bars replace the host material in occupied cells.
A later geometry object can still overwrite parts of the bars.

The following example builds a steel mesh inside a concrete slab:

.. code-block:: python

    import gprMax

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.Domain(p1=(0.24, 0.18, 0.16)))
    scene.add(gprMax.PMLThickness(thickness=10))
    scene.add(gprMax.TimeWindow(time=3e-9))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e9, id="w"))
    scene.add(gprMax.Material(er=5, se=0, mr=1, sm=0, id="concrete"))
    scene.add(gprMax.Box(
        p1=(0.01, 0.01, 0.01),
        p2=(0.23, 0.17, 0.11),
        material_id="concrete",
    ))
    scene.add(gprMax.RebarMesh(
        p1=(0.02, 0.02, 0.02),
        p2=(0.22, 0.16, 0.10),
        spacing=(0.04, 0.04),
        diameter=0.008,
        cover=0.02,
        layer_spacing=0.04,
        material_id="pec",
        tag="reinforcement",
    ))
    scene.add(gprMax.HertzianDipole(
        polarisation="z",
        p1=(0.12, 0.09, 0.12),
        waveform_id="w",
    ))
    scene.add(gprMax.Rx(p1=(0.16, 0.09, 0.12), id="rx1"))

The equivalent input-file commands are:

.. code-block:: none

    #waveform: ricker 1 1.5e9 w
    #box: 0.01 0.01 0.01 0.23 0.17 0.11 concrete
    #rebar_mesh: 0.02 0.02 0.02 0.22 0.16 0.10 0.04 0.04 0.008 0.02 0.04 pec n reinforcement
    #hertzian_dipole: z 0.12 0.09 0.12 w
    #rx: 0.16 0.09 0.12 rx1 Ex Ey Ez

A complete runnable model is provided in
:download:`rebar_mesh.py <../../examples/features/rebar_mesh/rebar_mesh.py>` and
:download:`rebar_mesh.in <../../examples/features/rebar_mesh/rebar_mesh.in>`.

Inspect the generated layout without starting a solve:

.. code-block:: python

    mesh = gprMax.RebarMesh(
        p1=(0.02, 0.02, 0.02),
        p2=(0.22, 0.16, 0.10),
        spacing=(0.04, 0.04),
        diameter=0.008,
        cover=0.02,
        layer_spacing=0.04,
        material_id="pec",
    )
    print(mesh.bar_count, mesh.total_length)
    for bar in mesh.bars:
        print(bar.layer, bar.axis, bar.p1, bar.p2, bar.length)

Discretisation guidance
=======================

The cylinder builder uses a cell-centre inclusion test. A bar that is very
thin compared with the grid can disappear completely, and the geometry
validation will reject the model with a message naming the affected bar. For
electrically meaningful steel reinforcement, use several cells across the bar
diameter. A diameter of one or two cells only approximates the PEC surface at
the staircased cell locations and does not resolve the conductor radius.

The mesh spacing and layer separation are geometric distances in metres and
are independent of ``#dx_dy_dz``. Adjust both the grid and the layout when
refining a model.

Limitations
===========

* The mesh is three-dimensional and is rejected in 2D TM and TE models.
* Only two orthogonal layers are generated.
* Bars are straight circular cylinders; hooks, bends, couplers, and welded
  joints are not represented.
* There is no automatic bottom cover or clipping against irregular slabs.
* The mesh does not create a separate steel material. Use ``pec`` or define a
  dispersive or conductive steel model explicitly.
