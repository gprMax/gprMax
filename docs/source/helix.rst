.. _helix:

********************************************
Helical wire geometry
********************************************

``Helix`` builds a circular-cross-section wire whose centreline is a helix.
The helix starts at ``p1`` and advances along the axis from ``p1`` towards
``p2``. Only the direction of ``p2`` is used; its distance from ``p1`` does
not affect the geometry. In the Python API the object is
:class:`gprMax.user_objects.cmds_geometry.helix.Helix`; the equivalent hash
command is ``#helix``.

Geometry
========

The helix centreline is a right circular helix of radius ``coil_radius``
around the axis. ``pitch`` is the axial advance per complete turn and
``turns`` is the number of turns, so the axial length is ``pitch * turns``.
``turns`` may be fractional. With ``pitch`` zero the wire closes into
``turns`` loops in the plane through ``p1``. Handedness is selected with
``handedness='r'`` (right-handed) or ``handedness='l'`` (left-handed), case
insensitively. The wire cross-section has radius ``wire_radius`` and the axis
orientation is unrestricted.

The object follows the same ordered overwrite semantics as other volume
primitives: later objects replace earlier ones in shared cells, dielectric
smoothing is controlled with ``averaging`` (``y`` or ``n``), and the optional
``tag`` writes semantic metadata to occupied cells. A helix can be given one
isotropic material with ``material_id`` or three directional materials with
``material_ids``. Because a helix is not invariant along either transverse
axis it can only be used in 3D models.

When the helix material is PEC in all three directions, the rasterised PEC
Yee-edge network is checked after construction, including across MPI
partitions. If the requested wire radius produces more than one connected
edge component, the helix is rejected as under-resolved. The check uses the
actual ``dx``, ``dy`` and ``dz`` and never changes ``wire_radius``.

For a helix that is not PEC in all three directions, the wire diameter must
be at least the smallest grid cell size. If it is smaller than the largest
grid cell size, an explicit under-resolution warning is issued. Connectivity
does not imply geometric accuracy: a connected one-cell-thick wire can still
be a coarse approximation of the requested cross-section. If only some
material directions are PEC, PEC connectivity is not checked and a warning is
issued.

Python API
==========

.. code-block:: python

    scene.add(gprMax.Helix(
        p1=(0.06, 0.06, 0.005),
        p2=(0.06, 0.06, 0.045),
        coil_radius=8e-3,
        pitch=5e-3,
        turns=4.0,
        wire_radius=1.5e-3,
        handedness="r",
        material_id="pec",
    ))

Hash command
============

.. code-block:: none

    #helix: p1x p1y p1z p2x p2y p2z coil_radius pitch turns wire_radius handedness material_id [averaging] [tag]

The last two optional entries are the dielectric-smoothing switch and the
geometry tag, in that order. Directional materials use
``material_x material_y material_z`` in place of ``material_id``.

Parallel execution
==================

``Helix`` rasterises directly into the local grid of each MPI rank, so
distributed models produce the same voxels as a serial run. See
:ref:`numerical-comparisons` for a validated three-turn helix antenna
compared with an independent method-of-moments solution.
