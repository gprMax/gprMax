Unit Tests — Geometry Primitives
================================

**Branch:** ``feat/unit-tests-geometry-primitives``

**Modules under test:**
   - ``gprMax/cython/geometry_primitives.pyx`` (~1,420 lines) — the
     low-level Cython rasterisation layer
   - ``gprMax/user_objects/cmds_geometry/cmds_geometry.py`` — shared
     helpers (averaging flag, rotation maths)
   - the ``build()`` dispatch of ``edge.py``, ``plate.py``,
     ``triangle.py``, ``cylinder.py``, ``cone.py``,
     ``cylindrical_sector.py``, ``geometry_objects_read.py`` — the
     glue between the user objects (constructors covered by the
     user-objects suite) and the Cython layer

**Covered transitively:**
   - ``gprMax/cython/yee_cell_setget_rigid.pyx`` — ``cdef`` (C-only)
     helpers that flip individual slots in the ``rigidE`` / ``rigidH``
     arrays; unreachable from Python, exercised through every setter
     and builder below
   - ``gprMax/user_inputs.py`` ``MainGridUserInput`` — the dispatch
     tests drive the real discretisation/bounds-checking object, not a
     mock

**Test files:**
   - ``tests/unit/geometry_primitives/test_predicates.py`` (46 tests)
   - ``tests/unit/geometry_primitives/test_voxel_setters.py`` (14 tests)
   - ``tests/unit/geometry_primitives/test_shape_builders.py`` (34 tests)
   - ``tests/unit/geometry_primitives/test_array_builders.py`` (8 tests)
   - ``tests/unit/geometry_primitives/test_cmds_geometry_helpers.py`` (27 tests)
   - ``tests/unit/geometry_primitives/test_build_dispatch.py`` (33 tests)

**Shared fixtures:** ``tests/unit/geometry_primitives/conftest.py``

Scope
-----

Verifies the Cython functions that turn shape descriptions into
per-cell writes on the four grid arrays the FDTD solver reads:

.. list-table::
   :header-rows: 1

   * - Array
     - Shape
     - Dtype
     - Holds
   * - ``solid``
     - ``(nx, ny, nz)``
     - ``uint32``
     - numeric ID of the smoothed (averaged) material filling each voxel
   * - ``rigidE``
     - ``(12, nx, ny, nz)``
     - ``int8``
     - E-field edges excluded from dielectric averaging
   * - ``rigidH``
     - ``(6, nx, ny, nz)``
     - ``int8``
     - H-field edges excluded from averaging
   * - ``ID``
     - ``(6, nx+1, ny+1, nz+1)``
     - ``uint32``
     - material ID per field component (Ex, Ey, Ez, Hx, Hy, Hz); the
       ``+1`` padding lets the last voxel stamp its far corners

Every builder follows the same recipe: compute an integer bounding box
in cell coordinates, clamp it to the domain, and for each cell whose
*centre* satisfies the shape's inside-check call ``build_voxel`` (or a
face setter for zero-thickness shapes). With ``averaging=True`` a cell
is written to ``solid`` and its rigid flags are cleared; with
``averaging=False`` the rigid flags are set and all 24 ``ID`` entries
of the cell are stamped.

The tests pin the exact set of array slots each function writes on a
small grid (typically 8 × 8 × 8 cells at 1 mm discretisation) — cell
selection at shape boundaries, domain clamping, the
averaging/hard split, and the Yee-cell neighbour bookkeeping.

On top of the Cython layer, the suite covers the dispatch layer that
feeds it: the ``build()`` methods of the seven shape wrappers whose
rasterisation path was not exercised by the user-objects suite, and
the shared helpers in ``cmds_geometry.py``. Those tests run the whole
chain — continuous coordinates → real ``MainGridUserInput``
discretisation → material/averaging resolution → Cython call — and
assert the final state of the grid arrays.

Tests do not validate physics (whether a rasterised sphere is a
"good" sphere at FDTD resolution is integration territory) and do not
touch the fractal machinery that generates the arrays consumed by
``build_voxels_from_array*`` (a later PR).
``GeometryObjectsRead.build()`` beyond its parameter guard is also out
of scope — it needs real HDF5/materials files and the scene builder;
its array stamping is covered directly via ``build_voxels_from_array``.

Test Infrastructure
-------------------

``tests/unit/geometry_primitives/conftest.py`` is deliberately small —
the Cython module reads no global config and no grid object, so a
namespace of freshly zeroed numpy arrays is a complete test
environment:

``grid_arrays``
   Factory fixture returning a ``SimpleNamespace`` with the four
   arrays at production shapes and dtypes (default 8 × 8 × 8 cells),
   plus ``nx`` / ``ny`` / ``nz``.

``nonzero_set(arr)``
   Helper returning the set of index tuples at which an array is
   nonzero — every "which slots changed" assertion compares one of
   these against a hand-derived expected set.

``DL``
   Module constant ``0.001`` (1 mm) used as the uniform spatial
   discretisation, so cell index ``i`` maps to coordinate ``i * DL``
   and cell centres to ``(i + 0.5) * DL``.

``dispatch_grid``
   Factory fixture for the ``build()`` dispatch tests. Extends the
   ``grid_arrays`` namespace with what the real ``MainGridUserInput``
   and the ``build()`` methods read: ``dx`` / ``dy`` / ``dz``, ``dl``
   (numpy array), ``size``, a ``within_bounds`` implementation with
   the ``FDTDGrid`` contract (raises ``ValueError`` carrying the axis
   letter), ``averagevolumeobjects = True``, and a materials list
   (``pec`` / ``free_space`` builtins plus averagable ``metal``,
   ``mat_a``, ``mat_b`` stubs built by the ``make_material`` helper).
   Because the stub is not a subgrid or MPI grid, ``_create_uip``
   dispatches to the real ``MainGridUserInput`` without any config
   patching.

The Cython functions take single-precision (C ``float``) arguments,
so all shape parameters are chosen to be exactly representable in
float32 and to keep every cell centre well clear of shape boundaries
— verdicts stay stable across the float arithmetic.

Test Catalog — ``test_predicates.py``
-------------------------------------

The pure geometric predicates that underpin the sector rasteriser and
the fractal-surface machinery. No side effects — every test is a
straight input → boolean check.

TestAreClockwise
^^^^^^^^^^^^^^^^

``test_sign_convention`` (7 parameter sets)
   Pins the 2D cross-product convention of
   ``are_clockwise(v1x, v1y, v2x, v2y)``: True iff v2 lies *strictly*
   clockwise of v1. Covers both orientations, collinear and
   anti-parallel vectors (cross product exactly zero → False, the
   comparison is strict) and zero vectors. Source:
   ``geometry_primitives.pyx:42-57``.

TestIsWithinRadius
^^^^^^^^^^^^^^^^^^

``test_membership`` (6 parameter sets)
   Pins the inclusive circle test: centre always inside, a 3-4-5
   point exactly on the boundary is inside (``<=``), just-outside
   points excluded, and ``radius == 0`` keeps only the centre.
   Source: ``geometry_primitives.pyx:60-75``.

TestIsInsideSector
^^^^^^^^^^^^^^^^^^

Sectors are defined anti-clockwise from the start arm; openings up to
π use the AND of the two arm tests, reflex openings use the OR
branch. Test points sit on 45-degree diagonals so verdicts are robust
to float32 trig error in the arm coordinates. Source:
``geometry_primitives.pyx:78-126``.

``test_quarter_sector_at_origin`` (5 parameter sets)
   π/2 opening from the +x axis: quadrant-I diagonal inside, the
   three other quadrants outside, in-direction point beyond the
   radius outside.

``test_half_plane_sector`` (5 parameter sets)
   ``angle == π`` — the largest sector still on the AND branch —
   accepts the upper half-plane and rejects the lower.

``test_reflex_sector_uses_or_branch`` (5 parameter sets)
   270-degree sector: quadrants I, II, III inside, the excluded
   quadrant IV outside, radius still enforced. Exercises the OR
   composition at ``geometry_primitives.pyx:120-126``.

``test_nonzero_start_angle`` (2 parameter sets)
   Start arm on the +y axis: the sector rotates with the start
   angle (quadrant II accepted, quadrant I rejected).

``test_offset_centre``
   The point is tested relative to ``(ctrx, ctry)``, not the origin.

``test_radius_limits_diagonal_point``
   A point 0.707 from the centre is rejected by a 0.5 radius.

TestPointInPolygon
^^^^^^^^^^^^^^^^^^

Ray-casting with explicit vertex and horizontal-boundary
short-circuits. Source: ``geometry_primitives.pyx:129-179``.

``test_square_interior_and_exterior`` (4 parameter sets)
   Unit square: centre inside; right / below / far-field outside.

``test_vertex_counts_as_inside``
   Points exactly on polygon vertices return True via the vertex
   short-circuit at ``geometry_primitives.pyx:149``.

``test_point_on_horizontal_edge_counts_as_inside``
   Points strictly between the endpoints of a horizontal edge return
   True via the boundary check at ``geometry_primitives.pyx:153-163``.

``test_triangle`` (4 parameter sets)
   Slanted edges on both sides plus a point above the apex.

``test_concave_polygon`` (4 parameter sets)
   L-shaped hexagon: both arms inside, the notch and the far exterior
   outside. Concavity exercises the multi-crossing path of the
   ray-caster.

Test Catalog — ``test_voxel_setters.py``
----------------------------------------

The atomic writes every shape rasteriser bottoms out in. Each test
calls a setter for a single cell of a 4³ grid and pins the exact set
of array slots that change.

TestBuildEdges
^^^^^^^^^^^^^^

``test_edge_x_interior_cell`` / ``test_edge_y_interior_cell`` /
``test_edge_z_interior_cell``
   An edge at ``(i, j, k)`` is shared with up to three neighbouring
   cells; each test pins the exact four ``rigidE`` slots flipped (the
   home slot plus the mirrored slots in the -y/-z, -x/-z or -x/-y
   neighbours) and the single ``ID`` entry stamped with the
   directional material ID. ``rigidH`` and ``solid`` must stay
   untouched. Source: ``geometry_primitives.pyx:182-242`` over
   ``yee_cell_setget_rigid.pyx:94-136``.

``test_edge_at_origin_skips_neighbour_writes`` (parametrised over the
three edge setters)
   At ``(0, 0, 0)`` the neighbour cells do not exist; only the base
   rigid slot of the origin cell is flipped. Verifies the ``if j != 0``
   -style guards in the ``set_rigid_E*`` helpers.

TestBuildFaces
^^^^^^^^^^^^^^

``test_face_yz_interior_cell`` / ``test_face_xz_interior_cell`` /
``test_face_xy_interior_cell``
   A face setter rigidifies the four E-edges bounding the face — two
   in the home cell, two in the +1 neighbours along the face plane —
   and stamps the four matching ``ID`` entries. Each test enumerates
   the full 16-slot ``rigidE`` fan-out and the 4 ``ID`` writes with
   their per-direction material IDs. Source:
   ``geometry_primitives.pyx:245-326``.

``test_face_yz_at_origin_skips_neighbour_writes``
   The origin face writes only the six rigid slots whose neighbour
   cells exist.

TestBuildVoxelAveraging
^^^^^^^^^^^^^^^^^^^^^^^

``test_writes_solid_and_clears_rigid_column``
   With ``rigidE`` / ``rigidH`` pre-filled, an averaged voxel write
   sets ``solid`` and clears exactly the 12 + 6 rigid slots of that
   one cell, leaving ``ID`` untouched — the smoothed-material path at
   ``geometry_primitives.pyx:353-356``.

TestBuildVoxelHard
^^^^^^^^^^^^^^^^^^

``test_stamps_solid_rigid_and_all_24_id_entries``
   The hard-boundary path (``geometry_primitives.pyx:358-391``):
   ``solid`` written, all 18 rigid flags of the cell set, and all 24
   ``ID`` entries stamped — six field components at the four Yee-cell
   corners each component touches, with the correct directional
   material ID per component. The expected 24-slot map is derived in
   the test and compared exactly, so any reordering of the index
   arithmetic fails immediately.

``test_far_corner_writes_into_id_padding``
   A hard voxel at the last cell of the domain stamps corners at
   ``nx`` / ``ny`` / ``nz`` — exactly what the ``+1`` padding on every
   ``ID`` dimension exists for.

``test_overwrite_flips_averaged_cell_to_hard``
   Geometry builds in arrival order and later objects overwrite
   earlier ones: an averaged write followed by a hard write leaves
   the cell rigid with the new material.

Test Catalog — ``test_shape_builders.py``
-----------------------------------------

TestBuildBox
^^^^^^^^^^^^

``test_averaging_writes_solid_and_clears_rigid``
   A 3 × 2 × 1 averaged box writes exactly its half-open cell range to
   ``solid`` and clears the rigid flags of exactly those cells;
   ``ID`` stays untouched. Source: ``geometry_primitives.pyx:711-717``.

``test_hard_box_stamps_interior_and_trailing_faces``
   The hard path (``geometry_primitives.pyx:718-760``): interior
   cells receive all six ``ID`` components, then six trailing loops
   close the far (+x / +y / +z) faces of the box — the ``ID`` writes
   at ``xf`` / ``yf`` / ``zf`` that the half-open interior loop does
   not reach. The test derives the complete expected ``ID`` map
   (interior plus every trailing write) and compares it exactly.

``test_empty_range_is_a_noop``
   ``xs == xf`` with averaging leaves all four arrays untouched.

``test_full_domain_box``
   A box spanning the whole domain fills every ``solid`` cell.

TestBuildSphere
^^^^^^^^^^^^^^^

Source: ``geometry_primitives.pyx:1177-1240``.

``test_radius_of_one_cell_marks_the_eight_corner_cells``
   A sphere of radius one cell centred on a grid vertex contains
   exactly the eight cell centres touching that vertex (each
   ``sqrt(0.75)`` ≈ 0.87 cells away).

``test_cells_match_the_inside_check``
   A 2.5-cell sphere writes exactly the cells whose centres satisfy
   the documented inside-check, cross-checked against an independent
   distance computation over the whole grid.

``test_sub_half_cell_radius_writes_nothing``
   A 0.4-cell sphere contains no cell centre at all — the inside
   check samples centres, not corners.

``test_sphere_clamped_at_domain_corner``
   Centred at the domain origin the bounding box goes negative and is
   clamped; only the in-domain octant is written, without error.

``test_hard_sphere_sets_rigid_at_written_cells``
   ``averaging=False`` propagates to ``build_voxel``: every written
   cell is fully rigid and ``ID`` is stamped.

TestBuildEllipsoid
^^^^^^^^^^^^^^^^^^

Source: ``geometry_primitives.pyx:1243-1310``.

``test_cells_match_the_ellipsoid_equation``
   Semi-axes of 3 / 2 / 1 cells: written cells match an independent
   evaluation of the ellipsoid equation at every cell centre.

``test_equal_semiaxes_reduce_to_a_sphere``
   With ``xr == yr == zr`` the ellipsoid writes the same ``solid``
   array as ``build_sphere`` with that radius.

TestBuildCylinder
^^^^^^^^^^^^^^^^^

Source: ``geometry_primitives.pyx:763-957``. The builder detects
axis-aligned cylinders (face centres rounding to the same cell in two
of three axes) and uses a direct radial test per cross-section cell;
arbitrary axes fall through to a vector-projection branch.

``test_axis_aligned_cylinder`` (parametrised over x / y / z)
   A 1.5-cell-radius cylinder running four cells along the axis
   writes exactly the four cross-section cells around the axis vertex
   for each of the four axis cells (half-open along the axis).

``test_degenerate_point_cylinder_writes_nothing``
   Coincident face centres produce an empty axis range — no writes.

``test_arbitrary_axis_cylinder``
   A cylinder diagonal in the xy-plane exercises the projection
   branch: on-axis cells near both face centres and at mid-length are
   written; cells well off the axis are not.

``test_hard_cylinder_sets_rigid_at_written_cells``
   The averaging flag propagates through the aligned branch.

TestBuildCone
^^^^^^^^^^^^^

Source: ``geometry_primitives.pyx:960-1174``. Same alignment
structure as the cylinder plus linear radius interpolation between
the two faces.

``test_z_aligned_cone_shrinks_layer_by_layer``
   Radius interpolating from 2.5 to 0.5 cells over four layers gives
   per-layer radii 2.5 / 2.0 / 1.5 / 1.0 and layer populations
   16 / 12 / 4 / 4; the widest (4 × 4 block) and narrowest (2 × 2
   core) layers are pinned exactly.

``test_equal_radii_reduce_to_a_cylinder``
   With ``r1 == r2`` the z-aligned cone writes the same ``solid``
   array as ``build_cylinder``.

``test_equal_radii_arbitrary_axis_matches_cylinder``
   The same equivalence through the vector-projection branch, using a
   diagonal axis.

``test_x_aligned_cone_shrinks_along_x``
   The layer progression holds along x as well — the interpolation
   follows the aligned axis.

TestBuildCylindricalSector
^^^^^^^^^^^^^^^^^^^^^^^^^^

Source: ``geometry_primitives.pyx:537-679``. One branch per
``normal`` value; ``thickness == 0`` writes faces instead of voxels.

``test_full_circle_is_a_disk``
   ``sectorangle = 2π`` fills the complete 16-cell disk for each of
   the two thickness layers.

``test_quarter_sector_normal_z`` / ``_normal_x`` / ``_normal_y``
   A π/2 sector keeps exactly the quadrant-I quarter of the disk; the
   three tests pin the plane/level mapping of each ``normal`` branch
   (sector plane in (x, y) / (y, z) / (x, z), extrusion along the
   remaining axis).

``test_zero_thickness_writes_a_face_not_voxels``
   No ``solid`` writes; instead the xy-face edges of every sector
   cell at the level plane are stamped — the exact ``ID[0]`` /
   ``ID[1]`` footprints (including the +1-neighbour edges each face
   shares) are compared as sets.

``test_disk_clamped_at_domain_corner``
   A disk centred one cell from the domain corner has its bounding
   box clamped on both plane axes; the surviving cells match an
   independent radial check.

TestBuildTriangle
^^^^^^^^^^^^^^^^^

Source: ``geometry_primitives.pyx:394-534``. A 3-4-5 right triangle
(vertices at cells (2, 2), (6, 2), (2, 5)) is used throughout — its
hypotenuse never passes through a cell centre, so the strict
inside-check yields a stable six-cell staircase.

``test_normal_z_one_cell_thick``
   The six staircase cells are written at exactly the level layer.

``test_thickness_extrudes_a_prism``
   Two-cell thickness extrudes the same footprint over two layers.

``test_normal_x_maps_vertices_to_yz_plane`` /
``test_normal_y_maps_vertices_to_xz_plane``
   The same triangle expressed in the (y, z) / (x, z) planes writes
   the same staircase in the corresponding orientation — pins the
   per-``normal`` coordinate mapping.

``test_zero_thickness_writes_a_face_not_voxels``
   Face path: no ``solid`` writes, exact ``ID[0]`` / ``ID[1]``
   face-edge footprints at the level plane.

``test_degenerate_collinear_triangle_writes_nothing``
   Three collinear vertices have zero area; no cell passes the strict
   inside-check and nothing is written.

``test_hard_triangle_sets_rigid_at_written_cells``
   The averaging flag propagates to the extruded voxels.

Test Catalog — ``test_array_builders.py``
-----------------------------------------

TestBuildVoxelsFromArray
^^^^^^^^^^^^^^^^^^^^^^^^

Source: ``geometry_primitives.pyx:1313-1369``. Stamps a pre-computed
3D block of material indices into the grid (used by
``GeometryObjectsRead`` and ``FractalBox``).

``test_data_block_lands_at_the_offset``
   ``data[i, j, k]`` maps to ``solid[xs+i, ys+j, zs+k]`` offset by
   ``numexistmaterials`` — the full 2 × 2 × 2 mapping is compared
   cell by cell.

``test_negative_values_are_skipped_as_no_material``
   Negative entries are the "no material here" sentinel: the cell is
   not written and keeps its rigid flags, while neighbouring
   non-negative entries are built normally.

``test_block_overhanging_the_far_boundary_is_truncated``
   A block placed so it overhangs the +x boundary keeps its values in
   the cells that fit — truncated at the boundary, with each surviving
   cell holding its original data value.

``test_hard_write_stamps_id_with_the_offset_material``
   ``averaging=False`` stamps all six ``ID`` components with the
   offset material ID.

TestBuildVoxelsFromArrayMask
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Source: ``geometry_primitives.pyx:1372-1420``. Same stamping through
a per-cell mask (used by ``AddGrass``).

``test_mask_selects_data_water_grass_or_skip``
   The four mask values dispatch correctly: ``1`` → the data value
   (no material offset), ``2`` → ``waternumID``, ``3`` →
   ``grassnumID``, anything else → cell untouched.

``test_masked_out_cells_keep_their_rigid_flags``
   The averaging path clears rigid flags only at cells the mask
   selects.

``test_hard_write_stamps_id_with_the_mask_material``
   ``averaging=False`` propagates to ``build_voxel`` with the
   mask-selected material.

``test_all_zero_mask_writes_nothing``
   A fully masked-out block leaves all four arrays untouched.

Test Catalog — ``test_cmds_geometry_helpers.py``
------------------------------------------------

The shared helpers in ``cmds_geometry/cmds_geometry.py`` — pure
functions used by the rotatable geometry classes and the hash parser
path.

TestCheckAveraging
^^^^^^^^^^^^^^^^^^

``test_yes_maps_to_true`` / ``test_no_maps_to_false`` (parametrised
over both cases)
   ``check_averaging`` folds case: ``"y"``/``"Y"`` → ``True``,
   ``"n"``/``"N"`` → ``False``. Source: ``cmds_geometry.py:30-48``.

TestRotatePoint
^^^^^^^^^^^^^^^

``test_90_degrees_about_z`` / ``_about_x`` / ``_about_y``
   Unit-vector rotations pin the axis convention of the
   ``scipy.spatial.transform`` Euler rotation. Source:
   ``cmds_geometry.py:51-78``.

``test_rotation_about_an_offset_origin``
   The point is translated to the rotation origin, rotated, and
   translated back.

``test_360_degrees_is_identity``
   Full turn returns the original point.

TestRotate2PointObject
^^^^^^^^^^^^^^^^^^^^^^

Source: ``cmds_geometry.py:81-136`` — the rotation applied by every
two-point rotatable geometry object (``Edge``, ``Plate``, ``Box``, …).

``test_90_about_z_with_explicit_origin``
   An x-aligned segment rotated about its first point becomes
   y-aligned, re-sorted to (lower-left, upper-right) form.

``test_default_origin_is_the_object_centre``
   With no origin given, rotation happens about the object's centre:
   a 2 × 4 box rotated 90 degrees swaps its extents around a fixed
   centre.

``test_coordinates_along_the_rotation_axis_are_preserved``
   Rotation about x cannot change the object's x-extents — verifies
   the invariant-direction reset at ``cmds_geometry.py:124-134``.

``test_result_is_sorted_lower_left_to_upper_right``
   A 180-degree rotation still yields ``new_pts[0] <= new_pts[1]``
   component-wise.

``test_non_multiple_of_90_raises`` (45, 91, 30) /
``test_angle_outside_0_360_raises`` (-90, 450) /
``test_invalid_axis_raises``
   The three validation guards at ``cmds_geometry.py:98-110``.

TestRotatePolarisation
^^^^^^^^^^^^^^^^^^^^^^

Source: ``cmds_geometry.py:139-177``.

``test_90_degree_polarisation_remap`` (all six axis/polarisation pairs)
   Rotating 90 degrees about an axis perpendicular to the current
   polarisation remaps it to the remaining axis (x + z-rotation → y,
   etc.).

``test_returns_point_pair_one_cell_along_the_polarisation``
   The returned point pair extends the input point by one grid cell
   in the current polarisation direction — the segment the rotation
   then acts on.

``test_uppercase_polarisation_accepted``
   The polarisation letter is case-folded.

Test Catalog — ``test_build_dispatch.py``
-----------------------------------------

The ``build()`` methods of the seven geometry wrappers whose
rasterisation path was not covered by the user-objects suite. Each
test drives the full chain — continuous coordinates → real
``MainGridUserInput`` discretisation → material/averaging resolution
→ Cython rasteriser — against the ``dispatch_grid`` stub and asserts
the final state of the grid arrays.

TestEdgeBuild
^^^^^^^^^^^^^

Source: ``edge.py:58-119``.

``test_axis_oriented_edge_stamps_id_cells`` (parametrised x / y / z)
   An edge from cell 2 to cell 5 along one axis stamps exactly the
   three half-open ``ID`` entries of the matching component with the
   material's ``numID``, sets ``rigidE`` and leaves ``solid``
   untouched.

``test_diagonal_edge_raises``
   Two points differing in more than one axis fail the orientation
   check at ``edge.py:93-99``.

``test_unknown_material_raises`` / ``test_missing_kwargs_raises``
   The material-lookup guard and the three-parameter guard.

``test_do_rotate_turns_an_x_edge_into_a_y_edge``
   ``rotate("z", 90)`` about the edge's first point followed by
   ``_do_rotate`` rewrites the ``p1``/``p2`` kwargs through
   ``rotate_2point_object``: the x-oriented edge becomes y-oriented.

TestPlateBuild
^^^^^^^^^^^^^^

Source: ``plate.py:59-160``.

``test_xy_plate_stamps_face_edges``
   An xy-plane plate over cells (1..3, 1..2) at z-level 2 runs
   ``build_face_xy`` per cell; the exact ``ID[0]`` / ``ID[1]``
   face-edge footprints (including the +1-neighbour edges) are
   compared as sets, with ``solid`` untouched.

``test_anisotropic_plate_uses_per_direction_materials``
   With ``material_ids=["mat_a", "mat_b"]`` the first material feeds
   the x-edges and the second the y-edges — the uniaxial branch at
   ``plate.py:148-150``.

``test_volume_raises`` / ``test_line_raises``
   The orientation check at ``plate.py:96-102`` rejects
   three-dimensional extents and degenerate lines.

``test_unknown_material_raises``
   Material lookup failure raises ``ValueError``.

TestTriangleBuild
^^^^^^^^^^^^^^^^^

Source: ``triangle.py:67-241``. Uses the same 3-4-5 staircase
triangle as the direct ``build_triangle`` tests, so the two suites
pin the same cells from opposite ends of the chain.

``test_prism_writes_solid_with_grid_default_averaging``
   With no ``averaging`` kwarg, ``grid.averagevolumeobjects`` and the
   material's ``averagable`` flag combine to the smoothed path:
   ``solid`` written, rigid untouched, no ``ID``.

``test_averaging_off_kwarg_sets_rigid``
   ``averaging="n"`` switches to the hard path via
   ``check_averaging``.

``test_zero_thickness_builds_a_patch``
   Zero thickness dispatches to the face path — no ``solid`` writes,
   ``ID[0]`` / ``ID[1]`` stamped.

``test_non_coplanar_vertices_raise``
   Vertices that do not share a plane fail the orientation check at
   ``triangle.py:109-124``.

``test_unknown_material_raises`` / ``test_missing_thickness_raises``
   Material lookup and required-kwargs guards.

TestCylinderBuild
^^^^^^^^^^^^^^^^^

Source: ``cylinder.py:55-159``.

``test_z_aligned_cylinder_matches_direct_rasterisation``
   The same z-aligned cylinder as the direct ``build_cylinder`` test
   produces the same 16-voxel set through the full dispatch chain,
   with the material's ``numID`` in ``solid``.

``test_averaging_off_kwarg_sets_rigid``
   ``averaging="n"`` propagates to the Cython call.

``test_non_positive_radius_raises`` /
``test_unknown_material_raises`` / ``test_missing_radius_raises``
   The radius guard at ``cylinder.py:91-93``, material lookup, and
   required-kwargs guards.

TestConeBuild
^^^^^^^^^^^^^

Source: ``cone.py:57-160``.

``test_tapering_cone_matches_direct_rasterisation``
   The tapering cone reproduces the 16 / 12 / 4 / 4 layer counts of
   the direct ``build_cone`` test through the dispatch chain.

``test_both_radii_zero_raises`` / ``test_negative_radius_raises`` /
``test_missing_radius_raises``
   The three radius guards at ``cone.py:94-108`` and the
   required-kwargs guard.

TestCylindricalSectorBuild
^^^^^^^^^^^^^^^^^^^^^^^^^^

Source: ``cylindrical_sector.py:64-233``.

``test_quarter_sector_matches_direct_rasterisation``
   ``start=0, end=90`` (degrees, converted to radians by the wrapper)
   reproduces the quarter-disk voxel set of the direct
   ``build_cylindrical_sector`` test.

``test_full_circle_end_angle_raises`` / ``test_zero_end_angle_raises``
   The wrapper caps sector angles strictly inside (0, 360) degrees —
   the guards at ``cylindrical_sector.py:132-141``.

``test_invalid_normal_raises``
   A normal outside x/y/z fails validation (via the dimension check
   in ``check_thickness``).

``test_missing_kwargs_raises``
   Required-kwargs guard.

TestGeometryObjectsReadBuild
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_missing_kwargs_raises``
   The five-parameter guard at ``geometry_objects_read.py:61-67``.
   The rest of ``build()`` (HDF5 + materials files, scene building)
   is integration territory; its array stamping is covered directly
   in ``test_array_builders.py``.

Running
-------

From the repository root, with the project installed in editable mode
(``pip install -e .``) so the compiled Cython extensions are
importable::

    python -m pytest tests/unit/ -v

Filter to just this PR's suite::

    python -m pytest tests/unit/geometry_primitives/ -v

Run a single file::

    python -m pytest tests/unit/geometry_primitives/test_shape_builders.py -v

Run a single test::

    python -m pytest tests/unit/geometry_primitives/test_shape_builders.py::TestBuildBox::test_hard_box_stamps_interior_and_trailing_faces -v

Stop on first failure (useful while iterating)::

    python -m pytest tests/unit/ -x
