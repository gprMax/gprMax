Unit Tests — Grid and Subgrids
==============================

**Branch:** ``feat/unit-tests-grid-subgrids``

**Modules under test:**
   - ``gprMax/grid/fdtd_grid.py`` — ``FDTDGrid``, the Yee lattice: size and
     discretisation properties, PML thickness bookkeeping, source and receiver
     dispatch, bounds and coordinate helpers, the CFL time step, array
     allocation, memory estimation, the 2D TM modes, PML slab construction and
     the ``build()`` assembly line
   - ``gprMax/model.py`` — ``Model``, the thin owner of one grid
   - ``gprMax/subgrids/grid.py`` — ``SubGridBaseGrid``, the subgrid's
     constructor arithmetic
   - ``gprMax/subgrids/user_objects.py`` — ``SubGridBase`` and the
     ``#subgrid_hsg`` command
   - ``gprMax/subgrids/subgrid_hsg.py`` — ``SubGridHSG``, Inner/Outer Surface
     field stitching
   - ``gprMax/subgrids/precursor_nodes.py`` — ``PrecursorNodes`` and
     ``PrecursorNodesFiltered``
   - ``gprMax/subgrids/updates.py`` — ``create_updates``, ``SubgridUpdater``,
     ``SubgridUpdates``

**Covered transitively:**
   - ``gprMax/cython/fields_updates_hsg.pyx`` ``update_is`` /
     ``update_electric_os`` / ``update_magnetic_os`` — the IS/OS tests drive
     the real OpenMP kernels against real arrays, not mocks
   - ``gprMax/pml.py`` ``PML`` and ``CFS`` — constructed for real by
     ``_construct_pml``; the coefficient internals belong to a later PR
   - ``gprMax/updates/cpu_updates.py`` ``CPUUpdates`` — the base
     ``SubgridUpdater`` extends

**Test files:**
   - ``tests/unit/grid/test_grid_properties.py`` (74 tests)
   - ``tests/unit/grid/test_grid_arrays.py`` (58 tests)
   - ``tests/unit/grid/test_grid_build.py`` (45 tests)
   - ``tests/unit/grid/test_grid_dt.py`` (33 tests)
   - ``tests/unit/grid/test_model.py`` (35 tests)
   - ``tests/unit/subgrids/test_precursor_nodes.py`` (80 tests)
   - ``tests/unit/subgrids/test_subgrid_hsg.py`` (48 tests)
   - ``tests/unit/subgrids/test_subgrid_base.py`` (47 tests)
   - ``tests/unit/subgrids/test_subgrid_commands.py`` (44 tests)
   - ``tests/unit/subgrids/test_subgrid_updates.py`` (41 tests)

**Total: 505 tests, all passing, no** ``xfail``.

**Shared fixtures:** ``tests/unit/grid/conftest.py``,
``tests/unit/subgrids/conftest.py``

Scope
-----

Every previous suite tested something that *draws on* the grid — waveforms,
materials, sources and receivers, the hash parser, user objects, geometry
primitives, fractals. None of them tested the grid itself.

``FDTDGrid`` is the Yee lattice plus the rules for using it. It knows how big
the domain is, how fine the discretisation is, how long a time step lasts, and
it owns every array the solver will ever touch. A subgrid is the same class
refined: a small region running at ``ratio`` times the resolution, stitched
into the main grid through a Huygens surface so the seam is invisible.

Grid code has a characteristic failure mode: **it does not crash, it silently
returns the wrong answer.** An off-by-one in an array shape, a ``dl[0]`` where
``dl[1]`` was meant, a time step a hair above the stability limit — none of
these raise. They produce a plausible-looking result that is wrong. That is
why this layer is worth pinning at the unit level, and why the assertions
below are mostly exact rather than approximate.

This document is written to be read **when a test fails**. Each catalog entry
states what that test expects, in concrete terms, so the failure message can
be matched against the intended contract without opening the test file. Each
catalog section ends with a *When these fail* block listing the causes that
actually produce failures in that file.

Diagnosing a Failure
--------------------

Start here before reading the catalog. Most failures in this suite fall into
one of six patterns, and four of them are environmental rather than genuine
regressions.

**The whole file errors during collection.**
   Wrong interpreter. This suite needs the project environment — the base
   conda env has no ``cython`` (so ``import gprMax`` fails at
   ``gprMax/config.py``), and ``gprMax-devel`` has no ``pytest``. Use the
   ``gprMax`` environment.

**Every test in one directory fails on a missing config attribute.**
   Both suites patch ``gprMax.config`` through an autouse fixture
   (``grid_config``, ``subgrid_config``). If new source code reads a config key
   the fixture does not supply, every test that touches that code path fails
   at once with ``AttributeError`` or ``KeyError`` on the config object. The
   fix is to add the key to the fixture — not to weaken the assertion that
   exposed it. ``config.sim_config`` is ``None`` until a real run initialises
   it, so nothing works without the patch.

**The IS/OS tests all pass, suspiciously easily.**
   This is the dangerous one, because it looks like success. The HSG Cython
   kernels multiply the incoming precursor value by an update coefficient
   looked up through the ``ID`` array. If the update-coefficient arrays are
   zeroed, or the subgrid's materials list is emptied, every kernel becomes a
   no-op and the locality assertions pass *vacuously*.
   ``TestCoupledGridsFixture::test_both_grids_have_usable_update_coefficients``
   exists to catch exactly this; if you change the fixture and that test still
   passes while the locality tests stop being meaningful, check the
   coefficients by hand.

**A set-equality assertion fails.**
   Tests that assert *which cells changed* use the ``nonzero_set`` helper and
   compare whole sets of index tuples. The failure message prints both sets, so
   the difference tells you precisely which cells moved or stopped moving —
   usually an off-by-one in an index or a changed loop bound. Do not relax these
   to "some cell changed"; the exactness is the point.

**A** ``caplog`` **test fails.**
   ``print_info`` and ``dispersion_analysis`` report through the logger rather
   than by returning values. If one of these fails, check the log *level* first
   — a message moved from ``info`` to ``debug`` will fail a test that captures
   at ``INFO`` even though the text is unchanged.

**You fixed a known bug and nothing went red.**
   Expected. Several source defects are deliberately left uncovered rather than
   pinned (see `Deliberately Untested Paths`_), because this suite does not
   assert broken behaviour and ships no ``xfail`` markers. Fixing one of those
   defects requires *adding* tests; the maintainers' bug write-ups list the
   specific tests to add in each case.

The CFL Time Step
-----------------

The most consequential line in the grid is ``calculate_dt()``. FDTD leapfrogs
electric and magnetic fields one cell and one tick at a time, so information
travels exactly one cell per tick. A time step large enough for light to cross
more than one cell does not merely lose accuracy — the solution diverges,
typically within a few hundred iterations.

The Courant-Friedrichs-Lewy limit prevents it:

.. code-block:: text

   dt = 1 / (c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))

gprMax then rounds the result **down**, to one decimal place fewer than the
hardware maximum (``decimal.getcontext().prec - 1``, via ``ROUND_FLOOR``),
because a value that lands a hair above the limit through binary
representation is as fatal as one that is genuinely too large.

Four branches: 3D, and the three 2D TM modes, each dropping its invariant axis
from the sum under the square root. Dropping a term always *relaxes* the limit,
so a 2D model legitimately takes larger steps than the equivalent 3D one.

Source: ``fdtd_grid.py:865-891``.

The Huygens Subgrid
-------------------

A subgrid resolves a small region at ``ratio`` times the main grid's
resolution. Because the CFL limit scales with cell size, it also runs ``ratio``
time steps for every main step. Two nested surfaces carry information between
the grids:

.. list-table::
   :header-rows: 1

   * - Surface
     - Position
     - Direction
   * - **IS** — Inner Surface
     - the subgrid's inner boundary
     - main grid → subgrid
   * - **OS** — Outer Surface
     - ``is_os_sep`` main cells further out
     - subgrid → main grid

``ratio`` must be **odd** so that one fine cell centre coincides with the
coarse cell centre; with an even ratio the two lattices disagree by half a cell
everywhere. Every subgrid dimension follows from it:

.. code-block:: text

   s_is_os_sep      = is_os_sep * ratio             # IS-OS gap, subgrid cells
   d_to_pml         = s_is_os_sep + pml_separation
   n_boundary_cells = d_to_pml + subgrid_pml_thickness
   nwx              = (i1 - i0) * ratio             # working region
   nx               = 2 * n_boundary_cells_x + nwx  # total, both sides
   iterations       = model.iterations * ratio

The **precursor nodes** bridge the rate difference. The main grid produces
field values once per main step; the subgrid needs them ``ratio`` times. Each
precursor holds two pages — ``<field>_0`` at the previous main tick and
``<field>_1`` at the current one — and blends them for each intermediate
sub-step:

.. code-block:: text

   c1, c2 = (ratio - m) / ratio, m / ratio
   field  = c1 * field_0 + c2 * field_1

``c1 + c2 == 1`` always. There is a spatial interpolation too, a
``RectBivariateSpline`` across each face, because the face carries ``ratio``
times more subgrid cells than main cells.

Sources: ``subgrids/grid.py:28-68``, ``subgrids/precursor_nodes.py:28-241``.

Test Infrastructure
-------------------

``tests/unit/grid/conftest.py``:

``grid_config`` (autouse)
   Patches ``gprMax.config`` to a predictable environment: double precision
   (``sim_config.dtypes["float_or_double"]``), a single OpenMP thread, 3D mode,
   zero dispersive poles, and the stock ``numdispersion`` thresholds.
   ``config.sim_config`` is ``None`` outside a real run, so this is mandatory
   for anything beyond bare construction. ``config.c`` is a module-level
   constant and needs no patch.

``make_grid``
   Factory returning a **real** ``FDTDGrid`` with ``size`` / ``dl`` set and
   geometry and field arrays allocated. Unlike the geometry and fractals
   suites, which had to stub the grid, here the grid is the class under test,
   so no stub is used. ``arrays=False`` inspects bare constructor state;
   ``pml_thickness=`` sets all six slabs.

``nonzero_set(arr)``
   Set of index tuples at which an array is nonzero — the idiom carried over
   from the geometry-primitives and fractals suites for every "which cells were
   written" assertion.

``DL``, ``DL_ANISO``
   ``0.001`` (1 mm) uniform, and ``(0.001, 0.002, 0.004)`` anisotropic. The
   anisotropic triple is deliberate: three distinct values, none a multiple of
   another, so a getter or kernel reading the wrong axis of ``dl`` cannot
   coincidentally produce the right answer.

``tests/unit/subgrids/conftest.py``:

``subgrid_config`` (autouse)
   The same patch as ``grid_config`` — ``SubGridBaseGrid`` inherits
   ``FDTDGrid``, so it reads the same three config surfaces, and ``ompthreads``
   is passed straight into the HSG Cython kernels.

``subgrid_kwargs``
   The eight keyword arguments ``SubGridBaseGrid.__init__`` requires, returned
   fresh so a test can drop or override exactly one. Defaults ``ratio=3``,
   ``is_os_sep=1``, ``pml_separation=2``, ``subgrid_pml_thickness=2``,
   ``interpolation=1``, giving ``n_boundary_cells == 7``.

``make_subgrid``
   A ``SubGridHSG`` grid with its sizes filled in. The grid class alone
   computes only the boundary-cell counts; the working region and total size
   are set by the *user object* during ``setup()``, so this factory does the
   same by hand. With ``arrays=True`` it also attaches a materials list and
   fills the update-coefficient arrays with ones — both required for the HSG
   kernels to have any observable effect.

``make_main_grid``
   A main ``FDTDGrid`` of 30³ cells, large enough to contain the subgrid's
   Inner Surface plus its ``is_os_sep`` margin, with non-zero update
   coefficients and a computed ``dt``.

``coupled_grids``
   The important one: a main grid and an HSG subgrid wired as parent and child
   with consistent indices, real field / ``ID`` / coefficient arrays on both,
   and a real ``PrecursorNodes`` (or ``PrecursorNodesFiltered``) built from the
   pair. Returns a namespace with ``main``, ``sub`` and ``precursors``.
   Everything in the IS/OS catalog depends on it, so it is asserted directly by
   ``TestCoupledGridsFixture`` before any behavioural test uses it.

``spy_updater``
   Replaces every step of a ``SubgridUpdater``, its precursors and its subgrid
   with a recorder, returning the ordered list of calls. Names are **prefixed**
   by owner (``precursors.``, ``sub.``, bare for the updater) because the three
   collaborators share method names — both the updater and the precursors have
   ``update_magnetic``, and without the prefix the counts silently conflate two
   different steps.

Test Catalog — ``test_grid_properties.py``
------------------------------------------

The parts of ``FDTDGrid`` that need no field arrays: the ``size`` / ``dl``
property views, PML thickness bookkeeping, source and receiver dispatch, bounds
checking, and the coordinate helpers every geometry command in earlier PRs
reached through a stub. Source: ``fdtd_grid.py:64-648``.

TestSizeProperties
^^^^^^^^^^^^^^^^^^

``nx`` / ``ny`` / ``nz`` are views onto the ``size`` array
(``fdtd_grid.py:139-161``).

``test_defaults_to_zero``
   A fresh ``FDTDGrid`` expects ``(nx, ny, nz) == (0, 0, 0)``.

``test_getter_reads_size`` (3 parameter sets)
   Expects ``g.nx == g.size[0]``, and the same for the other two axes.

``test_setter_writes_size`` (3 parameter sets)
   Expects ``g.nx = 17`` to leave ``size[0] == 17``.

``test_round_trip`` (3 parameter sets)
   Expects a value written through the property to read back unchanged.

``test_axes_are_independent``
   Setting ``ny = 99`` on a ``(4, 5, 6)`` grid expects ``(4, 99, 6)`` — the
   other two axes untouched.

TestDiscretisationProperties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``dx`` / ``dy`` / ``dz`` are views onto ``dl`` (``fdtd_grid.py:163-185``).

``test_defaults_to_unity``
   A fresh grid expects ``(dx, dy, dz) == (1.0, 1.0, 1.0)``.

``test_getter_reads_correct_axis`` (3 parameter sets)
   With ``dl == (0.001, 0.002, 0.004)``, expects ``dx == 0.001``,
   ``dy == 0.002``, ``dz == 0.004``. The anisotropic values make an axis
   mix-up impossible to pass by luck.

``test_setter_writes_correct_axis`` (3 parameter sets)
   Expects ``g.dy = 0.007`` to write ``dl[1]``, and correspondingly for the
   other two.

``test_round_trip`` (3 parameter sets)
   Expects a value written through the property to read back unchanged.

TestSetPmlThickness
^^^^^^^^^^^^^^^^^^^

Only the input forms that work are pinned — a scalar and a 6-element sequence.
See `Deliberately Untested Paths`_ for the others.
Source: ``fdtd_grid.py:187-199``.

``test_constructor_default_is_ten``
   Expects all six slab thicknesses to be ``10`` on a fresh grid.

``test_scalar_sets_all_six``
   ``set_pml_thickness(4)`` expects every value to be ``4``.

``test_six_element_sequence_maps_in_documented_order``
   ``(1, 2, 3, 4, 5, 6)`` expects exactly
   ``{"x0": 1, "y0": 2, "z0": 3, "xmax": 4, "ymax": 5, "zmax": 6}`` — the
   order is ``x0, y0, z0, xmax, ymax, zmax``, not ``x0, xmax, y0, …``.

``test_zero_thickness_is_allowed``
   All-zero is how the PML is turned off, and ``build()`` checks for exactly
   that state.

``test_key_order_is_stable``
   Expects the ``OrderedDict`` key order to be unchanged by a write. The
   docstring at ``fdtd_grid.py:114-119`` explains why: PML *update* order must
   not vary between models, because a different summation order changes the
   floating-point result.

``test_values_are_coerced_to_int``
   Expects every stored thickness to be a builtin ``int``.

TestAddSource
^^^^^^^^^^^^^

``add_source`` routes five source types to five lists
(``fdtd_grid.py:229-241``).

``test_voltage_source``, ``test_hertzian_dipole``, ``test_magnetic_dipole``
   Each expects its source to land in the matching list.

``test_transmission_line``, ``test_discrete_plane_wave``
   The same, for the two source types whose constructors take arguments:
   ``TransmissionLine`` takes ``(iterations, dt)`` and ``DiscretePlaneWave``
   takes the grid itself.

``test_unknown_type_raises_type_error``
   Expects ``TypeError`` for anything that is not a known ``Source`` subclass.

``test_each_source_lands_in_exactly_one_list``
   Adding one ``HertzianDipole`` expects exactly one of the five lists to be
   non-empty. A source in two lists would be updated twice per time step.

``test_sources_accumulate_in_order``
   Expects insertion order to be preserved.

TestAddReceiver
^^^^^^^^^^^^^^^

``test_appends``, ``test_accumulates_in_order``
   Expects receivers to append to ``rxs`` in order.

TestWithinBounds
^^^^^^^^^^^^^^^^

The contract the geometry and fractals suites imitated in their stubs, tested
here against the real implementation (``fdtd_grid.py:569-588``).

``test_interior_point_returns_true``, ``test_origin_is_inside``
   Expects ``True`` (a builtin ``bool``, from a literal ``return True``).

``test_upper_bound_is_inclusive``
   On an ``(8, 9, 10)`` grid, expects ``(8, 9, 10)`` to be inside. The check is
   ``p > n``, so ``p == n`` is legal — a point may sit on the far face.

``test_out_of_bounds_raises_naming_the_axis`` (6 parameter sets)
   Expects ``ValueError`` whose message is the offending axis letter, for each
   axis below zero and above the size.

``test_x_is_checked_before_y``
   With both x and y out of range, expects the error to name ``x`` —
   establishing that checks run in x, y, z order.

TestDiscretisePoint
^^^^^^^^^^^^^^^^^^^

``test_on_lattice_point_is_exact``, ``test_origin``
   Expects ``(3*DL, 4*DL, 5*DL)`` to discretise to ``(3, 4, 5)`` exactly.

``test_uses_the_matching_axis_of_dl``
   With anisotropic ``dl``, expects each coordinate to be divided by its own
   axis spacing.

``test_returns_plain_ints``
   Expects builtin ``int``, not a numpy integer.

``test_rounds_halves_toward_zero``
   With ``dl == 1.0``, expects ``(2.5, 3.5, 4.5)`` to give ``(2, 3, 4)``.
   ``round_value`` uses ``ROUND_HALF_DOWN`` — ties go toward zero. This is
   neither half-up (which gives 3, 4, 5) nor banker's rounding (which gives
   2, 4, 4).

``test_rounds_to_nearest_when_not_a_tie``
   Expects ``(2.4, 3.6, 4.5001)`` to give ``(2, 4, 5)``.

TestRoundToGrid
^^^^^^^^^^^^^^^

``test_on_lattice_point_round_trips``
   Expects a point already on the lattice to survive discretise-then-multiply
   unchanged.

``test_snaps_off_lattice_point_to_nearest_cell``
   Expects ``(2.4, 3.6, 0.0)`` at ``dl == 1.0`` to give ``(2.0, 4.0, 0.0)``.

``test_uses_the_matching_axis_of_dl``
   Anisotropic round trip.

``test_is_idempotent``
   Expects rounding an already-rounded point to be a no-op.

TestWithinPml
^^^^^^^^^^^^^

On a 20³ grid with 2-cell slabs (``fdtd_grid.py:619-635``). The method returns
``np.bool_`` rather than a builtin ``bool``, so the assertions coerce with
``bool()``.

``test_centre_is_not_in_pml``
   Expects the domain centre to be outside every slab.

``test_points_inside_each_slab`` (6 parameter sets)
   Expects one point inside each of the six slabs to report ``True``.

``test_inner_face_of_low_slab_is_outside``
   The test is ``p < thickness``, so ``p == thickness`` is already interior.

``test_inner_face_of_high_slab_is_outside``
   The test is ``p > n - thickness``, so ``p == n - thickness`` is interior.

``test_zero_thickness_means_nothing_is_in_pml``
   With the PML off, expects even the origin to report ``False``.

TestGetWaveformById
^^^^^^^^^^^^^^^^^^^

``test_returns_the_matching_waveform``
   Expects the waveform whose ``ID`` matches, by identity.

``test_returns_the_first_match``
   With two waveforms sharing an ID, expects the first.

TestGridIdentity
^^^^^^^^^^^^^^^^

``test_default_name``
   Expects ``"main_grid"``.

``test_id_lookup_covers_all_six_components``
   Expects ``IDlookup == {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4,
   "Hz": 5}``. These indices are used directly by the Cython kernels.

``test_collections_start_empty``
   Expects ``materials``, ``mixingmodels``, ``fractalvolumes``, ``waveforms``,
   ``rxs`` and ``snapshots`` all empty.

``test_collections_are_not_shared_between_instances``
   Adding a receiver to one grid expects another grid's list to stay empty. A
   class-level mutable default would leak sources between models in a
   multi-model run.

``test_average_volume_objects_defaults_true``
   Expects ``averagevolumeobjects is True``.

When these fail
~~~~~~~~~~~~~~~

**An axis-specific getter or setter test** — an index changed in the property
block. ``DL_ANISO`` is chosen so a wrong axis always produces a wrong number;
if one of these fails, compare the property's index against the axis in its
name. Note the same defect already exists in ``Model`` (see
`Deliberately Untested Paths`_).

**A rounding test** — ``round_value``'s behaviour changed. It is
``ROUND_HALF_DOWN`` for integers and ``ROUND_FLOOR`` for decimal places. Both
are deliberate: the second is what keeps the CFL time step below its limit, so
changing the rounding mode is a stability change, not a formatting one.

**A** ``within_bounds`` **test** — the inclusive upper bound is easy to lose in
a refactor from ``>`` to ``>=``. Geometry commands rely on being able to place
an object flush against the far face.

**A** ``set_pml_thickness`` **test** — check the ``OrderedDict`` is still
ordered and that the 6-tuple order is still ``x0, y0, z0, xmax, ymax, zmax``.
Reordering silently changes which face gets which thickness.

**``test_each_source_lands_in_exactly_one_list``** — a new branch in
``add_source`` that falls through to two appends, or an ``isinstance`` chain
whose order lets a subclass match the wrong arm.

Test Catalog — ``test_grid_arrays.py``
--------------------------------------

Array allocation, memory estimation and the 2D modes. The Yee lattice has one
more *node* than *cell* along each axis, which is why field and ``ID`` arrays
are ``(nx+1, ny+1, nz+1)`` while the cell-centred ``solid`` and ``rigid``
arrays are ``(nx, ny, nz)``. Source: ``fdtd_grid.py:650-863``.

TestInitialiseGeometryArrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_solid_is_cell_centred``
   Expects ``solid.shape == (nx, ny, nz)``.

``test_rigid_e_has_twelve_components``
   Expects ``rigidE.shape == (12, nx, ny, nz)`` — twelve edges per Yee cell.

``test_rigid_h_has_six_components``
   Expects ``rigidH.shape == (6, nx, ny, nz)`` — six faces per Yee cell.

``test_id_is_node_centred_with_six_components``
   Expects ``ID.shape == (6, nx+1, ny+1, nz+1)`` — six field components on a
   node-centred lattice.

``test_solid_starts_as_free_space``, ``test_id_starts_as_free_space``
   Expects every element to be ``1``. Material 1 is free space; ``0`` is PEC,
   so initialising to zero would fill the domain with metal.

``test_rigid_arrays_start_permissive``
   Expects every element to be ``0`` — zero means dielectric smoothing is
   allowed at that edge or face.

``test_dtypes`` (4 parameter sets)
   Expects ``solid`` and ``ID`` to be ``uint32``, both rigid arrays ``int8``.

``test_reallocates_on_a_second_call``
   A second call expects the arrays to be reset, not preserved.

TestInitialiseFieldArrays
^^^^^^^^^^^^^^^^^^^^^^^^^

``test_all_six_components_are_node_centred`` (6 parameter sets)
   Expects each of ``Ex``…``Hz`` to be ``(nx+1, ny+1, nz+1)``.

``test_all_six_components_start_at_zero`` (6 parameter sets)
   Expects a quiescent initial field.

``test_dtype_comes_from_config`` (6 parameter sets)
   Expects ``config.sim_config.dtypes["float_or_double"]``, not a hard-coded
   dtype — this is what single/double precision switching relies on.

``test_components_are_distinct_arrays``
   Writing ``Ex`` expects ``Ey``, ``Ez`` and ``Hx`` to stay zero. Aliasing any
   two would couple unrelated field components.

TestInitialiseUpdateCoeffArrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_shape_follows_material_count``
   With three materials, expects ``updatecoeffsE.shape == (3, 5)`` and the
   same for ``updatecoeffsH``. The row index is the material's ``numID``.

``test_starts_at_zero``
   Expects zero-filled arrays.

``test_empty_material_list_gives_empty_arrays``
   Expects ``(0, 5)``. Worth knowing: with no materials the array has no rows,
   so a kernel indexing by ``ID`` reads out of bounds.

TestInitialiseDispersiveArrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_shape_includes_the_pole_count``
   With ``maxpoles == 2``, expects ``Tx.shape == (2, nx+1, ny+1, nz+1)`` and
   the same for ``Ty`` and ``Tz``.

``test_dtype_comes_from_model_config``
   Expects ``config.get_model_config().materials["dispersivedtype"]``.

``test_update_coeff_array_has_three_entries_per_pole``
   With two materials and two poles, expects
   ``updatecoeffsdispersive.shape == (2, 6)``.

TestResetFields
^^^^^^^^^^^^^^^

``test_zeroes_the_field_arrays``
   Expects previously written field values to be cleared.

``test_preserves_the_shapes``
   Expects the array shapes to survive the reset.

``test_does_not_touch_geometry_arrays``
   Expects a written ``solid`` cell to survive. Only fields are cleared between
   runs of a multi-model job; the built geometry is reused.

``test_allocates_dispersive_arrays_when_poles_present``
   With ``maxpoles == 1``, expects ``Tx`` to be allocated by the reset.

TestMemoryEstimates
^^^^^^^^^^^^^^^^^^^

``test_basic_matches_hand_arithmetic``
   For a 4³ grid with no PML, expects exactly ``solid + rigid + fields``
   bytes, where ``solid = 4³ × 4``, ``rigid = 18 × 4³ × 1`` and
   ``fields = 12 × 5³ × 8``.

``test_basic_grows_with_pml_thickness``, ``test_basic_grows_with_domain_size``
   Directional checks.

``test_dispersive_matches_hand_arithmetic``
   With two poles, expects ``3 × 2 × 5³ × itemsize(complex128)``.

``test_dispersive_is_zero_without_poles``, ``test_fractals_is_zero_with_no_volumes``
   Expects zero when the feature is unused.

TestTwoDimensionalModes
^^^^^^^^^^^^^^^^^^^^^^^

Each 2D TM mode makes one axis invariant by forcing the two in-plane electric
components to PEC (material ``0``) on the first two node layers.
Source: ``fdtd_grid.py:835-863``.

``test_tmx_zeroes_ey_and_ez_on_the_first_two_x_layers``
   Expects ``ID[1, 0:2] == 0`` and ``ID[2, 0:2] == 0``.

``test_tmy_zeroes_ex_and_ez_on_the_first_two_y_layers``
   Expects ``ID[0, :, 0:2] == 0`` and ``ID[2, :, 0:2] == 0``.

``test_tmz_zeroes_ex_and_ey_on_the_first_two_z_layers``
   Expects ``ID[0, :, :, 0:2] == 0`` and ``ID[1, :, :, 0:2] == 0``.

``test_tmx_leaves_ex_untouched``, ``test_tmy_leaves_ey_untouched``, ``test_tmz_leaves_ez_untouched``
   Expects the out-of-plane component to remain entirely ``1``.

``test_magnetic_components_are_never_touched`` (3 parameter sets)
   Expects ``ID[3:] == 1`` after any of the three modes — only the electric
   components are forced to PEC.

``test_tmx_changes_exactly_the_documented_cells``
   Pins the whole footprint, not a sample: expects the changed set to be
   exactly components ``{1, 2}`` × ``i`` in ``{0, 1}`` × all ``j`` × all ``k``.

``test_modes_are_idempotent``
   Applying a mode twice expects no further change.

When these fail
~~~~~~~~~~~~~~~

**A shape assertion** — the array layout changed. The ``+1`` is not padding: a
Yee cell has one more node than cell per axis, and ``ID`` is node-centred while
``solid`` is cell-centred. If you meant to change a layout, every Cython kernel
indexing that array has to change with it, and the memory estimates will be
wrong too.

**An initial-value assertion** — ``solid`` and ``ID`` start at ``1``, not
``0``. Initialising them to zero fills the domain with PEC, which does not
raise; it produces a model that reflects everything.

**A dtype assertion** — the field dtype must come from config, not be
hard-coded, or the ``-precision`` option silently stops working.

**``test_components_are_distinct_arrays``** — an allocation refactor that
reuses one buffer, typically ``np.broadcast_to`` or a shared ``zeros`` result.

**A ``mem_est_*`` assertion** — either the array layout changed (see above) or
the estimate drifted from it. These two must move together; the estimate is
what the pre-run memory check reports to the user.

**A 2D-mode assertion** — check which component indices and which two layers
were zeroed. ``test_tmx_changes_exactly_the_documented_cells`` prints the exact
cell set, so a diff shows whether the slice moved or widened.

Test Catalog — ``test_grid_dt.py``
----------------------------------

The CFL time step and the Yee contour current sums. See `The CFL Time Step`_
for the formula. Source: ``fdtd_grid.py:865-945``.

TestCalculateDt3D
^^^^^^^^^^^^^^^^^

``test_matches_the_closed_form``
   With ``dl == (0.001, 0.002, 0.004)``, expects
   ``dt == 1 / (c * sqrt(1/dx² + 1/dy² + 1/dz²))``.

``test_isotropic_grid``
   Expects ``dt == DL / (c * sqrt(3))``.

``test_never_exceeds_the_cfl_limit``
   Expects ``dt <=`` the exact limit. The rounding is ``ROUND_FLOOR``
   precisely so the stored value can never land above it.

``test_stays_within_the_limit_for_various_spacings`` (4 parameter sets)
   Both the equality and the inequality, across isotropic and strongly
   anisotropic spacings.

``test_finer_grid_gives_a_smaller_time_step``
   Directional.

``test_halving_the_spacing_halves_the_time_step``
   For an isotropic grid the limit is linear in the spacing.

``test_rounded_to_one_less_than_hardware_precision``
   Expects ``dt`` to be unchanged by re-quantising at
   ``decimal.getcontext().prec - 1`` places with ``ROUND_FLOOR``.

TestCalculateDt2D
^^^^^^^^^^^^^^^^^

``test_tmx_uses_y_and_z``, ``test_tmy_uses_x_and_z``, ``test_tmz_uses_x_and_y``
   Each expects the closed form over only its two in-plane axes.

``test_2d_time_step_is_larger_than_3d`` (3 parameter sets)
   Dropping a term from the sum under the square root always relaxes the limit.

``test_unknown_mode_falls_back_to_3d``
   An unrecognised mode string expects the 3D formula — the terminal ``else``.

TestCalculateCurrents
^^^^^^^^^^^^^^^^^^^^^

``calculate_Ix`` / ``Iy`` / ``Iz`` sum the magnetic field around a Yee contour,
with guards returning exactly zero on the low faces where the contour would
need a cell at index ``-1``.

``test_ix_is_zero_on_the_low_y_or_z_faces`` (3 parameter sets)
   Expects exactly ``0`` — not merely small — when ``y == 0`` or ``z == 0``.

``test_iy_is_zero_on_the_low_x_or_z_faces`` (3 parameter sets)
   The same guard for ``x == 0`` or ``z == 0``.

``test_iz_is_zero_on_the_low_x_or_y_faces`` (3 parameter sets)
   The same guard for ``x == 0`` or ``y == 0``.

``test_ix_contour_sum``, ``test_iy_contour_sum``, ``test_iz_contour_sum``
   Each expects the documented two-term sum, for example
   ``Ix = dy*(Hy[x,y,z-1] - Hy[x,y,z]) + dz*(Hz[x,y,z] - Hz[x,y-1,z])``,
   evaluated against hand-built ramp fields.

``test_uniform_field_gives_zero_current``
   A spatially constant H has no curl, so every contour sum expects zero.

``test_zero_field_gives_zero_current``
   Expects exactly ``0``.

``test_current_scales_with_field_magnitude``
   The contour sum is linear in H: doubling the field expects double the
   current.

``test_uses_the_matching_axes_of_dl``
   ``Ix`` is weighted by ``dy`` and ``dz``, never ``dx``. With anisotropic
   spacing, expects ``dy*1 + dz*1`` for a two-cell excitation.

When these fail
~~~~~~~~~~~~~~~

**Any ``calculate_dt`` assertion** — treat this as a stability change, never a
rounding tweak. A ``dt`` above the CFL limit does not fail loudly; the
simulation diverges after a few hundred iterations, and the output looks like a
physical instability rather than a bug. If you deliberately changed the
formula, ``test_never_exceeds_the_cfl_limit`` is the one assertion that must
keep passing whatever else moves.

**A 2D-mode assertion** — check which two axes the branch uses. Each mode drops
its *invariant* axis: TMx drops x, so it uses ``dy`` and ``dz``.

**``test_rounded_to_one_less_than_hardware_precision``** — the rounding mode
changed from ``ROUND_FLOOR``. Rounding to nearest is not acceptable here; it
can round *up* past the limit.

**A contour-sum assertion** — check the sign convention and which neighbour
cell each term reads. The guards returning zero on the low faces are not an
approximation; the contour genuinely does not exist there.

Test Catalog — ``test_grid_build.py``
-------------------------------------

PML slab construction, the ``build()`` assembly line, and dispersion-analysis
reporting. ``build()``'s sub-steps are heavy — Cython Yee-cell builders, the
whole PML stack, progress bars — so the orchestration tests patch them out and
assert *which* steps run under *which* configuration.
Source: ``fdtd_grid.py:246-424``, ``947-1144``.

TestConstructPml
^^^^^^^^^^^^^^^^

Each of the six slab IDs maps to a specific box. Note ``PML.__init__`` runs
``check_kappamin()``, which sums ``kappa.min`` over the grid's CFS list and
rejects a total below one — so a grid with no CFS cannot construct a PML at
all. The fixture installs a default ``CFS``, exactly as ``build()`` does before
calling ``_build_pmls``.

``test_direction`` (6 parameter sets)
   Expects ``x0`` to give direction ``"xminus"``, ``xmax`` to give
   ``"xplus"``, and so on for the other four.

``test_x0_box``
   On a ``(20, 21, 22)`` grid with thickness 4, expects ``xs, xf == 0, 4`` and
   the transverse extent to span the whole face.

``test_xmax_box_is_measured_from_the_far_face``
   Expects ``xs, xf == 16, 20`` — measured inward from ``nx``.

``test_y0_box``, ``test_ymax_box_is_measured_from_the_far_face``, ``test_z0_box``, ``test_zmax_box_is_measured_from_the_far_face``
   The same for the other two axes.

``test_thickness_is_honoured`` (6 parameter sets)
   Expects ``pml.thickness == 7`` for a requested 7.

``test_unknown_id_raises``
   Expects ``ValueError`` matching ``"Unknown PML ID"``.

``test_returns_the_requested_type``
   Expects a ``PML`` instance.

``test_slab_spans_the_full_transverse_extent``
   A slab always covers the whole face it sits on.

TestBuildOrchestration
^^^^^^^^^^^^^^^^^^^^^^

``test_runs_the_standard_steps``
   Expects ``_build_components``, ``_tm_grid_update`` and ``_build_materials``
   all to run.

``test_installs_a_default_cfs_when_none_given``
   Expects exactly one ``CFS`` to be created when the user supplied none.

``test_keeps_a_user_supplied_cfs``
   Expects a user-provided CFS list to be left alone.

``test_builds_pmls_when_any_slab_is_non_zero``, ``test_builds_pmls_when_only_one_slab_is_non_zero``
   Expects ``_build_pmls`` to run if *any* thickness is non-zero.

``test_skips_pmls_when_all_thicknesses_are_zero``
   Expects ``_build_pmls`` to be skipped entirely.

``test_averaging_gates_component_building``
   With ``averagevolumeobjects = False``, expects ``_build_components`` not to
   run.

``test_allocates_field_arrays``, ``test_allocates_update_coefficient_arrays``
   Expects ``Ex.shape == (9, 9, 9)`` on an 8³ grid, and
   ``updatecoeffsE.shape == (2, 5)`` for two materials.

``test_skips_dispersive_arrays_without_poles``
   With ``maxpoles == 0``, expects ``Tx`` never to be created.

``test_allocates_dispersive_arrays_with_poles``
   With ``maxpoles == 2``, expects ``Tx.shape == (2, 9, 9, 9)`` and
   ``updatecoeffsdispersive.shape == (2, 6)``.

``test_initialises_snapshots``
   Expects ``initialise_snapfields()`` to be called on every registered
   snapshot.

TestDispersionAnalysisWaveformBranches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The branches reached before any FFT.

``test_no_waveform``
   Expects ``results["error"] == "no waveform detected."``.

``test_impulse_waveform``
   Expects ``"impulse waveform used."``.

``test_user_waveform``
   Expects ``"user waveform detected."``.

``test_continuous_waveforms_use_four_times_the_frequency`` (2 parameter sets)
   For ``sine`` and ``contsine`` at 1 GHz, expects ``maxfreq == 4e9``. A
   material must be present: once ``maxfreq`` is populated the method looks up
   the highest-permittivity material with a bare ``next()``.

``test_results_keys``
   Expects exactly ``{"deltavp", "N", "material", "maxfreq", "error"}``.

``test_no_waveform_leaves_metrics_unset``
   Expects ``N`` and ``deltavp`` to remain ``None``.

TestDispersionAnalysisReporting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``_dispersion_analysis`` is stubbed so each reporting branch can be driven
directly, without constructing a waveform whose spectrum lands in a particular
place.

``test_error_is_warned_not_raised``
   An error string expects a ``WARNING`` containing ``"not carried out"``, and
   no exception.

``test_undersampled_grid_raises``
   ``N == 1`` against ``mingridsampling == 3`` expects ``ValueError``.

``test_sampling_at_the_threshold_does_not_raise``
   ``N == 3`` expects no exception — the comparison is ``<``, not ``<=``.

``test_large_phase_error_is_warned``
   ``deltavp == 50`` against a 2 % threshold expects a ``WARNING`` mentioning
   numerical dispersion.

``test_small_phase_error_is_reported_at_info``
   ``deltavp == 0.5`` expects an ``INFO`` line mentioning the phase-velocity
   error.

When these fail
~~~~~~~~~~~~~~~

**A ``_construct_pml`` box assertion** — an index changed. The ``max`` slabs
are measured *inward from the far face* (``nx - thickness`` to ``nx``); the
``0`` slabs from the origin. Getting that backwards puts the absorber in the
wrong place without raising.

**Any ``_construct_pml`` test failing with a bare ``ValueError``** — most
likely ``check_kappamin``: the grid has no CFS. That is a fixture problem, not
a source regression.

**A ``build()`` gating assertion** — a condition changed. Each corresponds to a
documented user-facing switch: ``averagevolumeobjects`` is ``#averaging``,
``maxpoles`` follows from the materials in the model, and all-zero PML
thickness is how ``#pml_cells 0`` turns the boundary off.

**A dispersion-analysis message assertion** — the exact error strings are
matched. If you rewrite a message, update the test; these strings are what the
user sees when a model is undersampled.

**A reporting test raising instead of warning** — the severity of a diagnostic
changed. ``dispersion_analysis`` is advisory: only the ``mingridsampling``
breach is meant to raise.

Test Catalog — ``test_model.py``
--------------------------------

``Model`` is a thin owner of exactly one ``FDTDGrid``: almost every attribute
is a property forwarding to ``self.G``. ``Model.__init__`` calls
``set_omp_threads``, which reads host CPU information unrelated to the class
under test, so it is patched out. Source: ``model.py:53-193``.

TestConstruction
^^^^^^^^^^^^^^^^

``test_creates_an_fdtd_grid_for_the_cpu_solver``
   Expects ``model.G`` to be an ``FDTDGrid``.

``test_defaults``
   Expects ``title == ""``, ``dt_mod == 1.0``, ``iteration == 0``.

``test_collections_start_empty``
   Expects ``subgrids``, ``geometryviews`` and ``geometryobjects`` all empty.

``test_collections_are_not_shared_between_instances``, ``test_each_model_owns_a_distinct_grid``
   Two models expect independent state — the multi-model run depends on it.

TestSizeForwarding
^^^^^^^^^^^^^^^^^^

``test_setter_writes_through_to_grid`` (3 parameter sets)
   Expects ``model.nx = 12`` to leave ``model.G.nx == 12``.

``test_getter_reads_from_grid`` (3 parameter sets)
   Expects a value written on the grid to be visible through the model.

``test_round_trip`` (3 parameter sets)
   Expects ``model.nx`` and ``model.G.nx`` to stay in step both ways.

``test_axes_are_independent``
   ``set_size([3, 4, 5])`` expects ``(3, 4, 5)``.

TestSetSize
^^^^^^^^^^^

``test_unpacks_all_three_axes``, ``test_accepts_a_plain_sequence``
   Expects both a numpy array and a plain list to work.

TestCells
^^^^^^^^^

``test_is_the_product_of_the_three_axes``
   For ``(2, 3, 4)`` expects ``24``.

``test_is_uint64``
   Expects ``np.uint64``. The count is deliberately widened — a large 3D domain
   overflows a 32-bit product.

``test_zero_sized_model_has_no_cells``
   Expects ``0``.

``test_does_not_overflow_on_a_large_domain``
   For ``2000³`` expects exactly ``8_000_000_000``.

TestDiscretisationForwarding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_dx_getter_reads_the_x_spacing``
   Expects ``model.dx == dl[0]``.

``test_setters_write_the_correct_axis`` (3 parameter sets)
   Expects ``model.dy = 0.009`` to write ``dl[1]``, and correspondingly for
   ``dx`` and ``dz``. All three *setters* are correct.

``test_dl_forwards_the_whole_array``, ``test_dl_setter_writes_through``
   Expects the whole ``dl`` array to round-trip.

No test reads ``model.dy`` or ``model.dz`` back — those getters are defective;
see `Deliberately Untested Paths`_.

TestTimeForwarding
^^^^^^^^^^^^^^^^^^

``test_dt_round_trips``, ``test_iterations_round_trips``, ``test_timewindow_round_trips``
   Each expects the value to reach the grid and read back unchanged.

``test_dt_mod_is_model_level_not_grid_level``
   Expects the grid *not* to have a ``dt_mod``. It is one of the few attributes
   ``Model`` owns outright, and subgrids inherit it during ``setup()``.

TestStepForwarding
^^^^^^^^^^^^^^^^^^

``test_srcsteps_round_trips``, ``test_rxsteps_round_trips``
   Expects the step arrays to reach the grid intact.

TestCreateGrid
^^^^^^^^^^^^^^

``test_cpu_solver_gives_a_plain_fdtd_grid``
   Expects exactly ``FDTDGrid`` for ``solver == "cpu"``. The cuda / opencl /
   metal branches need real device handles and belong to a later PR.

``test_returns_a_new_grid_each_call``
   Expects a fresh instance per call.

When these fail
~~~~~~~~~~~~~~~

**A forwarding round-trip** — a property lost its delegation, or picked up the
wrong index. This file exists because that class of slip is invisible in normal
use: ``Model`` has sixteen near-identical forwarding properties, and two of
them are already wrong in the source.

**``test_is_uint64`` or the overflow test** — the accumulator dtype narrowed. A
2000³ domain is realistic and overflows a 32-bit product; the failure would
surface as a nonsensical memory estimate rather than an exception.

**``test_dt_mod_is_model_level_not_grid_level``** — ``dt_mod`` moved onto the
grid. That is a design change: it would give each subgrid an independent
stability factor rather than inheriting the model's.

Test Catalog — ``test_subgrid_base.py``
---------------------------------------

``SubGridBaseGrid``'s constructor arithmetic. Every dimension of a Huygens
subgrid derives from ``ratio``; this file pins that derivation against
hand-computed values, so a refactor that changes a size formula has to change
these numbers deliberately. Source: ``subgrids/grid.py:28-88``.

TestRatioValidation
^^^^^^^^^^^^^^^^^^^

``test_odd_ratios_are_accepted`` (5 parameter sets)
   Expects ``ratio`` in ``{1, 3, 5, 7, 9}`` to construct and be stored.

``test_even_ratios_are_rejected`` (4 parameter sets)
   Expects ``ValueError`` for ``{2, 4, 6, 8}``. With an even ratio the fine and
   coarse cell centres never coincide.

``test_zero_ratio_is_rejected``
   Expects ``ValueError`` — ``0 % 2 == 0``, so the parity guard catches it.

TestRequiredKwargs
^^^^^^^^^^^^^^^^^^

``test_each_kwarg_is_required`` (7 parameter sets)
   Dropping any one of ``ratio``, ``id``, ``filter``, ``is_os_sep``,
   ``pml_separation``, ``subgrid_pml_thickness`` or ``interpolation`` expects
   ``KeyError``. The constructor reads them all from ``kwargs`` with no
   defaults.

``test_all_kwargs_present_constructs``
   The positive control.

TestSizeDerivation
^^^^^^^^^^^^^^^^^^

``test_s_is_os_sep_scales_with_ratio``
   With ``is_os_sep=4, ratio=3``, expects ``s_is_os_sep == 12``.

``test_s_is_os_sep_for_various_ratios`` (4 parameter sets)
   Expects ``is_os_sep * ratio`` for ratios 1, 3, 5, 7.

``test_n_boundary_cells_sums_the_three_gaps``
   With ``ratio=3, is_os_sep=2, pml_separation=4, subgrid_pml_thickness=5``,
   expects ``n_boundary_cells == 6 + 4 + 5 == 15``.

``test_per_axis_boundary_cells_match_the_scalar`` (3 parameter sets)
   All six PML thicknesses come from one kwarg, so the three per-axis counts
   are equal on construction.

``test_boundary_cells_grow_with_pml_thickness``
   Increasing the thickness by 8 expects the boundary count to grow by 8.

TestPmlThickness
^^^^^^^^^^^^^^^^

``test_all_six_faces_take_the_single_kwarg``
   Expects every slab to equal ``subgrid_pml_thickness``.

``test_overrides_the_fdtd_grid_default_of_ten``
   Expects the inherited default of 10 to be replaced.

``test_zero_thickness_is_allowed``
   Expects all six to be ``0``.

TestConstructorState
^^^^^^^^^^^^^^^^^^^^

``test_name_comes_from_the_id_kwarg``, ``test_name_overrides_the_fdtd_grid_default``
   Expects ``name`` to be the ``id`` kwarg, not ``"main_grid"``.

``test_iterations_start_at_zero``
   Expects ``0``; the real count is set later by the user object.

``test_filter_flag_is_stored``, ``test_interpolation_is_stored``, ``test_is_os_sep_is_stored``
   Constructor bookkeeping.

TestAbstractBase
^^^^^^^^^^^^^^^^

``test_base_class_cannot_be_instantiated``
   Expects ``TypeError`` — ``SubGridBaseGrid`` is an ABC with five abstract
   methods.

``test_subclass_missing_an_abstract_method_cannot_be_instantiated``
   A subclass implementing four of the five expects ``TypeError``.

``test_complete_subclass_can_be_instantiated``
   Implementing all five expects success.

``test_hsg_implements_the_whole_interface``
   Expects ``SubGridHSG`` to provide all five as callables.

TestInheritsFdtdGrid
^^^^^^^^^^^^^^^^^^^^

A subgrid *is* an ``FDTDGrid``, so the whole grid surface applies to it.

``test_is_an_fdtd_grid``
   Expects ``isinstance(sg, FDTDGrid)``.

``test_size_properties_work``
   With the fixture defaults, expects ``(nx, ny, nz) == (32, 32, 32)``.

``test_discretisation_is_ratio_times_finer``
   Expects ``dx == DL / 3``.

``test_within_bounds_contract_is_inherited``
   Expects ``True`` inside, and ``ValueError`` naming ``x`` outside.

``test_array_initialisers_are_inherited``
   Expects ``solid.shape == (32, 32, 32)`` and ``Ex.shape == (33, 33, 33)``.

``test_calculate_dt_uses_the_finer_spacing``
   Expects ``(DL / 3) / (c * sqrt(3))`` — a ratio-3 subgrid takes a time step
   three times smaller than its parent.

When these fail
~~~~~~~~~~~~~~~

**A size-derivation assertion** — one of the formulas changed. They chain, so a
single change cascades: ``s_is_os_sep`` feeds ``d_to_pml``, which feeds
``n_boundary_cells``, which feeds ``nx``. Work out which link moved before
adjusting any expected number, and check `The Huygens Subgrid`_ for the chain.

**A ratio-validation assertion** — the parity guard changed. Odd is not a
style preference; with an even ratio the two lattices disagree by half a cell
everywhere and the Huygens surface cannot be consistent.

**A ``KeyError`` test now passing a different exception** — a kwarg gained a
default. That is a real API change: it means a caller can omit the argument and
silently get a value they did not choose.

**An inherited-behaviour assertion** — a change in ``FDTDGrid`` reached the
subgrid. Check whether the corresponding test in ``tests/unit/grid/`` also
failed; if it did, fix that first.

Test Catalog — ``test_subgrid_commands.py``
-------------------------------------------

The ``#subgrid_hsg`` user object: child collection, the size helpers, and
``setup()``, which turns a parsed command into a wired-up grid registered on
the model. Source: ``subgrids/user_objects.py:42-223``.

TestCommandIdentity
^^^^^^^^^^^^^^^^^^^

``test_order``
   Expects ``18``, the construction order among user objects.

``test_hash``
   Expects ``"#subgrid_hsg"``.

``test_is_not_single_use``
   Expects ``False`` — a model may contain more than one subgrid.

``test_is_a_subgrid_base``
   Expects the command to subclass ``SubGridBase``.

TestKwargPassThrough
^^^^^^^^^^^^^^^^^^^^

``test_arguments_reach_kwargs`` (6 parameter sets)
   Expects ``ratio``, ``id``, ``is_os_sep``, ``subgrid_pml_thickness``,
   ``interpolation`` and ``filter`` to arrive in ``kwargs`` unchanged.

``test_points_reach_kwargs``
   Expects ``p1`` and ``p2`` to arrive unchanged.

``test_defaults_are_applied``
   Expects ``ratio == 3``, ``is_os_sep == 3``,
   ``subgrid_pml_thickness == 6``, ``filter is True``.

   ``pml_separation`` is deliberately absent from both tests — see
   `Deliberately Untested Paths`_.

TestChildCollection
^^^^^^^^^^^^^^^^^^^

``test_starts_with_no_children``
   Expects all three child lists empty.

``test_geometry_child_is_routed``, ``test_grid_child_is_routed``, ``test_output_child_is_routed``
   Expects each category to land in its own list.

``test_geometry_is_checked_before_grid``
   ``GeometryUserObject`` subclasses ``GridUserObject``, so the ``isinstance``
   chain must test the more specific type first. Expects a geometry child *not*
   to appear in ``children_grid``.

``test_unknown_child_raises``
   Expects ``ValueError`` for an object that is none of the three.

``test_children_accumulate_in_order``
   Expects insertion order preserved.

TestSetDiscretisation
^^^^^^^^^^^^^^^^^^^^^

``test_each_axis_is_divided_by_ratio`` (3 parameter sets)
   For ratios 3, 5, 7 expects each of ``dx``, ``dy``, ``dz`` to be the main
   grid's spacing divided by ``ratio``.

``test_anisotropic_main_grid_is_preserved``
   Expects ``(0.001, 0.002, 0.004) / 3`` — the anisotropy carries through.

TestCellCountHelpers
^^^^^^^^^^^^^^^^^^^^

``test_working_region_scales_by_ratio``
   For an IS from ``(2, 3, 4)`` to ``(6, 9, 14)`` at ratio 3, expects
   ``(nwx, nwy, nwz) == (12, 18, 30)``.

``test_total_cells_bracket_the_working_region``
   Expects ``nx == 2 * n_boundary_cells + nwx``, and likewise for y and z.

``test_iterations_scale_by_ratio`` (3 parameter sets)
   With 100 main iterations, expects ``100 * ratio``.

``test_name_comes_from_the_id_kwarg``
   Expects the grid's name to be set from the command's ``id``.

TestSetMainGridIndices
^^^^^^^^^^^^^^^^^^^^^^

``test_stores_the_cell_indices``
   Expects ``(i0, j0, k0)`` and ``(i1, j1, k1)`` to hold the two corners.

``test_stores_the_rounded_coordinates``
   Expects ``(x1, y1, z1)`` and ``(x2, y2, z2)`` to hold the same corners in
   metres, snapped to the lattice.

TestBuild
^^^^^^^^^

``build()`` constructs the grid and runs the shared ``setup()``.

``test_returns_an_hsg_grid``
   Expects a ``SubGridHSG`` grid instance.

``test_registers_the_subgrid_on_the_model``
   Expects ``model.subgrids == [sg]``.

``test_wires_the_parent_grid``
   Expects ``sg.parent_grid is model.G``.

``test_stores_the_subgrid_on_the_command``
   Expects ``cmd.subgrid is sg``, so nested children can find their grid.

``test_discretisation_is_ratio_times_finer``
   Expects ``sg.dx == model.G.dx / 3``.

``test_iterations_scale_by_ratio``
   Expects ``model.iterations * 3``.

``test_time_step_respects_the_stability_factor``
   With ``model.dt_mod == 0.5``, expects the subgrid's CFL step multiplied by
   ``0.5`` — the factor is inherited from the model, not recomputed.

``test_working_region_matches_the_requested_box``
   A ``p1``-to-``p2`` span of 6 main cells at ratio 3 expects ``nwx == 18``.

``test_copies_builtin_materials``
   Expects only materials whose ``type`` is ``"builtin"`` to reach the subgrid.

``test_materials_are_copied_not_aliased``
   Expects ``sg.materials[0] is not`` the main grid's object. The subgrid must
   be able to diverge without mutating the parent's materials.

``test_non_builtin_materials_are_not_copied``
   A user-defined material expects an empty subgrid materials list.

``test_two_subgrids_of_the_same_type_are_allowed``
   Expects two ``SubGridHSG`` subgrids to coexist.

``test_mixing_subgrid_types_raises``
   Expects ``ValueError`` when a second, differently-typed subgrid is added.

When these fail
~~~~~~~~~~~~~~~

**A kwarg pass-through assertion** — an argument stopped reaching ``kwargs``,
usually because it was consumed or overwritten in ``__init__``. That is exactly
the shape of the ``pml_separation`` defect already present in this file, so
check for an unconditional assignment before assuming the test is wrong.

**``test_geometry_is_checked_before_grid``** — the ``isinstance`` chain in
``add`` was reordered. Because ``GeometryUserObject`` *is* a
``GridUserObject``, testing the general case first silently swallows every
geometry child.

**``test_materials_are_copied_not_aliased``** — a ``copy`` was dropped from the
comprehension in ``setup()``. Aliasing here is a genuine bug: material property
calculation mutates the object, so the subgrid would corrupt the main grid's
materials.

**``test_time_step_respects_the_stability_factor``** — ``dt_mod`` stopped being
applied, or is now applied twice. Both are silent: the model still runs, just
at the wrong time step.

Test Catalog — ``test_precursor_nodes.py``
------------------------------------------

The translator between the two grids' time steps. See `The Huygens Subgrid`_
for the mechanism. Source: ``subgrids/precursor_nodes.py:28-283``.

TestWeightingCoefficients
^^^^^^^^^^^^^^^^^^^^^^^^^

``calculate_weighting_coefficients(x1, x)`` returns
``((x - x1) / x, x1 / x)`` — a linear partition of unity.

``test_at_the_start_all_weight_is_on_the_previous_value``
   Expects ``(1.0, 0.0)`` at ``m == 0``.

``test_at_the_end_all_weight_is_on_the_current_value``
   Expects ``(0.0, 1.0)`` at ``m == ratio``.

``test_midpoint_splits_evenly``
   Expects ``(0.5, 0.5)`` at ``m == 2, ratio == 4``.

``test_weights_sum_to_one`` (12 parameter sets)
   Expects ``c1 + c2 == 1`` for every combination of ratio and sub-step. Any
   other sum would scale the injected field.

``test_intermediate_values`` (3 parameter sets)
   Expects ``(2/3, 1/3)`` at ``m=1, ratio=3``, and similar.

``test_is_monotone_in_m``
   Expects the current-step weight to increase steadily. A non-monotone blend
   would make the source jitter within a main time step.

TestPrecursorConstruction
^^^^^^^^^^^^^^^^^^^^^^^^^

``test_copies_the_scaling_parameters_from_the_subgrid``, ``test_copies_the_is_indices``
   Expects ``ratio``, ``nwx``, ``interpolation`` and the six IS indices to
   match the subgrid.

``test_holds_references_to_the_main_grid_fields``
   Expects ``precursors.Ex is main.Ex`` — references, not copies, so the
   precursors see the main grid's live arrays.

``test_half_sub_cell_offset``
   Expects ``d == 1 / (2 * ratio)``, the Yee stagger offset.

``test_left_and_right_weights_partition_the_ratio``
   Expects ``l_weight + r_weight == ratio``.

``test_left_weight_is_the_floor_of_half``
   Expects ``ratio // 2``.

TestFieldArrayShapes
^^^^^^^^^^^^^^^^^^^^

Shapes follow the Yee stagger: a component has one fewer sample along its own
direction and one more on each transverse axis.

``test_front_face_electric_shapes``
   Expects ``ex_front_1.shape == (nwx, nwz + 1)`` and
   ``ez_front_1.shape == (nwx + 1, nwz)``.

``test_left_face_electric_shapes``
   Expects ``(nwy, nwz + 1)`` and ``(nwy + 1, nwz)``.

``test_bottom_face_electric_shapes``
   Expects ``(nwx, nwy + 1)`` and ``(nwx + 1, nwy)``.

``test_magnetic_shapes_are_the_transverse_swap``
   Expects ``hx_front`` to be shaped like ``ez_front``, and ``hz_front`` like
   ``ex_front``.

``test_opposite_faces_have_equal_shapes``
   Expects front/back, left/right and top/bottom to match.

``test_all_arrays_start_at_zero``
   Expects both time pages of all 24 named fields to be zero.

``test_previous_and_current_pages_are_distinct_arrays``
   Expects ``<field>_0 is not <field>_1``. If they aliased, the time
   interpolation would collapse to the current value and the subgrid would see
   a stepped source instead of a smooth one.

``test_faces_are_distinct_arrays``
   Writing one face expects the opposite face to stay zero.

TestFieldNameTables
^^^^^^^^^^^^^^^^^^^

``test_twelve_electric_names``, ``test_twelve_magnetic_names``
   Expects 12 entries each — 6 faces × 2 tangential components.

``test_names_are_unique``
   Expects no duplicates.

``test_every_name_resolves_to_both_pages``
   Expects ``<name>_0`` and ``<name>_1`` to exist for each.

``test_electric_and_magnetic_names_do_not_overlap``
   Expects the two tables to be disjoint.

``test_all_six_faces_appear_in_each_table``
   Expects front, back, left, right, top and bottom in both.

TestTimeInterpolation
^^^^^^^^^^^^^^^^^^^^^

Driven with ``_0 == 10.0`` and ``_1 == 20.0`` throughout.

``test_start_of_step_gives_the_previous_value``
   Expects ``10.0`` at ``m == 0``.

``test_end_of_step_gives_the_current_value``
   Expects ``20.0`` at ``m == ratio``.

``test_intermediate_step_blends``
   At ``m=1, ratio=3`` expects ``10 * 2/3 + 20 * 1/3``.

``test_electric_interpolation_leaves_magnetic_alone``
   Expects the magnetic working fields not to exist yet.

``test_magnetic_interpolation_covers_every_magnetic_name``, ``test_electric_interpolation_covers_every_electric_name``
   Expects all 12 working fields to be produced.

``test_exact_electric_takes_the_current_page``, ``test_exact_magnetic_takes_the_current_page``
   Expects ``20.0`` — the exact sample bypasses interpolation entirely.

``test_exact_field_is_a_copy_not_a_view``, ``test_interpolated_field_is_not_a_view``
   Writing the working field expects the stored pages to be unchanged. A view
   here would let a sub-step corrupt the sample it was derived from.

TestPageRotation
^^^^^^^^^^^^^^^^

``test_current_page_is_copied_into_previous``
   Expects ``_1``'s contents to appear in ``_0``.

``test_rotation_copies_by_value``
   Overwriting ``_1`` afterwards expects ``_0`` to keep the old value.

``test_rotation_covers_every_named_field``
   Expects all 12 fields of the requested type to rotate.

TestInterpolatedCoords
^^^^^^^^^^^^^^^^^^^^^^

``test_mid_branch_offsets_the_first_axis``
   Expects ``x[0] == 0.5`` and ``z[0] == 0.0``.

``test_non_mid_branch_offsets_the_second_axis``
   Expects ``x[0] == 0.0`` and ``z[0] == 0.5``.

``test_mid_branch_output_lengths``
   Expects ``len(x_sg) == n_x * ratio`` and
   ``len(z_sg) == (n_y - 1) * ratio + 1``.

``test_non_mid_branch_output_lengths``
   The mirror image.

``test_sample_coords_match_the_field_shape``
   Expects the sample coordinate arrays to match the input field's dimensions.

TestSpatialInterpolation
^^^^^^^^^^^^^^^^^^^^^^^^

``test_output_shape_follows_the_requested_coords``
   Expects the output to match the requested sub-grid coordinate lengths.

``test_constant_field_interpolates_to_the_same_constant``
   Expects ``2.5`` everywhere.

``test_linear_field_is_reproduced_by_linear_interpolation``
   With ``interpolation == 1`` the spline is linear, so a linear ramp must come
   back exactly. This is the strongest available correctness check that needs
   no reference implementation.

``test_zero_field_interpolates_to_zero``
   Expects zero out.

TestSliceTables
^^^^^^^^^^^^^^^

The tables describing how to slice the main grid for each face. Both
``PrecursorNodes`` and ``PrecursorNodesFiltered`` are checked (2 parameter sets
each).

``test_twelve_slices_per_field_type``
   Expects 12 magnetic and 12 electric slices.

``test_every_slice_targets_a_real_attribute``
   Expects each slice's target name to resolve.

``test_slice_targets_are_current_step_pages``
   Expects every target to end in ``_1`` — the tables fill the current page;
   the bare working names come later from the time interpolation.

``test_coords_are_resolved_at_construction``
   Expects ``obj[1]`` to be a 4-tuple. It starts as a boolean ``mid`` flag and
   is replaced in place by the resolved coordinates at the end of each table
   builder, so reading the update methods alone is misleading.

``test_magnetic_slices_carry_two_index_tuples``
   Expects 5 elements — H is averaged across the IS, so it needs samples either
   side.

``test_electric_slices_carry_one_index_tuple``
   Expects 4 elements.

TestUpdateFromMainGrid
^^^^^^^^^^^^^^^^^^^^^^

``test_update_electric_runs_and_fills_the_current_page``, ``test_update_magnetic_runs_and_fills_the_current_page``
   With the main grid's fields set to a constant, expects the precursor's
   current page to carry that constant.

``test_update_electric_rotates_the_pages_first``
   Expects the previous page to receive the old current page before the new
   sample is taken.

``test_zero_main_grid_gives_zero_precursors``
   Expects zero out.

``test_both_precursor_types_update`` (2 parameter sets)
   Expects both classes to run and produce correctly shaped output.

TestPrecursorTypes
^^^^^^^^^^^^^^^^^^

``test_unfiltered_type``, ``test_filtered_type``
   Expects ``create_updates``' choice to map to the right class.

``test_both_share_the_same_field_shapes``
   Expects the filter not to change any array shape.

When these fail
~~~~~~~~~~~~~~~

**A weighting-coefficient assertion** — the interpolation is no longer a
partition of unity, which scales the field injected into the subgrid. Amplitude
errors at a Huygens surface look like a reflection, not like a bug.

**A shape assertion** — the Yee stagger. ``ex_front`` is ``(nwx, nwz+1)``
while ``ez_front`` is ``(nwx+1, nwz)``; a component is offset by half a cell
along *its own* direction only. Get a ``+1`` wrong and the code still runs, with
the field wrong along one edge of one face — the class of bug no integration
test notices.

**``test_previous_and_current_pages_are_distinct_arrays``** — an allocation
refactor replaced a ``np.copy`` with an assignment. This silently disables time
interpolation entirely.

**A copy-not-view assertion** — same cause, one layer up. The working field
must not alias the stored page it was blended from.

**``test_linear_field_is_reproduced_by_linear_interpolation``** — either the
spline degree changed or the coordinate construction did. Check
``interpolation`` first; degree 2 or 3 will not reproduce a ramp exactly at the
domain edges.

**``test_coords_are_resolved_at_construction``** — the in-place replacement of
``obj[1]`` was removed or moved. The update methods pass ``obj[1]`` straight to
the interpolator, so an unresolved boolean there fails deep inside scipy with a
confusing message.

Test Catalog — ``test_subgrid_hsg.py``
--------------------------------------

Inner/Outer Surface field stitching, driven through the real OpenMP Cython
kernels against real arrays. Assertions are about **locality and effect**,
never physical correctness — verifying the stitched field is numerically right
needs an analytic reference solution and belongs to an integration suite.
Source: ``subgrids/subgrid_hsg.py:30-705``.

TestCoupledGridsFixture
^^^^^^^^^^^^^^^^^^^^^^^

The IS/OS tests are only meaningful if the fixture is self-consistent, so it is
asserted directly before anything relies on it. **If these fail, fix the
fixture before reading any other failure in this file.**

``test_subgrid_knows_its_parent``
   Expects ``sub.parent_grid is main``.

``test_inner_surface_sits_inside_the_main_grid``
   Expects ``0 < i0 < i1 < main.nx`` on all three axes.

``test_outer_surface_also_fits_inside_the_main_grid``
   Expects ``i0 - is_os_sep >= 0`` and ``i1 + is_os_sep <= main.nx``.

``test_precursor_slices_stay_inside_the_main_grid``
   Expects ``i0 - 1 >= 0`` — the magnetic slices reach one main cell below the
   IS, so the IS cannot sit on the domain boundary.

``test_working_region_scales_by_ratio``, ``test_total_size_brackets_the_working_region``
   The size chain, re-checked on the coupled pair.

``test_subgrid_time_step_is_finer``
   Expects ``sub.dt == main.dt / ratio``.

``test_both_grids_have_usable_update_coefficients``
   Expects the coefficient arrays to be non-zero. **This is the guard against
   vacuous passes** — with zero coefficients every kernel becomes a no-op and
   the locality assertions below would pass without testing anything.

``test_field_arrays_start_at_zero``
   Expects a quiescent starting state.

TestNoSpuriousInjection
^^^^^^^^^^^^^^^^^^^^^^^

``test_magnetic_is_with_zero_precursors``, ``test_electric_is_with_zero_precursors``
   With every precursor zero, expects the subgrid arrays to be bit-identical
   afterwards. A non-zero result would mean the Huygens surface is radiating on
   its own.

``test_electric_os_with_zero_subgrid_fields``, ``test_magnetic_os_with_zero_subgrid_fields``
   The same outward: a quiet subgrid expects to leave the main grid untouched.

TestInnerSurfaceLocality
^^^^^^^^^^^^^^^^^^^^^^^^

One excited precursor cell must change exactly one subgrid cell. With
``n = n_boundary_cells``, the mapping is
``precursor[a, b]`` → subgrid ``(n + a, n + b, layer)``.

``test_bottom_face_changes_exactly_one_cell``
   ``ex_bottom[5, 5] = 1`` expects the changed set to be exactly
   ``{(n + 5, n + 5, n - 1)}``.

``test_top_face_changes_exactly_one_cell``
   ``ex_top[5, 5] = 1`` expects exactly ``{(n + 5, n + 5, n + nwz)}``.

``test_bottom_and_top_are_independent``
   Exciting the bottom face expects no change on the top layer.

``test_mapping_holds_across_the_face`` (3 parameter sets)
   The same exact set equality at ``(0, 0)``, ``(3, 7)`` and ``(17, 17)``.

``test_two_excited_cells_change_two_cells``
   Expects exactly the two corresponding cells.

``test_effect_is_linear_in_the_precursor_value``
   Doubling the precursor expects double the change.

``test_electric_is_changes_cells_in_the_subgrid``
   The electric counterpart, driven by the magnetic precursors, expects a
   non-empty change.

``test_electric_is_stays_within_the_working_region``
   Expects all changed indices to lie between ``n - 1`` and
   ``n + nwx + 1``.

TestOuterSurfaceLocality
^^^^^^^^^^^^^^^^^^^^^^^^

The OS updates write into the **main** grid, and only near the Outer Surface.

``test_electric_os_writes_into_the_main_grid``
   Expects a non-empty change when the subgrid carries a field.

``test_electric_os_stays_within_the_outer_surface`` (3 parameter sets)
   For each of ``Ex``, ``Ey``, ``Ez``, expects every changed index to lie
   within ``[i0 - is_os_sep, i1 + is_os_sep]``.

``test_magnetic_os_stays_within_the_outer_surface`` (3 parameter sets)
   The same for ``Hx``, ``Hy``, ``Hz``, but reaching one cell further on the
   low side — magnetic nodes sit half a cell back.

``test_far_field_is_untouched``
   Expects the domain's origin and far corner to stay exactly zero.

``test_electric_os_does_not_disturb_the_magnetic_field``, ``test_magnetic_os_does_not_disturb_the_electric_field``
   Expects each OS update to touch only its own field type.

``test_os_effect_is_linear``
   Doubling the subgrid field expects double the main-grid change.

TestRatioVariations
^^^^^^^^^^^^^^^^^^^

Every supported refinement factor (3 parameter sets each, ratios 3, 5, 7).

``test_inner_surface_mapping_holds``
   Expects the same exact single-cell mapping at every ratio.

``test_working_region_scales``
   Expects ``nwx == (i1 - i0) * ratio``.

``test_outer_surface_stays_local``
   Expects the OS writes to stay bounded at every ratio.

TestPrintInfo
^^^^^^^^^^^^^

``print_info`` **logs** and returns ``None``, so these use ``caplog``.

``test_returns_none``
   Expects ``None``.

``test_reports_the_ratio``
   Expects ``"1:3"`` in the log text.

``test_names_the_grid``
   Expects the subgrid's name.

``test_reports_the_working_region_cell_count``
   Expects ``nwx * nwy * nwz`` to appear.

``test_reports_the_time_step``
   Expects ``"Time step"``.

When these fail
~~~~~~~~~~~~~~~

**Any ``TestCoupledGridsFixture`` assertion** — stop and fix the fixture. Every
other failure in this file is meaningless until the coupled pair is consistent.

**A ``TestNoSpuriousInjection`` assertion** — the most serious failure in this
file. It means the Huygens surface injects energy with no source, which in a
real model appears as a spurious reflection growing from the subgrid boundary.

**An exact single-cell locality assertion** — an index changed in the kernel or
in ``n_boundary_cells``. The failure prints both cell sets: if the expected and
actual differ by a constant offset, the boundary-cell count moved; if they
differ in shape, a loop bound did.

**All locality tests pass but the suite feels too fast** — check
``test_both_grids_have_usable_update_coefficients``. Zeroed coefficients make
every assertion vacuous.

**An OS-bound assertion** — the write region grew. That is not necessarily
wrong, but it must be deliberate: the OS box is defined by ``is_os_sep``, and
writing outside it means the subgrid is modifying main-grid cells that the
Huygens formulation says it should not.

**A ``print_info`` assertion** — check the log level before the message text.

Test Catalog — ``test_subgrid_updates.py``
------------------------------------------

The HSG choreography. Each main time step drives the subgrid twice: ``hsg_1``
carries it from the main electric update to the main magnetic update, and
``hsg_2`` mirrors that back. Within each half the subgrid takes
``(ratio - 1) / 2`` *interpolated* steps and one final *exact* step, then hands
its result outward across the Outer Surface.

The sequence tests use ``spy_updater``, which records calls with an
owner prefix (``precursors.``, ``sub.``, bare for the updater) because the
three collaborators share method names.
Source: ``subgrids/updates.py:33-173``.

TestCreateUpdates
^^^^^^^^^^^^^^^^^

``test_returns_subgrid_updates``
   Expects a ``SubgridUpdates`` instance.

``test_one_updater_per_subgrid``
   With two subgrids, expects two updaters.

``test_filtered_subgrid_gets_filtered_precursors``
   ``filter=True`` expects ``PrecursorNodesFiltered``.

``test_unfiltered_subgrid_gets_plain_precursors``
   ``filter=False`` expects ``PrecursorNodes`` and *not* the filtered subclass.

``test_non_subgrid_raises``
   A non-HSG object in ``model.subgrids`` expects ``ValueError``.

``test_no_subgrids_gives_no_updaters``
   Expects an empty updater list.

``test_updater_holds_the_main_grid``, ``test_updater_holds_the_subgrid``
   Expects both references to be wired.

TestSubgridUpdaterState
^^^^^^^^^^^^^^^^^^^^^^^

``test_iteration_starts_at_zero``
   Expects ``0``.

``test_wires_up_its_three_collaborators``
   Expects ``grid``, ``precursors`` and ``G`` to point at the subgrid, the
   precursors and the main grid respectively.

``test_electric_sources_advance_the_iteration``
   Expects ``iteration == 1`` afterwards — the electric source update is the
   one that advances the subgrid clock.

``test_magnetic_sources_do_not_advance_the_iteration``, ``test_store_outputs_does_not_advance_the_iteration``
   Expects ``iteration == 0`` — these only read it.

TestHsgPhaseOne
^^^^^^^^^^^^^^^

``test_starts_by_sampling_the_main_electric_field``
   Expects the first call to be ``precursors.update_electric``.

``test_ends_by_pushing_out_across_the_outer_surface``
   Expects the last call to be ``sub.update_electric_os``.

``test_final_magnetic_sample_is_exact_not_interpolated``
   Expects ``precursors.calc_exact_magnetic_in_time`` to occur *after* the last
   ``precursors.interpolate_magnetic_in_time``.

``test_interpolated_step_count_is_half_the_ratio`` (3 parameter sets)
   Expects 1, 2 and 3 interpolated samples at ratios 3, 5 and 7 —
   ``(ratio - 1) / 2``.

``test_electric_substeps_equal_interpolated_plus_one`` (3 parameter sets)
   Expects 2, 3 and 4 electric sub-steps: each loop iteration plus the trailing
   exact step.

``test_injects_across_the_inner_surface_each_substep``
   At ratio 5, expects 3 calls to ``sub.update_electric_is``.

``test_pml_is_updated_alongside_the_fields``
   Expects both PML updates to appear.

``test_outer_surface_is_pushed_exactly_once``
   Expects exactly one ``sub.update_electric_os``, however many sub-steps run.

``test_does_not_touch_the_magnetic_outer_surface``
   Expects ``sub.update_magnetic_os`` never to be called in this phase.

TestHsgPhaseTwo
^^^^^^^^^^^^^^^

The mirror image.

``test_starts_by_sampling_the_main_magnetic_field``
   Expects ``precursors.update_magnetic`` first.

``test_ends_by_pushing_out_across_the_outer_surface``
   Expects ``sub.update_magnetic_os`` last.

``test_final_electric_sample_is_exact``
   Expects ``precursors.calc_exact_electric_in_time`` to be used.

``test_interpolated_step_count_is_half_the_ratio`` (3 parameter sets)
   Expects ``(ratio - 1) / 2`` electric interpolations.

``test_magnetic_substeps_equal_interpolated_plus_one`` (3 parameter sets)
   Expects 2, 3 and 4 updater-level magnetic updates. Note the prefix matters:
   ``precursors.update_magnetic`` is a different call from the updater's
   ``update_magnetic``.

``test_outer_surface_is_pushed_exactly_once``, ``test_does_not_touch_the_electric_outer_surface``
   The mirrored bounds.

TestSubgridUpdatesFanOut
^^^^^^^^^^^^^^^^^^^^^^^^

``test_phase_one_reaches_every_updater``, ``test_phase_two_reaches_every_updater``
   Expects each phase to be forwarded to every registered updater exactly once.

``test_holds_the_main_grid``
   Expects the main grid reference.

``test_no_updaters_is_a_no_op``
   Expects no exception with an empty updater list.

When these fail
~~~~~~~~~~~~~~~

**A call-count assertion off by exactly one** — check the owner prefix first.
``update_magnetic`` exists on both the updater and the precursors; conflating
them is the single easiest mistake in this file, and it was made once while
writing the suite.

**An interpolated-step count** — the loop bound
``upper_m = int(ratio / 2 - 0.5)`` changed. It must be ``(ratio - 1) / 2``, so
that the subgrid takes ``ratio`` sub-steps in total across both phases.

**A first-or-last call assertion** — the phase's boundaries moved. Sampling the
main grid must happen *before* any sub-step, and the OS push *after* all of
them; reordering either breaks the leapfrog interleaving that makes HSG stable.

**``test_final_*_sample_is_exact``** — the trailing exact sample was replaced by
an interpolated one. The last sub-step lands exactly on the main grid's tick,
so interpolating there introduces error where none is needed.

**``test_electric_sources_advance_the_iteration``** — the subgrid's clock moved
to a different method. Source waveforms are evaluated against this counter, so
advancing it in the wrong place shifts every source in the subgrid by a
sub-step.

Deliberately Untested Paths
---------------------------

Eight source defects were found while writing this suite. **No test asserts
broken behaviour and none is marked** ``xfail`` — where a defect made a
contract untestable, the test was omitted and the defect recorded separately
for the maintainers, with its cause and a suggested fix.

This section exists so the resulting coverage holes are not mistaken for
oversights. If you fix one of these, expect **no test to turn red**; the fix
needs new tests, and the maintainers' write-ups list them.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Location
     - Not covered, and why
   * - ``model.py:116``, ``:124``
     - ``Model.dy`` / ``Model.dz`` getters return ``dl[0]``. The setters are
       correct, so only the setters are pinned; no test reads those two
       getters back.
   * - ``fdtd_grid.py:187-199``
     - ``set_pml_thickness`` with a 1-element sequence raises ``TypeError``
       (the branch calls ``int()`` on the sequence), and lengths 2-5 are
       silently ignored. Only the scalar and 6-element forms are tested.
   * - ``fdtd_grid.py:637-648``
     - ``get_waveform_by_id`` raises ``StopIteration`` for an unknown ID
       rather than a message. Only the happy path is tested.
   * - ``fdtd_grid.py:1051-1071``
     - The undersampling handler catches ``ValueError`` where the guarded
       expression raises ``IndexError``, so it can never fire.
   * - ``fdtd_grid.py:1110-1125``
     - ``delta`` is unbound if the mode is 2D but no axis is one cell, giving
       ``UnboundLocalError``.
   * - ``subgrids/user_objects.py:199-205``
     - ``SubGridHSG.__init__`` overwrites the caller's ``pml_separation``
       unconditionally, so no test passes one and expects it to survive.
   * - ``model.py:178-193``
     - ``_create_grid`` has no terminal ``else``; an unknown solver leaves
       ``grid`` unbound.
   * - ``subgrids/grid.py:32-36``
     - The ratio guard tests parity only, so a negative odd ratio passes
       validation.

Five of the eight share one shape: an ``if`` / ``elif`` chain enumerating the
expected cases with **no terminal** ``else``, so an unexpected value falls
through to an ``UnboundLocalError`` or a silent no-op instead of a message
naming what went wrong. That is worth addressing as a single pattern rather
than as eight separate tickets.

Out of Scope
------------

- **GPU and MPI grids.** ``cuda_grid.py``, ``opencl_grid.py``,
  ``metal_grid.py`` and ``mpi_grid.py`` need real device handles or a live
  communicator, and are planned behind hardware guards in a later PR.
- **PML internals.** ``pml.py``'s coefficient construction is a separate PR;
  here the PML is built for real but only ``_construct_pml``'s box geometry is
  asserted.
- **Physics validation.** The suite asserts that HSG moves the right cells in
  the right direction, never that the stitched field is numerically correct.
- **The solver loop.** ``Model.solve``, ``contexts.py`` and ``solvers.py`` are
  orchestrators with no meaningful unit-level surface.
