Unit Tests — Geometry Fractals
==============================

**Branch:** ``feat/unit-tests-geometry-fractals``

**Modules under test:**
   - ``gprMax/cython/fractals_generate.pyx`` — ``generate_fractal2D`` /
     ``generate_fractal3D``, the spectral filter at the heart of every
     fractal object
   - ``gprMax/fractals/fractal_surface.py`` — ``FractalSurface``, the
     rough height-map applied to a face of a fractal box
   - ``gprMax/fractals/fractal_volume.py`` — ``FractalVolume``, the
     binned material distribution filling a fractal box
   - ``gprMax/fractals/grass.py`` — ``Grass``, blade and root geometry
   - ``gprMax/user_objects/cmds_geometry/fractal_box.py`` —
     ``FractalBox.pre_build()`` / ``.build()``
   - ``gprMax/user_objects/cmds_geometry/add_surface_roughness.py``
   - ``gprMax/user_objects/cmds_geometry/add_surface_water.py``
   - ``gprMax/user_objects/cmds_geometry/add_grass.py``

**Covered transitively:**
   - ``gprMax/user_inputs.py`` ``MainGridUserInput`` — the ``build()``
     tests drive the real discretisation and bounds-checking object,
     not a mock
   - ``gprMax/cython/geometry_primitives.pyx``
     ``build_voxels_from_array`` / ``build_voxels_from_array_mask`` —
     the endpoint of the fractal-box build chain (pinned directly by
     the geometry-primitives suite)
   - ``gprMax/materials.py`` ``ListMaterial``, ``create_water``,
     ``create_grass`` — exercised as the mixing model and the built-in
     materials the surface modifiers instantiate

**Test files:**
   - ``tests/unit/fractals/test_fractals_generate.py`` (19 tests)
   - ``tests/unit/fractals/test_grass.py`` (26 tests)
   - ``tests/unit/fractals/test_fractal_surface.py`` (33 tests)
   - ``tests/unit/fractals/test_fractal_volume.py`` (41 tests)
   - ``tests/unit/fractals/test_fractal_box.py`` (43 tests)
   - ``tests/unit/fractals/test_surface_modifiers.py`` (90 tests)

**Shared fixtures:** ``tests/unit/fractals/conftest.py``

Scope
-----

Fractals are how gprMax models ground that is neither flat nor
homogeneous: a ``#fractal_box`` fills a region with a spatially
correlated mixture of materials, ``#add_surface_roughness`` replaces one
of its faces with a random height-map, ``#add_surface_water`` floods the
dips in that height-map, and ``#add_grass`` plants blades on it.

Every one of those is built from the same four-step recipe, in 2D for a
surface and 3D for a volume:

1. fill an array with white noise from a seeded generator;
2. FFT it into the frequency domain;
3. divide each coefficient by ``distance_from_centre ** dimension`` —
   this suppresses the fine detail and leaves the broad features, and is
   the entire fractal (``generate_fractal2D`` / ``generate_fractal3D``);
4. inverse-FFT back, then rescale linearly so the result spans exactly
   the range the user asked for.

The output is random, but the contract around it is not, and the
contract is what these tests pin:

.. list-table::
   :header-rows: 1

   * - Property
     - What is asserted
   * - Reproducibility
     - The same seed regenerates a byte-identical array. The RNG is
       constructed fresh inside each generate method, so this holds
       across instances and across repeated calls.
   * - Seed sensitivity
     - A different seed produces a different array — the seed is
       actually plumbed through.
   * - Output range
     - The final rescale is arithmetic, not statistics: the minimum and
       maximum of a generated surface land exactly on the requested
       ``fractalrange``, for any seed.
   * - Shape and axis mapping
     - A surface on the ``x = const`` plane has shape ``(ny, nz)``, and
       so on for the other two orientations.
   * - Bin count
     - A volume generated with ``nbins = N`` holds only whole numbers in
       ``[0, N-1]`` — these are indices into the mixing model's material
       table.
   * - Dimension knob
     - Raising the fractal dimension yields a measurably smoother
       surface (asserted directionally, not against a target number).
   * - Validation
     - Every ``ValueError`` / ``KeyError`` branch of the four user
       objects, which is most of the code by line count and fully
       deterministic.

The Cython kernel is tested against the same formula re-evaluated in
numpy, including its zero-norm fallback and the offset/global-size
parameters that let an MPI rank compute a sub-block identically to the
serial path.

Out of scope: the MPI subclasses (``MPIFractalSurface``,
``MPIFractalVolume``, ``fractals/mpi_utilities.py``), which need a live
communicator and the optional ``mpi4py_fft`` package; statistical
validation of "fractalness" (whether the spectrum follows the intended
power law is a physics question, not a unit-test one); and golden-file
regression on generated arrays, which would pin numpy's and scipy's
RNG/FFT internals rather than gprMax's.

The Two-Pass Build
------------------

``FractalBox`` is the only user object whose ``build()`` means two
different things depending on how many times it has been called, and the
tests are shaped around that.

A fractal box cannot stamp itself into the grid when it is parsed,
because a later ``#add_surface_roughness`` or ``#add_grass`` may still
grow it. ``Scene.process_geometry_objects`` (``scene.py:140-153``)
therefore runs the fractal family twice:

**Pass 1** — ``FractalBox.build()`` sees ``do_pre_build == True`` and
delegates to ``pre_build()``: it validates the parameters, resolves the
mixing model (creating its N materials on the grid), constructs a
``FractalVolume`` and registers it on ``grid.fractalvolumes``. Nothing is
written to the grid arrays. The three surface modifiers then run, each
finding its box *by string ID* in that list and attaching a generated
``FractalSurface`` to it.

**Pass 2** — ``FractalBox.build()`` takes the other branch: it extends
the volume's bounds to cover every attached surface's range, generates
the fractal volume, builds the 3D mask, carves the rough surfaces, water
and grass into that mask, and finally stamps the result into ``solid`` /
``rigidE`` / ``rigidH`` / ``ID`` through the Cython array builders.

The mask is where a rough surface actually becomes geometry. It carries
four values, and the material each one selects:

.. list-table::
   :header-rows: 1

   * - Mask
     - Meaning
     - Material stamped
   * - ``0``
     - outside the box (above the rough surface)
     - none — cell skipped
   * - ``1``
     - box interior
     - ``fractalvolume[i, j, k]`` — the binned soil material
   * - ``2``
     - surface water
     - ``waternumID``
   * - ``3``
     - grass blade or root
     - ``grassnumID``

Test Infrastructure
-------------------

``tests/unit/fractals/conftest.py``:

``fractal_config`` (autouse)
   Patches ``gprMax.config`` to a predictable environment: double
   precision (``sim_config.dtypes["float_or_double"]``), a single OpenMP
   thread (``get_model_config().ompthreads``, passed straight through to
   the Cython kernel), and a ``materials`` dict that ``create_water`` /
   ``create_grass`` can bump ``maxpoles`` on. These two values are
   everything the fractal modules read from global config.

``grid_arrays``
   Factory fixture returning the four grid arrays at production shapes
   and dtypes, zero-initialised (default 16 × 16 × 16 cells).

``fractal_grid``
   Factory fixture for the ``build()`` tests: the four arrays plus the
   surface the fractal dispatch layer reads — ``dx`` / ``dy`` / ``dz``,
   ``dl``, ``size``, a time step small enough to resolve the Debye
   relaxation times of water and grass, a materials list, and empty
   ``fractalvolumes`` / ``mixingmodels`` registries. It implements the
   ``FDTDGrid`` ``within_bounds`` contract (raises ``ValueError``
   carrying the axis letter) and the two fractal factory methods
   (``add_fractal_volume``, ``create_fractal_surface``), which on the
   real grid are plain constructors. Because the stub is not a subgrid or
   MPI grid, ``_create_uip`` dispatches to the real
   ``MainGridUserInput`` with no config patching.

``add_mixing_model(grid, ...)``
   Registers a ``ListMaterial`` mixing model — the simplest of the three,
   since it collects existing materials by ID rather than synthesising
   new ones, so ``calculate_properties`` just populates ``matID``.

``nonzero_set(arr)``
   Set of index tuples at which an array is nonzero; every "which cells
   were written" assertion compares one of these against an expected set.

``DL``
   Module constant ``0.001`` (1 mm), the uniform discretisation, so cell
   index ``i`` maps to coordinate ``i * DL``.

Test Catalog — ``test_fractals_generate.py``
--------------------------------------------

The Cython spectral filter. Pure arithmetic over numpy arrays — no grid,
no config, no randomness. Each test hands the kernel a hand-built array
and compares against the documented formula re-evaluated in numpy:

.. code-block:: text

   v2  = weighting * ((index + offset + global_size // 2) % global_size)
   rr  = ||v2 - v1||
   B   = rr ** D              (B = 0.9 if B == 0)
   out = A / B

Source: ``fractals_generate.pyx:27-132``.

TestGenerateFractal2D
^^^^^^^^^^^^^^^^^^^^^

``test_matches_the_reference_formula``
   Random complex 8 × 8 input against the numpy reference.

``test_zero_dimension_is_the_identity``
   ``rr ** 0 == 1`` for every cell, so the array passes through
   untouched — a clean check that nothing else is applied.

``test_centre_cell_uses_the_zero_norm_fallback``
   The FFT shift (``sx = gx // 2``) maps cell ``n // 2`` to position
   zero; placing ``v1`` at the origin makes ``rr == 0`` there, and the
   kernel substitutes ``B = 0.9`` rather than dividing by zero.

``test_only_the_centre_cell_uses_the_fallback``
   Exactly one cell takes that branch.

``test_higher_dimension_suppresses_high_frequencies_harder``
   The cell furthest from the spectral centre shrinks as ``D`` rises.

``test_weighting_stretches_the_distance_metric_per_axis``
   Doubling the x-weighting suppresses a cell offset along x harder,
   while one offset along y is unchanged.

``test_offsets_select_a_sub_block_of_the_full_domain``
   The ``ox`` / ``oy`` / ``gx`` / ``gy`` parameters exist so an MPI rank
   holding a sub-block computes the same values the serial code would: a
   sub-block computed with an offset equals the corresponding slice of
   the full-domain result.

``test_non_square_arrays``, ``test_output_dtype_is_preserved``,
``test_input_array_is_not_mutated``
   Shape independence, ``complex128`` output, and read-only input.

TestGenerateFractal3D
^^^^^^^^^^^^^^^^^^^^^

The same battery one dimension up (9 tests): reference formula,
zero-dimension identity, the zero-norm fallback and its uniqueness,
dimension monotonicity, MPI sub-block equivalence, non-cubic arrays,
per-axis weighting, and input immutability.

Test Catalog — ``test_grass.py``
--------------------------------

``Grass`` holds the random geometry of a clump of blades in a
``(numblades, 6)`` table: columns 0-1 are the curvature scale in x and y
(``10 + 20 * U(0,1)``), columns 2-3 the curvature direction (±1), and
columns 4-5 the running position of the root, which starts at zero and is
walked one random step per call. Source: ``grass.py:28-105``.

TestGeometryParameters
^^^^^^^^^^^^^^^^^^^^^^

``test_table_shape_and_dtype``
   ``(numblades, 6)`` in the model's float type.

``test_numblades_and_seed_are_stored``, ``test_zero_blades_gives_an_empty_table``,
``test_six_generators_are_created``
   Constructor bookkeeping.

``test_curvature_scales_lie_in_the_documented_range``
   Columns 0-1 fall in ``[10, 30)`` over 50 blades.

``test_direction_columns_are_plus_or_minus_one``
   Columns 2-3 draw only from ``{-1, +1}``.

``test_root_accumulators_start_at_zero``
   Columns 4-5 are zero until the root walk runs.

``test_same_seed_gives_the_same_table``,
``test_different_seed_gives_a_different_table``
   Reproducibility and seed sensitivity of the whole table.

TestBladeGeometry
^^^^^^^^^^^^^^^^^

``test_a_blade_starts_directly_above_its_root``
   Height zero gives offset ``(0, 0)``.

``test_matches_the_quadratic_formula``,
``test_matches_the_formula_at_every_height`` (5 parameter sets)
   The displacement is ``direction * (height / scale) ** 2``, rounded to
   a cell — checked against the table entries at a range of heights.

``test_displacement_grows_with_height``
   The blade curves progressively away from vertical.

``test_offset_sign_follows_the_direction_column``
   Over 30 blades, the sign of the offset is the direction column's.

``test_returns_integer_cell_offsets``
   The result is a cell offset, not a float.

``test_is_a_pure_function_of_blade_and_height``
   Asking twice gives the same answer and leaves the table untouched —
   the contrast with the root walk below.

TestRootGeometry
^^^^^^^^^^^^^^^^

``test_root_walk_advances_the_accumulator``
   The first call moves the root off zero.

``test_each_step_moves_the_accumulator_by_at_most_one_cell``
   The step is drawn from ``U(-1, 1)``, checked over ten steps.

``test_returns_the_rounded_accumulator``
   The return value is the accumulator rounded to a cell.

``test_successive_calls_are_a_random_walk_not_a_pure_function``
   Roots wander: the same arguments give a different position each call,
   because the accumulator has moved on.

``test_roots_are_independent_per_blade``
   Walking one root leaves the others at zero.

``test_walk_is_reproducible_across_instances``
   Two identically seeded ``Grass`` objects produce the same six-step
   walk.

Test Catalog — ``test_fractal_surface.py``
------------------------------------------

``FractalSurface`` is the rough height-map applied to one face of a
fractal box. Source: ``fractal_surface.py:38-200``.

TestConstruction
^^^^^^^^^^^^^^^^

``test_start_and_stop_are_stored_as_int32``,
``test_dimension_and_seed_are_stored``, ``test_defaults``,
``test_surface_ids_are_the_six_faces``
   Bounds stored as ``int32`` arrays; fresh instances carry
   ``weighting == [1, 1]``, ``fractalrange == (0, 0)``, ``filldepth == 0``,
   no grass, ``complex128`` working dtype; the six legal face names.

TestCoordinateProperties
^^^^^^^^^^^^^^^^^^^^^^^^

``test_getters_read_through_to_start_and_stop``,
``test_setters_write_through`` (6 parameter sets),
``test_size_and_extents``
   ``xs`` … ``zf`` are views onto the ``start`` / ``stop`` arrays in both
   directions; ``size`` / ``nx`` / ``ny`` / ``nz`` derive from them.

TestGetSurfaceDims
^^^^^^^^^^^^^^^^^^

``test_x_plane_surface_spans_y_and_z`` (and the ``y`` / ``z`` cases)
   The axis mapping everything downstream depends on: a surface on a
   constant-``x`` plane is dimensioned ``(ny, nz)``.

TestGenerateFractalSurface
^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_shape_follows_the_plane`` (3 parameter sets)
   Generated shape matches ``get_surface_dims()`` for all three
   orientations.

``test_output_dtype_comes_from_config``
   Computed in ``complex128``, delivered in the model's float type.

``test_output_spans_the_requested_range_exactly``,
``test_range_is_exact_for_any_limits`` (3 parameter sets),
``test_every_value_lies_within_the_range``
   The rescale is exact arithmetic: minimum and maximum land on
   ``fractalrange`` and nothing escapes it.

``test_the_surface_is_not_flat``
   The height-map actually varies.

``test_same_seed_reproduces_the_same_surface``,
``test_different_seed_gives_a_different_surface``,
``test_generating_twice_gives_the_same_surface``
   The reproducibility contract: identical arrays from identical seeds,
   across instances and across repeated calls on one object (the RNG is
   rebuilt inside the method, so no shared stream advances).

``test_higher_dimension_gives_a_smoother_surface``
   Mean cell-to-cell variation falls as the dimension rises, on a
   32 × 32 surface with the seed held fixed.

``test_weighting_is_not_mutated_by_generation``,
``test_weighting_changes_the_surface``
   Anisotropic weighting changes the result and survives the call.

``test_a_single_cell_wide_surface_still_generates``
   A 1 × 8 surface generates and still spans its range exactly.

``test_returns_true``
   The serial generator always reports success (the MPI subclass is the
   one that can decline).

Test Catalog — ``test_fractal_volume.py``
-----------------------------------------

``FractalVolume`` is the interior of a fractal box: a 3D array of bin
indices into the mixing model's material table. Source:
``fractal_volume.py:39-280``.

TestConstruction
^^^^^^^^^^^^^^^^

``test_start_and_stop_are_stored_as_int32``, ``test_defaults``,
``test_dimension_and_seed_are_stored``
   As for the surface, plus ``averaging == False``, ``nbins == 0``, no
   mixing model and no attached surfaces.

``test_original_bounds_snapshot_the_constructor_arguments``,
``test_extending_the_volume_leaves_the_originals_alone``
   The ``original_*`` pair remembers the user's box after a rough surface
   has grown the working bounds — the distinction the volume mask is
   built from.

TestCoordinateProperties
^^^^^^^^^^^^^^^^^^^^^^^^

``test_getters_read_through``, ``test_setters_write_through``
(12 parameter sets), ``test_size_and_extents``
   Both the working bounds and the ``original*`` bounds read and write
   through to their backing arrays.

TestFilterScaling
^^^^^^^^^^^^^^^^^

Before generating, the volume scales its ``weighting`` by
``min(dims) / dims`` so the spectral filter is isotropic in physical
terms rather than in cells.

``test_a_cubic_volume_keeps_its_weighting``
   A cube needs no correction.

``test_an_elongated_volume_is_scaled_by_the_shortest_axis``
   An 8 × 4 × 4 volume ends up weighted ``[0.5, 1, 1]``.

``test_flat_volumes_hold_the_flat_axis_at_one`` (3 parameter sets)
   The three single-cell-thick branches, which insert a ``1`` at the flat
   axis and scale only the other two.

TestGenerateFractalVolume
^^^^^^^^^^^^^^^^^^^^^^^^^

``test_shape_matches_the_volume_extents``,
``test_output_dtype_comes_from_config``,
``test_a_single_cell_thick_volume_still_generates``, ``test_returns_true``
   Shape, dtype and degenerate-extent handling.

``test_values_are_bin_indices_in_range`` (3 parameter sets),
``test_values_are_whole_numbers``, ``test_every_bin_is_populated``
   With ``nbins = N`` the output holds only whole numbers in
   ``[0, N-1]`` — anything outside would index past the end of the
   mixing model's material table — and a reasonably sized volume fills
   every bin.

``test_same_seed_reproduces_the_same_volume``,
``test_different_seed_gives_a_different_volume``,
``test_different_dimension_gives_a_different_volume``
   Reproducibility and the sensitivity of the result to both knobs.

TestGenerateVolumeMask
^^^^^^^^^^^^^^^^^^^^^^

``test_mask_shape_and_dtype``
   ``int8``, sized to the working volume.

``test_an_unextended_volume_is_masked_solid``
   With no surface attached the whole box is interior.

``test_only_the_original_footprint_is_masked``,
``test_the_footprint_is_offset_into_the_extended_volume``
   When a rough surface has grown the volume, the mask is ``1`` exactly
   over the user's original box and ``0`` across the extension —
   including when the growth is in the minus direction, where the
   footprint is offset into the array.

``test_mask_is_regenerated_from_scratch_each_call``
   The mask is rebuilt, not patched.

Test Catalog — ``test_fractal_box.py``
--------------------------------------

Both passes of ``FractalBox.build()``, driven end-to-end against a stub
grid carrying real numpy arrays. The box spans cells 2..10 in every axis
on a 16-cell grid and draws from a four-material mixing model. Source:
``fractal_box.py:35-709``.

TestPreBuildRegistersTheVolume
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_a_fractal_volume_is_registered_on_the_grid``
   The volume appears on ``grid.fractalvolumes`` and is findable by ID —
   the whole contract of pass 1, since that lookup is the only link
   between a box and its surface modifiers.

``test_volume_bounds_come_from_the_discretised_points``,
``test_volume_attributes_are_populated``
   Bounds, ID, ``operatingonID``, ``nbins``, dimension, seed, weighting
   and mixing model are all transferred to the volume.

``test_nothing_is_written_to_the_grid_on_the_first_pass``
   ``solid`` / ``rigidE`` / ``ID`` remain untouched.

``test_the_mixing_model_materials_are_resolved``
   ``calculate_properties`` runs, populating ``matID`` with the numIDs
   the fractal volume will index.

``test_pre_build_runs_only_once``
   ``do_pre_build`` flips after the first call and no second volume is
   registered.

``test_a_normal_material_can_be_used_instead_of_a_mixing_model``,
``test_missing_seed_leaves_the_volume_unseeded``
   The single-material path, and the unseeded (non-reproducible) path.

TestAveraging
^^^^^^^^^^^^^

``test_averaging_is_off_by_default``
   Unlike every other geometry object, a fractal box does *not* inherit
   the grid's averaging default — dielectric smoothing is off unless
   asked for.

``test_averaging_can_be_switched_on``,
``test_averaging_can_be_switched_off_explicitly``
   The ``averaging`` kwarg overrides in both directions.

TestPreBuildValidation
^^^^^^^^^^^^^^^^^^^^^^

``test_missing_parameters_raise`` (7 parameter sets)
   Each required kwarg, individually omitted, raises ``KeyError``.

``test_negative_fractal_dimension_raises``,
``test_negative_weighting_raises`` (3 parameter sets),
``test_negative_material_count_raises``
   The numeric guards, including each weighting component separately.

``test_unknown_mixing_model_raises``
   Neither a mixing model nor a material with that ID exists.

``test_a_mixing_model_with_one_bin_raises``
   A mixing model needs more than one material to distribute.

``test_more_bins_than_materials_in_the_list_raises``
   A ``ListMaterial`` cannot supply more bins than it holds materials.

``test_out_of_bounds_points_raise``, ``test_inverted_points_raise``
   Bounds and ordering, enforced by the real ``MainGridUserInput``.

TestBuildWithoutSurfaces
^^^^^^^^^^^^^^^^^^^^^^^^

``test_a_single_material_box_with_no_surfaces_raises``
   One material and no roughness is just a ``#box``, and the code says so.

``test_the_box_footprint_is_stamped_into_solid``
   The exact set of written cells is the box's cell range and nothing
   else.

``test_only_mixing_model_materials_are_stamped``,
``test_all_four_soil_materials_appear``
   Every stamped value is one of the mixing model's numIDs, and all four
   bins reach the grid.

``test_the_stamped_materials_match_the_generated_volume``
   The ``solid`` array equals the generated ``fractalvolume`` over the
   box footprint — the binned distribution arrives intact.

``test_averaging_off_marks_the_cells_rigid``,
``test_averaging_on_clears_the_rigid_arrays``
   The hard path sets the rigid flags and stamps ``ID``; the smoothed
   path leaves both rigid arrays empty.

``test_the_result_is_reproducible``
   Two independent builds with the same seed produce identical grids.

TestBuildWithASurface
^^^^^^^^^^^^^^^^^^^^^

A rough surface on the box's z+ face, with heights ranging over cells
9..13.

``test_the_volume_is_extended_to_cover_the_surface``
   Pass 2 grows the volume's ``zf`` from 10 to 13 while ``originalzf``
   stays at 10 — the growth the mask later distinguishes.

``test_the_solid_footprint_is_limited_to_the_box_in_x_and_y``
   The roughness extends the box in z only.

``test_cells_below_the_roughness_range_are_solid_soil``
   Everything under the lowest possible surface height is box interior,
   whatever the height-map did.

``test_the_surface_is_rough_not_flat``
   Within the roughness band, different columns are filled to different
   heights — the height-map has reached the grid.

``test_nothing_is_written_above_the_roughness_range``
   The upper limit is respected.

``test_a_single_material_box_with_a_surface_uses_that_material``
   With ``nbins == 1`` the fractal generation is skipped entirely and the
   volume is filled with one material's numID.

``test_the_result_is_reproducible``
   The whole box-plus-roughness chain reproduces from the seeds.

Test Catalog — ``test_surface_modifiers.py``
--------------------------------------------

The three commands that modify a fractal box's surface. All run in pass
1, after the box has registered its volume and before it stamps itself:
they locate the volume by ID, attach a ``FractalSurface``, and touch no
grid arrays. Tests drive ``build()`` through the real
``MainGridUserInput``. Sources: ``add_surface_roughness.py:68-230``,
``add_surface_water.py:63-177``, ``add_grass.py:69-256``.

TestAddSurfaceRoughnessFaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each of the six faces is parametrised with its plane, roughness limits
and expected fractal range.

``test_the_requested_face_is_identified`` (6 parameter sets)
   A plane is matched against the volume's own bounds to yield
   ``xminus`` / ``xplus`` / ``yminus`` / ``yplus`` / ``zminus`` /
   ``zplus``.

``test_the_fractal_range_spans_the_limits`` (6 parameter sets)
   The range the surface may vary over, in cells, on each face.

``test_the_height_map_spans_the_other_two_axes`` (6 parameter sets)
   The generated map is dimensioned by the two axes the face lies in.

``test_the_height_map_stays_inside_the_fractal_range`` (6 parameter sets)
   Its minimum and maximum land exactly on the requested range.

TestAddSurfaceRoughnessAttachment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_the_surface_is_attached_to_the_volume``,
``test_surface_attributes_are_populated``,
``test_surface_bounds_are_the_discretised_plane``
   The surface reaches ``volume.fractalsurfaces`` carrying its dimension,
   seed, weighting, ``operatingonID`` and discretised bounds.

``test_no_grid_arrays_are_touched``
   Roughness is bookkeeping; the geometry arrives in pass 2.

``test_two_different_faces_can_be_roughened``
   Two faces attach independently, in order.

``test_missing_seed_leaves_the_surface_unseeded``,
``test_the_height_map_is_reproducible``
   Seed plumbing, and identical height-maps from two independent runs.

TestAddSurfaceRoughnessValidation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_missing_parameters_raise`` (6 parameter sets)
   Each required kwarg raises ``KeyError``.

``test_unknown_fractal_box_raises``
   The ID lookup is the only link to the box; a typo is fatal.

``test_negative_fractal_dimension_raises``,
``test_negative_weighting_raises`` (2 parameter sets)
   Numeric guards.

``test_a_volume_rather_than_a_plane_raises``,
``test_a_line_rather_than_a_plane_raises``
   Exactly one coordinate pair must match — zero (a volume) or two (a
   line) are rejected.

``test_an_internal_plane_raises``
   A plane through the middle of the box is not one of its faces.

``test_roughness_below_the_box_raises``,
``test_roughness_above_the_box_raises``
   Roughness may not extend past the box's own bounds, in either
   direction.

``test_roughness_outside_the_model_domain_raises``
   Nor past the edge of the model.

``test_roughening_the_same_face_twice_raises``
   One height-map per face.

TestAddSurfaceWater
^^^^^^^^^^^^^^^^^^^

``test_the_fill_depth_is_recorded_on_the_surface``
   The water level, discretised to cells, lands on ``surface.filldepth``
   — the value the mask later reads to decide which dips flood.

``test_water_is_created_as_a_material``,
``test_water_is_only_created_once``,
``test_a_debye_pole_is_registered_for_water``
   The built-in single-pole Debye water material is created on demand,
   once, and bumps the model's ``maxpoles``.

``test_no_grid_arrays_are_touched``
   As with roughness.

``test_missing_parameters_raise`` (4 parameter sets),
``test_unknown_fractal_box_raises``,
``test_non_positive_depth_raises`` (2 parameter sets)
   Parameter guards.

``test_water_on_a_face_with_no_roughness_raises``
   Water needs a rough surface to sit in — a flat face has no dips.

``test_a_fill_depth_outside_the_roughness_range_raises``
   The level must fall within the range the surface actually varies over.

``test_a_volume_rather_than_a_plane_raises``,
``test_an_internal_plane_raises``
   Orientation guards, as for roughness.

``test_too_large_a_time_step_for_water_raises``
   The model's time step must resolve water's Debye relaxation time.

TestAddGrassBuild
^^^^^^^^^^^^^^^^^

The success path: the surface is marked ``grass``, a ``Grass`` object
carrying the blade count is attached, the fractal range spans the
requested blade heights, and the height-map becomes a sparse field of
discrete blade heights (zero where there is no blade). The built-in grass
material is created on demand and registers a Debye pole; no grid arrays
are touched; the result is reproducible from the seed.

These nine tests are currently marked ``xfail``. ``AddGrass.build()``
cannot complete under NumPy ≥ 2: at ``add_grass.py:227`` it assigns
``R.randint(..., size=1)`` — a one-element array — into a scalar element
of the height-map, which NumPy 2 rejects (it was a ``DeprecationWarning``
in NumPy 1.25). Every valid input reaches that line. The tests describe
the intended contract and will pass unchanged once the assignment is made
scalar.

TestAddGrassValidation
^^^^^^^^^^^^^^^^^^^^^^

All of these raise before reaching that line and run normally.

``test_missing_parameters_raise`` (6 parameter sets),
``test_unknown_fractal_box_raises``,
``test_negative_fractal_dimension_raises``,
``test_negative_blade_heights_raise`` (2 parameter sets)
   Parameter guards.

``test_grass_on_a_negative_facing_surface_raises`` (3 parameter sets)
   Grass only grows on the positive-facing faces (``xplus`` / ``yplus`` /
   ``zplus``); the three minus faces are rejected.

``test_an_internal_plane_raises``,
``test_a_volume_rather_than_a_plane_raises``
   Orientation guards.

``test_more_blades_than_surface_cells_raises``
   An 8 × 8 face has room for 64 blades, and no more.
