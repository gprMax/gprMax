Unit Tests — User Objects
=========================

**Branch:** ``feat/unit-tests-user-objects``

**Modules under test:**
   - ``gprMax/user_objects/user_objects.py`` (abstract bases)
   - ``gprMax/user_objects/rotatable.py`` (rotation mixin)
   - ``gprMax/user_objects/cmds_singleuse.py`` (12 once-per-model objects)
   - ``gprMax/user_objects/cmds_multiuse.py`` (19 repeatable objects)
   - ``gprMax/user_objects/cmds_output.py`` (3 output objects)
   - ``gprMax/user_objects/cmds_geometry/*.py`` (11 primitives + 3 modifiers)

**Test files:**
   - ``tests/unit/user_objects/test_user_objects_base.py`` (25 tests)
   - ``tests/unit/user_objects/test_cmds_singleuse.py`` (75 tests)
   - ``tests/unit/user_objects/test_cmds_multiuse.py`` (72 tests)
   - ``tests/unit/user_objects/test_cmds_output.py`` (24 tests)
   - ``tests/unit/user_objects/test_cmds_geometry.py`` (69 tests)

**Shared fixtures:** ``tests/unit/user_objects/conftest.py``

Scope
-----

Verifies the user-object class hierarchy that the hash parser and the
Python API both feed: the five abstract base classes
(``UserObject``, ``ModelUserObject``, ``GridUserObject``,
``OutputUserObject``, ``GeometryUserObject``), the ``RotatableMixin``,
and every concrete user-object class under ``gprMax/user_objects/``.

For each concrete class we exercise the constructor → attribute /
``kwargs`` mirroring contract, the ``order`` and ``hash`` properties,
the ``__str__()`` round-trip used by ``params_str`` and downstream
logging, and the validation branches in ``build()``.

Tests do not drive the FDTD solver, the Cython geometry primitives
(``build_box``, ``build_sphere``, etc.), or any disk I/O. The
``build()`` chain is short-circuited one frame above the Cython call
by stubbing ``_create_uip``, ``model.add_*`` factories, and the small
set of grid attributes each ``build()`` reads — letting us cover the
Python-side validation logic without standing up a real ``FDTDGrid``.

Test Infrastructure
-------------------

``tests/unit/user_objects/conftest.py`` defines one autouse fixture
and two stub builders:

``user_object_config`` (autouse)
   Monkeypatches ``gprMax.config`` for the duration of each test:

   - ``config.sim_config.general`` → ``{"solver": "cpu", "precision":
     "double", "subgrid": False}``
   - ``config.sim_config.dtypes["float_or_double"]`` → ``np.float64``
   - ``config.sim_config.em_consts`` → ``{c, e0, m0, z0}`` from
     ``scipy.constants``
   - ``config.sim_config.args.autotranslate`` → ``False``
   - ``config.sim_config.input_file_path`` → a tmp_path sentinel
   - ``config.get_model_config()`` → a ``SimpleNamespace`` with
     ``mode``, ``ompthreads``, ``materials={"maxpoles": 0}`` and a
     ``set_output_file_path`` ``MagicMock``

   Returns a ``SimpleNamespace`` carrying ``sim_config`` and
   ``model_config`` for tests that need to assert side effects on the
   model config or override solver / autotranslate per test.

``stub_grid``
   Minimal ``SimpleNamespace`` standing in for ``FDTDGrid``. Carries:

   - ``dx = dy = dz = 0.001``, ``dl = [0.001]*3``
   - ``dt = 1.927e-12``, ``timewindow = 1e-9``, ``iterations = 100``
   - ``nx = ny = nz = 50``, ``size = [50]*3``
   - ``averagevolumeobjects = True``
   - ``materials`` pre-populated with ``pec`` and ``free_space``
   - empty ``waveforms``, ``mixingmodels``, ``discreteplanewaves``
   - ``pmls = {"formulation": "HORIPML", "thickness": {…}, "cfs": []}``
   - ``MagicMock``-backed ``add_source``, ``add_receiver``,
     ``set_pml_thickness``, ``calculate_dt``, ``within_bounds``

``stub_model``
   ``SimpleNamespace`` standing in for ``Model``. Mirrors ``stub_grid``
   for spatial / temporal attributes plus ``MagicMock``-backed
   ``set_size``, ``add_snapshot``, ``add_geometry_view_voxels``,
   ``add_geometry_view_lines``, ``add_geometry_object``. ``model.G``
   points at the ``stub_grid``.

Test helpers ``make_material`` and ``make_waveform`` build
``SimpleNamespace`` stand-ins with just the attributes the
user-object ``build()`` methods read (``ID``, ``numID``,
``averagable``, ``er``, ``se``, ``mr``, ``sm`` for materials; ``ID``
for waveforms).

Test Catalog — ``user_objects.py`` + ``rotatable.py``
-----------------------------------------------------

TestUserObjectABCEnforcement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_cannot_instantiate_userobject_directly``
   Asserts ``UserObject()`` raises ``TypeError`` — the abstract ``order``
   and ``hash`` properties block instantiation. Source:
   ``user_objects.py:31-56``.

``test_subclass_missing_abstracts_cannot_instantiate``
   Asserts a bare ``UserObject`` subclass (no ``order``/``hash``) also
   raises ``TypeError``. Documents that subclassing alone does not
   satisfy the contract.

TestUserObjectKwargsAndDefaults
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_init_stores_kwargs_dict_verbatim``
   Asserts ``UserObject(a=1, b="two", c=(0.1, 0.2)).kwargs`` is exactly
   ``{"a": 1, "b": "two", "c": (0.1, 0.2)}``. Source:
   ``user_objects.py:58-60``. The kwargs dict is the parser-side
   contract — every concrete class round-trips through it.

``test_autotranslate_defaults_to_true``
   Asserts a fresh instance has ``autotranslate == True``. Source:
   ``user_objects.py:60``. Per-object overrides defeat the global
   autotranslate flag in ``_create_uip``.

``test_no_kwargs_yields_empty_kwargs_dict``
   Asserts ``UserObject().kwargs == {}``. Sanity check that the dict
   is freshly built per instance (not a class-level mutable default).

TestUserObjectOrderingAndPrinting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_lt_sorts_by_order``
   With ``Low.order == 1`` and ``High.order == 9``, asserts
   ``Low() < High()``, ``not High() < Low()``, and ``sorted([High(),
   Low()])`` yields the ascending-``order`` list. Source:
   ``user_objects.py:62-63``. Failure means the scene build sequence
   has drifted.

``test_str_joins_scalar_kwargs_after_hash``
   Asserts ``str(obj)`` with ``a=1, b="two", c=3.5`` is
   ``"#fake: 1 two 3.5"``. Source: ``user_objects.py:65-75``. This is
   the round-trip used by every concrete class's logging output.

``test_str_expands_tuple_and_list_kwargs``
   Asserts a tuple ``p1=(0.1, 0.2, 0.3)`` is expanded into three
   space-separated tokens in the ``__str__`` output. Verifies the
   inner ``if isinstance(value, (tuple, list))`` branch at
   ``user_objects.py:69-71``.

``test_str_skips_none_valued_kwargs``
   Asserts a ``None``-valued kwarg is omitted from the ``__str__``
   output. Verifies the ``elif value is not None`` guard at
   ``user_objects.py:72-73``.

``test_params_str_returns_hash_and_kwargs_repr``
   Asserts ``params_str()`` returns ``"#fake: "`` followed by the
   ``str(kwargs)`` dict-repr. Source: ``user_objects.py:77-79``. Used
   by ``logger.exception(self.params_str())`` throughout the package.

TestCreateUIPDispatch
^^^^^^^^^^^^^^^^^^^^^

Verifies the three-way dispatch at ``user_objects.py:81-103``: pick
``SubgridUserInput`` for sub-grids when both global and per-object
autotranslate are on, ``MPIUserInput`` for MPI grids, and
``MainGridUserInput`` otherwise. Each test patches the corresponding
constructor in ``user_objects`` and asserts it was called with the
grid.

``test_main_grid_returns_main_grid_user_input``
   With a plain ``MagicMock`` grid (not spec'd to any specific grid
   class), asserts ``MainGridUserInput(grid)`` is constructed.

``test_mpi_grid_returns_mpi_user_input``
   With ``MagicMock(spec=MPIGrid)``, asserts ``MPIUserInput(grid)``
   is constructed.

``test_subgrid_with_autotranslate_returns_subgrid_user_input``
   With global ``autotranslate=True``, per-object
   ``autotranslate=True``, and ``MagicMock(spec=SubGridBaseGrid)``,
   asserts ``SubgridUserInput(grid)`` is constructed.

``test_subgrid_without_global_autotranslate_falls_through_to_main``
   With global ``autotranslate=False`` and a subgrid, asserts the
   dispatch falls through to ``MainGridUserInput`` — the global flag
   trumps subgrid detection.

``test_subgrid_with_object_autotranslate_disabled_falls_through``
   With global ``autotranslate=True`` but per-object
   ``autotranslate=False``, asserts the dispatch falls through to
   ``MainGridUserInput`` — per-object overrides the global flag.

TestSubclassAbstractEnforcement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three tests confirming the three concrete-grid-typed base classes
each require ``build`` to be implemented before instantiation:

``test_modeluserobject_without_build_cannot_instantiate``
   ``ModelUserObject`` with concrete ``order``/``hash`` but no
   ``build(model)`` raises ``TypeError``. Source:
   ``user_objects.py:106-116``.

``test_griduserobject_without_build_cannot_instantiate``
   ``GridUserObject`` with concrete ``order``/``hash`` but no
   ``build(grid)`` raises ``TypeError``. Source:
   ``user_objects.py:119-140``.

``test_outputuserobject_without_build_cannot_instantiate``
   ``OutputUserObject`` with concrete ``order``/``hash`` but no
   ``build(model, grid)`` raises ``TypeError``. Source:
   ``user_objects.py:143-164``.

TestGridNameHelper
^^^^^^^^^^^^^^^^^^

``test_grid_name_empty_for_main_grid``
   Asserts ``grid_name(SimpleNamespace())`` returns ``""``. Source:
   ``user_objects.py:126-140``. The logging prefix is empty for the
   main grid.

``test_grid_name_brackets_subgrid_name``
   With ``MagicMock(spec=SubGridBaseGrid)`` and ``grid.name = "sub1"``,
   asserts ``grid_name(grid) == "[sub1] "``. Used by every concrete
   ``GridUserObject.build()`` to prefix subgrid log lines.

TestGeometryUserObject
^^^^^^^^^^^^^^^^^^^^^^

``test_geometry_order_is_one``
   Asserts ``GeometryUserObject.order == 1``. Source:
   ``user_objects.py:167-176``. Geometry objects build in arrival
   order (the rendering order matters because later shapes overwrite
   earlier ones in the same cell), so they all share ``order = 1``.

TestRotatableMixinDefaults
^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_defaults``
   Asserts a fresh rotatable object has ``axis="x"``, ``angle=0``,
   ``origin=None``, ``do_rotate=False``. Source: ``rotatable.py:40-45``.
   The ``do_rotate`` flag stays False until ``rotate()`` is called —
   ``build()`` reads it to decide whether to invoke ``_do_rotate``.

TestRotatableMixinRotateSetter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_rotate_without_origin``
   ``obj.rotate("y", 90)`` sets ``axis="y"``, ``angle=90``,
   ``origin=None`` and flips ``do_rotate=True``. Source:
   ``rotatable.py:47-60``.

``test_rotate_with_origin``
   ``obj.rotate("z", 180, origin=(0.1, 0.2, 0.3))`` sets all four
   attributes including the explicit origin. Documents the optional
   origin path.

TestRotatableMixinAbstractEnforcement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_without_do_rotate_implementation_cannot_instantiate``
   Asserts a subclass that omits ``_do_rotate`` raises ``TypeError``.
   Source: ``rotatable.py:62-65``. Forces concrete rotatable classes
   to implement the rotation logic.

Test Catalog — ``cmds_singleuse.py``
------------------------------------

12 once-per-model user-object classes. Every class is tested with the
same four-part pattern: constructor → attribute / kwargs mirror,
``order`` and ``hash`` properties, ``__str__`` round-trip, and
``build(model)`` validation.

TestTitle / TestOutputDir / TestPMLFormulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Single-attribute classes with the same minimal contract:

``test_constructor_stores_attribute_and_kwargs``
   Asserts both the mirrored attribute (e.g. ``Title("x").title``) and
   ``self.kwargs[<name>]`` are populated. The parser-side and
   ``build()``-side contracts must stay in sync.

``test_order_and_hash``
   Asserts the documented ``order`` and ``hash`` values:
   ``Title`` (1, ``#title``), ``OutputDir`` (10, ``#output_dir``),
   ``PMLFormulation`` (7, ``#pml_formulation``).

``test_str_round_trip``
   Asserts ``str(obj) == "#hash: value"``.

``test_build_assigns_title_to_model`` / ``test_build_calls_set_output_file_path``
   ``Title.build`` writes ``self.title`` onto ``model.title``;
   ``OutputDir.build`` calls ``config.get_model_config().set_output_file_path(self.output_dir)``.
   For ``PMLFormulation``, ``test_build_accepts_known_formulations``
   (parametrised over ``"HORIPML"`` / ``"MRIPML"``) and
   ``test_build_rejects_unknown_formulation`` verify the membership
   check against ``PML.formulations`` at ``cmds_singleuse.py:327``.

TestDiscretisation
^^^^^^^^^^^^^^^^^^

``test_constructor_stores_attribute_and_kwargs``
   ``Discretisation((0.001, 0.002, 0.003)).discretisation == (0.001,
   0.002, 0.003)``; ``kwargs == {"p1": (0.001, 0.002, 0.003)}``.

``test_order_and_hash``
   ``order == 2``, ``hash == "#dx_dy_dz"``.

``test_str_round_trip``
   Asserts the tuple is expanded by ``UserObject.__str__`` into
   ``"#dx_dy_dz: 0.001 0.002 0.003"``.

``test_build_sets_model_dl``
   Asserts ``build(model)`` populates ``model.dl`` as an
   ``np.array`` matching the discretisation tuple. Source:
   ``cmds_singleuse.py:88-96``.

TestDomain
^^^^^^^^^^

``test_constructor_stores_attribute_and_kwargs``
   ``Domain((0.2, 0.3, 0.4)).domain_size == (0.2, 0.3, 0.4)``;
   ``kwargs["p1"] == (0.2, 0.3, 0.4)``.

``test_order_and_hash``
   ``order == 3``, ``hash == "#domain"``.

``test_str_round_trip``
   Tuple-expansion round-trip.

``test_build_calls_model_set_size``
   With ``_create_uip`` patched to return a UIP whose
   ``discretise_static_point`` yields ``[100, 100, 100]``, asserts
   ``model.set_size`` is called once and ``model.G.calculate_dt`` is
   called once. Source: ``cmds_singleuse.py:123-166``.

``test_build_raises_when_a_dimension_is_zero``
   With ``nx=0``, asserts ``ValueError``. Pins the "at least one cell
   per dimension" guard at ``cmds_singleuse.py:130-131``.

``test_build_sets_mode`` (parametrised: ``nx,ny,nz`` × expected mode)
   ``(1, 50, 50)`` → ``"2D TMx"``; ``(50, 1, 50)`` → ``"2D TMy"``;
   ``(50, 50, 1)`` → ``"2D TMz"``; ``(50, 50, 50)`` → ``"3D"``.
   Verifies the 2D/3D branch at ``cmds_singleuse.py:141-156`` and the
   side-effect on ``model_config.mode``.

TestTimeStepStabilityFactor
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_constructor_stores_attribute_and_kwargs`` /
``test_order_and_hash`` / ``test_str_round_trip``
   ``stability_factor`` mirror; ``order == 4``, ``hash ==
   "#time_step_stability_factor"``.

``test_build_applies_factor_to_model_dt``
   With ``f=0.5``, asserts ``model.dt_mod == 0.5`` and ``model.dt`` is
   halved. Source: ``cmds_singleuse.py:193-203``.

``test_build_rejects_out_of_range`` (parametrised over 0.0, -0.1,
1.01, 2.0)
   Asserts ``ValueError`` for any value outside ``(0, 1]``. Verifies
   the guard at ``cmds_singleuse.py:194-198``.

TestTimeWindow
^^^^^^^^^^^^^^

``test_default_constructor_both_none``
   ``TimeWindow().time is None`` and ``iterations is None``; both
   present in ``kwargs``. Documents the "deferred check" pattern —
   the validation runs in ``build()``, not ``__init__``.

``test_constructor_time_only`` / ``test_constructor_iterations_only``
   Each kwarg can be supplied independently.

``test_order_and_hash``
   ``order == 5``, ``hash == "#time_window"``.

``test_build_time_mode_sets_iterations``
   With ``time=1e-9``, asserts ``model.timewindow == 1e-9`` and
   ``model.iterations == ceil(1e-9 / dt) + 1``. Source:
   ``cmds_singleuse.py:237-241``.

``test_build_iterations_mode_sets_timewindow``
   With ``iterations=100``, asserts ``model.iterations == 100`` and
   ``model.timewindow == 99 * dt``. Verifies the ``(iterations - 1) *
   dt`` formula at ``cmds_singleuse.py:247``.

``test_build_both_none_raises``
   Asserts ``ValueError`` when neither is set. Source:
   ``cmds_singleuse.py:249-250``.

``test_build_negative_time_raises``
   Asserts ``ValueError`` for ``time=-1.0``.

``test_build_both_set_uses_time_branch``
   With ``time=2e-9`` and ``iterations=100``, asserts
   ``model.timewindow == 2e-9`` — ``time`` wins over ``iterations``
   per the documented behaviour. Verifies the warn-and-prefer-time
   branch at ``cmds_singleuse.py:252-255``.

TestOMPThreads
^^^^^^^^^^^^^^

``test_constructor_stores_attribute_and_kwargs``
   ``OMPThreads(4).omp_threads == 4``; ``kwargs == {"n": 4}``.

``test_order``
   ``order == 6``.

``test_build_sets_model_config_ompthreads``
   With ``set_omp_threads`` patched to return ``4``, asserts
   ``model_config.ompthreads == 4``. Source: ``cmds_singleuse.py:287-293``.

``test_build_rejects_zero_threads``
   Asserts ``ValueError`` for ``n=0``. Verifies the guard at
   ``cmds_singleuse.py:288-289``.

TestPMLThickness
^^^^^^^^^^^^^^^^

``test_constructor_scalar_thickness`` / ``test_constructor_tuple_thickness``
   Both forms supported: ``PMLThickness(10)`` and
   ``PMLThickness((10,)*6)``.

``test_order_and_hash``
   ``order == 7``, ``hash == "#pml_cells"``.

``test_build_scalar_calls_set_pml_thickness`` /
``test_build_tuple_calls_set_pml_thickness``
   Asserts ``grid.set_pml_thickness`` is invoked with the original
   thickness value. Source: ``cmds_singleuse.py:365-373``.

``test_build_rejects_wrong_tuple_length`` (parametrised over 2, 3, 4, 5, 7)
   Asserts ``ValueError`` for any tuple length other than 1 or 6.
   Verifies the guard at ``cmds_singleuse.py:368-371``.

``test_build_rejects_pml_thicker_than_half_domain``
   With ``grid.pmls["thickness"]["x0"] = 30`` and ``model.nx = 50``
   (so ``2 * 30 >= 50``), asserts ``ValueError``. Pins the "half the
   grid" guard at ``cmds_singleuse.py:378-386``.

TestPMLProps
^^^^^^^^^^^^

The deprecated catch-all for legacy ``#pml_properties`` input.

``test_formulation_only``
   With ``formulation="HORIPML"``, asserts ``pml_formulation`` is a
   ``PMLFormulation`` and ``pml_thickness is None``.

``test_thickness_only``
   With ``thickness=10``, asserts ``pml_thickness`` is a
   ``PMLThickness`` and ``pml_formulation is None``.

``test_six_face_thicknesses``
   With ``x0=1, y0=2, z0=3, xmax=4, ymax=5, zmax=6``, asserts
   ``pml_thickness.thickness == (1, 2, 3, 4, 5, 6)``. Verifies the
   per-face composition at ``cmds_singleuse.py:469-477``.

``test_thickness_wins_over_face_kwargs``
   With both ``thickness`` and all six face kwargs supplied,
   ``thickness`` takes precedence. Documents the precedence rule at
   ``cmds_singleuse.py:467-468``.

``test_no_args_raises``
   With no kwargs, asserts ``ValueError``. Verifies the guard at
   ``cmds_singleuse.py:481-484``.

``test_partial_face_kwargs_raises``
   With only five of six face kwargs supplied, asserts ``ValueError``.
   Verifies that the fall-through "all six required" branch fires.

``test_order_and_hash``
   ``order == 7``, ``hash == "#pml_properties"``.

TestSrcSteps / TestRxSteps
^^^^^^^^^^^^^^^^^^^^^^^^^^

Symmetric test pair for the two step-size classes. Each has:

``test_constructor_stores_attribute_and_kwargs``
   ``step_size`` mirror and ``kwargs["p1"]``.

``test_order_and_hash``
   ``SrcSteps`` (8, ``#src_steps``); ``RxSteps`` (9, ``#rx_steps``).

``test_build_writes_to_model_<attr>``
   With ``_create_uip`` patched to return a UIP that discretises to
   ``[10, 20, 30]``, asserts ``model.srcsteps`` /
   ``model.rxsteps`` ends up equal to that array. Source:
   ``cmds_singleuse.py:520-528`` and ``557-565``.

Test Catalog — ``cmds_multiuse.py``
-----------------------------------

19 repeatable user-object classes. Most take ``**kwargs`` and forward
to ``super().__init__`` (attribute mirroring happens during
``build()``), so the constructor test is "kwargs survive verbatim".
Five classes use ``RotatableMixin``: ``VoltageSource``,
``HertzianDipole``, ``MagneticDipole``, ``TransmissionLine``, ``Rx``.

TestExcitationFile
^^^^^^^^^^^^^^^^^^

``test_constructor_stores_attributes_and_kwargs``
   With ``filepath``, ``kind``, ``fill_value`` supplied, asserts all
   three are mirrored and appear in ``kwargs``. Source:
   ``cmds_multiuse.py:78-96``.

``test_optional_kwargs_default_to_none``
   With ``ExcitationFile("x.txt")``, asserts ``kind is None`` and
   ``fill_value is None``.

``test_order_and_hash``
   ``order == 1``, ``hash == "#excitation_file"``.

TestWaveform
^^^^^^^^^^^^

``test_constructor_stores_kwargs``
   Asserts ``Waveform(wave_type="gaussian", amp=1.0, freq=1e9,
   id="wf1").kwargs`` is the four-key dict.

``test_order_and_hash``
   ``order == 2``, ``hash == "#waveform"``.

``test_build_builtin_appends_waveform``
   With a stub grid (empty ``waveforms``), asserts ``build(grid)``
   appends a single ``Waveform`` with the right ``ID``, ``type``,
   ``amp``, ``freq``. Source: ``cmds_multiuse.py:237-250``.

``test_build_unknown_wavetype_raises``
   Asserts ``ValueError`` for ``wave_type="notarealwave"``. Pins the
   membership check at ``cmds_multiuse.py:211-215``.

``test_build_missing_wavetype_raises``
   With ``wave_type`` absent, asserts ``KeyError``. Source:
   ``cmds_multiuse.py:204-210``.

``test_build_zero_frequency_raises``
   With ``freq=0``, asserts ``ValueError``. Verifies the
   "frequency > 0" guard at ``cmds_multiuse.py:227-232``.

``test_build_duplicate_id_raises``
   With an existing waveform of the same ``ID`` already in the grid,
   asserts ``ValueError``. Source: ``cmds_multiuse.py:233-235``.

TestVoltageSource
^^^^^^^^^^^^^^^^^

``test_constructor_stores_attributes_and_kwargs``
   Asserts ``point``, ``polarisation``, ``resistance``,
   ``waveform_id`` mirrors plus ``start``/``stop`` defaulting to
   ``None``; ``kwargs`` carries the same entries.

``test_order_and_hash``
   ``order == 3``, ``hash == "#voltage_source"``.

``test_rotatable_defaults``
   Inherits ``RotatableMixin``: fresh instance has ``do_rotate ==
   False``, ``axis == "x"``, ``angle == 0``, ``origin is None``.

``test_rotate_flips_do_rotate``
   ``v.rotate("z", 90, origin=(0, 0, 0))`` flips ``do_rotate=True``
   and stores the rotation parameters. Documents the rotation setter
   used before ``build()``.

``test_validate_rejects_unknown_polarisation``
   With ``polarisation="w"``, asserts ``_validate_parameters`` raises
   ``ValueError``. Pins the membership check at
   ``cmds_multiuse.py:346-348``.

``test_validate_rejects_negative_resistance``
   With ``resistance=-1``, asserts ``ValueError``. Source:
   ``cmds_multiuse.py:356-360``.

``test_validate_rejects_missing_waveform``
   With ``waveform_id="ghost"`` (not in ``grid.waveforms``), asserts
   ``ValueError``. Source: ``cmds_multiuse.py:362-366``.

``test_validate_rejects_negative_start``
   With ``start=-1.0, stop=1.0``, asserts ``ValueError``.

``test_validate_rejects_zero_duration``
   With ``start=1.0, stop=1.0`` (``stop - start <= 0``), asserts
   ``ValueError``. Source: ``cmds_multiuse.py:368-382``.

TestHertzianDipole / TestMagneticDipole
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_constructor_stores_attributes_and_kwargs``
   ``HertzianDipole(p1=(0,0,0), polarisation="Y", waveform_id="wf1")``
   normalises ``polarisation`` to ``"y"`` and stores the kwargs.
   Documents the case-fold at ``cmds_multiuse.py:471``.

``test_order_and_hash``
   ``HertzianDipole`` (4, ``#hertzian_dipole``); ``MagneticDipole``
   (5, ``#magnetic_dipole``).

``test_validate_rejects_missing_waveform`` /
``test_validate_rejects_bad_polarisation``
   Pin the same validation branches as ``VoltageSource``, slightly
   tailored per class.

TestTransmissionLine
^^^^^^^^^^^^^^^^^^^^

``test_constructor_stores_attributes_and_kwargs`` /
``test_order_and_hash``
   ``order == 6``, ``hash == "#transmission_line"``; ``resistance``
   mirror.

``test_validate_rejects_cuda_solver``
   Overrides ``config.sim_config.general["solver"]`` to ``"cuda"`` and
   asserts ``_validate_parameters`` raises ``ValueError``. Pins the
   GPU-incompatibility guard at ``cmds_multiuse.py:792-798``.

``test_validate_rejects_zero_resistance``
   With ``resistance=0``, asserts ``ValueError``. Verifies the
   "strictly positive" lower bound at ``cmds_multiuse.py:812``.

``test_validate_rejects_resistance_above_free_space_impedance``
   With ``resistance=400.0`` (above ≈ 376.73 Ω free-space impedance),
   asserts ``ValueError``. Verifies the upper-bound guard at
   ``cmds_multiuse.py:812-817``.

TestDiscretePlaneWaveAngles / Vector / Axial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three plane-wave variants sharing the same ``**kwargs`` constructor
contract. Each has:

``test_constructor_stores_kwargs``
   Asserts all required kwargs survive into ``self.kwargs``.

``test_order_and_hash``
   ``DiscretePlaneWaveAngles`` (19, ``#plane_wave_angles``);
   ``DiscretePlaneWaveAxial`` (20, ``#plane_wave_axial``);
   ``DiscretePlaneWaveVector`` (22, ``#plane_wave_vector``).

TestRx
^^^^^^

``test_constructor_stores_attributes_and_kwargs``
   With ``p1=(0.05, 0.05, 0.05), id="rx1", outputs=["Ex", "Ey"]``,
   asserts ``point``, ``id``, ``outputs`` mirror plus ``kwargs["p1"]``.

``test_optional_kwargs_default_to_none``
   With ``Rx(p1=(0,0,0))``, asserts ``id is None`` and
   ``outputs is None``.

``test_order_and_hash``
   ``order == 7``, ``hash == "#rx"``.

``test_rotatable_defaults``
   Inherits ``RotatableMixin``: ``do_rotate == False`` by default.

TestRxArray
^^^^^^^^^^^

``test_constructor_stores_attributes_and_kwargs``
   ``RxArray(p1=(0,0,0), p2=(0.1, 0.1, 0.1), dl=(0.01, 0.01, 0.01))``
   mirrors ``lower_point``, ``upper_point``, ``dl``.

``test_order_and_hash``
   ``order == 8``, ``hash == "#rx_array"``.

TestMaterial
^^^^^^^^^^^^

``test_constructor_stores_kwargs``
   Asserts the five required kwargs (``er``, ``se``, ``mr``, ``sm``,
   ``id``) survive.

``test_order_and_hash``
   ``order == 10``, ``hash == "#material"``.

``test_build_appends_material``
   With a fresh stub grid, asserts ``build(grid)`` appends a new
   material whose ``ID`` matches the kwarg. Source:
   ``cmds_multiuse.py:1629-1647``.

``test_build_rejects_low_er``
   With ``er=0.5``, asserts ``ValueError``. Pins the
   "permittivity ≥ 1" guard at ``cmds_multiuse.py:1602-1606``.

``test_build_rejects_low_mr``
   With ``mr=0.5``, asserts ``ValueError``. Source:
   ``cmds_multiuse.py:1616-1620``.

``test_build_rejects_negative_sm``
   With ``sm=-1.0``, asserts ``ValueError``. Source:
   ``cmds_multiuse.py:1621-1623``.

``test_build_accepts_infinite_conductivity_string``
   With ``se="inf"``, asserts the new material has ``se ==
   float("inf")`` and ``averagable is False``. Documents the
   PEC-sentinel branch at ``cmds_multiuse.py:1607-1615`` and the
   "no averaging for PEC" rule at ``1636-1638``.

``test_build_rejects_duplicate_id``
   With a material of the same ``id`` already in the grid, asserts
   ``ValueError``. Source: ``cmds_multiuse.py:1625-1627``.

``test_build_missing_kwarg_raises``
   With ``sm`` and ``id`` omitted, asserts ``KeyError``. Source:
   ``cmds_multiuse.py:1592-1600``.

TestAddDebyeDispersion / TestAddLorentzDispersion / TestAddDrudeDispersion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_constructor_stores_kwargs``
   Asserts the required kwargs for each model survive into
   ``self.kwargs`` (Debye: ``poles``, ``er_delta``, ``tau``,
   ``material_ids``; Lorentz adds ``omega``, ``delta``; Drude:
   ``poles``, ``omega``, ``alpha``, ``material_ids``).

``test_order_and_hash``
   Debye (11, ``#add_dispersion_debye``); Lorentz (12,
   ``#add_dispersion_lorentz``); Drude (13, ``#add_dispersion_drude``).

``test_build_negative_poles_raises``
   With ``poles=-1``, asserts ``ValueError``. Source:
   ``cmds_multiuse.py:1685-1687``.

``test_build_unknown_material_raises``
   With ``material_ids=["ghost"]`` (not in the grid), asserts
   ``ValueError``. Source: ``cmds_multiuse.py:1690-1695``.

TestSoilPeplinski
^^^^^^^^^^^^^^^^^

``test_constructor_stores_kwargs``
   Asserts seven kwargs (``sand_fraction``, ``clay_fraction``,
   ``bulk_density``, ``sand_density``, ``water_fraction_lower``,
   ``water_fraction_upper``, ``id``) survive.

``test_order_and_hash``
   ``order == 14``, ``hash == "#soil_peplinski"``.

TestMaterialRange / TestMaterialList
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_constructor_stores_kwargs``
   ``MaterialRange`` accepts the eight range kwargs plus ``id``;
   ``MaterialList`` accepts ``list_of_materials`` and ``id``.

``test_order_and_hash``
   ``MaterialRange`` (15, ``#material_range``).

TestPMLCFS
^^^^^^^^^^

``test_constructor_stores_kwargs``
   Asserts all twelve scaling kwargs survive.

``test_order_and_hash``
   ``order == 19``, ``hash == "#pml_cfs"``.

Test Catalog — ``cmds_output.py``
---------------------------------

Three output user-object classes that handle the simulation's data
products: ``Snapshot`` (field samples at a point in time),
``GeometryView`` (mesh dump for visualisation), and
``GeometryObjectsWrite`` (geometry export for re-import).

TestSnapshot
^^^^^^^^^^^^

``test_constructor_stores_attributes``
   Asserts ``lower_bound``, ``upper_bound``, ``dl``, ``filename``,
   ``iterations`` mirror, with ``time`` and ``outputs`` defaulting to
   ``None``. Source: ``cmds_output.py:65-93``.

``test_constructor_stores_kwargs``
   Asserts every constructor kwarg appears in ``self.kwargs``,
   including the optional ``fileext`` and ``outputs``.

``test_order_and_hash``
   ``order == 9``, ``hash == "#snapshot"``.

TestSnapshotCalculateUpperBound
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Verifies the pure-math helper at ``cmds_output.py:95-98`` used to
align an arbitrary upper bound to the snapshot step size.

``test_returns_start_plus_step_times_ceil_size_over_step``
   With ``start=[0,0,0], step=[3,3,3], size=[10,10,10]``, asserts the
   helper returns ``[12, 12, 12]`` (``ceil(10/3) = 4 → 3·4 = 12``).

``test_aligned_size_returns_size``
   With ``size`` already a multiple of ``step``, asserts the helper
   returns ``size`` unchanged.

TestSnapshotBuildValidation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each test patches ``_create_uip`` with a ``MagicMock`` UIP that
returns canned bounds / dl arrays so we can exercise the validation
branches without touching real grid geometry.

``test_build_rejects_subgrid``
   With ``MagicMock(spec=SubGridBaseGrid)``, asserts ``ValueError``.
   Pins the "no snapshots in subgrids" guard at ``cmds_output.py:101-102``.

``test_build_rejects_iterations_above_grid``
   With ``iterations=200`` against a grid carrying ``iterations=100``,
   asserts ``ValueError``. Source: ``cmds_output.py:191-193``.

``test_build_rejects_zero_iterations``
   With ``iterations=0``, asserts ``ValueError``.

``test_build_rejects_negative_time``
   With ``time=-1.0``, asserts ``ValueError``. Source:
   ``cmds_output.py:197-200``.

``test_build_rejects_missing_iterations_and_time``
   With neither ``iterations`` nor ``time`` supplied, asserts
   ``ValueError``. Source: ``cmds_output.py:203-204``.

``test_build_rejects_invalid_fileext``
   With ``fileext=".bogus"``, asserts ``ValueError``. Pins the
   membership check against ``Snapshot.fileexts`` at
   ``cmds_output.py:208-212``.

``test_build_rejects_invalid_output``
   With ``outputs=["Notreal"]``, asserts ``ValueError``. Source:
   ``cmds_output.py:219-225``.

``test_build_defaults_fileext_to_vtkhdf``
   With ``fileext`` omitted, asserts the build path completes and
   ``self.file_extension == ".vtkhdf"`` (the first entry of
   ``Snapshot.fileexts``). Verifies the default-set at
   ``cmds_output.py:206-207`` and that ``model.add_snapshot`` is
   called.

TestGeometryView
^^^^^^^^^^^^^^^^

``test_constructor_stores_attributes_and_kwargs``
   Asserts ``lower_bound``, ``upper_bound``, ``output_type``,
   ``filename`` mirror plus ``kwargs["output_type"]``.

``test_order_and_hash``
   ``order == 17``, ``hash == "#geometry_view"``.

``test_build_rejects_unknown_type``
   With ``output_type="x"`` (only ``"n"`` and ``"f"`` are valid),
   asserts ``ValueError``. Source: ``cmds_output.py:335-338``.

``test_build_fine_requires_dl_equals_one``
   With ``output_type="f"`` and ``dl=[2,2,2]``, asserts ``ValueError``.
   Pins the "fine view requires native discretisation" guard at
   ``cmds_output.py:314-318``.

``test_build_normal_calls_add_voxels``
   With ``output_type="n"``, asserts ``model.add_geometry_view_voxels``
   is called and ``add_geometry_view_lines`` is not.

``test_build_fine_calls_add_lines``
   With ``output_type="f"`` and ``dl=[1,1,1]``, asserts
   ``model.add_geometry_view_lines`` is called and
   ``add_geometry_view_voxels`` is not.

``test_build_rejects_negative_dl``
   With ``dl=[-1, 1, 1]``, asserts ``ValueError``. Verifies the
   non-negativity guard at ``cmds_output.py:303-304``.

TestGeometryObjectsWrite
^^^^^^^^^^^^^^^^^^^^^^^^

``test_constructor_stores_attributes_and_kwargs``
   Asserts ``lower_bound``, ``upper_bound``, ``basefilename`` mirror
   plus ``kwargs["filename"]``.

``test_order_and_hash``
   ``order == 18``, ``hash == "#geometry_objects_write"``.

``test_build_rejects_subgrid``
   With ``MagicMock(spec=SubGridBaseGrid)``, asserts ``ValueError``.
   Pins the "no geometry write from subgrids" guard at
   ``cmds_output.py:384-385``.

``test_build_calls_add_geometry_object``
   With a stub grid and patched UIP, asserts
   ``model.add_geometry_object`` is called.

Test Catalog — ``cmds_geometry/``
---------------------------------

14 classes living under ``cmds_geometry/``: 11 primitive shapes
(``Edge``, ``Plate``, ``Triangle``, ``Box``, ``Cylinder``, ``Cone``,
``CylindricalSector``, ``Sphere``, ``Ellipsoid``, ``FractalBox``,
``GeometryObjectsRead``) plus three modifiers
(``AddSurfaceRoughness``, ``AddSurfaceWater``, ``AddGrass``) that
decorate a ``FractalBox``. All extend ``GeometryUserObject`` so they
share ``order = 1`` (geometry primitives build in arrival order). Six
of them mix in ``RotatableMixin``.

TestCommonContract
^^^^^^^^^^^^^^^^^^

Three parametrised checks that apply uniformly to all 14 classes:

``test_order_is_one`` (parametrised over every class)
   Asserts ``cls().order == 1``. Pins the inherited value from
   ``GeometryUserObject.order``.

``test_hash_matches`` (parametrised over every class with its hash)
   Asserts ``cls().hash`` matches the documented per-class value
   (e.g. ``Box`` → ``"#box"``, ``AddGrass`` → ``"#add_grass"``).

``test_constructor_stores_kwargs_verbatim`` (parametrised over every class)
   With ``cls(arbitrary=1, payload="x", values=(0.1, 0.2))``, asserts
   ``self.kwargs`` is exactly the supplied dict. Pins the common
   ``__init__(**kwargs) → super().__init__(**kwargs)`` contract.

TestBox
^^^^^^^

``test_constructor_kwargs``
   Asserts ``Box(p1=..., p2=..., material_id="free_space")`` mirrors
   the kwargs.

``test_rotatable_defaults``
   ``Box`` inherits ``RotatableMixin``: ``do_rotate == False`` by
   default.

``test_build_missing_p1_raises``
   With ``p1`` and ``p2`` absent, asserts ``KeyError``. Pins the
   "specify two points" guard at ``box.py:64-69``.

``test_build_missing_material_raises``
   With neither ``material_id`` nor ``material_ids`` set, asserts
   ``KeyError`` from the inner ``except KeyError`` chain at
   ``box.py:76-84``.

``test_build_unknown_material_raises``
   With ``material_id="ghost"`` (not in the grid), asserts
   ``ValueError``. Pins the material-lookup error at ``box.py:111-114``.

``test_build_returns_early_when_box_not_in_grid``
   With the UIP returning ``grid_contains_box=False``, asserts
   ``build()`` returns ``None`` without raising. Verifies the early
   return at ``box.py:99-100`` — the Cython call is skipped when the
   box is outside the current grid (relevant for MPI / sub-grids).

TestSphere
^^^^^^^^^^

``test_constructor_kwargs``
   ``Sphere(p1=(0.05, 0.05, 0.05), r=0.02, material_id="free_space")``
   kwargs survive.

``test_build_missing_p1_raises`` /
``test_build_missing_radius_raises``
   Asserts ``KeyError`` for each missing required kwarg. Pins the
   guard at ``sphere.py:53-58``.

``test_build_missing_material_raises`` /
``test_build_unknown_material_raises``
   Same pattern as ``Box``. Source: ``sphere.py:70-91``.

TestEllipsoid
^^^^^^^^^^^^^

``test_constructor_kwargs``
   With ``p1``, ``xr``, ``yr``, ``zr``, ``material_id``, asserts the
   kwargs survive.

``test_build_missing_semiaxis_raises``
   With ``zr`` omitted, asserts ``KeyError``. Pins the
   "point and three semiaxes" guard at ``ellipsoid.py:55-63``.

TestCylinder / TestCone / TestCylindricalSector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_constructor_kwargs``
   Each class accepts its documented parameter shape:

   - ``Cylinder``: ``p1``, ``p2``, ``r``, ``material_id``
   - ``Cone``: ``p1``, ``p2``, ``r1``, ``r2``, ``material_id``
   - ``CylindricalSector``: ``axis``, ``ctr1``, ``ctr2``, ``t1``,
     ``t2``, ``r``, ``sectorstartangle``, ``sectorangle``,
     ``material_id``

   For each class, ``test_constructor_kwargs`` verifies the
   class-specific parameters survive verbatim.

TestEdge / TestPlate / TestTriangle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_constructor_kwargs``
   Three 1D/2D/2D-with-thickness primitives, each with their
   class-specific kwargs: ``Edge`` (two points), ``Plate`` (two
   points, optional averaging), ``Triangle`` (three points,
   ``thickness``).

``test_rotatable_defaults``
   ``Edge`` and ``Plate`` and ``Triangle`` all inherit
   ``RotatableMixin``: ``do_rotate == False`` by default.

TestFractalBox
^^^^^^^^^^^^^^

``test_constructor_kwargs``
   With ``p1``, ``p2``, ``frac_dim``, ``weighting``, ``n_materials``,
   ``mixing_model_id``, ``id``, ``seed``, asserts every kwarg appears
   in ``self.kwargs``.

``test_do_pre_build_default``
   Asserts ``FractalBox().do_pre_build == True``. Source:
   ``fractal_box.py:65``. ``FractalBox`` is the one geometry class
   with a two-phase build (``pre_build`` then ``build``); the flag
   says whether ``Scene.build()`` should run the pre-phase.

``test_pre_build_missing_kwarg_raises``
   With ``mixing_model_id`` absent, asserts ``KeyError``. Pins the
   guard at ``fractal_box.py:74-85``.

TestModifiers
^^^^^^^^^^^^^

Three fractal-box modifiers — surface roughness, surface water,
grass — each tested with one ``test_add_<modifier>_kwargs`` checking
the modifier's specific kwargs survive (``fractal_box_id``,
``limits``, ``depth``, ``n_blades``, ``seed`` depending on the
modifier). All three inherit ``RotatableMixin`` and
``GeometryUserObject``.

TestGeometryObjectsRead
^^^^^^^^^^^^^^^^^^^^^^^

``test_constructor_kwargs``
   ``GeometryObjectsRead(p1=..., geofile="objs.h5",
   matfile="objs_materials.txt")`` mirrors all kwargs into
   ``self.kwargs``. Used to re-import the output of
   ``GeometryObjectsWrite`` into a fresh model.

Running
-------

From the repository root, with the project installed in editable mode
(``pip install -e .``)::

    python -m pytest tests/unit/ -v

Filter to just this PR's suite::

    python -m pytest tests/unit/user_objects/ -v

Run a single file::

    python -m pytest tests/unit/user_objects/test_cmds_singleuse.py -v

Run a single test::

    python -m pytest tests/unit/user_objects/test_cmds_multiuse.py::TestVoltageSource::test_rotate_flips_do_rotate -v

Stop on first failure (useful while iterating)::

    python -m pytest tests/unit/ -x
