Unit Tests — Hash Command Parser
================================

**Branch:** ``feat/unit-tests-hashparser``

**Modules under test:**
   - ``gprMax/hash_cmds_file.py``
   - ``gprMax/hash_cmds_singleuse.py``
   - ``gprMax/hash_cmds_multiuse.py``
   - ``gprMax/hash_cmds_geometry.py``

**Test files:**
   - ``tests/unit/hash_parser/test_hash_cmds_file.py`` (25 tests)
   - ``tests/unit/hash_parser/test_hash_cmds_singleuse.py`` (36 tests)
   - ``tests/unit/hash_parser/test_hash_cmds_multiuse.py`` (78 tests)
   - ``tests/unit/hash_parser/test_hash_cmds_geometry.py`` (52 tests)

**Shared fixtures:** ``tests/unit/hash_parser/conftest.py``

Scope
-----

Verifies the front-door translator that turns a ``.in`` hash-command
text file into the same user-object graph the Python API produces.

The four dispatcher functions covered:

* ``process_python_include_code(inputfile, usernamespace)`` — strips
  comments and blank lines, executes ``#python:`` blocks and captures
  their printed commands, defers to ``process_include_files`` for
  ``#include_file:`` resolution.
* ``process_include_files(hashcmds)`` — replaces every
  ``#include_file: path`` line with the contents of the named file.
* ``check_cmd_names(processedlines, checkessential)`` — bucketises each
  line into the singleuse dict, multiuse dict, or geometry list;
  validates names, the colon-space separator, single-instance rule, and
  essential-commands presence.
* ``process_singlecmds`` / ``process_multicmds`` /
  ``process_geometrycmds`` — turn the bucketed strings into instantiated
  user-objects.

Tests do **not** exercise ``parse_hash_commands(scene)`` or
``write_processed_file(...)`` because both touch
``config.sim_config.input_file_path`` / ``config.get_model_config()`` and
real disk I/O — orchestration-level, out of unit scope for this PR.

Test Infrastructure
-------------------

``tests/unit/hash_parser/conftest.py`` adds two fixtures and two helper
tuples (``SINGLE_KEYS``, ``MULTI_KEYS``):

``singlecmds_template``
   A fresh ``dict.fromkeys(SINGLE_KEYS, None)``. A test sets the one
   command it cares about; every other branch in ``process_singlecmds``
   short-circuits on the ``is not None`` check. New dict per test so
   in-place mutation can't leak.

``multicmds_template``
   ``{key: [] for key in MULTI_KEYS}``. Empty lists produce no scene
   objects; populated lists are iterated by the dispatcher exactly as
   ``check_cmd_names`` would have built them.

No autouse ``config`` patch — the three dispatchers don't read globals.
The one test that needs ``config.sim_config.input_file_path`` (the
fallback include-search root) does the monkeypatch locally.

Test Catalog — ``hash_cmds_singleuse.py``
-----------------------------------------

TestNonePassthrough
^^^^^^^^^^^^^^^^^^^

``test_all_none_yields_empty_list``
   With every key set to ``None``, ``process_singlecmds`` returns ``[]``.
   Source: ``hash_cmds_singleuse.py:50-180``. Pins the short-circuit
   pattern that every command branch relies on.

``test_single_command_does_not_create_others``
   Setting only ``#title`` produces exactly one ``Title`` object.

TestTitle
^^^^^^^^^

``test_title_string_stored``
   ``#title: my model`` → ``Title.title == "my model"``. Source:
   ``hash_cmds_singleuse.py:53-56``.

``test_title_cast_to_str``
   The dispatcher wraps the value in ``str(...)`` so a non-string
   payload survives (mostly defensive; ``check_cmd_names`` always hands
   over strings).

TestOutputDir
^^^^^^^^^^^^^

``test_output_dir_stored``
   ``#output_dir: results/run1`` → ``OutputDir.kwargs["dir"] ==
   "results/run1"``. Source: ``hash_cmds_singleuse.py:58-61``.

TestOMPThreads
^^^^^^^^^^^^^^

``test_single_thread_count_accepted``
   ``#omp_threads: 4`` → ``OMPThreads.omp_threads == 4``. Source:
   ``hash_cmds_singleuse.py:64-74``.

``test_two_tokens_rejected``
   Two-token payload raises ``ValueError``. Pins the
   "exactly one parameter" arity check.

``test_non_integer_token_rejected``
   ``int("abc")`` raises ``ValueError`` inside the dispatcher.

TestDiscretisation
^^^^^^^^^^^^^^^^^^

``test_three_floats_become_tuple``
   ``#dx_dy_dz: 0.001 0.002 0.004`` → ``Discretisation.discretisation ==
   (0.001, 0.002, 0.004)`` and the same value lands under
   ``kwargs["p1"]``. Source: ``hash_cmds_singleuse.py:76-85``.

``test_wrong_arity_rejected`` (parametrised over 2 / 4 tokens)
   Anything other than exactly 3 raises ``ValueError``.

TestDomain
^^^^^^^^^^

``test_three_floats_become_tuple``
   ``#domain: 0.2 0.3 0.4`` → ``Domain.domain_size == (0.2, 0.3, 0.4)``.
   Source: ``hash_cmds_singleuse.py:87-96``.

``test_wrong_arity_rejected`` (parametrised over 2 / 4 tokens)
   Same arity contract as discretisation.

TestTimeStepStabilityFactor
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_first_token_becomes_float_factor``
   ``#time_step_stability_factor: 0.5`` →
   ``TimeStepStabilityFactor.stability_factor == 0.5``. Source:
   ``hash_cmds_singleuse.py:98-102``.

TestTimeWindow
^^^^^^^^^^^^^^

``test_integer_token_routes_to_iterations``
   ``"100"`` is parseable as ``int`` so the dispatcher routes to
   ``TimeWindow(iterations=100)``. ``.time is None``.

``test_float_token_routes_to_time``
   ``"1e-9"`` fails ``int()``; the ``except ValueError`` branch routes
   to ``TimeWindow(time=1e-9)``.

``test_decimal_token_routes_to_time``
   ``"5.0"`` similarly routes to ``time``.

``test_lowercase_normalisation_does_not_strip_sign``
   ``.lower()`` is applied before the parse — ``"1E-9"`` survives.

``test_multi_token_rejected``
   Two tokens raises ``ValueError``.

``test_garbage_token_rejected``
   Neither ``int()`` nor the fallback ``float()`` can parse ``"abc"`` →
   the float retry re-raises ``ValueError``. Source:
   ``hash_cmds_singleuse.py:104-124``.

TestPMLFormulation
^^^^^^^^^^^^^^^^^^

``test_formulation_string_stored``
   ``#pml_formulation: HORIPML`` → ``PMLFormulation.formulation ==
   "HORIPML"``. Source: ``hash_cmds_singleuse.py:126-134``.

``test_multi_token_rejected``
   Two tokens raises ``ValueError``.

TestPMLCells
^^^^^^^^^^^^

``test_uniform_thickness_single_token``
   ``#pml_cells: 10`` → ``PMLThickness.thickness == 10``. Source:
   ``hash_cmds_singleuse.py:137-145``.

``test_invalid_arity_rejected`` (parametrised over 2 / 3 / 4 / 5 tokens)
   Any count outside ``[1, 6]`` raises ``ValueError``.

TestPMLCellsSixArgBranchBug
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_six_token_form_currently_raises_type_error``
   **Tripwire for ``hash_cmds_singleuse.py:147-154``.**
   ``PMLThickness.__init__`` (``cmds_singleuse.py:355``) accepts a single
   positional ``thickness`` argument. The 6-token branch calls
   ``PMLThickness(x0=, y0=, z0=, xmax=, ymax=, zmax=)`` — unknown
   kwargs, ``TypeError``. The 6-token form is currently unreachable as
   valid input. When ``PMLThickness`` is extended (or the dispatcher
   passes ``thickness=(x0, ..., zmax)``) this test must flip.

TestSrcSteps
^^^^^^^^^^^^

``test_three_floats_become_tuple``
   ``#src_steps: 0.01 0.02 0.03`` → ``SrcSteps.kwargs["p1"] == (0.01,
   0.02, 0.03)``. Source: ``hash_cmds_singleuse.py:158-167``.

``test_wrong_arity_rejected`` (parametrised over 2 / 4 tokens)

TestRxSteps
^^^^^^^^^^^

``test_three_floats_become_tuple``
   ``#rx_steps: 0.01 0.02 0.03`` → ``RxSteps.kwargs["p1"] == (0.01,
   0.02, 0.03)``. Source: ``hash_cmds_singleuse.py:169-178``.

``test_wrong_arity_rejected`` (parametrised over 2 / 4 tokens)

TestObjectOrder
^^^^^^^^^^^^^^^

``test_title_then_output_dir_then_threads_then_grid``
   With ``#title``, ``#output_dir``, ``#omp_threads``, ``#dx_dy_dz``,
   ``#domain``, ``#time_window`` all set, the result list is in the
   fixed source-defined order (``Title``, ``OutputDir``, ``OMPThreads``,
   ``Discretisation``, ``Domain``, ``TimeWindow``). Downstream code
   (e.g. ``Title.build`` running before ``Domain.build``) relies on this.

Test Catalog — ``hash_cmds_multiuse.py``
----------------------------------------

TestEmptyDispatch
^^^^^^^^^^^^^^^^^

``test_empty_dict_yields_empty_list`` / ``test_unrelated_family_does_not_pollute``
   Confirm the per-family ``is not None`` short-circuit and that
   populating one family does not leak into others.

TestWaveform
^^^^^^^^^^^^

``test_four_tokens_become_waveform``
   ``#waveform: gaussian 2.0 1e9 wf1`` →
   ``Waveform.kwargs == {wave_type, amp, freq, id}``.
   Source: ``hash_cmds_multiuse.py:61-72``.

``test_wrong_arity_rejected`` (parametrised over 3 / 5 tokens)

``test_multiple_instances_all_dispatched``
   Two waveform strings → two ``Waveform`` objects, IDs in input order.

TestVoltageSource
^^^^^^^^^^^^^^^^^

``test_six_token_short_form``
   No window: ``polarisation="x"``, ``point=(0.05, 0.05, 0.05)``,
   ``resistance=50.0``, ``waveform_id="wf1"``, ``start is None``,
   ``stop is None``. Source: ``hash_cmds_multiuse.py:74-100``.

``test_eight_token_with_window``
   With window: ``start == 1e-9``, ``stop == 5e-9``.

``test_polarisation_lowercased``
   Dispatcher applies ``.lower()`` to ``tmp[0]``; ``"X"`` → ``"x"``.

``test_invalid_arity_rejected`` (parametrised over 7 / 9 tokens)

TestHertzianDipole
^^^^^^^^^^^^^^^^^^

``test_five_token_short_form``
   Sets polarisation, point, waveform; ``start``/``stop`` absent.
   Source: ``hash_cmds_multiuse.py:102-136``.

``test_seven_token_with_window``
   Window captured.

``test_invalid_arity_rejected`` (parametrised over 4 / 8 tokens)

TestMagneticDipole
^^^^^^^^^^^^^^^^^^

Mirrors HertzianDipole. Source: ``hash_cmds_multiuse.py:138-172``.

TestTransmissionLine
^^^^^^^^^^^^^^^^^^^^

``test_six_token_short_form``
   Sets polarisation, point, resistance, waveform. Source:
   ``hash_cmds_multiuse.py:174-206``.

``test_eight_token_with_window_strings``
   **Notable**: ``start``/``stop`` are passed through as *strings*, not
   floats, by this branch (the only one in the file with that
   asymmetry). Pinned so downstream changes don't silently coerce.

``test_invalid_arity_rejected`` (parametrised over 5 / 9 tokens)

TestPlaneWaveAngles
^^^^^^^^^^^^^^^^^^^

``test_ten_token_minimum``
   Minimum arity sets theta/phi/psi and ``waveform_id``. Source:
   ``hash_cmds_multiuse.py:208-255``.

``test_eleven_token_with_material``
   Eleventh token → ``material_id``.

``test_thirteen_token_with_window``
   Tokens 11–12 → ``start``, ``stop``.

``test_invalid_arity_rejected`` (parametrised over 9 / 14 tokens)

TestPlaneWaveAxial
^^^^^^^^^^^^^^^^^^

``test_nine_token_minimum`` / ``test_eleven_token_with_window``
   9- and 11-token forms; ``axis`` is ``.lower()``'d before being
   forwarded. Source: ``hash_cmds_multiuse.py:305-340``.

``test_invalid_arity_rejected`` (parametrised over 8 / 10 / 12 tokens)

TestPlaneWaveVector
^^^^^^^^^^^^^^^^^^^

``test_eleven_token_minimum``
   ``m_vec`` is ``(int, int, int)``, ``psi`` is ``float``. Source:
   ``hash_cmds_multiuse.py:258-300``.

``test_twelve_token_with_material``
   Material id captured.

``test_invalid_arity_rejected`` (parametrised over 10 / 14 tokens)

TestPlaneWaveVectorIndexBug
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_thirteen_token_branch_currently_index_errors``
   **Tripwire for ``hash_cmds_multiuse.py:294``.** The 13-token branch
   reaches ``stop=float(tmp[13])`` — out of bounds for a length-13
   list. ``IndexError``. The branch is unreachable through valid input.
   When fixed (``stop=float(tmp[12])`` plus a 14-token branch for the
   four-extra-token form) this test should flip to assert populated
   ``start`` / ``stop``.

TestExcitationFile
^^^^^^^^^^^^^^^^^^

``test_single_token_filepath_only``
   ``#excitation_file: my.txt`` →
   ``ExcitationFile.kwargs["filepath"] == "my.txt"``. Source:
   ``hash_cmds_multiuse.py:344-357``.

``test_three_token_with_kind_and_fill``
   3-token form populates ``filepath``, ``kind``, ``fill_value``.

``test_invalid_arity_rejected`` (parametrised over 2 / 4 tokens)

TestRx
^^^^^^

``test_three_token_minimal``
   Position only; ``id is None``, ``outputs is None``. Source:
   ``hash_cmds_multiuse.py:359-382``.

``test_five_token_with_id_and_outputs``
   ``"0.05 0.05 0.05 my_rx Ex Ey"`` → ``id="my_rx"``,
   ``outputs=["Ex", "Ey"]``. The dispatcher slices ``tmp[4:]`` for
   outputs, so any number of trailing field names is accepted.

``test_invalid_arity_rejected`` (parametrised over 2 / 4 tokens — the
condition ``len != 3 and len < 5`` rejects both)

TestRxArray
^^^^^^^^^^^

``test_nine_token_form``
   Pins ``p1``, ``p2``, ``dl`` tuples. Source:
   ``hash_cmds_multiuse.py:384-399``.

``test_invalid_arity_rejected`` (parametrised over 8 / 10 tokens)

TestSnapshot
^^^^^^^^^^^^

``test_integer_iterations_branch``
   When ``int(tmp[9])`` succeeds the dispatcher routes to
   ``Snapshot(iterations=...)`` and infers ``fileext`` from the trailing
   ``.<ext>`` of the filename. Source:
   ``hash_cmds_multiuse.py:401-437``.

``test_float_time_branch``
   ``"1e-9"`` fails ``int()`` → ``except ValueError`` routes to
   ``Snapshot(time=...)``.

``test_filename_without_extension_sets_none``
   No dot in filename → ``fileext is None``.

``test_wrong_arity_rejected``
   10 tokens raises ``ValueError``.

TestMaterial
^^^^^^^^^^^^

``test_five_token_form``
   Full kwargs match: ``er``, ``se``, ``mr``, ``sm``, ``id``. Source:
   ``hash_cmds_multiuse.py:439-452``.

``test_invalid_arity_rejected`` (parametrised over 4 / 6 tokens)

TestAddDispersionDebye
^^^^^^^^^^^^^^^^^^^^^^

``test_single_pole``
   1 pole × 2 floats + 1 material id. Source:
   ``hash_cmds_multiuse.py:454-482``.

``test_two_poles_multiple_materials``
   ``poles=2`` ⇒ 4 floats sliced into two ``er_delta`` / ``tau`` pairs,
   trailing tokens collected as ``material_ids``.

``test_below_minimum_arity_rejected``
   Fewer than 4 tokens raises ``ValueError``.

TestAddDispersionLorentz
^^^^^^^^^^^^^^^^^^^^^^^^

``test_single_pole``
   1 pole × 3 floats (``er_delta``, ``omega``, ``delta``) + materials.
   Source: ``hash_cmds_multiuse.py:484-518``.

``test_below_minimum_arity_rejected``

TestAddDispersionDrude
^^^^^^^^^^^^^^^^^^^^^^

``test_single_pole``
   1 pole × 2 floats (``omega``, ``alpha``) + materials. Source:
   ``hash_cmds_multiuse.py:520-548``.

``test_below_minimum_arity_rejected``

TestSoilPeplinski
^^^^^^^^^^^^^^^^^

``test_seven_token_form``
   All seven named kwargs populated. Source:
   ``hash_cmds_multiuse.py:550-574``.

``test_invalid_arity_rejected`` (parametrised over 6 / 8 tokens)

TestMaterialRange
^^^^^^^^^^^^^^^^^

``test_nine_token_form``
   Pins ``er_lower``/``er_upper``/``id``. Source:
   ``hash_cmds_multiuse.py:618-644``.

``test_wrong_arity_rejected``

TestMaterialList
^^^^^^^^^^^^^^^^

``test_variadic_last_token_is_id``
   ``"mat1 mat2 mat3 mixed"`` → ``list_of_materials=["mat1", "mat2",
   "mat3"]``, ``id="mixed"``. Source:
   ``hash_cmds_multiuse.py:646-663``.

``test_two_tokens_is_minimum``

``test_single_token_rejected``

TestGeometryView
^^^^^^^^^^^^^^^^

``test_eleven_token_form``
   Captures ``filename`` and ``output_type``. Source:
   ``hash_cmds_multiuse.py:576-596``.

``test_wrong_arity_rejected``

TestGeometryObjectsWrite
^^^^^^^^^^^^^^^^^^^^^^^^

``test_seven_token_form``
   Pins ``filename``. Source: ``hash_cmds_multiuse.py:598-616``.

``test_wrong_arity_rejected``

TestPMLCFS
^^^^^^^^^^

``test_twelve_token_form``
   All 12 tokens routed to their named kwargs *as strings* (no float
   coercion in this dispatcher branch). Source:
   ``hash_cmds_multiuse.py:665-697``.

``test_wrong_arity_rejected``

TestMultiFamilyDispatch
^^^^^^^^^^^^^^^^^^^^^^^

``test_independent_families_combine_in_source_order``
   Verifies the dispatcher walks families in source order
   (waveform → … → rx → … → material). Locked because downstream uses
   the ordering for stable scene-object IDs.

Test Catalog — ``hash_cmds_geometry.py``
----------------------------------------

TestEmptyDispatch
^^^^^^^^^^^^^^^^^

``test_empty_list_yields_empty_list``
   No commands → no objects. Pins the loop-with-no-default behaviour.

TestGeometryObjectsRead
^^^^^^^^^^^^^^^^^^^^^^^

``test_six_token_form``
   Pins ``p1``, ``geofile``, ``matfile``. Source:
   ``hash_cmds_geometry.py:59-69``.

``test_wrong_arity_rejected``

TestEdge
^^^^^^^^

``test_eight_token_form``
   ``p1``, ``p2``, ``material_id`` captured. Source:
   ``hash_cmds_geometry.py:71-82``.

``test_wrong_arity_rejected``

TestPlate
^^^^^^^^^

``test_eight_token_isotropic`` / ``test_nine_token_anisotropic``
   8-token branch uses ``material_id`` (string); 9-token branch uses
   ``material_ids`` (list). Source: ``hash_cmds_geometry.py:84-109``.

``test_wrong_arity_rejected``

TestTriangle
^^^^^^^^^^^^

``test_twelve_token_isotropic_no_averaging``
   ``thickness`` and singular ``material_id``. Source:
   ``hash_cmds_geometry.py:111-144``.

``test_thirteen_token_isotropic_with_averaging``
   Adds ``averaging``.

``test_fourteen_token_anisotropic``
   Plural ``material_ids`` (3 strings).

``test_wrong_arity_rejected``

TestBox
^^^^^^^

``test_eight_token_isotropic`` / ``test_nine_token_isotropic_with_averaging`` /
``test_ten_token_anisotropic``
   3 valid widths. Source: ``hash_cmds_geometry.py:146-170``.

``test_wrong_arity_rejected`` (too few)

``test_too_many_tokens_rejected``

TestCylinder
^^^^^^^^^^^^

``test_nine_token_isotropic`` / ``test_ten_token_isotropic_with_averaging`` /
``test_eleven_token_anisotropic``
   3 valid widths. Source: ``hash_cmds_geometry.py:172-197``.

``test_wrong_arity_rejected``

TestCone
^^^^^^^^

``test_ten_token_isotropic`` / ``test_eleven_token_isotropic_with_averaging`` /
``test_twelve_token_anisotropic``
   ``r1`` and ``r2`` set independently. Source:
   ``hash_cmds_geometry.py:199-232``.

``test_wrong_arity_rejected``

TestCylindricalSector
^^^^^^^^^^^^^^^^^^^^^

``test_ten_token_isotropic`` / ``test_eleven_token_isotropic_with_averaging`` /
``test_twelve_token_anisotropic``
   ``normal`` is ``.lower()``'d. Source:
   ``hash_cmds_geometry.py:234-295``.

``test_wrong_arity_rejected``

TestSphere
^^^^^^^^^^

``test_six_token_isotropic`` / ``test_seven_token_isotropic_with_averaging``
   2 documented widths. Source: ``hash_cmds_geometry.py:297-321``.

``test_too_few_tokens_rejected`` / ``test_too_many_tokens_rejected``

TestSphereAnisotropicWrongKwargBug
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_anisotropic_branch_stores_list_under_singular_kwarg``
   **Tripwire for ``hash_cmds_geometry.py:315``.** The 8-token branch
   passes ``material_id=tmp[5:]`` — singular kwarg, list value. Every
   other shape's anisotropic branch uses the plural ``material_ids=``.
   At ``Sphere.build`` time the singular wraps the list in another list
   and materials lookup fails. The test asserts the buggy state:
   ``kwargs["material_id"] == ["mx", "my", "mz"]`` and
   ``"material_ids" not in kwargs``. Fix by renaming the kwarg.

TestEllipsoid
^^^^^^^^^^^^^

``test_eight_token_treated_as_isotropic``
   8 tokens lands on the first matching ``elif``. Source:
   ``hash_cmds_geometry.py:323-356``.

``test_nine_token_with_averaging``

``test_too_few_tokens_rejected``

TestEllipsoidAnisotropicDeadCodeBug
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_8_tokens_route_to_isotropic_not_anisotropic``
   **Tripwire for ``hash_cmds_geometry.py:349``.** The branches list
   ``elif len(tmp) == 8`` *twice* — the second one (the anisotropic
   path) is dead code. With 8 tokens the dispatcher always lands on the
   first ``elif`` and stores a single string ``material_id``. Test pins
   the dead-branch reality: ``kwargs["material_id"]`` is the last
   token (a string), ``"material_ids"`` is never populated. Fix by
   changing the duplicate to ``len == 10`` or similar.

TestFractalBox
^^^^^^^^^^^^^^

``test_fourteen_token_minimal``
   Minimum form: ``frac_dim``, ``mixing_model_id``, ``id``, plus a
   ``weighting`` array. Source: ``hash_cmds_geometry.py:358-412``.

``test_fifteen_token_with_seed`` / ``test_sixteen_token_with_seed_and_averaging``
   Optional ``seed`` and ``averaging`` captured.

``test_too_few_tokens_rejected``

TestFractalBoxSeedTypeBug
^^^^^^^^^^^^^^^^^^^^^^^^^

``test_fractal_box_seed_stored_as_string``
   **Tripwire for ``hash_cmds_geometry.py:393``.** The dispatcher
   forwards ``seed=tmp[14]`` *uncast*. The sibling commands
   ``#add_surface_roughness`` (line 453) and ``#add_grass`` (line 517)
   cast to ``int(tmp[N])``. So the same conceptual field lands as
   ``str`` here and ``int`` there. Test pins
   ``isinstance(kwargs["seed"], str)``. Fix by casting in the dispatcher.

TestFractalBoxModifiers
^^^^^^^^^^^^^^^^^^^^^^^

``test_surface_roughness_attached_to_fractal_box``
   ``#fractal_box`` + ``#add_surface_roughness`` (same id) →
   ``[FractalBox, AddSurfaceRoughness]`` in that order.

``test_surface_water_attached_to_fractal_box``
   Same pattern for ``AddSurfaceWater``.

``test_grass_attached_to_fractal_box``
   Same pattern for ``AddGrass``.

``test_modifier_with_mismatched_id_skipped``
   Modifier targets ``fb_other``; current ``#fractal_box`` is ``fb1``.
   Only the ``FractalBox`` lands in the output — the modifier is
   skipped by the ``if tmp[N] != ID: continue`` guard inside the
   nested loop.

``test_modifier_without_parent_fractal_box_silently_dropped``
   Modifier appears without any ``#fractal_box`` line. The outer
   ``for object in geometry`` loop has no ``#add_*`` branch — the line
   is silently dropped. Pinned as intentional (modifiers are only
   recognised inside a fractal-box scope).

TestFractalBoxModifierArity
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_surface_roughness_wrong_arity_rejected`` /
``test_surface_water_wrong_arity_rejected`` /
``test_grass_wrong_arity_rejected``
   Each modifier's arity guard raises ``ValueError`` on short input.

TestUnknownCommandSilentlyDropped
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_unknown_command_yields_nothing``
   The outer ``if/elif`` chain has no final ``else``. Unknown geometry
   commands produce no scene object and no error. Pinned so a future
   regression (e.g. a typo'd new shape) doesn't go unnoticed.

Test Catalog — ``hash_cmds_file.py``
------------------------------------

TestProcessPythonIncludeCode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Driven with ``io.StringIO`` so no real files are needed.

``test_double_hash_comments_stripped``
   Lines starting with ``##`` are dropped before any other processing.
   Source: ``hash_cmds_file.py:51-54``.

``test_blank_lines_stripped``
   ``line.rstrip("\n")`` non-empty filter removes blanks.

``test_non_hash_lines_dropped``
   Plain prose without a leading ``#`` is silently skipped.

``test_python_block_emits_printed_commands``
   ``#python:`` / ``#end_python:`` block executes via ``exec``;
   ``print(...)`` output is sliced into hash lines (kept) and other
   lines (logged). Source: ``hash_cmds_file.py:65-111``.

``test_python_block_namespace_passthrough``
   ``usernamespace`` reaches the executed code: ``f'#title: hello
   {NAME}'`` resolves ``NAME`` from the namespace dict.

``test_missing_end_python_raises_syntax_error``
   Unterminated block walks off the end of ``inputlines`` and raises
   ``SyntaxError``. Source: ``hash_cmds_file.py:79-83``.

``test_stdout_reset_to_os_stdout_after_python_block``
   The dispatcher swaps ``sys.stdout`` for an internal ``StringIO`` and
   then resets it via ``sys.stdout = sys.__stdout__``. Pinned so a
   future change that forgets to restore is caught.

TestProcessIncludeFiles
^^^^^^^^^^^^^^^^^^^^^^^

``test_no_include_lines_passes_through``
   List without any ``#include_file:`` round-trips identically.

``test_include_file_inlines_contents``
   ``#include_file: <abspath>`` replaced with the file's lines, each
   newline-terminated. Uses ``tmp_path``. Source:
   ``hash_cmds_file.py:127-172``.

``test_include_file_drops_comments_and_blanks``
   The included file is filtered through the same ``##``/blank
   filter as the top-level file.

``test_wrong_arity_rejected``
   ``#include_file: a b`` raises ``ValueError``.

``test_relative_path_falls_back_to_input_file_parent``
   First lookup (``Path(includefile)``) misses; the dispatcher then
   re-resolves under ``config.sim_config.input_file_path.parent``.
   Patched via ``monkeypatch.setattr(config, "sim_config", ...)``.
   Source: ``hash_cmds_file.py:152-156``.

TestCheckCmdNames
^^^^^^^^^^^^^^^^^

``test_single_cmd_routed_to_single_dict``
   Singleuse commands land in the singleuse dict; other buckets stay
   empty. Source: ``hash_cmds_file.py:196-354``.

``test_multi_cmd_appended_to_multi_dict``
   Two ``#waveform`` lines → two entries in ``multi["#waveform"]``,
   in input order.

``test_geometry_cmd_appended_to_geometry_list``
   Geometry commands preserved as full strings (with the trailing
   colon) in the geometry list, in input order.

``test_duplicate_single_cmd_rejected``
   Second occurrence of a singleuse command raises ``SyntaxError``.

``test_unknown_command_rejected``
   Unknown command name raises ``SyntaxError``.

``test_missing_space_after_colon_rejected``
   ``#title:demo`` (no space) raises ``SyntaxError`` per the
   first-character-is-space rule.

``test_missing_colon_exits``
   No ``:`` in the line → ``exit(1)`` (``SystemExit``). Note: the
   ``exit(1)`` call is inconsistent with the rest of the function
   which uses ``raise SyntaxError``; pinned for future cleanup.

``test_missing_essentials_rejected_when_check_enabled``
   ``checkessential=True`` and not all of ``#domain``, ``#dx_dy_dz``,
   ``#time_window`` present → ``SyntaxError``.

``test_missing_essentials_accepted_when_check_disabled``
   ``checkessential=False`` lets a sub-essential set through.

``test_returns_three_distinct_containers``
   Return signature: ``(dict, dict, list)``.

TestGetUserObjects
^^^^^^^^^^^^^^^^^^

``test_essentials_produce_singleuse_objects``
   Just the three essentials in → ``[Discretisation, Domain,
   TimeWindow]`` in source order.

``test_mixed_command_buckets_all_appear``
   Singleuse + multiuse + geometry input → all three categories
   produce objects, concatenated as ``single + multi + geometry``.
   Pinned so the concatenation order doesn't silently flip.

``test_skip_essential_check_allows_minimal_input``
   ``checkessential=False`` routes through both dispatchers without
   requiring essentials.

Running
-------

From the repository root, with the project installed in editable mode
(``pip install -e .``)::

    python -m pytest tests/unit/ -v

Filter to just this PR's suites::

    python -m pytest tests/unit/hash_parser/ -v

Run a single test::

    python -m pytest tests/unit/hash_parser/test_hash_cmds_multiuse.py::TestPlaneWaveVectorIndexBug -v

Stop on first failure::

    python -m pytest tests/unit/ -x
