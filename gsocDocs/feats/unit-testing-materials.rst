Unit Tests — Materials
======================

**Branch:** ``feat/unit-testing-materials``

**Module under test:** ``gprMax/materials.py``

**Test file:** ``tests/unit/materials/test_materials.py``

**Shared fixtures:** ``tests/unit/conftest.py``

**Scoped fixtures:** ``tests/unit/materials/conftest.py``

Scope
-----

Verifies the five material classes (``Material``, ``DispersiveMaterial``,
``PeplinskiSoil``, ``RangeMaterial``, ``ListMaterial``) and the five
module-level helpers (``calculate_water_properties``,
``create_built_in_materials``, ``create_water``, ``create_grass``,
``process_materials``).

Tests do not exercise the FDTD solver, source injection, or any I/O. They
are pure-Python regression checks against closed-form references where the
underlying math allows (Material/DispersiveMaterial coefficient formulae,
Debye/Lorentz/Drude permittivity at DC and high-frequency limits) and
direct attribute / table-shape checks otherwise.

Test Infrastructure
-------------------

``tests/unit/conftest.py`` provides three new factory fixtures (alongside
the existing ``make_waveform``):

``make_material(ID="test", numID=0, er=1.0, se=0.0, mr=1.0, sm=0.0)``
   Factory returning a configured non-dispersive ``Material``. Defaults
   describe free space; tests override only the parameters they care
   about.

``make_dispersive(ID="disp", numID=0, model="debye", er=1.0, se=0.0, poles=())``
   Factory returning a configured ``DispersiveMaterial``. ``poles`` is a
   sequence of ``(deltaer, tau, alpha)`` triples whose interpretation
   depends on ``model``:

   - ``"debye"`` — ``tau`` is the relaxation time; ``alpha`` is unused
   - ``"lorentz"`` — ``tau`` is the pole frequency; ``alpha`` is the damping coefficient
   - ``"drude"`` — ``tau`` is the pole frequency; ``alpha`` is the inverse relaxation time

``fake_grid(dt=1e-12, dx=1e-3, dy=1e-3, dz=1e-3, materials=None)``
   ``SimpleNamespace`` stand-in for ``gprMax.grid.fdtd_grid.FDTDGrid``.
   Materials methods only read ``dt``, ``dx``, ``dy``, ``dz``, and a
   ``materials`` list; the full grid object is not required at the unit
   level.

``tests/unit/materials/conftest.py`` adds one auto-applied fixture:

``material_config`` (autouse)
   Monkeypatches ``gprMax.config`` for the duration of each test so the
   Material methods that read global state run against a predictable
   environment. Specifically it patches:

   - ``config.sim_config.em_consts["e0"]`` → ``scipy.constants.epsilon_0``
   - ``config.sim_config.em_consts["m0"]`` → ``scipy.constants.mu_0``
   - ``config.get_model_config().materials["maxpoles"]`` → ``1``
   - ``config.get_model_config().materials["dispersivedtype"]`` → ``np.complex128``

   The fixture also yields a ``SimpleNamespace`` so individual tests can
   override ``maxpoles`` or read the constants without re-importing scipy.

Test Catalog
------------

Each entry below lists the assertion, the property it verifies, and the
most likely source location to inspect on failure.

TestMaterialDefaults
^^^^^^^^^^^^^^^^^^^^

``test_init_stores_numID_and_ID``
   Asserts ``Material(7, "concrete")`` sets ``numID == 7`` and
   ``ID == "concrete"``. Source: ``materials.py:34-42``.

``test_init_defaults_to_free_space``
   Asserts the default constitutive parameters are ``er=1.0``, ``se=0.0``,
   ``mr=1.0``, ``sm=0.0``. Free space is the EM vacuum — any change here
   silently shifts the baseline for every simulation. Source:
   ``materials.py:47-51``.

``test_init_defaults_averagable_true``
   Asserts ``averagable is True`` by default. Material averaging at
   cell boundaries depends on this flag; PEC and Debye-built materials
   override it to ``False``.

``test_init_type_starts_empty``
   Asserts ``type == ""``. The ``type`` field is set later by callers
   (``"builtin"``, ``"debye"``, ``"lorentz"``, ``"drude"``).

TestMaterialEquality
^^^^^^^^^^^^^^^^^^^^

``test_equal_when_IDs_match``
   Asserts two ``Material`` instances with the same ``ID`` compare equal
   even when ``numID`` differs. Source: ``materials.py:53-59``.

``test_not_equal_when_IDs_differ``
   Asserts inequality follows from differing ``ID``.

``test_eq_against_non_material_raises_typeerror``
   Asserts ``Material == <non-Material>`` raises ``TypeError``. Catches a
   silent miscompare against, e.g., a raw string ``"sand"``.

TestMaterialOrdering
^^^^^^^^^^^^^^^^^^^^

Verifies the three-rule ordering documented at ``materials.py:61-79``:

``test_two_non_compound_ordered_by_numID``
   Two non-compound materials sort by ``numID``.

``test_two_compound_ordered_by_ID``
   Two compound materials sort alphabetically by ``ID``. The ``numID`` of
   a compound material is not guaranteed to be consistent across MPI
   ranks, so ID-based ordering is the only stable choice.

``test_non_compound_less_than_compound``
   Mixed comparison: a non-compound material is always less than a
   compound material, regardless of ``numID`` values.

``test_lt_against_non_material_raises_typeerror``
   Same as for ``__eq__``: comparing against a non-Material raises.

TestCompoundMaterials
^^^^^^^^^^^^^^^^^^^^^

``test_is_compound_material`` (parametrised over four IDs)
   Asserts the ``"+"``-based detection at ``materials.py:100-112``
   correctly returns ``True`` for ``"sand+clay"``, ``"a+b+c"`` and
   ``False`` for ``"sand"``, ``""``.

TestCreateCompoundID
^^^^^^^^^^^^^^^^^^^^

``test_two_materials_doubles_and_sorts``
   For two inputs, the compound ID lists each material twice
   (``"clay+clay+sand+sand"``) per the docstring contract at
   ``materials.py:114-131``.

``test_three_materials_sorted_alphabetically``
   For three or more inputs, the IDs are sorted alphabetically and
   joined with ``"+"`` once each (``"air+clay+sand"``).

TestUpdateCoeffsH
^^^^^^^^^^^^^^^^^

Closed-form references for the magnetic update coefficients at
``materials.py:133-146``. With ``sm = 0``, ``HA == HB`` so ``DA = 1`` and
``DBx = dt / (m0 * mr * dx)``.

``test_free_space_DA_is_unity``
   Asserts ``DA == 1.0`` for free space. Failure indicates the
   ``HB / HA`` ratio at ``materials.py:142`` no longer matches when
   ``sm = 0``.

``test_free_space_DBx_matches_closed_form``
   Asserts ``DBx == dt / (m0 * dx)`` exactly. Failure indicates either
   the ``HA`` formula at ``materials.py:140`` or the spatial-step term
   at ``materials.py:143`` has drifted.

``test_free_space_DB_components_scale_with_inverse_spacing``
   Asserts ``DBy = DBx/2`` and ``DBz = DBx/4`` when ``dy = 2·dx`` and
   ``dz = 4·dx``. Verifies the per-axis ``1/dl`` factors at
   ``materials.py:143-145`` are not transposed.

``test_lossy_magnetic_DA_less_than_one``
   With ``sm > 0`` we have ``HB < HA`` so ``0 < DA < 1`` — magnetic loss
   damps the leapfrog. Failure indicates a sign flip in either ``HA`` or
   ``HB``.

TestUpdateCoeffsE
^^^^^^^^^^^^^^^^^

Closed-form references for the electric update coefficients at
``materials.py:148-169``.

``test_free_space_CA_is_unity``
   Asserts ``CA == 1.0`` for free space.

``test_free_space_CBx_matches_closed_form``
   Asserts ``CBx == dt / (e0 * dx)`` exactly.

``test_conductive_dielectric_CA_less_than_one``
   With finite ``se``, ``EB < EA`` so ``0 < CA < 1``.

``test_pec_by_ID_zeros_all_coefficients``
   With ``ID == "pec"`` the short-circuit at ``materials.py:158`` zeros
   ``CA``, ``CBx/y/z``, and ``srce``. Models metal walls — the field
   cannot penetrate.

``test_pec_by_infinite_conductivity_zeros_all_coefficients``
   Same coefficients zero when ``se == float("inf")``. There are two
   independent paths to PEC; both must zero the coefficients or some
   metals will silently radiate.

TestMaterialCalculateER
^^^^^^^^^^^^^^^^^^^^^^^

``test_non_dispersive_returns_static_er``
   Asserts ``calculate_er(freq)`` returns ``self.er`` regardless of
   ``freq``. The base ``Material`` is frequency-independent by design;
   ``DispersiveMaterial`` overrides this method.

TestDispersiveDefaults
^^^^^^^^^^^^^^^^^^^^^^

``test_inherits_material_defaults``
   Asserts a fresh ``DispersiveMaterial`` carries the same free-space
   defaults as ``Material``.

``test_pole_lists_start_empty``
   Asserts ``poles == 0`` and ``deltaer``, ``tau``, ``alpha`` are empty
   lists. Pole configuration is added later by callers.

TestDispersiveCalculateER
^^^^^^^^^^^^^^^^^^^^^^^^^

Spot checks against textbook limits of the closed-form expressions at
``materials.py:299-313``.

``test_debye_dc_limit_returns_static_permittivity``
   At ``f → 0``, a Debye material with ``er_inf = 4.9`` and one pole of
   ``deltaer = 73.2`` returns ``Re(er) ≈ 78.1``. Verifies the term
   ``deltaer / (1 + j·w·tau) → deltaer`` when ``w → 0``.

``test_debye_high_frequency_limit_returns_er_infinity``
   At ``f = 1 PHz``, ``Re(er) → er_inf = 4.9`` because the Debye term
   collapses. Failure indicates the ``j·w·tau`` denominator has been
   inverted.

``test_lorentz_dc_limit_returns_static_permittivity``
   At ``w = 0`` the Lorentz expression collapses to
   ``deltaer · tau² / tau² = deltaer``, so ``Re(er) = er_inf + deltaer``.
   Failure indicates the ``tau² + 2j·w·alpha - w²`` denominator at
   ``materials.py:307`` has drifted.

TestDispersiveDrudeBug
^^^^^^^^^^^^^^^^^^^^^^

``test_two_poles_match_current_buggy_formula``
   **Tripwire for the multi-pole Drude bug at ``materials.py:309-313``.**
   The line ``er -= ersum`` is *inside* the pole loop, so on iteration
   ``n`` the cumulative sum (including pole 0) is subtracted again,
   double-counting every earlier pole. With two poles the result is
   ``1 - 2·pole₀ - pole₁`` instead of the correct ``1 - pole₀ - pole₁``.
   Single-pole Drude is unaffected. When the source is fixed, this test
   must be inverted in the same PR to assert the correct sum-of-poles
   formula.

TestDispersiveUpdateCoeffsE
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_debye_single_pole_assigns_finite_CA``
   Asserts ``CA`` and ``CBx`` are finite (not NaN, not Inf) after a
   single-pole Debye material runs through
   ``calculate_update_coeffsE``. Catches divide-by-zero or overflow in
   the per-pole formulas at ``materials.py:244-265``.

``test_debye_zero_pole_recovers_non_dispersive_CA``
   A "dispersive" material configured with ``deltaer = 0`` (and ``se = 0``)
   collapses to a plain dielectric: ``CA == 1``. Verifies that
   ``DispersiveMaterial.calculate_update_coeffsE`` reduces to its
   parent's behaviour at the degenerate pole.

TestDispersiveDrudeSelfMutationBug
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_se_grows_between_consecutive_calls``
   **Tripwire for the in-place ``self.se`` mutation at
   ``materials.py:258``.** The line ``self.se += wp2 / self.alpha[x]``
   runs every time ``calculate_update_coeffsE`` is called, so a second
   call on the same instance produces a different ``self.se`` than the
   first. Update-coefficient computation should be idempotent for a
   fixed grid. When the source is fixed, this test must assert
   ``se_after_second == se_after_first``.

TestCalculateWaterProperties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Numerical references come from running the formula at
``materials.py:582-598`` directly at ``T = 25``, ``S = 0`` — they are
**not** independent textbook values, so a future change to the formula
will (correctly) fail these tests and force a code review.

``test_fresh_water_at_25C_eri_is_4p9``
   Asserts ``eri == 4.9`` — the canonical high-frequency limit of the
   extended Debye model.

``test_fresh_water_at_25C_static_er_matches_formula``
   Asserts ``er`` equals ``88.045 - 0.4147·T + 6.295e-4·T² + 1.075e-5·T³``
   at ``T = 25``. Failure indicates a coefficient in the cubic
   polynomial has changed.

``test_fresh_water_conductivity_is_zero``
   At ``S = 0``, ``sig_25s = 0`` so ``sig = 0``. Pure water is a perfect
   insulator in this model.

``test_saline_water_has_positive_conductivity``
   At ``S = 35`` (seawater) the conductivity is strictly positive.
   Verifies the salinity branch of the model fires.

TestCreateBuiltIns
^^^^^^^^^^^^^^^^^^

``test_appends_pec_and_free_space``
   Asserts ``create_built_in_materials(G)`` appends exactly two
   materials with IDs ``"pec"`` and ``"free_space"``, in that order.
   Source: ``materials.py:551-566``.

``test_pec_marked_non_averagable``
   Asserts ``pec.averagable is False`` and ``pec.se == float("inf")``.
   PEC must not be averaged with neighbouring cells — that would
   smear the perfect-conductor boundary.

TestCreateWater
^^^^^^^^^^^^^^^

``test_appends_single_dispersive_water_material``
   Asserts a single ``DispersiveMaterial`` with ``ID == "water"`` and
   ``poles == 1`` is appended. Source: ``materials.py:601-623``.

TestCreateGrass
^^^^^^^^^^^^^^^

``test_appends_single_dispersive_grass_material``
   Asserts a single ``DispersiveMaterial`` with ``ID == "grass"`` and
   ``poles == 1`` is appended. Source: ``materials.py:626-649``.

TestPeplinskiSoilInit
^^^^^^^^^^^^^^^^^^^^^

``test_stores_constructor_arguments``
   Asserts every constructor argument is stored verbatim on the
   instance. Source: ``materials.py:323-346``.

TestPeplinskiSoilProperties
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_generates_nbins_dispersive_materials``
   For ``nbins = 5``, asserts ``len(G.materials) == 5`` and every entry
   is a single-pole Debye ``DispersiveMaterial``. Source:
   ``materials.py:348-421``.

``test_matID_records_all_generated_materials``
   Asserts ``soil.matID`` records the ``numID`` of every generated
   material in append order. Failure here means downstream geometry
   commands cannot locate the soil materials by numID.

TestRangeMaterialInit
^^^^^^^^^^^^^^^^^^^^^

``test_stores_all_ranges``
   Asserts the four range tuples are stored as ``er``, ``sig``, ``mu``,
   ``ro``. Note the attribute naming inconsistency: ``se_range`` is
   stored as ``sig``, ``mr_range`` as ``mu``, ``sm_range`` as ``ro``.

TestRangeMaterialProperties
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_generates_nbins_new_materials``
   For ``nbins = 4``, asserts ``len(G.materials) == 4``. Source:
   ``materials.py:449-510``.

``test_generated_er_values_monotonically_increase``
   Asserts the generated materials' ``er`` values are sorted ascending,
   verifying the ``linspace``-based binning produces monotonic outputs.

TestListMaterialInit
^^^^^^^^^^^^^^^^^^^^

``test_stores_list_of_material_IDs``
   Asserts ``ID``, ``mat``, and an empty ``matID`` are stored. Source:
   ``materials.py:519-529``.

TestListMaterialLookup
^^^^^^^^^^^^^^^^^^^^^^

``test_looks_up_existing_materials_by_ID``
   Given a grid pre-populated with two materials and a ``ListMaterial``
   referencing them by ID, asserts ``lm.matID`` contains both
   ``numID`` values in declared order. Source: ``materials.py:531-548``.

TestListMaterialMissingMaterialBug
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_missing_material_raises_attribute_error``
   **Tripwire for the bug at ``materials.py:544-548``.** The code calls
   ``self.matID.append(material.numID)`` *before* checking whether
   ``material`` is ``None``. Looking up a missing ID raises
   ``AttributeError`` instead of the intended ``ValueError`` documented
   at ``materials.py:547-548``. When fixed, this test should assert
   ``pytest.raises(ValueError)``.

TestProcessMaterials
^^^^^^^^^^^^^^^^^^^^

End-to-end smoke tests for the orchestrator at ``materials.py:652-759``.

``test_fills_E_coeffs_for_each_material``
   Builds a grid with two free-space materials and asserts that after
   ``process_materials(G)`` the first column of ``G.updatecoeffsE``
   (``CA``) is ``1.0`` for both rows. Verifies the per-material loop at
   ``materials.py:696-715`` writes into the correct row.

``test_returns_table_with_header_and_one_row_per_material``
   Asserts the returned table has ``1 + N`` rows: one header followed by
   one row per material. Failure indicates either a missing
   ``append(materialtext)`` at ``materials.py:757`` or a malformed
   header at ``materials.py:665-694``.

Running
-------

From the repository root, with the project installed in editable mode
(``pip install -e .``)::

    python -m pytest tests/unit/ -v

Filter to the materials suite::

    python -m pytest tests/unit/materials/ -v

Run a single test::

    python -m pytest tests/unit/materials/test_materials.py::TestUpdateCoeffsE::test_pec_by_ID_zeros_all_coefficients -v

Stop on first failure (useful while iterating)::

    python -m pytest tests/unit/ -x