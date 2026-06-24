Unit Tests — Sources and Receivers
==================================

**Branch:** ``feat/unit-tests-sources-receivers``

**Modules under test:**
   - ``gprMax/sources.py``
   - ``gprMax/receivers.py``

**Test files:**
   - ``tests/unit/sources/test_sources.py`` (59 tests)
   - ``tests/unit/receivers/test_receivers.py`` (21 tests)

**Shared fixtures:** ``tests/unit/conftest.py``

**Scoped fixtures:**
   - ``tests/unit/sources/conftest.py``
   - ``tests/unit/receivers/conftest.py``

Scope
-----

Verifies the five source classes (``Source`` base, ``VoltageSource``,
``HertzianDipole``, ``MagneticDipole``, ``TransmissionLine``,
``DiscretePlaneWave``) and the receiver class (``Rx``) along with the
four device-transfer helpers (``htod_src_arrays``, ``htod_rx_arrays``,
``dtoh_rx_array``, and the pure-Python parts of
``DiscretePlaneWave.calculate_waveform_values``).

Tests do not exercise the FDTD solver, cython kernels, real GPU
devices, or HDF5 I/O. They are pure-Python regression checks against
the documented update formulas and array shapes. The CUDA paths of the
device-transfer helpers are exercised through a ``sys.modules`` shim
that injects a fake ``pycuda.gpuarray`` so the assertions can read what
*would* have been shipped to the GPU.

Test Infrastructure
-------------------

``tests/unit/conftest.py`` gains three additions on top of the
materials-suite fixtures:

``fake_grid`` (extended)
   Now accepts the keyword arguments source / receiver methods read:
   ``iterations``, ``timewindow``, ``waveforms``, ``voltagesources``,
   ``hertziandipoles``, ``magneticdipoles``, ``transmissionlines``,
   ``IDlookup``, ``ID``, ``rxs``. Defaults: ``dx=dy=dz=1``,
   ``iterations=10``, ``IDlookup`` mapping the six standard field
   components to indices 0..5. Any extra ``**kwargs`` are attached as
   plain attributes, so a test that needs ``G.calculate_Iz`` can pass
   it directly.

``_ConstantWaveform`` / ``make_constant_waveform``
   Test double for ``gprMax.waveforms.Waveform``.
   ``calculate_value(time, dt)`` returns ``value`` whenever ``time >= 0``
   and ``0`` otherwise. Eliminates dependency on the gaussian / ricker
   coefficient math; tests of the source-time-window logic rely on
   ``Source.start`` / ``Source.stop`` instead of the waveform itself.

``tests/unit/sources/conftest.py`` adds one auto-applied fixture:

``source_config`` (autouse)
   Monkeypatches ``gprMax.config`` for the duration of each test:

   - ``config.sim_config.general["solver"]`` → ``"cpu"`` (override per test)
   - ``config.sim_config.dtypes["float_or_double"]`` → ``np.float64``
   - ``config.c``, ``config.e0``, ``config.m0`` → scipy constants

``tests/unit/receivers/conftest.py`` mirrors the same pattern with a
smaller patch surface (no ``c``/``e0``/``m0`` needed).

Test Catalog — ``sources.py``
-----------------------------

TestSourceBase
^^^^^^^^^^^^^^

``test_init_defaults``
   Asserts a fresh ``Source()`` has ``polarisation=None``, ``start=0.0``,
   ``stop=0.0``, ``waveformID=None`` and both waveform-value arrays are
   ``None``. Source: ``sources.py:45-59``. Failure indicates a default
   has been changed in a way that downstream subclasses no longer
   override correctly.

``test_coord_arrays_are_zero_int32``
   Asserts ``coord`` and ``coordorigin`` are zero-filled ``np.int32``
   arrays of shape ``(3,)``. The solver uses these to index 4D field
   arrays; a wrong dtype silently coerces or crashes. Source:
   ``sources.py:51-52``.

``test_coord_property_round_trips`` (parametrised over x/y/z)
   Asserts ``setattr(s, "xcoord", 7)`` writes to ``s.coord[0]`` and
   ``s.xcoord`` reads it back. Source: ``sources.py:61-83``.

``test_coordorigin_property_round_trips`` (parametrised over x/y/z)
   Same as above for the origin coordinate (used for sub-grid offsets).
   Source: ``sources.py:85-107``.

TestVoltageSourceInit
^^^^^^^^^^^^^^^^^^^^^

``test_inherits_source_defaults``
   Asserts a fresh ``VoltageSource()`` carries the base-class defaults
   AND adds ``resistance=None``. Source: ``sources.py:117-119``.

TestVoltageSourceCalculateWaveformValues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_populates_both_arrays_inside_window``
   With ``start=0`` and ``stop=timewindow``, asserts both
   ``waveformvalues_wholedt`` and ``waveformvalues_halfdt`` are filled
   with the constant waveform value, and each array has shape
   ``(iterations + 1,)``. Source: ``sources.py:121-158``. Failure
   indicates the loop bounds drifted (e.g. ``range(iterations)``
   instead of ``range(iterations + 1)``) or the dtype changed.

``test_zero_outside_window``
   With a window covering only iterations 3..6, asserts in-window
   entries equal the constant value and out-of-window entries are zero.
   Verifies the ``time >= self.start and time <= self.stop`` guard at
   ``sources.py:151``.

``test_reuses_precomputed_values_from_matching_source``
   Pre-populates ``G.voltagesources`` with another ``VoltageSource``
   carrying the same ``waveformID``; asserts a new source's
   ``waveformvalues_halfdt`` / ``waveformvalues_wholedt`` are
   identity-equal numpy views of the pre-existing arrays. Verifies the
   reuse path at ``sources.py:131-138``. Failure means the optimisation
   has broken — every source pays the recomputation cost.

TestVoltageSourceUpdateElectric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_resistive_decrements_E_by_documented_formula`` (parametrised over x/y/z)
   With ``resistance=2``, ``waveformvalues_halfdt[1]=4``, ``coeff=1``,
   unit spacing, asserts ``E[1,1,1] == -2.0`` after one
   ``update_electric`` call (``-= 1 * 4 / (2 * 1 * 1) = -2``). Source:
   ``sources.py:179-205``. Failure indicates the ``1 / (R * d_a * d_b)``
   scaling has drifted.

``test_hard_source_overwrites_E_with_negated_waveform`` (parametrised over x/y/z)
   With ``resistance=0`` and ``waveformvalues_wholedt[1]=3``, asserts
   ``E[1,1,1] == -1.5`` after one update (``-1 * 3 / dx`` with
   ``dx=2``), and the assignment *overwrites* a sentinel value of 999
   rather than decrementing it. Verifies the hard-source branch at
   ``sources.py:187``.

``test_outside_window_is_noop``
   With ``start=2``, ``stop=3``, asserts an ``update_electric`` call at
   iteration 0 (``time=0``) leaves the field untouched. Verifies the
   guard at ``sources.py:173``.

TestVoltageSourceCreateMaterial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_resistance_zero_returns_without_modifying_grid``
   With ``resistance=0``, asserts ``create_material(G)`` returns
   immediately and ``G.materials`` is unchanged. Source: ``sources.py:217-218``.
   A hard source is the field directly — no synthetic material needed.

``test_resistive_appends_new_material_with_added_conductivity``
   (parametrised over x/y/z)
   With ``resistance=50``, asserts a new ``Material`` is appended with
   ``ID = "<base>+<vs>"``, ``averagable=False``, and ``se`` increased
   by the polarisation-specific factor (``dx/(R·dy·dz)`` for x, etc.).
   Verifies the deepcopy + conductivity-add at ``sources.py:226-238``.

TestHertzianDipoleInit
^^^^^^^^^^^^^^^^^^^^^^

``test_dl_default_is_zero``
   Asserts ``HertzianDipole().dl == 0.0``. Source: ``sources.py:247-249``.

TestHertzianDipoleCalculateWaveformValues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_only_halfdt_array_populated``
   Asserts ``waveformvalues_halfdt`` is filled with the constant
   waveform value and ``waveformvalues_wholedt`` stays ``None``.
   Hertzian dipoles use only half-step values. Source: ``sources.py:251-283``.

TestHertzianDipoleUpdateElectric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_decrements_E_by_documented_formula`` (parametrised over x/y/z)
   With ``dl=2``, ``coeff=1``, unit spacing, ``waveform=3``, asserts
   ``E[1,1,1] == -6.0`` after one update (``-1 * 3 * 2 / 1 = -6``).
   Source: ``sources.py:303-325``. Failure indicates either the
   ``dl`` factor was dropped or the per-cell volume normalisation
   changed.

TestMagneticDipoleCalculateWaveformValues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_only_wholedt_array_populated``
   Symmetric to the Hertzian half-dt test: ``waveformvalues_wholedt``
   filled, ``waveformvalues_halfdt`` stays ``None``. Magnetic dipoles
   use whole-step values. Source: ``sources.py:331-361``.

TestMagneticDipoleUpdateMagnetic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_decrements_H_by_documented_formula`` (parametrised over x/y/z)
   With ``coeff=1``, unit spacing, ``waveform=7``, asserts
   ``H[1,1,1] == -7.0``. Source: ``sources.py:382-401``.

TestHtodSrcArraysCpuBug
^^^^^^^^^^^^^^^^^^^^^^^

``test_cpu_solver_raises_unbound_local``
   **Tripwire for the missing CPU branch at ``sources.py:404-473``.**
   The function builds ``srcinfo1``/``srcinfo2``/``srcwaves`` then
   only assigns the ``_dev`` locals inside ``cuda``/``opencl``/
   ``metal`` branches. For ``solver="cpu"`` the final ``return``
   accesses unbound locals. When fixed (CPU branch added that returns
   the host numpy arrays), this test must flip to assert the returned
   arrays.

TestHtodSrcArraysVoltageSourceDeadCodeBug
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_hard_voltage_source_srcwaves_currently_uses_halfdt``
   **Tripwire for the dead-code override at ``sources.py:448-449``.**
   Sets ``solver="cuda"``, injects a fake ``pycuda.gpuarray`` that
   returns its input untouched, and creates a hard ``VoltageSource``
   (``resistance=0``) with distinguishable ``waveformvalues_halfdt``
   and ``waveformvalues_wholedt``. Asserts the returned ``srcwaves[0]``
   equals ``waveformvalues_halfdt`` — the *intended* hard-source path
   at ``sources.py:446-447`` writes ``waveformvalues_wholedt``, but
   lines 448-449 unconditionally overwrite. When fixed (the two
   redundant lines deleted), this test must assert the wholedt array.

TestTransmissionLineInit
^^^^^^^^^^^^^^^^^^^^^^^^

``test_dl_equals_sqrt3_c_dt``
   Asserts ``dl == √3 · c · dt``. Source: ``sources.py:498``. The
   ``√3`` factor avoids the FDTD "magic time step" instabilities for
   certain impedances.

``test_nl_equals_round_two_thirds_iterations``
   With ``iterations=1000``, asserts ``nl == 667``. Source:
   ``sources.py:502``. The 1D line is initially long so the incident
   voltage and current can be computed without reflection contamination.

``test_voltage_and_current_arrays_sized_nl``
   Asserts ``voltage`` and ``current`` have shape ``(nl,)``. Source:
   ``sources.py:510-511``.

``test_incident_arrays_sized_iterations_plus_one``
   Asserts ``Vinc``, ``Iinc``, ``Vtotal``, ``Itotal`` all have shape
   ``(iterations + 1,)``. Source: ``sources.py:512-515``.

``test_default_positions``
   Asserts ``srcpos == 5``, ``antpos == 10``, ``abcv0 == 0``,
   ``abcv1 == 0``. Source: ``sources.py:493-508``. These define the
   relative positions of the one-way injector excitation and the
   antenna port along the 1D line.

TestTransmissionLineCalculateWaveformValues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_populates_both_arrays``
   Asserts both ``waveformvalues_wholedt`` and ``waveformvalues_halfdt``
   are filled. Transmission lines use both — wholedt for the
   ``update_voltage`` injection at line 603, halfdt for the
   ``update_current`` injection at line 626. Source: ``sources.py:517-554``.

TestTransmissionLineUpdateABC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_abc_coefficient_matches_closed_form``
   With seeded ``voltage[0]=0``, ``voltage[1]=1``, ``abcv0=0.5``,
   ``abcv1=0.25``, asserts ``voltage[0] == h·(1-0.5) + 0.25`` where
   ``h = (c·dt - dl) / (c·dt + dl)``, and that ``abcv0``/``abcv1``
   advance to the previous ``voltage[0]``/``voltage[1]``. Source:
   ``sources.py:574-585``. Failure means the Mur 1st-order ABC has
   drifted; the line will radiate energy back from its end.

TestTransmissionLineCalculateIncidentVI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_shortens_nl_to_antpos_plus_one``
   Asserts ``nl == antpos + 1 == 11`` after
   ``calculate_incident_V_I``. Source: ``sources.py:572``. After the
   incident V/I is recorded, the line is shortened so it does not
   contaminate the main-grid simulation.

TestTransmissionLineUpdateElectric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_sets_E_from_voltage`` (parametrised over x/y/z)
   With ``voltage[antpos] = 0.4`` and unit spacing, asserts
   ``E[1,1,1] == -0.4``. Verifies the assignment at
   ``sources.py:653-660``. Note: ``update_electric`` first calls
   ``update_voltage`` — the test pre-seeds the line so that call
   leaves ``voltage[antpos]`` unchanged.

TestTransmissionLineUpdateMagnetic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_pulls_current_from_grid_calculate_I`` (parametrised over x/y/z)
   Stubs ``G.calculate_Ix`` / ``Iy`` / ``Iz`` to return a known value
   per polarisation; asserts ``current[antpos]`` matches after the
   update call. Source: ``sources.py:682-689``.

TestDiscretePlaneWaveInit
^^^^^^^^^^^^^^^^^^^^^^^^^

``test_m_array_shape_and_dtype``
   Asserts ``m.shape == (4,)`` and ``m.dtype == np.int32``. The
   fourth element stores ``max(|m_x|, |m_y|, |m_z|)``. Source:
   ``sources.py:730``.

``test_origin_array_shape_and_zero_default``
   Asserts ``origin.shape == (3,)``, ``int32``, all-zero. Source:
   ``sources.py:731-734``.

``test_projections_array_is_float64_length_6``
   Asserts ``projections.shape == (6,)`` and dtype is ``float64``.
   The projections store direction cosines + impedance-normalised
   counterparts used during plane-wave sourcing. Source:
   ``sources.py:737``.

``test_default_scalar_fields``
   Asserts ``materialID == 1`` (default background), ``pml_cells == 20``,
   ``buffercells_axial == 5``, ``axial == 0``, ``speed == c``. Source:
   ``sources.py:739-751``.

TestFindDpwIntegersOptimized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Verifies the pure-math continued-fractions integer solver at
``sources.py:1968-2131``.

``test_propagation_along_plus_x_axis``
   With ``theta=90``, ``phi=0`` (along +x), asserts the returned
   ``m_vec`` has magnitudes ``[1, 0, 0]`` and total error is ``≈ 0``.
   Sign is left free because the algorithm chooses by quadrant.

``test_propagation_along_plus_z_axis``
   With ``theta=0`` (along +z), asserts magnitudes ``[0, 0, 1]`` and
   total error ``≈ 0``.

``test_negative_tolerance_returns_none``
   With ``max_total_error_deg = -1.0`` (unsatisfiable since errors are
   always non-negative), asserts ``m_vec is None`` and all returned
   angles / errors are ``NaN``. Verifies the empty-candidate guard at
   ``sources.py:2095-2097``.

``test_oblique_uniform_grid_produces_small_integers``
   With ``theta=45``, ``phi=45`` on a uniform grid and a 1° tolerance,
   asserts ``max(|m_i|) < 10`` and total error ``≤ 1°``. Verifies the
   "smallest valid" selection at ``sources.py:2099-2104``.

TestDpwCalculateWaveformValuesNonCython
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_array_shapes``
   With ``cythonize=False`` and ``iterations=3``, ``m=[1,1,1,1]``,
   asserts both arrays have shape ``(4, 3, 1)``. Verifies the pure-Python
   fallback at ``sources.py:1194-1233``.

``test_zero_outside_window``
   With ``start=0``, ``stop=0.5`` and ``dt=10``, asserts iteration 1's
   waveform values are zero (the corresponding ``time1`` / ``time2``
   land outside the window). Verifies the ``time >= start and time <= stop``
   guards at ``sources.py:1207`` and ``1226``.

TestDpwNonCythonTypoBug
^^^^^^^^^^^^^^^^^^^^^^^

``test_time2_offset_uses_doubled_same_index``
   **Tripwire for the duplicated-index typo at ``sources.py:1221``.**
   The ``time2`` calculation reads ``np.abs(self.m[(dimension)]) +
   np.abs(self.m[(dimension)])`` — the same index twice. By symmetry
   with ``time1`` (which uses ``m[(dim+1)%3]`` and ``m[(dim+2)%3]``),
   this is almost certainly meant to be the cross-axis sum. The test
   confirms iteration 1 / dim 0 / r 0 lands a nonzero value with the
   current formula. When the source is fixed, this test must be
   updated to reflect the new formula's window behaviour.

Test Catalog — ``receivers.py``
-------------------------------

TestRxAllowableOutputs
^^^^^^^^^^^^^^^^^^^^^^

``test_allowableoutputs_lists_nine_components_in_order``
   Asserts ``Rx.allowableoutputs`` is exactly
   ``["Ex","Ey","Ez","Hx","Hy","Hz","Ix","Iy","Iz"]``. This list is
   part of the public contract — the HDF5 writer and the GPU output
   kernel index into it. Source: ``receivers.py:31``.

``test_defaultoutputs_is_first_six``
   Asserts ``defaultoutputs == ["Ex","Ey","Ez","Hx","Hy","Hz"]``. The
   trailing currents only make sense for transmission-line attachments.
   Source: ``receivers.py:32``.

``test_allowableoutputs_dev_matches_defaultoutputs``
   Asserts ``allowableoutputs_dev == defaultoutputs`` — both are the
   first 6 entries of ``allowableoutputs``. Documents the dup
   (probably a refactor leftover); failure would mean one slice
   changed without the other.

TestRxDefaults
^^^^^^^^^^^^^^

``test_outputs_starts_as_empty_dict``
   Asserts ``Rx().outputs == {}``. Caller populates this after
   construction. Source: ``receivers.py:37``.

``test_coord_array_is_zero_int32``
   Asserts ``coord.shape == (3,)``, ``dtype == np.int32``, all-zero.
   The solver indexes 4D field arrays with these; non-int32 silently
   coerces. Source: ``receivers.py:38``.

``test_coordorigin_array_is_zero_int32``
   Same for ``coordorigin`` (sub-grid offset). Source: ``receivers.py:39``.

``test_outputs_is_per_instance_not_shared``
   Two ``Rx()`` instances must have independent ``outputs`` dicts.
   Defends against a regression to a class-level mutable default,
   which would silently couple every receiver's recorded data.

TestRxCoordProperties
^^^^^^^^^^^^^^^^^^^^^

``test_coord_property_round_trips`` (parametrised over x/y/z)
   Asserts ``setattr(rx, "xcoord", 12)`` writes to ``coord[0]`` and
   the getter reads it back. Source: ``receivers.py:41-63``.

``test_coordorigin_property_round_trips`` (parametrised over x/y/z)
   Same for the origin coordinate. Source: ``receivers.py:65-87``.

``test_coord_setter_preserves_int32_dtype``
   After ``rx.xcoord = 5``, asserts ``coord.dtype`` is still ``int32``.
   The setter writes into the existing array in place; a regression to
   ``rx.coord = np.array([5, 0, 0])`` would lose the dtype.

TestHtodRxArraysCpuBug
^^^^^^^^^^^^^^^^^^^^^^

``test_cpu_solver_raises_unbound_local``
   **Tripwire for the missing CPU branch at ``receivers.py:90-140``.**
   Mirrors the analogous ``htod_src_arrays`` bug. Function only
   assigns ``rxcoords_dev`` / ``rxs_dev`` inside ``cuda``/``opencl``/
   ``metal`` branches, so on CPU the final ``return`` accesses
   unbound locals. When fixed (CPU branch added), this test must flip.

TestHtodRxArraysCuda
^^^^^^^^^^^^^^^^^^^^

``test_returns_arrays_with_documented_shapes``
   With ``solver="cuda"`` and a fake ``pycuda.gpuarray``, asserts the
   returned arrays have shapes ``(n_rxs, 3)`` (int32) and
   ``(6, iterations, n_rxs)`` (float64), and the rxs array starts
   zero-filled. Source: ``receivers.py:104-117``. Failure indicates the
   shape contract has drifted; the GPU kernels are hard-coded to these
   shapes.

``test_packs_receiver_coords_in_declaration_order``
   With three receivers, asserts ``rxcoords_dev`` rows appear in the
   same order as ``G.rxs``. Verifies the loop at
   ``receivers.py:106-109``.

TestDtohRxArrayHostPath
^^^^^^^^^^^^^^^^^^^^^^^

The non-Metal branch at ``receivers.py:199-213`` currently only works
when both ``rxs_dev`` and ``rxcoords_dev`` are already host numpy
arrays (the ``.get()`` calls needed to materialise CUDA/OpenCL
gpuarrays are commented out at lines 200-201). Tests here exercise
the host-array case directly.

``test_copies_requested_outputs_into_rx_outputs``
   With two receivers each requesting ``Ex``, seeds ``rxs_dev`` with
   distinguishable time series and asserts each ``rx.outputs["Ex"]``
   ends up with the right slice. Source: ``receivers.py:210-213``.

``test_copies_multiple_outputs_for_one_rx``
   One receiver requesting both ``Ex`` and ``Hy``; asserts both
   outputs are copied independently.

``test_skips_rxs_whose_coords_do_not_match_any_row``
   With one receiver whose coords appear in ``rxcoords_dev`` and one
   whose coords don't appear in any row, asserts the matched receiver
   gets its data and the unmatched receiver's outputs are unchanged
   (a 77.0 sentinel survives). Verifies the coordinate-equality guard
   at ``receivers.py:205-208``.

TestDtohRxArrayLoopBoundBug
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_more_rxs_than_rxcoords_raises_indexerror``
   **Tripwire for the inner-loop bound at ``receivers.py:204``.** The
   inner loop is ``for rxd in range(len(G.rxs))`` but it indexes
   ``rxcoords_dev[rxd, ...]``. If ``len(G.rxs) > len(rxcoords_dev)``
   (e.g. a receiver dropped during MPI domain decomposition) the line
   raises ``IndexError``. Correct fix would be
   ``range(len(rxcoords_dev))`` or an explicit guard. When fixed, this
   test should assert the no-op behaviour instead.

Running
-------

From the repository root, with the project installed in editable mode
(``pip install -e .``)::

    python -m pytest tests/unit/ -v

Filter to just this PR's suites::

    python -m pytest tests/unit/sources/ tests/unit/receivers/ -v

Run a single test::

    python -m pytest tests/unit/sources/test_sources.py::TestVoltageSourceUpdateElectric::test_hard_source_overwrites_E_with_negated_waveform -v

Stop on first failure (useful while iterating)::

    python -m pytest tests/unit/ -x