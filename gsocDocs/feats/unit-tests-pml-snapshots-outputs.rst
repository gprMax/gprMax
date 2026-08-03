Unit Tests — PML, Snapshots and Outputs
=======================================

**Branch:** ``feat/ssingh-pml-snapshots-outputs``

**Modules under test:**
   - ``gprMax/pml.py`` — ``CFSParameter``, ``CFS`` and ``PML``: the absorption
     gradient, the two RIPML formulations' update coefficients, the Cython
     dispatch, and ``MPIPML``
   - ``gprMax/snapshots.py`` — ``Snapshot`` and ``MPISnapshot``: field volumes
     frozen at one iteration, both output formats, and the device-transfer
     helpers
   - ``gprMax/fields_outputs.py`` — receiver traces, transmission-line
     voltages and currents, and the ``.h5`` output writer
   - ``gprMax/geometry_outputs/grid_view.py`` — ``GridView`` and
     ``MPIGridView``, the rectangular window every exporter looks through
   - ``gprMax/geometry_outputs/geometry_views.py`` — ``GeometryView``,
     ``Metadata`` and ``MPIMetadata``
   - ``gprMax/geometry_outputs/geometry_view_voxels.py`` —
     ``GeometryViewVoxels``, the per-cell VTK ImageData exporter
   - ``gprMax/geometry_outputs/geometry_view_lines.py`` —
     ``GeometryViewLines``, the per-cell-edge VTK UnstructuredGrid exporter
   - ``gprMax/geometry_outputs/geometry_objects.py`` — ``GeometryObject``, the
     raw-array exporter for geometry reuse
   - ``gprMax/geometry_outputs/geometry_objects_read.py`` —
     ``ReadGeometryObject``, the matching reader

**Covered transitively:**
   - ``gprMax/cython/pml_build.pyx`` ``pml_average_er_mr`` /
     ``pml_sum_er_mr`` — driven as real OpenMP kernels against real arrays
   - ``gprMax/cython/snapshots.pyx`` ``calculate_snapshot_fields`` — the Yee
     interpolation stencil, driven for real
   - ``gprMax/cython/geometry_outputs.pyx`` ``get_line_properties`` — the
     point-ID walk, hand-checked on 1×1×1 and 2×1×1 grids
   - ``gprMax/vtkhdf_filehandlers/`` ``VtkImageData`` and
     ``VtkUnstructuredGrid`` — every ``write_vtk`` test writes a real file and
     reads it back; the package itself is tested directly in a later PR
   - ``gprMax/cython/pml_updates_{electric,magnetic}_{HORIPML,MRIPML}`` — not
     executed, but every function name the dispatch convention can produce is
     resolved against the compiled extensions

**Test files:**
   - ``tests/unit/pml/test_cfs.py`` (80 tests)
   - ``tests/unit/pml/test_pml_construction.py`` (65 tests)
   - ``tests/unit/pml/test_pml_coeffs.py`` (71 tests)
   - ``tests/unit/pml/test_pml_updates.py`` (81 tests)
   - ``tests/unit/pml/test_pml_build_kernel.py`` (24 tests)
   - ``tests/unit/pml/test_mpi_pml.py`` (23 tests)
   - ``tests/unit/outputs/test_grid_view.py`` (91 tests)
   - ``tests/unit/outputs/test_mpi_grid_view.py`` (62 tests)
   - ``tests/unit/outputs/test_snapshots.py`` (67 tests)
   - ``tests/unit/outputs/test_snapshot_files.py`` (43 tests)
   - ``tests/unit/outputs/test_snapshot_devices.py`` (45 tests)
   - ``tests/unit/outputs/test_fields_outputs.py`` (76 tests)
   - ``tests/unit/outputs/test_geometry_views.py`` (55 tests)
   - ``tests/unit/outputs/test_geometry_view_voxels.py`` (26 tests)
   - ``tests/unit/outputs/test_geometry_view_lines.py`` (40 tests)
   - ``tests/unit/outputs/test_geometry_objects.py`` (36 tests)
   - ``tests/unit/outputs/test_geometry_objects_read.py`` (38 tests)

**Total: 923 tests** from 662 test functions across 121
classes, all passing, **no** ``xfail``. Three tests skip where ``h5py`` is
built without MPI support; see *Deliberately Untested Paths*.

**Shared fixtures:** ``tests/unit/pml/conftest.py``,
``tests/unit/outputs/conftest.py``

Scope
-----

Nine PRs have tested how a gprMax model is *described* and *built* — waveforms,
materials, sources and receivers, the hash parser, user objects, geometry
primitives, fractals, and the grid itself. None of them tested what happens at
the **edges** of the domain, or what leaves the simulation as a **file**.

This suite covers both, and they turn out to share a shape: each fails
silently, plausibly, and visibly only to a domain expert. A PML whose
coefficients came out zero does not crash — it stops absorbing, and the
radargram grows echoes off the domain wall that look like real reflectors. A
snapshot read with the wrong bound convention does not crash — it shifts every
field half a cell. A geometry view whose material table is misindexed does not
crash — it renders the model in the wrong materials.

Nearly all of it is exactly assertable. The PML coefficients are closed-form
algebra in ``e0``, ``dt`` and three profile arrays. The view sizing is one
``ceil``. The snapshot interpolation is a fixed four-versus-two averaging
ratio. The file layouts are fixed dataset names that every downstream tool
depends on, and a written file can simply be read back.

Two directories rather than three: ``tests/unit/pml/`` is self-contained, while
``tests/unit/outputs/`` holds snapshots, field outputs and the whole
``geometry_outputs`` package because all three compose the same ``GridView``.
Splitting them would mean building that fixture in one suite and importing it
into another.

Diagnosing a Failure
--------------------

The traps that cost the most time are cross-cutting. Check these before
reading the per-test entries.

**A whole file errors at collection.** Wrong interpreter. The suite needs the
``gprMax`` conda environment — the base environment has no ``cython``, and
``gprMax-devel`` has no ``pytest``. ``python -m pytest`` from the wrong prompt
fails at ``import gprMax.config``.

**Every test in a directory fails on a config attribute.** New source code
reads a ``config`` key that the autouse ``pml_config`` or ``outputs_config``
fixture does not supply. ``config.sim_config`` is ``None`` until a real run
initialises it, so the fixture is mandatory rather than convenient. Add the
key to the fixture; do not weaken the assertion. The complete surface each
suite needs is listed in *Test Infrastructure*.

**The whole process aborts with "Attempting to use an MPI routine before
initializing MPI".** ``MPI4PY_RC_INITIALIZE=0`` is set in the environment. The
MPI tests construct genuine communicators, which requires mpi4py to have
initialised on import. This PR removed that variable from
``.github/workflows/tests.yml`` for exactly this reason;
``MPI4PY_RC_FINALIZE=0`` is retained and is harmless.

**A test hangs rather than failing.** An MPI collective was entered without a
matching call. With the real single-rank ``COMM_SELF`` this cannot happen, but
a hand-written communicator double that never returns from ``Ibcast``, or a
code path that enters a collective on some ranks only, will block forever.

**A VTK value comparison fails by what looks like a transpose.** It is one.
VTKHDF stores datasets ZYX-major, so every ``write_vtk`` path transposes on the
way out while the plain ``.h5`` writers do not. Compare against ``array.T`` for
VTK output and against ``array`` for HDF5 output.

**A device-array size is wrong, and the number looks like another test's.**
``Snapshot.nx_max``/``ny_max``/``nz_max`` are mutable **class** attributes that
only ever grow. The outputs conftest restores them after every test; without
that, one test silently enlarges every later allocation.

**A materials lookup raises ``KeyError`` or ``IndexError``.** ``FDTDGrid``
initialises both ``ID`` and ``solid`` to **1**, not 0 — free space in a full
model. A test grid that defines only material 0 must set both arrays, or
``np.unique(ID)`` indexes past the end of the materials list.

**Fixing a known defect turns no test red.** That is by design. No test in this
suite asserts broken behaviour and none is marked ``xfail``; where a defect
made a contract untestable the test was omitted. The twelve affected paths are
listed under *Deliberately Untested Paths*, and each has a write-up carrying
the tests its fix should add.

The PML Gradient
----------------

A PML is a band of cells around the domain edge whose job is to be a wall that
does not look like one. The absorption has to be **ramped in gradually** — a
sudden jump from vacuum to very absorbing is itself an impedance step, and an
impedance step is a mirror. So ``pml.py`` is almost entirely the arithmetic of
that ramp.

Three parameters shape it, together forming a **CFS** (Complex Frequency
Shifted) set:

``sigma``
   Conductivity — the actual absorption. Quartic ramp by default, with its
   maximum derived rather than given.

``kappa``
   Stretches real distance, improving absorption at grazing angles. Constant 1
   by default, which is off.

``alpha``
   Shifts the pole off the real axis, suppressing slow late-time drift.
   Constant 0 by default, which is off.

Out of the box only ``sigma`` does anything, and its maximum follows a
published closed form::

   sigma_max = 0.8 · (m + 1) / (z0 · d · sqrt(er · mr))

where ``m`` is the polynomial order read from the *profile name*
(``"quartic"`` → 4), ``d`` the cell size along the slab's normal, ``z0`` the
impedance of free space, and ``er``/``mr`` the average material properties
behind the slab — computed by the OpenMP kernel ``pml_average_er_mr``.

Two details govern most of the assertions in the PML catalog.

**The ramp is sampled twice per cell.** ``calculate_values`` allocates
``thickness + 1`` samples, hands them to ``scaling_polynomial`` which builds one
``linspace`` of ``2n`` points, splits the even entries to the electric profile
and the odd entries to the magnetic one, then drops the final sample. For
``thickness == 4`` and a linear profile that leaves E at ``0, ¼, ½, ¾`` and H
at ``⅛, ⅜, ⅝, ⅞`` — H trailing E by half a cell, which is the Yee stagger.

**Reversal is not symmetric.** For a slab on the far side of the domain both
arrays are reversed, and the magnetic one is then rolled one element left. That
roll restores the half-cell offset which reversal breaks. It is six lines of
numpy and completely opaque unless someone writes it down.

Finally, ``HORIPML`` and ``MRIPML`` are **not two classes**. They are two
strings in ``G.pmls["formulation"]`` selecting two sets of coefficient formulas
inside one ``PML`` class, and also selecting which compiled Cython module is
imported at update time. Both reduce the three profiles to eight coefficient
arrays ``ERA``…``HRF``, each shaped ``(len(CFS), thickness)``.

The Grid View
-------------

Everything in the outputs half looks at the grid through the same window. A
``GridView`` is a start, a stop and a stride; given those it answers what shape
the region is and hands over slices of the grid's arrays. ``Snapshot``, every
``GeometryView`` and ``GeometryObject`` each *hold* one and delegate all their
coordinate arithmetic to it.

Three of its properties account for most of what can go wrong.

**Size is a ceiling.**

.. code-block:: python

   size = np.ceil((stop - start) / step).astype(np.int32)

A view from 0 to 10 with step 3 spans **four** cells, not three — the final
partial cell is kept. Substituting integer division silently shortens every
exported array by one in every axis on any non-dividing view.

**There are two slice families, and they mean different things.**
``getter_slice``/``setter_slice`` index the *grid's* arrays in grid
coordinates. ``get_output_slice``/``get_read_slice`` index the *view's* own
output buffer, always starting at zero. In the serial class the members of each
pair are literally the same function; only ``MPIGridView`` makes them diverge.
The equivalences are asserted in ``test_grid_view.py`` precisely so the MPI
overrides have a baseline to be measured against.

**``upper_bound_exclusive`` decides whether one extra node is fetched.**
Cell-centred arrays (``solid``, ``rigidE``, ``rigidH``) are read exclusively;
node-centred ones (``ID``) and all six field arrays inclusively, which reaches
one whole *step* beyond ``stop``. The snapshot kernel needs that extra node
because it averages across it. Getting this backwards does not crash; it shifts
every exported field half a cell.

Under MPI the same class trims each rank's view to the part that rank owns,
keeping the sample points aligned to the stride through a modulo, and records
an ``offset`` saying where the local block belongs in the global dataset. The
``global_*`` attributes describe the view as requested; the plain ones describe
this rank's share.

Test Infrastructure
-------------------

``tests/unit/pml/conftest.py``
   **``pml_config`` (autouse)** patches exactly four configuration keys, which
   is the module's entire surface: ``em_consts["e0"]``, ``em_consts["z0"]``,
   ``dtypes["float_or_double"]`` and ``ompthreads``. The electromagnetic
   constants are the real ones from ``scipy.constants``, so the closed-form
   coefficient assertions check gprMax's algebra rather than a made-up ``e0``.

   **``make_cfs``** builds a ``CFS`` with any of its three parameters
   overridden. The stock defaults switch ``alpha`` and ``kappa`` off, so a test
   that needs those terms to contribute has to say so.

   **``make_pml_grid``** builds a real ``FDTDGrid`` with ``dl``, ``dt``,
   materials, field and update-coefficient arrays, and a default ``[CFS()]``.
   That default is not optional: ``PML.__init__`` runs ``check_kappamin()``,
   which rejects an empty CFS list, so a grid without one cannot construct a
   slab at all. ``FDTDGrid.build()`` installs the same default in production.

   **``make_pml``** constructs a slab on any of the six faces, sized along its
   own normal and spanning the whole face on the other two axes — the same
   arrangement ``FDTDGrid._construct_pml`` produces.

``tests/unit/outputs/conftest.py``
   **``outputs_config`` (autouse)** covers a wider surface: ``dtypes``,
   ``general["progressbars"]``, ``general["solver"]``, ``input_file_path``,
   ``output_file_path``, ``appendmodelnumber``, ``ompthreads``, ``device`` and
   ``set_snapshots_dir()``. Both path settings point into ``tmp_path``, so any
   test that lets a filename be derived writes somewhere harmless.

   **``reset_snapshot_class_state`` (autouse)** restores
   ``Snapshot.nx_max``/``ny_max``/``nz_max``/``bpg`` after every test. Needing
   this fixture is itself a finding — production has no equivalent reset, so
   those maxima grow across every model in a run.

   **``make_view_grid``** builds a real ``FDTDGrid`` with geometry and field
   arrays and a materials list. The field arrays are filled with a
   distinct-per-cell ramp by default, so a slicing assertion can name exactly
   which cells were read.

   **``make_mpi_grid``** is the important one. ``MPIGridView.__init__``
   contains ``assert isinstance(comm, MPI.Intracomm)``, so a mock communicator
   is rejected outright. The fixture therefore supplies a **real**
   ``MPI.COMM_SELF`` and fakes only the grid's own methods —
   ``local_to_global_coordinate``, ``get_grid_coord_from_coordinate``,
   ``negative_halo_offset`` and ``size``. Because the halo-clamping arithmetic
   depends on those and not on rank count, setting a halo offset makes a
   one-rank view behave exactly as a mid-domain rank would, and every branch
   becomes reachable.

   **``read_h5``** reopens a written file and returns ``(attrs, datasets)`` as
   plain dicts with group paths flattened, so round-trip assertions read as
   data comparisons rather than h5py mechanics.

   **``make_rx``**, **``make_tl``** and **``null_pbar``** supply a named
   receiver, a transmission-line stand-in, and a progress bar that records the
   byte counts pushed to it. Every receiver is given an explicit ID, because
   ``Rx.__init__`` only annotates ``self.ID`` and never assigns it.

Both suites reuse ``nonzero_set`` from the geometry-primitives, fractals and
grid suites: the set of index tuples at which an array is non-zero, which is
how every "which cells were written" assertion is expressed.

Test Catalog — ``test_cfs.py``
------------------------------

**80 tests** from 43 test functions across 9 classes.

``CFSParameter`` and ``CFS`` — the PML absorption gradient.

The PML works by ramping absorption in gradually: a hard step from vacuum to
"very absorbing" is itself an impedance discontinuity, and an impedance
discontinuity is a mirror. Everything in this file is the arithmetic of that
ramp, and all of it is closed-form.

Two facts govern the expected values throughout, and both are worth stating
once here rather than repeating in thirty docstrings.

**The profile is sampled twice per cell.** ``calculate_values`` allocates
``thickness + 1`` samples, hands them to ``scaling_polynomial`` which builds
a single ``linspace`` of ``2n`` points, splits the even entries to the
electric profile and the odd entries to the magnetic one, then drops the
final sample. For ``thickness == 4`` and a linear profile that leaves E at
``0, ¼, ½, ¾`` and H at ``⅛, ⅜, ⅝, ⅞`` — H trailing E by half a cell, which
is exactly the Yee stagger.

**Reversal is not symmetric.** For a slab on the far side of the domain both
arrays are reversed, and the magnetic one is then rolled one element left.
That roll is the half-cell offset reasserting itself: reverse an E sample
and it lands where an E sample belongs, reverse an H sample and it lands
half a cell out.

TestCFSParameterDefaults
^^^^^^^^^^^^^^^^^^^^^^^^

``test_all_arguments_default``
   Expects a bare ``CFSParameter`` to be an inert, unnamed, zero-valued
   polynomial parameter with no profile chosen.

``test_each_argument_is_stored_verbatim``
   Expects every constructor argument to land on the attribute of the same
   name, unmodified. (6 parameter sets)

TestScalingProfileTable
^^^^^^^^^^^^^^^^^^^^^^^

``test_there_are_exactly_nine_profiles``
   Expects ``scalingprofiles`` to hold nine entries — a change here changes
   what every profile name means.

``test_name_maps_to_its_polynomial_order``
   Expects ``"linear" -> 1``, ``"quartic" -> 4``, and so on: the order is
   the position of the name in the sequence, starting at ``"constant" ->
   0``. (9 parameter sets)

``test_scaling_directions_are_forward_and_reverse``
   Expects exactly two directions, in that order.

TestCFSDefaults
^^^^^^^^^^^^^^^

The stock CFS: sigma does the absorbing, kappa and alpha are off.

``test_alpha_is_a_constant_zero``
   Expects ``alpha`` constant with ``min == max == 0`` — the frequency-shift
   term is switched off by default.

``test_kappa_is_a_constant_one``
   Expects ``kappa`` constant with ``min == max == 1`` — a stretch factor of
   one is no stretching, so it too is off.

``test_sigma_is_a_quartic_ramp_with_max_unset``
   Expects ``sigma`` quartic, ``min == 0`` and ``max is None`` — the
   ``None`` is the sentinel that makes ``calculate_update_coeffs`` derive
   the optimum from the underlying material.

``test_two_instances_do_not_share_parameters``
   Expects each ``CFS`` to own its three parameters — a shared
   ``CFSParameter`` would let one PML slab's auto-computed ``sigma.max``
   leak into another's.

TestCalculateSigmamax
^^^^^^^^^^^^^^^^^^^^^

``sigma_max = 0.8·(m+1) / (z0·d·sqrt(er·mr))``.

``test_matches_the_closed_form_for_the_default_quartic``
   Expects the published optimum for a quartic profile in free space:
   ``0.8·5 / (z0·d)``.

``test_numerator_follows_the_profile_order``
   Expects the ``(m + 1)`` factor to come from ``sigma``'s own profile name,
   so the name-to-order table is pinned end to end. (9 parameter sets)

``test_inversely_proportional_to_cell_size``
   Expects halving ``d`` to double ``sigma_max`` — a finer PML needs a
   steeper ramp to absorb as much over a shorter distance.

``test_scales_with_one_over_root_er_mr``
   Expects a PML backing ``er = 4`` to need half the conductivity of one
   backing free space, since ``sqrt(4·1) == 2``.

``test_er_and_mr_enter_symmetrically``
   Expects ``(er=4, mr=1)`` and ``(er=1, mr=4)`` to give the same answer —
   they appear only as the product under the root.

``test_writes_through_to_the_parameter``
   Expects the result to be stored on ``sigma.max`` rather than returned —
   the caller relies on the mutation.

TestScalingPolynomial
^^^^^^^^^^^^^^^^^^^^^

The interleaved ``linspace``: even samples to E, odd samples to H.

``test_matches_an_independently_written_formula``
   Expects agreement with a longhand reimplementation of the
   ``linspace``/stride construction. (3 parameter sets)

``test_linear_profile_samples_e_at_whole_cells``
   Expects ``0, ¼, ½, ¾, 1`` for a linear ramp over five samples — the
   electric profile sits on cell boundaries.

``test_linear_profile_samples_h_half_a_cell_later``
   Expects ``⅛, ⅜, ⅝, ⅞, 1⅛`` — the magnetic profile trails the electric one
   by half a cell, which is the Yee stagger.

``test_the_two_profiles_interleave``
   Expects every H sample to fall strictly between its neighbouring E
   samples — the defining property of a staggered pair.

``test_order_zero_is_flat``
   Expects a constant profile to raise everything to the zeroth power,
   giving all ones — including the ``0 ** 0 == 1`` sample at the origin.

``test_higher_orders_ramp_later``
   Expects a steeper polynomial to hold the profile nearer zero in the
   interior — a quartic reaches ½ much later than a linear does. (4
   parameter sets)

``test_returns_new_arrays_rather_than_filling_the_inputs``
   Expects the passed-in arrays to be untouched: the function returns
   replacements, and the arguments serve only to carry the length.

TestCalculateValuesLength
^^^^^^^^^^^^^^^^^^^^^^^^^

``test_output_length_equals_thickness``
   Expects one value per PML cell: the extra sample allocated to get the
   staggering right is dropped before returning. (5 parameter sets)

``test_uses_the_configured_float_dtype``
   Expects ``float64`` under the double-precision fixture — the arrays feed
   straight into Cython buffers typed by the same setting.

TestCalculateValuesConstant
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``constant`` profile short-circuits: it uses ``max`` directly and never
consults ``min``.

``test_default_kappa_is_all_ones``
   Expects ``[1, 1, 1, 1]`` — a stretch factor of one, i.e. off.

``test_default_alpha_is_all_zeros``
   Expects ``[0, 0, 0, 0]`` — the frequency shift is off.

``test_constant_takes_max_not_min``
   Expects every entry to equal ``max`` even when ``min`` differs — the
   constant branch fires before any min/max rescaling.

``test_electric_and_magnetic_agree``
   Expects both profiles identical: a flat ramp has no stagger to express.

``test_constant_profile_wins_over_the_scaling_field``
   Expects ``scalingprofile == "constant"`` to be checked first, so it
   applies whatever ``scaling`` says.

TestCalculateValuesPolynomial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The polynomial branch, then rescaling into ``[min, max]``.

``test_linear_kappa_spans_min_to_max``
   Expects ``min + (max-min)·t`` at the E sample points ``0, ¼, ½, ¾`` — for
   ``min=1, max=5`` that is ``1, 2, 3, 4``.

``test_linear_kappa_magnetic_profile_is_offset``
   Expects the H samples at ``⅛, ⅜, ⅝, ⅞`` rescaled the same way — ``1.5,
   2.5, 3.5, 4.5``.

``test_every_polynomial_profile_matches_the_longhand_formula``
   Expects agreement with an independently written reference for all eight
   non-constant profiles, rescaled into ``[0, 1]``. (8 parameter sets)

``test_starts_at_min``
   Expects the first electric sample to be exactly ``min``: the ramp begins
   at the inner face of the PML, where it must be invisible.

``test_is_monotonic``
   Expects absorption to increase strictly outward — a non-monotonic ramp
   would create an internal reflection.

``test_rescaling_is_selected_by_parameter_id``
   Expects the min/max pair to be looked up by ``parameter.ID``, so a
   parameter whose ID does not match any of the three is left on the raw
   ``[0, 1]`` profile.

``test_alpha_rescales_against_alpha``
   Expects ``alpha``'s own min/max to be used, not sigma's — the three
   branches must not cross-wire.

``test_sigma_rescales_against_sigma``
   Expects ``sigma``'s own min/max to be used, mirroring the alpha case from
   the other side.

TestCalculateValuesReverse
^^^^^^^^^^^^^^^^^^^^^^^^^^

Slabs on the far side of the domain ramp the other way.

``test_electric_profile_is_the_forward_one_reversed``
   Expects ``[1, ¾, ½, ¼]`` where forward gives ``[0, ¼, ½, ¾]`` — note the
   endpoints differ, because the dropped extra sample comes off the other
   end.

``test_magnetic_profile_is_rolled_one_element_left``
   Expects ``[⅞, ⅝, ⅜, ⅛]``. Reversing alone would give ``[1⅛, ⅞, ⅝, ⅜]``;
   the ``np.roll(-1)`` discards the out-of-range ``1⅛`` sample and restores
   the half-cell stagger.

``test_reverse_is_monotonically_decreasing``
   Expects absorption to increase toward index 0 instead of away from it —
   the mirror image of the forward case.

``test_h_still_falls_between_neighbouring_e_samples``
   Expects the stagger to survive reversal — this is precisely what the roll
   exists to guarantee, and dropping it would leave every H value outside
   its E interval.

``test_constant_profile_is_unaffected_by_reversal``
   Expects a flat ramp reversed to still be flat — reversal is a no-op on a
   constant, which makes it a useful control.

``test_forward_and_reverse_electric_profiles_are_mirror_images``
   Expects ``reverse(thickness+1 samples)[:-1]`` rather than
   ``forward[::-1]`` — the truncation happens after the reversal, so the two
   are not simply each other's ``[::-1]``.

When these fail
~~~~~~~~~~~~~~~

**A profile's values are off by half a cell.** ``scaling_polynomial``
returns one ``linspace`` of ``2n`` points and splits it: even entries to the
electric profile, odd to the magnetic. Swapping the two strides, or using
``tmp[0::2]`` instead of ``tmp[0:-1:2]``, shifts one of the two profiles by
half a cell. The interleaving test catches it; the length tests do not.

**A reversed profile's magnetic values are wrong.** After reversal the
magnetic array is rolled one element left. That is not symmetry — it
restores the Yee stagger, which reversal breaks. If
``test_h_still_falls_between_neighbouring_e_samples`` goes red, the roll has
been removed or its direction flipped.

**Everything in one parameter goes to zero.** The rescaling in
``calculate_values`` is selected by ``parameter.ID``, so renaming a
``CFSParameter``'s ID silently drops it onto the raw ``[0, 1]`` profile.
``test_rescaling_is_selected_by_parameter_id`` pins that behaviour.

**``sigma_max`` is out by a constant factor.** The ``(m + 1)`` term comes
from ``sigma``'s *own* ``scalingprofile`` name, not from the parameter being
calculated. Changing the default from ``"quartic"`` changes every derived
``sigma_max`` in the suite.

Test Catalog — ``test_pml_construction.py``
-------------------------------------------

**65 tests** from 38 test functions across 7 classes.

``PML`` construction, validation, field-array allocation and reporting.

A ``PML`` is one slab: a rectangular box of cells on one face of the domain,
plus the direction its absorption increases in. Everything the constructor
does is bookkeeping — pick the right cell size for the normal axis, work out
the thickness, validate the CFS list, allocate four auxiliary field arrays.

Two things are worth knowing before reading the assertions.

**The direction string is what selects the axis.** ``PML.d`` is the grid
spacing along the slab's normal and ``PML.thickness`` its extent along the
same axis, both chosen by ``direction[0]``. The anisotropic grid used
throughout makes an axis mix-up impossible to pass by luck: ``dx``, ``dy``
and ``dz`` are 1 mm, 2 mm and 4 mm, none a multiple of the others in a way
that could coincide.

**``check_kappamin`` runs before anything is allocated.** It sums
``kappa.min`` across the CFS list and rejects a total below one, so a grid
with no CFS terms cannot construct a PML at all. That is why every fixture
installs a default ``CFS()`` — exactly as ``FDTDGrid.build()`` does in
production.

TestClassTables
^^^^^^^^^^^^^^^

``test_two_formulations_are_available``
   Expects ``["HORIPML", "MRIPML"]`` — the two published RIPML variants,
   selected by string rather than by subclass.

``test_six_boundary_ids``
   Expects the six slab names in the order ``x0, y0, z0, xmax, ymax, zmax``
   — the same order ``FDTDGrid.set_pml_thickness`` writes its
   ``OrderedDict`` in.

``test_six_directions``
   Expects the three ``minus`` directions before the three ``plus`` ones,
   matching the boundary-ID ordering.

TestExtents
^^^^^^^^^^^

``test_stores_all_six_bounds``
   Expects the six extent arguments to land verbatim on the instance.

``test_cell_counts_are_the_extent_differences``
   Expects ``nx == xf - xs`` and likewise for y and z.

``test_defaults_give_an_empty_slab``
   Expects all six bounds to default to zero, so an argument-free slab has
   no cells at all.

``test_id_and_direction_are_stored``
   Expects ``ID`` and ``direction`` kept verbatim — the first names the slab
   in log output, the second selects the Cython kernel.

``test_grid_is_held_by_reference``
   Expects ``pml.G`` to be the same object, not a copy — the update methods
   pass the grid's live field arrays into Cython.

TestDirectionSelectsTheAxis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``d`` and ``thickness`` both follow ``direction[0]``.

``test_d_is_the_spacing_along_the_normal``
   Expects ``d`` to be ``dx`` for the two x slabs, ``dy`` for the two y
   slabs and ``dz`` for the two z slabs. The anisotropic 1/2/4 mm grid means
   reading the wrong axis cannot coincidentally match. (6 parameter sets)

``test_thickness_is_the_extent_along_the_normal``
   Expects ``thickness`` to equal the requested depth on every face, taken
   from the axis the slab is normal to rather than from the two it spans. (6
   parameter sets)

``test_thickness_ignores_the_spanning_axes``
   Expects a slab four cells deep in x but eleven wide in y and z to report
   ``thickness == 4``.

``test_both_x_directions_take_the_x_spacing``
   Expects only the first character of the direction to matter, so
   ``xminus`` and ``xplus`` behave alike here. (2 parameter sets)

TestCheckKappamin
^^^^^^^^^^^^^^^^^

The sum of ``kappa.min`` across all CFS terms must reach one.

``test_default_cfs_passes``
   Expects the stock ``CFS()`` (``kappa.min == 1``) to be accepted.

``test_empty_cfs_list_is_rejected``
   Expects ``ValueError``: an empty list sums to zero, so a grid with no CFS
   terms can never build a PML.

``test_kappamin_below_one_is_rejected``
   Expects ``ValueError`` for a single term with ``kappa.min == 0.5``.

``test_two_terms_summing_to_one_are_accepted``
   Expects two half-kappa terms to pass: the check is on the *sum* across
   the multi-pole list, not on each term individually.

``test_two_terms_summing_below_one_are_rejected``
   Expects ``ValueError`` for ``0.4 + 0.4`` — just under the limit.

``test_the_rejection_message_goes_only_to_the_log``
   Expects the explanatory text in the log record, not on the exception: the
   code calls ``logger.exception(...)`` and then ``raise ValueError`` with
   no argument, so ``str(exc)`` is empty. Assert on ``caplog``, never on the
   message.

``test_cfs_list_is_shared_with_the_grid``
   Expects ``pml.CFS`` to *be* ``G.pmls["cfs"]`` rather than a copy, so a
   ``sigma.max`` computed on one slab is visible to the next.

   Note the name: ``PML.CFS`` is a *list of* ``CFS`` instances, while
   ``CFS`` is the class. ``isinstance(pml.CFS, CFS)`` reads plausibly and is
   always ``False``.

TestInitialiseFieldArrays
^^^^^^^^^^^^^^^^^^^^^^^^^

Four auxiliary arrays per slab, shaped by the slab's own normal.

``test_x_direction_shapes``
   Expects, for a slab of ``(nx, ny, nz)`` cells normal to x: ``EPhi1 (1,
   nx+1, ny, nz+1)``, ``EPhi2 (1, nx+1, ny+1, nz)``, ``HPhi1 (1, nx, ny+1,
   nz)``, ``HPhi2 (1, nx, ny, nz+1)``.

``test_y_direction_shapes``
   Expects the y-normal arrangement: ``EPhi1 (1, nx, ny+1, nz+1)``, ``EPhi2
   (1, nx+1, ny+1, nz)``, ``HPhi1 (1, nx+1, ny, nz)``, ``HPhi2 (1, nx, ny,
   nz+1)``.

``test_z_direction_shapes``
   Expects the z-normal arrangement: ``EPhi1 (1, nx, ny+1, nz+1)``, ``EPhi2
   (1, nx+1, ny, nz+1)``, ``HPhi1 (1, nx+1, ny, nz)``, ``HPhi2 (1, nx, ny+1,
   nz)``.

``test_arrays_start_at_zero``
   Expects every auxiliary field to begin empty — these accumulate the PML
   correction over time and must not start with debris. (4 parameter sets)

``test_arrays_use_the_configured_float_dtype``
   Expects ``float64`` under the double-precision fixture. (4 parameter
   sets)

``test_leading_axis_is_the_cfs_order``
   Expects one page per CFS term: a two-pole PML gets ``shape[0] == 2``. (3
   parameter sets)

``test_allocated_during_construction``
   Expects the four arrays to exist straight after ``__init__`` —
   ``initialise_field_arrays`` is called by the constructor, so callers
   never have to.

``test_reinitialising_replaces_the_arrays``
   Expects a fresh zeroed allocation rather than an in-place clear, so any
   Cython buffer already holding the old array keeps pointing at the old
   memory.

TestPrintPmlInfo
^^^^^^^^^^^^^^^^

A string builder with three distinct output shapes.

``test_all_zero_thickness_reports_switched_off``
   Expects ``"PML boundaries [main_grid]: switched off"`` and nothing about
   formulation or order.

``test_uniform_thickness_prints_a_single_number``
   Expects ``thickness (cells): 10`` — one value, not six, when every face
   agrees.

``test_mixed_thickness_prints_every_face``
   Expects a comma-separated ``key: value`` list covering all six faces,
   with no trailing comma.

``test_reports_the_formulation``
   Expects the active formulation string to appear verbatim.

``test_order_is_the_cfs_count``
   Expects ``order`` to report ``len(pmls["cfs"])`` — two CFS terms make a
   second-order PML.

``test_names_the_grid``
   Expects the grid's own name in brackets, so subgrid PMLs are
   distinguishable from the main grid's in the log.

``test_returns_a_string_rather_than_logging``
   Expects the function to *return* its text and emit nothing — the caller
   in ``fdtd_grid.py`` does the logging.

``test_ends_with_a_newline``
   Expects a trailing newline in both the switched-off and the normal form,
   since the caller concatenates it into a larger report.

TestConstructionIsIndependentOfGridSize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_thickness_drives_the_normal_axis_only``
   Expects the two spanning axes to keep the grid's full face size whatever
   the depth. (4 parameter sets)

``test_every_face_constructs``
   Expects all six slabs to build without error and report the direction the
   factory paired with their ID. (6 parameter sets)

``test_uniform_and_anisotropic_grids_agree_on_shape``
   Expects the cell *spacing* to affect ``d`` but never the array shapes,
   which depend only on cell counts.

When these fail
~~~~~~~~~~~~~~~

**``d`` or ``thickness`` is wrong on one axis.** Both are chosen by
``direction[0]``. The tests run on the anisotropic ``DL_ANISO`` grid (1 mm /
2 mm / 4 mm) precisely so a wrong axis cannot coincidentally give the right
number — on a uniform grid these tests would pass with any axis.

**Every PML test in the file errors at construction.** ``PML.__init__``
calls ``check_kappamin()``, which rejects a CFS list whose ``kappa.min``
values sum below one — and an *empty* list sums to zero. If a fixture change
drops the default ``CFS()``, every slab fails to construct. That is the same
thing ``FDTDGrid.build()`` guards against by installing a default CFS.

**A ``check_kappamin`` test fails on the message.** The message is only in
the log; the exception carries an empty string. Assert with ``caplog``,
never with ``str(exc)``. See `notes/bugs/check-kappamin-bare-valueerror.md`.

**An auxiliary array shape is off by one.** The twelve shapes differ per
direction and are not symmetric — the ``x`` case gives ``EPhi1`` an extra
node in x and z but not y. Read the shape table in
``TestInitialiseFieldArrays`` rather than reasoning it out.

Test Catalog — ``test_pml_coeffs.py``
-------------------------------------

**71 tests** from 31 test functions across 6 classes.

``PML.calculate_update_coeffs`` — the eight coefficient arrays.

This is where the CFS profiles become numbers the Cython kernels multiply
by. Two formulations produce eight arrays each, all closed-form in ``e0``,
``dt`` and the three profiles, so every assertion here is exact arithmetic
rather than a shape check.

Both formulations are reimplemented longhand at the top of this file. That
duplication is deliberate: a test that calls the function under test to
compute its own expectation proves only that the function is deterministic.

**Why the non-default CFS matters.** With the stock ``CFS()`` — ``alpha``
constant 0, ``kappa`` constant 1 — several formula terms vanish, and two
pairs of coefficients collapse into each other (``ERA == ERB`` under
HORIPML; ``ERB == ERE == 1`` under MRIPML). A suite that only ever used the
defaults would pass with whole terms deleted from the source. The tests
below check both: the collapsed default case, because that is what most
models actually run, *and* a fully populated CFS where every term
contributes.

TestArrayShapesAndDtypes
^^^^^^^^^^^^^^^^^^^^^^^^

``test_shape_is_cfs_order_by_thickness``
   Expects ``(len(CFS), thickness)`` — one row per CFS term, one column per
   PML cell. (8 parameter sets)

``test_dtype_matches_the_configured_precision``
   Expects ``float64`` under the double-precision fixture, matching the
   fused type the Cython kernels are compiled for. (8 parameter sets)

``test_column_count_follows_thickness``
   Expects one coefficient per cell of depth. (4 parameter sets)

``test_row_count_follows_the_cfs_order``
   Expects a row per CFS term, so a two-pole PML gets two rows. (3 parameter
   sets)

``test_recalculating_replaces_the_arrays``
   Expects fresh allocations on each call rather than in-place updates. (8
   parameter sets)

TestSigmamaxAutoCalculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_unset_sigma_max_is_derived_from_the_material``
   Expects the ``None`` sentinel on a stock ``CFS`` to be replaced by the
   closed-form optimum for the backing material.

``test_an_explicit_sigma_max_is_left_alone``
   Expects a user-supplied ``sigma.max`` to survive: the guard is ``if not
   cfs.sigma.max``, so any truthy value suppresses the auto-calculation.

``test_the_backing_material_changes_the_result``
   Expects a PML backing ``er = 4`` to derive half the ``sigma.max`` of one
   backing free space.

``test_calling_twice_does_not_recompute``
   Expects the second call to reuse the value cached on the CFS: after the
   first call ``sigma.max`` is truthy, so the guard no longer fires.

   This makes the method non-idempotent in an important way — passing a
   *different* material the second time silently has no effect on
   ``sigma.max``.

``test_logs_the_derived_value_at_debug``
   Expects a debug record naming the slab and the value, once per CFS term.

``test_the_derived_value_uses_the_slab_normal_spacing``
   Expects ``d`` — the spacing along the slab's own normal — in the
   denominator, so a y-slab on an anisotropic grid gets a different
   ``sigma.max`` from an x-slab.

TestHoripmlAgainstTheClosedForm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default formulation, checked term by term.

``test_every_coefficient_matches_the_longhand_formula``
   Expects agreement with an independently written HORIPML reference for a
   fully populated CFS — every one of the eight arrays. (8 parameter sets)

``test_era_with_the_default_cfs_is_two_e0_over_tmp``
   Expects ``ERA = 2·e0 / (2·e0 + dt·sigma)``: with ``alpha == 0`` the
   numerator's ``dt·alpha`` term vanishes and ``kappa == 1`` drops out of
   the denominator.

``test_era_equals_erb_for_the_default_cfs``
   Expects the two to coincide when ``alpha == 0`` and ``kappa == 1``.

   This is exactly why the formula tests above use ``rich_cfs``: with the
   defaults these two arrays are indistinguishable, so a suite built only on
   defaults could not tell the ``ERA`` and ``ERB`` expressions apart.

``test_era_and_erb_differ_once_alpha_is_on``
   Expects the collapse above to be a property of the defaults, not of the
   formulas.

``test_ere_is_one_where_sigma_is_zero``
   Expects the innermost cell to be transparent: the quartic sigma profile
   starts at zero, so there ``ERE == 1`` and the PML applies no correction
   at all.

``test_erf_is_zero_where_sigma_is_zero``
   Expects no loss term at the inner face, for the same reason.

``test_erf_grows_outward``
   Expects the loss coefficient to increase monotonically with depth — the
   whole point of the graded ramp.

``test_ere_shrinks_outward``
   Expects the retention coefficient to fall as absorption rises.

``test_magnetic_coefficients_differ_from_electric``
   Expects the two sets to disagree, because the profiles they are built
   from are sampled half a cell apart.

TestMripmlAgainstTheClosedForm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The multipole formulation — a different algebra from the same profiles.

``test_every_coefficient_matches_the_longhand_formula``
   Expects agreement with an independently written MRIPML reference for a
   fully populated CFS. (8 parameter sets)

``test_erb_is_exactly_one_for_the_default_cfs``
   Expects ``ERB = 2·e0 / (2·e0 + dt·alpha) == 1`` when ``alpha`` is zero
   everywhere.

``test_ere_is_exactly_one_for_the_default_cfs``
   Expects ``ERE = (2·e0 - dt·alpha) / (2·e0 + dt·alpha) == 1`` for the same
   reason.

``test_era_starts_at_kappa``
   Expects ``ERA = kappa + dt·sigma/(2·e0)``, which at the transparent inner
   face (``sigma == 0``) is exactly ``kappa``, i.e. 1 by default.

``test_erf_is_sigma_dt_over_e0_for_the_default_cfs``
   Expects ``ERF = 2·sigma·dt / (2·e0) == sigma·dt/e0`` once the ``alpha``
   term drops out of the denominator.

``test_the_two_formulations_disagree``
   Expects genuinely different numbers from the same CFS — otherwise the
   formulation switch would be decorative.

TestScalingWithDt
^^^^^^^^^^^^^^^^^

``test_a_zero_time_step_makes_the_pml_transparent``
   Expects ``A == B == E == 1`` and ``F == 0`` when ``dt == 0``: every
   correction term carries a factor of ``dt``, so nothing is absorbed.

   A useful degenerate control — it isolates the ``dt``-free part of each
   formula.

``test_erf_is_linear_in_dt_for_small_steps``
   Expects doubling ``dt`` to roughly double ``ERF`` while ``dt·sigma``
   stays small against ``2·e0`` — the numerator is linear in ``dt`` and the
   denominator barely moves.

TestMultipole
^^^^^^^^^^^^^

``test_each_cfs_term_fills_its_own_row``
   Expects two CFS terms with different sigma maxima to produce two distinct
   rows, in list order.

``test_rows_are_independent``
   Expects the second term's row to match a single-term PML built from the
   same CFS — no cross-talk between poles.

``test_a_debug_record_is_emitted_per_term``
   Expects one ``sigma.max set to`` record for each CFS term.

When these fail
~~~~~~~~~~~~~~~

**One coefficient is wrong but the default-CFS tests still pass.** With the
stock ``CFS()`` — ``alpha`` constant 0, ``kappa`` constant 1 — several terms
vanish and two pairs collapse (``ERA == ERB`` under HORIPML; ``ERB == ERE ==
1`` under MRIPML). That is why the formula tests use the ``rich_cfs``
fixture, where all three parameters ramp and nothing cancels. **If you add a
coefficient test, use ``rich_cfs``.**

**Every coefficient is zero.** Check ``G.pmls["formulation"]``. An
unrecognised string falls through both branches and leaves all eight arrays
zeroed, with nothing raised — see `notes/bugs/pml-unknown-formulation-
silent-zeros.md`. There is deliberately no test for that path.

**A ``sigma_max`` assertion fails after a second call.**
``calculate_update_coeffs`` only derives ``sigma.max`` when it is falsy, and
it *writes it back onto the CFS*. A second call with a different backing
material silently reuses the first value.
``test_calling_twice_does_not_recompute`` pins this.

**``TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'``.**
A stock ``sigma`` has ``max=None`` until ``calculate_update_coeffs`` derives
it. Any test reconstructing the sigma profile must do so *after* that call.

Test Catalog — ``test_pml_updates.py``
--------------------------------------

**81 tests** from 24 test functions across 4 classes.

``PML.update_electric`` / ``update_magnetic`` — the Cython wiring.

Neither method does any arithmetic. Each assembles a module path and a
function name from the grid's formulation, the CFS order and the slab's
direction, imports the module, and forwards twenty-two positional arguments
to the kernel it finds there:

gprMax.cython.pml_updates_<polarity>_<formulation>.order<N>_<direction>

That naming convention is the entire contract, and it is invisible until it
breaks — a renamed direction or a changed CFS count produces an
``AttributeError`` from deep inside an import, with nothing pointing at the
slab that caused it.

So the tests here patch ``import_module`` and inspect the resolved names and
the forwarded argument list. Driving the real kernels is deliberately not
attempted: they mutate whole field arrays in an OpenMP parallel region, so
asserting on them would be a solver test rather than a wiring test. What
*is* checked against reality is that every name the convention can produce
for the supported orders actually exists in the compiled extensions.

TestModulePathResolution
^^^^^^^^^^^^^^^^^^^^^^^^

``test_electric_uses_the_electric_module``
   Expects ``gprMax.cython.pml_updates_electric_HORIPML`` for a default-
   formulation grid.

``test_magnetic_uses_the_magnetic_module``
   Expects ``gprMax.cython.pml_updates_magnetic_HORIPML`` — the two
   polarities live in separate extensions.

``test_formulation_is_appended_to_the_module_name``
   Expects the grid's ``pmls["formulation"]`` string to become the module
   suffix, so the formulation switch selects compiled code as well as
   coefficient algebra. (2 parameter sets)

``test_the_formulation_is_read_at_call_time``
   Expects a formulation changed after construction to take effect on the
   next update — the module path is rebuilt on every call rather than
   cached.

TestFunctionNameResolution
^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_direction_selects_the_kernel``
   Expects ``order1_<direction>`` for a single-pole PML on each of the six
   faces. (6 parameter sets)

``test_cfs_count_selects_the_order``
   Expects ``order<N>`` to track ``len(CFS)``, so a two-pole PML calls a
   different kernel from a one-pole PML. (2 parameter sets)

``test_magnetic_resolves_the_same_function_name``
   Expects the two polarities to share a name and differ only by module —
   ``order1_zplus`` exists in both extensions.

TestForwardedArguments
^^^^^^^^^^^^^^^^^^^^^^

``test_electric_forwards_twenty_two_arguments``
   Expects exactly 22 positional arguments — the kernels take no keywords,
   so a signature change is silent unless pinned here.

``test_magnetic_forwards_twenty_two_arguments``
   Expects the magnetic kernel to take the same count.

``test_extents_lead_the_argument_list``
   Expects ``xs, xf, ys, yf, zs, zf`` in positions 0-5, matching the slab's
   own bounds.

``test_thread_count_follows_the_extents``
   Expects ``config.get_model_config().ompthreads`` in position 6 — one
   under the test fixture.

``test_electric_passes_the_electric_update_coefficients``
   Expects ``G.updatecoeffsE`` in position 7, not the magnetic set.

``test_magnetic_passes_the_magnetic_update_coefficients``
   Expects ``G.updatecoeffsH`` in the same slot — the one positional
   difference in the shared head of the two signatures.

``test_all_six_field_arrays_are_forwarded_in_order``
   Expects ``ID`` then ``Ex, Ey, Ez, Hx, Hy, Hz`` in positions 8-14. Both
   polarities receive all six: a PML correction couples E to H.

``test_electric_forwards_its_own_phi_arrays``
   Expects ``EPhi1, EPhi2`` in positions 15-16 — the electric accumulators,
   never the magnetic ones.

``test_magnetic_forwards_its_own_phi_arrays``
   Expects ``HPhi1, HPhi2`` in the same slots.

``test_electric_forwards_the_e_coefficients``
   Expects ``ERA, ERB, ERE, ERF`` in positions 17-20, in that order.

``test_magnetic_forwards_the_h_coefficients``
   Expects ``HRA, HRB, HRE, HRF`` in the same positions.

``test_spacing_is_the_final_argument``
   Expects ``d`` — the spacing along the slab normal — last.

``test_arrays_are_passed_by_reference``
   Expects the live arrays rather than copies: the kernels write the PML
   correction back into the grid in place.

``test_update_returns_none``
   Expects no return value — the kernels communicate entirely through in-
   place mutation.

TestTheConventionResolvesAgainstTheRealExtensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No mocking here: the compiled modules must actually contain the names the
convention generates.

``test_both_formulations_ship_both_polarities``
   Expects all four compiled extensions to import. (4 parameter sets)

``test_every_generated_name_exists``
   Expects ``order<N>_<direction>`` to resolve for orders 1 and 2 in all six
   directions, across both formulations and both polarities — 48
   combinations, every one reachable from a valid input file.

   Orders above 2 are deliberately absent from the extensions;
   ``cmds_multiuse.py`` caps the CFS list at two. (48 parameter sets)

``test_a_third_order_kernel_is_not_provided``
   Expects ``order3_xminus`` to be absent, documenting the supported ceiling
   rather than leaving it implicit.

When these fail
~~~~~~~~~~~~~~~

**A module path or function name assertion fails.** The convention is
``gprMax.cython.pml_updates_<polarity>_<formulation>.order<N>_<direction>``,
assembled fresh on every call. All four parts are read at call time, so a
formulation changed after construction takes effect immediately.

**An argument-position test fails.** The kernels take 22 positional
arguments and no keywords, so a signature change is invisible to Python and
shows up only here. The electric and magnetic calls differ in exactly three
slots: position 7 (``updatecoeffsE``/``H``), 15-16 (``EPhi``/``HPhi``) and
17-20 (``ER*``/``HR*``).

**``AttributeError: 'FDTDGrid' object has no attribute 'updatecoeffsE'``.**
The grid fixture must call ``initialise_std_update_coeff_arrays()``. Like
``Rx.ID``, that attribute is only *annotated* in ``__init__`` and assigned
by a separate initialiser.

**``TestTheConventionResolvesAgainstTheRealExtensions`` fails.** These 53
tests do no mocking — they import the compiled extensions and check every
name the convention can produce. A failure means the Cython extensions are
stale or were built without one of the four modules. Rebuild with ``pip
install -e .``.

Test Catalog — ``test_pml_build_kernel.py``
-------------------------------------------

**24 tests** from 19 test functions across 5 classes.

``cython/pml_build.pyx`` — averaging the material behind a PML slab.

``sigma_max`` depends on what the PML is backed by: a layer absorbing into
wet clay needs a different conductivity ramp from one absorbing into air.
These two kernels supply that number, reducing the 2D face of material IDs
behind a slab to a single mean permittivity and permeability.

Both take the same six arguments — two face dimensions, a thread count, the
``solid`` slice (material ID per cell), and two lookup tables indexed by
material ID. ``pml_average_er_mr`` divides by the cell count;
``pml_sum_er_mr`` does not.

Both are OpenMP ``prange`` loops with ``sumer``/``summr`` as reduction
variables, so these are real parallel kernels driven with real arrays. One
caveat worth recording: those accumulators are declared ``cdef double`` with
no initialiser, and OpenMP's ``reduction(+:)`` adds the *pre-existing* value
into the result. On this build they read as zero and every test below
passes; if they ever stop doing so, these are the tests that will go red,
and the cause will not be anything in this file.

TestAverageSingleMaterial
^^^^^^^^^^^^^^^^^^^^^^^^^

``test_uniform_face_returns_that_material``
   Expects a face made entirely of material 0 to average to exactly that
   material's ``er`` and ``mr``.

``test_free_space_face_returns_one_and_one``
   Expects ``(1.0, 1.0)`` for a face of free space — the common case, and
   the value that makes ``sigma_max`` reduce to ``0.8·(m+1)/(z0·d)``.

``test_er_and_mr_are_looked_up_independently``
   Expects the two tables to be indexed separately, so a material can have a
   high permittivity and unit permeability.

``test_face_size_does_not_change_a_uniform_average``
   Expects the mean of a constant face to be that constant whatever the
   face's size. (4 parameter sets)

TestAverageMixedMaterials
^^^^^^^^^^^^^^^^^^^^^^^^^

``test_half_and_half_gives_the_midpoint``
   Expects ``(1 + 9)/2 == 5`` for a face split evenly between two materials.

``test_weighting_follows_cell_counts``
   Expects ``(3·1 + 1·9)/4 == 3`` — a three-to-one split weights the mean
   toward the majority material.

``test_matches_numpy_on_a_random_face``
   Expects agreement with ``ers[solid].mean()`` — an independent formulation
   of the same reduction.

``test_a_non_square_face_divides_by_the_product``
   Expects the divisor to be ``n1·n2``, not ``n1`` or ``max(n1, n2)`` — the
   two face dimensions are independent.

``test_only_the_first_n1_by_n2_cells_are_read``
   Expects the dimensions to govern the traversal rather than the array's
   own shape, so a larger buffer can be passed with a smaller window read
   out of its top-left corner.

TestSumVersusAverage
^^^^^^^^^^^^^^^^^^^^

``test_sum_is_the_average_times_the_cell_count``
   Expects the two kernels to differ only by the ``n1·n2`` divisor —
   ``pml_sum_er_mr`` exists so MPI ranks can add partial sums before
   dividing once globally.

``test_sum_matches_numpy``
   Expects agreement with ``ers[solid].sum()``.

``test_partial_sums_compose``
   Expects summing two halves of a face to equal summing the whole — the
   property the MPI path relies on.

TestThreading
^^^^^^^^^^^^^

``test_result_is_independent_of_thread_count``
   Expects identical answers however the ``prange`` is split — the
   accumulators are OpenMP reduction variables, so a missing reduction
   clause would show up here as a race. (3 parameter sets)

``test_repeated_calls_are_deterministic``
   Expects the same answer every time. A drifting result would mean the
   reduction accumulator is carrying state between calls — see the module
   docstring.

``test_interleaving_different_faces_does_not_contaminate``
   Expects a small face's answer to be unaffected by a large call in between
   — the strongest available check that nothing leaks across invocations.

TestDtypes
^^^^^^^^^^

``test_accepts_double_precision_lookups``
   Expects ``float64`` tables to bind to the double specialisation of the
   fused ``float_or_double`` type.

``test_accepts_single_precision_lookups``
   Expects ``float32`` tables to bind to the float specialisation — gprMax
   compiles both, selected by the run's precision setting.

``test_returns_python_floats``
   Expects plain floats, since the result feeds straight into
   ``CFS.calculate_sigmamax``'s scalar arithmetic.

``test_solid_must_be_unsigned_32_bit``
   Expects a typed-memoryview rejection for the wrong integer width — the
   ``solid`` array is ``uint32`` throughout gprMax, and a silent
   reinterpretation would index the lookup tables with garbage.

When these fail
~~~~~~~~~~~~~~~

**An average is subtly wrong, or drifts between runs.** ``sumer`` and
``summr`` are ``cdef double`` with no initialiser, used as OpenMP
``reduction(+:)`` variables. OpenMP adds the *pre-existing* value into the
result, so this is undefined behaviour that currently happens to read as
zero. If these tests go red for no apparent reason, the cause is not in the
test file — see `notes/bugs/pml-build-uninitialised-reduction.md`.

**A thread-count test fails.**
``test_result_is_independent_of_thread_count`` is the race detector. A
failure means the reduction clause was lost, not that the arithmetic
changed.

**A dtype test fails.** ``solid`` must be ``uint32`` and the lookup tables
must be a single consistent float width. The fused ``float_or_double`` type
has both specialisations compiled; mixing widths within one call has no
matching signature.

Test Catalog — ``test_mpi_pml.py``
----------------------------------

**23 tests** from 20 test functions across 5 classes.

``MPIPML`` — one rank derives ``sigma_max``, everyone else is told.

Under MPI the domain is split across ranks, and a PML slab that straddles
the split would otherwise have each rank compute ``sigma_max`` from only the
material it can see locally. The ranks would then disagree about how
absorbing their share of the same slab is, and the seam would reflect.

``MPIPML.calculate_update_coeffs`` fixes that by having rank 0 compute the
value and broadcast it, before delegating the rest of the work to the base
class unchanged.

**Why the broadcast is non-blocking.** A rank holding two slabs reaches the
second broadcast only after finishing the first, while a rank holding one
slab is already waiting. A blocking ``Bcast`` would deadlock; ``Ibcast``
plus ``Wait`` will not. The comment in the source says as much, and the
tests below pin the mechanism (``Ibcast`` is used, and the value that
arrives is the one that gets used) rather than trying to reproduce a
deadlock.

**What one rank can and cannot show.** These tests run on ``MPI.COMM_SELF``,
where rank 0 is the only rank, so the coordinator branch is exercised and
the follower branch is not. What that *does* establish is the property that
matters most: with one rank the MPI path must agree exactly with the serial
one, so the override cannot have changed any arithmetic.

TestClassSurface
^^^^^^^^^^^^^^^^

``test_extends_pml``
   Expects ``MPIPML`` to inherit the whole serial surface and override only
   ``calculate_update_coeffs``.

``test_only_the_coefficient_method_is_overridden``
   Expects construction, validation, array allocation and both update
   methods to be inherited unchanged — the MPI concern is confined to one
   method.

``test_rank_zero_coordinates``
   Expects rank 0 to be the one that derives ``sigma_max``.

``test_constructs_like_a_serial_pml``
   Expects the inherited constructor to set up ``d``, ``thickness`` and the
   four auxiliary arrays exactly as the base class does.

TestCoordinatorPath
^^^^^^^^^^^^^^^^^^^

``test_derives_sigma_max_on_the_coordinator``
   Expects rank 0 to compute the same optimum a serial PML would, from the
   same backing material.

``test_the_broadcast_value_is_what_gets_stored``
   Expects ``sigma.max`` to be read back out of the receive buffer rather
   than kept from the local computation — on one rank the two coincide,
   which is exactly why the round trip must be lossless.

``test_an_explicit_sigma_max_skips_the_broadcast``
   Expects the guard ``if not cfs.sigma.max`` to suppress the exchange
   entirely when the user supplied a value — no collective is entered, so no
   rank can block on one.

``test_uses_a_non_blocking_broadcast``
   Expects ``Ibcast(...).Wait()`` rather than ``Bcast``. A rank holding two
   slabs reaches its second broadcast late; the blocking form would deadlock
   against a rank already waiting.

``test_broadcasts_from_the_coordinator_rank``
   Expects ``COORDINATOR_RANK`` to be passed as the broadcast root, so every
   rank agrees on who is authoritative.

``test_one_broadcast_per_cfs_term``
   Expects a two-pole PML to exchange two values — each CFS term has its own
   ``sigma_max``.

TestFollowerPath
^^^^^^^^^^^^^^^^

Ranks other than 0 allocate an empty buffer and take what arrives.

``test_a_follower_adopts_the_broadcast_value``
   Expects a non-coordinator rank to skip the local computation and use
   whatever the broadcast delivers.

   A one-rank ``COMM_SELF`` cannot produce a real follower, so the
   communicator is faked to report a non-zero rank and to fill the receive
   buffer the way a real broadcast would.

``test_a_follower_does_not_compute_locally``
   Expects the received value to win even when it differs from what the
   local material would have given — the whole point of the exchange.

``test_the_received_value_drives_the_coefficients``
   Expects the broadcast ``sigma_max`` to flow through into ``ERF``, so
   every rank builds identical coefficient arrays for a shared slab.

TestAgreementWithTheSerialPath
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The override must not change any arithmetic.

``test_single_rank_matches_a_plain_pml``
   Expects every coefficient array to be identical to the serial one at one
   rank, for both formulations. (2 parameter sets)

``test_delegates_to_the_base_implementation``
   Expects the eight arrays to be allocated by ``super()`` — the override
   adds a broadcast and changes nothing else.

``test_the_backing_material_still_matters``
   Expects a different ``er`` to reach ``calculate_sigmamax`` through the
   override unchanged.

``test_every_face_agrees_with_serial``
   Expects parity on each axis, so the direction-dependent ``d`` is picked
   up identically by both paths. (3 parameter sets)

TestRealCommunicator
^^^^^^^^^^^^^^^^^^^^

``MPI.COMM_SELF`` rather than a hand-written double.

``test_a_real_ibcast_round_trips_the_value``
   Expects the genuine mpi4py non-blocking broadcast to deliver the
   coordinator's value back into the buffer at one rank.

``test_comm_self_reports_one_rank``
   Expects a single-rank communicator, which is what makes the coordinator
   branch the one under test here.

``test_repeated_calls_do_not_re_broadcast``
   Expects the second call to find ``sigma.max`` already truthy and skip the
   collective — important, because an unmatched collective on one rank would
   hang every other rank.

When these fail
~~~~~~~~~~~~~~~

**A test hangs instead of failing.** Anything that enters an MPI collective
without a matching call on every rank blocks forever. At one rank that
cannot happen with the real ``COMM_SELF``, but a hand-written communicator
double that forgets to return from ``Ibcast`` will.

**A broadcast assertion fails.** ``MPIPML`` uses ``Ibcast(...).Wait()``, not
``Bcast``. The non-blocking form is required: a rank holding two slabs
reaches its second broadcast after a rank holding one is already waiting,
and the blocking form deadlocks. ``test_uses_a_non_blocking_broadcast``
fails the test outright if ``Bcast`` is called.

**``TestAgreementWithTheSerialPath`` fails.** ``MPIPML`` must not change any
arithmetic — it adds a broadcast and delegates. At one rank its coefficients
must equal the serial class's exactly, for both formulations.

**A second call unexpectedly broadcasts.** After the first call
``sigma.max`` is truthy, so the collective is skipped. An unmatched
collective on one rank would hang every other rank, which is why
``test_repeated_calls_do_not_re_broadcast`` guards it.

Test Catalog — ``test_grid_view.py``
------------------------------------

**91 tests** from 56 test functions across 11 classes.

``GridView`` — the rectangular window every exporter looks through.

A ``GridView`` is a start, a stop and a stride. Given those it answers two
questions: what shape is this region, and hand me that slice of an array.
Every snapshot, every geometry view and every geometry object holds one and
delegates all its coordinate arithmetic to it, which makes this the single
highest-traffic class in the PR.

Three things govern the assertions below.

**Size is a ceiling, not a floor.** ``size = ceil((stop - start) / step)``.
A view from 0 to 10 with step 3 spans *four* cells, not three: the final
partial cell is kept. Substituting integer division would silently shorten
every exported array by one in every axis on any non-dividing view.

**There are two slice families, and they mean different things.**
``getter_slice``/``setter_slice`` index the *grid's* arrays in grid
coordinates. ``get_output_slice``/``get_read_slice`` index the *view's* own
output buffer, always starting at zero. In the serial class the members of
each pair are literally the same function; only ``MPIGridView`` makes them
diverge, which is precisely why the equivalences are asserted here — they
are the baseline the MPI overrides are measured against.

**``upper_bound_exclusive=False`` fetches one extra step.** Node-centred
arrays (``ID``) and the six field arrays are read this way, because a Yee
cell has one more node than cell along each axis and the snapshot kernel
averages across that extra node. Cell-centred arrays (``solid``, ``rigidE``,
``rigidH``) are read exclusively. Getting this backwards does not crash; it
shifts every exported field half a cell.

TestConstruction
^^^^^^^^^^^^^^^^

``test_stores_start_stop_and_step``
   Expects the nine coordinate arguments to become three int32 triples.

``test_step_defaults_to_one``
   Expects an unstrided view when no step is given — the common case for a
   geometry view of the whole domain.

``test_coordinate_arrays_are_int32``
   Expects ``int32`` throughout: these feed HDF5 extent attributes that
   downstream readers type-check.

``test_holds_the_grid_by_reference``
   Expects ``view.grid`` to be the same object — the view slices the grid's
   live arrays, so a copy would silently detach every setter.

``test_the_id_cache_starts_empty``
   Expects ``_ID`` to be ``None`` until first requested — the slice is built
   lazily and then cached.

``test_logs_its_creation_at_debug``
   Expects a debug record naming the grid and all four coordinate triples,
   which is the only trace a view leaves in a normal run.

TestSizeArithmetic
^^^^^^^^^^^^^^^^^^

``test_unit_step_size_is_the_extent``
   Expects ``stop - start`` when the step is one.

``test_exact_division_gives_the_quotient``
   Expects ``(0, 8)`` step 2 to give four cells.

``test_non_dividing_extent_rounds_up``
   Expects ``ceil(10/3) == 4``, not ``10 // 3 == 3``.

   This is the assertion that distinguishes the actual implementation from
   the one most readers assume. The tenth cell is a partial step and it is
   still counted.

``test_ceiling_boundaries``
   Expects the size to tick over exactly one element past each multiple of
   the step. (7 parameter sets)

``test_axes_are_sized_independently``
   Expects three different steps to give three different sizes — an axis
   mix-up cannot survive this.

``test_a_zero_width_axis_gives_zero_cells``
   Expects ``start == stop`` to produce an empty axis rather than one cell.

``test_offset_start_does_not_change_the_count``
   Expects size to depend on the extent, not on where it begins.

TestCoordinateProperties
^^^^^^^^^^^^^^^^^^^^^^^^

``test_each_property_reads_its_own_axis``
   Expects the twelve scalar accessors to index the right element of the
   right triple. Anisotropic steps make a wrong axis impossible to miss. (12
   parameter sets)

``test_the_three_size_properties_are_distinct``
   Expects ``nx``, ``ny`` and ``nz`` to disagree on an anisotropic view — a
   sanity check on the parametrised test above.

TestGetterSlice
^^^^^^^^^^^^^^^

``test_exclusive_upper_bound_stops_at_stop``
   Expects ``slice(start, stop, step)`` — the default, used for cell-centred
   arrays.

``test_inclusive_upper_bound_adds_one_step``
   Expects ``slice(start, stop + step, step)`` — one extra sample, for node-
   centred arrays and the field arrays.

``test_the_extra_sample_is_a_whole_step``
   Expects a strided view to extend by its own step, not by one cell — so a
   step-3 view reaches ``stop + 3``.

``test_each_dimension_uses_its_own_coordinates``
   Expects the slice for axis ``d`` to be built from ``start[d]``,
   ``stop[d]`` and ``step[d]``. (3 parameter sets)

TestSetterSliceMatchesGetterSlice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the serial class the two are the same function.

``MPIGridView`` overrides ``setter_slice`` with halo-aware logic, so these
equivalences are the contrast the MPI tests are measured against. Asserting
them here means a future divergence in the *base* class shows up as a
failure rather than as a silent behaviour change.

``test_setter_slice_delegates_to_getter_slice``
   Expects identical slices from both, on every axis and both bound
   conventions. (6 parameter sets)

``test_read_slice_delegates_to_output_slice``
   Expects ``get_read_slice`` and ``get_output_slice`` to coincide: a single
   process has no separate read and write partitioning. (3 parameter sets)

``test_3d_read_slice_matches_3d_output_slice``
   Expects the tuple forms to agree as well.

TestOutputSlice
^^^^^^^^^^^^^^^

``test_starts_at_zero``
   Expects output slices to index the view's own buffer, so they begin at
   zero however far into the grid the view sits.

``test_length_is_the_view_size``
   Expects the slice to span exactly ``size[dimension]``.

``test_inclusive_bound_adds_exactly_one``
   Expects ``size + 1``, not ``size + step`` — output slices index a dense
   buffer, so the extra node is one element regardless of stride.

``test_3d_form_returns_one_slice_per_axis``
   Expects a three-tuple in x, y, z order.

TestArraySlicing
^^^^^^^^^^^^^^^^

``test_slices_the_last_three_dimensions``
   Expects a leading component axis to pass through untouched, so a ``(6,
   nx, ny, nz)`` array keeps all six components.

``test_extracts_the_requested_region``
   Expects the values from exactly the named cells — the ramp fill makes
   every cell distinguishable.

``test_stride_takes_every_nth_cell``
   Expects a step of two to select alternate cells, not the first half.

``test_result_is_contiguous``
   Expects ``np.ascontiguousarray`` to have been applied — the arrays go
   straight into HDF5 and typed memoryviews, both of which require it.

``test_result_is_a_copy_not_a_view``
   Expects mutations of the slice not to reach the grid: strided slicing
   plus ``ascontiguousarray`` necessarily copies, and callers rely on being
   able to remap material IDs without corrupting the model.

``test_set_array_slice_writes_back``
   Expects the setter to reach the grid's own array, and only the cells the
   view covers.

``test_set_array_slice_leaves_the_rest_alone``
   Expects cells outside the view to keep their previous values.

``test_get_and_set_round_trip``
   Expects reading a region and writing it straight back to be a no-op.

TestTypedArrayAccessors
^^^^^^^^^^^^^^^^^^^^^^^

Which arrays are fetched with which bound convention.

``test_cell_centred_arrays_use_the_exclusive_bound``
   Expects ``solid``, ``rigidE`` and ``rigidH`` to span exactly the view's
   size in each spatial axis — they hold one value per *cell*. (3 parameter
   sets)

``test_field_arrays_fetch_one_extra_node``
   Expects ``size + 1`` in every axis. The snapshot kernel averages
   neighbouring samples to bring the six staggered components onto a common
   point, so it needs the node past the edge. (6 parameter sets)

``test_id_fetches_one_extra_node``
   Expects ``(6, size+1, size+1, size+1)`` — ``ID`` is node-centred and
   carries all six components.

``test_rigid_arrays_keep_their_leading_axes``
   Expects ``rigidE`` to keep 12 components and ``rigidH`` 6 — the leading
   axis is not spatial and must not be sliced.

``test_setters_write_back_to_the_grid``
   Expects each typed setter to land in the grid array of the same name. (3
   parameter sets)

``test_set_id_uses_the_inclusive_bound``
   Expects the ``ID`` setter to cover ``size + 1`` nodes, matching its
   getter — a mismatched pair would raise on assignment.

TestIdCaching
^^^^^^^^^^^^^

``test_the_first_call_populates_the_cache``
   Expects ``_ID`` to be filled on first access.

``test_repeat_calls_return_the_cached_array``
   Expects the same object back, so ``initialise_materials`` and a user call
   do not each rebuild the slice.

``test_force_refresh_rebuilds``
   Expects a new array when asked, so geometry built after the first access
   is picked up.

``test_the_cache_is_stale_after_the_grid_changes``
   Expects the cached copy *not* to track later grid mutations.

   ``get_array_slice`` copies, so this is inherent rather than a defect —
   but it means anything reading ``ID`` after geometry changes must pass
   ``force_refresh``, which is exactly what ``initialise_materials`` does.

TestInitialiseMaterials
^^^^^^^^^^^^^^^^^^^^^^^

``test_unfiltered_takes_every_material``
   Expects all of the grid's materials, whether or not they appear in the
   view — what ``GeometryViewLines`` asks for.

``test_filtered_keeps_only_what_the_view_contains``
   Expects a view over free-space-only cells to report one material, even
   though the grid defines three.

``test_filtering_reads_the_id_array_afresh``
   Expects ``force_refresh`` on the internal ``get_ID`` call, so geometry
   written after an earlier ``get_ID()`` is still seen.

``test_materials_are_sorted``
   Expects ascending order, so the exported material table is stable between
   runs regardless of definition order.

``test_builds_a_dense_index_map``
   Expects the map to renumber sparse grid IDs onto ``0..n-1``, which is
   what makes an exported file self-contained.

``test_map_preserves_dtype``
   Expects ``uint32`` in, ``uint32`` out — ``np.vectorize`` would otherwise
   widen to int64 and break the HDF5 layout.

``test_map_preserves_shape``
   Expects an elementwise mapping, not a flattening.

``test_unfiltered_map_is_the_identity_for_contiguous_ids``
   Expects no renumbering when the grid's own IDs are already dense —
   materials 0, 1, 2 map to 0, 1, 2.

``test_an_unmapped_id_raises``
   Expects ``KeyError`` for a material outside the filtered set — the map is
   a plain dict lookup, so an ID the view never saw is a hard error rather
   than a silently wrong colour.

TestAnisotropicViews
^^^^^^^^^^^^^^^^^^^^

``test_each_axis_slices_independently``
   Expects three different steps to produce three different lengths in the
   sliced result.

``test_field_slices_add_one_node_per_axis``
   Expects ``(13, 7, 5)`` for the same view — one extra sample on each axis
   regardless of that axis's stride.

``test_spacing_does_not_affect_shapes``
   Expects an anisotropic *physical* discretisation to leave every shape
   unchanged: ``GridView`` counts cells and never consults ``dl``.

When these fail
~~~~~~~~~~~~~~~

**Every exported array is one short in every axis.** ``size`` is
``ceil((stop - start) / step)``, not ``//``. A view from 0 to 10 at step 3
has *four* cells; the final partial cell is kept.
``test_non_dividing_extent_rounds_up`` is the assertion that distinguishes
the two, and it is the single most likely thing to break here.

**A field is shifted half a cell.** Cell-centred arrays (``solid``,
``rigidE``, ``rigidH``) are fetched with ``upper_bound_exclusive=True``;
node-centred ones (``ID``) and all six field arrays with ``False``, which
fetches one extra *step*. Swapping the two does not crash.

**A base-class equivalence test fails.** In the serial class
``setter_slice`` *is* ``getter_slice`` and ``get_read_slice`` *is*
``get_output_slice``. Those tests exist to be the baseline ``MPIGridView``
diverges from. If the base class genuinely needs to diverge, the MPI tests
need revisiting at the same time.

**``KeyError`` from ``map_to_view_materials``.** The map is a plain dict
built from the materials the view can see. An ID outside that set is a hard
error by design — but note ``ID`` and ``solid`` both initialise to **1**, so
a grid defining only material 0 will raise unless the test sets them.

**A cached ``ID`` test fails.** ``get_ID`` caches, and ``get_array_slice``
copies — so the cache does not track later grid mutations.
``initialise_materials`` passes ``force_refresh`` for exactly that reason.

Test Catalog — ``test_mpi_grid_view.py``
----------------------------------------

**62 tests** from 54 test functions across 11 classes.

``MPIGridView`` — the same window, split across ranks.

Under MPI each rank owns a slab of the domain plus a *halo*: a border of
cells mirroring its neighbours' edges, needed so the field update can reach
one cell past the rank boundary. A geometry view or snapshot spanning the
whole domain therefore has to be trimmed on each rank to the part that rank
actually owns, without double-counting halo cells and without breaking the
view's stride.

That trimming is what this class adds, and it is pure numpy:

- ``global_*`` records the view as the user asked for it, in global
  coordinates

- ``has_negative_neighbour`` / ``has_positive_neighbour`` mark which faces
  abut another rank rather than the true domain edge

- ``start`` and ``stop`` are pulled back inside the local grid, *staying
  aligned to the step* — that modulo is the fiddly part

- ``offset`` says where this rank's block belongs inside the global output

**Why these tests work at one rank.** ``MPIGridView.__init__`` asserts
``isinstance(comm, MPI.Intracomm)``, so a mock communicator is rejected
outright. The fixtures hand it a genuine ``MPI.COMM_SELF`` and fake only the
*grid* — and the clamping arithmetic depends on ``negative_halo_offset`` and
``grid.size``, not on how many ranks exist. Setting a halo offset makes a
one-rank view behave exactly as a mid-domain rank would, so every branch
below is reachable. What one rank cannot show is cross-rank agreement; those
tests assert the local arithmetic and the collective call contract instead.

TestConstruction
^^^^^^^^^^^^^^^^

``test_extends_the_serial_grid_view``
   Expects ``MPIGridView`` to inherit the whole serial surface.

``test_creates_a_cartesian_communicator``
   Expects ``comm`` to be a real ``Cartcomm``, built from the range of MPI
   grid coordinates the view spans.

``test_the_communicator_is_a_new_one``
   Expects a fresh sub-communicator rather than the grid's own, so
   collectives over a view do not involve ranks outside it.

``test_requires_a_real_intracomm``
   Expects the ``assert isinstance(comm, MPI.Intracomm)`` guard to reject a
   stand-in communicator.

   This is why every fixture here supplies a genuine ``MPI.COMM_SELF``: the
   class cannot be tested with a mock.

``test_logs_its_creation_at_debug``
   Expects a debug record carrying both the global and the local coordinate
   triples — the only way to see the clamping in a real run.

TestGlobalCoordinates
^^^^^^^^^^^^^^^^^^^^^

``test_global_start_is_the_local_start_mapped_out``
   Expects ``local_to_global_coordinate(start)`` — with the fixture's origin
   of 100, a local 0 becomes a global 100.

``test_global_stop_is_the_local_stop_mapped_out``
   Expects the requested upper bound in global coordinates, before any
   clamping: 12 local becomes 112 global.

``test_global_size_is_the_unclamped_size``
   Expects ``ceil((12 - 0)/2) == 6`` — the size of the view *as requested*,
   captured before the local clamp shrinks ``size``.

   This is the shape of the collective output dataset, so it must reflect
   the whole view rather than this rank's share of it.

``test_global_size_exceeds_local_size_when_clamped``
   Expects the local block to be strictly smaller once both halo faces are
   trimmed — six global cells, four local.

``test_they_agree_when_nothing_is_clamped``
   Expects global and local sizes to coincide for a rank owning the whole
   view.

``test_global_size_properties``
   Expects ``gx``/``gy``/``gz`` to index ``global_size``, mirroring
   ``nx``/``ny``/``nz`` on the local size. (3 parameter sets)

TestNeighbourDetection
^^^^^^^^^^^^^^^^^^^^^^

``test_a_start_inside_the_halo_means_a_negative_neighbour``
   Expects ``start < negative_halo_offset`` to flag the low face: a view
   beginning at 0 with a 2-cell halo is asking for cells that belong to the
   rank below.

``test_a_start_outside_the_halo_means_no_negative_neighbour``
   Expects the flag to clear once the view begins at or past the halo
   boundary.

``test_a_stop_past_the_grid_means_a_positive_neighbour``
   Expects ``stop > grid.size`` to flag the high face.

``test_a_stop_within_the_grid_means_no_positive_neighbour``
   Expects the flag to clear for a view ending inside the local grid.

``test_the_two_faces_are_detected_independently``
   Expects a view abutting a neighbour on one side only to set exactly one
   flag.

``test_axes_are_detected_independently``
   Expects a per-axis decision, so a view can abut a neighbour in x and the
   domain edge in z.

TestClamping
^^^^^^^^^^^^

``test_start_is_pulled_out_of_the_negative_halo``
   Expects ``start`` to move from 0 to 2, the first cell this rank actually
   owns.

``test_the_clamped_start_stays_aligned_to_the_step``
   Expects ``halo + ((start - halo) % step)``, which keeps the sample points
   on the same lattice the user asked for.

   With ``start=1``, ``halo=2``, ``step=3``: ``(1-2) % 3 == 2``, so the
   clamped start is 4 — not 2. Snapping naively to the halo boundary would
   shift every exported sample by one cell.

``test_stop_is_pulled_back_into_the_local_grid``
   Expects ``stop`` to move from 12 to 10, the local grid size.

``test_the_clamped_stop_stays_aligned_to_the_step``
   Expects ``grid.size + ((stop - grid.size) % step)``, which can land
   *past* the grid size to preserve the stride.

   With ``stop=16``, ``size=12``, ``step=3``: ``(16-12) % 3 == 1``, so the
   clamped stop is 13.

``test_nothing_is_clamped_at_the_domain_edge``
   Expects a view matching the grid exactly to keep its own bounds.

``test_size_is_recomputed_after_clamping``
   Expects ``ceil((10 - 2)/2) == 4`` — the size attribute is overwritten
   once the bounds settle.

``test_clamping_is_per_axis``
   Expects each axis to be clamped against its own halo and grid extent.

TestOffset
^^^^^^^^^^

``test_offset_places_the_local_block_in_the_global_output``
   Expects ``(global(start) - global_start) // step``: the clamped start is
   two cells in, at a step of two, so this rank's block begins at index 1 of
   the global dataset.

``test_offset_is_zero_when_nothing_was_clamped``
   Expects a rank owning the whole view to write from index 0.

``test_offset_is_measured_in_output_cells_not_grid_cells``
   Expects the division by ``step``, so a stride-3 view moved three grid
   cells shifts by one output cell.

``test_offset_plus_size_fits_inside_the_global_size``
   Expects this rank's block to lie wholly within the global dataset — an
   offset or size overshoot would corrupt a neighbour's data.

TestGetterSliceOverride
^^^^^^^^^^^^^^^^^^^^^^^

The extra node is suppressed when a neighbour will supply it.

``test_exclusive_bound_behaves_as_in_the_base_class``
   Expects ``slice(start, stop, step)`` — unchanged from serial.

``test_inclusive_bound_is_suppressed_with_a_positive_neighbour``
   Expects *no* extra step: the node past this rank's edge belongs to the
   neighbour, which will contribute it. Taking it here would double-count.

``test_inclusive_bound_applies_at_the_domain_edge``
   Expects the extra step to be taken when there is no neighbour — matching
   the serial class exactly.

``test_the_decision_is_per_axis``
   Expects the extra node on an axis at the domain edge and not on one
   abutting a neighbour, within the same view.

TestSetterSliceOverride
^^^^^^^^^^^^^^^^^^^^^^^

Reading back extends *downward* into the negative halo.

``test_start_reaches_back_one_step_with_a_negative_neighbour``
   Expects ``start - step``: when writing data in, this rank must fill its
   own halo cell too, so the neighbour's edge value is present locally for
   the next field update.

``test_start_is_unchanged_at_the_domain_edge``
   Expects no reach-back where there is no neighbour below.

``test_the_inclusive_bound_always_extends``
   Expects ``stop + step`` regardless of the positive neighbour — the
   setter, unlike the getter, does not suppress the extra node.

``test_setter_and_getter_slices_diverge``
   Expects the two to differ once halos are in play.

   In the serial class ``setter_slice`` is literally ``getter_slice``. This
   is the assertion that shows the override is doing something.

TestOutputSliceOverride
^^^^^^^^^^^^^^^^^^^^^^^

``test_starts_at_the_rank_offset``
   Expects ``slice(offset, offset + size)`` rather than starting at zero —
   each rank writes into its own region of a shared dataset.

``test_the_extra_node_is_suppressed_with_a_positive_neighbour``
   Expects no ``+1``, matching ``getter_slice``: the two must agree or the
   write would be shape-mismatched.

``test_the_extra_node_applies_at_the_domain_edge``
   Expects ``size + 1`` where no neighbour contributes it.

``test_output_slice_length_matches_the_getter_slice_length``
   Expects the destination window and the source data to be the same length
   — the invariant that makes ``dset[out] = get_solid()`` valid.

TestReadSliceOverride
^^^^^^^^^^^^^^^^^^^^^

``test_reaches_back_with_a_negative_neighbour``
   Expects ``offset - 1`` and one extra element, so the halo cell is read
   back along with the owned block.

``test_is_unchanged_at_the_domain_edge``
   Expects a plain ``slice(0, size)`` where there is no halo to fill.

``test_the_inclusive_bound_always_extends``
   Expects ``+1`` for the inclusive bound regardless of the positive
   neighbour, mirroring ``setter_slice``.

``test_read_slice_length_matches_the_setter_slice_length``
   Expects the source window and the destination to agree, the mirror of the
   output/getter invariant.

``test_read_and_output_slices_diverge``
   Expects the two to differ — in the serial class they are the same
   function, so this is the override's fingerprint.

TestDegeneratesToTheSerialClass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With no halo and no overrun, MPI and serial must agree exactly.

``test_getter_slices_match``
   Expects identical getter slices on both bound conventions. (3 parameter
   sets)

``test_setter_slices_match``
   Expects identical setter slices. (3 parameter sets)

``test_output_slices_match``
   Expects identical output slices. (3 parameter sets)

``test_sizes_match``
   Expects the same cell counts from both classes.

TestMaterialsAcrossRanks
^^^^^^^^^^^^^^^^^^^^^^^^

``initialise_materials`` is collective; at one rank it must still work.

``test_the_coordinator_collects_every_material``
   Expects rank 0 to end up holding the deduplicated union — with one rank
   that is simply its own list.

``test_filtering_keeps_only_what_this_rank_sees``
   Expects an all-free-space block to report one material even though the
   grid defines three.

``test_builds_a_working_map``
   Expects the local-to-global map to renumber the single visible material
   to index 0.

``test_the_map_preserves_dtype``
   Expects ``uint32`` to survive the ``np.vectorize`` round trip, as in the
   serial class.

``test_materials_are_deduplicated_and_sorted``
   Expects ``np.unique`` ordering on the gathered union, so every rank
   agrees on the global numbering.

When these fail
~~~~~~~~~~~~~~~

**``AssertionError`` from inside ``MPIGridView.__init__``.** Line 561
asserts ``isinstance(comm, MPI.Intracomm)``. Mock communicators are rejected
outright. The fixtures hand it a **real** ``MPI.COMM_SELF`` and fake only
the *grid* — that is the whole strategy for this file, and it works because
the clamping arithmetic depends on ``negative_halo_offset`` and
``grid.size``, not on rank count.

**A clamped bound is off by a cell.** Both clamps preserve step alignment
through a modulo: ``halo + ((start - halo) % step)`` and ``grid.size +
((stop - grid.size) % step)``. The second can land *past* the grid size.
Snapping naively to the boundary shifts every exported sample.

**A slice length mismatch.** Four invariants hold pairwise and are tested
directly: ``get_output_slice`` must match ``getter_slice`` in length, and
``get_read_slice`` must match ``setter_slice``. Breaking either makes
``dset[out] = get_solid()`` raise a shape error at write time.

**``global_size`` disagrees with ``size``.** That is expected on a clamped
view. ``global_size`` is captured *before* the local clamp and describes the
whole collective dataset; ``size`` is this rank's share.

**A test aborts the whole process.** Running under
``MPI4PY_RC_INITIALIZE=0`` gives *"Attempting to use an MPI routine before
initializing MPI"*. That variable was removed from the CI workflow in this
PR for exactly this reason.

Test Catalog — ``test_snapshots.py``
------------------------------------

**67 tests** from 51 test functions across 9 classes.

``Snapshot`` — freezing the fields inside a window at one iteration.

A snapshot is a ``GridView`` plus six output flags plus a time. Construction
and property access are thin delegation; the interesting part is
``store()``, which drives a real OpenMP Cython kernel.

**Why the kernel averages, and why 4 versus 2.** In a Yee lattice the six
field components do not live at the same place. An electric component sits
on a cell *edge*, offset from the cell centre along the two axes that are
not its own; a magnetic component sits on a cell *face*, offset along only
its own axis. Writing them out unaveraged would give six pictures of six
slightly different places. So the kernel interpolates each component onto
the cell centre:

Exsnap[i,j,k] = (Ex[i,j,k] + Ex[i,j+1,k] + Ex[i,j,k+1] + Ex[i,j+1,k+1]) / 4
Hxsnap[i,j,k] = (Hx[i,j,k] + Hx[i+1,j,k]) / 2

Four neighbours for E, two for H — and which axes are stepped differs per
component. That is the whole Yee convention in six lines, and it is why
``GridView.get_Ex()`` fetches one extra node in every direction.

The tests below pin the ratio, the axes, and the dependence on the extra
node, by filling the grid with fields whose values vary along exactly one
axis at a time.

TestClassSurface
^^^^^^^^^^^^^^^^

``test_six_allowable_outputs``
   Expects exactly the six field components — snapshots have no current
   outputs, unlike receivers.

``test_two_file_extensions``
   Expects ``.vtkhdf`` and ``.h5``, the two formats ``write_file``
   dispatches on.

``test_max_dimensions_start_at_zero``
   Expects the shared GPU sizing attributes to begin cleared.

   These are *class* attributes mutated by ``htod_snapshot_array``, so the
   suite resets them between tests — see the conftest.

``test_default_threads_per_block``
   Expects ``(1, 1, 1)`` until a GPU run overrides it.

``test_the_grid_view_type_is_the_serial_one``
   Expects a plain ``GridView``; ``MPISnapshot`` overrides this to return
   ``MPIGridView``.

TestConstruction
^^^^^^^^^^^^^^^^

``test_builds_a_grid_view_from_the_extents``
   Expects the nine coordinate arguments to be handed straight to a
   ``GridView`` — the snapshot itself stores no coordinates.

``test_stores_the_iteration``
   Expects ``time`` to be the iteration index, kept verbatim.

``test_stores_the_output_flags``
   Expects the outputs dict as given, so a caller can request a subset.

``test_stores_the_file_extension``
   Expects ``fileext`` kept separately from the filename — it is what
   ``write_file`` dispatches on.

``test_byte_count_starts_at_zero``
   Expects ``nbytes`` to be zero before ``initialise_snapfields``
   accumulates it.

``test_snapfields_starts_empty``
   Expects no arrays until explicitly initialised.

``test_no_validation_is_performed``
   Expects construction to accept an inverted extent without complaint — all
   geometry checking lives in the user-object layer, which PR 6 covers, not
   here.

``test_grid_is_reached_through_the_view``
   Expects ``snap.grid`` to be a property forwarding to ``grid_view.grid``
   rather than a stored reference.

TestFilename
^^^^^^^^^^^^

``test_the_extension_is_applied``
   Expects ``snap1`` plus ``.h5``.

``test_an_existing_suffix_is_replaced_not_appended``
   Expects ``with_suffix`` semantics: ``snap.vtkhdf`` with ``.h5`` becomes
   ``snap.h5``, **not** ``snap.vtkhdf.h5``.

   Worth pinning — a user filename containing a dot is silently truncated at
   that dot.

``test_a_dotted_name_loses_everything_after_the_dot``
   Expects ``run.2.field`` to become ``run.2.h5`` — only the final component
   is treated as a suffix.

``test_directory_components_are_preserved``
   Expects only the final path component to be re-suffixed.

``test_the_result_is_a_path``
   Expects a ``Path``, since ``save_snapshots`` later joins a directory onto
   it.

TestDelegatedProperties
^^^^^^^^^^^^^^^^^^^^^^^

``test_each_property_forwards_to_the_grid_view``
   Expects all twelve accessors to return the view's value, so the snapshot
   holds no duplicate coordinate state. (12 parameter sets)

``test_sizes_follow_the_ceiling_rule``
   Expects ``nx == ceil((xf-xs)/dx)``: a 0-to-10 window at step 3 is four
   cells, inherited straight from ``GridView``.

``test_axes_are_independent``
   Expects three different steps to give three different sizes.

TestInitialiseSnapfields
^^^^^^^^^^^^^^^^^^^^^^^^

``test_requested_outputs_get_full_sized_arrays``
   Expects one array of the view's shape per requested component.

``test_unrequested_outputs_get_a_dummy_array``
   Expects a ``(1, 1, 1)`` placeholder rather than no entry at all.

   The Cython kernel takes all twelve arrays positionally whatever the flags
   say, so every key must exist — but there is no reason to allocate a full
   volume for a component nobody asked for.

``test_all_six_keys_are_present_regardless``
   Expects every component to have an entry even when none were requested.

``test_arrays_start_zeroed``
   Expects a clean buffer — ``store()`` writes every cell, but a partially-
   requested snapshot leaves the dummies untouched.

``test_uses_the_configured_float_dtype``
   Expects ``float64`` under the double-precision fixture, matching the
   grid's own field arrays — a mismatch would be rejected by the kernel's
   fused-type memoryview.

``test_nbytes_counts_only_requested_outputs``
   Expects the dummy arrays to be excluded, so the progress bar totals what
   will actually be written.

``test_nbytes_sums_across_components``
   Expects six requested components to total six times one.

``test_calling_twice_double_counts``
   Expects ``nbytes`` to accumulate rather than reset — the method is
   written to be called exactly once, and callers do.

TestStoreAveraging
^^^^^^^^^^^^^^^^^^

The real Cython kernel, against fields varying on one axis at a time.

``test_a_constant_field_is_unchanged``
   Expects averaging four equal values to give that value — the simplest
   possible check that the divisor is right.

``test_ex_averages_over_y_and_z``
   Expects ``Ex`` to average its two *transverse* axes. A field varying only
   along x must therefore pass through untouched.

``test_ex_smooths_a_ramp_along_y``
   Expects the midpoint of adjacent y samples: a ramp ``0,1,2,3,4`` along y
   averages to ``0.5, 1.5, 2.5, 3.5``.

``test_ey_averages_over_x_and_z``
   Expects ``Ey`` to leave a y-varying field alone, by symmetry with ``Ex``.

``test_ez_averages_over_x_and_y``
   Expects ``Ez`` to leave a z-varying field alone.

``test_hx_averages_over_x_only``
   Expects a magnetic component to be averaged along its *own* axis — the
   opposite convention from the electric ones.

``test_hx_ignores_the_transverse_axes``
   Expects a y-varying ``Hx`` to pass through unchanged.

``test_hy_averages_over_y_only``
   Expects the y analogue of the ``Hx`` case.

``test_hz_averages_over_z_only``
   Expects the z analogue.

``test_electric_averaging_uses_four_neighbours``
   Expects a single unit spike to be spread over four cells at ¼ each — the
   defining signature of the electric stencil.

``test_magnetic_averaging_uses_two_neighbours``
   Expects a spike to be spread over two cells at ½ each.

``test_the_extra_node_is_read``
   Expects the last output cell to depend on the node *past* the view's stop
   bound.

   This is why ``get_Ex`` fetches with ``upper_bound_exclusive=False``.
   Placing a value only at index 4 of a 0-to-4 view still changes the result
   at output index 3.

TestStoreFlags
^^^^^^^^^^^^^^

``test_an_unrequested_component_is_not_written``
   Expects the dummy array to stay zeroed when the flag is false — the
   kernel skips the whole assignment.

``test_a_requested_component_is_written``
   Expects the ramp-filled grid to produce non-zero output.

``test_each_component_can_be_requested_alone``
   Expects the six flags to be independent, so requesting one leaves the
   other five untouched. (6 parameter sets)

``test_store_returns_none``
   Expects in-place population of ``snapfields`` rather than a return value.

``test_store_is_repeatable``
   Expects a second call on unchanged fields to give the same answer — the
   kernel assigns rather than accumulates.

``test_a_later_store_picks_up_new_field_values``
   Expects the slices to be re-fetched each call, so a snapshot object
   reused across iterations sees current data.

TestStoreWithStride
^^^^^^^^^^^^^^^^^^^

``test_a_strided_snapshot_samples_rather_than_averages_the_gap``
   Expects a step-2 view to produce half-sized output, with each cell still
   averaged from its own immediate neighbours rather than over the skipped
   cells.

``test_output_shape_follows_the_ceiling_rule``
   Expects a non-dividing strided view to keep its partial final cell.

``test_an_offset_window_reads_the_right_region``
   Expects a window starting at 2 to average cells 2 and 3, not 0 and 1.

TestSinglePrecision
^^^^^^^^^^^^^^^^^^^

``test_store_works_with_float32_arrays``
   Expects the kernel's fused ``float_or_double`` type to bind to the
   single-precision specialisation when the run is configured for it.

   Both the grid arrays and the snapshot buffers must agree; the shared
   ``config`` key guarantees they do.

When these fail
~~~~~~~~~~~~~~~

**A snapshot is shifted half a cell.** The kernel averages **four**
neighbours for each electric component and **two** for each magnetic one,
and the axes stepped differ per component: ``Ex`` over y and z, ``Hx`` over
x. The tests fill the grid with a field varying along one axis at a time so
each stencil is isolated. Get the axes wrong and the shape tests still pass.

**``KeyError`` from ``store()``.** ``initialise_snapfields()`` must run
first. ``store()`` indexes all six keys of both ``outputs`` and
``snapfields`` with hardcoded literals, whatever the flags say.

**A dtype error from the Cython call.** The grid's field arrays and the
snapshot buffers must be the same float width. Both come from
``config.sim_config.dtypes["float_or_double"]``, so patch that one key
rather than either array.

**The last output cell is wrong.** ``get_Ex`` over-fetches by one node in
every direction because the stencil reaches past the view's stop bound.
``test_the_extra_node_is_read`` pins it.

**A ``nbytes`` assertion is double what you expect.**
``initialise_snapfields`` accumulates rather than resets, so calling it
twice doubles the count. It is written to be called once.

Test Catalog — ``test_snapshot_files.py``
-----------------------------------------

**43 tests** from 38 test functions across 6 classes.

Writing snapshots to disk — both formats, round-tripped through
``tmp_path``.

Every test here writes a real file and reads it back with ``h5py``. Nothing
is mocked, because the on-disk layout *is* the contract: dataset names,
shapes, dtypes and attributes are what ParaView and every downstream
analysis script depend on, and a mock-based test would confirm only that a
method was called.

Two formats share one dispatcher:

- ``.h5`` — a flat HDF5 file with one root dataset per requested component

- ``.vtkhdf`` — VTK ImageData, written through
  ``gprMax/vtkhdf_filehandlers/``, with cell data under ``VTKHDF/CellData``
  and the geometry carried in root attributes

``vtkhdf_filehandlers`` belongs to a later PR; here it is exercised
transitively, which is the point — these tests establish what the current
layout is so that PR 11 can refactor the writer with something to check
against.

TestWriteFileDispatch
^^^^^^^^^^^^^^^^^^^^^

``test_h5_extension_produces_an_hdf5_file``
   Expects ``.h5`` to route to ``write_hdf5``, leaving a readable file.

``test_vtkhdf_extension_produces_a_vtk_file``
   Expects ``.vtkhdf`` to route to ``write_vtk``, producing the VTKHDF group
   structure rather than root datasets.

``test_the_two_formats_write_different_layouts``
   Expects the same field data to land at different paths in the two formats
   — root-level in HDF5, under ``VTKHDF/CellData`` in VTK.

TestHdf5Attributes
^^^^^^^^^^^^^^^^^^

``test_records_the_gprmax_version``
   Expects the writing version stamped at the root, so a file can be traced
   to the build that produced it.

``test_records_the_cell_counts``
   Expects ``nx_ny_nz`` to be the *view's* size, not the grid's.

``test_records_the_physical_spacing``
   Expects ``dx_dy_dz == step * grid.dl`` — the physical size of one
   snapshot cell, which for a strided snapshot is larger than one grid cell.

``test_spacing_follows_an_anisotropic_grid``
   Expects each axis to take its own discretisation.

``test_records_the_simulation_time_not_the_iteration``
   Expects ``time == iteration * dt`` in seconds. The constructor takes an
   iteration index; the file carries physical time.

``test_time_zero_is_written``
   Expects a snapshot at iteration 0 to record ``0.0`` rather than omitting
   the attribute.

TestHdf5Datasets
^^^^^^^^^^^^^^^^

``test_requested_components_become_root_datasets``
   Expects one dataset per requested component, named for it.

``test_unrequested_components_are_absent``
   Expects the dummy ``(1,1,1)`` placeholder arrays *not* to be written —
   they exist only to satisfy the Cython signature.

``test_each_component_can_be_written_alone``
   Expects the six flags to be independent at the file level. (6 parameter
   sets)

``test_dataset_shape_is_the_view_size``
   Expects ``(nx, ny, nz)`` of the view.

``test_dataset_dtype_matches_the_configured_precision``
   Expects ``float64`` under the double-precision fixture.

``test_values_match_the_in_memory_snapfields``
   Expects the file to carry exactly what ``store()`` computed — no scaling,
   reordering or truncation on the way out.

``test_a_strided_snapshot_writes_the_reduced_shape``
   Expects the output to be the number of *snapshot* cells.

TestVtkLayout
^^^^^^^^^^^^^

``test_cell_data_is_named_for_the_component``
   Expects each requested component under ``VTKHDF/CellData``.

``test_unrequested_components_are_absent``
   Expects the same flag filtering as the HDF5 writer.

``test_declares_the_image_data_type``
   Expects ``Type == b"ImageData"`` on the ``VTKHDF`` group — the marker
   that tells VTK how to read the file.

``test_whole_extent_spans_the_view``
   Expects ``[0, nx, 0, ny, 0, nz]`` — the extent is relative to the
   snapshot's own origin, not to the grid.

``test_origin_is_the_physical_start_of_the_window``
   Expects ``start * grid.dl`` in metres, so a snapshot of a sub-region
   lands in the right place when overlaid on the full model.

``test_origin_follows_an_anisotropic_grid``
   Expects each axis scaled by its own discretisation.

``test_spacing_is_the_physical_cell_size``
   Expects ``step * grid.dl``, matching the HDF5 writer's ``dx_dy_dz``.

``test_direction_defaults_to_the_identity``
   Expects an axis-aligned lattice — gprMax never rotates a snapshot.

``test_values_are_written_in_zyx_order``
   Expects the array **transposed** relative to ``snapfields``.

   The VTKHDF specification stores datasets ZYX-major, so the writer
   transposes on the way out. This is the one place in the PR where the on-
   disk array is not element-for-element what was computed, and reading such
   a file back without transposing gives a model reflected through its main
   diagonal — plausible-looking and completely wrong.

``test_the_transpose_is_visible_in_the_dataset_shape``
   Expects ``(nz, ny, nx)`` on disk for an ``(nx, ny, nz)`` snapshot — the
   cheapest way to notice the reordering.

``test_the_hdf5_writer_does_not_transpose``
   Expects the plain ``.h5`` format to store ``(nx, ny, nz)`` as computed —
   the two formats genuinely differ here, and a script reading both must
   account for it.

TestProgressReporting
^^^^^^^^^^^^^^^^^^^^^

``test_bytes_are_reported_per_component``
   Expects one ``update`` call per written component, each carrying that
   array's byte count.

``test_reported_bytes_total_nbytes``
   Expects the progress total to match the ``nbytes`` the bar was sized with
   — otherwise the bar finishes short or overruns.

``test_unrequested_components_are_not_reported``
   Expects no progress update for a component that is not written.

TestSaveSnapshots
^^^^^^^^^^^^^^^^^

The orchestrator: make a directory, relocate each file, write it.

``test_creates_the_snapshot_directory``
   Expects the directory from ``set_snapshots_dir()`` to be created if
   absent.

``test_tolerates_an_existing_directory``
   Expects ``mkdir(exist_ok=True)`` semantics, so a second model in the same
   run does not fail.

``test_relocates_each_file_into_that_directory``
   Expects the snapshot directory to be prepended to each filename — the
   snapshot is constructed with a bare name and only learns its directory
   here.

``test_writes_every_snapshot``
   Expects one file per snapshot in the list.

``test_the_written_files_are_readable``
   Expects a complete, valid file rather than a truncated one — a writer
   that never closes its handle could leave the last one short.

``test_an_empty_list_writes_nothing``
   Expects the directory to be created but left empty — a model with no
   snapshots must not fail here.

``test_logs_the_directory``
   Expects the resolved path in the log, since it is the only place the user
   learns where the files went.

``test_vtk_snapshots_are_saved_too``
   Expects the orchestrator to be format-agnostic — it defers to
   ``write_file``.

When these fail
~~~~~~~~~~~~~~~

**A VTK value comparison fails by what looks like a transpose.** It *is* a
transpose. The VTKHDF specification stores datasets ZYX-major, so
``write_vtk`` transposes on the way out while ``write_hdf5`` does not. This
is the one place in the PR where the on-disk array is not element-for-
element what was computed. Compare against ``snapfields[name].T`` for VTK
and against ``snapfields[name]`` for HDF5.

**No file was written and nothing raised.** Check ``fileext``.
``write_file``'s ``if``/``elif`` has no ``else``, so an unrecognised
extension is a silent no-op — see `notes/bugs/snapshot-write-file-silent-no-
op.md`.

**A progress-bar total disagrees with ``nbytes``.** Only *requested*
components are written and reported; the ``(1, 1, 1)`` placeholder arrays
are neither.

**``save_snapshots`` writes to the wrong place.** It prepends
``get_model_config().set_snapshots_dir()`` to each snapshot's filename, so
the filename changes as a side effect of saving. A snapshot saved twice
would be nested one directory deeper.

Test Catalog — ``test_snapshot_devices.py``
-------------------------------------------

**45 tests** from 40 test functions across 7 classes.

Snapshot device transfer and the MPI snapshot.

Two areas that share nothing except living in ``snapshots.py``.

**``htod_snapshot_array`` / ``dtoh_snapshot_array``** move snapshot buffers
to and from an accelerator. The host-side half — deciding how large the
shared device array must be, and how many time slices it holds — is plain
numpy and fully testable here. The device half branches on the solver name
and imports ``pycuda`` / ``pyopencl`` inside the branch, so those paths are
driven with injected stand-in modules and a stand-in Metal device. That
tests the wiring, not the hardware; PR 12 covers the accelerators
themselves.

``Snapshot.nx_max``/``ny_max``/``nz_max`` are **class** attributes, sized to
the largest snapshot in the model so one device allocation serves them all.
They are therefore global mutable state, and the suite's autouse fixture
restores them after each test — without that, a large snapshot in one test
silently enlarges every allocation in the next.

**``MPISnapshot``** overrides the grid-view type, records its Cartesian
neighbours, and exchanges halo data before averaging. At one rank there are
no neighbours, so what is established here is the wiring and the degenerate-
case agreement with the serial class. The ``driver="mpio"`` write path
cannot run at all in this environment — ``h5py.get_config().mpi`` is
``False`` — so it is guarded rather than faked.

TestMaximumDimensions
^^^^^^^^^^^^^^^^^^^^^

``test_records_the_largest_snapshot``
   Expects the class-level maxima to end up at the largest extent seen on
   each axis — one device allocation has to fit every snapshot.

``test_maxima_are_taken_per_axis``
   Expects an axis-wise maximum rather than the single largest snapshot: a
   tall thin snapshot and a short wide one together demand a box big enough
   for both.

``test_a_single_snapshot_sets_its_own_size``
   Expects the common case to be exact rather than padded.

``test_the_maxima_only_grow``
   Expects a second call with smaller snapshots to leave the maxima alone —
   the comparison is one-sided.

   This is exactly why the suite resets these between tests: without the
   reset, the values would ratchet upward across the whole session.

``test_they_are_class_level_not_instance_level``
   Expects the sizing to be visible on the class itself, and therefore on
   every other snapshot in the process.

TestDeviceArrayShape
^^^^^^^^^^^^^^^^^^^^

``test_one_time_slice_when_copying_back_each_iteration``
   Expects a leading axis of 1 when ``snapsgpu2cpu`` is set: the device
   holds one snapshot at a time and the host takes each away.

``test_one_time_slice_per_snapshot_when_kept_on_device``
   Expects a leading axis equal to the snapshot count when they are all
   retained on the accelerator.

``test_spatial_axes_use_the_maxima``
   Expects the allocation to be sized by the largest snapshot, so every
   snapshot fits in the shared buffer.

``test_six_buffers_are_allocated``
   Expects one per field component.

``test_buffers_use_the_configured_precision``
   Expects ``float64`` under the double-precision fixture.

``test_byte_length_matches_the_array``
   Expects the length handed to the device to be the array's own ``nbytes``
   — a mismatch would truncate or overrun the buffer.

``test_returns_one_handle_per_component``
   Expects a six-tuple in ``Ex, Ey, Ez, Hx, Hy, Hz`` order.

TestSolverDispatch
^^^^^^^^^^^^^^^^^^

``test_cuda_sets_blocks_per_grid``
   Expects ``bpg`` sized from the total cell count over the threads per
   block, as a 3-tuple with singleton y and z.

``test_cuda_uploads_every_component``
   Expects six ``to_gpu`` calls, one per field.

``test_opencl_sets_a_workgroup_size``
   Expects ``wgs`` to be the plain cell count.

   Note the asymmetry: CUDA writes ``bpg``, OpenCL writes ``wgs``, and Metal
   writes neither. The three backends do not share an attribute.

``test_opencl_passes_the_queue_through``
   Expects the caller's queue to reach ``to_device`` — CUDA and Metal take
   no queue, OpenCL does.

``test_metal_reads_its_device_from_config``
   Expects the Metal branch to fetch the device from
   ``get_model_config().device["dev"]`` rather than from an argument.

``test_metal_sets_neither_bpg_nor_wgs``
   Expects the Metal path to leave the CUDA and OpenCL sizing attributes
   untouched, since it dispatches differently.

TestDtohSnapshotArray
^^^^^^^^^^^^^^^^^^^^^

Pure slicing: pull one snapshot's window out of the shared device array.

``test_populates_all_six_components``
   Expects every entry of ``snapfields`` to be replaced.

``test_each_component_reads_its_own_buffer``
   Expects the six buffers to map to the six components in order — the
   offsets make a crossed pair impossible to miss.

``test_extracts_the_snapshot_window``
   Expects the ``xs:xf`` window, so a snapshot from 1 to 3 gives a 2-cube.

``test_selects_the_requested_time_index``
   Expects the leading index to choose which stored snapshot is pulled back.

``test_values_come_from_the_right_cells``
   Expects an exact match against the same slice taken by hand.

``test_returns_none``
   Expects in-place mutation of ``snapfields``.

TestMpiSnapshotConstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_extends_the_serial_snapshot``
   Expects the whole serial surface to be inherited.

``test_uses_an_mpi_grid_view``
   Expects ``GRID_VIEW_TYPE`` to be overridden, so the snapshot's coordinate
   arithmetic is halo-aware.

``test_asserts_the_view_type``
   Expects the explicit ``assert isinstance`` in ``__init__`` to hold — it
   is what guarantees ``self.comm`` exists.

``test_takes_its_communicator_from_the_view``
   Expects the Cartesian communicator built by ``MPIGridView`` to be reused
   rather than a second one created.

``test_records_neighbours_on_three_axes``
   Expects a ``(3, 2)`` table — two directions on each of three axes.

``test_a_single_rank_has_no_neighbours``
   Expects every entry negative: ``Cartcomm.Shift`` returns
   ``MPI.PROC_NULL`` where there is no neighbour, and ``has_neighbour``
   tests for a non-negative rank.

``test_has_neighbour_is_false_at_one_rank``
   Expects ``has_neighbour`` to report ``False`` on every face of a single-
   rank domain. (6 parameter sets)

``test_distinct_message_tags``
   Expects four distinct tags, so the halo exchanges for H and the three E
   components cannot be confused with one another.

TestMpiSnapshotStore
^^^^^^^^^^^^^^^^^^^^

``test_stores_without_neighbours``
   Expects a single-rank store to complete: with no neighbours every halo
   exchange is skipped and the local averaging runs alone.

``test_a_constant_field_survives_averaging``
   Expects the same answer as the serial path for a uniform field —
   averaging equal values changes nothing however the domain is split.

``test_logs_the_iteration_at_debug``
   Expects a debug record naming the iteration, the only trace the MPI store
   leaves.

TestMpiSnapshotWriting
^^^^^^^^^^^^^^^^^^^^^^

Both MPI write paths need parallel HDF5, not just the ``.h5`` one.

``MPISnapshot.write_hdf5`` opens ``h5py.File(..., driver="mpio")`` directly,
and ``write_vtk`` reaches the same call through ``VtkImageData(...,
comm=...)``. Where ``h5py`` is built without MPI support — this environment
and CI both — the open raises before any gprMax logic runs, so there is
nothing these tests could assert. They ship guarded so they execute wherever
parallel HDF5 does exist.

What *is* covered unconditionally is everything upstream of the write:
construction, neighbour discovery, the halo-free store, and the global
versus local size arithmetic in ``TestMpiGridView``.

``test_vtk_write_uses_global_dimensions``
   Expects ``WholeExtent`` to describe the *whole* view rather than this
   rank's share, so all ranks agree on one dataset shape.

``test_vtk_write_places_data_at_the_rank_offset``
   Expects the local block written at ``grid_view.offset``; at one rank that
   offset is zero and the whole dataset is filled.

``test_hdf5_write_uses_the_mpio_driver``
   Expects a parallel-HDF5 write producing one shared file.

``test_the_mpio_driver_is_genuinely_unavailable_here``
   Expects ``h5py.get_config().mpi`` to be the thing gating the three tests
   above, recorded explicitly so the skips are not mistaken for an
   oversight.

``test_global_size_is_what_the_writers_would_use``
   Expects the size the guarded writers pass to HDF5 to be computed
   correctly even though the write itself cannot run — the arithmetic is
   reachable, only the I/O is not.

When these fail
~~~~~~~~~~~~~~~

**A device-array size assertion fails, and the number looks like a previous
test's.** ``Snapshot.nx_max``/``ny_max``/``nz_max`` are **class** attributes
that only ever grow, and nothing in production resets them. The suite's
autouse ``reset_snapshot_class_state`` fixture restores them; if it is
removed or bypassed, one test silently enlarges every later allocation. See
`notes/bugs/snapshot-max-dimensions-class-state.md`.

**``UnboundLocalError`` from ``htod_snapshot_array``.** The solver is not
one of cuda / opencl / metal. There is no ``else``, so none of the six
``*_dev`` names is bound — see `notes/bugs/htod-snapshot-array-unbound-
locals.md`. The tests drive the Metal branch (which needs no third-party
import) and inject stand-in modules for the other two.

**Three tests are skipped.** Both MPI write paths need parallel HDF5.
``MPISnapshot.write_hdf5`` opens with ``driver="mpio"`` directly and
``write_vtk`` reaches the same call through ``VtkImageData(..., comm=...)``.
``h5py.get_config().mpi`` is ``False`` here and on CI, so the open raises
before any gprMax logic runs. This is an environment limit, not a gap in the
suite.

**``Snapshot.bpg`` is set when you expected ``wgs``.** CUDA writes ``bpg``,
OpenCL writes ``wgs`` (never declared on the class), Metal writes neither.

Test Catalog — ``test_fields_outputs.py``
-----------------------------------------

**76 tests** from 57 test functions across 10 classes.

``fields_outputs.py`` — the receiver traces a GPR user actually plots.

The smallest file in the PR and the closest to the end user. Three jobs:

- ``store_outputs`` runs once per iteration, copying one field value per
  receiver into a growing time series;

- ``Ix``/``Iy``/``Iz`` compute a current from a contour of magnetic field
  values, for receivers that asked for a current rather than a field;

- ``write_hdf5_outputfile`` writes the whole lot at the end — receiver
  traces, source positions, transmission-line voltages and currents, and one
  group per subgrid.

**The duplicated current formulas.** ``Ix``/``Iy``/``Iz`` here are a second
implementation of ``FDTDGrid.calculate_Ix``/``Iy``/``Iz``, tested in PR 9.
Nothing in the codebase checks that the two agree, and a fix applied to one
would silently leave the other behind, so a cross-check is included below.

**Receivers must be named.** ``write_hd5_data`` sorts ``grid.rxs`` by
``rx.ID``, but ``Rx.__init__`` only *annotates* ``self.ID: str`` — it never
assigns it. An unnamed receiver therefore raises ``AttributeError`` from
inside the writer. Every receiver built here is given an explicit ID; the
defect is recorded for the maintainers rather than asserted.

TestCurrentBoundaryGuards
^^^^^^^^^^^^^^^^^^^^^^^^^

Each component returns exactly zero on the two faces it cannot close a
contour around.

``test_ix_is_zero_on_its_guarded_faces``
   Expects ``Ix == 0`` whenever ``y == 0`` or ``z == 0`` — the contour would
   need a cell outside the domain. (3 parameter sets)

``test_iy_is_zero_on_its_guarded_faces``
   Expects ``Iy == 0`` whenever ``x == 0`` or ``z == 0``. (3 parameter sets)

``test_iz_is_zero_on_its_guarded_faces``
   Expects ``Iz == 0`` whenever ``x == 0`` or ``y == 0``. (3 parameter sets)

``test_each_component_guards_the_two_axes_that_are_not_its_own``
   Expects ``Ix`` to be unguarded in x: a current along x is computed from a
   contour in the y-z plane, so ``x == 0`` is perfectly fine.

``test_the_guard_returns_an_integer_zero``
   Expects a plain ``0`` rather than an array or ``0.0`` — it is assigned
   straight into a float time series either way, but the literal is what the
   code returns.

TestCurrentFormulas
^^^^^^^^^^^^^^^^^^^

``test_ix_matches_the_contour_sum``
   Expects ``dy·(Hy[x,y,z-1] - Hy[x,y,z]) + dz·(Hz[x,y,z] - Hz[x,y-1,z])`` —
   a loop around the x-directed cell edge.

``test_iy_matches_the_contour_sum``
   Expects ``dx·(Hx[x,y,z] - Hx[x,y,z-1]) + dz·(Hz[x-1,y,z] - Hz[x,y,z])``.

``test_iz_matches_the_contour_sum``
   Expects ``dx·(Hx[x,y-1,z] - Hx[x,y,z]) + dy·(Hy[x,y,z] - Hy[x-1,y,z])``.

``test_a_uniform_field_gives_no_current``
   Expects zero from a constant magnetic field: every difference in the
   contour cancels. This is the physical sanity check — no curl, no current.

``test_each_term_is_weighted_by_its_own_spacing``
   Expects the anisotropic ``dy`` and ``dz`` to scale their own terms — a
   swapped pair would change the answer by a factor of two here.

``test_currents_read_only_the_magnetic_field``
   Expects no dependence on the electric field — the grid argument supplies
   spacings only.

TestAgreementWithTheGridImplementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``FDTDGrid`` carries its own copy of these three formulas.

``test_ix_agrees``
   Expects the module function and ``FDTDGrid.calculate_Ix`` to give
   identical answers.

   The two are independent copies of the same algebra. Nothing else in the
   suite would notice them drifting apart. (3 parameter sets)

``test_iy_agrees``
   Expects agreement with ``FDTDGrid.calculate_Iy``. (3 parameter sets)

``test_iz_agrees``
   Expects agreement with ``FDTDGrid.calculate_Iz``. (3 parameter sets)

``test_the_boundary_guards_agree_too``
   Expects both implementations to return zero on the guarded faces.

TestStoreOutputs
^^^^^^^^^^^^^^^^

``test_copies_a_field_value_into_the_time_series``
   Expects the receiver's own cell of the named field array to land at index
   ``iteration``.

``test_writes_at_the_requested_iteration``
   Expects index 3 to be written and the earlier slots left alone.

``test_each_field_component_is_resolved_by_name``
   Expects the output key to select the matching grid array. The lookup goes
   through ``locals()``, so the six local names must match the six allowable
   outputs exactly. (6 parameter sets)

``test_current_outputs_are_dispatched_to_the_module_functions``
   Expects a key containing ``I`` to route to the module-level function of
   the same name rather than to a grid array. (3 parameter sets)

``test_multiple_outputs_on_one_receiver``
   Expects every requested output to be filled in a single pass.

``test_multiple_receivers_are_all_stored``
   Expects each receiver to read its own cell.

``test_successive_iterations_build_a_series``
   Expects a changing field to produce a changing trace — the point of the
   whole function.

``test_transmission_line_totals_are_sampled_at_the_antenna``
   Expects ``Vtotal``/``Itotal`` to take the line's voltage and current at
   ``antpos``, not at index 0.

``test_a_grid_with_no_receivers_is_a_no_op``
   Expects no error when nothing is being recorded.

TestWriteOutputFileTopLevel
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_records_the_gprmax_version``
   Expects the writing version at the file root.

``test_records_the_title``
   Expects the model title as given.

``test_records_the_iteration_count``
   Expects ``Iterations`` from the *model*, since it is the model that owns
   the time window.

``test_records_the_source_and_receiver_steps``
   Expects the per-model translation steps used for a B-scan.

``test_logs_the_written_filename``
   Expects a ``basic``-level record naming the file, which is how the user
   learns where the output went.

   ``logger.basic`` is a custom level 25 added by
   ``gprMax/utilities/logging.py``, between INFO and WARNING.

TestWriteGridMetadata
^^^^^^^^^^^^^^^^^^^^^

``test_records_the_cell_counts``
   Expects ``nx_ny_nz`` from the grid.

``test_records_the_discretisation``
   Expects ``dx_dy_dz`` per axis, from the anisotropic fixture.

``test_records_the_time_step``
   Expects the CFL time step, needed to convert sample index to time.

``test_counts_all_four_source_types``
   Expects ``nsrc`` to include transmission lines alongside the three source
   lists — one voltage source, one dipole, one line.

``test_counts_receivers``
   Expects ``nrx`` to be the receiver count.

TestWriteSources
^^^^^^^^^^^^^^^^

``test_sources_are_numbered_from_one``
   Expects ``srcs/src1`` and ``srcs/src2`` — user-facing numbering, not
   zero-based.

``test_the_source_type_is_the_class_name``
   Expects ``type(src).__name__``, so a reader can tell a voltage source
   from a dipole.

``test_positions_are_in_metres``
   Expects cell indices multiplied by the per-axis discretisation, so the
   file carries physical coordinates.

``test_transmission_lines_are_not_in_the_source_group``
   Expects lines to be excluded from ``srcs`` and given their own group —
   they carry extra data no other source has.

TestWriteTransmissionLines
^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_records_the_line_resistance``
   Expects the characteristic impedance, needed to interpret the voltages.

``test_records_the_line_discretisation``
   Expects the 1D line's own cell size, which is not the grid's.

``test_records_the_position_in_metres``
   Expects the same index-times-spacing convention as sources.

``test_writes_all_four_traces``
   Expects incident and total voltage and current — four datasets, the pairs
   a user subtracts to get the reflected wave.

``test_trace_values_are_preserved``
   Expects the arrays written verbatim.

TestWriteReceivers
^^^^^^^^^^^^^^^^^^

``test_records_the_receiver_name``
   Expects the user's ``#rx`` label, so traces can be identified.

``test_records_the_position_in_metres``
   Expects index-times-spacing, as for sources.

``test_one_dataset_per_requested_output``
   Expects ``rxs/rx1/<output>`` for each key of ``rx.outputs``.

``test_trace_values_are_preserved``
   Expects the in-memory series written verbatim.

``test_receivers_are_sorted_by_id``
   Expects ``rx1`` to be the alphabetically first ID, not the first one
   added.

   The sort exists so that a multi-rank MPI run, where receivers arrive in
   arbitrary order, always writes them in the same sequence.

``test_the_sort_mutates_the_grids_receiver_list``
   Expects ``grid.rxs`` itself to be reordered — the writer sorts in place,
   so a caller holding the list sees it change as a side effect of writing
   output.

TestWriteSubgrids
^^^^^^^^^^^^^^^^^

``test_creates_a_group_per_subgrid``
   Expects ``/subgrids/<name>`` named for the subgrid.

``test_subgrid_receivers_are_written``
   Expects the subgrid's own traces alongside the main grid's.

``test_records_the_refinement_ratio``
   Expects ``ratio``, without which the subgrid's spacing and time step
   cannot be interpreted.

``test_records_the_huygens_surface_separation``
   Expects ``is_os_sep`` — the gap between the inner and outer Huygens
   surfaces, in main-grid cells.

``test_records_the_subgrid_pml_thickness_from_the_x0_slab``
   Expects a single value taken from ``pmls["thickness"]["x0"]``: a
   subgrid's six PML slabs are all built from one setting, so one is
   representative.

``test_records_its_own_iteration_count``
   Expects the subgrid's ``iterations``, which is ``ratio`` times the main
   grid's — the subgrid steps faster.

``test_records_the_interpolation_and_filter_settings``
   Expects both precursor settings, since they change the numerical result
   at the seam.

``test_a_model_with_no_subgrids_writes_no_group``
   Expects no ``/subgrids`` group at all for a plain model.

When these fail
~~~~~~~~~~~~~~~

**``AttributeError: 'Rx' object has no attribute 'ID'``.** The receiver was
created without an ID. ``Rx.__init__`` only *annotates* ``self.ID: str`` and
never assigns it, so ``write_hd5_data``'s sort raises. The ``make_rx``
fixture always names receivers; see `notes/bugs/rx-id-annotation-never-
assigned.md`.

**A current value is wrong on a boundary.** Each of ``Ix``/``Iy``/``Iz``
returns exactly ``0`` on the two faces it cannot close a contour around —
and is *unguarded* on its own axis. ``Ix`` at ``x == 0`` is a real number.

**``TestAgreementWithTheGridImplementation`` fails.** ``Ix``/``Iy``/``Iz``
here are a second copy of ``FDTDGrid.calculate_Ix``/``Iy``/``Iz``. Nothing
in the source keeps them in step, so a fix applied to one file must be
applied to the other. These nine tests are the only thing that would notice.

**A receiver group is numbered differently than expected.**
``write_hd5_data`` sorts ``grid.rxs`` **in place** by ID before writing, so
``rx1`` is the alphabetically first receiver, not the first added — and the
caller's list is permanently reordered as a side effect of writing.

**An unexpected ``/subgrids`` group appears.** If *any* subgrid has
receivers, groups are written for *every* subgrid. Whether that is intended
is an open question — see `notes/bugs/subgrid-groups-written-for-every-
subgrid.md`.

Test Catalog — ``test_geometry_views.py``
-----------------------------------------

**55 tests** from 55 test functions across 9 classes.

``GeometryView`` and ``Metadata`` — the self-describing part of an export.

A geometry view exports what a model is *made of*. The pixels are the easy
part; the reason the files are usable in ParaView without a separate legend
is ``Metadata``, which attaches version, discretisation, domain size,
material names, PML depths, and every source and receiver position as VTKHDF
field data.

Three things here repay attention.

**``pml_gv_comment`` reports the PML depth visible *in this view*, not the
grid's PML thickness.** A view of the model's interior sees none of it and
gets zeros; a view overlapping a slab reports how far in it reaches. So the
answer depends on the view's bounds as well as the grid's settings, and the
six faces are computed by six separate comparisons.

**Empty means absent, not zero.** A model with no sources writes no
``source_ids`` field rather than an empty one, because
``srcs_rx_gv_comment`` returns ``None`` and ``write_to_vtkhdf`` skips on
``None``. Same for receivers and for PMLs that are switched off.

**``materials_comment`` prefers the view's material list if there is one.**
``GeometryViewLines`` and ``GeometryObject`` call ``initialise_materials``
first, so their metadata carries the view's remapped list;
``GeometryViewVoxels`` does not, so its metadata falls back to the grid's
whole unfiltered list. The two exporters therefore describe their materials
differently — see ``test_geometry_view_voxels.py``.

TestGeometryViewConstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_is_abstract``
   Expects ``prep_vtk`` and ``write_vtk`` to be abstract, so the base class
   cannot be used directly as an exporter.

``test_builds_a_grid_view_from_the_extents``
   Expects the nine coordinate arguments handed to a ``GridView``.

``test_stores_the_filename_base``
   Expects the user's name kept separately from the resolved path —
   ``set_filename`` combines it with the model number later.

``test_starts_with_no_prepared_data``
   Expects ``nbytes``, ``material_data`` and ``materials`` all unset until
   ``prep_vtk`` runs.

``test_grid_is_reached_through_the_view``
   Expects ``view.grid`` to forward to ``grid_view.grid``.

``test_the_file_extension_is_vtkhdf``
   Expects ``.vtkhdf`` for every geometry view — unlike snapshots, there is
   no HDF5 alternative.

TestSetFilename
^^^^^^^^^^^^^^^

``test_uses_the_output_directory``
   Expects the file to land beside the model's output file, not in the
   working directory.

``test_uses_the_user_supplied_base_name``
   Expects the stem to come from ``filenamebase`` rather than from the model
   name.

``test_appends_the_model_number``
   Expects the per-model suffix, so a B-scan's many models do not overwrite
   each other's geometry files.

``test_applies_the_vtkhdf_extension``
   Expects ``.vtkhdf`` whatever the base name.

``test_replaces_an_existing_suffix``
   Expects ``with_suffix`` semantics — a base name containing a dot is
   truncated at it, as for snapshots.

TestMetadataBasics
^^^^^^^^^^^^^^^^^^

``test_records_the_gprmax_version``
   Expects the writing version, so a file can be traced to its build.

``test_records_the_grid_discretisation``
   Expects ``grid.dl`` — the *grid's* spacing, not the view's stride.

``test_records_the_whole_domain_size``
   Expects ``grid.size``, so the metadata locates a partial view within the
   full model.

``test_materials_only_skips_the_extra_sections``
   Expects PML, source and receiver information not to be computed at all
   when ``materials_only`` is set — ``GeometryViewLines`` uses this.

``test_the_full_form_computes_them``
   Expects the three extra attributes present by default.

``test_grid_is_reached_through_the_view``
   Expects the ``grid`` property to forward to ``grid_view.grid``.

TestMaterialsComment
^^^^^^^^^^^^^^^^^^^^

``test_falls_back_to_the_grids_material_list``
   Expects the grid's whole list when the view has not called
   ``initialise_materials`` — the ``GeometryViewVoxels`` situation.

``test_prefers_the_views_material_list``
   Expects the view's filtered list once it exists — the
   ``GeometryViewLines`` and ``GeometryObject`` situation.

``test_reports_material_names``
   Expects the user-facing ``#material`` identifiers, not numeric IDs — the
   whole point of the table.

``test_smoothed_materials_are_hidden_by_default``
   Expects the automatically generated dielectric-smoothing materials to be
   omitted: they are an implementation detail of averaging, not something
   the user defined.

``test_averaged_materials_includes_them``
   Expects the full list when the caller asks for averaged materials.

``test_a_none_material_list_is_passed_through``
   Expects ``None`` rather than a crash — a non-coordinating MPI rank ends
   up with no material list at all.

TestPmlComment
^^^^^^^^^^^^^^

``test_returns_none_when_no_slabs_were_built``
   Expects ``None`` for a model with PMLs switched off, so the field is
   omitted from the file entirely.

``test_reports_six_depths``
   Expects one entry per face, in the ``pmls["thickness"]`` key order.

``test_a_full_domain_view_sees_the_whole_pml``
   Expects every face to report the grid's own thickness when the view
   covers the whole domain.

``test_an_interior_view_sees_none_of_it``
   Expects all zeros for a view entirely inside the absorbing shell — there
   is no PML to draw.

``test_a_partial_overlap_reports_the_visible_depth``
   Expects ``thickness - xs``: a view starting one cell into a 4-cell PML
   shows three cells of it.

``test_the_high_faces_are_measured_from_the_far_edge``
   Expects ``xf - (nx - thickness)`` for the max faces, so a view reaching
   one cell past the PML's inner edge reports one.

``test_the_six_faces_are_independent``
   Expects a view clipped on one axis only to report a depth on that axis
   and zero on the others.

``test_the_result_is_int64``
   Expects an integer array, since these are cell counts.

TestSourceAndReceiverComment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``test_returns_none_for_an_empty_list``
   Expects ``None`` rather than empty arrays, so nothing is written.

``test_records_receiver_names``
   Expects the user's ``#rx`` labels.

``test_records_receiver_positions_in_metres``
   Expects ``coord * grid.dl``, so ParaView places the marker at the
   physical location.

``test_positions_are_one_row_per_object``
   Expects an ``(n, 3)`` array.

``test_all_four_source_types_are_combined``
   Expects dipoles, voltage sources and transmission lines in one list — the
   file makes no distinction between them.

``test_sources_and_receivers_are_kept_separate``
   Expects two independent groups, so ParaView can style them differently.

TestWriteToVtkhdf
^^^^^^^^^^^^^^^^^

``test_always_writes_the_four_core_fields``
   Expects version, spacing, size and material list unconditionally.

``test_omits_the_pml_field_when_there_is_none``
   Expects ``pml_thickness`` absent rather than zeroed.

``test_writes_the_pml_field_when_present``
   Expects the six depths written when slabs were built.

``test_omits_source_fields_when_there_are_none``
   Expects both ``source_ids`` and ``sources`` absent — they are written as
   a pair or not at all.

``test_writes_source_fields_as_a_pair``
   Expects names and positions together, so neither is meaningful alone.

``test_writes_receiver_fields_as_a_pair``
   Expects the receiver equivalent.

``test_materials_only_writes_nothing_extra``
   Expects exactly the four core fields even when the grid has PMLs, sources
   and receivers to report.

TestMpiMetadata
^^^^^^^^^^^^^^^

``test_extends_the_serial_metadata``
   Expects only the three rank-dependent methods to be overridden.

``test_domain_size_is_the_global_one``
   Expects ``grid.global_size`` rather than this rank's local size, so every
   rank writes the same value.

``test_pml_depths_are_reduced_across_ranks``
   Expects an ``Allgather`` followed by an elementwise maximum: a rank that
   sees no PML must not veto a rank that does.

   With one rank the maximum is that rank's own value, and an all-zero
   result still collapses to ``None``.

``test_a_visible_pml_survives_the_reduction``
   Expects a non-zero depth to be reported after the reduction.

``test_sources_are_gathered_and_sorted_by_name``
   Expects ``allgather`` of a name-to-position dict, then a sort — so every
   rank writes the same order regardless of who owns what.

``test_positions_are_converted_to_global_coordinates``
   Expects ``local_to_global_coordinate`` applied before scaling, so a
   rank's local index maps to the right place in the whole model.

``test_an_empty_list_still_gives_none``
   Expects ``None`` when no rank contributed anything, matching the serial
   behaviour so the field is omitted.

TestSaveGeometryViews
^^^^^^^^^^^^^^^^^^^^^

``test_prepares_and_writes_each_view``
   Expects ``set_filename``, ``prep_vtk`` then ``write_vtk``, in that order
   — the filename must exist before the writer opens it.

``test_handles_several_views``
   Expects every view in the list to be written.

``test_an_empty_list_is_a_no_op``
   Expects no error for a model with no geometry views.

``test_logs_blank_spacer_lines``
   Expects two ``info`` records framing the progress bars — the only output
   this orchestrator produces.

When these fail
~~~~~~~~~~~~~~~

**A PML depth is reported where you expected zero.** ``pml_gv_comment``
reports the PML depth *visible in this view*, not the grid's thickness. A
view of the interior sees none of it; a view overlapping a slab reports how
far in it reaches. The six faces are six independent comparisons, and the
``z0`` one is written with its operands reversed (``thickness - zs > 0``
rather than ``zs < thickness``) — equivalent, but it reads as if it differs.

**A field is missing from a written file rather than empty.** Empty means
absent: ``srcs_rx_gv_comment`` returns ``None`` for an empty list, and
``write_to_vtkhdf`` skips on ``None``. Sources and receivers are written as
name/position *pairs* or not at all.

**A material list is longer or shorter than expected.**
``materials_comment`` prefers ``grid_view.materials`` if the attribute
exists and falls back to ``grid.materials`` otherwise, and it filters out
``dielectric-smoothed`` entries unless ``averaged_materials=True``. Those
two behaviours interact badly in the voxel exporter — see `notes/bugs/voxel-
view-material-index-misalignment.md`.

**``set_filename`` puts the file somewhere unexpected.** It uses
``get_model_config().output_file_path`` plus ``appendmodelnumber``, and
applies ``with_suffix`` — so a base name containing a dot is truncated at
it.

Test Catalog — ``test_geometry_view_voxels.py``
-----------------------------------------------

**26 tests** from 26 test functions across 5 classes.

``GeometryViewVoxels`` — one coloured brick per cell.

The compact geometry export: a VTK ImageData file holding a single
``Material`` value per cell, plus the metadata block. It is what you want
for a quick look at a whole domain.

**It writes raw ``solid`` values.** Unlike ``GeometryViewLines`` and
``GeometryObject``, this exporter never calls ``initialise_materials``, so
the cell data carries the grid's own material numbering rather than a
compacted per-view one. That is self-consistent — ``Metadata`` then falls
back to the grid's full unfiltered material list, so index *n* in the table
still names the material with ``numID == n``. But it means the two view
types describe their materials differently, and a reader that assumes a
filtered table for one and applies it to the other gets the wrong colours.
The tests below pin both halves of that consistency.

**Subgrids get an absolute origin.** A subgrid's own coordinates start at
zero, so a naive origin would stack every subgrid on top of the main grid at
the model origin. ``prep_vtk`` detects a ``SubGridBaseGrid`` and offsets by
the subgrid's position in the parent, scaled by the refinement ratio.

TestClassSurface
^^^^^^^^^^^^^^^^

``test_extends_geometry_view``
   Expects the shared base, so filename handling is inherited.

``test_does_not_override_the_constructor``
   Expects the base nine-coordinate signature, unlike ``GeometryViewLines``
   which forces a unit step.

``test_implements_both_abstract_methods``
   Expects a concrete, instantiable exporter.

TestPrepVtk
^^^^^^^^^^^

``test_material_data_is_the_solid_array``
   Expects the raw per-cell material IDs from ``get_solid()``.

``test_material_data_is_cell_shaped``
   Expects ``(nx, ny, nz)`` — one value per cell, with no extra node.

``test_material_data_is_not_remapped``
   Expects the grid's own material numbering to survive.

   ``initialise_materials`` is never called here, so no compaction happens.
   A view containing only material 2 still reports 2, not 0.

``test_the_metadata_material_table_matches_that_numbering``
   Expects the metadata to list the *grid's* full material set, so index *n*
   of the table still names material ``numID == n``.

   This is the consistency that makes the unremapped cell data usable.

``test_origin_is_the_physical_start_of_the_window``
   Expects ``start * grid.dl`` in metres.

``test_origin_follows_an_anisotropic_grid``
   Expects each axis scaled by its own discretisation.

``test_spacing_is_the_physical_cell_size``
   Expects ``step * grid.dl``, so a strided view reports larger bricks.

``test_byte_count_is_the_material_array_size``
   Expects ``nbytes`` to size the progress bar from the only array this
   exporter writes.

``test_metadata_is_the_full_form``
   Expects PML, source and receiver information to be collected —
   ``materials_only`` is left at its default, unlike the lines exporter.

TestSubgridOrigin
^^^^^^^^^^^^^^^^^

``test_origin_uses_the_parent_position``
   Expects ``i0 * dx * ratio`` — the subgrid's location in the parent,
   expressed in the subgrid's own (finer) spacing.

   Without this, every subgrid would be drawn at the model origin, stacked
   on top of the main grid.

``test_the_three_axes_use_their_own_indices``
   Expects ``i0``, ``j0`` and ``k0`` to drive x, y and z respectively — the
   distinct fixture values make a mix-up visible.

``test_spacing_is_unaffected``
   Expects the subgrid branch to change only the origin — the cell size is
   still ``step * dl``.

TestWriteVtk
^^^^^^^^^^^^

``test_writes_a_readable_file``
   Expects a complete VTKHDF file on disk.

``test_cell_data_is_named_material``
   Expects the single cell array under ``VTKHDF/CellData/Material`` — the
   name ParaView colours by.

``test_cell_data_is_written_in_zyx_order``
   Expects the transpose the VTKHDF specification requires, as for
   snapshots.

``test_whole_extent_spans_the_view``
   Expects ``[0, nx, 0, ny, 0, nz]``.

``test_origin_and_spacing_reach_the_file``
   Expects both geometry attributes written, so the brick lattice lands in
   the right physical place.

``test_metadata_is_attached``
   Expects the four core field-data entries alongside the cell data, so the
   file is self-describing.

``test_material_values_survive_the_round_trip``
   Expects the exact material IDs written, so a reader can map them back
   through the metadata table.

``test_a_strided_view_writes_the_reduced_shape``
   Expects one brick per sampled cell, not per grid cell.

TestMpiVariant
^^^^^^^^^^^^^^

``test_extends_the_serial_exporter``
   Expects only the grid-view type to be overridden — the MPI variant adds
   halo awareness and nothing else.

``test_uses_an_mpi_grid_view``
   Expects ``MPIGridView``, so the exporter's coordinates are trimmed to
   this rank's share of the domain.

``test_origin_uses_global_coordinates``
   Expects ``global_start * dl`` rather than the local start, so each rank's
   block lands in the right place in the shared file.

When these fail
~~~~~~~~~~~~~~~

**Cell values do not match the material table.** This exporter writes raw
``solid`` values and never calls ``initialise_materials``, so the cell data
uses the grid's own numbering. The accompanying table must therefore be the
grid's *complete* list — but ``Metadata`` filters out smoothed materials by
default, which misaligns the two. This is a live defect, not a test
artefact: `notes/bugs/voxel-view-material-index-misalignment.md`.

**Values look transposed.** They are — VTKHDF is ZYX-major, as for
snapshots. Compare against ``material_data.T``.

**A subgrid is drawn at the model origin.** ``prep_vtk`` detects a
``SubGridBaseGrid`` and offsets the origin by ``i0 * dx * ratio``. Without
that branch every subgrid stacks on top of the main grid. The test uses a
concrete ``SubGridBaseGrid`` subclass rather than ``ABCMeta.register``,
which would raise ``RuntimeError: Refusing to create an inheritance cycle``
because ``SubGridBaseGrid`` already inherits ``FDTDGrid``.

Test Catalog — ``test_geometry_view_lines.py``
----------------------------------------------

**40 tests** from 40 test functions across 5 classes.

``GeometryViewLines`` — every cell edge drawn separately.

The bulky geometry export: a VTK UnstructuredGrid holding three line
segments per cell, one along each axis, each carrying the material of the
corresponding ``ID`` component. It is what you need when debugging why an
antenna is not resonating where it should, because it shows the staircase
discretisation of a curved object rather than smoothing it into bricks.

**The step is forced to one.** ``__init__`` overrides the base signature and
hard-codes ``dx = dy = dz = 1``. Drawing individual cell edges at a stride
would be meaningless — you would be drawing edges that do not exist.

**The point-ID walk is the fiddly part.** ``get_line_properties`` numbers
points over a ``(nx+1, ny+1, nz+1)`` lattice while iterating cells over
``(nx, ny, nz)``, so it has to skip the far-edge points that are the source
of no line. The strides are ``z_step = 1``, ``y_step = nz + 1`` and ``x_step
= (nz + 1)(ny + 1)``; after each k-loop the walk skips one point, and after
each j-loop it skips a further ``nz + 1``. Small grids make this hand-
checkable, and the tests below do exactly that.

**Materials are remapped here.** Unlike the voxel exporter, this one calls
``initialise_materials(filter_materials=False)`` and then maps the raw
material data through the resulting index, so cell values are positions in
the metadata table rather than grid material IDs.

TestClassSurface
^^^^^^^^^^^^^^^^

``test_extends_geometry_view``
   Expects the shared base.

``test_the_constructor_takes_no_step``
   Expects six coordinates plus a filename, not nine — the step is not the
   caller's to choose.

``test_the_step_is_forced_to_one``
   Expects a unit stride however the view is built: individual cell edges
   only exist at the grid's own resolution.

TestLinePropertiesKernel
^^^^^^^^^^^^^^^^^^^^^^^^

``get_line_properties``, hand-checked on the smallest possible grids.

``test_a_single_cell_gives_three_lines``
   Expects one line per axis for one cell.

``test_a_single_cell_connectivity_is_hand_computable``
   Expects ``[0,4, 0,2, 0,1]``.

   For ``nx=ny=nz=1`` the strides are ``x_step = (1+1)(1+1) = 4``, ``y_step
   = 1+1 = 2`` and ``z_step = 1``. All three edges start at point 0 and end
   one stride away along their own axis.

``test_two_cells_along_x_skip_the_far_edge_points``
   Expects ``[0,4,0,2,0,1, 4,8,4,6,4,5]``.

   After the first cell the walk advances one point, then skips one for the
   ``(i, j, nz)`` plane and ``nz + 1`` more for the ``(i, ny, ·)`` row —
   landing on point 4, which is exactly ``x_step``.

``test_line_count_is_three_per_cell``
   Expects ``3 * nx * ny * nz`` lines for any grid.

``test_each_line_takes_its_own_id_component``
   Expects the x, y and z edges of a cell to read ``ID[0]``, ``ID[1]`` and
   ``ID[2]`` respectively — the three components in order.

``test_higher_id_components_are_ignored``
   Expects only the first three of the six ``ID`` components to be read —
   components 3-5 are the magnetic ones, and a cell edge is an electric
   quantity.

``test_cells_are_visited_in_x_then_y_then_z_order``
   Expects the innermost loop to be z, so consecutive line triples walk
   along z first.

``test_connectivity_is_int32``
   Expects ``int32`` point indices, as the VTKHDF connectivity array
   requires.

``test_material_data_is_uint32``
   Expects ``uint32`` material IDs, matching the ``ID`` array.

``test_every_point_index_is_within_the_lattice``
   Expects no index past ``(nx+1)(ny+1)(nz+1) - 1`` — an overrun would
   reference a point that was never written.

TestPrepVtk
^^^^^^^^^^^

``test_builds_one_point_per_lattice_node``
   Expects ``(nx+1)(ny+1)(nz+1)`` points — lines join nodes, and there is
   one more node than cell along each axis.

``test_points_are_three_dimensional``
   Expects an ``(n, 3)`` coordinate array.

``test_points_are_in_metres``
   Expects lattice indices scaled by ``grid.dl``, so the drawing lands at
   the model's physical scale.

``test_points_are_offset_by_the_view_start``
   Expects a view of the model's interior to be drawn there rather than at
   the origin.

``test_points_follow_an_anisotropic_grid``
   Expects each axis scaled independently.

``test_three_lines_per_cell``
   Expects ``3 * nx * ny * nz`` cell types.

``test_every_cell_is_a_line``
   Expects a uniform VTK cell type — this exporter draws nothing but line
   segments.

``test_cell_offsets_step_by_two``
   Expects ``0, 2, 4, …`` — every line has exactly two endpoints, so the
   offsets into the connectivity array advance in pairs.

``test_cell_offsets_have_one_more_entry_than_cells``
   Expects ``n_lines + 1`` offsets, the standard VTK convention where the
   final entry closes the last cell.

``test_connectivity_has_two_entries_per_line``
   Expects ``2 * n_lines``.

``test_material_data_has_one_entry_per_line``
   Expects one material per drawn edge.

``test_materials_are_remapped_to_the_view_index``
   Expects the raw ``ID`` values replaced by positions in the metadata
   table.

   The exporter calls ``initialise_materials(filter_materials=False)``, so
   with the grid's own IDs already dense the map is the identity — but the
   mapping step is what makes the file self-describing when they are not.

``test_metadata_is_materials_only``
   Expects PML, source and receiver information to be skipped — this
   exporter asks for ``materials_only``, unlike the voxel one.

``test_metadata_includes_averaged_materials``
   Expects dielectric-smoothed materials in the table, because the drawn
   edges can reference them.

``test_byte_count_sums_every_written_array``
   Expects points, cell types, connectivity, offsets and materials — all
   five arrays this exporter writes.

TestWriteVtk
^^^^^^^^^^^^

``test_writes_a_readable_unstructured_grid``
   Expects a VTKHDF file declaring the UnstructuredGrid type.

``test_writes_the_point_coordinates``
   Expects a ``Points`` dataset of shape ``(n, 3)``.

``test_writes_the_connectivity``
   Expects the flat endpoint list, two entries per line.

``test_writes_the_cell_offsets``
   Expects the offsets array alongside the connectivity.

``test_writes_the_cell_types``
   Expects a ``Types`` dataset, one entry per line.

``test_declares_the_counts``
   Expects ``NumberOfCells``, ``NumberOfPoints`` and
   ``NumberOfConnectivityIds`` to agree with the arrays.

``test_cell_data_is_named_material``
   Expects the per-line material under ``VTKHDF/CellData/Material``, the
   same name the voxel exporter uses.

``test_material_values_survive_the_round_trip``
   Expects the per-line materials written verbatim — a 1D array needs no
   transpose, unlike the voxel exporter's 3D one.

``test_metadata_is_attached``
   Expects the four core field-data entries and nothing more, since this
   exporter asks for ``materials_only``.

TestMpiVariant
^^^^^^^^^^^^^^

``test_extends_the_serial_exporter``
   Expects the MPI variant to override only the grid-view type and the two
   write-side methods.

``test_uses_an_mpi_grid_view``
   Expects ``MPIGridView``, so points are generated for this rank's share of
   the domain only.

``test_points_use_global_coordinates``
   Expects ``global_start + offset`` rather than the local start, so each
   rank's edges land in the right place in the shared file.

When these fail
~~~~~~~~~~~~~~~

**A connectivity assertion fails.** The point-ID walk numbers points over an
``(nx+1, ny+1, nz+1)`` lattice while iterating cells over ``(nx, ny, nz)``,
so it skips one point after each k-loop and ``nz + 1`` more after each
j-loop. The strides are ``z_step = 1``, ``y_step = nz + 1``, ``x_step = (nz
+ 1)(ny + 1)``. The 1×1×1 and 2×1×1 cases are small enough to work out by
hand; do that rather than adjusting the expectation.

**A material is wrong on one axis.** The three edges of a cell read
``ID[0]``, ``ID[1]`` and ``ID[2]`` in that order. Components 3-5 are
magnetic and are deliberately not read.

**An off-by-one in the offsets array.** VTK wants ``n_lines + 1`` offsets —
the final entry closes the last cell. ``cell_offsets`` steps by two because
every line has exactly two endpoints.

**The step is not what you set.** ``GeometryViewLines.__init__`` overrides
the base signature and hard-codes ``dx = dy = dz = 1``. Drawing individual
cell edges at a stride would mean drawing edges that do not exist.

Test Catalog — ``test_geometry_objects.py``
-------------------------------------------

**36 tests** from 35 test functions across 6 classes.

``GeometryObject`` — exporting a model's raw arrays for later reuse.

Not a picture: a working copy. This writer dumps ``solid``, ``rigidE``,
``rigidH`` and ``ID`` into a plain ``.h5``, alongside a ``_materials.txt``
that names what the numbers mean in gprMax's own input syntax. The point is
that an expensive antenna geometry can be built once and then read straight
back into later models with ``#geometry_objects_read``.

**The materials file is executable input.** Each line is a literal
``#material:`` or ``#add_dispersion_*:`` command. That is what makes the
pair self-contained — the ``.h5`` holds indices, the ``.txt`` turns them
back into physics, and both are consumed by the reader tested in
``test_geometry_objects_read.py``.

**Material IDs are compacted.** ``write_hdf5`` calls
``initialise_materials()`` with filtering on, so the exported arrays are
renumbered from zero over just the materials present. An exported object
therefore carries no reference to materials that were only used elsewhere in
the source model.

**``rigidE`` has 12 components and ``rigidH`` 6.** The byte-size arithmetic
folds them into a single factor of 18, which is worth naming because ``18``
appearing alone in a size calculation is otherwise unexplained.

TestConstruction
^^^^^^^^^^^^^^^^

``test_builds_a_grid_view_from_the_extents``
   Expects the six coordinates handed to a ``GridView``; unlike the view
   exporters there is no stride argument at all.

``test_the_hdf5_filename_takes_the_h5_suffix``
   Expects ``<name>.h5`` for the array file.

``test_the_materials_filename_is_suffixed_and_txt``
   Expects ``<name>_materials.txt`` beside it — the naming the reader relies
   on to find the pair.

``test_files_land_beside_the_input_file``
   Expects the input file's directory, not the output directory — geometry
   objects are inputs to later runs.

``test_grid_is_reached_through_the_view``
   Expects the usual forwarding property.

TestSizeArithmetic
^^^^^^^^^^^^^^^^^^

``test_solid_size_is_one_uint32_per_cell``
   Expects ``nx·ny·nz · 4`` bytes.

``test_rigid_size_covers_both_arrays``
   Expects ``18 · nx·ny·nz · 1`` bytes.

   The 18 is ``rigidE``'s 12 components plus ``rigidH``'s 6 — the two are
   written together and sized together.

``test_id_size_uses_the_node_count``
   Expects ``6 · (nx+1)(ny+1)(nz+1) · 4`` bytes — ``ID`` is node-centred, so
   it has one more entry per axis than ``solid``.

``test_total_is_the_sum_of_the_three``
   Expects ``datawritesize`` to size the progress bar from everything that
   will be written.

``test_sizes_are_floats``
   Expects floats, since ``tqdm`` scales them into human units.

``test_sizes_track_the_view_not_the_grid``
   Expects a partial view to report its own extent, so a small export from a
   large model does not claim the whole model's bytes.

TestWriteMetadata
^^^^^^^^^^^^^^^^^

``test_records_the_gprmax_version``
   Expects the writing version stamped at the root.

``test_records_the_title``
   Expects the model title as given.

``test_records_the_discretisation``
   Expects ``dx_dy_dz`` per axis.

   The reader checks this against the importing model's own spacing and
   refuses to build if they differ — a geometry object is a fixed lattice of
   cells, not a scalable shape.

TestWriteHdf5Arrays
^^^^^^^^^^^^^^^^^^^

``test_writes_all_four_arrays``
   Expects ``/data``, ``/rigidE``, ``/rigidH`` and ``/ID`` — everything
   needed to rebuild the geometry without re-running the build step.

``test_data_is_cell_shaped``
   Expects ``(nx, ny, nz)`` for the solid array.

``test_data_is_int16``
   Expects a *signed* type, because ``-1`` means "background, build nothing
   here" — an unsigned array could not express that.

``test_id_is_node_shaped_with_six_components``
   Expects ``(6, nx+1, ny+1, nz+1)``.

``test_rigid_arrays_keep_their_component_counts``
   Expects 12 components for ``rigidE`` and 6 for ``rigidH``, matching the
   18 in the byte arithmetic.

``test_arrays_are_not_transposed``
   Expects plain ``(x, y, z)`` ordering — this is a raw HDF5 file, not
   VTKHDF, so none of the ZYX reordering applies.

``test_material_ids_are_compacted``
   Expects renumbering from zero over the materials actually present.

   A view containing only material 2 exports it as 0, so the file's indices
   line up with its own materials list.

``test_progress_is_reported_in_three_steps``
   Expects one update after the solid array, one after both rigid arrays,
   and one after ``ID``.

``test_reported_bytes_total_the_declared_size``
   Expects the progress total to match ``datawritesize``.

TestMaterialsFile
^^^^^^^^^^^^^^^^^

``test_writes_one_line_per_material``
   Expects a line for each material in the compacted list.

``test_lines_are_valid_material_commands``
   Expects gprMax input syntax — ``#material: er se mr sm name`` — because
   the reader feeds these lines straight back through the parser.

``test_the_constitutive_parameters_are_written``
   Expects permittivity, conductivity, permeability and magnetic loss in
   that order, followed by the name.

``test_dispersive_materials_get_a_second_line``
   Expects a dispersion command alongside the material line, so the
   frequency dependence survives the round trip. (2 parameter sets)

``test_lorenz_dispersion_writes_three_values_per_pole``
   Expects ``deltaer``, ``tau`` and ``alpha`` for a Lorentz pole — one more
   than Debye needs.

   Note the command is spelled ``#add_dispersion_lorenz``, matching gprMax's
   own input syntax.

``test_the_material_name_ends_each_dispersion_line``
   Expects the material ID appended, so the command binds to the right
   material when re-parsed.

``test_non_dispersive_materials_get_no_second_line``
   Expects a plain material to produce exactly one line.

``test_materials_are_written_in_compacted_order``
   Expects the file's line order to match the index the arrays use, so line
   *n* describes material *n*.

TestMpiVariant
^^^^^^^^^^^^^^

``test_extends_the_serial_writer``
   Expects only the grid-view type and the write method to be overridden.

``test_uses_an_mpi_grid_view``
   Expects ``MPIGridView``, so each rank exports its own share.

``test_size_arithmetic_is_inherited``
   Expects the byte counts to be computed by the base constructor,
   unchanged.

``test_the_parallel_write_needs_parallel_hdf5``
   Expects ``MPIGeometryObject.write_hdf5`` to open with ``driver="mpio"``.

   That is unavailable here — ``h5py.get_config().mpi`` is ``False`` — so
   the write path itself is out of reach in this environment. Recorded
   explicitly rather than left as an unexplained coverage hole.

When these fail
~~~~~~~~~~~~~~~

**``IndexError`` or ``KeyError`` from ``initialise_materials``.**
``write_hdf5`` filters materials to those present in the view, and both
``ID`` and ``solid`` initialise to **1**. A test grid defining only material
0 must set both arrays to 0, or ``np.unique(ID)`` indexes past the end of
the materials list.

**A byte-size assertion fails.** The three are computed differently on
purpose: ``solidsize`` is one ``uint32`` per *cell*, ``rigidsize`` folds
``rigidE``'s 12 components and ``rigidH``'s 6 into a factor of 18, and
``IDsize`` uses the *node* count ``(size + 1)`` because ``ID`` is node-
centred.

**A materials-file line does not parse.** Each line is literal gprMax input
syntax — ``#material:`` and ``#add_dispersion_*:`` — because the reader
feeds them straight back through the parser. Note the Lorentz command is
spelled ``#add_dispersion_lorenz``, and each dispersion line ends with the
material name so the command binds correctly.

**The MPI writer cannot be exercised.** ``MPIGeometryObject.write_hdf5``
opens with ``driver="mpio"``, unavailable here. The one test that touches it
inspects the source rather than running it, so the coverage hole is
explicit.

Test Catalog — ``test_geometry_objects_read.py``
------------------------------------------------

**38 tests** from 35 test functions across 6 classes.

``ReadGeometryObject`` — importing a previously exported geometry.

The other half of the round trip ``test_geometry_objects.py`` writes. Given
an exported ``.h5``, this reader slices out the part that belongs to the
importing grid and writes it into that grid's ``solid``, ``rigidE``,
``rigidH`` and ``ID`` arrays.

**There are two files with this name.** PR 8 tested
``gprMax/user_objects/cmds_geometry/geometry_objects_read.py`` — the
``#geometry_objects_read`` *command*, which parses the materials text file
and drives the import. This is
``gprMax/geometry_outputs/geometry_objects_read.py``, the *file reader* that
command delegates to. Same idea, different layer, identical filename.

**Materials are re-based on import.** The importing model already has its
own materials, so every ID read from the file is shifted by
``num_existing_materials``. Without that, an imported object would silently
reference whatever material happened to occupy those indices in the host
model.

**Ranks that do not overlap get no view at all.** Under MPI, a rank whose
local domain does not intersect the object's bounding box sets ``grid_view =
None`` and every read method short-circuits. It still calls ``comm.Split``
first, because ``MPIGridView`` would call it on the other ranks and an
unmatched collective deadlocks.

TestConstruction
^^^^^^^^^^^^^^^^

``test_opens_the_file``
   Expects a live h5py handle for the duration of the read.

``test_derives_the_extent_from_the_data_shape``
   Expects ``stop = start + data.shape`` — the caller supplies only the
   insertion point, and the file itself says how big the object is.

``test_builds_a_serial_grid_view_for_a_plain_grid``
   Expects a ``GridView`` rather than the MPI variant.

``test_stores_the_material_offset``
   Expects the existing-material count kept for use by every read.

``test_is_a_context_manager``
   Expects ``__enter__`` to return the reader itself.

``test_exiting_closes_the_file``
   Expects the handle released on exit, so the file is not left locked.

``test_close_can_be_called_directly``
   Expects the same effect without the ``with`` block, for callers that
   manage the lifetime themselves.

TestValidation
^^^^^^^^^^^^^^

``test_matching_discretisation_is_accepted``
   Expects ``True`` when the file's spacing equals the grid's.

``test_mismatched_discretisation_is_rejected``
   Expects ``False`` — a geometry object is a fixed lattice of cells, so
   importing it into a differently-discretised model would silently change
   its physical size.

``test_a_single_mismatched_axis_is_rejected``
   Expects all three axes checked, not just the first.

``test_detects_an_id_array``
   Expects ``True`` when the file carries ``/ID``.

   The caller uses this to choose between a fast path that reads the stored
   arrays directly and a slow one that rebuilds them from ``data`` with the
   voxel builder.

``test_detects_a_missing_id_array``
   Expects ``False`` for an older file with only ``/data``.

``test_detects_rigid_arrays``
   Expects ``True`` only when *both* rigid arrays are present.

``test_detects_missing_rigid_arrays``
   Expects ``False`` when they are absent.

TestReadData
^^^^^^^^^^^^

``test_writes_the_solid_array_into_the_grid``
   Expects the imported values to land in ``grid.solid`` at the requested
   position.

``test_places_the_object_at_the_requested_start``
   Expects the object offset into the grid, leaving everything before it
   untouched.

``test_shifts_material_ids_by_the_existing_count``
   Expects every imported ID increased by ``num_existing_materials``, so it
   names the material the importing model just added rather than one it
   already had.

``test_get_data_returns_without_writing``
   Expects the array back and the grid untouched — the caller uses this when
   it needs to rebuild the rigid and ID arrays itself.

``test_get_data_does_not_apply_the_material_offset``
   Expects the raw file values.

   ``read_data`` adds the offset; ``get_data`` does not, leaving it to the
   caller. Mixing the two up would double-shift every material.

``test_data_is_converted_to_int16``
   Expects a signed 16-bit result even from an unsigned file.

   ``-1`` means "background, build nothing here", and files exported by
   other tools (AustinMan/Woman) store ``uint16``. Reading one of those
   without the conversion would make every background cell material 65535.

``test_an_int16_file_is_left_alone``
   Expects no conversion when the file already uses the right type.

TestReadRigidAndId
^^^^^^^^^^^^^^^^^^

``test_reads_rigid_e``
   Expects all 12 components written into the grid's ``rigidE``.

``test_reads_rigid_h``
   Expects all 6 components written into ``rigidH``.

``test_rigid_arrays_are_not_material_shifted``
   Expects the offset *not* applied — the rigid arrays hold flags, not
   material indices.

``test_reads_id_with_the_inclusive_bound``
   Expects ``(nx+1)`` nodes per axis, because ``ID`` is node-centred — the
   reader asks for a read slice with ``upper_bound_exclusive=False``.

``test_id_is_material_shifted``
   Expects the offset applied here, unlike the rigid arrays — ``ID`` does
   hold material indices.

``test_rigid_and_id_land_at_the_requested_start``
   Expects the same offsetting as ``read_data``, so all four arrays describe
   the same region.

``test_a_full_import_writes_all_four_arrays``
   Expects the complete fast path to reconstruct the geometry without re-
   running the voxel builder.

TestRoundTrip
^^^^^^^^^^^^^

``test_an_exported_object_reads_back_unchanged``
   Expects a full write-then-read cycle to reproduce the source geometry.

   This is the property the whole pair exists for: build an expensive
   geometry once, export it, and get exactly the same cells back in a later
   model.

``test_the_round_trip_respects_a_material_offset``
   Expects every imported material shifted, so the object's materials sit
   after the host model's own.

TestMpiPaths
^^^^^^^^^^^^

``test_an_overlapping_rank_gets_an_mpi_grid_view``
   Expects ``MPIGridView`` when this rank's domain intersects the object's
   bounding box.

``test_a_non_overlapping_rank_gets_no_view``
   Expects ``grid_view = None`` when this rank owns none of the object.

   The rank still calls ``comm.Split(MPI.UNDEFINED)`` first: the ranks that
   *do* overlap will split their communicator inside ``MPIGridView``, and an
   unmatched collective would hang every one of them.

``test_validation_passes_trivially_without_a_view``
   Expects ``True`` — a rank with nothing to read cannot disagree about the
   discretisation, and returning ``False`` would abort the run for everyone.

``test_every_read_short_circuits_without_a_view``
   Expects each reader to return immediately rather than raise. (4 parameter
   sets)

``test_get_data_returns_none_without_a_view``
   Expects ``None`` rather than an empty array, so the caller can tell
   "nothing for this rank" from "an empty object".

When these fail
~~~~~~~~~~~~~~~

**You are looking at the wrong file.** There are two
``geometry_objects_read.py``. PR 8 tested
``gprMax/user_objects/cmds_geometry/geometry_objects_read.py`` — the
``#geometry_objects_read`` *command*. This file tests
``gprMax/geometry_outputs/geometry_objects_read.py``, the *file reader* that
command delegates to.

**Imported materials name the wrong thing.** Every ID read from the file is
shifted by ``num_existing_materials`` — but only in ``read_data`` and
``read_ID``. ``get_data`` deliberately does *not* apply the offset, leaving
it to the caller, and the rigid arrays never do because they hold flags
rather than material indices. Mixing these up double-shifts or under-shifts.

**Background cells come back as 65535.** ``-1`` means "build nothing here",
so the data must be ``int16``. Files exported by other tools
(AustinMan/Woman) store ``uint16``; the reader converts, and
``test_data_is_converted_to_int16`` pins it.

**An MPI test does not take the branch you expect.** The dispatch is
``isinstance(grid, MPIGrid)`` against the name imported into the reader
module. ``MPIGrid`` is a concrete class with no ABC registration hook, so
the fixture rebinds that one name with ``monkeypatch``.

**A non-overlapping rank must still call ``comm.Split``.** Ranks that *do*
overlap split their communicator inside ``MPIGridView``; an unmatched
collective hangs all of them. That is why the no-view path calls
``Split(MPI.UNDEFINED)`` before setting ``grid_view = None``.

Deliberately Untested Paths
---------------------------

The standing rule from PR 9 onward is that **no test asserts broken behaviour
and none is marked** ``xfail``. Where a defect made a contract untestable, the
test was omitted. That leaves coverage holes which would otherwise look like
oversights, so they are named here.

Each has a maintainer write-up carrying the tests its fix should add, so
closing one of these is a matter of applying the fix and pasting in the test.

**An unknown PML formulation.**
   ``calculate_update_coeffs`` dispatches on ``G.pmls["formulation"]`` with no
   terminal ``else``, so an unrecognised string leaves all eight coefficient
   arrays zero and raises nothing. The PML then absorbs nothing while the run
   reports success. Nothing tests this path because the only assertion
   available would be "produces silently wrong output".

**An unknown PML direction.**
   ``PML.__init__`` and ``initialise_field_arrays`` both branch on
   ``direction[0]`` with no ``else``, so an invalid direction produces a slab
   missing ``d``, ``thickness`` and all four auxiliary arrays. The failure
   surfaces later as an ``AttributeError`` from inside the coefficient builder.

**A non-polynomial CFS scaling.**
   ``CFS.calculate_values`` checks ``scalingprofile == "constant"`` and then
   ``scaling == "polynomial"`` — two *different* attributes — with no ``else``.
   Anything else yields a silently all-zero profile.

**A snapshot with an unrecognised file extension.**
   ``write_file``'s two-way dispatch has no ``else``, so the snapshot is not
   written and nothing is raised. A run producing no output files exits
   successfully.

**``htod_snapshot_array`` on a non-accelerator solver.**
   The three-way solver branch has no ``else``, so with ``solver == "cpu"``
   none of the six device names is bound and the ``return`` raises
   ``UnboundLocalError``. The Metal, CUDA and OpenCL branches *are* tested,
   with a recording device stand-in and injected stand-in modules.

**Unnamed receivers.**
   ``Rx.__init__`` writes ``self.ID: str`` — an annotation, not an assignment —
   so the attribute never exists. ``write_hd5_data`` sorts by it and raises
   ``AttributeError`` at the very end of a run. Every receiver in this suite is
   given an explicit ID.

**Voxel geometry views of models using dielectric smoothing.**
   ``GeometryViewVoxels`` writes raw ``solid`` values but its ``Metadata``
   filters smoothed materials out of the accompanying table, so the cell values
   and the table index differently. Reproduced directly; the table can end up
   *shorter* than the indices the data uses.

**Subgrids without their own outputs.**
   ``write_hdf5_outputfile`` writes a group for every subgrid whenever any one
   of them has receivers. Whether that is intended or an oversight needs a
   maintainer's decision, so no test asserts either reading.

**The error path of two HDF5 writers.**
   ``write_hdf5_outputfile`` never closes its file and ``Snapshot.write_hdf5``
   closes it only on success. Refcounting hides this today.

**Cross-model growth of the snapshot maxima.**
   ``Snapshot.nx_max`` and friends only ever increase and nothing in production
   resets them, so a B-scan sizes every model's device buffers from the largest
   snapshot anywhere in the run. The suite's autouse reset fixture is the
   workaround; production has none.

**The parallel-HDF5 write paths.**
   ``MPISnapshot.write_hdf5`` opens with ``driver="mpio"``, and
   ``MPISnapshot.write_vtk`` reaches the same call through
   ``VtkImageData(..., comm=...)``; ``MPIGeometryObject.write_hdf5`` does the
   same. ``h5py.get_config().mpi`` is ``False`` in this environment and on CI,
   so the open raises before any gprMax logic runs. Three tests ship behind
   ``skipif`` and will execute wherever parallel HDF5 exists. **This is an
   environment constraint, not a decision** — everything upstream of the write
   is covered unconditionally.

**Genuinely multi-rank behaviour.**
   Every MPI test runs at one rank against a real ``MPI.COMM_SELF``. The
   halo-clamping and offset arithmetic is fully exercised, because it depends
   on ``negative_halo_offset`` and ``grid.size`` rather than on rank count. What
   one rank cannot show is cross-rank agreement — a real neighbour on the other
   side of a collective. Those tests assert the local arithmetic and the
   collective *call contract* instead.

Four of the first five above are the same shape: an ``if``/``elif`` chain
enumerating the expected cases with **no terminal** ``else``. With PR 9's five
that is nine instances across two PRs, and it is worth raising as one issue
about the codebase's dispatch idiom rather than as nine tickets.
