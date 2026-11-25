*************
Release notes
*************

Unreleased
==========

These changes reduce per-time-step host overhead in the Apple Metal backend
and correct the modes of eigenmode ports cut by PMC symmetry planes.

Execution
---------

* Metal far-field (KSIR and equivalent-current) frequency-domain collection no
  longer spends most of its time in Python: whether each kernel argument is a
  buffer view is now cached per type, instead of being tested with ``hasattr``
  on PyObjC buffer proxies for every argument, surface component and time step.
  The same applies to the Metal SAR collector.
* Metal far-field frequency-domain accumulation is submitted without waiting
  for each dispatch to complete; the command queue is synchronised before any
  download, as before.
* Metal eigenmode-port monitors synchronise only when their DFT phases are
  re-anchored (every 1024 steps) rather than on every time step.
* On an Apple M2 Pro, the User Guide waveguide-fed horn (eigenmode port,
  virtual waveguide, equivalent-current far field) now solves in 5.3 s
  instead of 45.9 s in the reported benchmark. Before/after Metal outputs
  were unchanged; comparison with CPU double precision gave a maximum
  relative difference of 2.36e-5 and a directivity difference of 0.0014 dB.
  Speedups depend on the model and hardware.

Eigenmode ports
---------------

* Eigenmode ports whose cross-section touches a PMC symmetry boundary now
  solve the correct modes. The port window was closed with artificial PEC on
  every edge, which is right on a PEC symmetry plane but turned a PMC plane
  into an electric wall: for a rectangular guide split along its broad wall,
  TE10 became evanescent across its band. Window edges on a PMC face now leave
  tangential E free and use the image-theory ghost value of tangential H,
  matching the FDTD PMC symmetry update. A quarter model of the User Guide
  horn, with PEC and PMC symmetry planes and the feed guide continued into the
  PML, reproduces the full model's S11 to within 0.0004 dB and 0.002 degrees.
* Surface-impedance walls that cross a PMC symmetry plane keep their boundary
  equations on the cut. A rectangular guide with 20 ohm walls, halved by a PMC
  plane, now gives the modal attenuation of the full guide to round-off.
* Virtual waveguides now reject a port whose aperture is cut by a PMC symmetry
  boundary, rather than closing it with PEC. A window edge on a PMC face that
  a physical PEC wall covers completely is still accepted, as in 4.0.2.
* Automatic mode verification treats a window edge on a PMC symmetry plane as
  a physical wall, so a closed guide reduced by symmetry is classified like the
  full guide instead of reporting insufficient exterior geometry.

The `changes since 4.0.2
<https://github.com/gprMax/gprMax/compare/v.4.0.2...master>`_ lists revisions
merged since the last release.

4.0.2 (Caol Ila)
================

This maintenance-led release collects corrections since 4.0.1, improvements
to eigenmode tracking and geometry conversion, and smaller modelling and
post-processing additions. Automatic eigenmode tracking remains experimental
and opt-in; ``tracking="legacy"`` is still the default.

For installation and upgrades, follow :ref:`installation`. Supported Python
versions remain 3.11--3.13, with CPU wheels for Linux x86-64, Windows x86-64,
and Intel and Apple Silicon macOS. CUDA, OpenCL, Metal and MPI still require
their optional bindings and system runtimes; see :doc:`accelerators` and
:doc:`hpc`. No new mandatory CAD dependency is added to the core installation.

Before reusing reference results, read the changes below to DPW termination,
PMC/PML intersections, eigenmode acceptance, seeded fractal/grass geometry,
CAD voxelisation and antenna placement. Study receiver selection also adds
output metadata that third-party readers may need to handle. This release
does not change the historical main-grid CFL timestep calculation.

NTFF temporal windows
---------------------

* KSIR, equivalent-current NTFF, and layered NTFF frequency transforms accept
  ``hann_end`` in the Python API and hash commands. It leaves the first 80%
  of the sample interval unchanged, then applies a falling half-cosine taper.
  The default remains ``rectangular`` and the full symmetric ``hann`` is
  unchanged. Full Hann is not recommended for transient amplitudes/RCS;
  end-only tapering also needs care and does not repair insufficient decay.
  See :ref:`ntff-frequency-windows`. Port-normalised gain/efficiency still
  require ``rectangular``.

SAR and radiometry temporal windows
-----------------------------------

* The Python API accepts the same optional ``hann_end`` temporal taper for
  SAR and radiometry as for NTFF, with ``WindowTaperFraction=0.2`` recorded in
  HDF5. Rectangular remains the default; existing full Hann weights are
  unchanged. Both hash commands continue to use rectangular windows.
* Document that full Hann can bias source-normalised transient absorption
  and spatial weights, and that an end taper must not suppress genuine
  late-time fields. See :ref:`sar-temporal-windows`. SFCW/FMCW frequency-band
  windows are different operations and are unchanged.
* Enforce the existing rectangular-only restriction for incident/accepted
  port-power normalisation before time stepping instead of failing during
  finalisation after the solve.

Input validation and rounding
-----------------------------

* ``inf`` in the centre of ``#sphere`` and ``#ellipsoid`` now follows the documented 2D
  single-point convention (interior reference layer on the invariant axis, domain faces on
  in-plane axes) instead of crashing with a raw ``decimal.InvalidOperation``; in 3D models
  it is rejected with the standard "'inf' is only allowed in a coordinate when the model is
  in 2D mode" error. NaN centres are rejected with a clear message.
* The spatial discretisation of ``#snapshot`` and ``#geometry_view``, and the step size of
  ``#rx_array``, must be finite; ``inf``/NaN previously crashed in the internal rounding
  helper and now raise a clear "step size must be a finite value" error. In 2D models,
  write any finite value on the invariant axis for snapshots (one plane is exported
  regardless) and zero for ``#rx_array`` axes that should not step.
* The internal rounding helper ``round_int`` now raises ``ValueError`` for non-finite
  input instead of exposing a raw ``decimal.InvalidOperation`` or the cryptic
  "cannot convert NaN to integer" error. Its finite rounding behaviour is unchanged.
* Decimal-boundary flooring in ``round_float`` no longer drops a value one
  unit below an exact decimal boundary. The main-grid CFL calculation retains
  its historical rounding explicitly, rather than inheriting this helper
  change. Invalid decimal-place and allocation-rounding arguments now fail
  with informative errors.
* Domain dimensions, time windows, waveform parameters, material-bin counts,
  grass-blade counts, subgrid refinement ratios and single-use hash-command
  arguments receive earlier validation. Invalid or missing values produce
  clearer errors instead of failing later in numerical routines. Input files
  and included text are read as UTF-8, with clearer decoding errors.
* Material and mixing-model builders, and selected geometry commands, attach
  the validation reason to ``ValueError`` instead of logging it and raising
  an empty exception. Logging outside exception handlers no longer emits a
  spurious ``NoneType: None`` traceback. These diagnostic changes preserve
  the validation conditions, including the Peplinski density checks below.
* ``#rx_array`` rejects negative steps before rounding and rejects sub-cell
  steps on axes with a nonzero extent. Zero remains valid on a fixed axis;
  the coordinate tolerance scales with cell size rather than assuming a
  millimetre-scale model.
* Peplinski soils require positive bulk density below particle density and
  sand/clay fractions whose sum does not exceed one. Invalid combinations
  now fail before material generation. Permittivity evaluation rejects
  non-finite or non-positive evaluation frequencies; this is not a blanket
  ban on signed Fourier frequencies used internally by broadband sources.
* Transmission lines reject time windows too short for their auxiliary
  line, and a DPW angle request that cannot meet its specified tolerance
  reports the failure explicitly. Material databases reject smoothing for
  conductor entries. Nested directional material assignments resolve the
  constituent for each axis correctly.

Toolbox antenna models
----------------------

* The GSSI 2 GHz toolbox model now rounds its centre-bottom placement once,
  preserving its integer-cell geometry and one-cell feed gaps. Previously
  off-grid positions could distort the antenna or collapse its receiver edge.
  Off-grid placements warn with the actual position; existing grid-aligned
  placements at the default orientation retain their geometry and loading.
* An optional ``rotation`` keyword accepts 0, 90, 180 or 270 degrees and
  rotates this antenna in the x-y plane, including both feeds and its receiver.
  At 90/270 degrees the field output
  is ``Ex`` rather than ``Ey``. Port voltages retain the global positive-axis
  convention; see :doc:`inc_GPRAntennaModels` for polarity and survey guidance.
  Other antenna models and the solver's coordinate rounding are unchanged.
* The GSSI 400 MHz subgrid example uses the contributed antenna's supported
  2 mm resolution. The antenna design itself has not been rescaled to 1 mm.

Fractal volumes and rough surfaces
----------------------------------

* ``#add_surface_roughness`` and ``#add_grass`` now reject, with a clear
  error message, a surface footprint that discretises to a single cell in
  both in-plane directions (including footprints that only round down to
  it, e.g. 1.4 x 1.4 cells at the model discretisation). Such a footprint
  has a one-sample spectrum, so after the DC component is removed the
  height field is identically zero and the generator previously crashed
  with a raw ``FloatingPointError: divide by zero``. Footprints of at
  least 2 cells in one in-plane direction - including 1-cell-wide strips
  (1 x N or N x 1 cells) - are unaffected by this guard.
  The same guard protects the serial, 2D TE-shadow and MPI-distributed
  fractal-surface generators.
* **Same-seed fractal geometry can change:** fractal volumes and rough
  surfaces now centre their spectral filters on the actual FFT DC position.
  Regenerating them with the same seed can therefore give different geometry
  when any fractal-generation dimension is odd-sized or a single cell. These
  are the dimensions of the fractal being generated, not necessarily those of
  the overall model domain. Fractals whose generation dimensions are all even
  are unaffected, and previously exported geometry is unchanged; reuse it
  where earlier geometry must be reproduced exactly. The separate grass
  random-stream correction below can change grass even on even-sized grids.
* **Seeded grass geometry changes:** blade curvature and root-walk generators
  now use six independently derived random streams instead of restarting
  each stream with the same seed. This removes artificial correlations that
  constrained the bending and root walks. A specified seed still reproduces
  the geometry with unchanged inputs in this release, but does not reproduce
  the older correlated geometry. Keep exported geometry if the earlier
  realisation is required.
* Grass construction supports MPI domain decomposition, with consistent
  geometry across ranks, including runs without an explicit seed. Whole-number
  fractal seeds are validated before generation; the two-cell TE construction
  continues to preserve invariance.

Discrete plane waves
--------------------

* **New default termination of the auxiliary 1D grid:** the absorbing layer
  at the end of each plane wave's 1D grid is now designed for the solver
  precision: ``sum(|m_i|) * N`` cells with ``N = 160`` in double and ``N = 20``
  in single precision, grading order 6 and a design target of -320 dB for the
  reflection of the conductivity profile. The previous peak conductivity was
  too large by a factor of the relative permittivity of the background medium.
  The reflection reached by a discrete broadband pulse depends on the model:
  in the test models, the double-precision error relative to an infinite 1D
  grid falls from between -121 and -191 dB to between -214 and -302 dB.
  Single-precision results change only at round-off level.
  ``termination="legacy"`` restores the previous termination and reproduces
  the previous results exactly, except that the axial-y ``Ex`` correction
  below is kept. See :ref:`dpw-termination`.
* **Cost of the new default:** in double precision the layer is eight times
  longer than before (160 instead of 20 cells per unit of ``sum(|m_i|)``).
  This improves absorption but increases the memory of the 1D grid and can
  increase CPU time substantially for plane waves with large integer direction
  mappings. It is not an execution-speed improvement; ``pml_cells=20`` or
  ``40`` keeps most of the earlier cost at reduced accuracy.
* New optional Python-API keyword arguments ``termination``, ``pml_cells``,
  ``pml_order`` and ``pml_r0_db`` on ``DiscretePlaneWaveAngles``,
  ``DiscretePlaneWaveVector`` and ``DiscretePlaneWaveAxial``, set per plane
  wave. The environment variable ``GPRMAX_DPW_TERMINATION`` applies them to a
  whole run, including hash-command input files.
* Axial plane waves: the hard source of the 1D grid now sits a fixed distance
  before the total-field box, so the incident pulse arrives at the same time
  for every termination with at least 20 cells and for ``"extended"``. A layer
  thinner than 20 cells advances the arrival by ``(20 - N) * sum(|m_i|)`` 1D
  cells and is reported with a warning; use at least 20 cells when the arrival
  time matters.
* **Fixed leakage of axial plane waves along y:** the electric TF/SF
  correction for ``Ex`` on the y faces used the material of ``Ey`` and the
  wrong index on the upper face. Axial y plane waves with an ``Ex`` component
  leaked -12 to -23 dB into the scattered-field region; leakage is now below
  -295 dB in double precision. Fixed on CPU/MPI, CUDA, OpenCL and Metal. The
  correction applies whatever the termination, including ``"legacy"``, so
  earlier results of affected models change. Re-run affected models.
* ``#plane_wave_angles`` and ``#plane_wave_vector`` now reject TFSF boxes
  whose face lies on a domain boundary, with a message naming the face. A
  face on the boundary has no scattered-field region beyond it, and the
  non-axial auxiliary 1D grid has no cells outside the domain projection:
  the correction at such a face wrote onto the wall plane (its magnetic-side
  partner is structurally skipped there) while sampling the 1D grid one cell
  outside the buffer, since the kernels compile with bounds checking off.
  Affected models previously built and produced silently wrong fields - in
  test models, total-field amplitudes off by factors of 2.2 to 33 with the
  incident wave leaking into the scattered-field region. Move the box at
  least one cell inside on every axis, or use ``#plane_wave_axial`` for
  normally incident waves that fill or half-fill the domain, whose box may
  span the domain (its two-sided 1D grid keeps valid samples beyond both
  faces).

Scenes, studies and subgrids
----------------------------

* ``gprMax.run(scenes=[scene], n=N)`` can repeat a single scene definition
  without a bare ``IndexError``. Source and receiver stepping work with or
  without ``geometry_fixed=True``; omitting geometry reuse rebuilds the
  definition each time. Run counts are validated before model construction,
  and warnings identify likely redundant repeats or unused scene definitions.
  Multiple distinct scenes are not automatically cycled to satisfy ``n``.
* Completed models no longer remain alive through subgrid references held by
  their scene definitions. This prevents memory accumulating across multi-scene
  runs while retaining reusable scene definitions and geometry-fixed runs.
* Snapshot times are resolved again when a scene is rebuilt, avoiding a
  previously cached iteration being interpreted as a physical time.
* Studies can select receiver recording per case using ``record=False``
  (Python API or study CSV). A muted receiver retains its identity and output
  group but has ``Recorded=0`` and no field datasets. Older files without this
  attribute remain readable. Merged output uses ``MergedOutputSchemaVersion=2``
  and stores unrecorded samples as **NaN columns, not zeros**, with per-input
  flags in ``trace_metadata/rxs/rxN/Recorded``. See :doc:`studies` and
  :doc:`output` before adapting external readers.

* A subgrid whose Outer Surface (``is_os_sep`` main grid cells outside it) is
  not at least one main grid cell inside the domain is now rejected when the
  model is built, with a message naming the axis. Previously such a model
  failed once time stepping started, with a segmentation fault, an
  ``OverflowError`` or an ``IndexError``. A subgrid whose Outer Surface lay
  exactly on the domain's upper edge used to run; it is now rejected too.
* Unsupported MPI/subgrid combinations are rejected before a distributed
  solve starts, with configuration-specific messages. This does not add
  MPI domain-decomposed subgrid support.

PMC symmetry boundaries and PML
-------------------------------

* Fixed missing PML corrections where an upper PMC symmetry boundary meets
  a perpendicular PML. These intersections could produce incorrect fields
  and late-time instability. Electric and magnetic corrections now include
  the required staggered boundary samples on CPU/MPI, CUDA, OpenCL and Metal;
  ordinary PMLs and internal MPI rank boundaries retain their existing extents.
* ``#pml_cells`` now validates and stores the thickness of the faces that
  carry PML. The invariant axis of a 2D model and faces with a declared
  ``#symmetry_boundary`` carry none, so the documented single-value form
  works in 2D models, requested invariant-axis cells are dropped with a
  message, and a thickness that only overflows through a symmetry face is
  accepted. Previously every 2D model raised "too many cells" for any
  positive single value, and the only escape was the six-value form with
  explicit zeros.
* Explicit ``sigma.max=0`` is honoured in distributed PML construction rather
  than being mistaken for an omitted value. Invalid thickness and kappa
  inputs report the relevant constraint, including ``kappamin >= 1``.

Geometry commands and saved models
----------------------------------

* A volumetric hash command such as ``#box: ... a n c`` now raises a clear
  ambiguity error when ``n`` is a material ID, instead of silently replacing
  the directional assignment with isotropic material ``a`` and tag ``c``.
  The guard also covers ``y``, ``Y``, and ``N`` and the other tagged volume
  primitives. Use ``... a n tag=c`` for an explicitly tagged isotropic volume,
  descriptive directional material names, or the unambiguous Python API.
  Ordinary tagged geometry and existing material IDs remain supported.
* ``#magnetic_edge`` / ``MagneticEdge`` supports active magnetic components
  in 2D: the invariant component in TE and in-plane components in TM. Inactive
  orientations are rejected. The documented ``inf`` convention places or
  spans the appropriate reduced-axis samples.
* ``#helix`` / ``Helix`` adds a 3D helical volume with a circular wire section,
  arbitrary axis, handedness and fractional turns. It is a rasterised volume,
  not an unresolved thin-wire source. Fully PEC helices are checked for
  connectivity of their rasterised Yee edges; other materials have resolution
  checks and warnings. See :doc:`helix` for the geometric definition and
  limitations.
* Impedance geometry can round-trip through ``GeometryObjectsWrite`` and
  ``GeometryObjectsRead`` with its companion JSON material database. The
  continuous surface model and boundary markers are restored, the timestep
  margin is applied, and conflicting destination model IDs are rejected.
  Existing SIBC solver/backend restrictions remain; see
  :doc:`impedance_surfaces`.
* Geometry imported into a two-cell TE model preserves invariant voxel and
  Yee-component layers, including files with component arrays. Thickness
  clipping and volume-loop bounds are corrected for partially out-of-domain
  primitives.
* Geometry-view and geometry-object filenames preserve dots in the requested
  basename, including numbered runs, rather than treating everything after
  the final dot as a replaceable extension. VTKHDF tuple shapes are corrected
  for scalar and vector field-data arrays.

STLtoVoxel and geometry import
------------------------------

* **STLtoVoxel cells now match the STL:** the converter fills the cells whose
  centres are inside each closed surface, on a grid whose lower corner is a
  whole number of cells from the origin, using STEPtoVoxel's voxeliser.
  Previously objects were 8--21 % too large and shifted by up to one cell
  (a 20 mm cube at 2 mm gave 1101 cells instead of 1000). Converted files
  therefore differ in cell count and grid shape; re-convert STL geometry used
  in existing models and place it with the printed ``p1``.
* **``#geometry_objects_read`` clips placements outside the domain:** a file
  placed with ``p1`` outside the model domain, which the toolboxes suggest
  whenever the source model has negative coordinates, used to shift the whole
  object onto the domain face (voxel-only files), crash with an
  ``IndexError``, or silently write its cells into the middle of the grid
  (files with ``/ID``, ``/rigidE`` and ``/rigidH``; a cube spanning
  [-2, 2] mm at 1 mm placed at the printed origin wrote all 64 cells at the
  origin face instead of the 8 inside the domain). The in-domain part is now
  imported with its own values and the rest is dropped; a file that misses
  the domain entirely on any axis is rejected with a clear error. Under MPI
  such placements still abort during the build; that needs a separate fix in
  the geometry view construction.
* STLtoVoxel writes the grid origin as ``origin_xyz`` and prints the
  ``GeometryObjectsRead`` ``p1`` that keeps the STL coordinates. New options
  ``--rotate``, ``--rotate-about``, ``--scale``, ``--translate``,
  ``--supersample`` and ``--pad-cells``; the settings are recorded in the
  ``conversion_settings`` attribute. The ``parallel`` argument of the
  ``convert`` functions is deprecated and ignored.
* STLtoVoxel accepts ``-o`` / ``--output-dir`` / ``--output`` to choose the
  directory for its HDF5 geometry and companion JSON material database,
  creating missing parent directories. The default remains beside the input
  STL. Voxelisation, placement and existing material-database contents are
  unchanged by the directory choice.
* Imported-geometry grids from STLtoVoxel, STEPtoVoxel and GeometryImport no
  longer gain an empty upper cell when a bound lies on a cell face.
* **Faces on cell centres are voxelised consistently** (STEPtoVoxel and
  STLtoVoxel): a face lying on the plane of the cell centres, such as the top
  of 35 um copper on 70 um cells, was cut as its own triangles and could come
  out as wedges (an 8-element patch array lost 23 % of its copper, the
  ``PROBE_FED`` example half its patch at 70 um), and whether it did depended
  on the grid's origin, so ``--pad-cells`` changed the result. A cell centre on
  a face square to an axis is now inside when the face is a lower side of the
  solid and outside when it is an upper side (a thin part left empty is kept by
  thin-feature preservation, as before), whatever the origin. Edges within
  rounding of a cell centre are decided the same way. Geometry converted again
  can therefore differ where a face or an edge lies on, or within rounding of,
  a cell centre (``PROBE_FED`` at 0.1 mm: 394 of 263,000 cells). A sample
  exactly on an oblique face can still be classified differently by different
  sweep axes, as before.
* ``GeometryObjectsRead`` names both cell sizes when the file's resolution
  differs from the model's, and warns when ``p1`` is rounded to a cell corner.

Eigenmode tracking
------------------

* The opt-in automatic tracker supports symmetry-aware subspace continuation
  for PEC, dielectric and SIBC ports. Symmetry is tested about the certified
  active port region, not necessarily the centre of the full numerical
  window. Unequal exterior PEC or unretained SIBC padding can be excluded only
  when proved inactive; active air padding remains part of the problem.
  Additional native Yee-grid symmetry groups and mirror orientations are
  supported. This does not extend SIBC to unsupported execution backends.
* Automatic tracking now resolves longitudinal families and symmetry subgroups
  before selecting polarization within each eligible doublet. Crossing recovery
  excludes spatially unresolved reference candidates, even when their
  eigenproblem residual is small.

* Broadband automatic tracking checks the Maxwell residual of interpolated
  propagating profiles at interval midpoints. ``midpoint_tolerance`` defaults
  to ``1e-3``; exceedances warn without changing the accepted anchors unless
  ``midpoint_refinement=True`` is requested. Refinement uses the shared solve
  budget. Cutoff handling excludes ineligible channels from excitation and
  physical modal references while retaining successfully tracked profiles
  for diagnostics; inspect the stored validity masks near cutoff. See
  :doc:`eigenmode_tracking` for controls and output diagnostics.

* Shared two-direction ``mode_polarizations`` inputs remain supported in both
  legacy and automatic tracking during the deprecation period, including the
  Python and hash-command forms. Both directions and their signs are preserved;
  valid nonorthogonal pairs remain available. Per-mode/per-group mappings
  require legacy tracking. These forms emit a visible deprecation warning and
  will be removed in **gprMax 5.0**. Prefer one global direction when its inferred
  orthogonal partner matches the intended basis. For a +z-normal port, ``"y"``
  implies y/-x and is not phase-equivalent to an explicit y/+x pair. See
  :doc:`eigenmode_tracking` for compatibility and migration examples.

* Residual acceptance limits change: the general ``residual_tolerance`` default
  increases from ``1e-8`` to ``1e-7``. Mixed-mode acceptance, previously limited
  to a fixed ``1e-9``, now uses the configurable tolerance, including legacy
  degenerate groups. At the default this is a 100-fold relaxation of the mixed
  eigen-residual limit, alongside additional reconstructed Maxwell-residual
  checks where available. Higher-order modes and lossy metallic-waveguide modes
  can have larger numerical residuals, so the previous limits could reject
  otherwise admissible modes. The new ``1e-7`` limit remains a stringent
  residual requirement while accommodating these cases; spatial resolution,
  field consistency, degeneracy, propagation, and power checks still apply.
  To retain the previous mixed-mode limit, explicitly
  set ``EigenmodeTrackingConfig(residual_tolerance=1e-9)``; this also tightens
  the other residual checks. The degeneracy and propagation-branch constraints
  still apply. See :doc:`eigenmode_tracking_theory`.

STEPtoVoxel
-----------

* STEP files can be read with OpenCascade installed by pip:
  ``pip install "gprMax[step]"`` installs cadquery-ocp-novtk as an alternative to
  conda's pythonocc-core. Both give the same components, names, colours and voxels.
  See :doc:`inc_STEPtoVoxel`.

* ``--rotate RX RY RZ``, ``--rotate-about`` and ``--rotate-order`` (and the
  matching ``ConversionConfig`` fields) rotate the model before voxelisation.
  ``convert`` also prints the ``GeometryObjectsRead`` position that keeps the
  CAD coordinates.

* Assembly parts that are not split into several solids keep their assembly
  placement in the reported CAD bounding boxes, and in the voxels when the
  model is rotated. With pythonocc-core, CAD colours are now read; every part
  was previously reported grey.

Output processing and plotting
------------------------------

* ``evaluate_saved_surface_dft`` reads KSIR surface DFTs saved by scene
  objects as well as stand-alone monitors, so saved fields can be reevaluated
  at other observation angles without rerunning FDTD. Missing saved DFTs,
  ambiguous names and incompatible conventions receive explicit errors.
  This correction does not add reevaluation of equivalent-current surface
  DFTs.
* Merge and trace-export tools select only the requested numbered file series,
  avoiding similarly prefixed models and previous merged outputs. Receiver
  paths, absent ``nrx``/``Name`` attributes and muted receivers are handled
  consistently in plotting, exports and Marimo. Invalid grid selections and
  missing or unrecorded receivers produce useful diagnostics rather than
  unrelated lookup errors.
* Gathered B-scan plots no longer overwrite the first individual receiver's
  plot. Waveform previews use the solver's
  time-window-to-sample-count convention and accept ``-o`` / ``--output`` to
  save a PNG at a chosen path.
* A-scan and waveform spectra retain all non-negative FFT bins for zero,
  impulse and DC-peaked traces instead of showing an empty or truncated
  frequency axis. Marimo applies the receiver half-step fallback when reading
  magnetic fields from legacy files without explicit timing metadata.
* ImpulseResponse supports the built-in ``gauspulse`` waveform and preserves
  raw CSV waveform identifiers. AntennaPatterns dB rings are anchored at the
  0 dB peak and critical-angle calculations handle their permitted limits.
* Geometry conversion, head extraction, AntennaPatterns, SFCW, FMCW,
  ImpulseResponse and SEG-2/SEG-Y/DT1 exports add input/output alias checks
  and safer staged or recoverable writes. These protect source data and
  existing destinations against the covered validation/write failures;
  multi-file exports are not a single filesystem-wide atomic transaction.
  DT1/HD and PNG-to-HDF5 outputs preserve dotted basenames.
* Head-only geometry extraction crops component IDs, rigid masks and tag
  arrays together with the voxel data, retaining a consistent geometry file.

Execution and documentation
---------------------------

* Metal submits command buffers without waiting after every dispatch and
  sizes PML launches to their slabs, while retaining synchronization before
  host reads and required transfers. This reduces unnecessary host/device
  waiting; the speedup depends on the model and device.
* Snapshot device allocation validates the selected backend before allocating
  arrays. Example-resource errors identify the operation and path more clearly.
* Eigenmode source/receiver and PML logs format coordinates as ordinary numbers
  rather than NumPy scalar representations; this changes logging only.
* Python user-object argument documentation is expanded for model-building
  tools, and source-install instructions follow the default ``master`` branch.

Review numerical reference results and regenerate affected derived data where
appropriate. For migration from v3, use :doc:`migration_v3_v4`. The
`4.0.2 change history
<https://github.com/gprMax/gprMax/compare/v.4.0.1...v.4.0.2>`_ will identify the
released revisions once the 4.0.2 tag is published.

4.0.1 (Caol Ila)
================

This maintenance release collects corrections since 4.0.0 and refinements to
the existing eigenmode-port functionality. Automatic mode tracking remains
experimental and opt-in; ``tracking="legacy"`` remains the default.

For installation and upgrades, follow :ref:`installation`. Python 3.11--3.13
remains supported, with CPU wheels for Linux x86-64, Windows x86-64, and Intel
and Apple Silicon macOS. GPU and MPI execution still require their optional
bindings and system runtimes; see :doc:`accelerators` and :doc:`hpc`.

Corrections that can change model results
------------------------------------------

* **Discrete plane waves on unequal cell spacings:** correct the z-derivative
  coefficient in the ordinary, non-dispersive CPU auxiliary ``Ey`` update.
  The previous expression could produce incorrect fields or numerical
  instability when ``dx != dz`` and that derivative was active. MPI uses
  the same corrected CPU path. The axial, dispersive and GPU implementations
  already used the correct coefficient. Models with ``dx == dz`` are
  unaffected by this particular correction. Re-run affected CPU/MPI models
  rather than reusing their previous traces.
  (`PR #864 <https://github.com/gprMax/gprMax/pull/864>`_)

* **PML calibration:** automatic boundary-PML ``sigma.max`` calibration now
  samples the entrance plane on the negative faces, consistently with the
  positive faces. Calibration can change when the sampled material
  cross-section differs along the PML normal. Models whose material
  cross-section is constant along that normal are unaffected. This correction
  does not make material discontinuities along the PML normal a
  reflection-free termination.
  (`PR #856 <https://github.com/gprMax/gprMax/pull/856>`_)

* **Modal boundaries:** correct PEC/PMC modal constraints, the default
  eigenvalue guess, and agreement between eigenmode-port windows and their
  virtual waveguides. These corrections also apply with legacy tracking,
  so keeping that tracker does not guarantee identical results to 4.0.0.
  User-specified eigenvalue guesses remain unchanged.
  (`PR #846 <https://github.com/gprMax/gprMax/pull/846>`_,
  `PR #848 <https://github.com/gprMax/gprMax/pull/848>`_)

.. important::

   Every modal port window has a PEC transverse rim. An SIBC wall exactly on
   that rim is treated as PEC in the modal solve and virtual guide, removing
   its modelled surface loss there. To retain its impedance, place the wall
   inside the window with at least one opaque voxel beyond it. Open structures
   also need enough transverse space that the artificial PEC rim does not
   materially affect the mode. See :doc:`eigenmode_port`.

Eigenmode tracking and surface-impedance refinements
----------------------------------------------------

* ``tracking="auto"`` adds experimental frequency-dependent mode assignment,
  automatic degeneracy detection and mode-quality diagnostics. Review those
  diagnostics when using it; legacy tracking remains the default.
  ``anchors="auto"`` selects frequency anchors and does **not** enable
  automatic tracking. See :ref:`eigenmode-mode-tracking` and the new TE11 and
  mode-crossing examples in :doc:`examples_advanced`.

* Degenerate-mode polarization controls accept a shared direction, two shared
  directions, or exact mode assignments. Exact mode mappings require
  ``tracking="legacy"`` and declared degenerate pairs; with automatic
  tracking, use shared directions instead. Invalid, non-transverse or
  insufficiently distinct directions are rejected. See :doc:`eigenmode_port`
  for the Python API and equivalent hash-command forms.

* CPU SIBC/PML and virtual-guide coupling support uniformly extruded
  heterogeneous, lossy and electrically dispersive retained hosts, subject
  to the documented material and geometry restrictions. SIBC remains
  unsupported on GPU backends, MPI and subgrids. See
  :doc:`impedance_surfaces` and :ref:`virtual-waveguide`.

Reliability, installation and toolboxes
---------------------------------------

* DebyeFit uses loss magnitude when normalising imaginary-part error, avoiding
  a singularity at unit loss. Optimizer adapters honour the iteration budget,
  preserve keyword arguments and support parallel differential evolution.
  Inputs are validated before fitting; invalid parameters raise ``ValueError``
  instead of exiting Python. Zero-strength poles are omitted from exported
  materials, including lossless fits. The Rawdata conductivity/sign convention
  and the legacy weight solver's limitations are clarified in
  :doc:`inc_DebyeFit`. Re-run affected fits rather than reusing their old error
  estimates or automatic pole counts.

* Accept valid asymmetric PML thicknesses while rejecting overlapping
  opposing slabs. MPI validates global thicknesses before partitioning and
  coordinates checks that a boundary slab fits its owning partition.
  Internal ranks with no native PML are not required to contain one.
  (`PR #845 <https://github.com/gprMax/gprMax/pull/845>`_,
  `PR #857 <https://github.com/gprMax/gprMax/pull/857>`_,
  `PR #858 <https://github.com/gprMax/gprMax/pull/858>`_)

* Guard source-build setup execution for Windows multiprocessing, and include
  the CUDA complex-number header only for models that need complex dispersive
  coefficients.
  (`PR #842 <https://github.com/gprMax/gprMax/pull/842>`_,
  `PR #844 <https://github.com/gprMax/gprMax/pull/844>`_)

* GeometryImport and STLtoVoxel assignment-template preparation expose
  ``--overwrite`` and report existing files cleanly instead of producing an
  unhandled traceback. Existing templates remain protected unless replacement
  is explicitly requested.
  (`PR #851 <https://github.com/gprMax/gprMax/pull/851>`_,
  `PR #862 <https://github.com/gprMax/gprMax/pull/862>`_)

* DT1 export tolerates non-ASCII metadata in the companion HD text file by
  replacing unsupported characters, rather than failing the export. This
  does not introduce a new Unicode DT1 format. A-scan plotting errors identify
  the actual receiver, and DebyeFit/optimizer option dictionaries no longer
  share mutable defaults across calls.
  (`PR #847 <https://github.com/gprMax/gprMax/pull/847>`_,
  `PR #850 <https://github.com/gprMax/gprMax/pull/850>`_,
  `PR #854 <https://github.com/gprMax/gprMax/pull/854>`_)

* Align retained development-only Python plane-wave initialization and curl
  coefficients with the corresponding Cython expressions. Normal solver
  execution continues to use the compiled, precomputed path; this does not
  enable a supported alternative Python solver.
  (`PR #860 <https://github.com/gprMax/gprMax/pull/860>`_,
  `PR #864 <https://github.com/gprMax/gprMax/pull/864>`_)

* Repair hosted documentation builds, expand real-hardware Metal NTFF
  qualification, and clean up duplicate kernel metadata, MPI initialization
  and type annotations without introducing runtime circular imports.
  (`PR #839 <https://github.com/gprMax/gprMax/pull/839>`_,
  `PR #841 <https://github.com/gprMax/gprMax/pull/841>`_,
  `PR #852 <https://github.com/gprMax/gprMax/pull/852>`_,
  `PR #853 <https://github.com/gprMax/gprMax/pull/853>`_,
  `PR #861 <https://github.com/gprMax/gprMax/pull/861>`_)

Review numerical reference results for models affected by the corrections
above. For the larger transition from version 3, use :doc:`migration_v3_v4`.
The `complete change history
<https://github.com/gprMax/gprMax/compare/v.4.0.0...v.4.0.1>`_ records the
changes between the release tags once 4.0.1 is published.
