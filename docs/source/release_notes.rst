*************
Release notes
*************

Unreleased / next release
=========================

Fractal volumes and rough surfaces
----------------------------------

* **Same-seed fractal geometry can change:** fractal volumes and rough
  surfaces now centre their spectral filters on the actual FFT DC position.
  Regenerating them with the same seed can therefore give different geometry
  when any fractal-generation dimension is odd-sized or a single cell. These
  are the dimensions of the fractal being generated, not necessarily those of
  the overall model domain. Fractals whose generation dimensions are all even
  are unaffected, and previously exported geometry is unchanged; reuse it
  where earlier geometry must be reproduced exactly.

PMC symmetry boundaries and PML
-------------------------------

* Fixed missing PML corrections where an upper PMC symmetry boundary meets
  a perpendicular PML. These intersections could produce incorrect fields
  and late-time instability. Electric and magnetic corrections now include
  the required staggered boundary samples on CPU/MPI, CUDA, OpenCL and Metal;
  ordinary PMLs and internal MPI rank boundaries retain their existing extents.

STLtoVoxel and geometry import
------------------------------

* **STLtoVoxel cells now match the STL:** the converter fills the cells whose
  centres are inside each closed surface, on a grid whose lower corner is a
  whole number of cells from the origin, using STEPtoVoxel's voxeliser.
  Previously objects were 8--21 % too large and shifted by up to one cell
  (a 20 mm cube at 2 mm gave 1101 cells instead of 1000). Converted files
  therefore differ in cell count and grid shape; re-convert STL geometry used
  in existing models and place it with the printed ``p1``.
* STLtoVoxel writes the grid origin as ``origin_xyz`` and prints the
  ``GeometryObjectsRead`` ``p1`` that keeps the STL coordinates. New options
  ``--rotate``, ``--rotate-about``, ``--scale``, ``--translate``,
  ``--supersample`` and ``--pad-cells``; the settings are recorded in the
  ``conversion_settings`` attribute. The ``parallel`` argument of the
  ``convert`` functions is deprecated and ignored.
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

* Automatic tracking now resolves longitudinal families and symmetry subgroups
  before selecting polarization within each eligible doublet. Crossing recovery
  excludes spatially unresolved reference candidates, even when their
  eigenproblem residual is small.

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
  ``pip install "gprMax[step]"`` installs cadquery-ocp as an alternative to
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
