*************
Release notes
*************

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
