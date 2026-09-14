.. _migration-v3-v4:

****************************
Migrating from gprMax 3 to 4
****************************

This guide covers migration from gprMax 3.1.7 to gprMax 4.0. Changes affecting
only earlier v4 development builds are identified separately below.

Version 4 retains the hash-command model interface, but it is not a
bit-for-bit replacement for version 3. Model-building validation, source
timing, some material calculations, the Python API and output processing have
changed. A model that runs successfully can therefore produce different
results, even when its geometry and input parameters are unchanged.

This guide explains what existing projects need to change and verify. Use
:doc:`features` for an overview of v4 capabilities and the command references
for the complete supported syntax.

At a glance
===========

.. list-table:: Migration checklist
    :header-rows: 1
    :widths: 25 45 30

    * - Area
      - Action
      - Details
    * - Installation
      - Keep v3 separate; install v4 and its matching toolboxes together.
      - :ref:`migration-installation`
    * - Launchers and Python scripts
      - Use ``gprMax.run``; review MPI, restart and removed CLI options.
      - :ref:`migration-launchers`
    * - Geometry files
      - Convert exported HDF5/text-material pairs to keyed HDF5/JSON pairs.
      - :ref:`migration-geometry`
    * - Results and plots
      - Expect ``.h5`` results; verify receiver identities and physical time axes.
      - :ref:`migration-results`
    * - Numerical reference data
      - Recheck source timing, magnetic/Drude materials and snapshot times.
      - :ref:`migration-numerics`
    * - Repeated experiments
      - Keep simple scan controls or select a suitable study; do not reuse
        geometry for changes to materials, loads or mesh.
      - :ref:`migration-studies`

.. _migration-installation:

1. Install v4 without replacing v3
==================================

Preserve the v3 environment, input files, imported geometry, material tables,
custom antenna scripts and a representative set of outputs. Record the exact
gprMax version, Python environment, grid spacing, time step, precision and
solver backend. Work on copies in a separate v4 results directory.

Follow the complete installation guide
--------------------------------------

Use :ref:`Installation <installation>` for the full sequence: Python
prerequisites, environment selection, installation and running an example.
Choose **one** route there:

* :ref:`PyPI installation <installation-pypi>` for ordinary modelling and
  toolbox use. It explains both ``venv`` and direct installation without one,
  including OS restrictions and when activation is needed.
* :ref:`Conda installation <installation-conda>` if you want to keep using
  Conda. The normal v4 environment name is ``gprMax-v4``; you do not need the
  larger development environment merely to run a released package.
* :ref:`Source installation <installation-source>` when modifying gprMax or
  when a suitable wheel is unavailable. Use a new v4 checkout. Its environment
  YAML also defaults to ``gprMax-v4``; choose a distinct name if you need both
  wheel and editable source installations at once.

Neither Conda nor ``venv`` is required by gprMax. A separate environment is
nevertheless recommended for migration so that v3 and its dependencies remain
available. Do not upgrade the old environment in place if you intend to keep
both versions. Installation and upgrade commands belong to the installation
guide rather than being repeated here without their prerequisites.

Existing v3 Conda users
-----------------------

Keep your existing Conda installation; there is no need to uninstall or
replace it. Follow :ref:`keeping v3 and v4 side by side
<installation-coexistence>` to create/select the new environment and switch
between versions. Both use ``python -m gprMax``: the selected interpreter and
import paths choose the version, not which version was installed first.

If the old installation was editable, keep its source checkout too: the v3
environment still imports from those files. Do not update that checkout to
v4, apply the v4 environment YAML to the old environment, or use Conda's
``base`` for the project.

Avoid importing the old checkout by mistake
-------------------------------------------

Work outside the v3 checkout, remove unintended old checkout entries from
``PYTHONPATH``, and select the v4 interpreter in your IDE or notebook kernel.
Changing environments alone does not prevent a local ``gprMax`` package from
shadowing the installed one. For a one-time migration or troubleshooting
check, inspect the interpreter, imported version and code location:

.. code-block:: console

    python -c "import sys, gprMax; print(sys.executable); print(gprMax.__version__); print(gprMax.__file__)"

This is a setup/troubleshooting check, not a requirement before every run.
Keep v4 toolboxes with their matching solver in the same environment.

Optional runtimes
------------------

The core package runs on CPU/OpenMP without MPI. Optional extras such as
``gprMax[cuda]``, ``gprMax[opencl]``, ``gprMax[metal]`` and ``gprMax[mpi]``
install the corresponding Python bindings; they do not supply every system
driver, compiler or MPI runtime. Distributed fractal generation additionally
requires a working MPI/FFTW stack, and parallel HDF5 writes require
MPI-enabled h5py. Use :doc:`accelerators` and :doc:`hpc` for those paths.

Update toolbox imports and commands
-----------------------------------

The installed toolbox namespace is now ``gprMax.toolboxes``, not the generic
top-level name ``toolboxes``. Update both Python imports and module commands:

.. code-block:: python

    # Old import
    # from toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_1500
    from gprMax.toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_1500

.. code-block:: console

    python -m gprMax.toolboxes.Plotting.plot_Ascan model.h5
    python -m gprMax.toolboxes.MaterialDatabase --help

These commands work from your project directory; you do not need to locate
or enter the installed package directory. Update stored optimisation builder
module names and other import strings too. There is deliberately no installed
``toolboxes`` alias: such an alias could conflict with another distribution or
a user's local package. Prefer a fresh environment when replacing an older
editable or development installation; do not remove another distribution's
``toolboxes`` files to make gprMax work.

To find toolbox-specific examples and the MATLAB utility folder:

.. code-block:: console

    python -c "from importlib.resources import files; print(files('gprMax.toolboxes'))"

Copy editable examples/assets to your own working directory. The separate
main model examples can be listed and copied with:

.. code-block:: console

    python -m gprMax.examples list
    python -m gprMax.examples copy my-gprmax-work

The destination is ``my-gprmax-work/examples``; this command does not copy
every toolbox directory. Toolbox code and compact examples are included in
the package, but some large datasets and optional dependencies are not. See
the corresponding toolbox reference for its requirements. In a source
checkout, toolbox files are now in ``gprMax/toolboxes/``; update filesystem
paths in scripts as well as imports.

.. _migration-launchers:

2. Update launchers and model-building scripts
================================================

Hash-command inputs
-------------------

Start with an ordinary run of a copied input:

.. code-block:: console

    python -m gprMax model.in
    python -m gprMax model.in -gpu 0

The second command selects CUDA device 0 and requires a working CUDA
installation. The existing ``-n``, ``--geometry-only``, ``--geometry-fixed``
and ``--write-processed`` options remain available. Run
``python -m gprMax --help`` for the current command-line interface.

.. list-table:: Launcher changes to review
    :header-rows: 1
    :widths: 32 68

    * - v3 usage
      - v4 action
    * - ``-restart N`` / Python ``restart=N``
      - Use ``-i N`` / ``i=N``. For ordinary runs, ``n`` is the number of
        models to execute starting at N, not the last model number.
    * - ``-task N`` / Python ``task=N``
      - Review the scheduler script. ``-i N -n 1`` executes one ordinary
        model starting at N, but does not append N to the output filename.
        Give each task a unique output basename, for example ``-o model_N``.
        Also check Python blocks that use the total run count.
    * - ``-mpi N`` or ``--mpi-no-spawn`` for independent models
      - Launch the ranks externally and use the boolean ``--taskfarm``
        switch. Do not translate this to domain decomposition.
    * - Python ``mpi=N`` for the old task farm
      - Use ``taskfarm=True`` under the MPI launcher. In v4, ``mpi=(x,y,z)``
        instead splits one model into spatial subdomains.
    * - ``-benchmark``
      - Use the repository's :doc:`benchmarking` drivers; this is no longer
        a solver CLI flag.
    * - ``--opt-taguchi``
      - Port the optimisation setup to :doc:`inc_Optimisation`. The old
        solver flag is not supported; this is not a one-flag replacement.

For example, these are two different MPI experiments:

.. code-block:: console

    mpiexec -n 3 python -m gprMax model.in -n 10 --taskfarm
    mpiexec -n 4 python -m gprMax model.in --mpi 2 2 1

The first uses one coordinator and two workers for ten independent models.
The second divides one model across four ranks. Retire old worker/spawn
arguments and use the current launcher examples in :doc:`hpc`.

Python API
----------

For a script that launches a hash-command input, replace the old entry point:

.. code-block:: python

    # v3
    from gprMax.gprMax import api
    api("model.in", n=1)

with:

.. code-block:: python

    # v4
    import gprMax
    gprMax.run(inputfile="model.in", n=1)

Use keyword arguments: the first positional argument to ``run`` is
``scenes``, not ``inputfile``. New object-based models use ``gprMax.Scene``,
``scene.add(...)`` and ``gprMax.run(scenes=[scene], ...)``. A wholesale rewrite
of a valid hash-command input into the object API is not required.

Custom geometry/antenna helpers written for v3 may print hash commands rather
than return scene objects. Port those explicitly; do not assume they have the
same return contract as the current :doc:`inc_GPRAntennaModels` functions.
Also review direct imports from old internal modules.

Built-in model constructors and antenna optimisation parameters reject
unknown keyword names, even when their value is ``None``, ``False``, zero or
an empty string. For example, ``averagin=False`` raises ``TypeError``;
``averaging=False`` is the supported spelling. Correct the spelling or remove
an unsupported option rather than suppressing the exception. See
:doc:`input_api` and :doc:`input_hash_cmds` for the accepted parameters.

Reduced 2-D models
------------------

The legacy one-cell-thick domain convention remains supported as TM. For new
or revised inputs, make the invariant axis and reduction explicit:

.. code-block:: none

    #domain_mode: TM
    #domain: 0.5 0.5 inf

Use ``TE`` only when that is the intended physical polarisation, not as a
migration shortcut. ``inf`` is resolved to the required internal thickness;
it does not allocate an infinite domain. Sources and receivers can use
``inf`` on that same invariant axis. Avoid hard-coded internal TE layer
coordinates, especially in parameterised models. Consult the domain-mode
entries in :doc:`input_hash_cmds` before changing source components.

.. _migration-geometry:

3. Convert exported geometry, not ordinary material commands
=============================================================

An ordinary ``#material`` declaration in a model remains a hash command.
The conversion requirement concerns geometry exported with a companion text
material table, not every model containing materials.

Keep the original pair and run:

.. code-block:: console

    python -m gprMax.toolboxes.MaterialDatabase convert-geometry geometry.h5 materials.txt

The default outputs are ``geometry_converted.h5`` and
``geometry_materials.json``. Update the import, retaining the original
insertion position and any averaging option:

.. code-block:: none

    #geometry_objects_read: 0 0 0 geometry_converted.h5 geometry_materials

The final token is the database name, not the old text filename. For the API,
use ``material_database="geometry_materials"`` instead of ``matfile=...``.
Keep the converted HDF5/JSON pair together.

The converter preserves geometry indices and associates them with material
keys; it does not assume that v3 and v4 built-in numeric material IDs match.
It requires the complete original material table. Do not manually renumber
indices or rename a text file to JSON. See :ref:`legacy_geometry_conversion`
for accepted commands and validation limits.

.. include:: _includes/geometry_output_filenames.rstinc

.. _migration-results:

4. Upgrade result readers, plots and port processing
=====================================================

Filenames and receiver identity
--------------------------------

v3 normally wrote ``model.out``; v4 normally writes ``model.h5`` for a single
run. Both are HDF5, but changing the extension does not convert the schema.
Update file globs, archive names, merge commands, plotting scripts and
external readers. Multiple ordinary runs have numbered output basenames.

Public receivers use ``/rxs/rxN`` paths, but those numbers are file-local
selectors, not permanent sensor identities. v4 writes construction-order and
layout metadata independent of MPI partitioning. Arrays traverse x/y/z with
z varying fastest; individual Rx commands are built before RxArray commands.
This is not literal line order across mixed command types.

Name receivers explicitly when comparing independently built models. For
example, a receiver declared as ``measurement`` can be read with:

.. code-block:: python

    from gprMax.toolboxes.SFCW.processing import load_receiver
    trace = load_receiver("model.h5", "name:measurement", "Ez")

Match identity, component and position before subtraction, background removal,
stacking or merging. For moving scans, positions may change while the sensor
identity remains the same. The current tools retain per-trace mapping metadata
and reject ambiguous matches; do not bypass those checks merely to produce a
plot. See :ref:`receiver-numbering` for Python and MATLAB selectors.

.. note::

    **Earlier v4 development files:** some used lexicographic receiver-name
    sorting. v3 used insertion order. Neither supplies the current ordering
    metadata. Do not describe alphabetical sorting as a v3 behaviour or assume
    ``rx1`` selects the same sensor in all three cases. Old files are not
    rewritten; inspect their labels and positions explicitly.

Physical time axes and ports
----------------------------

For current receiver datasets, calculate time from that dataset's attributes:

.. code-block:: python

    import h5py
    import numpy as np

    with h5py.File("model.h5", "r") as output:
        field = output["rxs/rx1/Ez"]  # First verify this receiver's identity.
        values = field[...]
        time = (np.arange(values.shape[0]) * field.attrs["SampleInterval"]
                + field.attrs["TimeSampleOffset"])

Electric samples represent E at n*dt; magnetic fields and loop currents have
an offset of -dt/2. Source groups additionally describe the exact excitation
and its waveform-evaluation time. Older files without these attributes need
a version-aware reader or a verified legacy convention, not invented metadata.

Port histories are not interchangeable with raw receiver fields. An electric
field is in V/m; the source-oriented gap voltage is -E_parallel*dl. Current
port processing also accounts for its particular feed, staggering and
reference impedance. Use stored ``time``, ``time_current`` and ``frequency``
arrays where supplied, and each quantity's validity mask for spectra.

In supported 3-D models, finite-resistance voltage sources produce automatic
voltage-port outputs; their aligned histories use N-1 half-step samples.
Supported hard-voltage
ports retain N voltage samples, including time zero, and a separately timed
current history. Transmission lines retain ``/tls/tlN`` outputs. Do not move
all feed readers to ``/ports`` or assume all source types have circuit ports.
See :doc:`sources_ports` and :doc:`output` for each source's contract.

A zero-amplitude **finite-resistance** voltage source is a passive loaded
receiver, not a hard clamp. Use its measured voltage, but do not infer its own
source-normalised S11 from a zero incident spectrum. Drive that port in a
separate case when characterising its input. The current GSSI 2 GHz toolbox
model exposes ``gssi2000_tx`` and ``gssi2000_rx`` ports; see the receiver-port
update in :doc:`inc_GPRAntennaModels` before changing old antenna scripts or
NTFF power normalisation.

Do not treat additional HDF5 groups or changed material indices as an automatic
physics failure. For example, distributed SAR/radiometry can use output-local
material IDs with their own catalogue. Compare material identities through
that catalogue and physical arrays separately. Existing result files without
newly requested outputs cannot acquire those quantities by renaming the file.

.. _migration-numerics:

5. Revalidate numerical reference results
=========================================

There is no single switch that restores every v3 numerical behaviour. The
following changes can affect results even when parsing succeeds:

* **Source clocks:** impressed electric-current sources (including finite-R
  voltage sources) use the half-step current lattice; magnetic-current
  sources use the full-step lattice. Their waveform evaluation differs from
  v3. Revisit fitted delays, source deconvolution and mixed-source phase
  comparisons rather than applying one global empirical time shift.
* **Hard voltage sources:** resistance zero prescribes the electric field
  directly. An active source initialises E(0), then prescribes E at the new
  full time level after each electric update. The first receiver/snapshot
  sample is therefore not necessarily all zeros. A waveform that becomes
  zero still clamps the edge while the source is active. To apply a hard
  impulse only at time zero and then release the edge, use start=0 and
  0 < stop < dt; stop=dt also includes the next, zero-valued clamp. Use the
  owning grid's dt. Releasing a clamp changes the physical boundary condition.
* **Magnetic interfaces:** harmonic magnetic averaging is now the default.
  ``#magnetic_averaging: arithmetic`` selects the earlier mixing rule, but
  does not undo other source, magnetic-component placement or PEC-interface
  corrections. It is not a guarantee of exact v3 reproduction.
* **Drude materials:** the effective conductivity calculation includes the
  vacuum-permittivity factor and no longer accumulates into the declared
  conductivity when coefficients are rebuilt. Revalidate old Drude fits;
  do not compensate by arbitrarily rescaling material parameters.
* **Snapshots:** integer selections are zero-based electric time levels
  0 through N-1; a floating time maps to the nearest full time step. v3's
  off-by-one timing is not retained. Snapshot files now default to VTKHDF
  (``.vtkhdf``); use a suitable viewer and review scripts expecting ``.vti``.
  Native HDF5 is another documented output option.

Current hard-source excitation histories include the initial sample.
Earlier v4 development outputs used a different hard-source offset and
shorter port histories; that is a separate transition from the v3 source
clock changes. Honour stored metadata when working with those files.

Keep precision, mesh, physical observation point and waveform definition
controlled during comparisons. Do not require bitwise equality with v3 or
approve all differences as intentional. Compare suitable physical quantities
against analytical solutions, convergence results or validated measurements;
investigate remaining discrepancies before replacing reference data. The
:doc:`comparisons_analytical` examples provide starting points.

.. _migration-studies:

6. Move repeated experiments deliberately
==========================================

Simple ``#src_steps`` / ``#rx_steps`` scans remain supported. They need not
be rewritten as studies solely to migrate. If adopting the study API, select
the family for the experiment: GPR acquisitions, finite-resistance voltage
ports, other terminal-source states, eigenmode ports or plane-wave incidence.
The :doc:`studies` guide explains the supported state changes and outputs.

Studies keep geometry and discretisation fixed. Changes to material
properties, source resistance, network loads, mesh spacing or a complete
antenna's geometry require fresh model builds. Moving a point source does
not move the antenna around it. Study cases supply their run count; do not
combine a study with ordinary source/receiver stepping controls. Studies
currently do not support MPI task farming or domain decomposition.

Backend support is feature-specific. CPU and CUDA support HSG subgrids;
OpenCL, Metal and distributed MPI do not. Refining HSG regions require
double precision, while equal-resolution ratio=1 regions inherit the main
grid's supported precision. Do not assume a new backend supports every
source, output or study combination. Consult :doc:`accelerators` and the
particular command's reference.

Before adopting v4 for a project
=================================

1. Run one copied, small model on CPU and confirm the version, geometry,
   materials, receiver locations and output filenames.
2. Compare selected receiver identities and correctly timed physical signals
   against the saved v3 reference. Record explanations for accepted changes.
3. Recheck the project's magnetic, dispersive, hard-source and snapshot cases
   where applicable. Validate scientific results, not just file schemas.
4. Exercise the actual downstream workflow: merge, background subtraction,
   plots, exports and any port/antenna processing.
5. Repeat a representative case on the intended GPU/MPI configuration.
   Archive the validated model, environment, output metadata and comparison
   criteria before updating production reference data.

Keep the v3 archive available for reproducibility. Report a discrepancy with
a minimal input, both exact versions, backend/precision information, and the
specific physical comparison that fails.
