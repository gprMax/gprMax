Toolboxes is a sub-package where useful Python modules contributed by users are stored.

*********
Utilities
*********

Information
===========

This package contains various scripts and helper functions.

Package contents
================

HPC
---

This folder contains scripts to assist with running gprMax on high-performance computing (HPC) systems.

MATLAB
------

The ``MATLAB`` folder contains a general reader for the complete gprMax HDF5
output hierarchy and a converter to MATLAB v7.3 MAT files. Add the folder to
the MATLAB path and read an output without first converting it:

.. code-block:: matlab

    addpath("toolboxes/Utilities/MATLAB")
    free_space = gprmax_read_h5("antenna_free_space.h5");

    Ez = free_space.data.rxs.rx1.Ez;
    dt = free_space.header.dt;
    S11 = free_space.data.ports.feed.S11;

``gprmax_read_h5`` recursively supports receiver and source histories, ports,
S-parameter studies, NTFF, SAR/radiometry, subgrids, and other groups without
hard-coding those schemas. It reconstructs h5py complex datasets as native
MATLAB complex arrays and restores multidimensional arrays to gprMax's HDF5
dimension order. Thus a merged receiver remains ``(time, trace)``, an S matrix
remains ``(frequency, output port, input port)``, and an NTFF field remains
``(frequency, direction)``. Numeric precision, integer arrays, strings, and
all root/group/dataset attributes are retained.

HDF5 names that are not valid MATLAB fields are made valid and unique. The
``name_map`` and ``header_name_map`` members retain the original names and
their MATLAB equivalents. The normal structure is:

.. code-block:: text

    result.header          common root attributes such as dt and Iterations
    result.data            recursively mirrored HDF5 groups and datasets
    result.metadata        attributes and dimensions for every loaded object
    result.name_map        original HDF5 path to MATLAB field-path mapping
    result.info            source filename and reader information

``gprmax_h5_get`` resolves an original HDF5 path through this mapping, which
is useful when a user-selected object ID is not itself a valid MATLAB field:

.. code-block:: matlab

    [S11, metadata] = gprmax_h5_get(result, "/ports/feed-1/S11");

Large outputs can be read selectively so unused datasets are not loaded:

.. code-block:: matlab

    ports = gprmax_read_h5("antenna.h5", ...
        "Paths", ["/ports", "/eigenmode_ports"]);

Use ``gprmax_h5_to_mat`` when a self-contained MAT file is more convenient:

.. code-block:: matlab

    gprmax_h5_to_mat("antenna_free_space.h5")
    gprmax_h5_to_mat("antenna_halfspace.h5")
    load("antenna_free_space.mat")
    load("antenna_halfspace.mat")

Each MAT file contains a variable derived from its input filename, so the two
results above coexist as ``antenna_free_space`` and ``antenna_halfspace``.
Invalid filename characters are converted using ``matlab.lang.makeValidName``.
An explicit name and destination can instead be supplied:

.. code-block:: matlab

    gprmax_h5_to_mat("model.h5", ...
        "OutputFile", "free_space_result.mat", ...
        "VariableName", "free_space", ...
        "Paths", ["/rxs", "/ports"]);

Several HDF5 files can either be converted to separate MAT files in one call,
or collected under ``<variable>.runs`` in one MAT file:

.. code-block:: matlab

    files = ["free_space.h5", "dry_soil.h5", "wet_soil.h5"];
    gprmax_h5_to_mat(files)
    gprmax_h5_to_mat(files, ...
        "OutputFile", "soil_study.mat", ...
        "VariableName", "soil_study");

Existing MAT files are not replaced unless ``"Overwrite", true`` is given.
The converter uses MATLAB v7.3 because classic MAT files have restrictive size
limits. Conversion nevertheless materialises the selected arrays in memory;
for very large snapshots or dense volumetric outputs, select only the required
paths or continue to access HDF5 directly.

The A-scan and B-scan plotters are reusable functions built on the same reader.
With no arguments they retain the interactive file-picker workflow. They also
return the exact raw values and physical coordinates used in the plot:

.. code-block:: matlab

    [figures, traces] = plot_Ascan("model.h5", ...
        "Receiver", 1, ...
        "Outputs", "Ez", ...
        "FFT", true);

Receiver dataset ``SampleInterval`` and ``TimeSampleOffset`` metadata are used
to construct the physical time axis, including half-step magnetic-field
timing. Use ``"Grid", "/subgrids/fine"`` for receivers belonging to a
subgrid. The optional ``Path`` selects an arbitrary time-domain group instead,
allowing a voltage-port, transmission-line, or magnetic-frill history to be
plotted with the same function:

.. code-block:: matlab

    [figure_handle, feed] = plot_Ascan("antenna.h5", ...
        "Path", "/ports/feed", ...
        "Outputs", "Vtotal");

Figures can be rendered without a window and saved reproducibly:

.. code-block:: matlab

    plot_Ascan("model.h5", ...
        "Outputs", ["Ex", "Hy"], ...
        "Visible", false, ...
        "Save", true, ...
        "OutputDirectory", "plots");

``plot_Bscan`` expects the modern merged ``(time, trace)`` dataset and does
not transpose it again. When retained trace positions are available, its
horizontal axis is the cumulative three-dimensional receiver or terminal
path distance; otherwise it uses trace number. Both receiver fields and real
antenna-terminal voltages are supported:

.. code-block:: matlab

    [figure_handle, field_scan] = plot_Bscan( ...
        "survey_merged.h5", "Ez");

    [figure_handle, voltage_scan] = plot_Bscan( ...
        "antenna_merged.h5", "Vtotal", ...
        "Path", "/ports/receive", ...
        "OutputFile", "antenna_voltage.png");

The ``Path`` option can similarly identify ``/tls/tl1``, ``/frills/frill1``,
or a main-grid/subgrid receiver group. Only real time-domain matrices are
accepted; complex frequency-domain quantities are rejected.

``outputfile_converter.m`` is a legacy exporter to vendor GPR formats and is
unrelated to the general HDF5-to-MAT converter.


Paraview
--------

This folder contains a Python macro to be installed into ParaView. The macro
creates named threshold items for the materials in a gprMax VTKHDF geometry
file. When semantic geometry tags are present, it also creates initially hidden
``Tag - <name>`` threshold items so tagged objects can be viewed independently
of their electromagnetic materials. Sources, receivers, and any exterior PML
regions contained in the selected geometry view are added to the pipeline as
separate objects.


convert_png2h5.py
-----------------

This module converts discrete colours in a Portable Network Graphics (PNG)
image into a voxel-only HDF5 geometry file and companion versioned JSON
material database. Both files use the current
``GeometryObjectsRead``/``#geometry_objects_read`` schema. The resulting
geometry is two-dimensional but can be extended by a chosen number of cells
in the z (invariant) direction. Usage from the top-level gprMax directory is:

.. code-block:: none

    python -m toolboxes.Utilities.convert_png2h5 imagefile dx dy dz

where:

* ``imagefile`` is the PNG filename, including its path;
* ``dx dy dz`` are the three model spatial discretisations in metres.

There is an optional command line argument:

* ``--zcells`` is the number of cells in the z direction (default: one);
* ``--output-file`` selects the HDF5 destination.

For example create a HDF5 geometry objects file from the PNG image ``my_layers.png`` with a spatial discretisation of :math:`\Delta x = \Delta y = \Delta z = 0.002` metres, and extending 150 cells in the z-(invariate) direction of the model:

.. code-block:: none

    python -m toolboxes.Utilities.convert_png2h5 \
        my_layers.png 0.002 0.002 0.002 --zcells 150

The image is displayed interactively. Single-click each colour that should
become a material and then close the image. Selection order defines compact
material indices, while the selected RGB/RGBA values are retained in the JSON
metadata. Unselected pixels are stored as ``-1`` and therefore leave the
existing model material unchanged when the geometry is imported. Use a
palette-like image without anti-aliasing or colour gradients: every distinct
channel value is a different colour.

For the example above the utility writes ``my_layers.h5`` and
``my_layers_materials.json``. The HDF5 file contains ``/data`` and stable
``/material_keys`` and records the companion database identity. Image colours
do not determine electromagnetic properties, so the generated JSON entries
deliberately contain null constitutive values. Edit every selected material to
supply relative permittivity, electric conductivity in S/m, relative
permeability, and magnetic conductivity in S/m before running gprMax. Existing
edited JSON values are preserved when the same colour mapping is regenerated;
a changed mapping is rejected rather than silently detaching the database from
its geometry.

After completing the JSON file, import the image geometry with, for example:

.. code-block:: none

    #geometry_objects_read: 0 0 0 my_layers.h5 my_layers_materials y

or with the Python API:

.. code-block:: python

    scene.add(gprMax.GeometryObjectsRead(
        p1=(0, 0, 0),
        geofile="my_layers.h5",
        material_database="my_layers_materials",
        averaging="y",
    ))

The optional averaging choice belongs to ``GeometryObjectsRead`` rather than
the converter. Use ``"y"`` for smoothable dielectric interfaces and ``"n"``
when exact staircased voxel membership is required. The imported resolution
must match the simulation resolution.


get_host_spec.py
----------------

This module prints information about the host machine capabilities for OpenMP,
CUDA, OpenCL, and Apple Metal. It is safe to import; the hardware probing is
performed only when the module is run as a command.


outputfiles_merge.py
--------------------

gprMax produces a separate output file for each trace (A-scan) in a B-scan.
This module combines real, time-domain receiver and antenna-terminal traces
into a single two-dimensional HDF5 file and can remove the separate files
afterwards. Main-grid and subgrid outputs are supported. The files must have
matching output components, data types, metadata, iteration counts, and time
steps. Usage from the top-level gprMax directory is:

.. code-block:: none

    python -m toolboxes.Utilities.outputfiles_merge basefilename --remove-files

where:

* ``basefilename`` is the base name file of the output file series, e.g. for ``myoutput1.h5``, ``myoutput2.h5`` the base file name would be ``myoutput``
* ``remove-files`` is an optional argument (flag) that when given will remove the separate output files after the merge.

The columns of every merged receiver dataset correspond to the naturally
ordered input files. Per-trace physical and grid positions for receivers and
position-bearing sources are retained below ``/trace_metadata`` (and below
the corresponding subgrid's ``trace_metadata`` group). The original
single-position receiver or port attributes are retained for compatibility
and refer to the first trace. The merged grid attributes include ``ntraces``
and identify the content as
``real_time_domain_receivers_and_terminal_voltages``.

This is deliberately an A-scan-to-B-scan utility. It merges the ordinary
receiver components ``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, ``Hz``, ``Ix``,
``Iy``, and ``Iz``. It also merges the authoritative total terminal voltage
``Vtotal`` from voltage-source and rational-network ports under ``/ports``,
transmission-line feeds under ``/tls``, and magnetic-frill feeds under
``/frills``. This permits antenna B-scans to use a physical terminal response
without introducing a theoretical point receiver. A passive receiving antenna
can therefore be represented by its normal zero-excitation port and its
``Vtotal`` response.

The source types use different internal Yee-time history lengths. The merger
retains the physical ``N-1`` voltage samples of resistive/hard voltage-source
ports, all ``N`` rational-network and transmission-line samples, and the first
``N`` magnetic-frill samples (the frill's extra stored endpoint is not a
physical FDTD output sample). Sample interval, physical time offset, quantity,
and volt units are written on every merged voltage dataset.

Frequency-domain products such as S-parameters, impedance, spectra, NTFF
results, SAR/radiometry results, snapshots, and study data are not copied. A
file series with neither receiver nor terminal-voltage A-scans is rejected
rather than producing an empty or misleading file. Complex data, mixed
precision, and an already merged two-dimensional dataset are also rejected.

The resulting voltage B-scan can be plotted directly. For example:

.. code-block:: none

    python -m toolboxes.Plotting.plot_Bscan antenna_merged.h5 Vtotal \
        --trace-group ports/receive

The trace group can similarly be ``tls/tl1`` or ``frills/frill1``.


outputfiles_segy.py
-------------------

This module exports a naturally ordered series of gprMax A-scans as a
`SEG-Y revision 2.1 <https://seg.org/publications/seg-technical-standards/>`_
file by default, with an optional revision-1 GPR compatibility profile. It
writes one selected
receiver component per trace, preserves the source and receiver coordinates
from every model run, and stores samples as big-endian IEEE 32-bit floating
point values. For example:

.. code-block:: none

    python -m toolboxes.Utilities.outputfiles_segy myoutput Ez

This reads ``myoutput1.h5``, ``myoutput2.h5``, and so on, and writes
``myoutput_Ez.sgy``. The principal options are:

* ``--receiver 2`` selects receiver 2 (the default is receiver 1)
* ``--source srcs/src2`` selects the source whose position is written when a
  model contains more than one source
* ``--trace-group tls/tl1`` selects a position-bearing group other than an
  ordinary receiver, allowing a real time-domain voltage such as ``Vtotal``
  to be exported
* ``--grid subgrids/fine`` selects receiver and source data in a subgrid
* ``--output-file survey.sgy`` chooses the output name
* ``--line-number 15`` sets the SEG-Y line number
* ``--profile gpr`` writes the legacy GPR convention used by software such as
  GeoLitix, instead of the standards-compliant revision 2.1 default
* ``--overwrite`` permits replacement of an existing SEG-Y file

The original, separate A-scan files must still exist. Although the standard
merged gprMax output now retains per-trace source and receiver positions, the
interchange exporters deliberately consume original one-dimensional A-scans
so that every exported trace is validated independently.

Normal FDTD time steps are much shorter than one microsecond and cannot be
represented by SEG-Y's legacy integer-microsecond sample-interval fields. The
exporter therefore follows revision 2.1 and stores the exact interval in the
extended IEEE 64-bit sample-interval field in binary-header bytes 3273--3280;
the legacy field is zero unless the interval is exactly representable. Software
limited to SEG-Y revision 1 may consequently reject the sampling metadata or
require a non-standard workaround. The physical time of sample zero is
recorded in the textual header because the standard trace delay fields cannot
represent the sub-nanosecond Yee-grid time offsets used by gprMax.

Many GPR programs instead use a legacy adaptation in which the ordinary
SEG-Y interval fields contain integer **picoseconds**, despite those fields
being defined as microseconds by the seismic standard. GeoLitix describes its
output as using the common GPR SEG-Y format for this reason. For compatibility,
use:

.. code-block:: none

    python -m toolboxes.Utilities.outputfiles_segy myoutput Ez --profile gpr

This profile writes SEG-Y revision 1, generic live/production traces, and the
picosecond interval convention expected by legacy GPR readers. If the FDTD
time step is not an integer number of picoseconds, the exporter linearly
resamples each trace to the nearest integer-picosecond interval. The original
and exported intervals are recorded in the textual header. Resampling is never
performed by the default ``standard`` profile. The GPR convention is useful
for software interoperability but is intentionally not described as compliant
with the SEG-Y seismic standard.

Coordinates are written in metres with a scalar of ``-10000`` (0.1 mm
precision). The source and receiver z coordinates are written as elevations.
The trace samples must be real, uniformly sampled time-domain fields, currents,
or voltages. Complex or frequency-domain quantities such as ``S11`` and
``Zin`` are deliberately rejected. Samples retain their native SI quantity and
are not amplitude normalised; the selected quantity and its units are recorded
in the textual header. This is a synthetic-data interchange export, not a claim
that a field component is a calibrated instrument voltage.

A complete 151-trace 2D TM example containing four buried PEC metal bars is
provided in ``toolboxes/Utilities/examples/four_metal_bars.in``. In the 2D
model the circular cross-sections extend through the invariant direction and
therefore represent bars rather than finite pipes. The 0.60 m scan aperture
provides visible flanks for all four hyperbolic responses. It can be run and
exported with:

.. code-block:: none

    python -m gprMax toolboxes/Utilities/examples/four_metal_bars.in \
        -n 151 --geometry-fixed
    python -m toolboxes.Utilities.outputfiles_segy \
        toolboxes/Utilities/examples/four_metal_bars Ez

The input also requests a voxel geometry view named
``four_metal_bars.vtkhdf``. To generate and inspect only that file before
running the complete scan, use:

.. code-block:: none

    python -m gprMax toolboxes/Utilities/examples/four_metal_bars.in \
        --geometry-only

The resulting survey file contains four hyperbolic bar responses and can be opened
directly by a revision-2-aware SEG-Y reader. Export the SEG-Y file before
removing the individual A-scan HDF5 files.

For GeoLitix and other revision-1 GPR software, append ``--profile gpr`` to
the export command.


outputfiles_seg2.py
-------------------

This module exports the same naturally ordered gprMax A-scan series in
`SEG-2 revision 1 <https://seg.org/wp-content/uploads/2025/11/seg_2.pdf>`_.
SEG-2 was designed for shallow seismic and digital-radar data. Unlike the
legacy SEG-Y interval fields, its ``SAMPLE_INTERVAL`` trace keyword is a
floating-point value in seconds. The exporter can therefore retain a normal
FDTD time step exactly without resampling. For example:

.. code-block:: none

    python -m toolboxes.Utilities.outputfiles_seg2 myoutput Ez

This writes ``myoutput_Ez.sg2`` using little-endian IEEE 32-bit floating-point
samples. Source and receiver positions are written in metres for every trace,
and the component, native SI unit, source path, and physical time of sample
zero are retained as SEG-2 free-format metadata. The options ``--receiver``,
``--source``, ``--trace-group``, ``--grid``, ``--output-file``, and
``--overwrite`` have the same meanings as for ``outputfiles_segy.py``.

SEG-2 has no universal controlled vocabulary for all GPR metadata. Standard
keywords such as ``SAMPLE_INTERVAL``, ``RECEIVER_LOCATION``,
``SOURCE_LOCATION``, and ``TRACE_TYPE RADAR_DATA`` are accompanied by
explicit ``GPRMAX_*`` metadata. Readers are expected to retain unrecognised
free-format strings, but applications may display only a subset of them.


outputfiles_dt1.py
------------------

This module exports a gprMax A-scan series as the paired Sensors & Software
``.DT1`` data and ``.HD`` header files used by many GPR-processing programs:

.. code-block:: none

    python -m toolboxes.Utilities.outputfiles_dt1 myoutput Ez \
        --nominal-frequency 1000

This writes ``myoutput_Ez.DT1`` and ``myoutput_Ez.HD``. The optional nominal
frequency is in MHz; its default value of zero means unspecified. Source and
receiver coordinates are stored in every binary trace header. The scalar
survey position is the cumulative three-dimensional receiver path length, so
non-x-directed survey lines are not collapsed. The options ``--receiver``,
``--source``, ``--trace-group``, ``--grid``, ``--output-file``, and
``--overwrite`` have the same meanings as for the other exporters.

DT1 stores samples as signed 16-bit integers. The exporter therefore uses one
symmetric scale for the complete survey, maps the largest absolute value to
32767, and records ``GPRMAX COUNT SCALE`` and the reconstruction rule
``SI_VALUE = INTEGER_COUNT * GPRMAX_COUNT_SCALE`` in the HD file. This keeps
relative amplitudes between traces but introduces at most half of one count
of quantisation error. Software unaware of the gprMax extension will show the
integer counts, which remain suitable for conventional GPR processing. SEG-Y
or SEG-2 should be preferred when native floating-point amplitudes are
required.

DT1/HD is a vendor format rather than an open interchange standard. The
implementation follows the commonly supported 128-byte trace header layout
and has been checked with the open-source GPRPy reader. Exact gprMax sampling
and Yee-grid timing metadata are added after the conventional HD entries, so
readers that parse only the established fields remain compatible.

The four-metal-bar model shown in the SEG-Y section can be exported to either
additional format without rerunning it:

.. code-block:: none

    python -m toolboxes.Utilities.outputfiles_seg2 \
        toolboxes/Utilities/examples/four_metal_bars Ez
    python -m toolboxes.Utilities.outputfiles_dt1 \
        toolboxes/Utilities/examples/four_metal_bars Ez \
        --nominal-frequency 1000
