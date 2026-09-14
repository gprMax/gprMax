Reading and processing solver outputs
-------------------------------------

The objective receives ``evaluate(parameters, output)`` after a candidate finishes.
``output`` is a ``SimulationOutput`` object for that candidate; you do not search
a result directory to guess which file belongs to it.

.. list-table:: Public access points
   :header-rows: 1
   :widths: 28 72

   * - Interface
     - Result
   * - ``output.file``
     - Path to the main solver HDF5 file; pass it to h5py or another postprocessor.
   * - ``output.datasets()``
     - Dataset paths actually saved in that file; lists names without loading arrays.
   * - ``output.receiver(name, component, grid="/")``
     - One receiver trace, its sampling times and units.
   * - ``output.port(name, grid="/")``
     - Terminal S11 and input impedance spectra, frequency and validity masks.
   * - ``output.save_npz(name, **arrays)``
     - Save intermediate arrays for this evaluation, for example a filtered waveform.

The convenience readers live in ``quantities.py``. ``results.py`` supplies the
methods above; ``processing.py`` manages evaluation records and saved artifacts.
The same readers work without a campaign:

.. code-block:: python

   from gprMax.toolboxes.Optimisation import read_receiver, read_port

   trace = read_receiver("output.h5", "probe", "Ez")
   spectrum = read_port("output.h5", "frills/frill1")

These two calls illustrate different output types: your model must actually have
the named receiver or terminal. The directory name does not determine which
physical quantities exist.

ReceiverTrace fields
~~~~~~~~~~~~~~~~~~~~

``output.receiver("probe", "Ez")`` selects the receiver ID from your model and
one saved component. It returns materialised arrays; the reader closes the file
before returning.

.. list-table:: ReceiverTrace
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``values``
     - One-dimensional sample array for the selected component.
   * - ``time``
     - Matching one-dimensional time array, in seconds.
   * - ``unit``
     - Component unit: for example V/m for E, A/m for H, or A for current.
   * - ``dataset``
     - Exact HDF5 dataset path used by the reader.
   * - ``source``
     - Source HDF5 file path.

Sampling requires the dataset's explicit sample interval and offset metadata.
Electric and magnetic quantities need not have identical sample times. Compare
``trace.time`` arrays before subtracting signals; the waveform example shows
alignment and unit checks. This reader handles one-dimensional traces, not a
merged B-scan matrix. Use ``grid`` to select an explicit subgrid rather than
assuming its sampling matches the main grid.

PortSpectrum fields
~~~~~~~~~~~~~~~~~~~

Terminal ports, transmission lines and magnetic frills are selected under
``ports``, ``tls`` and ``frills`` respectively. Prefer a qualified name such as
``"ports/feed"`` or ``"frills/frill1"``. An unqualified ID works only if unique.
The name must match the saved solver output; the first magnetic frill is normally
``frill1``. Inspect the file if unsure.

.. list-table:: PortSpectrum
   :header-rows: 1
   :widths: 38 62

   * - Field
     - Meaning
   * - ``frequency``
     - One-dimensional frequency samples in Hz.
   * - ``s11``
     - Complex reflection coefficient, dimensionless; not dB.
   * - ``valid``
     - Boolean S11 validity mask with the same shape as frequency.
   * - ``impedance``
     - Complex input impedance in ohms.
   * - ``impedance_valid``
     - Separate Boolean impedance mask; do not substitute the S11 mask.
   * - ``reference_impedance``
     - Reference impedance in ohms used for the saved terminal spectrum.
   * - ``tail_relative_db``
     - Saved time-tail diagnostic in dB; NaN if absent.
   * - ``independent_frequency_resolution_hz``
     - Saved independent frequency resolution, or None if unavailable.
   * - ``source`` / ``group``
     - File path and exact terminal group path.

``spectrum.at(frequency_hz)`` returns complex S11 using adjacent valid frequency
samples. Interpolation acts on its real and imaginary parts. It refuses invalid
gaps and extrapolation; it does not improve independent frequency resolution.
Convert to dB explicitly in your criterion:

.. code-block:: python

   import numpy as np

   def evaluate(parameters, output):
       spectrum = output.port("frills/frill1")
       reflection = spectrum.at(3.1e9)
       return float(20 * np.log10(max(abs(reflection), 1e-12)))

This minimises S11 in dB at one frequency. It does not locate the deepest dip
anywhere in the spectrum. To plot or reduce a band, select valid samples inside
your band first. Never replace invalid bins with zero: zero reflection would
look like a perfect match. Missing groups, datasets or required attributes raise ``KeyError``. A missing
or ambiguous unqualified name, malformed data or invalid interpolation raises
``ValueError``.

Voltages, currents and other outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The terminal HDF5 groups can contain V/I time traces and spectra in addition to
S11 and Zin. ``PortSpectrum`` is a compact S11/Zin view, not a complete terminal
record. For the existing broader terminal reader, use:

.. code-block:: python

   from gprMax.toolboxes.Plotting.plot_port import discover_port_outputs, read_port_output

   print(discover_port_outputs(output.file))
   port = read_port_output(output.file, "frills/frill1")
   for trace in port.time_traces:
       print(trace.name, trace.quantity, trace.time.shape, trace.values.shape)

That reader lives in ``gprMax/toolboxes/Plotting/plot_port.py``. Its ``PortData`` contains
``time_traces``, ``spectral_traces``, ``s_parameters``, ``impedances``,
``admittances``, ``validity_masks``, ``diagnostics`` and ``metadata``. Depending on
the terminal type and solver output, traces can include incident and total V/I,
generator voltage and network or loop current. Use each time trace's own ``time``
array; source-specific V and I traces can differ in length and time offset.

.. list-table:: Current coverage in the optimisation toolbox
   :header-rows: 1
   :widths: 32 68

   * - Output family
     - How to use it now
   * - Receivers
     - ``receiver`` / ``read_receiver`` for one saved component trace.
   * - Source terminals
     - ``port`` / ``read_port`` for S11/Zin; the broader Plotting reader for V/I and other saved terminal fields.
   * - Other source records
     - Inspect and read the actual HDF5 datasets through ``output.file``.
   * - Eigenmode ports
     - Read modal data with a source-specific postprocessor or h5py; there is no dedicated modal reader in Optimisation yet.
   * - Antenna/scattering metrics
     - Configure the required monitors and postprocessing in your model/workflow, then reduce their results to your objective.

Gain, directivity, patterns, efficiency, RCS and SAR are not automatically computed
by this toolbox. Their definition and required fields depend on the experiment.
The objective may call an existing calculation using ``output.file`` or read
other artifacts your workflow explicitly creates. Raw h5py access exposes stored
data; interpreting axes, normalisation and units remains the postprocessor's
responsibility. A dedicated convenience reader for every output family has not
yet been implemented.

Inspect first, then write the calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`inspect_output.py <../../gprMax/toolboxes/Optimisation/examples/inspect_output.py>`
prints group attributes and dataset names, shapes, types and attributes without
loading all arrays. It can also demonstrate the public receiver/terminal reader:

.. code-block:: console

   python -m gprMax.toolboxes.Optimisation.examples.inspect_output path/to/output.h5
   python -m gprMax.toolboxes.Optimisation.examples.inspect_output path/to/output.h5 --group frills
   python -m gprMax.toolboxes.Optimisation.examples.inspect_output path/to/output.h5 --terminal frills/frill1 --frequency 3.1e9

Replace the path and selection with your actual file and source. For custom data,
copy an observed dataset path rather than assuming names or units:

.. code-block:: python

   import h5py

   with h5py.File(output.file, "r") as handle:
       print(list(handle.keys()))
       # values = handle["/the/dataset/path/you/inspected"][...]

Use ``output.save_npz("processed.npz", time_s=trace.time, signal=trace.values)``
to save your intermediate calculation in the objective. Names such as
``time_s`` and ``signal`` are yours, not mandatory toolbox fields. Optional
``units={"time_s": "s", "signal": "V/m"}`` records units with the processing
artifact. Return the final finite scalar after any filtering, transformation or
comparison. The optimiser interface stays the same.
