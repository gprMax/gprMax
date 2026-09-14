"""User-facing access to one completed model's output.

You use SimulationOutput inside evaluate(parameters, output). All readers are
optional: output.file can be passed directly to h5py or your own postprocessor.
See OUTPUT_READERS.rst and examples/inspect_output.py for the returned fields,
validity rules, raw-data inspection and a copyable command-line example.
"""

from .processing import ProcessingContext
from .quantities import read_port, read_receiver


class SimulationOutput:
    """The completed gprMax output passed to ``evaluate(parameters, output)``.

    ``file`` is the HDF5 path: any existing gprMax or user processing function can
    read it. Receiver/port methods are optional conveniences, not required output
    types. Names are chosen in the user's model, never inferred by the toolbox.
    """

    def __init__(self, run, context=None):
        """Expose a RunResult to a user objective without loading its arrays.

        context, when supplied, records intermediate processing artifacts.
        file points to the main HDF5 output; it is not the result directory.
        """
        self.file = run.output_file
        self._run = run
        self._context = context

    def receiver(self, name, component, *, grid="/"):
        """Read your named receiver; returns .time (s), .values and .unit.

        Example: ``output.receiver("probe", "Ez")`` matches
        ``gprMax.Rx(..., id="probe", outputs=["Ez"])`` in your model.
        """
        return read_receiver(self.file, name, component, grid=grid)

    def port(self, name, *, grid="/"):
        """Read saved S11/Zin, frequency and validity for a terminal.

        Use an exact, unique ID or a qualified name such as "frills/frill1".
        Voltage/network ports, transmission lines and magnetic frills share
        this interface; grid selects the main grid or an explicit subgrid.
        The returned fields are frequency, s11, valid, impedance,
        impedance_valid, reference_impedance, tail_relative_db,
        independent_frequency_resolution_hz, source and group.
        ``.at(frequency_hz)`` returns complex S11; it does not return dB.
        This convenience reader does not expose all V/I or modal quantities.
        """
        return read_port(self.file, name, grid=grid)

    def datasets(self):
        """List dataset paths in the main HDF5 file without loading arrays.

        This is exploration of raw contents, not a catalogue of physical units
        or attributes. The receiver and port readers interpret their supported
        schemas; other quantities can be read through h5py and self.file.
        """
        import h5py

        names = []
        with h5py.File(self.file, "r") as handle:
            handle.visititems(
                lambda name, obj: names.append("/" + name)
                if isinstance(obj, h5py.Dataset)
                else None
            )
        return names

    def save_npz(self, name, *, units=None, **arrays):
        """Save named arrays from the user's processing for later inspection.

        name is a fresh .npz filename; keyword names identify the user's arrays.
        Optional units is a name-to-unit mapping recorded by ProcessingContext.
        With an objective context, files belong to that evaluation's processing
        directory. Standalone simulate results use candidate/user_processing.
        """
        if self._context is None:
            directory = self._run.directory.parent.parent / "user_processing"
            directory.mkdir(parents=True, exist_ok=True)
            self._context = ProcessingContext(directory, {})
        return self._context.save_npz(name, units=units, **arrays)
