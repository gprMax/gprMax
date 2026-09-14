"""Optional extraction of physical quantities from gprMax HDF5 output.

User objectives normally reach these readers through output.receiver(...)
and output.port(...). read_receiver returns one E/H/loop-I history with
its physical time axis. read_port returns the saved S11/Zin subset from
terminal, transmission-line and magnetic-frill groups, with validity masks.

Dataset and attribute names below come from the gprMax output writers.
Receiver/port IDs come from the user's model. Neither sets a required
optimisation objective: evaluate may process output.file in any way and
return its own finite score. Proposed broader readers are not implemented
by this module yet. See CODE_WALKTHROUGH.md for the present interface.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReceiverTrace:
    """A receiver signal: values, physical time in seconds, and field/current unit.

    dataset and source identify the stored data for inspection and provenance.
    No filtering, normalisation or resampling has been applied.
    """

    values: object
    time: object
    unit: str
    dataset: str
    source: Path


def read_receiver(filename, name, component, *, grid="/"):
    """Read one named receiver with its stored physical time coordinates.

    Requires the explicit receiver timing metadata provided by current devel.
    No implicit resampling, normalisation or interpretation of invalid values.

    Parameters
    ----------
    filename : str or pathlib.Path
        One completed gprMax HDF5 output, not the campaign directory.
    name : str
        The ``id`` given to Rx in the model (stored as its Name attribute).
    component : str
        Ex/Ey/Ez in V/m, Hx/Hy/Hz in A/m, or loop Ix/Iy/Iz in A.
    grid : str
        HDF5 grid group: "/" for the main grid, or "/subgrids/<name>".

    Returns
    -------
    ReceiverTrace
        Independent in-memory arrays ``time`` and ``values``, each (samples,).
        The file is closed before returning. ``dataset`` records the full path.

    Raises
    ------
    KeyError
        A requested group/component or required timing attribute was not saved.
    ValueError
        The name is absent/ambiguous, or samples/timing are unusable.
    """
    import h5py
    import numpy as np

    units = {
        **{c: "V/m" for c in ("Ex", "Ey", "Ez")},
        **{c: "A/m" for c in ("Hx", "Hy", "Hz")},
        **{c: "A" for c in ("Ix", "Iy", "Iz")},
    }
    if component not in units:
        raise ValueError(f"Unsupported receiver component: {component}")
    filename = Path(filename).resolve()
    with h5py.File(filename, "r") as handle:
        # 1. Resolve the model's receiver ID. rx1/rx2 are storage numbering,
        # whereas Name is the user-chosen ID and is the lookup contract here.
        matches = []
        for receiver in handle[grid]["rxs"].values():
            receiver_name = receiver.attrs.get("Name")
            if isinstance(receiver_name, bytes):
                receiver_name = receiver_name.decode("utf-8")
            if receiver_name == name:
                matches.append(receiver)
        if len(matches) != 1:
            raise ValueError(f"Expected one receiver named {name!r}, found {len(matches)}")
        dataset = matches[0][component]
        # 2. Load this component only, together with its own sampling metadata.
        values = dataset[...]
        dt = float(dataset.attrs["SampleInterval"])
        offset = float(dataset.attrs["TimeSampleOffset"])
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Receiver trace must be a nonempty finite vector")
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(offset):
            raise ValueError("Receiver sampling metadata is invalid")
        # Use the component's stored timing. E, H and loop-current samples may
        # have different offsets, and subgrids have their own sampling intervals.
        time = offset + np.arange(values.size, dtype=float) * dt
        # 3. Return arrays and provenance together; no h5py object escapes the file.
        return ReceiverTrace(
            values=values, time=time, unit=units[component], dataset=dataset.name, source=filename
        )


@dataclass(frozen=True)
class PortSpectrum:
    """Saved complex S11 and input impedance on one frequency axis.

    frequency is in Hz; s11 is dimensionless; impedance (Zin) is in ohms.
    valid belongs to S11 and impedance_valid belongs to Zin. They must not
    be interchanged: a usable reflection coefficient can have singular Zin.
    reference_impedance is the port's real reference resistance in ohms.
    tail_relative_db describes residual time-domain signal level, while
    independent_frequency_resolution_hz describes the independent spectral
    resolution when recorded. source/group identify the original HDF5 data.
    No objective, resonance estimate or dB conversion is imposed here.
    """

    frequency: object
    s11: object
    valid: object
    impedance: object
    impedance_valid: object
    reference_impedance: float
    tail_relative_db: float
    source: Path
    group: str
    independent_frequency_resolution_hz: float | None = None

    def at(self, frequency):
        """Return complex S11 at a frequency in Hz, using adjacent valid bins.

        Exact saved frequencies use their own mask. Between bins, interpolate
        real/imaginary parts together in the complex plane. Never bridge an
        invalid gap or extrapolate. This does not improve independent frequency
        resolution; magnitude or dB conversion is the caller's choice.
        """
        import numpy as np

        frequency = float(frequency)
        if not np.isfinite(frequency) or not self.frequency[0] <= frequency <= self.frequency[-1]:
            raise ValueError("Requested frequency is outside the saved spectrum")
        index = int(np.searchsorted(self.frequency, frequency))
        if index < len(self.frequency) and self.frequency[index] == frequency:
            if not self.valid[index]:
                raise ValueError("S11 is invalid at the requested frequency")
            return complex(self.s11[index])
        if index == 0 or not (self.valid[index - 1] and self.valid[index]):
            raise ValueError("S11 interpolation requires two adjacent valid bins")
        weight = (frequency - self.frequency[index - 1]) / (
            self.frequency[index] - self.frequency[index - 1]
        )
        return complex((1 - weight) * self.s11[index - 1] + weight * self.s11[index])


def read_port(filename, port, *, grid="/"):
    """Read one terminal's saved S11/Zin and validity.

    port may be an exact ID ("feed", "tl1", "frill1") when unique in the
    chosen grid, or a qualified name ("ports/feed", "tls/tl1", "frills/frill1").
    grid is "/" for the main grid or an explicit subgrid group path. Source-plane
    ratios are not recomputed. Invalid bins are retained with false masks;
    nonfinite values in bins claimed to be valid are rejected as inconsistent.
    This convenience view does not return every voltage/current history;
    those remain available in the original HDF5 output.

    Returns a PortSpectrum with one-dimensional arrays on ``frequency`` (Hz).
    ``s11`` is complex and dimensionless; ``impedance`` is complex ohms.
    ``valid`` selects usable S11 bins; ``impedance_valid`` selects usable Zin.
    Missing groups/datasets raise KeyError. Ambiguous IDs and inconsistent
    arrays raise ValueError. Missing optional decay/resolution diagnostics are
    represented by NaN/None, never fabricated as a successful quality check.

    Example: ``read_port("model.h5", "frills/frill1").at(3.1e9)`` returns a
    complex reflection coefficient. The objective chooses its magnitude/dB.
    """
    import h5py
    import numpy as np

    if not isinstance(port, str) or not port:
        raise ValueError("Use an explicit terminal ID or family/ID")
    families = ("ports", "tls", "frills")
    parts = port.split("/")
    if len(parts) > 2 or not all(parts) or (len(parts) == 2 and parts[0] not in families):
        raise ValueError("Use an ID or ports/ID, tls/ID, frills/ID within the selected grid")
    filename = Path(filename).resolve()
    with h5py.File(filename, "r") as output:
        # 1. Resolve the grid, source family and terminal ID without guessing.
        owner = output[grid]
        if len(parts) == 2:
            group = owner[port]
        else:
            matches = [
                owner[family][port]
                for family in families
                if family in owner and port in owner[family]
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one terminal named {port!r}, found {len(matches)}; use family/ID to disambiguate"
                )
            group = matches[0]
        frequency = np.asarray(group["frequency"], dtype=float)
        if (
            frequency.ndim != 1
            or frequency.size < 2
            or not np.isfinite(frequency).all()
            or np.any(np.diff(frequency) <= 0)
        ):
            raise ValueError("Port frequency axis must be a finite increasing vector")
        # 2. Validate each saved quantity against its own mask and shared axis.
        arrays = {}
        masks = {}
        # Pair each quantity with its own solver-written validity mask. The
        # reader retains invalid bins; an objective must deliberately select data.
        for quantity, mask_name in (("S11", "valid_S11"), ("Zin", "valid_Zin")):
            values, flags = group[quantity][...], group[mask_name][...]
            if (
                values.shape != frequency.shape
                or flags.shape != frequency.shape
                or not np.isin(flags, [0, 1]).all()
            ):
                raise ValueError(f"Invalid shape or validity mask for {quantity}")
            flags = flags.astype(bool)
            if not np.isfinite(values[flags]).all():
                raise ValueError(f"Nonfinite {quantity} in bins marked valid")
            arrays[quantity] = values
            masks[quantity] = flags
        # 3. Attach source normalisation and optional spectral-quality metadata.
        # Voltage/network terminals use ReferenceImpedance; TL and frill
        # writers use their physical characteristic impedance Z0.
        reference_key = (
            "ReferenceImpedance" if group.parent.name.rsplit("/", 1)[-1] == "ports" else "Z0"
        )
        reference = float(group.attrs[reference_key])
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("Port reference impedance must be positive and finite")
        resolution = group.attrs.get("IndependentFrequencyResolution")
        if resolution is not None:
            resolution = float(resolution)
            if not np.isfinite(resolution) or resolution <= 0:
                raise ValueError("Independent frequency resolution must be positive and finite")
        return PortSpectrum(
            frequency=frequency,
            s11=arrays["S11"],
            valid=masks["S11"],
            impedance=arrays["Zin"],
            impedance_valid=masks["Zin"],
            reference_impedance=reference,
            tail_relative_db=float(group.attrs.get("TailRelativeLevelDB", float("nan"))),
            source=filename,
            group=group.name,
            independent_frequency_resolution_hz=resolution,
        )
