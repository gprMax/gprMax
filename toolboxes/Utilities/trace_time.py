"""Physical time-domain histories shared by plotting and trace exporters.

Raw source buffers are not always physical output histories: magnetic frills
and transmission lines have an extra allocation endpoint, and currents need not lie
on the receiver H/current lattice. Do not infer source timing from 'I' alone.
"""

from dataclasses import dataclass

import h5py
import numpy as np


@dataclass(frozen=True)
class TimeHistory:
    samples: np.ndarray
    dt: float
    offset: float

    @property
    def time(self):
        return self.offset + np.arange(self.samples.shape[0]) * self.dt


def receiver_time_offset(component, dt):
    """Legacy receiver fallback; explicit dataset metadata takes precedence."""
    return -0.5 * dt if component.startswith(("H", "I")) else 0.0


def _nearest_attribute(group, name):
    while True:
        if name in group.attrs:
            return group.attrs[name]
        if group.name == "/":
            return None
        group = group.parent


def read_time_history(dataset, *, allow_matrix=False):
    """Read a real history with validated sampling and source-specific length.

    Dataset metadata is authoritative; native terminal time vectors and group
    offsets are checked for consistency. Missing interval metadata falls back
    to the nearest owning grid, not unconditionally the root grid. Only native
    frill/TL buffers may have their extra endpoint trimmed. Merged matrices
    already contain physical samples and are never trimmed.
    """
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError("A time-domain HDF5 dataset is required")
    values = np.asarray(dataset)
    if values.ndim not in ((1, 2) if allow_matrix else (1,)) or values.shape[0] == 0:
        raise ValueError(
            f"{dataset.name} must be a nonempty {'one- or two-' if allow_matrix else 'one-'}dimensional history"
        )
    if np.iscomplexobj(values):
        raise ValueError(f"Complex-valued time-domain trace data are unsupported: {dataset.name}")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{dataset.name} must contain real numeric time-domain samples")
    group = dataset.parent
    family = group.parent.name.rsplit("/", 1)[-1]
    component = dataset.name.rsplit("/", 1)[-1]
    interval = dataset.attrs.get("SampleInterval", _nearest_attribute(group, "dt"))
    if interval is None or not np.isfinite(float(interval)) or float(interval) <= 0:
        raise ValueError(f"Invalid or missing sample interval for {dataset.name}")
    dt = float(interval)
    offset = None
    time_name = None
    if family == "tls":
        current = component.startswith("I")
        time_name = "time_current" if current else "time_voltage"
        offset = group.attrs.get("TimeCurrentOffset" if current else "TimeVoltageOffset")
        fallback = -0.5 * dt if current else 0.0
    elif family == "frills":
        time_name = "time"
        offset = group.attrs.get("TimeOffset")
        fallback = 0.0
    elif family == "ports":
        current = component.startswith("I")
        time_name = "time_current" if current and "time_current" in group else "time"
        offset = group.attrs.get("CurrentTimeSampleOffset") if current else None
        if offset is None:
            offset = group.attrs.get("TimeSampleOffset")
        fallback = 0.0
    else:
        fallback = receiver_time_offset(component, dt)
    if family in ("tls", "frills") and values.ndim == 1:
        iterations = _nearest_attribute(group, "Iterations")
        if iterations is None and time_name in group:
            iterations = group[time_name].shape[0]
        if iterations is not None:
            count = int(iterations)
            if count <= 0 or count != iterations or values.shape[0] not in (count, count + 1):
                kind = "frill" if family == "frills" else "transmission-line"
                raise ValueError(f"Invalid physical {kind} history length for {dataset.name}")
            values = values[:count]
    offset = dataset.attrs.get("TimeSampleOffset", offset)
    if time_name is not None and time_name in group:
        axis = np.asarray(group[time_name])
        if (
            axis.shape != (values.shape[0],)
            or not np.issubdtype(axis.dtype, np.number)
            or np.iscomplexobj(axis)
            or not np.all(np.isfinite(axis))
        ):
            raise ValueError(f"Invalid time axis {group.name}/{time_name} for {dataset.name}")
        if offset is None:
            offset = float(axis[0])
        expected = float(offset) + np.arange(values.shape[0]) * dt
        if not np.allclose(axis, expected, rtol=1e-6, atol=32 * np.finfo(float).eps * dt):
            raise ValueError(f"Time axis and sampling metadata disagree for {dataset.name}")
    offset = float(fallback if offset is None else offset)
    if not np.isfinite(offset):
        raise ValueError(f"Invalid sample-zero time offset for {dataset.name}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Trace data contain NaN or infinite values: {dataset.name}")
    return TimeHistory(values, dt, offset)
