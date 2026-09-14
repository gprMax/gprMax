"""Identity- and time-aware background subtraction for loaded A-scans."""

import numpy as np

from gprMax.toolboxes.Marimo.h5_reader import get_time_axis, get_trace
from gprMax.toolboxes.Marimo.processing import subtract_traces
from gprMax.toolboxes.Utilities.receiver_identity import match_receiver


def subtract_receiver_reference(samples, target, background, *, receiver, component):
    """Subtract the same declared receiver, not a coincident file-local rxN.

    Loaded files carry immutable receiver identities. Refuse missing/ambiguous
    identities or incompatible physical times instead of guessing a receiver.
    The dashboard catches these errors and reports an unapplied subtraction.
    """
    reference = target["receivers"][receiver]["identity"]
    catalogue = {f"rxs/{key}": info["identity"] for key, info in background["receivers"].items()}
    matched = match_receiver(reference, catalogue).path.split("/")[-1]
    a_time = get_time_axis(target, unit="s", receiver=receiver, component=component)
    b_time = get_time_axis(background, unit="s", receiver=matched, component=component)
    # Avoid an absolute tolerance in seconds that would hide differences for
    # very fine grids. Compare relative to the actual physical sample spacing.
    info = target["receivers"][receiver].get("component_meta", {}).get(component, {})
    dt = float(info.get("sample_interval", target["meta"]["dt"]))
    if (
        a_time.shape != b_time.shape
        or not np.all(np.isfinite(a_time))
        or not np.all(np.isfinite(b_time))
        or not np.isfinite(dt)
        or dt <= 0
        or not np.allclose(a_time, b_time, rtol=1e-9, atol=32 * np.finfo(float).eps * dt)
    ):
        raise ValueError("target and background sample times differ")
    return subtract_traces(samples, get_trace(background, component, matched))
