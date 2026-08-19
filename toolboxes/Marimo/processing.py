"""
processing.py: amplitude processing for gprMax traces and B-scan matrices.

No marimo dependency, same principle as h5_reader.py and trace_matrix.py:
pure numpy, testable with plain pytest, importable from any dashboard cell.

Every function here works on a single A-scan trace, shape (n_samples,), and
on an assembled B-scan matrix, shape (n_samples, n_traces) — the orientation
trace_matrix.stack_traces() produces. One implementation, not two, so a gain
applied to a trace in ascan_dashboard.py and the same gain applied across a
radargram in bscan_dashboard.py cannot drift apart.

Gain functions return the applied curve alongside the gained data. A gained
radargram without its curve is uninterpretable — the researcher has to be
able to see what was done to the amplitudes, not just the result.
"""

from __future__ import annotations

import numpy as np

# Gain kind -> unit of its `factor` argument, for labels and slider captions.
# Keys are the accepted `kind` values; build a dashboard dropdown from these
# rather than hardcoding a second list somewhere.
GAIN_KINDS: dict[str, str] = {
    "none": "",
    "constant": "x",
    "linear": "per ns",
    "exponential": "per ns",
    "db": "dB per ns",
}


def gain_curve(
    time_ns: np.ndarray,
    kind: str,
    factor: float = 1.0,
    start_ns: float = 0.0,
    max_gain: float | None = None,
) -> np.ndarray:
    """Build the time-varying amplification curve, shape (n_samples,).

    `factor` means something different per kind, hence GAIN_KINDS:
        none         ignored, curve is all ones
        constant     flat multiplier applied from start_ns onward
        linear       1 + factor * t, t in ns since start_ns
        exponential  exp(factor * t)
        db           10 ** (factor * t / 20), factor in dB per ns

    Time before `start_ns` gets a gain of exactly 1.0 so the direct wave can
    be left alone — amplifying it along with everything else just saturates
    the colour scale on a B-scan and hides the reflections underneath.

    `max_gain` clamps the curve. Exponential gain over a few ns reaches
    absurd numbers quickly (exp(5 * 3) is 3.3e6), so this is a real guard,
    not defensive padding.
    """
    t = np.asarray(time_ns, dtype=float)
    if t.ndim != 1:
        raise ValueError(f"time_ns must be 1D, got shape {t.shape}")

    kind = kind.lower()
    if kind not in GAIN_KINDS:
        raise ValueError(f"Unknown gain kind '{kind}'. Options: {list(GAIN_KINDS)}")

    factor = float(factor)
    elapsed = np.clip(t - float(start_ns), 0.0, None)

    if kind == "none":
        curve = np.ones_like(t)
    elif kind == "constant":
        curve = np.where(t >= float(start_ns), factor, 1.0)
    elif kind == "linear":
        curve = 1.0 + factor * elapsed
    elif kind == "exponential":
        curve = np.exp(factor * elapsed)
    else:  # db
        curve = 10.0 ** (factor * elapsed / 20.0)

    # Gain is never negative. A negative factor on linear or constant would
    # otherwise cross zero and flip trace polarity partway down, which reads
    # as a physical reflection that isn't there. Polarity inversion is a
    # separate control if it's ever wanted, not a side effect of gain.
    curve = np.clip(curve, 0.0, None)

    if max_gain is not None:
        curve = np.minimum(curve, float(max_gain))

    return curve


def apply_gain(
    data: np.ndarray,
    time_ns: np.ndarray,
    kind: str,
    factor: float = 1.0,
    start_ns: float = 0.0,
    max_gain: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a gain curve down the time axis. Returns (gained, curve).

    `data` is either one trace, shape (n_samples,), or a stacked B-scan
    matrix, shape (n_samples, n_traces). Axis 0 is time in both cases, so
    every trace in a matrix gets the identical curve.

    The returned curve is always 1D and always the full sample length, ready
    to plot on a second y-axis next to the gained trace.
    """
    arr = np.asarray(data, dtype=float)
    t = np.asarray(time_ns, dtype=float)

    if arr.ndim not in (1, 2):
        raise ValueError(f"data must be 1D or 2D, got shape {arr.shape}")
    if arr.shape[0] != t.shape[0]:
        raise ValueError(
            f"time axis has {t.shape[0]} samples, data has {arr.shape[0]} "
            f"along axis 0 (data shape {arr.shape}) — a B-scan matrix must be "
            f"(n_samples, n_traces)"
        )

    curve = gain_curve(t, kind, factor, start_ns, max_gain)
    gained = arr * curve if arr.ndim == 1 else arr * curve[:, None]

    return gained, curve


def gain_label(
    kind: str,
    factor: float = 1.0,
    start_ns: float = 0.0,
    max_gain: float | None = None,
) -> str:
    """One-line description of a gain setting, for plot titles and CSV headers.

    CSV export stays raw, so the header comment carrying this string is the
    only record of what the plot the user was looking at actually showed.
    """
    kind = kind.lower()
    if kind not in GAIN_KINDS:
        raise ValueError(f"Unknown gain kind '{kind}'. Options: {list(GAIN_KINDS)}")

    if kind == "none":
        return "no gain"

    if kind == "constant":
        text = f"constant gain x{factor:g}"
    else:
        text = f"{kind} gain {factor:g} {GAIN_KINDS[kind]}"

    if start_ns:
        text += f" from {start_ns:g} ns"
    if max_gain is not None:
        text += f", clamped at x{max_gain:g}"

    return text


def remove_mean_trace(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the mean trace from every column. Returns (result, mean_trace).

    The real-world background removal: with the antenna stepping across a
    surface, the direct wave and any flat-layer reflection are identical in
    every trace, so the across-trace mean is almost entirely that stationary
    signal. Subtracting it leaves the target response.

    This is the equivalent of target-minus-free-space subtraction for the
    common case where no separate free-space simulation exists.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2D (n_samples, n_traces), got shape {arr.shape}")

    mean_trace = arr.mean(axis=1)
    return arr - mean_trace[:, None], mean_trace
