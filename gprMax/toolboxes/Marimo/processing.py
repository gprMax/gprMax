"""
processing.py: amplitude processing for gprMax traces and B-scan matrices.

No marimo dependency, same principle as h5_reader.py and trace_matrix.py:
pure numpy, testable with plain pytest, importable from any dashboard cell.

Every function works on a single A-scan trace, shape (n_samples,), and on an
assembled B-scan matrix, shape (n_samples, n_traces) — the orientation
trace_matrix.stack_traces() produces. One implementation, not two, so a gain
applied to a trace in ascan_dashboard.py and the same gain applied across a
radargram in bscan_dashboard.py cannot drift apart.

Gain functions return the applied curve alongside the gained data. A gained
radargram without its curve is uninterpretable — the researcher has to see
what was done to the amplitudes, not just the result.

Naming follows the GPR processing literature rather than inventing new terms:
power gain t**b, exponential gain exp(a*t), and their product SEC (Spreading
and Exponential Compensation), which is the gain most published GPR flows
actually use. AGC is deliberately absent — it equalises amplitudes by local
window and destroys relative reflectivity, which is the one thing an FDTD
simulation gets exactly right and a field instrument does not.

Standard processing order, which the dashboards follow:
    background removal -> gain -> filtering -> display normalisation
"""

from __future__ import annotations

import numpy as np

# Gain kind -> UI metadata. Build the dashboard dropdown and decide which
# sliders to show from this, rather than hardcoding a second list elsewhere.
#   factor_unit  unit of the `factor` argument, None if unused
#   uses_power   whether the `power` argument applies
GAIN_KINDS: dict[str, dict] = {
    "none": {"label": "None", "factor_unit": None, "uses_power": False},
    "constant": {"label": "Constant", "factor_unit": "x", "uses_power": False},
    "linear": {"label": "Linear", "factor_unit": "per ns", "uses_power": False},
    "power": {"label": "Power (t^b)", "factor_unit": None, "uses_power": True},
    "exponential": {"label": "Exponential", "factor_unit": "per ns", "uses_power": False},
    "db": {"label": "dB", "factor_unit": "dB per ns", "uses_power": False},
    "sec": {"label": "SEC (spreading + exponential)", "factor_unit": "per ns", "uses_power": True},
}


def gain_curve(
    time_ns: np.ndarray,
    kind: str,
    factor: float = 1.0,
    power: float = 1.0,
    start_ns: float = 0.0,
    max_gain: float | None = None,
) -> np.ndarray:
    """Build the time-varying amplification curve, shape (n_samples,).

    With t the elapsed time in ns since `start_ns`:

        none         1
        constant     factor
        linear       1 + factor * t
        power        t ** power              spherical spreading compensation
        exponential  exp(factor * t)         ohmic attenuation compensation
        db           10 ** (factor * t / 20) same family, amplitude decibels
        sec          exp(factor * t) * t ** power

    SEC is the composition of the two physical corrections and is the gain
    most published GPR processing flows use. Setting power=1 reduces its
    spreading term to the textbook linear gain.

    Time before `start_ns` gets a gain of exactly 1.0 so the direct wave can
    be left alone. Amplifying it along with everything else saturates the
    colour scale on a B-scan and buries the reflections underneath.

    `max_gain` bounds the curve, which is standard practice — exponential
    gain over a few ns reaches absurd numbers quickly (exp(5 * 3) is 3.3e6).

    The power term is floored at 1.0 rather than evaluated as a bare t**b.
    Bare t**b is zero at t=0 and below 1 for all t<1 ns, so on a 3 ns gprMax
    window it would blank the first sample and attenuate the first third of
    the trace — the opposite of what a gain is for. Flooring keeps t**b exact
    everywhere it amplifies and holds at unity everywhere it would not.
    """
    t = np.asarray(time_ns, dtype=float)
    if t.ndim != 1:
        raise ValueError(f"time_ns must be 1D, got shape {t.shape}")

    kind = kind.lower()
    if kind not in GAIN_KINDS:
        raise ValueError(f"Unknown gain kind '{kind}'. Options: {list(GAIN_KINDS)}")

    factor = float(factor)
    power = float(power)
    elapsed = np.clip(t - float(start_ns), 0.0, None)

    if kind == "none":
        curve = np.ones_like(t)
    elif kind == "constant":
        curve = np.where(t >= float(start_ns), factor, 1.0)
    elif kind == "linear":
        curve = 1.0 + factor * elapsed
    elif kind == "power":
        curve = _power_term(elapsed, power)
    elif kind == "exponential":
        curve = np.exp(factor * elapsed)
    elif kind == "db":
        curve = 10.0 ** (factor * elapsed / 20.0)
    else:  # sec
        curve = np.exp(factor * elapsed) * _power_term(elapsed, power)

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
    power: float = 1.0,
    start_ns: float = 0.0,
    max_gain: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a gain curve down the time axis. Returns (gained, curve).

    `data` is either one trace, shape (n_samples,), or a stacked B-scan
    matrix, shape (n_samples, n_traces). Axis 0 is time in both cases, so
    every trace in a matrix gets the identical curve.

    The returned curve is always 1D and always the full sample length, ready
    to plot against the same time axis as the trace it was applied to.
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

    curve = gain_curve(t, kind, factor, power, start_ns, max_gain)
    gained = arr * curve if arr.ndim == 1 else arr * curve[:, None]

    return gained, curve


def gain_label(
    kind: str,
    factor: float = 1.0,
    power: float = 1.0,
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

    unit = GAIN_KINDS[kind]["factor_unit"]

    if kind == "constant":
        text = f"constant gain x{factor:g}"
    elif kind == "power":
        text = f"power gain t^{power:g}"
    elif kind == "sec":
        text = f"SEC gain a={factor:g} {unit}, b={power:g}"
    elif kind == "db":
        text = f"dB gain {factor:g} {unit}"
    else:
        text = f"{kind} gain {factor:g} {unit}"

    if start_ns:
        text += f" from {start_ns:g} ns"
    if max_gain is not None:
        text += f", clamped at x{max_gain:g}"

    return text


def remove_mean_trace(
    matrix: np.ndarray,
    window: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the average trace from every column. Returns (result, background).

    With the antenna stepping across a surface, the direct wave and any
    flat-layer reflection are identical in every trace, so the across-trace
    mean is almost entirely that stationary signal. Subtracting it leaves the
    target response. This is analogous to target-minus-background-reference
    subtraction when no separate target-free background run exists.

    `window` selects between the two conventions in the literature:
        None      mean over all traces. Appropriate when the direct wave and
                  layered background response are stationary over the scan.
        int       mean over a moving window of that many traces, the
                  convention for real survey lines where the background
                  drifts along the profile.

    The returned background is the full (n_samples, n_traces) array that was
    subtracted when a window is used, and the single (n_samples,) mean trace
    when it is not — in both cases exactly what was removed, so it can be
    displayed rather than taken on trust.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2D (n_samples, n_traces), got shape {arr.shape}")

    if window is None:
        background = arr.mean(axis=1)
        return arr - background[:, None], background

    window = int(window)
    if window < 1:
        raise ValueError(f"window must be at least 1 trace, got {window}")

    background = _moving_mean(arr, window)
    return arr - background, background


# Private helpers


def _power_term(elapsed: np.ndarray, power: float) -> np.ndarray:
    """t**power, floored at 1.0. See gain_curve for why the floor is there."""
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.power(elapsed, power)
    term = np.nan_to_num(term, nan=1.0, posinf=np.inf, neginf=1.0)
    return np.maximum(term, 1.0)


def _moving_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Per-column mean over a moving window of columns, same shape as `arr`.

    Windows are clamped at the profile ends rather than zero-padded, so edge
    traces average over a full-width window of their nearest neighbours
    instead of being biased toward zero.
    """
    n_traces = arr.shape[1]
    width = min(window, n_traces)
    half = width // 2

    cumsum = np.concatenate([np.zeros((arr.shape[0], 1)), np.cumsum(arr, axis=1)], axis=1)

    out = np.empty_like(arr)
    for j in range(n_traces):
        hi = min(n_traces, max(j - half, 0) + width)
        lo = max(0, hi - width)
        out[:, j] = (cumsum[:, hi] - cumsum[:, lo]) / (hi - lo)

    return out


# Frequency domain


def fft_spectrum(
    waveform: np.ndarray,
    dt: float,
    positive_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Power spectrum of one trace. Returns (freqs_hz, power_db, peak_db).

    The transform itself is gprMax's own `fft_power`, imported rather than
    reimplemented so this stays consistent with `plot_Ascan.py`. Three things
    about that function need handling at the call site:

    1. It normalises each trace against its own maximum, so every spectrum
       peaks at 0 dB no matter its absolute amplitude. Overlay a 1000 V/m
       trace with a 3 A/m one and they look identical. `peak_db` is that
       trace's absolute peak power in dB; adding it to `power_db` recovers
       absolute values, which is what makes overlaid spectra comparable.
    2. It returns the whole `fftfreq`, negative frequencies included, which
       plots as a mirrored spectrum. `positive_only` slices to the first
       half.
    3. An all-zero trace has no finite normalised spectrum: ``fft_power``
       returns ``-inf``. Ex in a 2D TMz model is exactly this case. ``peak_db``
       is None for such a trace so the caller can report it instead of drawing
       a meaningless spectrum.

    The import is deferred so that importing this module, and testing every
    other function in it, does not require a built gprMax.
    """
    arr = np.asarray(waveform, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"waveform must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("waveform is empty")

    from gprMax.utilities.utilities import fft_power

    if not np.any(arr):
        freqs = np.fft.fftfreq(arr.size, d=dt)
        power = np.zeros(arr.size, dtype=float)
        peak_db = None
    else:
        freqs, power = fft_power(arr, dt)
        peak_db = float(10.0 * np.log10(np.max(np.abs(np.fft.fft(arr)) ** 2)))

    if positive_only:
        half = (arr.size + 1) // 2
        freqs, power = freqs[:half], power[:half]

    return freqs, power, peak_db


def spectrum_view_limit(
    freqs: np.ndarray,
    power_db: np.ndarray,
    multiple: float = 4.0,
) -> float:
    """Upper frequency for a sensible default spectrum view, in Hz.

    With dt around 4.7 ps the Nyquist frequency is above 100 GHz, while a
    1.5 GHz antenna puts all of its energy in the first couple of percent of
    that axis. Plotting to Nyquist shows a spike against a flat floor.

    `plot_Ascan.py` solves this with `freqmaxpower * 4` as an array index,
    which runs past the end of the array whenever the peak sits beyond a
    quarter of the spectrum. Working in frequency and clamping to the
    available maximum cannot overshoot.
    """
    if freqs.size == 0 or power_db.size == 0:
        return 0.0

    peak_hz = float(freqs[int(np.argmax(power_db))])
    limit = multiple * peak_hz
    available = float(np.max(freqs))

    if limit <= 0.0 or limit > available:
        return available
    return limit


def subtract_traces(
    target: np.ndarray,
    reference: np.ndarray,
    dt_target: float | None = None,
    dt_reference: float | None = None,
    rtol: float = 1e-6,
) -> np.ndarray:
    """Elementwise `target - reference`, for isolating a buried target's response.

    Run the model twice, once with the target and once without, and subtract.
    What cancels is everything the two runs share: the source pulse, the direct
    coupling between antennas, and any reflection from a flat interface. What
    survives is the target.

    Works on a single trace, shape (n_samples,), and on two B-scan matrices of
    the same shape.

    Both time steps should be passed whenever they are known. Checking the
    sample count alone is not enough: two runs can produce the same number of
    samples from different `#time_window` and `#dx_dy_dz` settings, in which
    case the arrays line up index by index while representing different
    instants, and the subtraction silently returns nonsense. Comparing dt
    catches that.
    """
    a = np.asarray(target, dtype=float)
    b = np.asarray(reference, dtype=float)

    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: target {a.shape}, reference {b.shape}. "
            "Both runs must have the same iteration count."
        )

    if dt_target is not None and dt_reference is not None:
        if not np.isclose(float(dt_target), float(dt_reference), rtol=rtol, atol=0.0):
            raise ValueError(
                f"time steps differ: target dt={float(dt_target):.6e} s, "
                f"reference dt={float(dt_reference):.6e} s. The arrays are the same "
                "length but their samples are at different instants, so subtracting "
                "them index by index is meaningless."
            )

    return a - b
