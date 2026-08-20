"""
hyperbola.py: two-way travel time for a buried point or cylindrical target.

No marimo dependency, same principle as h5_reader.py, trace_matrix.py and
processing.py: pure numpy, testable with plain pytest, importable from any
dashboard or recipe cell.

Everything here is in metres and nanoseconds. Nanoseconds because that is what
h5_reader.get_time_axis() returns and what the dashboards plot against, so a
predicted curve can be overlaid on a radargram without a unit conversion at the
call site.

What this is for: given a soil permittivity and a target depth, predict where
the reflection hyperbola lands on a B-scan, and inversely, given a measured
apex time and a known depth, recover the permittivity. Both directions are
well posed. Solving for depth AND permittivity together from arrival times
alone is not: they trade off against each other almost exactly, and a fit that
also leaves the source delay free admits permittivities from roughly 4 to 7 on
the same data at sub-picosecond residual. Depth has to come from somewhere
else, which is why it is a required argument everywhere below.
"""

from __future__ import annotations

import numpy as np

# Speed of light in vacuum, m/ns. Working in ns throughout keeps every number
# in this module the same order of magnitude as the plotted time axis.
C_M_PER_NS = 0.299792458


def velocity(eps_r: float) -> float:
    """Wave speed in a non-magnetic medium, m/ns.

    v = c / sqrt(eps_r). Relative permeability is taken as 1, which holds for
    every material in the standard gprMax examples.
    """
    eps_r = float(eps_r)
    if eps_r <= 0:
        raise ValueError(f"eps_r must be positive, got {eps_r}")
    return C_M_PER_NS / np.sqrt(eps_r)


def permittivity(v_m_per_ns: float) -> float:
    """Relative permittivity implied by a wave speed, the inverse of velocity()."""
    v = float(v_m_per_ns)
    if v <= 0:
        raise ValueError(f"velocity must be positive, got {v}")
    return (C_M_PER_NS / v) ** 2


def ricker_delay(freq_hz: float) -> float:
    """Time offset of a gprMax ricker waveform, ns.

    gprMax defines the ricker with chi = sqrt(2) / freq and evaluates it at
    (t - chi), so the pulse peaks chi after the simulation starts rather than
    at t = 0. At 1.5 GHz that is 0.943 ns, which is 31% of the 3.005 ns window
    used by the standard cylinder examples. A predicted arrival that omits it
    lands a third of the record too early.

    The same chi applies to the gaussiandotdot and gaussiandotdotnorm
    waveforms. The plain gaussian family uses 1 / freq instead.

    This cannot be read back from a gprMax .h5 file: the output stores dt,
    iterations, grid and source position, but not the source waveform or its
    centre frequency. It has to come from the input file, or be estimated from
    the peak of the received spectrum.
    """
    freq_hz = float(freq_hz)
    if freq_hz <= 0:
        raise ValueError(f"freq_hz must be positive, got {freq_hz}")
    return np.sqrt(2.0) / freq_hz * 1e9


def travel_time(
    x_src: np.ndarray | float,
    x_target: float,
    depth: float,
    eps_r: float,
    offset: float = 0.0,
    radius: float = 0.0,
    delay_ns: float = 0.0,
) -> np.ndarray:
    """Two-way travel time to a buried target, ns. Accepts an array of source
    positions and returns the hyperbola.

    Bistatic by default: the receiver trails the source by `offset`, which is
    how gprMax B-scans are configured (`#src_steps` and `#rx_steps` move both
    together, so the separation is fixed). Set offset=0 for a monostatic
    survey.

    `depth` is measured from the surface to the target CENTRE. For a cylinder,
    `radius` shortens each ray to the nearest point on its surface, which is
    where the reflection actually comes from. Leaving radius at 0 treats the
    target as a point scatterer and predicts an arrival about 0.08 ns late for
    the 10 mm cylinder in the standard example.

    `delay_ns` is the source waveform offset, from ricker_delay().
    """
    xs = np.asarray(x_src, dtype=float)
    depth = float(depth)
    radius = float(radius)

    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    if radius < 0:
        raise ValueError(f"radius cannot be negative, got {radius}")
    if radius >= np.hypot(0.0, depth):
        raise ValueError(f"radius {radius} reaches the surface at depth {depth}")

    v = velocity(eps_r)
    to_src = np.hypot(x_target - xs, depth)
    to_rx = np.hypot(x_target - (xs + float(offset)), depth)

    return (to_src + to_rx - 2.0 * radius) / v + float(delay_ns)


def apex_source_x(x_target: float, offset: float = 0.0) -> float:
    """Source x at which the antenna midpoint sits over the target.

    With a bistatic pair the hyperbola turns over when the midpoint of the
    source and receiver crosses the target, which is half an offset before the
    source itself reaches it. Reading the apex off the source axis without this
    correction misplaces the target by offset/2.
    """
    return float(x_target) - float(offset) / 2.0


def apex_time(
    depth: float,
    eps_r: float,
    offset: float = 0.0,
    radius: float = 0.0,
    delay_ns: float = 0.0,
) -> float:
    """Travel time at the apex, ns. Independent of where the target sits
    laterally, so x_target is not needed."""
    return float(
        travel_time(
            apex_source_x(0.0, offset), 0.0, depth, eps_r, offset, radius, delay_ns
        )
    )


def permittivity_from_apex(
    t_apex_ns: float,
    depth: float,
    offset: float = 0.0,
    radius: float = 0.0,
    delay_ns: float = 0.0,
) -> tuple[float, float]:
    """Recover (eps_r, velocity) from a measured apex time and a known depth.

    This is the calculator direction: pick the apex off a radargram, supply the
    depth, and read back the soil permittivity. Well posed only because depth
    is given. Inverting for depth and permittivity together is degenerate.

    Raises ValueError if the apex arrives too early to be physical, which means
    faster than light in vacuum for that depth once the source delay is
    accounted for.
    """
    depth = float(depth)
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")

    path = 2.0 * np.hypot(float(offset) / 2.0, depth) - 2.0 * float(radius)
    elapsed = float(t_apex_ns) - float(delay_ns)

    if elapsed <= 0:
        raise ValueError(
            f"apex at {t_apex_ns} ns arrives before the source delay of {delay_ns} ns"
        )

    v = path / elapsed
    if v > C_M_PER_NS:
        raise ValueError(
            f"apex at {t_apex_ns} ns implies {v:.4f} m/ns over a {path:.4f} m path, "
            f"faster than light. Check the depth and the source delay."
        )

    return permittivity(v), v
