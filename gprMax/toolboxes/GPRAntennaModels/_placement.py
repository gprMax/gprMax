# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Rigid placement and quarter-turn rotation of antenna models on the grid.

An antenna model is built once with its reference point (the geometric centre
in x-y and the bottom of the skid in z) at the origin. Every coordinate of that
local model is then converted to whole cells with exact arithmetic and moved to
the placement point, which is rounded to the grid once. Because each part is
"placement + a fixed number of cells", the antenna has the same shape wherever
it is placed. Rounding each translated coordinate separately instead lets
floating-point noise decide half-cell ties, so parts can grow, shrink or move
by one cell depending on the placement.

Quarter turns about +z are exact integer permutations of the cell offsets.
"""

import math
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np

import gprMax
from gprMax.utilities.utilities import round_int

ROTATIONS = (0, 90, 180, 270)

# Objects whose two corner points must be ordered p1 <= p2 on every axis.
_ORDERED_CORNERS = (gprMax.Box, gprMax.Plate, gprMax.Edge)
# Objects whose points are moved as they are.
_FREE_POINTS = (gprMax.Cylinder, gprMax.Triangle)
# Point objects tied to one Yee edge through their polarisation.
_POLARISED = (
    gprMax.VoltageSource,
    gprMax.HertzianDipole,
    gprMax.MagneticDipole,
    gprMax.TransmissionLine,
)
# Objects without coordinates.
_NO_POINTS = (gprMax.Material, gprMax.Waveform, gprMax.ExcitationFile)

# Local metres are converted to an exact integer number of micrometres first.
_MICRONS = 10**6


def check_rotation(rotation, name):
    """Raise unless ``rotation`` is 0, 90, 180 or 270 degrees."""
    if isinstance(rotation, bool) or rotation not in ROTATIONS:
        raise ValueError(f"{name} rotation must be 0, 90, 180 or 270 degrees")
    return int(rotation)


def snap_placement(x, y, z, resolution, name):
    """Round the placement point to the grid once and return it in cells.

    Off-grid placements are rounded with gprMax's normal rule and emit a
    warning; the antenna dimensions are preserved.
    """
    placement = (float(x), float(y), float(z))
    if not all(math.isfinite(value) for value in placement):
        raise ValueError(f"{name} placement coordinates must be finite")
    anchor = tuple(round_int(value / resolution) for value in placement)
    snapped = tuple(value * resolution for value in anchor)
    if any(
        abs(value - actual) > 8 * math.ulp(max(abs(value), abs(actual), resolution))
        for value, actual in zip(placement, snapped)
    ):
        warnings.warn(
            f"{name} placement {placement} rounded to {snapped} m "
            f"on the {resolution * 1e3:g} mm grid; antenna dimensions are preserved.",
            UserWarning,
            stacklevel=3,
        )
    return anchor


def _local_cells(point, resolution):
    """Exact cell offsets of a local point; half-cell ties round down."""
    step = round(resolution * _MICRONS)
    cells = []
    for value in point:
        microns = round(value * _MICRONS)
        if abs(value * _MICRONS - microns) > 1e-3:
            raise ValueError(
                f"Antenna model coordinate {value!r} is not a whole number of micrometres"
            )
        # Nearest cell with exact ties going towards -infinity, so a shape keeps
        # its size on both sides of the reference point.
        cells.append(math.ceil(Fraction(microns, step) - Fraction(1, 2)))
    return cells


def _rotate(cells, quarter_turns):
    i, j, k = cells
    for _ in range(quarter_turns):
        i, j = -j, i
    return i, j, k


def _world(point, anchor, quarter_turns, resolution):
    offset = _rotate(_local_cells(point, resolution), quarter_turns)
    return tuple((origin + step) * resolution for origin, step in zip(anchor, offset))


def _edge(point, axis, anchor, quarter_turns, resolution):
    """Move the positive-axis Yee edge starting at ``point``.

    Returns the lower endpoint of the moved edge, its axis, and +1 or -1 for
    whether the original positive direction now points along the positive axis.
    """
    start = _rotate(_local_cells(point, resolution), quarter_turns)
    unit = [0, 0, 0]
    unit["xyz".index(axis)] = 1
    direction = _rotate(unit, quarter_turns)
    new_axis = "xyz"[next(index for index, step in enumerate(direction) if step)]
    sign = sum(direction)
    lower = tuple(s + min(d, 0) for s, d in zip(start, direction))
    return (
        tuple((origin + step) * resolution for origin, step in zip(anchor, lower)),
        new_axis,
        sign,
    )


def place_antenna(objects, anchor, rotation, resolution):
    """Move antenna objects built about the origin to ``anchor`` (in cells).

    Args:
        objects: user objects of an antenna model built with its reference
            point at the origin.
        anchor: placement point in cells, from :func:`snap_placement`.
        rotation: 0, 90, 180 or 270 degrees counterclockwise about +z.
        resolution: spatial resolution in metres.

    Returns:
        New user objects at the placement point. A source whose feed edge is
        reversed by the rotation has the amplitude of its waveform negated, so
        the antenna is driven with its original local polarity.
    """
    quarter_turns = rotation // 90
    placed = []
    waveform_signs = {}
    for obj in objects:
        kwargs = dict(obj.kwargs)
        if isinstance(obj, _NO_POINTS):
            placed.append(obj)
            continue
        if isinstance(obj, _ORDERED_CORNERS):
            q1 = _world(kwargs["p1"], anchor, quarter_turns, resolution)
            q2 = _world(kwargs["p2"], anchor, quarter_turns, resolution)
            kwargs["p1"] = tuple(min(a, b) for a, b in zip(q1, q2))
            kwargs["p2"] = tuple(max(a, b) for a, b in zip(q1, q2))
        elif isinstance(obj, _FREE_POINTS):
            for key in ("p1", "p2", "p3"):
                if key in kwargs:
                    kwargs[key] = _world(kwargs[key], anchor, quarter_turns, resolution)
        elif isinstance(obj, _POLARISED):
            kwargs["p1"], kwargs["polarisation"], sign = _edge(
                kwargs["p1"], kwargs["polarisation"], anchor, quarter_turns, resolution
            )
            if "waveform_id" in kwargs:
                previous = waveform_signs.setdefault(kwargs["waveform_id"], sign)
                if previous != sign:
                    raise ValueError(
                        f"Waveform {kwargs['waveform_id']!r} drives feeds of opposite polarity"
                    )
        elif isinstance(obj, gprMax.Rx):
            outputs = kwargs.get("outputs")
            axes = {name[1] for name in outputs or () if len(name) == 2 and name[0] in "EHI"}
            if quarter_turns and (
                len(axes) != 1 or len(axes) != len({name[0] for name in outputs})
            ):
                raise ValueError("Rotated antenna receivers must record one E, H or I component")
            if axes:
                kwargs["p1"], axis, _ = _edge(
                    kwargs["p1"], axes.pop(), anchor, quarter_turns, resolution
                )
                kwargs["outputs"] = [name[0] + axis for name in outputs]
            else:
                kwargs["p1"] = _world(kwargs["p1"], anchor, quarter_turns, resolution)
        else:
            raise TypeError(f"Antenna placement does not support {type(obj).__name__} objects")
        placed.append(type(obj)(**kwargs))

    reversed_ids = {waveform_id for waveform_id, sign in waveform_signs.items() if sign < 0}
    if reversed_ids:
        placed = _reverse_waveforms(placed, reversed_ids)
    return placed


def _reverse_waveforms(objects, reversed_ids):
    """Negate the waveforms with IDs in ``reversed_ids``.

    A waveform read from an excitation file cannot be scaled, so the file is
    read here and its columns are passed on as user waveforms with the same
    IDs, interpolation kind and fill value; the reversed columns are negated.
    """
    result = []
    found = set()
    for obj in objects:
        if isinstance(obj, gprMax.Waveform) and obj.kwargs.get("id") in reversed_ids:
            kwargs = dict(obj.kwargs)
            if kwargs.get("user_values") is not None:
                kwargs["user_values"] = -np.asarray(kwargs["user_values"], dtype=float)
            elif "user_func" in kwargs:
                user_func = kwargs["user_func"]
                kwargs["user_func"] = lambda t, f=user_func: -f(t)
            else:
                kwargs["amp"] = -kwargs["amp"]
            found.add(kwargs["id"])
            result.append(gprMax.Waveform(**kwargs))
        elif isinstance(obj, gprMax.ExcitationFile):
            ids = np.loadtxt(Path(obj.filepath), max_rows=1, dtype=str, ndmin=1)
            if not reversed_ids.intersection(ids):
                result.append(obj)
                continue
            if ids[0].lower() != "time":
                raise ValueError(
                    f"Excitation file {obj.filepath} needs a time column to drive a rotated feed"
                )
            values = np.loadtxt(Path(obj.filepath), skiprows=1, dtype=float, ndmin=2)
            for column, waveform_id in enumerate(ids[1:], start=1):
                sign = -1 if waveform_id in reversed_ids else 1
                found.add(str(waveform_id))
                result.append(
                    gprMax.Waveform(
                        wave_type="user",
                        user_values=sign * values[:, column],
                        user_time=values[:, 0],
                        kind=obj.kind,
                        fill_value=obj.fill_value,
                        id=str(waveform_id),
                    )
                )
        else:
            result.append(obj)
    missing = reversed_ids - found
    if missing:
        raise ValueError(f"Antenna waveform(s) {sorted(missing)} are not part of the model")
    return result
