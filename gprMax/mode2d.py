# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.

"""Shared geometry of the reduced 2-D Yee modes.

The reduced solvers retain the three-dimensional Yee arrays.  A TM model is
one cell thick on its invariant axis and its live field layer is index zero.
A TE model needs two cells so that the tangential electric fields and the
normal magnetic field share the interior plane at index one.  The extra TE
cell is storage topology, not a second physical 2-D sampling plane.
"""

from __future__ import annotations

from dataclasses import dataclass

AXES = "xyz"
ELECTRIC_COMPONENTS = ("Ex", "Ey", "Ez")
MAGNETIC_COMPONENTS = ("Hx", "Hy", "Hz")


@dataclass(frozen=True)
class Mode2DGeometry:
    """Live-plane and field-component description of one reduced mode."""

    mode: str
    polarisation: str
    invariant_axis: int
    live_index: int
    active_electric: tuple[str, ...]
    active_magnetic: tuple[str, ...]

    @property
    def invariant_axis_name(self) -> str:
        return AXES[self.invariant_axis]

    @property
    def collocation_strides(self) -> tuple[int, int, int]:
        """Yee-neighbour strides used to collocate fields at the live plane."""

        strides = [1, 1, 1]
        if self.polarisation == "TE":
            strides[self.invariant_axis] = 0
        return tuple(strides)


def mode2d_geometry(mode: str) -> Mode2DGeometry | None:
    """Return reduced-mode geometry, or ``None`` for a three-dimensional mode."""

    if not mode.startswith("2D"):
        return None
    parts = mode.split()
    if len(parts) != 2 or len(parts[1]) != 3:
        raise ValueError(f"invalid 2-D model mode {mode!r}")
    family = parts[1][:2]
    axis_name = parts[1][2].lower()
    if family not in ("TM", "TE") or axis_name not in AXES:
        raise ValueError(f"invalid 2-D model mode {mode!r}")
    invariant_axis = AXES.index(axis_name)
    if family == "TM":
        active_electric = (ELECTRIC_COMPONENTS[invariant_axis],)
        active_magnetic = tuple(
            component
            for axis, component in enumerate(MAGNETIC_COMPONENTS)
            if axis != invariant_axis
        )
        live_index = 0
    else:
        active_electric = tuple(
            component
            for axis, component in enumerate(ELECTRIC_COMPONENTS)
            if axis != invariant_axis
        )
        active_magnetic = (MAGNETIC_COMPONENTS[invariant_axis],)
        live_index = 1
    return Mode2DGeometry(
        mode=mode,
        polarisation=family,
        invariant_axis=invariant_axis,
        live_index=live_index,
        active_electric=active_electric,
        active_magnetic=active_magnetic,
    )
