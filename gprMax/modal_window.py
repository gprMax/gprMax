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

"""Shared Yee constraints at a modal port's artificial PEC rim."""

import numpy as np


def pec_electric_masks(cell_shape, invariant_local=None):
    """Return constrained E masks in the local (u, v, propagation) basis.

    Tangential E is zero at either end of each physical transverse axis.
    Normal E is cell-centred along that axis and remains free. The synthetic
    invariant dimension of a reduced model is not a physical PEC boundary.
    These constraints replace whole Ampere rows, never individual H terms.
    """
    masks = []
    for component in range(3):
        shape = tuple(size + int(axis != component) for axis, size in enumerate(cell_shape))
        mask = np.zeros(shape, dtype=bool)
        for axis in range(2):
            if axis == component or axis == invariant_local:
                continue
            face = [slice(None), slice(None)]
            for position in (0, -1):
                face[axis] = position
                mask[tuple(face)] = True
        masks.append(mask)
    return tuple(masks)


PMC_WINDOW_EDGES = ("u_min", "u_max", "v_min", "v_max")


def window_edge_face(edge):
    """Return the local transverse axis and end index of a named window edge."""
    return (0 if edge[0] == "u" else 1), (0 if edge.endswith("_min") else -1)


def release_pmc_edges(masks, pmc_edges):
    """Free the artificial PEC rim on window edges that lie on a PMC symmetry face.

    ``masks`` are the constrained E masks of :func:`pec_electric_masks`.
    Component ``c`` is tangential to the edges of the other transverse axis,
    and the propagation component to every edge. A sample that also lies on a
    non-PMC edge to which it is tangential keeps its artificial PEC. This is
    the one definition of the rim shared by the modal solver, the
    surface-impedance row filter and the virtual-waveguide check; physical
    PEC is held in separate masks and is never released here.
    """
    released = []
    for component, mask in enumerate(masks):
        original = np.asarray(mask, dtype=bool)
        result = original.copy()
        for edge in pmc_edges:
            axis, position = window_edge_face(edge)
            if axis == component:
                continue
            face = [slice(None), slice(None)]
            face[axis] = position
            result[tuple(face)] = False
        for edge in PMC_WINDOW_EDGES:
            if edge in pmc_edges:
                continue
            axis, position = window_edge_face(edge)
            if axis == component:
                continue
            face = [slice(None), slice(None)]
            face[axis] = position
            result[tuple(face)] |= original[tuple(face)]
        released.append(result)
    return tuple(released)


def pmc_symmetry_window_edges(grid, transverse_axes, start, stop, invariant_axis=None):
    """Return the window edges that lie on a domain face with a PMC symmetry boundary.

    Edges are named in the local (u, v) basis of the port window: ``u_min``,
    ``u_max``, ``v_min`` and ``v_max``. ``start`` and ``stop`` are cell indices
    in the same (global) frame as the grid size. A window edge on a PEC
    symmetry face needs nothing extra, since the window is already closed
    with artificial PEC.
    """
    boundaries = getattr(grid, "symmetry_boundaries", None) or {}
    if "pmc" not in boundaries.values():
        return ()
    size = np.asarray(getattr(grid, "global_size", getattr(grid, "size", None)))
    edges = []
    for local, (axis, low, high) in enumerate(zip(transverse_axes, start, stop)):
        if invariant_axis is not None and axis == invariant_axis:
            continue
        name = "xyz"[axis]
        if int(low) == 0 and boundaries.get(f"{name}0") == "pmc":
            edges.append(f"{'uv'[local]}_min")
        if int(high) == int(size[axis]) and boundaries.get(f"{name}max") == "pmc":
            edges.append(f"{'uv'[local]}_max")
    return tuple(edges)
