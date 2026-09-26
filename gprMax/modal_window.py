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

"""Shared Yee constraints at a surface-impedance port's artificial PEC rim."""

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


def sibc_window_pec_masks(system, plane, transverse_axes, start, stop, invariant_local=None):
    """Constrain artificial cuts, but retain complete physical SIBC walls.

    A surface row on the rim is a physical wall if its entire ordinary-H
    circulation is present in the port. If cropping removes any H sample,
    the artificial PEC wall replaces that complete electric row instead.
    """
    cell_shape = tuple(int(high - low) for low, high in zip(start, stop))
    masks = list(pec_electric_masks(cell_shape, invariant_local))
    local_axes = (*transverse_axes, next(axis for axis in range(3) if axis not in transverse_axes))
    magnetic_shapes = (
        (cell_shape[0] + 1, cell_shape[1]),
        (cell_shape[0], cell_shape[1] + 1),
        cell_shape,
    )
    for edge in system.edge_info:
        if edge[1 + local_axes[2]] != plane:
            continue
        electric_axis = local_axes.index(int(edge[0]))
        index = tuple(int(edge[1 + axis] - low) for axis, low in zip(transverse_axes, start))
        mask = masks[electric_axis]
        if not all(0 <= value < size for value, size in zip(index, mask.shape)):
            continue
        if not mask[index]:
            continue
        h_start, h_count = (int(value) for value in edge[4:6])
        if h_count == 0:
            continue
        for h in system.h_info[h_start : h_start + h_count]:
            magnetic_axis = local_axes.index(int(h[0]))
            h_index = tuple(int(h[1 + axis] - low) for axis, low in zip(transverse_axes, start))
            if not all(
                0 <= value < size
                for value, size in zip(h_index, magnetic_shapes[magnetic_axis])
            ):
                break
        else:
            mask[index] = False
    return tuple(masks)
