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
