# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

import numpy as np

from toolboxes.STLtoVoxel.convert import convert_meshes
from toolboxes.STLtoVoxel.slice import calculate_scale_shift


def test_parallel_conversion_matches_serial_conversion():
    mesh = np.array(
        [
            [[0, 0, 0], [10, 0, 0], [0, 10, 0]],
            [[0, 0, 0], [10, 0, 0], [0, 0, 10]],
            [[0, 0, 0], [0, 10, 0], [0, 0, 10]],
            [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        ],
        dtype=float,
    )

    serial, *_ = convert_meshes([mesh], (0.001, 0.001, 0.001), parallel=False)
    parallel, *_ = convert_meshes([mesh], (0.001, 0.001, 0.001), parallel=True)

    assert np.count_nonzero(serial >= 0) > 0
    np.testing.assert_array_equal(parallel, serial)


def test_anisotropic_discretisation_sets_each_axis_independently():
    mesh = np.array(
        [[[0, 0, 0], [10, 0, 0], [0, 20, 30]]],
        dtype=float,
    )

    scale, _, shape = calculate_scale_shift([mesh], (0.001, 0.002, 0.003))

    np.testing.assert_allclose(scale, [1, 0.5, 1 / 3])
    assert shape == [11, 11, 11]
