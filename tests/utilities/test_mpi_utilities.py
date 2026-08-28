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

import numpy as np
import pytest
from mpi4py import MPI

from gprMax.utilities.mpi import mpi_datatype_for_dtype


@pytest.mark.parametrize(
    ("numpy_dtype", "expected_mpi_dtype"),
    [(np.float32, MPI.FLOAT), (np.float64, MPI.DOUBLE)],
)
def test_mpi_datatype_matches_configured_precision(numpy_dtype, expected_mpi_dtype):
    mpi_dtype = mpi_datatype_for_dtype(numpy_dtype)

    assert mpi_dtype == expected_mpi_dtype
    assert mpi_dtype.Get_size() == np.dtype(numpy_dtype).itemsize


def test_mpi_datatype_rejects_unsupported_precision():
    with pytest.raises(TypeError, match="float32 or float64"):
        mpi_datatype_for_dtype(np.int32)
