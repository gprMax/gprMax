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

"""Compatibility check for the optional distributed-FFT dependency."""

import numpy as np
import pytest


@pytest.mark.unit
def test_mpi4py_fft_fftw_round_trip():
    """A one-rank FFTW transform exercises the compiled mpi4py-fft modules."""

    MPI = pytest.importorskip("mpi4py.MPI")
    mpi4py_fft = pytest.importorskip("mpi4py_fft")

    fft = mpi4py_fft.PFFT(
        MPI.COMM_SELF,
        (8, 8),
        axes=(0, 1),
        dtype=float,
        backend="fftw",
    )
    values = mpi4py_fft.newDistArray(fft, False)
    transformed = mpi4py_fft.newDistArray(fft, True)
    values[...] = np.arange(values.size).reshape(values.shape)
    expected = values.copy()

    fft.forward(values, transformed)
    fft.backward(transformed, values)

    assert np.allclose(values, expected)
