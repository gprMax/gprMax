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
