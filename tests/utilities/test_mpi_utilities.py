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
