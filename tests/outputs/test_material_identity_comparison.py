"""Keep the MPI regression's identity comparison strict without requiring MPI."""

import h5py
import numpy as np
import pytest

from tests.outputs.test_mpi_reduced_outputs import compare_group

pytestmark = pytest.mark.unit


@pytest.fixture
def outputs(tmp_path):
    with h5py.File(tmp_path / "serial.h5", "w") as serial, h5py.File(tmp_path / "mpi.h5", "w") as mpi:
        serial["material_id"] = [3, 4, 3]
        mpi["material_id"] = [1, 0, 1]
        mpi["materials/id"] = [0, 1]
        mpi["materials/name"] = np.array(["other", "tissue"], dtype=h5py.string_dtype())
        mpi["material_id"].attrs["Catalogue"] = "materials"
        for group in (serial, mpi):
            group["sar"] = [1.0, 2.0, 3.0]
        yield serial, mpi


def test_output_local_ids_are_compared_by_material_identity(outputs):
    compare_group(*outputs, "double", material_names={3: "tissue", 4: "other"})


@pytest.mark.parametrize(
    "error",
    [
        "name",
        "duplicate_id",
        "duplicate_name",
        "unmapped_id",
        "unused_id",
        "cell_identity",
        "physical_data",
        "schema",
        "pointer",
    ],
)
def test_identity_comparison_still_detects_corruption(outputs, error):
    serial, mpi = outputs
    if error == "name":
        mpi["materials/name"][0] = "wrong"
    elif error == "duplicate_id":
        mpi["materials/id"][1] = 0
    elif error == "duplicate_name":
        mpi["materials/name"][1] = "other"
    elif error == "unmapped_id":
        mpi["material_id"][0] = 99
    elif error == "unused_id":
        mpi["material_id"][:] = 0
    elif error == "cell_identity":
        mpi["material_id"][:] = [0, 1, 0]
    elif error == "physical_data":
        mpi["sar"][0] = 7
    elif error == "schema":
        mpi["unexpected"] = 1
    elif error == "pointer":
        mpi["material_id"].attrs["Catalogue"] = "wrong"
    with pytest.raises(AssertionError):
        compare_group(serial, mpi, "double", material_names={3: "tissue", 4: "other"})
