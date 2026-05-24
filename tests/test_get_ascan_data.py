import numpy as np
import h5py
import pytest
from tools.plot_Ascan import get_ascan_data


def test_get_ascan_data(tmp_path):
    # Create a minimal fake .out file matching gprMax HDF5 structure
    f_path = tmp_path / "test.out"
    with h5py.File(f_path, "w") as f:
        f.attrs["dt"] = 4.717e-12
        f.attrs["nrx"] = 1
        f.attrs["Iterations"] = 100
        rx = f.create_group("/rxs/rx1")
        rx.create_dataset("Ez", data=np.random.rand(100))

    data = get_ascan_data(str(f_path))

    assert data["dt"] == pytest.approx(4.717e-12)
    assert data["time"].shape == (100,)
    assert "rx1" in data["rxs"]
    assert "Ez" in data["rxs"]["rx1"]
    assert isinstance(data["rxs"]["rx1"]["Ez"], np.ndarray)
    assert data["rxs"]["rx1"]["Ez"].shape == (100,)


def test_get_ascan_data_values_match_hdf5():
    """Values returned must exactly match what's stored in the HDF5 file."""
    import h5py

    filepath = "user_models/cylinder_Ascan_2D.out"

    # read directly from HDF5
    with h5py.File(filepath, "r") as f:
        expected_dt = f.attrs["dt"]
        expected_Ez = f["/rxs/rx1/Ez"][:]

    # read via our function
    data = get_ascan_data(filepath)

    assert data["dt"] == expected_dt
    assert np.array_equal(data["rxs"]["rx1"]["Ez"], expected_Ez)
