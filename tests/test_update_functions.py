import numpy as np

from gprMax import config
from gprMax.grid import FDTDGrid
from gprMax.updates import CPUUpdates


class MockSimulationConfig:
    dtypes = {"float_or_double": np.float32}


class MockModelConfig:
    ompthreads = 1


def test_update_magnetic_cpu(monkeypatch):
    monkeypatch.setattr(config, "sim_config", MockSimulationConfig)
    monkeypatch.setattr(config, "get_model_config", MockModelConfig)

    # TODO: Move building the grid into a fixture
    grid = FDTDGrid()
    grid.nx = 100
    grid.ny = 100
    grid.nz = 100
    grid.initialise_geometry_arrays()
    grid.initialise_field_arrays()
    grid.initialise_std_update_coeff_arrays()

    cpu_updates = CPUUpdates(grid)

    expected_value = grid.Ex.copy()

    cpu_updates.update_magnetic()

    assert np.equal(grid.Ex, expected_value).all()
    assert np.equal(grid.Ey, expected_value).all()
    assert np.equal(grid.Ez, expected_value).all()
    assert np.equal(grid.Hx, expected_value).all()
    assert np.equal(grid.Hy, expected_value).all()
    assert np.equal(grid.Hz, expected_value).all()
