import numpy as np
import pytest

from gprMax import config
from gprMax.grid import FDTDGrid
from gprMax.updates import CPUUpdates


class MockSimulationConfig:
    dtypes = {"float_or_double": np.float32}


class MockModelConfig:
    ompthreads = 1


@pytest.fixture
def config_mock(monkeypatch):
    monkeypatch.setattr(config, "sim_config", MockSimulationConfig)
    monkeypatch.setattr(config, "get_model_config", MockModelConfig)


@pytest.fixture
def build_grid():
    def _build_grid(nx, ny, nz):
        grid = FDTDGrid()
        grid.nx = nx
        grid.ny = ny
        grid.nz = nz
        grid.initialise_geometry_arrays()
        grid.initialise_field_arrays()
        grid.initialise_std_update_coeff_arrays()
        return grid

    return _build_grid


def test_update_magnetic_cpu(config_mock, build_grid):
    grid = build_grid(100, 100, 100)

    expected_value = grid.Ex.copy()

    cpu_updates = CPUUpdates(grid)
    cpu_updates.update_magnetic()

    assert np.equal(grid.Ex, expected_value).all()
    assert np.equal(grid.Ey, expected_value).all()
    assert np.equal(grid.Ez, expected_value).all()
    assert np.equal(grid.Hx, expected_value).all()
    assert np.equal(grid.Hy, expected_value).all()
    assert np.equal(grid.Hz, expected_value).all()
