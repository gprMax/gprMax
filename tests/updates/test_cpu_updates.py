import argparse

import numpy as np
import pytest

from gprMax import config, gprMax
from gprMax.grid import FDTDGrid
from gprMax.materials import create_built_in_materials
from gprMax.model_build_run import GridBuilder
from gprMax.pml import CFS
from gprMax.updates.cpu_updates import CPUUpdates


def build_grid(nx, ny, nz, dl=0.001, dt=3e-9):
    grid = FDTDGrid()
    grid.nx = nx
    grid.ny = ny
    grid.nz = nz
    grid.dx = dl
    grid.dy = dl
    grid.dz = dl
    grid.dt = dt
    create_built_in_materials(grid)
    grid.initialise_geometry_arrays()
    grid.initialise_field_arrays()
    grid.initialise_std_update_coeff_arrays()
    grid.pmls["cfs"] = [CFS()]

    grid_builder = GridBuilder(grid)
    grid_builder.build_pmls()
    grid_builder.build_components()
    grid_builder.build_materials()

    return grid


@pytest.fixture
def config_mock(monkeypatch):
    def _mock_simulation_config():
        args = argparse.Namespace(**gprMax.args_defaults)
        args.inputfile = "test.in"
        return config.SimulationConfig(args)

    def _mock_model_config():
        model_config = config.ModelConfig()
        model_config.ompthreads = 1
        return model_config

    monkeypatch.setattr(config, "sim_config", _mock_simulation_config())
    monkeypatch.setattr(config, "get_model_config", _mock_model_config)


def test_update_magnetic(config_mock):
    grid = build_grid(100, 100, 100)

    expected_value = np.zeros((101, 101, 101))

    cpu_updates = CPUUpdates(grid)
    cpu_updates.update_magnetic()

    assert np.equal(grid.Ex, expected_value).all()
    assert np.equal(grid.Ey, expected_value).all()
    assert np.equal(grid.Ez, expected_value).all()
    assert np.equal(grid.Hx, expected_value).all()
    assert np.equal(grid.Hy, expected_value).all()
    assert np.equal(grid.Hz, expected_value).all()

    for pml in grid.pmls["slabs"]:
        assert np.equal(pml.HPhi1, 0).all()
        assert np.equal(pml.HPhi2, 0).all()
        assert np.equal(pml.EPhi1, 0).all()
        assert np.equal(pml.EPhi2, 0).all()


def test_update_magnetic_pml(config_mock):
    grid = build_grid(100, 100, 100)

    grid_expected_value = np.zeros((101, 101, 101))

    cpu_updates = CPUUpdates(grid)
    cpu_updates.update_magnetic_pml()

    assert np.equal(grid.Ex, grid_expected_value).all()
    assert np.equal(grid.Ey, grid_expected_value).all()
    assert np.equal(grid.Ez, grid_expected_value).all()
    assert np.equal(grid.Hx, grid_expected_value).all()
    assert np.equal(grid.Hy, grid_expected_value).all()
    assert np.equal(grid.Hz, grid_expected_value).all()

    for pml in grid.pmls["slabs"]:
        assert np.equal(pml.HPhi1, 0).all()
        assert np.equal(pml.HPhi2, 0).all()
        assert np.equal(pml.EPhi1, 0).all()
        assert np.equal(pml.EPhi2, 0).all()


def test_update_electric_pml(config_mock):
    grid = build_grid(100, 100, 100)

    grid_expected_value = np.zeros((101, 101, 101))

    cpu_updates = CPUUpdates(grid)
    cpu_updates.update_electric_pml()

    assert np.equal(grid.Ex, grid_expected_value).all()
    assert np.equal(grid.Ey, grid_expected_value).all()
    assert np.equal(grid.Ez, grid_expected_value).all()
    assert np.equal(grid.Hx, grid_expected_value).all()
    assert np.equal(grid.Hy, grid_expected_value).all()
    assert np.equal(grid.Hz, grid_expected_value).all()

    for pml in grid.pmls["slabs"]:
        assert np.equal(pml.HPhi1, 0).all()
        assert np.equal(pml.HPhi2, 0).all()
        assert np.equal(pml.EPhi1, 0).all()
        assert np.equal(pml.EPhi2, 0).all()
