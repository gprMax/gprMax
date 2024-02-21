import argparse

import numpy as np
import pytest
from pytest import MonkeyPatch

from gprMax import config, gprMax
from gprMax.grid import FDTDGrid
from gprMax.materials import create_built_in_materials
from gprMax.model_build_run import GridBuilder
from gprMax.pml import CFS
from gprMax.updates.cpu_updates import CPUUpdates


def build_grid(nx: int, ny: int, nz: int, dl: float = 0.001, dt: float = 3e-9) -> FDTDGrid:
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
def config_mock(monkeypatch: MonkeyPatch):
    def _mock_simulation_config() -> config.SimulationConfig:
        args = argparse.Namespace(**gprMax.args_defaults)
        args.inputfile = "test.in"
        return config.SimulationConfig(args)

    def _mock_model_config() -> config.ModelConfig:
        model_config = config.ModelConfig()
        model_config.ompthreads = 1
        return model_config

    monkeypatch.setattr(config, "sim_config", _mock_simulation_config())
    monkeypatch.setattr(config, "get_model_config", _mock_model_config)


def test_update_magnetic_zero_grid(config_mock):
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


def test_update_magnetic(config_mock):
    grid = build_grid(11, 11, 11)

    grid.updatecoeffsH[1] = 1

    grid.Ex = np.tile(np.array([[[1, 2], [2, 1]], [[2, 1], [1, 2]]], dtype=np.float32), (6, 6, 6))
    grid.Ey = np.tile(np.array([[[1, 3], [3, 1]], [[3, 1], [1, 3]]], dtype=np.float32), (6, 6, 6))
    grid.Ez = np.tile(np.array([[[1, 4], [4, 1]], [[4, 1], [1, 4]]], dtype=np.float32), (6, 6, 6))
    grid.Hx = np.tile(np.array([[[3, 1], [1, 3]], [[1, 3], [3, 1]]], dtype=np.float32), (6, 6, 6))
    grid.Hy = np.tile(np.array([[[1, 5], [5, 1]], [[5, 1], [1, 5]]], dtype=np.float32), (6, 6, 6))
    grid.Hz = np.tile(np.array([[[5, 3], [3, 5]], [[3, 5], [5, 3]]], dtype=np.float32), (6, 6, 6))

    expected_Ex = grid.Ex.copy()
    expected_Ey = grid.Ey.copy()
    expected_Ez = grid.Ez.copy()
    expected_Hx = grid.Hx.copy()
    expected_Hy = grid.Hy.copy()
    expected_Hz = grid.Hz.copy()
    expected_Hx[1:, :-1, :-1] = np.tile(np.array([[[2]]], dtype=np.float32), (11, 11, 11))
    expected_Hy[:-1, 1:, :-1] = np.tile(np.array([[[3]]], dtype=np.float32), (11, 11, 11))
    expected_Hz[:-1, :-1, 1:] = np.tile(np.array([[[4]]], dtype=np.float32), (11, 11, 11))

    # Why does fields_updates_normal use i+1, j+1 and k+1 everywhere?
    cpu_updates = CPUUpdates(grid)
    cpu_updates.update_magnetic()

    assert np.equal(grid.Ex, expected_Ex).all()
    assert np.equal(grid.Ey, expected_Ey).all()
    assert np.equal(grid.Ez, expected_Ez).all()
    assert np.equal(grid.Hx, expected_Hx).all()
    assert np.equal(grid.Hy, expected_Hy).all()
    assert np.equal(grid.Hz, expected_Hz).all()


def test_update_electric_a_non_dispersive_zero_grid(config_mock):
    grid = build_grid(100, 100, 100)

    expected_value = np.zeros((101, 101, 101))

    cpu_updates = CPUUpdates(grid)
    cpu_updates.update_electric_a()

    assert np.equal(grid.Ex, expected_value).all()
    assert np.equal(grid.Ey, expected_value).all()
    assert np.equal(grid.Ez, expected_value).all()
    assert np.equal(grid.Hx, expected_value).all()
    assert np.equal(grid.Hy, expected_value).all()
    assert np.equal(grid.Hz, expected_value).all()


def test_update_electric_a_non_dispersive(config_mock):
    grid = build_grid(11, 11, 11)

    print(grid.updatecoeffsE)
    print(grid.updatecoeffsE[1])
    grid.updatecoeffsE[1] = 1
    print(grid.updatecoeffsE[1])

    grid.Ex = np.tile(np.array([[[3, 1], [1, 3]], [[1, 3], [3, 1]]], dtype=np.float32), (6, 6, 6))
    grid.Ey = np.tile(np.array([[[1, 5], [5, 1]], [[5, 1], [1, 5]]], dtype=np.float32), (6, 6, 6))
    grid.Ez = np.tile(np.array([[[5, 3], [3, 5]], [[3, 5], [5, 3]]], dtype=np.float32), (6, 6, 6))
    grid.Hx = np.tile(np.array([[[1, 2], [2, 1]], [[2, 1], [1, 2]]], dtype=np.float32), (6, 6, 6))
    grid.Hy = np.tile(np.array([[[1, 3], [3, 1]], [[3, 1], [1, 3]]], dtype=np.float32), (6, 6, 6))
    grid.Hz = np.tile(np.array([[[1, 4], [4, 1]], [[4, 1], [1, 4]]], dtype=np.float32), (6, 6, 6))

    expected_Ex = grid.Ex.copy()
    expected_Ey = grid.Ey.copy()
    expected_Ez = grid.Ez.copy()
    expected_Hx = grid.Hx.copy()
    expected_Hy = grid.Hy.copy()
    expected_Hz = grid.Hz.copy()
    # Why is there not a full (11x11x11) section of the frid being updated?
    expected_Ex[:-1, 1:-1, 1:-1] = np.tile(np.array([[[2]]], dtype=np.float32), (11, 10, 10))
    expected_Ey[1:-1, :-1, 1:-1] = np.tile(np.array([[[3]]], dtype=np.float32), (10, 11, 10))
    expected_Ez[1:-1, 1:-1, :-1] = np.tile(np.array([[[4]]], dtype=np.float32), (10, 10, 11))

    cpu_updates = CPUUpdates(grid)
    cpu_updates.update_electric_a()

    assert np.equal(grid.Ex, expected_Ex).all()
    assert np.equal(grid.Ey, expected_Ey).all()
    assert np.equal(grid.Ez, expected_Ez).all()
    assert np.equal(grid.Hx, expected_Hx).all()
    assert np.equal(grid.Hy, expected_Hy).all()
    assert np.equal(grid.Hz, expected_Hz).all()


def test_update_electric_b_non_dispersive(config_mock):
    grid = build_grid(100, 100, 100)

    expected_value = np.zeros((101, 101, 101))

    cpu_updates = CPUUpdates(grid)
    cpu_updates.update_electric_b()

    assert np.equal(grid.Ex, expected_value).all()
    assert np.equal(grid.Ey, expected_value).all()
    assert np.equal(grid.Ez, expected_value).all()
    assert np.equal(grid.Hx, expected_value).all()
    assert np.equal(grid.Hy, expected_value).all()
    assert np.equal(grid.Hz, expected_value).all()


def test_update_electric_a_dispersive(config_mock):
    assert False


def test_update_electric_b_dispersive(config_mock):
    assert False


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


def test_update_magnetic_sources(config_mock):
    assert False


def test_update_electric_sources(config_mock):
    assert False


def test_dispersive_update_a(config_mock):
    assert False


def test_dispersive_update_b(config_mock):
    assert False
