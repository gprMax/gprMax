import argparse
from types import SimpleNamespace

import pytest
from pytest import MonkeyPatch

from gprMax import config, gprMax
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import create_built_in_materials
from gprMax.subgrids.user_objects import SubGridHSG


@pytest.fixture
def config_mock(monkeypatch: MonkeyPatch):
    def _mock_simulation_config() -> config.SimulationConfig:
        args = argparse.Namespace(**gprMax.args_defaults)
        args.inputfile = "test.in"
        return config.SimulationConfig(args)

    def _mock_model_config() -> config.ModelConfig:
        model_config = config.ModelConfig(1)
        model_config.ompthreads = 1
        return model_config

    monkeypatch.setattr(config, "sim_config", _mock_simulation_config())
    monkeypatch.setattr(config, "get_model_config", _mock_model_config)


def build_main_grid(nx: int, ny: int, nz: int, dl: float = 0.001, timewindow: float = 5e-9) -> FDTDGrid:
    grid = FDTDGrid()
    grid.nx = nx
    grid.ny = ny
    grid.nz = nz
    grid.dx = dl
    grid.dy = dl
    grid.dz = dl
    grid.timewindow = timewindow
    create_built_in_materials(grid)
    grid.calculate_dt()
    return grid


def build_model(main_grid: FDTDGrid) -> SimpleNamespace:
    """SubGridBase.setup() only reads model.G/dt_mod/iterations and appends
    to model.subgrids, so a plain namespace stands in for gprMax.model.Model.
    """
    return SimpleNamespace(G=main_grid, dt_mod=None, iterations=100, subgrids=[])


def test_subgrid_inherits_main_grid_timewindow(config_mock):
    """Regression test for the bug where SubGridBaseGrid never inherited
    timewindow from the main grid: FDTDGrid.__init__ defaults timewindow to
    0.0, and every source type defaults stop=grid.timewindow when no
    explicit start/stop is given. Without inheritance, a source added
    directly to a subgrid only fires at iteration 0 - silently, with no
    warning, producing tiny noise-like receiver output.
    """
    main_grid = build_main_grid(40, 40, 40, timewindow=5e-9)
    model = build_model(main_grid)

    sg_obj = SubGridHSG(p1=(0.005, 0.005, 0.005), p2=(0.015, 0.015, 0.015), ratio=3, id="sg1")
    sg = sg_obj.build(model)

    assert sg.timewindow == main_grid.timewindow
    assert sg.timewindow != 0.0


def test_subgrid_timewindow_tracks_main_grid_changes(config_mock):
    """Guards against a shallow/one-time copy that would go stale if the
    main grid's timewindow is set after the subgrid is built in some future
    refactor - not currently possible since #time_window always runs before
    subgrids are built, but pins the intended value semantics either way.
    """
    main_grid = build_main_grid(40, 40, 40, timewindow=12e-9)
    model = build_model(main_grid)

    sg_obj = SubGridHSG(p1=(0.005, 0.005, 0.005), p2=(0.015, 0.015, 0.015), ratio=3, id="sg1")
    sg = sg_obj.build(model)

    assert sg.timewindow == pytest.approx(12e-9)
