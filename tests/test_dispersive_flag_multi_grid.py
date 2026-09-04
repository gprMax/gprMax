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

"""Per-grid dispersion configuration and model-wide summary tests."""
from types import SimpleNamespace

import numpy as np
import pytest

from gprMax import config
from gprMax.model import Model


@pytest.fixture(autouse=True)
def _sim_config(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={
                "float_or_double": np.float64,
                "complex": np.complex128,
                "C_float_or_double": "double",
                "C_complex": "double2",
            }
        ),
    )


def _material(type_, poles=1):
    return SimpleNamespace(type=type_, poles=poles)


def _fake_model_config():
    return SimpleNamespace(
        materials={"maxpoles": 1, "drudelorentz": None},
        set_dispersive_material_types=lambda: None,
    )


def test_drudelorentz_true_when_earlier_grid_has_lorentz_and_later_grid_is_debye_only(
    monkeypatch,
):
    model_config = _fake_model_config()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)

    main_grid = SimpleNamespace(materials=[_material("lorentz")])
    subgrid = SimpleNamespace(materials=[_material("debye")])

    Model._check_for_dispersive_materials(None, [main_grid, subgrid])

    assert model_config.materials["drudelorentz"] is True
    assert main_grid.drudelorentz is True
    assert subgrid.drudelorentz is False
    assert main_grid.maxpoles == subgrid.maxpoles == 1


def test_drudelorentz_true_when_only_a_subgrid_has_drude(monkeypatch):
    model_config = _fake_model_config()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)

    main_grid = SimpleNamespace(materials=[_material("dielectric", poles=0)])
    subgrid = SimpleNamespace(materials=[_material("drude")])

    Model._check_for_dispersive_materials(None, [main_grid, subgrid])

    assert model_config.materials["drudelorentz"] is True
    assert main_grid.maxpoles == 0
    assert main_grid.dispersivedtype is None
    assert subgrid.maxpoles == 1
    assert subgrid.drudelorentz is True


def test_drudelorentz_false_when_no_grid_has_drude_or_lorentz(monkeypatch):
    model_config = _fake_model_config()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)

    main_grid = SimpleNamespace(materials=[_material("debye")])
    subgrid = SimpleNamespace(materials=[_material("debye")])

    Model._check_for_dispersive_materials(None, [main_grid, subgrid])

    assert model_config.materials["drudelorentz"] is False
    assert main_grid.drudelorentz is False
    assert subgrid.drudelorentz is False


def test_pole_counts_and_dtypes_are_independent_for_each_grid(monkeypatch):
    model_config = _fake_model_config()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)

    main_grid = SimpleNamespace(materials=[_material("dielectric", poles=0)])
    subgrid = SimpleNamespace(materials=[_material("debye", poles=3)])

    Model._check_for_dispersive_materials(None, [main_grid, subgrid])

    assert main_grid.maxpoles == 0
    assert main_grid.dispersivedtype is None
    assert subgrid.maxpoles == 3
    assert subgrid.dispersivedtype is config.sim_config.dtypes["float_or_double"]
    assert model_config.materials["maxpoles"] == 3


def test_material_crossing_interface_configures_both_grids(monkeypatch):
    model_config = _fake_model_config()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)

    main_grid = SimpleNamespace(materials=[_material("debye", poles=2)])
    subgrid = SimpleNamespace(materials=[_material("debye", poles=2)])

    Model._check_for_dispersive_materials(None, [main_grid, subgrid])

    assert main_grid.maxpoles == subgrid.maxpoles == 2
    assert main_grid.dispersivedtype is config.sim_config.dtypes["float_or_double"]
    assert subgrid.dispersivedtype is config.sim_config.dtypes["float_or_double"]


def test_material_only_on_main_grid_leaves_subgrid_plain(monkeypatch):
    model_config = _fake_model_config()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)

    main_grid = SimpleNamespace(materials=[_material("lorentz", poles=2)])
    subgrid = SimpleNamespace(materials=[_material("dielectric", poles=0)])

    Model._check_for_dispersive_materials(None, [main_grid, subgrid])

    assert main_grid.maxpoles == 2
    assert main_grid.drudelorentz is True
    assert subgrid.maxpoles == 0
    assert subgrid.dispersivedtype is None
