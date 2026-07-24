"""Regression test for Model._check_for_dispersive_materials()
(Codex-reported): materials["drudelorentz"] was overwritten on every
iteration of `for grid in grids: ...`, so its final value reflected only
the *last* grid checked (main grid + subgrids, in that order) rather than
whether *any* grid anywhere contains a Drude/Lorentz material.

materials["drudelorentz"]/["dispersivedtype"]/["dispersiveCdtype"] are
single, model-wide settings used by *every* grid's
FDTDGrid.initialise_dispersive_arrays() to choose the array dtype. Getting
the flag wrong for an earlier grid isn't just "wrong kernel selected" -
Lorentz/Drude pole coefficients (materials.py's material.eqt/zt) are
genuinely complex-valued; assigning them into a real-dtype
updatecoeffsdispersive array (allocated because a later, Debye-only grid
reset the global flag to False) silently truncates them to their real part
(numpy only raises a ComplexWarning on this assignment, not an error).

Example that used to break: main grid has a Lorentz material, a subgrid
has only a Debye material. The old loop set drudelorentz=True (main grid),
then overwrote it to False (subgrid, checked last) - so the main grid's own
Lorentz arrays would end up allocated with the wrong (real) dtype.

Fixed by aggregating with any(...) across every material in every grid,
not reassigning per grid in a loop.
"""
from types import SimpleNamespace

from gprMax import config
from gprMax.model import Model


def _material(type_):
    return SimpleNamespace(type=type_)


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


def test_drudelorentz_true_when_only_a_subgrid_has_drude(monkeypatch):
    model_config = _fake_model_config()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)

    main_grid = SimpleNamespace(materials=[_material("debye")])
    subgrid = SimpleNamespace(materials=[_material("drude")])

    Model._check_for_dispersive_materials(None, [main_grid, subgrid])

    assert model_config.materials["drudelorentz"] is True


def test_drudelorentz_false_when_no_grid_has_drude_or_lorentz(monkeypatch):
    model_config = _fake_model_config()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)

    main_grid = SimpleNamespace(materials=[_material("debye")])
    subgrid = SimpleNamespace(materials=[_material("debye")])

    Model._check_for_dispersive_materials(None, [main_grid, subgrid])

    assert model_config.materials["drudelorentz"] is False
