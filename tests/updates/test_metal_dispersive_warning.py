"""Regression test confirming create_solver() no longer warns about
dispersive materials on Metal (gprMax/solvers.py).

A stopgap warning used to fire here because Metal silently ignored
dispersive (Debye/Lorentz/Drude) materials - update_electric_a fell back
to the non-dispersive pipeline and update_electric_b was a bare `pass`.
Full dispersive support has since been implemented for Metal (see
tests/updates/test_metal_dispersive_dispatch.py), so create_solver()
should no longer emit this warning at all - it would now be actively
misleading, since Metal handles dispersive materials the same way every
other solver does.

Real Apple Metal hardware/PyObjC isn't available in this environment
(MetalGrid/MetalUpdates both call `import_module("Metal")` in __init__,
which fails immediately on non-macOS), so this test exercises
create_solver()'s dispatch logic directly with fake MetalGrid/MetalUpdates
stand-ins (monkeypatched into gprMax.solvers) rather than constructing the
real classes.
"""
import argparse

import pytest

import gprMax.solvers as solvers_mod
from gprMax import config, gprMax


class _FakeMetalGrid:
    pass


class _FakeMetalUpdates:
    def __init__(self, grid):
        self.grid = grid


@pytest.fixture
def config_mock(monkeypatch):
    args = argparse.Namespace(**gprMax.args_defaults)
    args.inputfile = "test.in"
    sim_config = config.SimulationConfig(args)
    monkeypatch.setattr(config, "sim_config", sim_config)

    # ModelConfig.__init__ reads the module-level config.sim_config directly,
    # so it must already be patched in before construction.
    model_config = config.ModelConfig(1)
    model_config.ompthreads = 1

    # A single shared instance (not a fresh ModelConfig() per call) so that
    # a test's config.get_model_config().materials["maxpoles"] = ... write
    # is visible to create_solver()'s own, separate get_model_config() call.
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    monkeypatch.setattr(solvers_mod, "MetalGrid", _FakeMetalGrid)
    monkeypatch.setattr(solvers_mod, "MetalUpdates", _FakeMetalUpdates)


class _FakeModel:
    def __init__(self, grid):
        self.G = grid


def test_metal_solver_no_longer_warns_when_dispersive_materials_present(config_mock, caplog):
    config.get_model_config().materials["maxpoles"] = 1

    with caplog.at_level("WARNING"):
        solvers_mod.create_solver(_FakeModel(_FakeMetalGrid()))

    assert not any("dispersive" in r.message.lower() for r in caplog.records)


def test_metal_solver_silent_when_no_dispersive_materials(config_mock, caplog):
    config.get_model_config().materials["maxpoles"] = 0

    with caplog.at_level("WARNING"):
        solvers_mod.create_solver(_FakeModel(_FakeMetalGrid()))

    assert not any("dispersive" in r.message.lower() for r in caplog.records)
