"""Auto-applied fixtures for the materials test suite.

`gprMax/materials.py` reads four pieces of global config at runtime:

    config.m0
    config.sim_config.em_consts["e0"]
    config.get_model_config().materials["maxpoles"]
    config.get_model_config().materials["dispersivedtype"]

In production these are initialised when a simulation starts. For unit
tests we monkeypatch them once per test so the methods under test see a
predictable, isolated environment.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import epsilon_0, mu_0


@pytest.fixture(autouse=True)
def material_config(monkeypatch, request):
    """Patch ``gprMax.config`` so materials methods run in isolation.

    Returns a ``SimpleNamespace`` exposing the values the patched config
    holds, so individual tests can reference them in assertions without
    re-importing scipy constants.
    """
    if request.node.get_closest_marker("unit") is None:
        return

    from gprMax import config

    em_consts = {
        "e0": epsilon_0,
        "m0": mu_0,
        "c": 299_792_458.0,
        "z0": float(np.sqrt(mu_0 / epsilon_0)),
    }

    materials_cfg = {
        "maxpoles": 1,
        "dispersivedtype": np.complex128,
        "dispersiveCdtype": None,
        "drudelorentz": None,
        "crealfunc": None,
    }

    model_cfg = SimpleNamespace(
        materials=materials_cfg, debye_averaging=True, dispersive_averaging=True
    )
    sim_cfg = SimpleNamespace(em_consts=em_consts)

    monkeypatch.setattr(config, "sim_config", sim_cfg)
    monkeypatch.setattr(config, "get_model_config", lambda: model_cfg)

    return SimpleNamespace(
        em_consts=em_consts,
        materials=materials_cfg,
        EPS0=epsilon_0,
        MU0=mu_0,
    )
