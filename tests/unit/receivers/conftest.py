"""Auto-applied fixtures for the receivers test suite.

``gprMax/receivers.py`` reads two pieces of global config at runtime:

    config.sim_config.dtypes["float_or_double"]   # numpy dtype for arrays
    config.sim_config.general["solver"]            # "cpu" / "cuda" / "opencl" / "metal"

Patched the same way as the sources suite — fresh dicts per test, so
tests that need to exercise GPU branches can mutate ``solver`` in place
without leaking state.
"""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def receiver_config(monkeypatch):
    """Patch ``gprMax.config`` so receivers methods run in isolation."""
    from gprMax import config

    sim_cfg = SimpleNamespace(
        general={"solver": "cpu", "precision": "double"},
        dtypes={"float_or_double": np.float64},
    )
    monkeypatch.setattr(config, "sim_config", sim_cfg)
    return SimpleNamespace(sim_config=sim_cfg)
