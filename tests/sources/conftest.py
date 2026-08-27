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

"""Auto-applied fixtures for the sources test suite.

``gprMax/sources.py`` reads three pieces of global config at runtime:

    config.sim_config.dtypes["float_or_double"]   # numpy dtype for arrays
    config.sim_config.general["solver"]            # "cpu" / "cuda" / "opencl" / "metal"
    config.c, config.m0, config.e0                 # scipy constants

In production these are initialised when a simulation starts. For unit
tests we monkeypatch them once per test so the methods under test see a
predictable, isolated environment.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c, epsilon_0, mu_0


@pytest.fixture(autouse=True)
def source_config(monkeypatch, request):
    """Patch ``gprMax.config`` so sources methods run in isolation.

    Defaults to the CPU solver and double-precision float arrays. Tests
    that need to exercise GPU branches can call
    ``monkeypatch.setattr(config.sim_config.general, "solver", "cuda")``
    inside the test body.
    """
    if request.node.get_closest_marker("unit") is None:
        return

    from gprMax import config

    sim_cfg = SimpleNamespace(
        general={"solver": "cpu", "precision": "double"},
        dtypes={"float_or_double": np.float64},
        em_consts={
            "c": c,
            "e0": epsilon_0,
            "m0": mu_0,
            "z0": float(np.sqrt(mu_0 / epsilon_0)),
        },
    )
    # Provide a minimal get_model_config that upstream code now calls.
    sim_cfg.get_model_config = lambda: SimpleNamespace(ompthreads=1, mode="3D")

    monkeypatch.setattr(config, "sim_config", sim_cfg)
    monkeypatch.setattr(config, "c", c, raising=False)
    monkeypatch.setattr(config, "e0", epsilon_0, raising=False)
    monkeypatch.setattr(config, "m0", mu_0, raising=False)

    return SimpleNamespace(sim_config=sim_cfg, C=c, EPS0=epsilon_0, MU0=mu_0)
