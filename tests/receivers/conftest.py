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
def receiver_config(monkeypatch, request):
    """Patch ``gprMax.config`` so receivers methods run in isolation."""
    if request.node.get_closest_marker("unit") is None:
        return

    from gprMax import config

    sim_cfg = SimpleNamespace(
        general={"solver": "cpu", "precision": "double"},
        dtypes={"float_or_double": np.float64},
    )
    sim_cfg.get_model_config = lambda: SimpleNamespace(ompthreads=1)
    monkeypatch.setattr(config, "sim_config", sim_cfg)
    return SimpleNamespace(sim_config=sim_cfg)
