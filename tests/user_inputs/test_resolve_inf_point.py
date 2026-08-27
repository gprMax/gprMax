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

"""Unit tests for UserInput.resolve_inf_point() (gprMax/user_inputs.py) -
the shared `inf`-coordinate resolution used by position-taking commands
(Box, HertzianDipole, Rx, etc.) ahead of discretisation.

Rules under test (see gprMax/user_inputs.py docstring for the full
rationale):
  - `inf` is only allowed when the model is in an active 2D mode (TM or
    TE, via #domain_mode). In 3D there is no ambiguity for `inf` to
    resolve - every coordinate already has one unambiguous meaning - so
    it is rejected with a clear error rather than silently doing a
    "snap to domain edge" convenience (which, among other problems,
    cannot be resolved correctly for subgrid-scoped objects).
  - Range endpoints (role="lower"/"upper"): purely positional, sign
    ignored. Works the same on the invariant axis of a 2D mode as on any
    other axis within it - "lower" and "upper" are still distinct, real
    positions there (0 and the axis extent), so no override is needed
    or applied for range endpoints.
  - Single points (role=None): sign carries the meaning (-inf -> 0,
    inf/+inf -> axis extent), EXCEPT on the invariant axis of the active
    2D mode, where any inf (regardless of sign) redirects to that
    mode's interior reference layer (index 0 for TM, index 1 for TE) -
    since the axis extent itself would be a dead PEC/PMC-forced
    boundary layer for TE, not a real single point to place a source.
"""
import pytest

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.user_inputs import MainGridUserInput

INF = float("inf")


class _FakeModelConfig:
    def __init__(self, mode):
        self.mode = mode


def _uip(monkeypatch, mode, dl=(1e-3, 1e-3, 1e-3), size=(10, 10, 10)):
    monkeypatch.setattr(config, "get_model_config", lambda: _FakeModelConfig(mode))
    grid = FDTDGrid()
    grid.dl = dl
    grid.nx, grid.ny, grid.nz = size
    return MainGridUserInput(grid)


def test_no_inf_returns_point_unchanged(monkeypatch):
    uip = _uip(monkeypatch, "3D")
    point = (0.001, 0.002, 0.003)
    assert uip.resolve_inf_point(point) == point


@pytest.mark.parametrize("role", [None, "lower", "upper"])
def test_inf_in_3d_mode_raises(monkeypatch, role):
    uip = _uip(monkeypatch, "3D")
    with pytest.raises(ValueError, match="2D"):
        uip.resolve_inf_point((INF, 0, 0), role=role)


def test_inf_on_non_invariant_axis_of_2d_model_raises_if_no_mode(monkeypatch):
    """Sanity check the guard is keyed off the resolved model mode, not
    just some flag - the unset ("3D") default is rejected too."""
    uip = _uip(monkeypatch, "3D")
    with pytest.raises(ValueError, match="2D"):
        uip.resolve_inf_point((0, INF, 0))


def test_range_lower_role_resolves_to_zero_regardless_of_sign(monkeypatch):
    uip = _uip(monkeypatch, "2D TEz", size=(10, 10, 2))
    assert uip.resolve_inf_point((INF, 0, 0), role="lower") == (0.0, 0, 0)
    assert uip.resolve_inf_point((-INF, 0, 0), role="lower") == (0.0, 0, 0)


def test_range_upper_role_resolves_to_axis_extent_regardless_of_sign(monkeypatch):
    uip = _uip(monkeypatch, "2D TEz", dl=(1e-3, 1e-3, 1e-3), size=(10, 20, 2))
    assert uip.resolve_inf_point((INF, 0, 0), role="upper") == (0.01, 0, 0)
    assert uip.resolve_inf_point((-INF, 0, 0), role="upper") == (0.01, 0, 0)


def test_single_point_sign_based_resolution_on_non_invariant_axis(monkeypatch):
    uip = _uip(monkeypatch, "2D TEz", dl=(1e-3, 1e-3, 1e-3), size=(10, 10, 2))
    assert uip.resolve_inf_point((INF, 0, 0)) == (0.01, 0, 0)
    assert uip.resolve_inf_point((-INF, 0, 0)) == (0.0, 0, 0)


def test_range_endpoints_span_full_tm_invariant_axis(monkeypatch):
    """No override for range endpoints - TM's 1-cell axis should get a
    real 0..1-cell span, not collapse to a single point."""
    uip = _uip(monkeypatch, "2D TMz", dl=(1e-3, 1e-3, 1e-3), size=(10, 10, 1))
    assert uip.resolve_inf_point((0, 0, INF), role="lower") == (0, 0, 0.0)
    assert uip.resolve_inf_point((0, 0, INF), role="upper") == (0, 0, 0.001)


def test_range_endpoints_span_full_te_invariant_axis(monkeypatch):
    """No override for range endpoints - TE's 2-cell axis should get a
    real 0..2-cell span, not collapse to the single interior layer."""
    uip = _uip(monkeypatch, "2D TEz", dl=(1e-3, 1e-3, 1e-3), size=(10, 10, 2))
    assert uip.resolve_inf_point((0, 0, INF), role="lower") == (0, 0, 0.0)
    assert uip.resolve_inf_point((0, 0, INF), role="upper") == (0, 0, 0.002)


def test_single_point_overridden_to_interior_layer_on_tm_invariant_axis(monkeypatch):
    uip = _uip(monkeypatch, "2D TMz", dl=(1e-3, 1e-3, 1e-3), size=(10, 10, 1))
    assert uip.resolve_inf_point((0, 0, INF)) == (0, 0, 0.0)
    assert uip.resolve_inf_point((0, 0, -INF)) == (0, 0, 0.0)


def test_single_point_overridden_to_interior_layer_on_te_invariant_axis(monkeypatch):
    uip = _uip(monkeypatch, "2D TEz", dl=(1e-3, 1e-3, 1e-3), size=(10, 10, 2))
    # both signs redirect to the interior layer (index 1), NOT to the
    # dead pec/pmc-forced outer walls (index 0 or 2) that the plain
    # sign-based rule would otherwise give.
    assert uip.resolve_inf_point((0, 0, INF)) == (0, 0, 0.001)
    assert uip.resolve_inf_point((0, 0, -INF)) == (0, 0, 0.001)


def test_single_point_non_invariant_axis_unaffected_by_invariant_axis_override(monkeypatch):
    uip = _uip(monkeypatch, "2D TEz", dl=(1e-3, 1e-3, 1e-3), size=(10, 10, 2))
    assert uip.resolve_inf_point((INF, 0, 0)) == (0.01, 0, 0)
    assert uip.resolve_inf_point((-INF, 0, 0)) == (0.0, 0, 0)
