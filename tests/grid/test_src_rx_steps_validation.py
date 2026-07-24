"""Regression tests for FDTDGrid._update_positions() (Codex-reported):

1. `if any(step_size > 0):` gated the entire method body (both the
   bounds check and the actual repositioning) - so an all-negative step
   vector (a valid request to move backward each model) was silently
   ignored: positions were never updated across model runs, with no
   error or warning, even though SrcSteps/RxSteps accepted and logged it
   as a "valid" step. Fixed to `any(step_size != 0)`.

2. The bounds check used `step_size * config.sim_config.model_end`, one
   step further than the actual last model run (index `model_end - 1`,
   since models run over `range(model_start, model_end)`) - rejecting
   some valid scans that fit exactly within the domain boundary. Fixed
   to use `model_end - 1`.

Uses a minimal FDTDGrid instance (bypassing __init__ via __new__, only
setting the nx/ny/nz attributes _update_positions/within_bounds actually
touch) and fake source-like items exposing .coord/.coordorigin, matching
the established pattern for isolated grid-method tests in this test
suite.
"""
import numpy as np
import pytest

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid


class _FakeItem:
    def __init__(self, coord):
        self.coord = np.array(coord, dtype=np.int32)
        self.coordorigin = self.coord.copy()


def _make_grid(nx=100, ny=100, nz=100):
    grid = FDTDGrid.__new__(FDTDGrid)
    grid.size = np.array([nx, ny, nz], dtype=np.int32)
    return grid


@pytest.fixture(autouse=True)
def _fake_sim_config(monkeypatch):
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.model_end = 5  # models run 0..4 (5 models, n=5)
    config.sim_config.model_start = 0  # no restart (see test_restart_stepping_bounds_check.py)


def test_all_negative_step_is_applied_not_ignored():
    grid = _make_grid()
    item = _FakeItem((50, 50, 50))
    step_size = np.array([-1, -1, -1], dtype=np.int32)

    # step_number == 0 is the bounds-check-only call (matches real usage
    # in update_simple_source_positions/update_receiver_positions).
    grid._update_positions([item], step_size, 0)
    assert tuple(item.coord) == (50, 50, 50)  # unchanged on step 0

    grid._update_positions([item], step_size, 3)
    assert tuple(item.coord) == (47, 47, 47)  # coordorigin + 3 * (-1)


def test_all_zero_step_is_genuinely_a_no_op():
    grid = _make_grid()
    item = _FakeItem((50, 50, 50))
    step_size = np.array([0, 0, 0], dtype=np.int32)

    grid._update_positions([item], step_size, 3)
    assert tuple(item.coord) == (50, 50, 50)


def test_bounds_check_uses_last_model_actually_run_not_model_end():
    # model_end=5 means models 0..4 run; the last step multiplier used
    # in practice is 4, not 5. A step size that lands exactly on the
    # boundary at step 4 (but would overshoot at step 5) must be
    # accepted, not rejected.
    grid = _make_grid(nx=100, ny=100, nz=100)
    item = _FakeItem((80, 50, 50))
    # 80 + 5*4 = 100 (== nx, within bounds); 80 + 5*5 = 105 (> nx, would
    # have been rejected by the old off-by-one check).
    step_size = np.array([5, 0, 0], dtype=np.int32)

    grid._update_positions([item], step_size, 0)  # should not raise
