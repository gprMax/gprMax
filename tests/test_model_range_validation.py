"""Regression tests for SimulationConfig._set_model_start_end() (Codex-reported):

1. No validation rejected n<=0 - a non-positive number of model runs
   silently produced an empty model_range, and the simulation reported
   normal completion having run zero models.

2. `if self.args.i:` was a truthy check, not `is not None` - so i=0 was
   treated identically to "i not given at all" (None), even though `i`
   is documented as a 1-indexed "model number to start/restart from"
   (so 0 is itself an invalid index, but should be rejected with a clear
   error, not silently ignored).
"""
import argparse

import pytest

import gprMax.config as config
from gprMax import gprMax as gprmax_mod


def _make_args(**overrides):
    args = argparse.Namespace(**gprmax_mod.args_defaults)
    args.inputfile = "test.in"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_zero_n_rejected():
    with pytest.raises(ValueError):
        config.SimulationConfig(_make_args(n=0))


def test_negative_n_rejected():
    with pytest.raises(ValueError):
        config.SimulationConfig(_make_args(n=-3))


def test_zero_i_rejected_not_silently_ignored():
    with pytest.raises(ValueError):
        config.SimulationConfig(_make_args(n=1, i=0))


def test_negative_i_rejected():
    with pytest.raises(ValueError):
        config.SimulationConfig(_make_args(n=1, i=-2))


def test_valid_n_and_i_still_work():
    sim_config = config.SimulationConfig(_make_args(n=5, i=10))
    # i=10 (1-indexed) -> modelstart=9, modelend=9+5=14
    assert sim_config.model_start == 9
    assert sim_config.model_end == 14


def test_n_without_i_still_works():
    sim_config = config.SimulationConfig(_make_args(n=3))
    assert sim_config.model_start == 0
    assert sim_config.model_end == 3
