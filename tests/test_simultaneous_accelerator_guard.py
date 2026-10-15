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

"""Regression test for gprMax/config.py's simultaneous-accelerator guard
(Codex-reported, "Medium"): SimulationConfig.__init__ used to check
`[self.args.gpu, self.args.opencl, self.args.metal].count(True)` to
reject requesting more than one non-CPU solver at once. `list.count()`
uses equality, and a list is never `== True`, so this only ever caught
the bool form (gpu=True, opencl=True, ...) - list forms like
gpu=[[0]], opencl=[[0]] (both non-None, i.e. both genuinely requested)
slipped straight past the guard. Both solver-selection branches later in
__init__ then ran, and whichever was checked last (opencl, then metal)
silently overrode the earlier one with no error raised at all.

Fixed by extracting the check into config._multiple_accelerators_requested(),
which counts arguments that are not None instead of counting values equal
to True - matching how "is this solver requested" is determined
everywhere else in this class (e.g.
`if self.gpu is not None: self.general["solver"] = "cuda"`).

The unit tests below exercise the extracted helper directly (no need to
construct a full SimulationConfig/detect real hardware). The constructor
tests use the application's argument defaults and confirm that every pair
of accelerators is rejected by this guard before hardware discovery.
"""
import argparse

import pytest

from gprMax import config
from gprMax.config import SimulationConfig, _multiple_accelerators_requested
from gprMax.gprMax import args_defaults


@pytest.mark.parametrize(
    "non_cpu_solvers,expected",
    [
        ([None, None, None], False),
        ([True, None, None], False),
        ([[[0]], None, None], False),
        ([True, True, None], True),
        ([[[0]], [[0]], None], True),
        ([True, None, [[0]]], True),
        # gpu=False/opencl=False still count as "requested" under "is not
        # None" semantics - matches the pre-existing solver-selection
        # gate elsewhere in this class (`if self.gpu is not None: ...`),
        # which also treats False the same as True. Not this guard's
        # concern to distinguish True from False, only None from
        # not-None.
        ([False, False, None], True),
    ],
)
def test_multiple_accelerators_requested(non_cpu_solvers, expected):
    assert _multiple_accelerators_requested(non_cpu_solvers) is expected


def _make_args(**overrides):
    # Match the API/CLI contract, including restart and Scene defaults, without
    # mutating the application's shared defaults or duplicating its argument list.
    defaults = dict(args_defaults, inputfile="model.in", outputfile="out")
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.parametrize("solvers", [("gpu", "opencl"), ("gpu", "metal"), ("opencl", "metal")])
@pytest.mark.parametrize(
    "values",
    [
        pytest.param((True, True), id="bool"),
        pytest.param(([[0]], [[0]]), id="list"),
        pytest.param((True, [[0]]), id="bool-list"),
        pytest.param(([[0]], True), id="list-bool"),
        pytest.param((False, False), id="false-is-not-none"),
    ],
)
def test_two_solvers_rejected_via_real_init(solvers, values, monkeypatch, caplog):
    def unexpected_hardware_probe(*args, **kwargs):
        pytest.fail("Conflicting accelerators must be rejected before hardware discovery")

    for probe in ("get_host_info", "detect_cuda_gpus", "detect_opencl", "detect_metal"):
        monkeypatch.setattr(config, probe, unexpected_hardware_probe)

    with caplog.at_level("ERROR", logger="gprMax.config"):
        with pytest.raises(ValueError):
            SimulationConfig(_make_args(**dict(zip(solvers, values))))

    # Another validation failure must not accidentally make this test pass.
    assert "You cannot use combinations of CUDA, OpenCl and Apple Metal solvers simultaneously." in caplog.text
