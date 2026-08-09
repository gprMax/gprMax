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
construct a full SimulationConfig/detect real hardware). The two
integration tests build a real SimulationConfig with a minimal fake args
Namespace, confirming the guard actually fires from __init__ before any
of the rest of the (much larger) constructor runs.
"""
from types import SimpleNamespace

import pytest

from gprMax.config import SimulationConfig, _multiple_accelerators_requested


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
    defaults = dict(
        geometry_fixed=False,
        geometry_only=False,
        gpu=None,
        mpi=None,
        n=1,
        opencl=None,
        metal=None,
        outputfile="out",
        taskfarm=False,
        write_processed=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_two_bool_solvers_rejected_via_real_init():
    with pytest.raises(ValueError):
        SimulationConfig(_make_args(gpu=True, opencl=True))


def test_two_list_solvers_rejected_via_real_init():
    with pytest.raises(ValueError):
        SimulationConfig(_make_args(gpu=[[0]], opencl=[[0]]))
