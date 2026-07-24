"""Regression test for gprMax/config.py's device-ID selection
(Codex-reported, "Medium"): ModelConfig.__init__ assumed
args.gpu/args.opencl/args.metal was always an iterable nested list,
which is only true for the CLI's action="append" + nargs="*" shape
(e.g. `-gpu 1` -> [[1]]). The documented Python API accepts "list/bool"
(gprMax.py's help text), which breaks the old logic three ways:

1. gpu=True: `any(isinstance(element, list) for element in devs)`
   iterates over a bool -> TypeError.
2. gpu=[1] (a flat list, exactly as the API docs describe): every
   element is an int, so `any(isinstance(element, list) ...)` is False
   -> the flattening branch is skipped -> `deviceID` is never assigned
   -> `deviceID = deviceID[0]` raises NameError -> silently swallowed by
   a bare `except:` -> falls through to deviceID=0, silently ignoring
   the user's requested device.
3. Only the CLI's doubly-nested form (e.g. [[1]]) worked correctly.

Fixed by extracting a small, independently-testable
config._resolve_device_id(devs) helper that normalises all of these
shapes (None/bool, flat list, list-of-lists) into a single deviceID,
used by ModelConfig.__init__ instead of the inline try/except.
"""
import pytest

from gprMax.config import _resolve_device_id


@pytest.mark.parametrize(
    "devs,expected",
    [
        (None, 0),
        (True, 0),
        (False, 0),
        ([1], 1),
        ([2], 2),
        ([1, 2], 1),
        ([[1]], 1),
        ([[1], [2]], 1),
        ([[]], 0),
        ([], 0),
    ],
)
def test_resolve_device_id(devs, expected):
    assert _resolve_device_id(devs) == expected
