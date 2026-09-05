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

"""Regression tests for best-effort process-memory reporting."""

from types import SimpleNamespace

import psutil
import pytest

from gprMax.model import _process_memory_bytes


class _USSAccessDenied:
    def memory_full_info(self):
        raise psutil.AccessDenied(pid=123)

    def memory_info(self):
        return SimpleNamespace(rss=456)


class _AllAccessDenied(_USSAccessDenied):
    def memory_info(self):
        raise psutil.AccessDenied(pid=123)


def test_process_memory_falls_back_to_rss_when_uss_is_denied():
    assert _process_memory_bytes(_USSAccessDenied()) == 456


def test_process_memory_is_optional_when_all_queries_are_denied():
    assert _process_memory_bytes(_AllAccessDenied()) is None


pytestmark = pytest.mark.unit
