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

import reframe as rfm

from reframe_tests.tests.mixins import TaskfarmMixin
from reframe_tests.tests.test_example_models import TestBscan

"""Reframe regression tests for taskfarm functionality
"""


@rfm.simple_test
class TestSingleNodeTaskfarm(TaskfarmMixin, TestBscan):
    tags = {
        "test",
        "mpi",
        "taskfarm",
        "steps",
        "waveform",
        "hertzian_dipole",
        "material",
        "box",
        "cylinder",
    }
    num_tasks = 8
    num_tasks_per_node = 8
    test_dependency = TestBscan


@rfm.simple_test
class TestMultiNodeTaskfarm(TaskfarmMixin, TestBscan):
    tags = {
        "test",
        "mpi",
        "taskfarm",
        "steps",
        "waveform",
        "hertzian_dipole",
        "material",
        "box",
        "cylinder",
    }
    num_tasks = 32
    num_tasks_per_node = 8
    test_dependency = TestBscan
