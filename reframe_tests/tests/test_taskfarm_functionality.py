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
