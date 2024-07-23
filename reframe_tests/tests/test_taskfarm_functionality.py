import reframe as rfm

from reframe_tests.tests.base_tests import GprMaxTaskfarmRegressionTest
from reframe_tests.tests.test_example_models import TestBscan

"""Reframe regression tests for taskfarm functionality
"""


@rfm.simple_test
class TestSingleNodeTaskfarm(GprMaxTaskfarmRegressionTest):
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
    serial_dependency = TestBscan
    model = serial_dependency.model
    num_models = serial_dependency.num_models


@rfm.simple_test
class TestMultiNodeTaskfarm(GprMaxTaskfarmRegressionTest):
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
    serial_dependency = TestBscan
    model = serial_dependency.model
    num_models = serial_dependency.num_models
