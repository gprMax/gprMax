import reframe as rfm
from reframe.core.builtins import parameter

from reframe_tests.tests.base_tests import GprMaxBScanRegressionTest, GprMaxRegressionTest
from reframe_tests.tests.mixins import MpiMixin, ReceiverMixin

"""Reframe regression tests for example models in gprMax documentation
"""


@rfm.simple_test
class TestAscan(ReceiverMixin, GprMaxRegressionTest):
    tags = {
        "test",
        "serial",
        "ascan",
        "2d",
        "hertzian_dipole",
        "waveform",
        "material",
        "box",
        "cylinder",
    }
    sourcesdir = "src/example_models"
    model = parameter(["cylinder_Ascan_2D"])


@rfm.simple_test
class TestBscan(GprMaxBScanRegressionTest):
    tags = {
        "test",
        "serial",
        "bscan",
        "steps",
        "waveform",
        "hertzian_dipole",
        "material",
        "box",
        "cylinder",
    }
    sourcesdir = "src/bscan_tests"
    model = parameter(["cylinder_Bscan_2D"])
    num_models = parameter([64])
