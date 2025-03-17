import reframe as rfm
from reframe.core.builtins import parameter

from reframe_tests.tests.mixins import BScanMixin, MpiMixin
from reframe_tests.tests.standard_tests import GprMaxRegressionTest

"""Reframe regression tests for example models in gprMax documentation
"""


@rfm.simple_test
class TestAscan(GprMaxRegressionTest):
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
class TestAscanMPI(MpiMixin, TestAscan):
    tags = {
        "test",
        "mpi",
        "ascan",
        "2d",
        "hertzian_dipole",
        "waveform",
        "material",
        "box",
        "cylinder",
    }
    mpi_layout = parameter([[2, 2, 1]])
    test_dependency = TestAscan


@rfm.simple_test
class TestBscan(BScanMixin, GprMaxRegressionTest):
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


@rfm.simple_test
class TestBscanMPI(MpiMixin, TestBscan):
    tags = {
        "test",
        "mpi",
        "bscan",
        "steps",
        "waveform",
        "hertzian_dipole",
        "material",
        "box",
        "cylinder",
    }
    mpi_layout = parameter([[2, 2, 1]])
    test_dependency = TestBscan
