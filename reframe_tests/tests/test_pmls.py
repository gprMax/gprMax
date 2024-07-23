import reframe as rfm
from reframe.core.builtins import parameter

from reframe_tests.tests.base_tests import GprMaxMPIRegressionTest, GprMaxRegressionTest

"""Reframe regression tests for models defining geometry
"""


@rfm.simple_test
class TestSingleCellPml(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "box", "pml"}
    sourcesdir = "src/pml_tests"
    model = parameter(["single_cell_pml_2d"])
    rx_outputs = ["Hx"]


"""Test MPI Functionality
"""


@rfm.simple_test
class TestSingleCellPmlMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "geometery", "box", "pml"}
    mpi_layout = parameter([[2, 2, 1], [3, 3, 1]])
    serial_dependency = TestSingleCellPml
    model = serial_dependency.model
    rx_outputs = ["Hx"]
