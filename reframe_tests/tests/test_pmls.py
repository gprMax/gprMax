import reframe as rfm
from reframe.core.builtins import parameter

from reframe_tests.tests.mixins import MpiMixin
from reframe_tests.tests.standard_tests import GprMaxRegressionTest

"""Reframe regression tests for models defining geometry
"""


@rfm.simple_test
class TestSingleCellPml(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "box", "pml"}
    sourcesdir = "src/pml_tests"
    model = parameter(["single_cell_pml_2d"])


"""Test MPI Functionality
"""


@rfm.simple_test
class TestSingleCellPmlMpi(MpiMixin, TestSingleCellPml):
    tags = {"test", "mpi", "geometery", "box", "pml"}
    mpi_layout = parameter([[2, 2, 1], [3, 3, 1]])
    test_dependency = TestSingleCellPml
