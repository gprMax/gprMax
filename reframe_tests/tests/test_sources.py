import reframe as rfm
from reframe.core.builtins import parameter

from reframe_tests.tests.mixins import MpiMixin
from reframe_tests.tests.standard_tests import GprMaxRegressionTest

"""Reframe regression tests for each gprMax source
"""


@rfm.simple_test
class TestHertzianDipoleSource(GprMaxRegressionTest):
    tags = {"test", "serial", "hertzian_dipole", "waveform"}
    sourcesdir = "src/source_tests"
    model = parameter(["hertzian_dipole_fs"])


@rfm.simple_test
class TestMagneticDipoleSource(GprMaxRegressionTest):
    tags = {"test", "serial", "magnetic_dipole", "waveform"}
    sourcesdir = "src/source_tests"
    model = parameter(["magnetic_dipole_fs"])


@rfm.simple_test
class TestTransmissionLineSource(GprMaxRegressionTest):
    tags = {"test", "serial", "transmission_line", "waveform"}
    sourcesdir = "src/source_tests"
    model = parameter(["transmission_line_fs"])


"""Test MPI Functionality
"""


@rfm.simple_test
class TestHertzianDipoleSourceMpi(MpiMixin, TestHertzianDipoleSource):
    tags = {"test", "mpi", "hertzian_dipole", "waveform"}
    mpi_layout = parameter([[3, 3, 3]])
    test_dependency = "TestHertzianDipoleSource"


@rfm.simple_test
class TestMagneticDipoleSourceMpi(MpiMixin, TestMagneticDipoleSource):
    tags = {"test", "mpi", "magnetic_dipole", "waveform"}
    mpi_layout = parameter([[3, 3, 3]])
    test_dependency = "TestMagneticDipoleSource"


@rfm.simple_test
class TestTransmissionLineSourceMpi(MpiMixin, TestTransmissionLineSource):
    tags = {"test", "mpi", "transmission_line", "waveform"}
    mpi_layout = parameter([[3, 3, 3]])
    test_dependency = "TestTransmissionLineSource"
