import reframe as rfm
from reframe.core.builtins import parameter

from reframe_tests.tests.base_tests import GprMaxMPIRegressionTest, GprMaxRegressionTest

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
class TestHertzianDipoleSourceMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "hertzian_dipole", "waveform"}
    mpi_layout = parameter([[3, 3, 3]])
    serial_dependency = TestHertzianDipoleSource
    model = serial_dependency.model


@rfm.simple_test
class TestMagneticDipoleSourceMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "magnetic_dipole", "waveform"}
    mpi_layout = parameter([[3, 3, 3]])
    serial_dependency = TestMagneticDipoleSource
    model = serial_dependency.model


@rfm.simple_test
class TestTransmissionLineSourceMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "transmission_line", "waveform"}
    mpi_layout = parameter([[3, 3, 3]])
    serial_dependency = TestTransmissionLineSource
    model = serial_dependency.model
