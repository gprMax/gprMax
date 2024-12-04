from pathlib import Path
from typing import Optional

import reframe.utility.typecheck as typ
from numpy import prod
from reframe import RegressionMixin
from reframe.core.builtins import parameter, required, run_after, variable
from typing_extensions import TYPE_CHECKING

from reframe_tests.tests.base_tests import GprMaxBaseTest
from reframe_tests.tests.regression_checks import (
    GeometryViewRegressionCheck,
    ReceiverRegressionCheck,
    RegressionCheck,
    SnapshotRegressionCheck,
)

# If using a static type checker, inherit from GprMaxBaseTest as the
# Mixin classes should always have access to resources from that class.
# However, during execution inherit from RegressionMixin.
if TYPE_CHECKING:
    GprMaxMixin = GprMaxBaseTest
else:
    GprMaxMixin = RegressionMixin


class ReceiverMixin(GprMaxMixin):
    number_of_receivers = variable(int, value=-1)

    @run_after("setup")
    def add_receiver_regression_checks(self):
        reference_file = self.build_reference_filepath(self.output_file)

        if self.number_of_receivers > 0:
            for i in range(self.number_of_receivers):
                regression_check = ReceiverRegressionCheck(
                    self.output_file, reference_file, f"r{i}"
                )
                self.regression_checks.append(regression_check)
        else:
            regression_check = RegressionCheck(self.output_file, reference_file)
            self.regression_checks.append(regression_check)


class SnapshotMixin(GprMaxMixin):
    """Add regression tests for snapshots.

    Attributes:
        snapshots (list[str]): List of snapshots to run regression
            checks on.
    """

    snapshots = variable(typ.List[str], value=[])

    def build_snapshot_filepath(self, snapshot: str) -> Path:
        """Build filepath to the specified snapshot.

        Args:
            snapshot: Name of the snapshot.
        """
        return Path(f"{self.model}_snaps", snapshot).with_suffix(".h5")

    @run_after("setup")
    def add_snapshot_regression_checks(self):
        """Add a regression check for each snapshot.

        The test will be skipped if no snapshots have been specified.
        """
        self.skip_if(
            len(self.snapshots) < 0,
            f"Must provide a list of snapshots.",
        )

        for snapshot in self.snapshots:
            snapshot_file = self.build_snapshot_filepath(snapshot)
            reference_file = self.build_reference_filepath(snapshot)
            regression_check = SnapshotRegressionCheck(snapshot_file, reference_file)
            self.regression_checks.append(regression_check)


class GeometryOnlyMixin(GprMaxMixin):
    """Run test with geometry only flag"""

    @run_after("setup")
    def add_geometry_only_flag(self):
        self.executable_opts += ["--geometry-only"]


class GeometryViewMixin(GprMaxMixin):
    """Add regression tests for geometry views.

    Attributes:
        geometry_views (list[str]): List of geometry views to run
            regression checks on.
    """

    geometry_views = variable(typ.List[str], value=[])

    def build_geometry_view_filepath(self, geometry_view: str) -> Path:
        """Build filepath to the specified geometry view.

        Args:
            geometry_view: Name of the geometry view.
        """
        return Path(geometry_view).with_suffix(".vtkhdf")

    @run_after("setup")
    def add_geometry_view_regression_checks(self):
        """Add a regression check for each geometry view.

        The test will be skipped if no geometry views have been specified.
        """
        self.skip_if(
            len(self.geometry_views) < 0,
            f"Must provide a list of geometry views.",
        )

        for geometry_view in self.geometry_views:
            geometry_view_file = self.build_geometry_view_filepath(geometry_view)
            reference_file = self.build_reference_filepath(geometry_view, ".vtkhdf")
            regression_check = GeometryViewRegressionCheck(geometry_view_file, reference_file)
            self.regression_checks.append(regression_check)


class PythonApiMixin(GprMaxMixin):
    """Use the GprMax Python API rather than a standard input file."""

    @run_after("setup")
    def use_python_input_file(self):
        """Input files for API tests will be python files."""
        self.executable = "time -p python"
        self.input_file = self.input_file.with_suffix(".py")


class MpiMixin(GprMaxMixin):
    """Run test using GprMax MPI functionality.

    Attributes:
        mpi_layout (parameter[list[int]]): ReFrame parameter to specify
            how MPI tasks should be arranged.
    """

    mpi_layout = parameter()

    @run_after("setup")
    def configure_mpi_tasks(self):
        """Set num_tasks and add MPI specific commandline arguments."""
        self.num_tasks = int(prod(self.mpi_layout))
        self.executable_opts += ["--mpi", *map(str, self.mpi_layout)]


class BScanMixin(GprMaxMixin):
    """Test a B-scan model - a model with a moving source and receiver.

    Attributes:
        num_models (parameter[int]): Number of models to run.
    """

    num_models = parameter()

    @run_after("setup")
    def setup_bscan_test(self):
        """Add B-scan specific commandline arguments and postrun cmds.

        Set the number of models to run, and merge the output files.
        """
        self.executable_opts += ["-n", str(self.num_models)]

        self.postrun_cmds += [
            f"python -m toolboxes.Utilities.outputfiles_merge {self.model}",
            f"mv {self.model}_merged.h5 {self.output_file}",
        ]

    def get_test_dependency_variant_name(self, **kwargs) -> Optional[str]:
        """Get unique ReFrame name of the test dependency variant.

        By default, filter test dependencies by the model name and the
        number of models.

        Args:
            **kwargs: Additional key-value pairs to filter the parameter
                space of the test dependency. The key is the test
                parameter name and the value is either a single value or
                a unary function that evaluates to True if the parameter
                point must be kept, False otherwise.

        Returns:
            variant_name: Unique name of the test dependency variant.
        """

        kwargs.setdefault("num_models", self.num_models)
        return super().get_test_dependency_variant_name(**kwargs)


class TaskfarmMixin(GprMaxMixin):
    """Run test using GprMax taskfarm functionality."""

    # TODO: Make this a required variabe, or create a new variable to
    # proxy it.
    # num_tasks = required

    @run_after("setup")
    def add_taskfarm_flag(self):
        """Add taskfarm specific commandline arguments."""
        self.executable_opts += ["--taskfarm"]


class AntennaModelMixin(GprMaxMixin):
    """Test an antenna model."""

    pass
