"""ReFrame base classes for GprMax tests

Usage (run all tests):
    cd gprMax/reframe_tests
    reframe -C configuration/{CONFIG_FILE} -c tests/ -r
"""

import os
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ
from reframe import RunOnlyRegressionTest, simple_test
from reframe.core.builtins import (
    parameter,
    performance_function,
    require_deps,
    required,
    run_after,
    run_before,
    sanity_function,
    variable,
)
from reframe.core.exceptions import DependencyError
from reframe.utility import osext, udeps

from reframe_tests.tests.regression_checks import RegressionCheck
from reframe_tests.utilities.deferrable import path_join

GPRMAX_ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
PATH_TO_PYENV = os.path.join(".venv", "bin", "activate")


@simple_test
class CreatePyenvTest(RunOnlyRegressionTest):
    """Create a fresh virtual environment for running the tests.

    The test checks for any errors from pip installing gprMax and its
    dependencies.
    """

    valid_systems = ["generic", "archer2:login"]
    valid_prog_environs = ["builtin", "PrgEnv-gnu"]
    modules = ["cray-python"]
    time_limit = "20m"

    prerun_cmds = [
        "python -m venv --system-site-packages --prompt gprMax .venv",
        f"source {PATH_TO_PYENV}",
        "CC=cc CXX=CC FC=ftn python -m pip install --upgrade pip",
        f"CC=cc CXX=CC FC=ftn python -m pip install -r {os.path.join(GPRMAX_ROOT_DIR, 'requirements.txt')}",
    ]
    executable = f"CC=cc CXX=CC FC=ftn python -m pip install -e {GPRMAX_ROOT_DIR}"

    @run_after("init")
    def install_system_specific_dependencies(self):
        """Install additional dependencies for specific systems."""
        if self.current_system.name == "archer2":
            """
            Needed to prevent a pip install error.
            dask 2022.2.1 (installed) requires cloudpickle>=1.1.1, which
            is not installed and is missed by the pip dependency checks.

            Not necessary for gprMax, but any error message is picked up
            by the sanity checks.
            """
            self.prerun_cmds.insert(3, "CC=cc CXX=CC FC=ftn python -m pip install cloudpickle")

            """
            A default pip install of h5py does not have MPI support so
            it needs to built as described here:
            https://docs.h5py.org/en/stable/mpi.html#building-against-parallel-hdf5
            """
            self.modules.append("cray-hdf5-parallel")
            self.prerun_cmds.insert(
                4, "CC=mpicc HDF5_MPI='ON' python -m pip install --no-binary=h5py h5py"
            )

    @sanity_function
    def check_requirements_installed(self):
        """Check packages were successfully installed.

        Check pip is up to date and gprMax dependencies from
        requirements.txt were successfully installed. Check gprMax was
        installed successfully and no other errors were thrown.
        """
        return (
            sn.assert_found(
                r"(Successfully installed pip)|(Requirement already satisfied: pip.*\n(?!Collecting pip))",
                self.stdout,
                "Failed to update pip",
            )
            and sn.assert_found(
                r"Successfully installed (?!(gprMax)|(pip))",
                self.stdout,
                "Failed to install requirements",
            )
            and sn.assert_found(
                r"Successfully installed gprMax", self.stdout, "Failed to install gprMax"
            )
            and sn.assert_not_found(r"finished with status 'error'", self.stdout)
            and sn.assert_not_found(r"(ERROR|error):", self.stderr)
        )


class GprMaxBaseTest(RunOnlyRegressionTest):
    """Base class that all GprMax tests should inherit from.

    Test functionality can be augmented by using Mixin classes.

    Attributes:
        model (parameter[str]): ReFrame parameter to specify the model
            name(s).
        sourcesdir (str): Relative path to the test's src directory.
        regression_checks (list[RegressionCheck]): List of regression
            checks to perform.
        test_dependency (type[GprMaxBaseTest] | None): Optional test
            dependency. If specified, regression checks will use
            reference files created by the test dependency.
    """

    valid_systems = ["archer2:compute"]
    valid_prog_environs = ["PrgEnv-gnu"]
    modules = ["cray-python"]
    time_limit = "10m"

    num_cpus_per_task = 16
    exclusive_access = True

    model = parameter()
    sourcesdir = required
    executable = "python -m gprMax"

    regression_checks = variable(typ.List[RegressionCheck], value=[])

    # TODO: Make this a ReFrame variable
    # Not currently possible as ReFrame does not think an object of type
    # reframe.core.meta.RegressionTestMeta is copyable, and so ReFrame
    # test classes cannot be specified in a variable.
    test_dependency: Optional[type["GprMaxBaseTest"]] = None
    # test_dependency = variable(type(None), type, value=None)

    def get_test_dependency_variant_name(self, **kwargs) -> Optional[str]:
        """Get unique ReFrame name of the test dependency variant.

        By default, filter test dependencies by the model name.

        Args:
            **kwargs: Additional key-value pairs to filter the parameter
                space of the test dependency. The key is the test
                parameter name and the value is either a single value or
                a unary function that evaluates to True if the parameter
                point must be kept, False otherwise.

        Returns:
            variant_name: Unique name of the test dependency variant.
        """
        if self.test_dependency is None:
            return None

        # Always filter by the model parameter (unless the test
        # dependency only runs a single model), but allow child classes
        # (or mixins) to override how models are filtered.
        if len(self.test_dependency.model.values) > 1:
            kwargs.setdefault("model", self.model)

        variant_nums = self.test_dependency.get_variant_nums(**kwargs)

        if len(variant_nums) < 1:
            raise DependencyError(
                f"No variant of '{self.test_dependency.__name__}' meets conditions: {kwargs}",
            )

        return self.test_dependency.variant_name(variant_nums[0])

    def get_test_dependency(self) -> Optional["GprMaxBaseTest"]:
        """Get correct ReFrame test case from the test dependency.

        Returns:
            test_case: ReFrame test case.
        """
        variant = self.get_test_dependency_variant_name()
        if variant is None:
            return None
        else:
            return self.getdep(variant)

    def build_reference_filepath(self, name: Union[str, os.PathLike], suffix: str = ".h5") -> Path:
        """Build path to the specified reference file.

        Reference files are saved in directories per test case. If this
        test does not specify a test dependency, it will save and manage
        its own reference files in its own directory. Otherwise, it will
        use reference files saved by its test dependency.

        Args:
            name: Name of the file.
            suffix: File extension. Default ".h5".

        Returns:
            filepath: Absolute path to the reference file.
        """
        target = self.get_test_dependency()
        if target is None:
            reference_dir = self.short_name
        else:
            reference_dir = target.short_name

        reference_file = Path("regression_checks", reference_dir, name).with_suffix(suffix)
        return reference_file.absolute()

    # TODO: Change CreatePyenvTest to a fixture instead of a test dependency
    @run_after("init")
    def inject_dependencies(self):
        """Specify test dependencies.

        All tests depend on the Python virtual environment building
        correctly and their own test dependency if specified.
        """
        self.depends_on("CreatePyenvTest", udeps.by_env)
        if self.test_dependency is not None:
            variant = self.get_test_dependency_variant_name()
            self.depends_on(variant, udeps.by_env)

    @require_deps
    def get_pyenv_path(self, CreatePyenvTest):
        """Add prerun command to load the built Python environment."""
        path_to_pyenv = os.path.join(CreatePyenvTest(part="login").stagedir, PATH_TO_PYENV)
        self.prerun_cmds.append(f"source {path_to_pyenv}")

    @run_after("init")
    def setup_env_vars(self):
        """Set necessary environment variables.

        Set OMP_NUM_THREADS environment variable from num_cpus_per_task
        and other system specific varaibles.
        """
        self.env_vars["OMP_NUM_THREADS"] = self.num_cpus_per_task

        if self.current_system.name == "archer2":
            # Avoid inheriting slurm memory environment variables from any previous slurm job (i.e. the reframe job)
            self.prerun_cmds.append("unset SLURM_MEM_PER_NODE")
            self.prerun_cmds.append("unset SLURM_MEM_PER_CPU")

            # Set the matplotlib cache to the work filesystem
            self.env_vars["MPLCONFIGDIR"] = "${HOME/home/work}/.config/matplotlib"

    def build_output_file_path(self, filename: str) -> Path:
        """Build output file Path object from filename.

        Using a function to build this allows mixins to reuse or
        override it if needed.

        Args:
            filename: Name of output file with no file extension.

        Returns:
            Path: Output file Path object
        """
        return Path(f"{filename}.h5")

    @run_after("init")
    def set_file_paths(self):
        """Set default test input and output files.

        These are set in a post-init hook to allow mixins to use them
        later in the pipeline.
        """
        self.input_file = Path(f"{self.model}.in")
        self.output_file = self.build_output_file_path(self.model)

    @run_before("run")
    def configure_test_run(self):
        """Configure gprMax commandline arguments and files to keep."""
        input_file = str(self.input_file)
        output_file = str(self.output_file)

        self.executable_opts += [
            input_file,
            "-o",
            output_file,
            "--log-level",
            "10",
            "--hide-progress-bars",
        ]

        regression_output_files = [str(r.output_file) for r in self.regression_checks]
        self.keep_files += [input_file, output_file, *regression_output_files]

        """
        if self.has_receiver_output:
            self.postrun_cmds = [
                f"python -m reframe_tests.utilities.plotting {self.output_file} {self.reference_file} -m {self.model}"
            ]
            self.keep_files += [self.output_file, f"{self.model}.pdf"]

        if self.is_antenna_model:
            self.postrun_cmds = [
                f"python -m toolboxes.Plotting.plot_antenna_params -save {self.output_file}"
            ]

            antenna_t1_params = f"{self.model}_t1_params.pdf"
            antenna_ant_params = f"{self.model}_ant_params.pdf"
            self.keep_files += [
                antenna_t1_params,
                antenna_ant_params,
            ]
        """

    @run_before("run")
    def combine_task_outputs(self):
        """Split output from each MPI rank.

        If running with multiple MPI ranks, split the output into
        seperate files and add postrun commands to combine the files
        after the simulation has run.
        """
        if self.num_tasks > 1:
            stdout = self.stdout.evaluate().split(".")[0]
            stderr = self.stderr.evaluate().split(".")[0]
            self.prerun_cmds.append(f"mkdir out")
            self.prerun_cmds.append(f"mkdir err")
            self.job.launcher.options = [
                f"--output=out/{stdout}_%t.out",
                f"--error=err/{stderr}_%t.err",
            ]
            self.executable_opts += ["--log-all-ranks"]
            self.postrun_cmds.append(f"cat out/{stdout}_*.out >> {self.stdout}")
            self.postrun_cmds.append(f"cat err/{stderr}_*.err >> {self.stderr}")

    def test_simulation_complete(self) -> Literal[True]:
        """Check simulation completed successfully.

        Returns:
            simulation_completed: Returns True if the simulation
                completed, otherwise it fails the test.

        Raises:
            reframe.core.exceptions.SanityError: If the simulation did
                not complete.
        """
        return sn.assert_not_found(
            r"(?i)error",
            self.stderr,
            f"An error occured. See '{path_join(self.stagedir, self.stderr)}' for details.",
        ) and sn.assert_found(
            r"=== Simulation completed in ", self.stdout, "Simulation did not complete"
        )

    def test_reference_files_exist(self) -> Literal[True]:
        """Check all reference files exist and create any missing ones.

        Returns:
            files_exist: Returns True if all reference files exist,
                otherwise it fails the test.

        Raises:
            reframe.core.exceptions.SanityError: If any reference files
                do not exist.
        """

        # Store error messages so all references files can be checked
        # (and created if necessary) before the test is failed.
        error_messages = []
        for check in self.regression_checks:
            if not check.reference_file_exists():
                if self.test_dependency is None and check.create_reference_file():
                    error_messages.append(
                        f"Reference file does not exist. Creating... '{check.reference_file}'"
                    )
                elif self.test_dependency is not None:
                    error_messages.append(
                        f"ERROR: Test dependency did not create reference file: '{check.reference_file}'"
                    )
                else:
                    error_messages.append(
                        f"ERROR: Unable to create reference file: '{check.reference_file}'"
                    )
        return sn.assert_true(len(error_messages) < 1, "\n".join(error_messages))

    @sanity_function
    def regression_check(self) -> bool:
        """Run sanity checks and regression checks.

        Checks will run in the following order:
        - Check the simulation completed.
        - Check all reference files exist.
        - Run all regression checks.

        If any of these checks fail, the test will fail and none of the
        other later checks will run.

        Returns:
            test_passed: Returns True if all checks pass.

        Raises:
            reframe.core.exceptions.SanityError: If any regression
                checks fail.
        """

        return (
            self.test_simulation_complete()
            and self.test_reference_files_exist()
            and sn.all(sn.map(lambda check: check.run(), self.regression_checks))
        )

    @performance_function("s", perf_key="run_time")
    def extract_run_time(self):
        """Extract total runtime from SLURM."""
        sactt_command = osext.run_command(
            [
                "sacct",
                "--format=JobID,JobName,State,Elapsed",
                "-j",
                self.job.jobid,
            ]
        )
        hours, minutes, seconds = sn.extractsingle_s(
            self.job.jobid
            + r"\.0\s+python\s+COMPLETED\s+(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)",
            sactt_command.stdout,
            ["hours", "minutes", "seconds"],
            int,
        )

        return hours * 3600 + minutes * 60 + seconds

    @performance_function("s", perf_key="simulation_time")
    def extract_simulation_time(self):
        """Extract average simulation time reported by gprMax."""
        return sn.round(self.extract_simulation_time_per_rank().sum() / self.num_tasks, 2)

    # @performance_function("s", perf_key="max_simulation_time")
    # def extract_max_simulation_time(self):
    #     """Extract maximum simulation time reported by gprMax."""
    #     return sn.round(self.extract_simulation_time_per_rank().max(), 2)

    # @performance_function("s", perf_key="min_simulation_time")
    # def extract_min_simulation_time(self):
    #     """Extract minimum simulation time reported by gprMax."""
    #     return sn.round(self.extract_simulation_time_per_rank().min(), 2)

    # @performance_function("s", perf_key="wall_time")
    # def extract_wall_time(self):
    #     """Extract total simulation time reported by gprMax."""
    #     return sn.round(self.extract_simulation_time_per_rank().sum(), 2)

    def extract_simulation_time_per_rank(self) -> npt.NDArray[np.float64]:
        """Extract simulation time reported by gprMax from each rank.

        Raises:
            ValueError: Raised if not all ranks report the simulation
                time.

        Returns:
            simulation_times: Simulation time for each rank in seconds.
        """
        simulation_time = sn.extractall(
            r"=== Simulation completed in "
            r"((?<= )(?P<hours>\d+) hours?)?\D*"
            r"((?<= )(?P<minutes>\d+) minutes?)?\D*"
            r"((?<= )(?P<seconds>[\d\.]+) seconds?)?\D*=+",
            self.stdout,
            ["hours", "minutes", "seconds"],
            lambda x: 0.0 if x is None else float(x),
        )

        # Check simulation time was reported by all ranks
        if sn.len(simulation_time) != self.num_tasks:
            raise ValueError(
                f"Simulation time not reported for all ranks. Found {sn.len(simulation_time)}, expected {self.num_tasks}"
            )

        # Convert hour and minute values to seconds
        simulation_time = np.array(simulation_time.evaluate())

        simulation_time[:, 0] *= 3600
        simulation_time[:, 1] *= 60

        # Return simulation time in seconds for each rank
        return simulation_time.sum(axis=1)

    @performance_function("GB", perf_key="total_memory_use")
    def extract_total_memory_use(self):
        """Extract total memory use across all ranks."""
        return sn.round(self.extract_memory_use_per_rank().sum(), 2)

    @performance_function("GB", perf_key="average_memory_use")
    def extract_average_memory_use(self):
        """Extract average memory use for each rank."""
        return sn.round(self.extract_memory_use_per_rank().sum() / self.num_tasks, 2)

    # @performance_function("GB", perf_key="min_memory_use")
    # def extract_min_memory_use(self):
    #     """Extract minimum memory use by a single rank."""
    #     return sn.round(self.extract_memory_use_per_rank().min(), 2)

    # @performance_function("GB", perf_key="max_memory_use")
    # def extract_max_memory_use(self):
    #     """Extract maximum memory use by a single rank."""
    #     return sn.round(self.extract_memory_use_per_rank().max(), 2)

    def extract_memory_use_per_rank(self) -> npt.NDArray[np.float64]:
        """Extract gprMax report of the estimated memory use per rank.

        Raises:
            ValueError: Raised if not all ranks report their estimated
                memory usage.

        Returns:
            usages: Estimated memory usage for each rank in GB.
        """
        memory_report = sn.extractall(
            r"Memory used \(estimated\): ~(?P<memory_usage>\S+) (?P<units>\S+)",
            self.stdout,
            ["memory_usage", "units"],
            [float, str],
        )

        # Check all ranks reported their estimated memory usage
        if sn.len(memory_report) != self.num_tasks:
            raise ValueError(
                f"Memory usage not reported for all ranks. Found {sn.len(memory_report)}, expected {self.num_tasks}"
            )

        usages = np.zeros(self.num_tasks)

        # Convert all values into GB
        for index, (value, unit) in enumerate(memory_report):
            if unit == "MB":
                value /= 1024
            elif unit == "KB":
                value /= 1048576
            elif unit != "GB":
                raise ValueError(f"Unknown unit of memory '{unit}'")

            usages[index] = value

        return usages
