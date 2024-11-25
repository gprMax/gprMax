"""ReFrame base classes for GprMax tests

Usage (run all tests):
    cd gprMax/reframe_tests
    reframe -C configuraiton/{CONFIG_FILE} -c tests/ -r
"""

import os
from pathlib import Path
from typing import Literal, Optional, Union

import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ
from numpy import prod
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
from reframe.utility import udeps

from reframe_tests.tests.regression_checks import RegressionCheck
from reframe_tests.utilities.deferrable import path_join

TESTS_ROOT_DIR = Path(__file__).parent
GPRMAX_ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
PATH_TO_PYENV = os.path.join(".venv", "bin", "activate")


@simple_test
class CreatePyenvTest(RunOnlyRegressionTest):
    valid_systems = ["generic", "archer2:login"]
    valid_prog_environs = ["builtin", "PrgEnv-gnu"]
    modules = ["cray-python"]

    prerun_cmds = [
        "python -m venv --system-site-packages --prompt gprMax .venv",
        f"source {PATH_TO_PYENV}",
        "CC=cc CXX=CC FC=ftn python -m pip install --upgrade pip",
        f"CC=cc CXX=CC FC=ftn python -m pip install -r {os.path.join(GPRMAX_ROOT_DIR, 'requirements.txt')}",
    ]
    executable = f"CC=cc CXX=CC FC=ftn python -m pip install -e {GPRMAX_ROOT_DIR}"

    @run_after("init")
    def install_system_specific_dependencies(self):
        """Install additional dependencies for specific systems"""
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
        """
        Check packages successfully installed from requirements.txt
        Check gprMax installed successfully and no other errors thrown
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


class GprMaxRegressionTest(RunOnlyRegressionTest):
    valid_systems = ["archer2:compute"]
    valid_prog_environs = ["PrgEnv-gnu"]
    modules = ["cray-python"]

    num_cpus_per_task = 16
    exclusive_access = True

    model = parameter()
    sourcesdir = required
    executable = "time -p python -m gprMax --log-level 10 --hide-progress-bars"

    regression_checks = variable(typ.List[RegressionCheck], value=[])

    test_dependency = variable(type(None), type, value=None)

    def get_test_dependency(self) -> Optional["GprMaxRegressionTest"]:
        """Get test variant with the same model and number of models"""
        if self.test_dependency is None:
            return None
        else:
            variant = self.test_dependency.variant_name(self.test_dependency.param_variant)
            return self.getdep(variant)

    def build_reference_filepath(self, name: Union[str, os.PathLike]) -> Path:
        target = self.get_test_dependency()
        if target is None:
            reference_dir = self.short_name
        else:
            reference_dir = target.short_name

        reference_file = Path("regression_checks", reference_dir, name).with_suffix(".h5")
        return reference_file.absolute()

    @run_after("init")
    def setup_env_vars(self):
        """Set OMP_NUM_THREADS environment variable from num_cpus_per_task"""
        self.env_vars["OMP_NUM_THREADS"] = self.num_cpus_per_task

        if self.current_system.name == "archer2":
            # Avoid inheriting slurm memory environment variables from any previous slurm job (i.e. the reframe job)
            self.prerun_cmds.append("unset SLURM_MEM_PER_NODE")
            self.prerun_cmds.append("unset SLURM_MEM_PER_CPU")

            # Set the matplotlib cache to the work filesystem
            self.env_vars["MPLCONFIGDIR"] = "${HOME/home/work}/.config/matplotlib"

    # TODO: Change CreatePyenvTest to a fixture instead of a test dependency
    @run_after("init")
    def inject_dependencies(self):
        """Test depends on the Python virtual environment building correctly"""
        self.depends_on("CreatePyenvTest", udeps.by_env)
        if self.test_dependency is not None:
            variant = self.test_dependency.variant_name(self.test_dependency.param_variant)
            self.depends_on(variant, udeps.by_env)

    @require_deps
    def get_pyenv_path(self, CreatePyenvTest):
        """Add prerun command to load the built Python environment"""
        path_to_pyenv = os.path.join(CreatePyenvTest(part="login").stagedir, PATH_TO_PYENV)
        self.prerun_cmds.append(f"source {path_to_pyenv}")

    @run_after("init")
    def configure_test_run(self):
        """Configure gprMax commandline arguments and plot outputs

        Set the input and output files and add postrun commands to plot
        the outputs.
        """
        self.input_file = f"{self.model}.in"
        self.output_file = f"{self.model}.h5"
        self.executable_opts = [self.input_file, "-o", self.output_file]
        self.keep_files = [self.input_file, self.output_file]

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
        """Split output from each MPI rank

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

    # @run_before("run")
    # def check_input_file_exists(self):
    #     """Skip test if input file does not exist"""
    #     # Current working directory will be where the reframe job was launched
    #     # However reframe assumes the source directory is relative to the test file
    #     with osext.change_dir(TESTS_ROOT_DIR):
    #         self.skip_if(
    #             not os.path.exists(self.sourcesdir),
    #             f"Source directory '{self.sourcesdir}' does not exist. Current working directory: '{os.getcwd()}'",
    #         )
    #         self.skip_if(
    #             not os.path.exists(os.path.join(self.sourcesdir, self.input_file)),
    #             f"Input file '{self.input_file}' not present in source directory '{self.sourcesdir}'",
    #         )

    def test_simulation_complete(self) -> Literal[True]:
        """Check simulation completed successfully"""
        return sn.assert_not_found(
            r"(?i)error",
            self.stderr,
            f"An error occured. See '{path_join(self.stagedir, self.stderr)}' for details.",
        ) and sn.assert_found(
            r"=== Simulation completed in ", self.stdout, "Simulation did not complete"
        )

    @sanity_function
    def regression_check(self) -> bool:
        """Perform regression check for the test output and snapshots

        If not all the reference files exist, then create all the
        missing reference files from the test output and fail the test.
        """
        error_messages = []
        for check in self.regression_checks:
            if not check.reference_file_exists():
                if check.create_reference_file():
                    error_messages.append(
                        f"Reference file does not exist. Creating... '{check.reference_file}'"
                    )
                else:
                    error_messages.append(
                        f"ERROR: Unable to create reference file: '{check.reference_file}'"
                    )

        return (
            self.test_simulation_complete()
            and sn.assert_true(len(error_messages) < 1, "\n".join(error_messages))
            and sn.all(sn.map(lambda check: check.run(), self.regression_checks))
        )

    @performance_function("s", perf_key="run_time")
    def extract_run_time(self):
        """Extract total runtime from the last task to complete"""
        return sn.extractsingle(
            r"real\s+(?P<run_time>\S+)", self.stderr, "run_time", float, self.num_tasks - 1
        )

    @performance_function("s", perf_key="simulation_time")
    def extract_simulation_time(self):
        """Extract simulation time reported by gprMax"""

        # sn.extractall throws an error if a group has value None.
        # Therefore have to handle the < 1 min, >= 1 min and >= 1 hour cases separately.
        timeframe = sn.extractsingle(
            r"=== Simulation completed in \S+ (?P<timeframe>hour|minute|second)",
            self.stdout,
            "timeframe",
        )
        if timeframe == "hour":
            simulation_time = sn.extractall(
                r"=== Simulation completed in (?P<hours>\S+) hours?, (?P<minutes>\S+) minutes? and (?P<seconds>\S+) seconds? =*",
                self.stdout,
                ["hours", "minutes", "seconds"],
                float,
            )
            hours = simulation_time[0][0]
            minutes = simulation_time[0][1]
            seconds = simulation_time[0][2]
        elif timeframe == "minute":
            hours = 0
            simulation_time = sn.extractall(
                r"=== Simulation completed in (?P<minutes>\S+) minutes? and (?P<seconds>\S+) seconds? =*",
                self.stdout,
                ["minutes", "seconds"],
                float,
            )
            minutes = simulation_time[0][0]
            seconds = simulation_time[0][1]
        else:
            hours = 0
            minutes = 0
            seconds = sn.extractsingle(
                r"=== Simulation completed in (?P<seconds>\S+) seconds? =*",
                self.stdout,
                "seconds",
                float,
            )
        return hours * 3600 + minutes * 60 + seconds


class GprMaxAPIRegressionTest(GprMaxRegressionTest):
    executable = "time -p python"

    @run_after("setup", always_last=True)
    def configure_test_run(self):
        """Input files for API tests will be python files"""
        super().configure_test_run(input_file_ext=".py")


class GprMaxBScanRegressionTest(GprMaxRegressionTest):
    num_models = parameter()

    @run_after("setup", always_last=True)
    def configure_test_run(self):
        """Add B-Scan specific commandline arguments and postrun cmds"""
        self.extra_executable_opts += ["-n", str(self.num_models)]
        super().configure_test_run()

        # Override postrun_cmds
        # Merge output files and create B-Scan plot
        self.postrun_cmds = [
            f"python -m toolboxes.Utilities.outputfiles_merge {self.model}",
            f"mv {self.model}_merged.h5 {self.output_file}",
            f"python -m toolboxes.Plotting.plot_Bscan -save {self.output_file} Ez",
        ]


class GprMaxTaskfarmRegressionTest(GprMaxBScanRegressionTest):
    serial_dependency: type[GprMaxRegressionTest]
    extra_executable_opts = ["-taskfarm"]
    sourcesdir = "src"  # Necessary so test is not skipped (set later)

    num_tasks = required

    def _get_variant(self) -> str:
        """Get test variant with the same model and number of models"""
        variant = self.serial_dependency.get_variant_nums(
            model=lambda m: m == self.model, num_models=lambda n: n == self.num_models
        )
        return self.serial_dependency.variant_name(variant[0])

    @run_after("init")
    def inject_dependencies(self):
        """Test depends on the serial version of the test"""
        self.depends_on(self._get_variant(), udeps.by_env)
        super().inject_dependencies()

    @run_after("init")
    def set_variables_from_serial_dependency(self):
        """Set test dependencies to the same as the serial test"""
        self.sourcesdir = str(self.serial_dependency.sourcesdir)
        self.has_receiver_output = bool(self.serial_dependency.has_receiver_output)
        self.snapshots = list(self.serial_dependency.snapshots)

    @run_after("setup")
    def setup_reference_files(self):
        """
        Set the reference file regression checks to the output of the
        serial test
        """
        target = self.getdep(self._get_variant())
        self.reference_file = os.path.join(target.stagedir, target.output_file)
        self.snapshot_reference_files = target.snapshot_reference_files


class GprMaxMPIRegressionTest(GprMaxRegressionTest):
    # TODO: Make this a variable
    serial_dependency: type[GprMaxRegressionTest]
    mpi_layout = parameter()
    sourcesdir = "src"  # Necessary so test is not skipped (set later)

    @run_after("setup", always_last=True)
    def configure_test_run(self):
        """Add MPI specific commandline arguments"""
        self.num_tasks = int(prod(self.mpi_layout))
        self.extra_executable_opts = ["--mpi", *map(str, self.mpi_layout)]
        super().configure_test_run()

    def _get_variant(self) -> str:
        """Get test variant with the same model"""
        # TODO: Refactor tests to work with benchmarks
        variant = self.serial_dependency.get_variant_nums(
            model=lambda m: m == self.model,
            # cpu_freq=lambda f: f == self.cpu_freq,
            # omp_threads=lambda o: o == 16,
        )
        return self.serial_dependency.variant_name(variant[0])

    @run_after("init")
    def inject_dependencies(self):
        """Test depends on the specified serial test"""
        self.depends_on(self._get_variant(), udeps.by_env)
        super().inject_dependencies()

    @run_after("init")
    def set_variables_from_serial_dependency(self):
        """Set test dependencies to the same as the serial test"""
        self.sourcesdir = str(self.serial_dependency.sourcesdir)
        self.has_receiver_output = bool(self.serial_dependency.has_receiver_output)
        self.snapshots = list(self.serial_dependency.snapshots)

    @run_after("setup")
    def setup_reference_files(self):
        """
        Set the reference file regression checks to the output of the
        serial test
        """
        target = self.getdep(self._get_variant())
        self.reference_file = os.path.join(target.stagedir, target.output_file)
        self.snapshot_reference_files = target.snapshot_reference_files
