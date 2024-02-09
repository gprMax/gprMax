import reframe as rfm
from base_tests import GprMaxRegressionTest
from reframe.core.builtins import parameter, run_after

"""ReFrame tests for basic functionality

    Usage:
        cd gprMax/reframe_tests
        reframe -C configuraiton/{CONFIG_FILE} -c reframe_tests.py -c base_tests.py -r
"""


@rfm.simple_test
class TaskfarmTest(GprMaxRegressionTest):
    tags = {"test", "mpi", "taskfarm"}

    model = parameter(["cylinder_Bscan_2D"])

    num_mpi_tasks = parameter([8, 16])
    num_cpus_per_task = 16

    @run_after("init")
    def setup_env_vars(self):
        self.num_tasks = self.num_mpi_tasks
        super().setup_env_vars()

    @run_after("init")
    def set_filenames(self):
        self.input_file = f"{self.model}.in"
        self.output_file = f"{self.model}_merged.h5"
        self.executable_opts = [self.input_file, "-n", "64", "-taskfarm"]
        self.postrun_cmds = [
            f"python -m toolboxes.Utilities.outputfiles_merge {self.model}",
            f"python -m toolboxes.Plotting.plot_Bscan -save {self.output_file} Ez",
        ]
        self.keep_files = [self.input_file, self.output_file, "{self.model}_merged.pdf"]


@rfm.simple_test
class BasicModelsTest(GprMaxRegressionTest):
    tags = {"test", "serial", "regression"}

    # List of available basic test models
    model = parameter(
        [
            "2D_ExHyHz",
            "2D_EyHxHz",
            "2D_EzHxHy",
            "cylinder_Ascan_2D",
            "hertzian_dipole_fs",
            "hertzian_dipole_hs",
            "hertzian_dipole_dispersive",
            "magnetic_dipole_fs",
        ]
    )
    num_cpus_per_task = 16

    @run_after("init")
    def set_filenames(self):
        self.input_file = f"{self.model}.in"
        self.output_file = f"{self.model}.h5"
        self.executable_opts = [self.input_file, "-o", self.output_file]
        self.postrun_cmds = [f"python -m toolboxes.Plotting.plot_Ascan -save {self.output_file}"]
        self.keep_files = [self.input_file, self.output_file, f"{self.model}.pdf"]
