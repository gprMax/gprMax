import reframe as rfm
from base_tests import GprmaxBaseTest
from reframe.core.builtins import parameter, run_after

"""ReFrame tests for basic functionality

    Usage:
        cd gprMax/tests
        reframe -C configuraiton/{CONFIG_FILE} -c reframe_tests.py -c base_tests.py -r
"""


@rfm.simple_test
class BScanTest(GprmaxBaseTest):
    tags = {"test", "mpi", "taskfarm"}

    executable_opts = "cylinder_Bscan_2D.in -n 64 -mpi".split()
    num_tasks = 8
    num_cpus_per_task = 16


@rfm.simple_test
class BasicModelsTest(GprmaxBaseTest):
    tags = {"test", "serial"}

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
        input_file = f"{self.model}.in"
        output_file = f"{self.model}.h5"
        self.executable_opts = [input_file, "-o", output_file]
        self.postrun_cmds = [f"python -m toolboxes.Plotting.plot_Ascan -save {output_file}"]
        self.keep_files = [input_file, output_file, f"{self.model}.pdf"]
