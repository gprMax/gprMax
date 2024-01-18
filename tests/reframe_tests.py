import reframe as rfm
from reframe.core.builtins import parameter

from base_tests import GprmaxBaseTest


"""ReFrame tests for benchmarking and basic functionality

    Usage:
        cd gprMax/tests
        reframe -C configuraiton/{CONFIG_FILE} -c reframe_tests.py -c base_tests.py -r
"""


@rfm.simple_test
class BScanTest(GprmaxBaseTest):

    executable_opts = "cylinder_Bscan_2D.in -n 64 -mpi".split()
    num_tasks = 8
    num_cpus_per_task = 16


@rfm.simple_test
class BasicModelsTest(GprmaxBaseTest):

    # List of available basic test models
    model = parameter([
        "2D_ExHyHz",
        "2D_EyHxHz",
        "2D_EzHxHy",
        "cylinder_Ascan_2D",
        "hertzian_dipole_fs",
        "hertzian_dipole_hs",
        "hertzian_dipole_dispersive",
        "magnetic_dipole_fs",
    ])
    num_cpus_per_task = 16

    @run_after("init")
    def set_filenames(self):
        input_file = f"{self.model}.in"
        output_file = f"{self.model}.h5"
        self.executable_opts = [input_file, "-o", output_file]
        self.postrun_cmds = [f"python -m toolboxes.Plotting.plot_Ascan -save {output_file}"]
        self.keep_files = [input_file, output_file, f"{self.model}.pdf"]


@rfm.simple_test
class BenchmarkTest(GprmaxBaseTest):

    num_tasks = 1
    omp_threads = parameter([1, 2, 4, 8, 16, 32, 64, 128])
    domain = parameter([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    time_limit = "4h"

    @run_after("init")
    def setup_omp(self):
        self.num_cpus_per_task = self.omp_threads
        super().setup_omp()
        
    @run_after("init")
    def create_model_file(self):
        input_file = "benchmark_model.in"
        new_input_file = f"benchmark_model_{self.domain}.in"

        self.prerun_cmds.append(f"sed -e 's/\$domain/{self.domain}/g' -e 's/\$src/{self.domain/2}/g' {input_file} > {new_input_file}")
        self.executable_opts = [new_input_file]
        self.keep_files = [new_input_file]

    @run_after("init")
    def set_cpu_freq(self):
        self.env_vars["SLURM_CPU_FREQ_REQ"] = 2250000        
