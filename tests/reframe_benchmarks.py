import reframe as rfm
from base_tests import GprmaxBaseTest
from reframe.core.builtins import parameter, run_after

"""ReFrame tests for performance benchmarking

    Usage:
        cd gprMax/tests
        reframe -C configuraiton/{CONFIG_FILE} -c reframe_benchmarks.py -c base_tests.py -r
"""


@rfm.simple_test
class SingleNodeBenchmark(GprmaxBaseTest):
    tags = {"benchmark", "single node", "openmp"}

    num_tasks = 1
    omp_threads = parameter([1, 2, 4, 8, 16, 32, 64, 128])
    domain = parameter([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    cpu_freq = parameter([2250000])
    time_limit = "4h"

    @run_after("init")
    def setup_omp(self):
        self.num_cpus_per_task = self.omp_threads
        super().setup_omp()

    @run_after("init")
    def create_model_file(self):
        input_file = f"benchmark_model_{self.domain}.in"
        self.executable_opts = [input_file]
        self.keep_files = [input_file]

    @run_after("init")
    def set_cpu_freq(self):
        self.env_vars["SLURM_CPU_FREQ_REQ"] = self.cpu_freq
