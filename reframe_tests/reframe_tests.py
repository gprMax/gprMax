import os

import reframe as rfm
from base_tests import GprMaxAPIRegressionTest, GprMaxRegressionTest
from reframe.core.builtins import parameter, require_deps, run_after, run_before
from reframe.utility import udeps

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
class BScanTest(GprMaxRegressionTest):
    tags = {"test", "bscan"}

    model = parameter(["cylinder_Bscan_2D"])

    num_cpus_per_task = 16

    @run_after("init")
    def set_filenames(self):
        self.input_file = f"{self.model}.in"
        self.output_file = f"{self.model}_merged.h5"
        self.executable_opts = [self.input_file, "-n", "64"]
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
            "2D_ExHyHz_hs",
            "cylinder_Ascan_2D",
            "hertzian_dipole_fs",
            "hertzian_dipole_hs",
            "hertzian_dipole_dispersive",
            "magnetic_dipole_fs",
            "magnetic_dipole_hs",
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


@rfm.simple_test
class AntennaModelsTest(GprMaxRegressionTest):
    tags = {"test", "serial", "regression", "antenna"}

    # List of available antenna test models
    model = parameter(
        [
            "antenna_wire_dipole_fs",
        ]
    )
    num_cpus_per_task = 16

    @run_after("init")
    def set_filenames(self):
        self.input_file = f"{self.model}.in"
        self.output_file = f"{self.model}.h5"
        self.executable_opts = [self.input_file, "-o", self.output_file]
        self.postrun_cmds = [f"python -m toolboxes.Plotting.plot_Ascan -save {self.output_file}"]
        self.postrun_cmds = [
            f"python -m toolboxes.Plotting.plot_antenna_params -save {self.output_file}"
        ]

        antenna_t1_params = f"{self.model}_t1_params.pdf"
        antenna_ant_params = f"{self.model}_ant_params.pdf"
        plot_ascan_output = f"{self.model}.pdf"
        geometry_view = f"{self.model}.vtu"
        self.keep_files = [
            self.input_file,
            self.output_file,
            antenna_t1_params,
            antenna_ant_params,
            plot_ascan_output,
            geometry_view,
        ]


@rfm.simple_test
class SubgridTest(GprMaxAPIRegressionTest):
    tags = {"test", "api", "serial", "regression", "subgrid"}

    # List of available subgrid test models
    model = parameter(
        [
            "cylinder_fs",
            # "gssi_400_over_fractal_subsurface",  # Takes ~1hr 30m on ARCHER2
        ]
    )
    num_cpus_per_task = 16

    @run_after("init")
    def set_filenames(self):
        self.input_file = f"{self.model}.py"
        self.output_file = f"{self.model}.h5"
        self.executable_opts = [self.input_file, "-o", self.output_file]
        self.postrun_cmds = [f"python -m toolboxes.Plotting.plot_Ascan -save {self.output_file}"]

        geometry_view = f"{self.model}.vti"
        subgrid_geometry_view = f"{self.model}_sg.vti"
        plot_ascan_output = f"{self.model}.pdf"
        self.keep_files = [
            self.input_file,
            self.output_file,
            geometry_view,
            subgrid_geometry_view,
            plot_ascan_output,
        ]


@rfm.simple_test
class MPIBasicModelsTest(GprMaxRegressionTest):
    tags = {"test", "mpi", "regression"}

    # List of available basic test models
    model = parameter(
        [
            "2D_ExHyHz",
            "2D_EyHxHz",
            "2D_EzHxHy",
            "2D_ExHyHz_hs",
            "cylinder_Ascan_2D",
            "hertzian_dipole_fs",
            "hertzian_dipole_hs",
            "hertzian_dipole_dispersive",
            "magnetic_dipole_fs",
            "magnetic_dipole_hs",
        ]
    )
    num_cpus_per_task = 16
    num_tasks = 8
    num_tasks_per_node = 4

    mpi_layout = "2 2 2"

    @run_after("init")
    def inject_dependencies(self):
        """Test depends on the Python virtual environment building correctly"""
        variant = BasicModelsTest.get_variant_nums(model=lambda m: m == self.model)
        self.depends_on(BasicModelsTest.variant_name(variant[0]), udeps.by_env)
        super().inject_dependencies()

    @run_after("init")
    def set_filenames(self):
        self.input_file = f"{self.model}.in"
        self.output_file = f"{self.model}.h5"
        self.executable_opts = ["-mpi", self.mpi_layout, self.input_file, "-o", self.output_file]
        self.postrun_cmds = [f"python -m toolboxes.Plotting.plot_Ascan -save {self.output_file}"]
        self.keep_files = [self.input_file, self.output_file, f"{self.model}.pdf"]

    @run_before("run")
    def setup_reference_file(self):
        """Add prerun command to load the built Python environment"""
        variant = BasicModelsTest.get_variant_nums(model=lambda m: m == self.model)
        target = self.getdep(BasicModelsTest.variant_name(variant[0]))
        self.reference_file = os.path.join(target.stagedir, str(self.output_file))
