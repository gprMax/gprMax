# Copyright (C) 2015-2023: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import logging
import sys
import warnings
from pathlib import Path

# Used to suppress CompilerWarning (sub-class of UserWarning) from pyopencl
warnings.filterwarnings("ignore", category=UserWarning)

import cython
import numpy as np
from colorama import Fore, Style, init

init()
from scipy.constants import c
from scipy.constants import epsilon_0 as e0
from scipy.constants import mu_0 as m0

from .utilities.host_info import detect_cuda_gpus, detect_opencl, get_host_info
from .utilities.utilities import get_terminal_width

logger = logging.getLogger(__name__)

# Single instance of SimConfig to hold simulation configuration parameters.
sim_config = None

# Instances of ModelConfig that hold model configuration parameters.
model_configs = []

# Each model in a simulation is given a unique number when the instance of
# ModelConfig is created
model_num = 0


def get_model_config():
    """Return ModelConfig instace for specific model."""
    if sim_config.args.mpi:
        return model_configs
    else:
        return model_configs[model_num]


class ModelConfig:
    """Configuration parameters for a model.
    N.B. Multiple models can exist within a simulation
    """

    def __init__(self):
        self.mode = "3D"
        self.grids = []
        self.ompthreads = None

        # Store information for CUDA or OpenCL solver
        #   dev: compute device object.
        #   snapsgpu2cpu: copy snapshot data from GPU to CPU during simulation.
        #     N.B. This will happen if the requested snapshots are too large to
        #           fit on the memory of the GPU. If True this will slow
        #           performance significantly.
        if sim_config.general["solver"] in ["cuda", "opencl"]:
            if sim_config.general["solver"] == "cuda":
                devs = sim_config.args.gpu
            elif sim_config.general["solver"] == "opencl":
                devs = sim_config.args.opencl

            # If a list of lists of deviceIDs is found, flatten it
            if any(isinstance(element, list) for element in devs):
                deviceID = [val for sublist in devs for val in sublist]

            # If no deviceID is given default to using deviceID 0. Else if either
            # a single deviceID or list of deviceIDs is given use first one.
            try:
                deviceID = deviceID[0]
            except:
                deviceID = 0

            self.device = {"dev": sim_config.set_model_device(deviceID), "snapsgpu2cpu": False}

        # Total memory usage for all grids in the model. Starts with 50MB overhead.
        self.mem_overhead = 65e6
        self.mem_use = self.mem_overhead

        self.reuse_geometry = False

        # String to print at start of each model run
        s = f"\n--- Model {model_num + 1}/{sim_config.model_end}, " f"input file: {sim_config.input_file_path}"
        self.inputfilestr = Fore.GREEN + f"{s} {'-' * (get_terminal_width() - 1 - len(s))}\n" + Style.RESET_ALL

        # Output file path and name for specific model
        self.appendmodelnumber = "" if sim_config.args.n == 1 else str(model_num + 1)  # Indexed from 1
        self.set_output_file_path()

        # Numerical dispersion analysis parameters
        #   highestfreqthres: threshold (dB) down from maximum power (0dB) of
        #                       main frequency used to calculate highest
        #                       frequency for numerical dispersion analysis.
        #   maxnumericaldisp: maximum allowable percentage physical
        #                       phase-velocity phase error.
        #   mingridsampling: minimum grid sampling of smallest wavelength for
        #                       physical wave propagation.
        self.numdispersion = {"highestfreqthres": 40, "maxnumericaldisp": 2, "mingridsampling": 3}

        # General information to configure materials
        #   maxpoles: Maximum number of dispersive material poles in a model.
        #   dispersivedtype: Data type for dispersive materials.
        #   dispersiveCdtype: Data type for dispersive materials in Cython.
        #   drudelorentz: True/False model contains Drude or Lorentz materials.
        #   cudarealfunc: String to substitute into CUDA kernels for fields
        #                   dependent on dispersive material type.
        self.materials = {
            "maxpoles": 0,
            "dispersivedtype": None,
            "dispersiveCdtype": None,
            "drudelorentz": None,
            "cudarealfunc": "",
        }

    def get_scene(self):
        if sim_config.scenes:
            return sim_config.scenes[model_num]
        else:
            return None

    def get_usernamespace(self):
        """Namespace only used with #python blocks which are deprecated."""
        tmp = {
            "number_model_runs": sim_config.model_end,
            "current_model_run": model_num + 1,
            "inputfile": sim_config.input_file_path.resolve(),
        }
        return dict(**sim_config.em_consts, **tmp)

    def set_dispersive_material_types(self):
        """Sets data type for disperive materials. Complex if Drude or Lorentz
        materials are present. Real if Debye materials.
        """
        if self.materials["drudelorentz"]:
            self.materials["crealfunc"] = ".real()"
            self.materials["dispersivedtype"] = sim_config.dtypes["complex"]
            self.materials["dispersiveCdtype"] = sim_config.dtypes["C_complex"]
        else:
            self.materials["crealfunc"] = ""
            self.materials["dispersivedtype"] = sim_config.dtypes["float_or_double"]
            self.materials["dispersiveCdtype"] = sim_config.dtypes["C_float_or_double"]

    def set_output_file_path(self, outputdir=None):
        """Sets output file path. Can be provided by the user via the API or an
            input file command. If they haven't provided one use the input file
            path instead.

        Args:
            outputdir: string of output file directory given by input file command.
        """

        if not outputdir:
            try:
                self.output_file_path = Path(self.args.outputfile)
            except AttributeError:
                self.output_file_path = sim_config.input_file_path.with_suffix("")
        else:
            try:
                Path(outputdir).mkdir(exist_ok=True)
                self.output_file_path = Path(outputdir, sim_config.input_file_path.stem)
            except AttributeError:
                self.output_file_path = sim_config.input_file_path.with_suffix("")

        parts = self.output_file_path.parts
        self.output_file_path = Path(*parts[:-1], parts[-1] + self.appendmodelnumber)
        self.output_file_path_ext = self.output_file_path.with_suffix(".h5")

    def set_snapshots_dir(self):
        """Sets directory to store any snapshots.

        Returns:
            snapshot_dir: Path to directory to store snapshot files in.
        """
        parts = self.output_file_path.with_suffix("").parts
        snapshot_dir = Path(*parts[:-1], parts[-1] + "_snaps")

        return snapshot_dir


class SimulationConfig:
    """Configuration parameters for a simulation.
    N.B. A simulation can consist of multiple models.
    """

    def __init__(self, args):
        """
        Args:
            args: Namespace with arguments from either API or CLI.
        """

        self.args = args

        if self.args.mpi and self.args.geometry_fixed:
            logger.exception("The geometry fixed option cannot be used with MPI.")
            raise ValueError

        if self.args.gpu and self.args.opencl:
            logger.exception("You cannot use both CUDA and OpenCl simultaneously.")
            raise ValueError

        # General settings for the simulation
        #   solver: cpu, cuda, opencl.
        #   precision: data type for electromagnetic field output (single/double).
        #   progressbars: progress bars on stdoout or not - switch off
        #                   progressbars when logging level is greater than
        #                   info (20)

        self.general = {"solver": "cpu", "precision": "single", "progressbars": args.log_level <= 20}

        self.em_consts = {
            "c": c,  # Speed of light in free space (m/s)
            "e0": e0,  # Permittivity of free space (F/m)
            "m0": m0,  # Permeability of free space (H/m)
            "z0": np.sqrt(m0 / e0), # Impedance of free space (Ohms)
        }  

        # Store information about host machine
        self.hostinfo = get_host_info()

        # CUDA
        if self.args.gpu is not None:
            self.general["solver"] = "cuda"
            # Both single and double precision are possible on GPUs, but single
            # provides best performance.
            self.general["precision"] = "single"
            self.devices = {"devs": [], "nvcc_opts": None}  # pycuda device objects  # nvcc compiler options
            # Suppress nvcc warnings on Microsoft Windows
            if sys.platform == "win32":
                self.devices["nvcc_opts"] = ["-w"]

            # Add pycuda available GPU(s)
            self.devices["devs"] = detect_cuda_gpus()

        # OpenCL
        if self.args.opencl is not None:
            self.general["solver"] = "opencl"
            self.general["precision"] = "single"
            self.devices = {"devs": [], "compiler_opts": None}  # pyopencl available device(s)

            # Suppress unused variable warnings on gcc
            # if sys.platform != 'win32': self.devices['compiler_opts'] = ['-w']

            # Add pyopencl available device(s)
            self.devices["devs"] = detect_opencl()

        # Subgrids
        try:
            self.general["subgrid"] = self.args.subgrid
            # Double precision should be used with subgrid for best accuracy
            self.general["precision"] = "double"
            if (self.general["subgrid"] and self.general["solver"] == "cuda") or (
                self.general["subgrid"] and self.general["solver"] == "opencl"
            ):
                logger.exception(
                    "You cannot currently use CUDA or OpenCL-based " "solvers with models that contain sub-grids."
                )
                raise ValueError
        except AttributeError:
            self.general["subgrid"] = False

        # Scenes parameter may not exist if user enters via CLI
        try:
            self.scenes = args.scenes
        except AttributeError:
            self.scenes = []

        # Set more complex parameters
        self._set_precision()
        self._set_input_file_path()
        self._set_model_start_end()

    def set_model_device(self, deviceID):
        """Specify pycuda/pyopencl object for model.

        Args:
            deviceID: int of requested deviceID of compute device.

        Returns:
            dev: requested pycuda/pyopencl device object.
        """

        found = False
        for ID, dev in self.devices["devs"].items():
            if ID == deviceID:
                found = True
                return dev

        if not found:
            logger.exception(f"Compute device with device ID {deviceID} does " "not exist.")
            raise ValueError

    def _set_precision(self):
        """Data type (precision) for electromagnetic field output.

        Solid and ID arrays use 32-bit integers (0 to 4294967295).
        Rigid arrays use 8-bit integers (the smallest available type to store true/false).
        Fractal arrays use complex numbers.
        Dispersive coefficient arrays use either float or complex numbers.
        Main field arrays use floats.
        """

        if self.general["precision"] == "single":
            self.dtypes = {
                "float_or_double": np.float32,
                "complex": np.complex64,
                "cython_float_or_double": cython.float,
                "cython_complex": cython.floatcomplex,
                "C_float_or_double": "float",
                "C_complex": None,
            }
            if self.general["solver"] == "cuda":
                self.dtypes["C_complex"] = "pycuda::complex<float>"
            elif self.general["solver"] == "opencl":
                self.dtypes["C_complex"] = "cfloat"

        elif self.general["precision"] == "double":
            self.dtypes = {
                "float_or_double": np.float64,
                "complex": np.complex128,
                "cython_float_or_double": cython.double,
                "cython_complex": cython.doublecomplex,
                "C_float_or_double": "double",
                "C_complex": None,
            }
            if self.general["solver"] == "cuda":
                self.dtypes["C_complex"] = "pycuda::complex<double>"
            elif self.general["solver"] == "opencl":
                self.dtypes["C_complex"] = "cdouble"

    def _set_model_start_end(self):
        """Sets range for number of models to run (internally 0 index)."""
        if self.args.i:
            modelstart = self.args.i - 1
            modelend = modelstart + self.args.n
        else:
            modelstart = 0
            modelend = modelstart + self.args.n

        self.model_start = modelstart
        self.model_end = modelend

    def _set_input_file_path(self):
        """Sets input file path for CLI or API."""
        # API
        if self.args.inputfile is None:
            self.input_file_path = Path(self.args.outputfile)
        # API/CLI
        else:
            self.input_file_path = Path(self.args.inputfile)
