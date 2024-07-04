# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
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
from typing import List, Optional, Union

import cython
import numpy as np
from colorama import Fore, Style, init

from gprMax.scene import Scene

init()
from scipy.constants import c
from scipy.constants import epsilon_0 as e0
from scipy.constants import mu_0 as m0

from .utilities.host_info import detect_cuda_gpus, detect_opencl, get_host_info
from .utilities.utilities import get_terminal_width

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration parameters for a model.
    N.B. Multiple models can exist within a simulation
    """

    def __init__(self, model_num):
        self.mode = "3D"
        self.grids = []
        self.ompthreads = None
        self.model_num = model_num

        # Store information for CUDA or OpenCL solver
        #   dev: compute device object.
        #   snapsgpu2cpu: copy snapshot data from GPU to CPU during simulation.
        #     N.B. This will happen if the requested snapshots are too large to
        #           fit on the memory of the GPU. If True this will slow
        #           performance significantly.
        if sim_config.general["solver"] in ["cuda", "opencl"]:
            if sim_config.general["solver"] == "cuda":
                devs = sim_config.args.gpu
            else:  # opencl
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

            self.device = {
                "dev": sim_config.get_model_device(deviceID),
                "deviceID": deviceID,
                "snapsgpu2cpu": False,
            }

        # Total memory usage for all grids in the model. Starts with 50MB overhead.
        self.mem_overhead = 65e6
        self.mem_use = self.mem_overhead

        # String to print at start of each model run
        s = (
            f"\n--- Model {model_num + 1}/{sim_config.model_end}, "
            f"input file: {sim_config.input_file_path}"
        )
        self.inputfilestr = (
            Fore.GREEN + f"{s} {'-' * (get_terminal_width() - 1 - len(s))}\n" + Style.RESET_ALL
        )

        # Output file path and name for specific model
        self.appendmodelnumber = (
            "" if sim_config.args.n == 1 else str(model_num + 1)
        )  # Indexed from 1
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
        #   crealfunc: String to substitute into CUDA/OpenCL kernels for fields
        #                   dependent on dispersive material type.
        self.materials = {
            "maxpoles": 0,
            "dispersivedtype": None,
            "dispersiveCdtype": None,
            "drudelorentz": None,
            "crealfunc": None,
        }

    def reuse_geometry(self):
        return self.model_num != 0 and sim_config.args.geometry_fixed

    def get_scene(self):
        return sim_config.get_scene(self.model_num)

    def get_usernamespace(self):
        """Namespace only used with #python blocks which are deprecated."""
        tmp = {
            "number_model_runs": sim_config.model_end,
            "current_model_run": self.model_num + 1,
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

        if outputdir is not None:
            Path(outputdir).mkdir(exist_ok=True)
            self.output_file_path = Path(outputdir, sim_config.input_file_path.stem)
        elif sim_config.args.outputfile is not None:
            self.output_file_path = Path(sim_config.args.outputfile).with_suffix("")
        else:
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

    # TODO: Make this an enum
    em_consts = {
        "c": c,  # Speed of light in free space (m/s)
        "e0": e0,  # Permittivity of free space (F/m)
        "m0": m0,  # Permeability of free space (H/m)
        "z0": np.sqrt(m0 / e0),  # Impedance of free space (Ohms)
    }

    def __init__(self, args):
        """
        Args:
            args: Namespace with arguments from either API or CLI.
        """

        self.args = args

        self.geometry_fixed: bool = args.geometry_fixed
        self.geometry_only: bool = args.geometry_only
        self.gpu: Union[List[str], bool] = args.gpu
        self.mpi: List[int] = args.mpi
        self.number_of_models: int = args.n
        self.opencl: Union[List[str], bool] = args.opencl
        self.output_file_path: str = args.outputfile
        self.taskfarm: bool = args.taskfarm
        self.write_processed_input_file: bool = (
            args.write_processed
        )  # For depreciated Python blocks

        if self.taskfarm and self.geometry_fixed:
            logger.exception("The geometry fixed option cannot be used with MPI taskfarm.")
            raise ValueError

        if self.gpu and self.opencl:
            logger.exception("You cannot use both CUDA and OpenCl simultaneously.")
            raise ValueError

        if self.mpi and hasattr(self.args, "subgrid") and self.args.subgrid:
            logger.exception("You cannot use subgrids with MPI.")
            raise ValueError

        # Each model in a simulation is given a unique number when the instance of ModelConfig is created
        self.current_model = 0

        # Instances of ModelConfig that hold model configuration parameters.
        # TODO: Consider if this would be better as a dictionary.
        # Or maybe a non fixed length list (i.e. append each config)
        self.model_configs: List[Optional[ModelConfig]] = [None] * self.number_of_models

        # General settings for the simulation
        #   solver: cpu, cuda, opencl.
        #   precision: data type for electromagnetic field output (single/double).
        #   progressbars: progress bars on stdoout or not - switch off
        #     progressbars when logging level is greater than info (20)
        #     or when specified by the user.

        if args.show_progress_bars and args.hide_progress_bars:
            logger.exception("You cannot both show and hide progress bars.")
            raise ValueError

        self.general = {
            "solver": "cpu",
            "precision": "single",
            "progressbars": (
                args.show_progress_bars or (args.log_level <= 20 and not args.hide_progress_bars)
            ),
        }

        # Store information about host machine
        self.hostinfo = get_host_info()

        # CUDA
        if self.gpu is not None:
            self.general["solver"] = "cuda"
            # Both single and double precision are possible on GPUs, but single
            # provides best performance.
            self.general["precision"] = "single"
            self.devices = {
                "devs": [],
                "nvcc_opts": None,
            }  # pycuda device objects; nvcc compiler options
            # Suppress nvcc warnings on Microsoft Windows
            if sys.platform == "win32":
                self.devices["nvcc_opts"] = ["-w"]

            # Add pycuda available GPU(s)
            self.devices["devs"] = detect_cuda_gpus()

        # OpenCL
        if self.opencl is not None:
            self.general["solver"] = "opencl"
            self.general["precision"] = "single"
            self.devices = {
                "devs": [],
                "compiler_opts": None,
            }  # pyopencl device device(s); compiler options

            # Suppress CompilerWarning (sub-class of UserWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            # Suppress unused variable warnings on gcc
            # if sys.platform != 'win32': self.devices['compiler_opts'] = ['-w']

            # Add pyopencl available device(s)
            self.devices["devs"] = detect_opencl()

        # Subgrids
        if hasattr(self.args, "subgrid") and self.args.subgrid:
            self.general["subgrid"] = self.args.subgrid
            # Double precision should be used with subgrid for best accuracy
            self.general["precision"] = "double"
            if (self.general["subgrid"] and self.general["solver"] == "cuda") or (
                self.general["subgrid"] and self.general["solver"] == "opencl"
            ):
                logger.exception(
                    "You cannot currently use CUDA or OpenCL-based solvers with models that contain sub-grids."
                )
                raise ValueError
        else:
            self.general["subgrid"] = False

        self.autotranslate_subgrid_coordinates = True
        if hasattr(args, "autotranslate"):
            self.autotranslate_subgrid_coordinates: bool = args.autotranslate

        # Scenes parameter may not exist if user enters via CLI
        self.scenes: List[Optional[Scene]]
        if hasattr(args, "scenes") and args.scenes is not None:
            self.scenes = args.scenes
        else:
            self.scenes = [None] * self.number_of_models

        # Set more complex parameters
        self._set_precision()
        self._set_input_file_path()
        self._set_model_start_end()

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

    def _set_input_file_path(self):
        """Sets input file path for CLI or API."""
        # API
        if self.args.inputfile is None:
            self.input_file_path = Path(self.args.outputfile)
        # API/CLI
        else:
            self.input_file_path = Path(self.args.inputfile)

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

    def get_model_device(self, deviceID):
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

    def get_model_config(self, model_num: Optional[int] = None) -> ModelConfig:
        """Return ModelConfig instance for specific model.

        Args:
            model_num: number of the model. If None, returns the config for the current model

        Returns:
            model_config: requested model config
        """
        if model_num is None:
            model_num = self.current_model

        model_config = self.model_configs[model_num]
        if model_config is None:
            logger.exception(f"Cannot get ModelConfig for model {model_num}. It has not been set.")
            raise ValueError

        return model_config

    def set_model_config(self, model_config: ModelConfig, model_num: Optional[int] = None) -> None:
        """Set ModelConfig instace for specific model.

        Args:
            model_num: number of the model. If None, sets the config for the current model
        """
        if model_num is None:
            model_num = self.current_model

        self.model_configs[model_num] = model_config

    def set_current_model(self, model_num: int) -> None:
        """Set the current model by it's unique identifier

        Args:
            model_num: unique identifier for the current model
        """
        self.current_model = model_num

    def get_scene(self, model_num: Optional[int] = None) -> Optional[Scene]:
        """Return Scene instance for specific model.

        Args:
            model_num: number of the model. If None, returns the scene for the current model

        Returns:
            scene: requested scene
        """
        if model_num is None:
            model_num = self.current_model

        return self.scenes[model_num]

    def set_scene(self, scene: Scene, model_num: Optional[int] = None) -> None:
        """Set Scene instace for specific model.

        Args:
            model_num: number of the model. If None, sets the scene for the current model
        """
        if model_num is None:
            model_num = self.current_model

        self.scenes[model_num] = scene


# Single instance of SimConfig to hold simulation configuration parameters.
sim_config: SimulationConfig = None


def get_model_config() -> ModelConfig:
    """Return ModelConfig instance for specific model."""
    return sim_config.get_model_config()
