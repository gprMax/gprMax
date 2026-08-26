# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
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

from .utilities.host_info import detect_cuda_gpus, detect_opencl, detect_metal, get_host_info
from .utilities.utilities import get_terminal_width

logger = logging.getLogger(__name__)


def _multiple_accelerators_requested(non_cpu_solvers: List) -> bool:
    """Whether more than one of gpu/opencl/metal was requested at once.

    Each is "requested" whenever it isn't None - this can be a bool
    (True/False) or a list of deviceIDs (flat, e.g. [1], or CLI-nested,
    e.g. [[1]]). Counting `is not None` (rather than equality to True)
    matters because list forms are never `== True`, so a naive
    `.count(True)` only ever caught the bool form.

    Args:
        non_cpu_solvers: [args.gpu, args.opencl, args.metal].

    Returns:
        True if more than one of the three is not None.
    """

    return sum(solver is not None for solver in non_cpu_solvers) > 1


def _resolve_device_id(devs) -> int:
    """Normalises the various shapes the gpu=/opencl=/metal= (Python API)
    and -gpu/-opencl/-metal (CLI) arguments can take into a single
    requested deviceID.

    Accepted forms:
        None or a bool (e.g. gpu=True): no specific device requested.
        A flat list of deviceIDs (Python API, e.g. gpu=[1]).
        A list of lists (CLI's action="append" + nargs="*", e.g.
            -gpu 1 -> [[1]], or repeated -gpu 1 -gpu 2 -> [[1], [2]]).

    Args:
        devs: value of args.gpu/args.opencl/args.metal.

    Returns:
        deviceID: int deviceID of the first requested device, or 0 if
                    none was requested.
    """

    deviceIDs = []
    if isinstance(devs, list):
        for element in devs:
            if isinstance(element, list):
                deviceIDs.extend(element)
            else:
                deviceIDs.append(element)

    return deviceIDs[0] if deviceIDs else 0


class ModelConfig:
    """Configuration parameters for a model.
    N.B. Multiple models can exist within a simulation
    """

    def __init__(self, model_num):
        self.mode = "3D"
        self.requested_2d_mode = None
        # Mixing rule for magnetic (H-field) material averaging at cell
        # boundaries during Yee-cell smoothing. Harmonic averaging is the
        # default; arithmetic averaging preserves results from older versions.
        self.magnetic_averaging_mode = "harmonic"
        # Permit dispersive materials to participate in electric-edge
        # averaging. This is opt-in because the exact compound material may
        # require more poles than any constituent and therefore increases the
        # dense model-wide dispersive storage allocation.
        self.dispersive_averaging = False
        self.grids = []
        self.ompthreads = None
        self.model_num = model_num

        # Store information for CUDA or OpenCL solver
        #   dev: compute device object.
        #   snapsgpu2cpu: copy snapshot data from GPU to CPU during simulation.
        #     N.B. This will happen if the requested snapshots are too large to
        #           fit on the memory of the GPU. If True this will slow
        #           performance significantly.
        if sim_config.general["solver"] in ["cuda", "opencl", "metal"]:
            if sim_config.general["solver"] == "cuda":
                devs = sim_config.args.gpu
            elif sim_config.general["solver"] == "opencl":
                devs = sim_config.args.opencl
            else:  # metal
                devs = sim_config.args.metal

            deviceID = _resolve_device_id(devs)

            self.device = {
                "dev": sim_config.set_model_device(deviceID),
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
            Fore.GREEN + f"{s} {'-' * (get_terminal_width() - 1 - len(s))}\n\n" + Style.RESET_ALL
        )

        # Output file path and name for specific model
        study_case_count = (
            len(sim_config.study.cases) if sim_config.study is not None else sim_config.args.n
        )
        self.appendmodelnumber = (
            "" if study_case_count == 1 else str(model_num + 1)
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
        self.numdispersion = {
            "highestfreqthres": 40,
            "maxnumericaldisp": 2,
            "mingridsampling": 3,
        }

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
        # Compare against the run's actual starting model number, not the
        # literal 0 - with a restart offset (-i/i=), the first model in
        # model_range can itself be non-zero, and that first model must
        # still be built, not treated as a reuse of a never-built geometry.
        return self.model_num != sim_config.model_start and sim_config.args.geometry_fixed

    def restore_geometry_derived_config(self, reference: "ModelConfig") -> None:
        """Copy configuration derived from building the geometry/materials
        onto this (freshly-created) ModelConfig, from `reference` - the
        ModelConfig of the model that actually built the geometry.

        A fresh ModelConfig is created for every model run, even when
        geometry_fixed=True causes Model.build() to skip build_geometry()
        entirely (see reuse_geometry()) - which is where Domain.build()
        sets `mode` and _check_for_dispersive_materials() sets
        materials["maxpoles"/"drudelorentz"/"dispersivedtype"/
        "dispersiveCdtype"/"crealfunc"]. Without this, every run after the
        first would silently see this fresh ModelConfig's defaults
        ("3D", maxpoles=0) instead of what the model's own geometry/
        materials actually require - e.g. CPUUpdates.mode2d (gprMax/
        updates/cpu_updates.py) would pick 3D kernels for a genuinely 2D
        model, and FDTDGrid.reset_fields() would skip reinitialising
        Tx/Ty/Tz for a genuinely dispersive material, leaking the
        previous run's polarisation-current state into the next run.

        Args:
            reference: ModelConfig of the model that built the geometry
                (i.e. sim_config.get_model_config(sim_config.model_start)).
        """

        self.mode = reference.mode
        self.requested_2d_mode = reference.requested_2d_mode
        self.magnetic_averaging_mode = reference.magnetic_averaging_mode
        self.dispersive_averaging = reference.dispersive_averaging
        self.materials = dict(reference.materials)

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
            Path(outputdir).mkdir(parents=True, exist_ok=True)
            self.output_file_path = Path(outputdir, sim_config.input_file_path.stem)
        elif sim_config.args.outputfile is not None:
            self.output_file_path = Path(sim_config.args.outputfile)
            if self.output_file_path.suffix.lower() == ".h5":
                self.output_file_path = self.output_file_path.with_suffix("")
        else:
            self.output_file_path = sim_config.input_file_path.with_suffix("")

        parts = self.output_file_path.parts
        self.output_file_path = Path(*parts[:-1], parts[-1] + self.appendmodelnumber)
        self.output_file_path_ext = self.output_file_path.with_name(
            self.output_file_path.name + ".h5"
        )

    def set_snapshots_dir(self):
        """Sets directory to store any snapshots.

        Returns:
            snapshot_dir: Path to directory to store snapshot files in.
        """
        snapshot_dir = self.output_file_path.with_name(self.output_file_path.name + "_snaps")

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
        self.study = getattr(args, "study", None)
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
            logger.error("The geometry fixed option cannot be used with MPI taskfarm.")
            raise ValueError

        non_cpu_solvers = [self.args.gpu, self.args.opencl, self.args.metal]
        if _multiple_accelerators_requested(non_cpu_solvers):
            logger.error("You cannot use combinations of CUDA, OpenCl and Apple Metal solvers simultaneously.")
            raise ValueError

        if self.mpi and hasattr(self.args, "subgrid") and self.args.subgrid:
            logger.error("You cannot use subgrids with MPI.")
            raise ValueError

        # Each model in a simulation is given a unique number when the instance of ModelConfig is created
        self.current_model = 0

        # Instances of ModelConfig that hold model configuration parameters.
        # TODO: Consider if this would be better as a dictionary.
        # Or maybe a non fixed length list (i.e. append each config)
        self.model_configs: List[Optional[ModelConfig]] = [None] * self.number_of_models

        # General settings for the simulation
        #   solver: cpu, cuda, opencl, metal.
        #   precision: data type for electromagnetic field output (single/double).
        #   progressbars: progress bars on stdoout or not - switch off
        #     progressbars when logging level is greater than info (20)
        #     or when specified by the user.

        if args.show_progress_bars and args.hide_progress_bars:
            logger.error("You cannot both show and hide progress bars.")
            raise ValueError

        self.general = {
            "solver": "cpu",
            # Deliberately "single" by default (not "double") to preserve
            # memory on large CPU models. Overridable via -cpu_precision/
            # cpu_precision= (this branch only - the CUDA/OpenCL/Metal and
            # subgrid branches below have their own, separate precision
            # arguments/overrides).
            "precision": args.cpu_precision,
            "progressbars": (
                args.show_progress_bars or (args.log_level <= 20 and not args.hide_progress_bars)
            ),
        }

        if self.mpi and self.general["progressbars"]:
            from mpi4py import MPI

            self.general["progressbars"] = MPI.COMM_WORLD.rank == 0

        # Store information about host machine
        self.hostinfo = get_host_info()

        # CUDA
        if self.gpu is not None:
            self.general["solver"] = "cuda"
            # Both single and double precision are possible on GPUs.
            # Deliberately "single" by default (best performance) - see
            # -gpu_precision/gpu_precision=.
            self.general["precision"] = args.gpu_precision
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
            self.general["precision"] = args.gpu_precision
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

        # Apple Metal
        if self.args.metal is not None:
            self.general["solver"] = "metal"
            self.general["precision"] = args.gpu_precision
            self.devices = {"devs": [], "compiler_opts": None}  # Apple Metal device object(s); compiler options

            # Apple GPU hardware has no native double-precision floating
            # point support, and the Metal Shading Language has no "double"
            # type at all - unlike CUDA/OpenCL, this isn't a gprMax-side
            # limitation to work around, it's a hard platform constraint.
            # Without this guard, requesting double precision here would
            # silently generate invalid Metal shader source (e.g. "device
            # const double& dx", "metal::complex<double>") that fails to
            # compile - and since Metal library-compile call sites discard
            # the compile error, that would surface later as an opaque
            # AttributeError on None.newFunctionWithName_ instead of this
            # clear diagnostic.
            if self.general["precision"] == "double":
                logger.error(
                    "The Metal solver does not support double precision - Apple GPU "
                    "hardware and the Metal Shading Language have no native double "
                    "type. Use the CPU, CUDA, or OpenCL solver for double precision."
                )
                raise ValueError

            # Add metal available device(s)
            self.devices["devs"] = detect_metal()

        # Subgrids
        if hasattr(self.args, "subgrid") and self.args.subgrid:
            self.general["subgrid"] = self.args.subgrid
            # Double precision should be used with subgrid for best accuracy
            # - always wins, regardless of -cpu_precision/-gpu_precision.
            if self.general["precision"] == "single":
                logger.warning(
                    "Sub-gridding requires double precision - overriding the requested"
                    " single precision."
                )
            self.general["precision"] = "double"
            if (self.general["subgrid"] and self.general["solver"] == "cuda") or (
                self.general["subgrid"] and self.general["solver"] == "opencl") or (
                self.general["subgrid"] and self.general["solver"] == "metal"
            ):
                logger.error(
                    "You cannot currently use CUDA, OpenCL, or Metal based solvers with models that contain sub-grids."
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

    def set_model_device(self, deviceID):
        """Specify pycuda/pyopencl/pyobjc object for model.

        Args:
            deviceID: int of requested deviceID of compute device.

        Returns:
            dev: requested pycuda/pyopencl/pyobjc device object.
        """

        found = False
        for ID, dev in self.devices["devs"].items():
            if ID == deviceID:
                found = True
                return dev

        if not found:
            logger.exception(f"Compute device with device ID {deviceID} does not exist.")
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
                self.dtypes["C_complex"] = "cfloat_t"
            elif self.general["solver"] == "metal":
                # Metal Shading Language has no native complex type - a
                # small custom struct (gprMaxComplex, with the needed
                # +/-/* operators and .real()) is defined in
                # knl_common_metal.tmpl instead.
                self.dtypes["C_complex"] = "gprMaxComplex"

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
                self.dtypes["C_complex"] = "cdouble_t"
            elif self.general["solver"] == "metal":
                # Unreachable in practice - the Metal branch above already
                # raises ValueError for double precision - kept consistent
                # with the single-precision case regardless.
                self.dtypes["C_complex"] = "gprMaxComplex"

        else:
            # The CLI protects against this via argparse's
            # choices=["single", "double"] on -cpu_precision/-gpu_precision,
            # but the Python API (cpu_precision=/gpu_precision=) bypasses
            # argparse entirely and accepts any string. Without this
            # branch, an invalid value left self.dtypes completely unset,
            # failing later with a confusing, unrelated
            # AttributeError/KeyError far from the actual bad input.
            logger.error(
                f"Precision '{self.general['precision']}' is not recognised - "
                "it must be 'single' or 'double'."
            )
            raise ValueError

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
        if self.args.n <= 0:
            logger.error(f"Number of models (n={self.args.n}) must be greater than zero.")
            raise ValueError

        # `is not None` (not a truthy check) - `i` is documented as a
        # 1-indexed "model number to start/restart from", so i=0 is
        # itself invalid (rejected below), but a bare truthy check would
        # ALSO treat a legitimately-supplied i=0 identically to "i not
        # given at all" (None), silently ignoring it instead of erroring.
        if self.args.i is not None:
            if self.args.i <= 0:
                logger.error(f"Model start/restart number (i={self.args.i}) must be greater than zero.")
                raise ValueError
            modelstart = self.args.i - 1
            modelend = modelstart + self.args.n
        else:
            modelstart = 0
            modelend = modelstart + self.args.n

        self.model_start = modelstart
        self.model_end = modelend

    def _list_index(self, model_num: int) -> int:
        """Converts an absolute model number to an index into model_configs/
        scenes.

        Both lists are sized to hold exactly `n` entries (one per model in
        *this* run), regardless of where model_num itself starts counting
        from - with a restart (-i/i=), model_num starts at model_start, not
        0. Indexing by model_num directly would run off the end of either
        list as soon as model_start != 0.
        """
        return model_num - self.model_start

    def get_model_config(self, model_num: Optional[int] = None) -> ModelConfig:
        """Return ModelConfig instance for specific model.

        Args:
            model_num: number of the model. If None, returns the config for the current model

        Returns:
            model_config: requested model config
        """
        if model_num is None:
            model_num = self.current_model

        model_config = self.model_configs[self._list_index(model_num)]
        if model_config is None:
            logger.error(f"Cannot get ModelConfig for model {model_num}. It has not been set.")
            raise ValueError

        return model_config

    def set_model_config(self, model_config: ModelConfig, model_num: Optional[int] = None) -> None:
        """Set ModelConfig instace for specific model.

        Args:
            model_num: number of the model. If None, sets the config for the current model
        """
        if model_num is None:
            model_num = self.current_model

        self.model_configs[self._list_index(model_num)] = model_config

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

        return self.scenes[self._list_index(model_num)]

    def set_scene(self, scene: Scene, model_num: Optional[int] = None) -> None:
        """Set Scene instace for specific model.

        Args:
            model_num: number of the model. If None, sets the scene for the current model
        """
        if model_num is None:
            model_num = self.current_model

        self.scenes[self._list_index(model_num)] = scene


# Single instance of SimConfig to hold simulation configuration parameters.
sim_config: SimulationConfig = None


def get_model_config() -> ModelConfig:
    """Return ModelConfig instance for specific model."""
    return sim_config.get_model_config()
