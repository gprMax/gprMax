import logging
from importlib import import_module

import humanize
import numpy as np
from jinja2 import Environment, PackageLoader
from ..utilities.utilities import hip_check
from hip import hip, hiprtc
from gprMax import config
from gprMax.cuda_opencl import (
    knl_fields_updates,
    knl_snapshots,
    knl_source_updates,
    knl_store_outputs,
)
from gprMax.grid.hip_grid import HIPGrid
from gprMax.receivers import dtoh_rx_array, htod_rx_arrays
from gprMax.snapshots import Snapshot, dtoh_snapshot_array, htod_snapshot_array
from gprMax.sources import htod_src_arrays
from gprMax.updates.updates import Updates
from gprMax.utilities.utilities import round32
from ..hip_code.load_hip_kernals import HipManager
import ctypes
logger = logging.getLogger(__name__)


class HIPUpdates(Updates[HIPGrid]):
    """HIP updates for the FDTD algorithm."""

    def __init__(self, G: HIPGrid):
        super().__init__(G)
        self.hip_manager = HipManager(G)
        self._set_field_knls()
        self._set_dispersive()
        self.floattype = config.sim_config.dtypes["C_float_or_double"]
        self.complextype = config.get_model_config().materials["dispersiveCdtype"]
        self.get_real = "hipCrealf"
        if self.complextype is None:
            self.complextype = "hipFloatComplex"
        if self.complextype == "float":
            self.get_real = ""
        self.subs_name_args = {
            "REAL": self.floattype,
            "COMPLEX": self.complextype,
        }
        self.block = hip.dim3(x=1024)
        self.grid_hip = hip.dim3(x=65536)
        self.subs_func = {
            "REAL": self.floattype,
            "COMPLEX": self.complextype,
            "CUDA_IDX": "int i = blockIdx.x * blockDim.x + threadIdx.x;",
            "NX_FIELDS": str(self.grid.nx + 1),
            "NY_FIELDS": str(self.grid.ny + 1),
            "NZ_FIELDS": str(self.grid.nz + 1),
            "NX_ID": str(self.grid.ID.shape[1]),
            "NY_ID": str(self.grid.ID.shape[2]),
            "NZ_ID": str(self.grid.ID.shape[3]),
            "N_updatecoeffsE": str(self.grid.updatecoeffsE.size), 
            "N_updatecoeffsH": str(self.grid.updatecoeffsH.size),
            "NY_MATCOEFFS": str(self.grid.updatecoeffsE.shape[1]),
            "NX_FIELDS": str(self.grid.nx + 1), 
            "NY_FIELDS": str(self.grid.ny + 1), 
            "NZ_FIELDS": str(self.grid.nz + 1), 
            "NX_ID": str(self.grid.ID.shape[1]), 
            "NY_ID": str(self.grid.ID.shape[2]), 
            "NZ_ID": str(self.grid.ID.shape[3]), 
            "GETREAL": self.get_real,
            "NX_T": self.NX_T, 
            "NY_T": self.NY_T, 
            "NZ_T": self.NZ_T,
        }
        self.env = Environment(loader=PackageLoader("gprMax", "cuda_opencl"))
        self._set_macros()
        self.knl_tmpls = {
            "update_e" : knl_fields_updates.update_electric,
            "update_m" : knl_fields_updates.update_magnetic,
            "update_electric_dispersive_A" : knl_fields_updates.update_electric_dispersive_A_hip,
            "update_electric_dispersive_B" : knl_fields_updates.update_electric_dispersive_B_hip,
            }
        self.knls = {
            "update_e" : None,
            "update_m" : None,
            "update_electric_dispersive_A" : None,
            "update_electric_dispersive_B" : None,
        }
        for knl_name in self.knl_tmpls:
            knl_tmpl = self.knl_tmpls[knl_name]
            self.knls[knl_name] = self._build_knl(knl_tmpl, knl_name)

    def _set_field_knls(self):
        self.grid.htod_geometry_arrays()
        self.grid.htod_field_arrays()
        if config.get_model_config().materials["maxpoles"] > 0:
            self.grid.htod_dispersive_arrays()
    
    def _build_knl(self, knl_func, name):
        name_plus_args = knl_func["args_hip"].substitute(self.subs_name_args)
        func_body = knl_func["func"].substitute(self.subs_func)
        bld = self.knl_common + "\n" + name_plus_args + "{" + func_body + "}"
        prog = hip_check(hiprtc.hiprtcCreateProgram(bld.encode(), name.encode(), 0, [], []))
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props,0))
        arch = props.gcnArchName
        cflags = [b"--offload-arch="+arch]
        err, = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
            raise RuntimeError(log.decode())
        code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
        code = bytearray(code_size)
        hip_check(hiprtc.hiprtcGetCode(prog, code))
        module = hip_check(hip.hipModuleLoadData(code))
        bld_kernel = hip_check(hip.hipModuleGetFunction(module, name.encode()))
        return bld_kernel
    def _set_dispersive(self):
        if config.get_model_config().materials["maxpoles"] > 0:
            self.NY_MATDISPCOEFFS = self.grid.updatecoeffsdispersive.shape[1]
            self.NX_T = self.grid.Tx.shape[1]
            self.NY_T = self.grid.Tx.shape[2]
            self.NZ_T = self.grid.Tx.shape[3]
        else:  # Set to one any substitutions for dispersive materials.
            self.NY_MATDISPCOEFFS = 1
            self.NX_T = 1
            self.NY_T = 1
            self.NZ_T = 1

    def _set_macros(self):
        """Common macros to be used in kernels."""

        # Set specific values for any dispersive materials
        if config.get_model_config().materials["maxpoles"] > 0:
            NY_MATDISPCOEFFS = self.grid.updatecoeffsdispersive.shape[1]
            NX_T = self.grid.Tx.shape[1]
            NY_T = self.grid.Tx.shape[2]
            NZ_T = self.grid.Tx.shape[3]
        else:  # Set to one any substitutions for dispersive materials.
            NY_MATDISPCOEFFS = 1
            NX_T = 1
            NY_T = 1
            NZ_T = 1

        self.knl_common = self.env.get_template("knl_common_hip.tmpl").render(
            REAL=self.floattype,
            N_updatecoeffsE=self.grid.updatecoeffsE.size,
            N_updatecoeffsH=self.grid.updatecoeffsH.size,
            NY_MATCOEFFS=self.grid.updatecoeffsE.shape[1],
            NY_MATDISPCOEFFS=NY_MATDISPCOEFFS,
            NX_FIELDS=self.grid.nx + 1,
            NY_FIELDS=self.grid.ny + 1,
            NZ_FIELDS=self.grid.nz + 1,
            NX_ID=self.grid.ID.shape[1],
            NY_ID=self.grid.ID.shape[2],
            NZ_ID=self.grid.ID.shape[3],
            NX_T=NX_T,
            NY_T=NY_T,
            NZ_T=NZ_T,
            NY_RXCOORDS=3,
            NX_RXS=6,
            NY_RXS=self.grid.iterations,
            NZ_RXS=len(self.grid.rxs),
            NY_SRCINFO=4,
            NY_SRCWAVES=self.grid.iterations,
            NX_SNAPS=Snapshot.nx_max,
            NY_SNAPS=Snapshot.ny_max,
            NZ_SNAPS=Snapshot.nz_max,
        )

    def store_outputs(self, iteration: int) -> None:
        """Stores field component values for every receiver and transmission line."""
        self.hip_manager.store_outputs_hip(iteration)

    
    def store_snapshots(self, iteration: int) -> None:
        """Stores any snapshots.

        Args:
            iteration: int for iteration number.
        """
        #print("store_snapshots not implemented in HIPUpdates")


    
    def update_magnetic(self) -> None:
        """Updates magnetic field components."""
        hip_check(
            hip.hipModuleLaunchKernel(
                self.knls["update_m"],
                *self.grid_hip, # grid
                *self.block,  # self.block
                sharedMemBytes=128,
                stream=None,
                kernelParams=None,
                extra=(
                    ctypes.c_int(self.grid.nx),
                    ctypes.c_int(self.grid.ny),
                    ctypes.c_int(self.grid.nz),
                    self.grid.ID_dev,
                    self.grid.Hx_dev,
                    self.grid.Hy_dev,
                    self.grid.Hz_dev,
                    self.grid.Ex_dev,
                    self.grid.Ey_dev,
                    self.grid.Ez_dev,
                    self.grid.updatecoeffsE_d,
                    self.grid.updatecoeffsH_d,
                )
            )
        )

    
    def update_magnetic_pml(self) -> None:
        """Updates magnetic field components with the PML correction."""
        for pml in self.grid.pmls["slabs"]:
            pml.update_magnetic()

    
    def update_magnetic_sources(self, iteration: int) -> None:
        """Updates magnetic field components from sources."""
        self.hip_manager.update_magnetic_dipole_hip(iteration)

    
    def update_electric_a(self) -> None:
        """Updates electric field components."""
        if config.get_model_config().materials["maxpoles"] == 0:
            hip_check(
                hip.hipModuleLaunchKernel(
                    self.knls["update_e"],
                    *self.grid_hip, # grid
                    *self.block,  # self.block
                    sharedMemBytes=128,
                    stream=None,
                    kernelParams=None,
                    extra=(
                        ctypes.c_int(self.grid.nx),
                        ctypes.c_int(self.grid.ny),
                        ctypes.c_int(self.grid.nz),
                        self.grid.ID_dev,
                        self.grid.Ex_dev,
                        self.grid.Ey_dev,
                        self.grid.Ez_dev,
                        self.grid.Hx_dev,
                        self.grid.Hy_dev,
                        self.grid.Hz_dev,
                        self.grid.updatecoeffsE_d,
                        self.grid.updatecoeffsH_d,
                    )
                )
            )
        else:
            hip_check(
                hip.hipModuleLaunchKernel(
                    self.knls["update_electric_dispersive_A"],
                    *self.grid_hip, # grid
                    *self.block,  # self.block
                    sharedMemBytes=128,
                    stream=None,
                    kernelParams=None,
                    extra=(
                        ctypes.c_int(self.grid.nx),
                        ctypes.c_int(self.grid.ny),
                        ctypes.c_int(self.grid.nz),
                        ctypes.c_int(config.get_model_config().materials["maxpoles"]),
                        self.grid.updatecoeffsdispersive_dev,
                        self.grid.Tx_d,
                        self.grid.Ty_d,
                        self.grid.Tz_d,
                        self.grid.ID_dev,
                        self.grid.Ex_dev,
                        self.grid.Ey_dev,
                        self.grid.Ez_dev,
                        self.grid.Hx_dev,
                        self.grid.Hy_dev,
                        self.grid.Hz_dev,
                        self.grid.updatecoeffsE_d,
                        self.grid.updatecoeffsH_d,
                    )
                )
            )


    
    def update_electric_pml(self) -> None:
        """Updates electric field components with the PML correction."""
        for pml in self.grid.pmls["slabs"]:
            pml.update_electric()

    
    def update_electric_sources(self, iteration: int) -> None:
        """Updates electric field components from sources -
        update any Hertzian dipole sources last.
        """
        if self.grid.hertziandipoles:
            self.hip_manager.update_hertzian_dipole_hip(iteration)
        if self.grid.voltagesources:
            self.hip_manager.update_voltage_source_hip(iteration)

    
    def update_electric_b(self) -> None:
        """If there are any dispersive materials do 2nd part of dispersive
        update - it is split into two parts as it requires present and
        updated electric field values. Therefore it can only be completely
        updated after the electric field has been updated by the PML and
        source updates.
        """
        if config.get_model_config().materials["maxpoles"] > 0:
            hip_check(
                hip.hipModuleLaunchKernel(
                    self.knls["update_electric_dispersive_B"],
                    *self.grid_hip, # grid
                    *self.block,  # self.block
                    sharedMemBytes=128,
                    stream=None,
                    kernelParams=None,
                    extra=(
                        ctypes.c_int(self.grid.nx),
                        ctypes.c_int(self.grid.ny),
                        ctypes.c_int(self.grid.nz),
                        ctypes.c_int(config.get_model_config().materials["maxpoles"]),
                        self.grid.updatecoeffsdispersive_dev,
                        self.grid.Tx_d,
                        self.grid.Ty_d,
                        self.grid.Tz_d,
                        self.grid.ID_dev,
                        self.grid.Ex_dev,
                        self.grid.Ey_dev,
                        self.grid.Ez_dev,
                        self.grid.updatecoeffsE_d,
                        self.grid.updatecoeffsH_d,
                    )
                )
            )

    def time_start(self) -> None:
        """Starts timer used to calculate solving time for model."""
        #print("time_start not implemented in HIPUpdates")

    def calculate_solve_time(self) -> float:
        """Calculates solving time for model."""
        #print("calculate_solve_time not implemented in HIPUpdates")
        return 0.0

    def finalise(self) -> None:
        """Finalise the updates, releasing any resources."""
        if self.grid.rxs:
            from gprMax.receivers import Rx
            rxs = np.zeros((len(Rx.allowableoutputs_dev), self.grid.iterations, len(self.grid.rxs)),
        dtype=config.sim_config.dtypes["float_or_double"])
        hip_check(hip.hipMemcpy(rxs, self.hip_manager.grid.rxs_dev, rxs.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
        for i in range(len(self.grid.rxs)):
            rx = self.grid.rxs[i].outputs
            for j, output in enumerate(Rx.allowableoutputs_dev):
                rx[output] = rxs[j, :, i]

    def cleanup(self) -> None:
        """Cleanup the updates, releasing any resources."""
        self.hip_manager.free_resources()

    def calculate_memory_used(self, iteration: int) -> int:
        #print("calculate_memory_used not implemented in HIPUpdates")
        return 0
