from hip import hip, hiprtc
from ..utilities.utilities import hip_check
from .hip_source import update_m, update_hertzian_dipole, store_outputs, update_voltage_source, update_magnetic_dipole, update_electric_dispersive_A, update_electric_dispersive_B
import numpy as np
import ctypes
from .. import config
from gprMax.grid.hip_grid import HIPGrid
from gprMax.sources import htod_src_arrays
from gprMax.receivers import dtoh_rx_array, htod_rx_arrays
import random
from gprMax.hip_code import E_HORIPML, M_HORIPML
from importlib import import_module
from gprMax.hip_code.macros import macros
from gprMax import config
floattype = 'float'
# complextype = config.get_model_config().materials["dispersiveCdtype"]


class HipManager:
    def __init__(self, G: HIPGrid):
        self.grid = G
        self.module = None
        self.kernel = None
        self.prog = None
        self.update_e_kernel = None
        self.update_m_kernel = None
        self.update_hertzian_dipole_kernel = None
        self.store_outputs_kernel = None
        self.update_voltage_source_kernel = None
        self.update_electric_dispersive_A_kernel = None
        self.update_electric_dispersive_B_kernel = None
        self.complextype = config.get_model_config().materials["dispersiveCdtype"]
        self.get_real = "hipCrealf"
        if self.complextype is None:
            self.complextype = "hipFloatComplex"
        if self.complextype is "float":
            self.get_real = ""
        print(config.get_model_config().materials["dispersiveCdtype"])
        self.subs_name_args = {
            "REAL": floattype,
            "COMPLEX": floattype,
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
        }
        # Substitutions in function bodies
        self.subs_func = {
            "REAL": floattype,
            "CUDA_IDX": "int i = blockIdx.x * blockDim.x + threadIdx.x;",
            "NX_FIELDS": str(self.grid.nx + 1),
            "NY_FIELDS": self.grid.ny + 1,
            "NZ_FIELDS": self.grid.nz + 1,
            "NX_ID": self.grid.ID.shape[1],
            "NY_ID": self.grid.ID.shape[2],
            "NZ_ID": self.grid.ID.shape[3],
            "N_updatecoeffsE": self.grid.updatecoeffsE.size, 
            "N_updatecoeffsH": self.grid.updatecoeffsH.size,
            "NY_MATCOEFFS": self.grid.updatecoeffsE.shape[1],
            "NX_FIELDS": self.grid.nx + 1, 
            "NY_FIELDS": self.grid.ny + 1, 
            "NZ_FIELDS": self.grid.nz + 1, 
            "NX_ID": self.grid.ID.shape[1], 
            "NY_ID": self.grid.ID.shape[2], 
            "NZ_ID": self.grid.ID.shape[3], 
        }
        self.macros = macros

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
        # self.ID_dev = hip_check(hip.hipMalloc(self.grid.ID.nbytes))
        # self.Ex_dev = hip_check(hip.hipMalloc(self.grid.Ex.nbytes))
        # self.Ey_dev = hip_check(hip.hipMalloc(self.grid.Ey.nbytes))
        # self.Ez_dev = hip_check(hip.hipMalloc(self.grid.Ez.nbytes))
        # self.Hx_dev = hip_check(hip.hipMalloc(self.grid.Hx.nbytes))
        # self.Hy_dev = hip_check(hip.hipMalloc(self.grid.Hy.nbytes))
        # self.Hz_dev = hip_check(hip.hipMalloc(self.grid.Hz.nbytes))
        # self.Tx_d = hip_check(hip.hipMalloc(self.grid.Tx.nbytes))
        # self.Ty_d = hip_check(hip.hipMalloc(self.grid.Ty.nbytes))
        # self.Tz_d = hip_check(hip.hipMalloc(self.grid.Tz.nbytes))
        # self.updatecoeffsdispersive_dev = hip_check(hip.hipMalloc(self.grid.updatecoeffsdispersive.nbytes))
        # self.updatecoeffsE_d = hip_check(hip.hipMalloc(self.grid.updatecoeffsE.nbytes))
        # self.updatecoeffsH_d = hip_check(hip.hipMalloc(self.grid.updatecoeffsH.nbytes))
        # hip_check(hip.hipMemcpy(self.ID_dev, self.grid.ID, self.grid.ID.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        # hip_check(hip.hipMemcpy(self.Ex_dev, self.grid.Ex, self.grid.Ex.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        # hip_check(hip.hipMemcpy(self.Ey_dev, self.grid.Ey, self.grid.Ey.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        # hip_check(hip.hipMemcpy(self.Ez_dev, self.grid.Ez, self.grid.Ez.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        # hip_check(hip.hipMemcpy(self.Hx_dev, self.grid.Hx, self.grid.Hx.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        # hip_check(hip.hipMemcpy(self.Hy_dev, self.grid.Hy, self.grid.Hy.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        # hip_check(hip.hipMemcpy(self.Hz_dev, self.grid.Hz, self.grid.Hz.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        # hip_check(hip.hipMemcpy(self.updatecoeffsE_d, self.grid.updatecoeffsE, self.grid.updatecoeffsE.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        # hip_check(hip.hipMemcpy(self.updatecoeffsH_d, self.grid.updatecoeffsH, self.grid.updatecoeffsH.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        # hip_check(hip.hipMemcpy(self.Tx_d, self.grid.Tx, self.grid.Tx.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        # hip_check(hip.hipMemcpy(self.Ty_d, self.grid.Ty, self.grid.Ty.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        # hip_check(hip.hipMemcpy(self.Tz_d, self.grid.Tz, self.grid.Tz.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        # hip_check(hip.hipMemcpy(self.updatecoeffsdispersive_dev, self.grid.updatecoeffsdispersive, self.grid.updatecoeffsdispersive.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        # self.rxcoords_dev, self.rxs_dev = htod_rx_arrays(self.grid)
        self.len_rxs = int(6 * self.grid.iterations * len(self.grid.rxs)/4)
        self.len_rxcoords = int(len(self.grid.rxs) * 3)
        self.block = hip.dim3(x=1024)
        self.grid_hip = hip.dim3(x=65536)

        self.compile_kernels()


    def _set_pml_knls(self):
        """PMLS - prepares kernels and gets kernel functions."""
        knl_pml_updates_electric = import_module(
            "gprMax.hip_code.E_" + self.grid.pmls["formulation"]
        )
        knl_pml_updates_magnetic = import_module(
            "gprMax.hip_code.M_" + self.grid.pmls["formulation"]
        )

        # Initialise arrays on GPU, set block per grid, and get kernel functions
        for pml in self.grid.pmls["slabs"]:
            pml.htod_field_arrays()
            pml.set_blocks_per_grid()
            knl_name = f"order{len(pml.CFS)}_{pml.direction}"
            self.subs_name_args["FUNC"] = knl_name

            knl_electric = getattr(knl_pml_updates_electric, knl_name)
            bld = self._construct_knl(knl_electric, self.subs_name_args, self.subs_func, self.macros)
            knlE = self._build_knl(bld, knl_name)
            pml.update_electric_dev = knlE

            knl_magnetic = getattr(knl_pml_updates_magnetic, knl_name)
            bld = self._construct_knl(knl_magnetic, self.subs_name_args, self.subs_func, self.macros)
            knlH = self._build_knl(bld, knl_name)
            pml.update_magnetic_dev = knlH
            print(f"Compiling {knl_name} Done")


    def _construct_knl(self, knl_func, subs_name_args, subs_func, macros):
        name_plus_args = macros.substitute(subs_name_args) + 'extern "C"' + knl_func["args_hip"].substitute(subs_name_args)
        func_body = knl_func["func"].substitute(subs_func)
        knl = name_plus_args + "{" + func_body + "}"

        return knl
    
    def _build_knl(self, bld, name):
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



    def compile_kernels(self):
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props,0))
        print(props)
        arch = props.gcnArchName
        print(f"Compiling kernel for {arch}")

        # self.compile_kernels_e()
        # self.compile_kernels_m()
        self.compile_kernels_hertzian_dipole()
        self.compile_store_outputs()
        self.compile_kernels_voltage_sources()
        self.compile_kernels_magnetic_sources()
        self.compile_kernels_e_dispersive_A()
        self.compile_kernels_e_dispersive_B()  
        self._set_pml_knls()
        print("compiling done")



    # def compile_kernels_e(self):
    #     source_update_e = update_e.substitute(REAL=floattype, COMPLEX=floattype, N_updatecoeffsE=self.grid.updatecoeffsE.size, N_updatecoeffsH=self.grid.updatecoeffsH.size, NY_MATCOEFFS=self.grid.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=self.NY_MATDISPCOEFFS, NX_FIELDS=self.grid.nx + 1, NY_FIELDS=self.grid.ny + 1, NZ_FIELDS=self.grid.nz + 1, NX_ID=self.grid.ID.shape[1], NY_ID=self.grid.ID.shape[2], NZ_ID=self.grid.ID.shape[3], NX_T=self.NX_T, NY_T=self.NY_T, NZ_T=self.NZ_T)
    #     self.prog = hip_check(hiprtc.hiprtcCreateProgram(source_update_e.encode(), b"update_e", 0, [], []))
    #     props = hip.hipDeviceProp_t()
    #     hip_check(hip.hipGetDeviceProperties(props,0))
    #     arch = props.gcnArchName
    #     cflags = [b"--offload-arch="+arch]
    #     err, = hiprtc.hiprtcCompileProgram(self.prog, len(cflags), cflags)
    #     if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
    #         log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(self.prog))
    #         log = bytearray(log_size)
    #         hip_check(hiprtc.hiprtcGetProgramLog(self.prog, log))
    #         raise RuntimeError(log.decode())
    #     code_size = hip_check(hiprtc.hiprtcGetCodeSize(self.prog))
    #     code = bytearray(code_size)
    #     hip_check(hiprtc.hiprtcGetCode(self.prog, code))
    #     self.module = hip_check(hip.hipModuleLoadData(code))
    #     self.update_e_kernel = hip_check(hip.hipModuleGetFunction(self.module, b"update_e"))
    #     print(f"Compiling update_e Done")
    
    # def compile_kernels_m(self):
    #     source_update_m = update_m.substitute(REAL=floattype, COMPLEX=floattype, N_updatecoeffsE=self.grid.updatecoeffsE.size, N_updatecoeffsH=self.grid.updatecoeffsH.size, NY_MATCOEFFS=self.grid.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=self.NY_MATDISPCOEFFS, NX_FIELDS=self.grid.nx + 1, NY_FIELDS=self.grid.ny + 1, NZ_FIELDS=self.grid.nz + 1, NX_ID=self.grid.ID.shape[1], NY_ID=self.grid.ID.shape[2], NZ_ID=self.grid.ID.shape[3], NX_T=self.NX_T, NY_T=self.NY_T, NZ_T=self.NZ_T)
    #     self.prog = hip_check(hiprtc.hiprtcCreateProgram(source_update_m.encode(), b"update_m", 0, [], []))
    #     props = hip.hipDeviceProp_t()
    #     hip_check(hip.hipGetDeviceProperties(props,0))
    #     arch = props.gcnArchName
    #     cflags = [b"--offload-arch="+arch]
    #     err, = hiprtc.hiprtcCompileProgram(self.prog, len(cflags), cflags)
    #     if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
    #         log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(self.prog))
    #         log = bytearray(log_size)
    #         hip_check(hiprtc.hiprtcGetProgramLog(self.prog, log))
    #         raise RuntimeError(log.decode())
    #     code_size = hip_check(hiprtc.hiprtcGetCodeSize(self.prog))
    #     code = bytearray(code_size)
    #     hip_check(hiprtc.hiprtcGetCode(self.prog, code))
    #     self.module = hip_check(hip.hipModuleLoadData(code))
    #     self.update_m_kernel = hip_check(hip.hipModuleGetFunction(self.module, b"update_m"))
    #     print(f"Compiling update_m Done")

    def compile_kernels_hertzian_dipole(self):
        source_update_hertzian_dipole = update_hertzian_dipole.substitute(REAL=floattype, COMPLEX=floattype, N_updatecoeffsE=self.grid.updatecoeffsE.size, N_updatecoeffsH=self.grid.updatecoeffsH.size, NY_MATCOEFFS=self.grid.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=self.NY_MATDISPCOEFFS, NX_FIELDS=self.grid.nx + 1, NY_FIELDS=self.grid.ny + 1, NZ_FIELDS=self.grid.nz + 1, NX_ID=self.grid.ID.shape[1], NY_ID=self.grid.ID.shape[2], NZ_ID=self.grid.ID.shape[3], NX_T=self.NX_T, NY_T=self.NY_T, NZ_T=self.NZ_T, NY_SRCINFO=4, NY_SRCWAVES=self.grid.iterations)
        self.prog = hip_check(hiprtc.hiprtcCreateProgram(source_update_hertzian_dipole.encode(), b"update_hertzian_dipole", 0, [], []))
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props,0))
        arch = props.gcnArchName
        cflags = [b"--offload-arch="+arch]
        err, = hiprtc.hiprtcCompileProgram(self.prog, len(cflags), cflags)
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(self.prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(self.prog, log))
            raise RuntimeError(log.decode())
        code_size = hip_check(hiprtc.hiprtcGetCodeSize(self.prog))
        code = bytearray(code_size)
        hip_check(hiprtc.hiprtcGetCode(self.prog, code))
        self.module = hip_check(hip.hipModuleLoadData(code))
        self.update_hertzian_dipole_kernel = hip_check(hip.hipModuleGetFunction(self.module, b"update_hertzian_dipole"))
        print(f"Compiling update_hertzian_dipole Done")

    def compile_store_outputs(self):
        source_store_outputs = store_outputs.substitute(REAL=floattype, NY_RXCOORDS=3, NX_RXS=6, NY_RXS=self.grid.iterations, NZ_RXS=len(self.grid.rxs), NY_FIELDS=self.grid.ny + 1, NZ_FIELDS=self.grid.nz + 1)
        self.prog = hip_check(hiprtc.hiprtcCreateProgram(source_store_outputs.encode(), b"store_outputs", 0, [], []))
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props,0))
        arch = props.gcnArchName
        cflags = [b"--offload-arch="+arch]
        err, = hiprtc.hiprtcCompileProgram(self.prog, len(cflags), cflags)
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(self.prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(self.prog, log))
            raise RuntimeError(log.decode())
        code_size = hip_check(hiprtc.hiprtcGetCodeSize(self.prog))
        code = bytearray(code_size)
        hip_check(hiprtc.hiprtcGetCode(self.prog, code))
        self.module = hip_check(hip.hipModuleLoadData(code))
        self.store_outputs_kernel = hip_check(hip.hipModuleGetFunction(self.module, b"store_outputs"))
        print(f"Compiling store_outputs Done")

    def compile_kernels_voltage_sources(self):
        source_update_voltage_source = update_voltage_source.substitute(REAL=floattype, COMPLEX=floattype, N_updatecoeffsE=self.grid.updatecoeffsE.size, N_updatecoeffsH=self.grid.updatecoeffsH.size, NY_MATCOEFFS=self.grid.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=self.NY_MATDISPCOEFFS, NX_FIELDS=self.grid.nx + 1, NY_FIELDS=self.grid.ny + 1, NZ_FIELDS=self.grid.nz + 1, NX_ID=self.grid.ID.shape[1], NY_ID=self.grid.ID.shape[2], NZ_ID=self.grid.ID.shape[3], NX_T=self.NX_T, NY_T=self.NY_T, NZ_T=self.NZ_T, NY_SRCINFO=4, NY_SRCWAVES=self.grid.iterations)
        self.prog = hip_check(hiprtc.hiprtcCreateProgram(source_update_voltage_source.encode(), b"update_voltage_source", 0, [], []))
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props,0))
        arch = props.gcnArchName
        cflags = [b"--offload-arch="+arch]
        err, = hiprtc.hiprtcCompileProgram(self.prog, len(cflags), cflags)
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(self.prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(self.prog, log))
            raise RuntimeError(log.decode())
        code_size = hip_check(hiprtc.hiprtcGetCodeSize(self.prog))
        code = bytearray(code_size)
        hip_check(hiprtc.hiprtcGetCode(self.prog, code))
        self.module = hip_check(hip.hipModuleLoadData(code))
        self.update_voltage_source_kernel = hip_check(hip.hipModuleGetFunction(self.module, b"update_voltage_source"))
        print(f"Compiling update_voltage_source Done")

    def compile_kernels_magnetic_sources(self):
        source_update_magnetic_dipole = update_magnetic_dipole.substitute(REAL=floattype, COMPLEX=floattype, N_updatecoeffsE=self.grid.updatecoeffsE.size, N_updatecoeffsH=self.grid.updatecoeffsH.size, NY_MATCOEFFS=self.grid.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=self.NY_MATDISPCOEFFS, NX_FIELDS=self.grid.nx + 1, NY_FIELDS=self.grid.ny + 1, NZ_FIELDS=self.grid.nz + 1, NX_ID=self.grid.ID.shape[1], NY_ID=self.grid.ID.shape[2], NZ_ID=self.grid.ID.shape[3], NX_T=self.NX_T, NY_T=self.NY_T, NZ_T=self.NZ_T, NY_SRCINFO=4, NY_SRCWAVES=self.grid.iterations)
        self.prog = hip_check(hiprtc.hiprtcCreateProgram(source_update_magnetic_dipole.encode(), b"update_magnetic_dipole", 0, [], []))
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props,0))
        arch = props.gcnArchName
        cflags = [b"--offload-arch="+arch]
        err, = hiprtc.hiprtcCompileProgram(self.prog, len(cflags), cflags)
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(self.prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(self.prog, log))
            raise RuntimeError(log.decode())
        code_size = hip_check(hiprtc.hiprtcGetCodeSize(self.prog))
        code = bytearray(code_size)
        hip_check(hiprtc.hiprtcGetCode(self.prog, code))
        self.module = hip_check(hip.hipModuleLoadData(code))
        self.update_magnetic_dipole_kernel = hip_check(hip.hipModuleGetFunction(self.module, b"update_magnetic_dipole"))
        print(f"Compiling update_magnetic_dipole Done")

    def compile_kernels_e_dispersive_A(self):
        source_update_electric_dispersive_A = update_electric_dispersive_A.substitute(GETREAL=self.get_real, REAL=floattype, COMPLEX=self.complextype, N_updatecoeffsE=self.grid.updatecoeffsE.size, N_updatecoeffsH=self.grid.updatecoeffsH.size, NY_MATCOEFFS=self.grid.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=self.NY_MATDISPCOEFFS, NX_FIELDS=self.grid.nx + 1, NY_FIELDS=self.grid.ny + 1, NZ_FIELDS=self.grid.nz + 1, NX_ID=self.grid.ID.shape[1], NY_ID=self.grid.ID.shape[2], NZ_ID=self.grid.ID.shape[3], NX_T=self.NX_T, NY_T=self.NY_T, NZ_T=self.NZ_T)
        self.prog = hip_check(hiprtc.hiprtcCreateProgram(source_update_electric_dispersive_A.encode(), b"update_electric_dispersive_A", 0, [], []))
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props,0))
        arch = props.gcnArchName
        cflags = [b"--offload-arch="+arch]
        err, = hiprtc.hiprtcCompileProgram(self.prog, len(cflags), cflags)
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(self.prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(self.prog, log))
            raise RuntimeError(log.decode())
        code_size = hip_check(hiprtc.hiprtcGetCodeSize(self.prog))
        code = bytearray(code_size)
        hip_check(hiprtc.hiprtcGetCode(self.prog, code))
        self.module = hip_check(hip.hipModuleLoadData(code))
        self.update_electric_dispersive_A_kernel = hip_check(hip.hipModuleGetFunction(self.module, b"update_electric_dispersive_A"))
        print(f"Compiling update_electric_dispersive_A Done")
    
    def compile_kernels_e_dispersive_B(self):
        source_update_electric_dispersive_B = update_electric_dispersive_B.substitute(GETREAL=self.get_real, REAL=floattype, COMPLEX=self.complextype, N_updatecoeffsE=self.grid.updatecoeffsE.size, N_updatecoeffsH=self.grid.updatecoeffsH.size, NY_MATCOEFFS=self.grid.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=self.NY_MATDISPCOEFFS, NX_FIELDS=self.grid.nx + 1, NY_FIELDS=self.grid.ny + 1, NZ_FIELDS=self.grid.nz + 1, NX_ID=self.grid.ID.shape[1], NY_ID=self.grid.ID.shape[2], NZ_ID=self.grid.ID.shape[3], NX_T=self.NX_T, NY_T=self.NY_T, NZ_T=self.NZ_T)
        self.prog = hip_check(hiprtc.hiprtcCreateProgram(source_update_electric_dispersive_B.encode(), b"update_electric_dispersive_B", 0, [], []))
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props,0))
        arch = props.gcnArchName
        cflags = [b"--offload-arch="+arch]
        err, = hiprtc.hiprtcCompileProgram(self.prog, len(cflags), cflags)
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(self.prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(self.prog, log))
            raise RuntimeError(log.decode())
        code_size = hip_check(hiprtc.hiprtcGetCodeSize(self.prog))
        code = bytearray(code_size)
        hip_check(hiprtc.hiprtcGetCode(self.prog, code))
        self.module = hip_check(hip.hipModuleLoadData(code))
        self.update_electric_dispersive_B_kernel = hip_check(hip.hipModuleGetFunction(self.module, b"update_electric_dispersive_B"))
        print(f"Compiling update_electric_dispersive_B Done")

    def free_resources(self):
        """Free resources allocated on the device."""
        hip_check(hip.hipFree(self.grid.ID_dev))
        hip_check(hip.hipFree(self.grid.Ex_dev))
        hip_check(hip.hipFree(self.grid.Ey_dev))
        hip_check(hip.hipFree(self.grid.Ez_dev))
        hip_check(hip.hipFree(self.grid.Hx_dev))
        hip_check(hip.hipFree(self.grid.Hy_dev))
        hip_check(hip.hipFree(self.grid.Hz_dev))
        hip_check(hip.hipModuleUnload(self.module))
        hip_check(hiprtc.hiprtcDestroyProgram(self.prog.createRef()))
        hip_check(hip.hipFree(self.grid.rxcoords_dev))
        hip_check(hip.hipFree(self.grid.rxs_dev))

    # def update_e_hip(self):
    #     hip_check(
    #         hip.hipModuleLaunchKernel(
    #             self.update_e_kernel,
    #             *self.grid_hip, # grid
    #             *self.block,  # self.block
    #             sharedMemBytes=128,
    #             stream=None,
    #             kernelParams=None,
    #             extra=(
    #                 ctypes.c_int(self.grid.nx),
    #                 ctypes.c_int(self.grid.ny),
    #                 ctypes.c_int(self.grid.nz),
    #                 self.grid.ID_dev,
    #                 self.grid.Ex_dev,
    #                 self.grid.Ey_dev,
    #                 self.grid.Ez_dev,
    #                 self.grid.Hx_dev,
    #                 self.grid.Hy_dev,
    #                 self.grid.Hz_dev,
    #                 self.grid.updatecoeffsE_d,
    #                 self.grid.updatecoeffsH_d,
    #             )
    #         )
    #     )

    # def update_m_hip(self):
    #     hip_check(
    #         hip.hipModuleLaunchKernel(
    #             self.update_m_kernel,
    #             *self.grid_hip, # grid
    #             *self.block,  # self.block
    #             sharedMemBytes=128,
    #             stream=None,
    #             kernelParams=None,
    #             extra=(
    #                 ctypes.c_int(self.grid.nx),
    #                 ctypes.c_int(self.grid.ny),
    #                 ctypes.c_int(self.grid.nz),
    #                 self.grid.ID_dev,
    #                 self.grid.Hx_dev,
    #                 self.grid.Hy_dev,
    #                 self.grid.Hz_dev,
    #                 self.grid.Ex_dev,
    #                 self.grid.Ey_dev,
    #                 self.grid.Ez_dev,
    #                 self.grid.updatecoeffsE_d,
    #                 self.grid.updatecoeffsH_d,
    #             )
    #         )
    #     )
        
    def update_hertzian_dipole_hip(self, iteration):
        (
            self.srcinfo1_dev,
            self.srcinfo2_dev,
            self.srcwaves_dev
        ) = htod_src_arrays(self.grid.hertziandipoles, self.grid)
        hip_check(
            hip.hipModuleLaunchKernel(
                self.update_hertzian_dipole_kernel,
                *self.grid_hip, # grid
                *self.block,  # self.block
                sharedMemBytes=128,
                stream=None,
                kernelParams=None,
                extra=(
                ctypes.c_int(len(self.grid.hertziandipoles)),
                ctypes.c_int(iteration),
                ctypes.c_float(self.grid.dx),
                ctypes.c_float(self.grid.dy),
                ctypes.c_float(self.grid.dz),
                self.srcinfo1_dev,
                self.srcinfo2_dev,
                self.srcwaves_dev,
                self.grid.ID_dev,
                self.grid.Ex_dev,
                self.grid.Ey_dev,
                self.grid.Ez_dev,
                self.grid.updatecoeffsE_d,
                self.grid.updatecoeffsH_d,
                )
            )
        )
    
    def store_outputs_hip(self, iteration):
        hip_check(
            hip.hipModuleLaunchKernel(
                self.store_outputs_kernel,
                *self.grid_hip, # grid
                *self.block,  # self.block
                sharedMemBytes=128,
                stream=None,
                kernelParams=None,
                extra=(
                    ctypes.c_int(len(self.grid.rxs)),
                    ctypes.c_int(iteration),
                    self.grid.rxcoords_dev,
                    self.grid.rxs_dev,
                    self.grid.Ex_dev,
                    self.grid.Ey_dev,
                    self.grid.Ez_dev,
                    self.grid.Hx_dev,
                    self.grid.Hy_dev,
                    self.grid.Hz_dev,
                )
            )
        )

    def update_voltage_source_hip(self, iteration):
        (
            self.srcinfo1_dev,
            self.srcinfo2_dev,
            self.srcwaves_dev
        ) = htod_src_arrays(self.grid.voltagesources, self.grid)
        hip_check(
            hip.hipModuleLaunchKernel(
                self.update_voltage_source_kernel,
                *self.grid_hip, # grid
                *self.block,  # self.block
                sharedMemBytes=128,
                stream=None,
                kernelParams=None,
                extra=(
                    ctypes.c_int(len(self.grid.voltagesources)),
                    ctypes.c_int(iteration),
                    ctypes.c_float(self.grid.dx),
                    ctypes.c_float(self.grid.dy),
                    ctypes.c_float(self.grid.dz),
                    self.srcinfo1_dev,
                    self.srcinfo2_dev,
                    self.srcwaves_dev,
                    self.grid.ID_dev,
                    self.grid.Ex_dev,
                    self.grid.Ey_dev,
                    self.grid.Ez_dev,
                    self.grid.updatecoeffsE_d,
                    self.grid.updatecoeffsH_d,
                )
            )
        )

    def update_magnetic_dipole_hip(self, iteration):
        (
            self.srcinfo1_dev,
            self.srcinfo2_dev,
            self.srcwaves_dev
        ) = htod_src_arrays(self.grid.magneticdipoles, self.grid)
        hip_check(
            hip.hipModuleLaunchKernel(
                self.update_magnetic_dipole_kernel,
                *self.grid_hip, # grid
                *self.block,  # self.block
                sharedMemBytes=128,
                stream=None,
                kernelParams=None,
                extra=(
                    ctypes.c_int(len(self.grid.magneticdipoles)),
                    ctypes.c_int(iteration),
                    ctypes.c_float(self.grid.dx),
                    ctypes.c_float(self.grid.dy),
                    ctypes.c_float(self.grid.dz),
                    self.srcinfo1_dev,
                    self.srcinfo2_dev,
                    self.srcwaves_dev,
                    self.grid.ID_dev,
                    self.grid.Hx_dev,
                    self.grid.Hy_dev,
                    self.grid.Hz_dev,
                    self.grid.updatecoeffsE_d,
                    self.grid.updatecoeffsH_d,
                )
            )
        )

    def update_electric_dispersive_A_hip(self):
        hip_check(
            hip.hipModuleLaunchKernel(
                self.update_electric_dispersive_A_kernel,
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
    def update_electric_dispersive_B_hip(self):
        hip_check(
            hip.hipModuleLaunchKernel(
                self.update_electric_dispersive_B_kernel,
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