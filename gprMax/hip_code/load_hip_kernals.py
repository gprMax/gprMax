from hip import hip, hiprtc
from ..utilities.utilities import hip_check
from .hip_source import update_e
import numpy as np
import ctypes
from .. import config
from gprMax.grid.hip_grid import HIPGrid
floattype = 'float'

class HipManager:
    def __init__(self, G: HIPGrid):
        self.G = G
        self.module = None
        self.kernel = None
        self.prog = None
        self.update_e_kernel = None
        if config.get_model_config().materials["maxpoles"] > 0:
            self.NY_MATDISPCOEFFS = self.G.updatecoeffsdispersive.shape[1]
            self.NX_T = self.G.Tx.shape[1]
            self.NY_T = self.G.Tx.shape[2]
            self.NZ_T = self.G.Tx.shape[3]
        else:  # Set to one any substitutions for dispersive materials.
            self.NY_MATDISPCOEFFS = 1
            self.NX_T = 1
            self.NY_T = 1
            self.NZ_T = 1
        self.compile_kernels()

    
    def compile_kernels(self):
        source = update_e.substitute(REAL=floattype, COMPLEX=floattype, N_updatecoeffsE=self.G.updatecoeffsE.size, N_updatecoeffsH=self.G.updatecoeffsH.size, NY_MATCOEFFS=self.G.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=self.NY_MATDISPCOEFFS, NX_FIELDS=self.G.nx + 1, NY_FIELDS=self.G.ny + 1, NZ_FIELDS=self.G.nz + 1, NX_ID=self.G.ID.shape[1], NY_ID=self.G.ID.shape[2], NZ_ID=self.G.ID.shape[3], NX_T=self.NX_T, NY_T=self.NY_T, NZ_T=self.NZ_T)
        self.prog = hip_check(hiprtc.hiprtcCreateProgram(source.encode(), b"update_e", 0, [], []))
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props,0))
        arch = props.gcnArchName

        print(f"Compiling kernel for {arch}")

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
        self.update_e_kernel = hip_check(hip.hipModuleGetFunction(self.module, b"update_e"))
        print(f"Compiling Done")
        self.ID_d = hip_check(hip.hipMalloc(self.G.ID.nbytes))
        self.Ex_d = hip_check(hip.hipMalloc(self.G.Ex.nbytes))
        self.Ey_d = hip_check(hip.hipMalloc(self.G.Ey.nbytes))
        self.Ez_d = hip_check(hip.hipMalloc(self.G.Ez.nbytes))
        self.Hx_d = hip_check(hip.hipMalloc(self.G.Hx.nbytes))
        self.Hy_d = hip_check(hip.hipMalloc(self.G.Hy.nbytes))
        self.Hz_d = hip_check(hip.hipMalloc(self.G.Hz.nbytes))
        hip_check(hip.hipMemcpy(self.ID_d, self.G.ID, self.G.ID.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.Ex_d, self.G.Ex, self.G.Ex.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.Ey_d, self.G.Ey, self.G.Ey.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.Ez_d, self.G.Ez, self.G.Ez.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.Hx_d, self.G.Hx, self.G.Hx.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.Hy_d, self.G.Hy, self.G.Hy.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.Hz_d, self.G.Hz, self.G.Hz.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))



    def update_e_hip(self):

        block = hip.dim3(x=8, y=8, z=8)
        grid = hip.dim3(x=32, y=32, z=32)

        hip_check(
            hip.hipModuleLaunchKernel(
                self.update_e_kernel,
                *grid, # grid
                *block,  # block
                sharedMemBytes=128,
                stream=None,
                kernelParams=None,
                extra=(
                    ctypes.c_int(self.G.nx),
                    ctypes.c_int(self.G.ny),
                    ctypes.c_int(self.G.nz),
                    self.ID_d,
                    self.Ex_d,
                    self.Ey_d,
                    self.Ez_d,
                    self.Hx_d,
                    self.Hy_d,
                    self.Hz_d,
                )
            )
        )
        # hip_check(hip.hipMemcpy(self.G.Ex, Ex_d, self.G.Ex.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
        # hip_check(hip.hipMemcpy(self.G.Ey, Ey_d, self.G.Ey.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
        # hip_check(hip.hipMemcpy(self.G.Ez, Ez_d, self.G.Ez.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
        # hip_check(hip.hipMemcpy(self.G.Hx, Hx_d, self.G.Hx.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
        # hip_check(hip.hipMemcpy(self.G.Hy, Hy_d, self.G.Hy.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
        # hip_check(hip.hipMemcpy(self.G.Hz, Hz_d, self.G.Hz.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
        # hip_check(hip.hipMemcpy(self.G.ID, ID_d, self.G.ID.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
        # hip_check(hip.hipFree(ID_d))
        # hip_check(hip.hipFree(Ex_d))
        # hip_check(hip.hipFree(Ey_d))
        # hip_check(hip.hipFree(Ez_d))
        # hip_check(hip.hipFree(Hx_d))
        # hip_check(hip.hipFree(Hy_d))
        # hip_check(hip.hipFree(Hz_d))
        # hip_check(hip.hipModuleUnload(self.module))
        # hip_check(hiprtc.hiprtcDestroyProgram(self.prog.createRef()))
