from hip import hip, hiprtc
from ..utilities.utilities import hip_check
from .hip_source import update_e
import numpy as np
import ctypes
from .. import config
floattype = 'float'

def update_e_hip(G):
    if config.get_model_config().materials["maxpoles"] > 0:
        NY_MATDISPCOEFFS = G.updatecoeffsdispersive.shape[1]
        NX_T = G.Tx.shape[1]
        NY_T = G.Tx.shape[2]
        NZ_T = G.Tx.shape[3]
    else:  # Set to one any substitutions for dispersive materials.
        NY_MATDISPCOEFFS = 1
        NX_T = 1
        NY_T = 1
        NZ_T = 1

    source = update_e.substitute(REAL=floattype, COMPLEX=floattype, N_updatecoeffsE=G.updatecoeffsE.size, N_updatecoeffsH=G.updatecoeffsH.size, NY_MATCOEFFS=G.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=NY_MATDISPCOEFFS, NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1, NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3], NX_T=NX_T, NY_T=NY_T, NZ_T=NZ_T)

    prog = hip_check(hiprtc.hiprtcCreateProgram(source.encode(), b"update_e", 0, [], []))

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props,0))
    arch = props.gcnArchName

    print(f"Compiling kernel for {arch}")

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
    kernel = hip_check(hip.hipModuleGetFunction(module, b"update_e"))
    print(f"Compiling Done")

    ID_d = hip_check(hip.hipMalloc(G.ID.nbytes))
    Ex_d = hip_check(hip.hipMalloc(G.Ex.nbytes))
    Ey_d = hip_check(hip.hipMalloc(G.Ey.nbytes))
    Ez_d = hip_check(hip.hipMalloc(G.Ez.nbytes))
    Hx_d = hip_check(hip.hipMalloc(G.Hx.nbytes))
    Hy_d = hip_check(hip.hipMalloc(G.Hy.nbytes))
    Hz_d = hip_check(hip.hipMalloc(G.Hz.nbytes))
    hip_check(hip.hipMemcpy(ID_d, G.ID, G.ID.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(Ex_d, G.Ex, G.Ex.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(Ey_d, G.Ey, G.Ey.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(Ez_d, G.Ez, G.Ez.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(Hx_d, G.Hx, G.Hx.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(Hy_d, G.Hy, G.Hy.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(Hz_d, G.Hz, G.Hz.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

    block = hip.dim3(x=8, y=8, z=8)
    grid = hip.dim3(x=4, y=4, z=4)
    
    hip_check(
        hip.hipModuleLaunchKernel(
            kernel,
            *grid, # grid
            *block,  # block
            sharedMemBytes=128,
            stream=None,
            kernelParams=None,
            extra=(
                ctypes.c_int(G.nx),
                ctypes.c_int(G.ny),
                ctypes.c_int(G.nz),
                ID_d,
                Ex_d,
                Ey_d,
                Ez_d,
                Hx_d,
                Hy_d,
                Hz_d,
            )
        )
    )
    hip_check(hip.hipMemcpy(G.Ex, Ex_d, G.Ex.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    hip_check(hip.hipMemcpy(G.Ey, Ey_d, G.Ey.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    hip_check(hip.hipMemcpy(G.Ez, Ez_d, G.Ez.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    hip_check(hip.hipMemcpy(G.Hx, Hx_d, G.Hx.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    hip_check(hip.hipMemcpy(G.Hy, Hy_d, G.Hy.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    hip_check(hip.hipMemcpy(G.Hz, Hz_d, G.Hz.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    hip_check(hip.hipMemcpy(G.ID, ID_d, G.ID.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    hip_check(hip.hipFree(ID_d))
    hip_check(hip.hipFree(Ex_d))
    hip_check(hip.hipFree(Ey_d))
    hip_check(hip.hipFree(Ez_d))
    hip_check(hip.hipFree(Hx_d))
    hip_check(hip.hipFree(Hy_d))
    hip_check(hip.hipFree(Hz_d))
    hip_check(hip.hipModuleUnload(module))
    hip_check(hiprtc.hiprtcDestroyProgram(prog.createRef()))
