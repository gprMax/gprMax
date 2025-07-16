from importlib import import_module

import numpy as np

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.pml import HIPPML
from ..utilities.utilities import hip_check
from hip import hip, hiprtc
from gprMax.receivers import dtoh_rx_array, htod_rx_arrays

class HIPGrid(FDTDGrid):
    def __init__(self):
        super().__init__()
        self.initialise_dispersive_arrays()
        self.initialise_dispersive_update_coeff_array()




    def _construct_pml(self, pml_ID: str, thickness: int) -> HIPPML:
        return super()._construct_pml(pml_ID, thickness, HIPPML)
    
    def htod_geometry_arrays(self):
        """Initialise an array for cell edge IDs (ID) on compute device."""
        self.ID_dev = hip_check(hip.hipMalloc(self.ID.nbytes))
        hip_check(hip.hipMemcpy(self.ID_dev, self.ID, self.ID.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        self.rxcoords_dev, self.rxs_dev = htod_rx_arrays(self)

    def htod_field_arrays(self):
        self.Ex_dev = hip_check(hip.hipMalloc(self.Ex.nbytes))
        self.Ey_dev = hip_check(hip.hipMalloc(self.Ey.nbytes))
        self.Ez_dev = hip_check(hip.hipMalloc(self.Ez.nbytes))
        self.Hx_dev = hip_check(hip.hipMalloc(self.Hx.nbytes))
        self.Hy_dev = hip_check(hip.hipMalloc(self.Hy.nbytes))
        self.Hz_dev = hip_check(hip.hipMalloc(self.Hz.nbytes))
        self.updatecoeffsE_d = hip_check(hip.hipMalloc(self.updatecoeffsE.nbytes))
        self.updatecoeffsH_d = hip_check(hip.hipMalloc(self.updatecoeffsH.nbytes))
        hip_check(hip.hipMemcpy(self.Ex_dev, self.Ex, self.Ex.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.Ey_dev, self.Ey, self.Ey.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.Ez_dev, self.Ez, self.Ez.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.Hx_dev, self.Hx, self.Hx.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.Hy_dev, self.Hy, self.Hy.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.Hz_dev, self.Hz, self.Hz.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.updatecoeffsE_d, self.updatecoeffsE, self.updatecoeffsE.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.updatecoeffsH_d, self.updatecoeffsH, self.updatecoeffsH.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

    def htod_dispersive_arrays(self):
        self.Tx_d = hip_check(hip.hipMalloc(self.Tx.nbytes))
        self.Ty_d = hip_check(hip.hipMalloc(self.Ty.nbytes))
        self.Tz_d = hip_check(hip.hipMalloc(self.Tz.nbytes))
        self.updatecoeffsdispersive_dev = hip_check(hip.hipMalloc(self.updatecoeffsdispersive.nbytes))
        hip_check(hip.hipMemcpy(self.Tx_d, self.Tx, self.Tx.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.Ty_d, self.Ty, self.Ty.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.Tz_d, self.Tz, self.Tz.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(self.updatecoeffsdispersive_dev, self.updatecoeffsdispersive, self.updatecoeffsdispersive.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))





