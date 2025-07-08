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
logger = logging.getLogger(__name__)


class HIPUpdates(Updates[HIPGrid]):
    """HIP updates for the FDTD algorithm."""

    def __init__(self, G: HIPGrid):
        super().__init__(G)
        self.hip_manager = HipManager(G)

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
        self.hip_manager.update_m_hip()

    
    def update_magnetic_pml(self) -> None:
        """Updates magnetic field components with the PML correction."""
        #print("update_magnetic_pml not implemented in HIPUpdates")

    
    def update_magnetic_sources(self, iteration: int) -> None:
        """Updates magnetic field components from sources."""
        self.hip_manager.update_magnetic_dipole_hip(iteration)

    
    def update_electric_a(self) -> None:
        """Updates electric field components."""
        if config.get_model_config().materials["maxpoles"] == 0:
            self.hip_manager.update_e_hip()
        else:
            self.hip_manager.update_electric_dispersive_A_hip()

    
    def update_electric_pml(self) -> None:
        """Updates electric field components with the PML correction."""
        #print("update_electric_pml not implemented in HIPUpdates")

    
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
        self.hip_manager.update_electric_dispersive_B_hip()

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
        hip_check(hip.hipMemcpy(rxs, self.hip_manager.rxs_dev, rxs.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
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
