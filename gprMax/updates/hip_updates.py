import logging
from importlib import import_module

import humanize
import numpy as np
from jinja2 import Environment, PackageLoader

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
from ..hip_code.load_hip_kernals import update_e_hip
logger = logging.getLogger(__name__)


class HIPUpdates(Updates[HIPGrid]):
    """HIP updates for the FDTD algorithm."""

    def __init__(self, G: HIPGrid):
        self.G = G
    
    def store_outputs(self, iteration: int) -> None:
        """Stores field component values for every receiver and transmission line."""
        #print("store_outputs not implemented in HIPUpdates")

    
    def store_snapshots(self, iteration: int) -> None:
        """Stores any snapshots.

        Args:
            iteration: int for iteration number.
        """
        #print("store_snapshots not implemented in HIPUpdates")


    
    def update_magnetic(self) -> None:
        """Updates magnetic field components."""
        #print("update_magnetic not implemented in HIPUpdates")

    
    def update_magnetic_pml(self) -> None:
        """Updates magnetic field components with the PML correction."""
        #print("update_magnetic_pml not implemented in HIPUpdates")

    
    def update_magnetic_sources(self, iteration: int) -> None:
        """Updates magnetic field components from sources."""
        #print("update_magnetic_sources not implemented in HIPUpdates")

    
    def update_electric_a(self) -> None:
        """Updates electric field components."""
        update_e_hip(self.G)
        print("update_electric_a on HIP")

    
    def update_electric_pml(self) -> None:
        """Updates electric field components with the PML correction."""
        #print("update_electric_pml not implemented in HIPUpdates")

    
    def update_electric_sources(self, iteration: int) -> None:
        """Updates electric field components from sources -
        update any Hertzian dipole sources last.
        """
        #print("update_electric_sources not implemented in HIPUpdates")

    
    def update_electric_b(self) -> None:
        """If there are any dispersive materials do 2nd part of dispersive
        update - it is split into two parts as it requires present and
        updated electric field values. Therefore it can only be completely
        updated after the electric field has been updated by the PML and
        source updates.
        """
        #print("update_electric_b not implemented in HIPUpdates")

    def time_start(self) -> None:
        """Starts timer used to calculate solving time for model."""
        #print("time_start not implemented in HIPUpdates")

    def calculate_solve_time(self) -> float:
        """Calculates solving time for model."""
        #print("calculate_solve_time not implemented in HIPUpdates")
        return 0.0

    def finalise(self) -> None:
        """Finalise the updates, releasing any resources."""
        #print("finalise not implemented in HIPUpdates")

    def cleanup(self) -> None:
        """Cleanup the updates, releasing any resources."""
        #print("cleanup not implemented in HIPUpdates")

    def calculate_memory_used(self, iteration: int) -> int:
        #print("calculate_memory_used not implemented in HIPUpdates")
        return 0
