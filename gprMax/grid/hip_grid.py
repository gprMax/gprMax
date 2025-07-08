from importlib import import_module

import numpy as np

from gprMax.grid.fdtd_grid import FDTDGrid

class HIPGrid(FDTDGrid):
    def __init__(self):
        super().__init__()
        self.initialise_dispersive_arrays()
        self.initialise_dispersive_update_coeff_array()
