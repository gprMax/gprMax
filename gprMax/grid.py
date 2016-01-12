# Copyright (C) 2015-2016: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
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

import numpy as np

from gprMax.constants import floattype, complextype
from gprMax.materials import Material


class FDTDGrid:
    """Holds attributes associated with the entire grid. A convenient way for accessing regularly used parameters."""
    
    def __init__(self):
        self.inputdirectory = ''
        self.title = ''
        self.messages = True
        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.dt = 0
        self.iterations = 0
        self.timewindow = 0
        self.nthreads = 0
        self.cfs = []
        self.pmlthickness = (10, 10, 10, 10, 10, 10)
        self.pmls = []
        self.materials = []
        self.mixingmodels = []
        self.averagevolumeobjects = True
        self.fractalvolumes = []
        self.geometryviews = []
        self.waveforms = []
        self.voltagesources = []
        self.hertziandipoles = []
        self.magneticdipoles = []
        self.transmissionlines = []
        self.txs = [] # Only used for converting old output files to HDF5 format
        self.srcstepx = 0
        self.srcstepy = 0
        self.srcstepz = 0
        self.rxstepx = 0
        self.rxstepy = 0
        self.rxstepz = 0
        self.rxs = []
        self.snapshots = []
        
    def initialise_std_arrays(self):
        """Initialise an array for volumetric material IDs (solid); boolean arrays for specifying whether materials can have dielectric smoothing (rigid);
            an array for cell edge IDs (ID); and arrays for the electric and magnetic field components. Solid and ID arrays are initialised to free_space (one); rigid arrays
            to allow dielectric smoothing (zero).
        """
        self.solid = np.ones((self.nx + 1, self.ny + 1, self.nz + 1), dtype=np.uint32)
        self.rigidE = np.zeros((12, self.nx + 1, self.ny + 1, self.nz + 1), dtype=np.int8)
        self.rigidH = np.zeros((6, self.nx + 1, self.ny + 1, self.nz + 1), dtype=np.int8)
        self.ID = np.ones((6, self.nx + 1, self.ny + 1, self.nz + 1), dtype=np.uint32)
        self.Ex = np.zeros((self.nx, self.ny + 1, self.nz + 1), dtype=floattype)
        self.Ey = np.zeros((self.nx + 1, self.ny, self.nz + 1), dtype=floattype)
        self.Ez = np.zeros((self.nx + 1, self.ny + 1, self.nz), dtype=floattype)
        self.Hx = np.zeros((self.nx + 1, self.ny, self.nz), dtype=floattype)
        self.Hy = np.zeros((self.nx, self.ny + 1, self.nz), dtype=floattype)
        self.Hz = np.zeros((self.nx, self.ny, self.nz + 1), dtype=floattype)
    
    def initialise_std_updatecoeff_arrays(self, nummaterials):
        """Initialise arrays for storing update coefficients.
            
        Args:
            nummaterials (int): Number of materials present in the model.
        """
        self.updatecoeffsE = np.zeros((nummaterials, 5), dtype=floattype)
        self.updatecoeffsH = np.zeros((nummaterials, 5), dtype=floattype)

    def initialise_dispersive_arrays(self, nummaterials):
        """Initialise arrays for storing coefficients when there are dispersive materials present.
            
        Args:
            nummaterials (int): Number of materials present in the model.
        """
        self.Tx = np.zeros((Material.maxpoles, self.nx, self.ny + 1, self.nz + 1), dtype=complextype)
        self.Ty = np.zeros((Material.maxpoles, self.nx + 1, self.ny, self.nz + 1), dtype=complextype)
        self.Tz = np.zeros((Material.maxpoles, self.nx + 1, self.ny + 1, self.nz), dtype=complextype)
        self.updatecoeffsdispersive = np.zeros((nummaterials, 3 * Material.maxpoles), dtype=complextype)


def Ix(x, y, z, Hy, Hz, G):
    """Calculates the x-component of current at a grid position.
            
    Args:
        x, y, z (float): Coordinates of position in grid.
        Hy, Hz (memory view): numpy array of magnetic field values.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    if y == 0 or z == 0:
        Ix = 0
        return Ix
    
    else:
        Ix = G.dy * (Hy[x, y, z - 1] - Hy[x, y, z]) + G.dz * (Hz[x, y, z] - Hz[x, y - 1, z])
        return Ix

def Iy(x, y, z, Hx, Hz, G):
    """Calculates the y-component of current at a grid position.
            
    Args:
        x, y, z (float): Coordinates of position in grid.
        Hx, Hz (memory view): numpy array of magnetic field values.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    if x == 0 or z == 0:
        Iy = 0
        return Iy
    
    else:
        Iy = G.dx * (Hx[x, y, z] - Hx[x, y, z - 1]) + G.dz * (Hz[x - 1, y, z] - Hz[x, y, z])
        return Iy

def Iz(x, y, z, Hx, Hy, G):
    """Calculates the z-component of current at a grid position.
            
    Args:
        x, y, z (float): Coordinates of position in grid.
        Hx, Hy (memory view): numpy array of magnetic field values.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    if x == 0 or y == 0:
        Iz = 0
        return Iz
    
    else:
        Iz = G.dx * (Hx[x, y - 1, z] - Hx[x, y, z]) + G.dy * (Hy[x, y, z] - Hy[x - 1, y, z])
        return Iz



