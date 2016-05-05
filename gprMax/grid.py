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
import matplotlib.pyplot as plt

from gprMax.constants import c, floattype, complextype
from gprMax.materials import Material


class FDTDGrid:
    """Holds attributes associated with the entire grid. A convenient way for accessing regularly used parameters."""
    
    def __init__(self):
        self.inputfilename = ''
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
        self.pmlthickness = 10
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
        self.IDlookup = {'Ex': 0, 'Ey': 1, 'Ez': 2, 'Hx': 3, 'Hy': 4, 'Hz': 5}
        self.ID = np.ones((6, self.nx + 1, self.ny + 1, self.nz + 1), dtype=np.uint32)
        self.Ex = np.zeros((self.nx, self.ny + 1, self.nz + 1), dtype=floattype)
        self.Ey = np.zeros((self.nx + 1, self.ny, self.nz + 1), dtype=floattype)
        self.Ez = np.zeros((self.nx + 1, self.ny + 1, self.nz), dtype=floattype)
        self.Hx = np.zeros((self.nx + 1, self.ny, self.nz), dtype=floattype)
        self.Hy = np.zeros((self.nx, self.ny + 1, self.nz), dtype=floattype)
        self.Hz = np.zeros((self.nx, self.ny, self.nz + 1), dtype=floattype)
    
    def initialise_std_updatecoeff_arrays(self):
        """Initialise arrays for storing update coefficients."""
        self.updatecoeffsE = np.zeros((len(self.materials), 5), dtype=floattype)
        self.updatecoeffsH = np.zeros((len(self.materials), 5), dtype=floattype)

    def initialise_dispersive_arrays(self):
        """Initialise arrays for storing coefficients when there are dispersive materials present."""
        self.Tx = np.zeros((Material.maxpoles, self.nx, self.ny + 1, self.nz + 1), dtype=complextype)
        self.Ty = np.zeros((Material.maxpoles, self.nx + 1, self.ny, self.nz + 1), dtype=complextype)
        self.Tz = np.zeros((Material.maxpoles, self.nx + 1, self.ny + 1, self.nz), dtype=complextype)
        self.updatecoeffsdispersive = np.zeros((len(self.materials), 3 * Material.maxpoles), dtype=complextype)


def dispersion_check(G):
    """Check for potential numerical dispersion. Is the smallest wavelength present in the simulation discretised by at least a factor of 10
        
    Args:
        G (class): Grid class instance - holds essential parameters describing the model.
    
    Returns:
        resolution (float): Potential numerical dispersion
    """
    
    # Minimum number of spatial steps to resolve smallest wavelength
    resolvedsteps = 10
    
    # Find maximum frequency
    maxfreqs = []
    for waveform in G.waveforms:
        
        # User-defined waveform
        if waveform.uservalues is not None:
            waveformvalues = waveform.uservalues
        
        # Built-in waveform
        else:
            time = np.linspace(0, 1, G.iterations)
            time *= (G.iterations * G.dt)
            waveformvalues = np.zeros(len(time))
            timeiter = np.nditer(time, flags=['c_index'])

            while not timeiter.finished:
                waveformvalues[timeiter.index] = waveform.calculate_value(timeiter[0], G.dt)
                timeiter.iternext()

        if waveform.type == 'sine' or waveform.type == 'contsine':
            maxfreqs.append(4 * waveform.freq)
        
        elif waveform.type =='impulse':
            pass
        
        else:
            # Calculate magnitude of frequency spectra of waveform
            power = 20 * np.log10(np.abs(np.fft.fft(waveformvalues))**2)
            freqs = np.fft.fftfreq(power.size, d=G.dt)

            # Shift powers so that frequency with maximum power is at zero decibels
            power -= np.amax(power)
            
            # Set maximum frequency to -60dB from maximum power
            freq = np.where((np.amax(power[1::]) - power[1::]) > 60)[0][0] + 1
            maxfreqs.append(freqs[freq])

    if maxfreqs:
        maxfreq = max(maxfreqs)
        
        # Find minimum wavelength
        ers = [material.er for material in G.materials]
        maxer = max(ers)

        # Minimum velocity
        minvelocity = c / np.sqrt(maxer)
        
        # Minimum wavelength
        minwavelength = minvelocity / maxfreq
        
        # Resolution of minimum wavelength
        resolution = minwavelength / resolvedsteps

    else:
        resolution = 0

    return resolution


def get_other_directions(direction):
    """Return the two other directions from x, y, z given a single direction
    
    Args:
        direction (str): Component x, y or z
        
    Returns:
        (tuple): Two directions from x, y, z
    """

    directions = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}

    return directions[direction]


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



