# Copyright (C) 2015-2017: The University of Edinburgh
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

from collections import OrderedDict

from colorama import init, Fore, Style
init()
import numpy as np
np.seterr(invalid='raise')

from gprMax.constants import c, floattype, complextype
from gprMax.materials import Material
from gprMax.pml import PML
from gprMax.utilities import round_value


class Grid(object):
    """Generic grid/mesh."""

    def __init__(self, grid):
        self.nx = grid.shape[0]
        self.ny = grid.shape[1]
        self.nz = grid.shape[2]
        self.dx = 1
        self.dy = 1
        self.dz = 1
        self.i_max = self.nx - 1
        self.j_max = self.ny - 1
        self.k_max = self.nz - 1
        self.grid = grid

    def n_edges(self):
        l = self.nx
        m = self.ny
        n = self.nz
        e = (l * m * (n - 1)) + (m * n * (l - 1)) + (l * n * (m - 1))
        return e

    def n_nodes(self):
        return self.nx * self.ny * self.nz

    def n_cells(self):
        return (self.nx - 1) * (self.ny - 1) * (self.nz - 1)

    def get(self, i, j, k):
        return self.grid[i, j, k]

    def within_bounds(self, **kwargs):
        for co, val in kwargs.items():
            if val < 0 or val > getattr(self, 'n' + co):
                raise ValueError(co)

    def calculate_coord(self, coord, val):
        co = round_value(float(val) / getattr(self, 'd' + coord))
        return co


class FDTDGrid(Grid):
    """Holds attributes associated with the entire grid. A convenient way for accessing regularly used parameters."""

    def __init__(self):
        self.inputfilename = ''
        self.inputdirectory = ''
        self.title = ''
        self.messages = True
        self.tqdmdisable = False

        # Threshold (dB) down from maximum power (0dB) of main frequency used to calculate highest frequency for disperion analysis
        self.highestfreqthres = 60
        # Maximum allowable percentage physical phase-velocity phase error
        self.maxnumericaldisp = 2

        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.dt = 0
        self.dimension = None
        self.iterations = 0
        self.timewindow = 0
        self.nthreads = 0
        self.cfs = []
        self.pmlthickness = OrderedDict((key, 10) for key in PML.slabs)
        self.pmls = []
        self.materials = []
        self.mixingmodels = []
        self.averagevolumeobjects = True
        self.fractalvolumes = []
        self.geometryviews = []
        self.geometryobjectswrite = []
        self.waveforms = []
        self.voltagesources = []
        self.hertziandipoles = []
        self.magneticdipoles = []
        self.transmissionlines = []
        self.rxs = []
        self.srcsteps = [0, 0, 0]
        self.rxsteps = [0, 0, 0]
        self.snapshots = []

    def initialise_geometry_arrays(self):
        """Initialise an array for volumetric material IDs (solid); boolean arrays for specifying whether materials can have dielectric smoothing (rigid);
            and an array for cell edge IDs (ID). Solid and ID arrays are initialised to free_space (one); rigid arrays to allow dielectric smoothing (zero).
        """
        self.solid = np.ones((self.nx + 1, self.ny + 1, self.nz + 1), dtype=np.uint32)
        self.rigidE = np.zeros((12, self.nx + 1, self.ny + 1, self.nz + 1), dtype=np.int8)
        self.rigidH = np.zeros((6, self.nx + 1, self.ny + 1, self.nz + 1), dtype=np.int8)
        self.IDlookup = {'Ex': 0, 'Ey': 1, 'Ez': 2, 'Hx': 3, 'Hy': 4, 'Hz': 5}
        self.ID = np.ones((6, self.nx + 1, self.ny + 1, self.nz + 1), dtype=np.uint32)

    def initialise_field_arrays(self):
        """Initialise arrays for the electric and magnetic field components."""
        self.Ex = np.zeros((self.nx, self.ny + 1, self.nz + 1), dtype=floattype)
        self.Ey = np.zeros((self.nx + 1, self.ny, self.nz + 1), dtype=floattype)
        self.Ez = np.zeros((self.nx + 1, self.ny + 1, self.nz), dtype=floattype)
        self.Hx = np.zeros((self.nx + 1, self.ny, self.nz), dtype=floattype)
        self.Hy = np.zeros((self.nx, self.ny + 1, self.nz), dtype=floattype)
        self.Hz = np.zeros((self.nx, self.ny, self.nz + 1), dtype=floattype)

    def initialise_std_update_coeff_arrays(self):
        """Initialise arrays for storing update coefficients."""
        self.updatecoeffsE = np.zeros((len(self.materials), 5), dtype=floattype)
        self.updatecoeffsH = np.zeros((len(self.materials), 5), dtype=floattype)

    def initialise_dispersive_arrays(self):
        """Initialise arrays for storing coefficients when there are dispersive materials present."""
        self.Tx = np.zeros((Material.maxpoles, self.nx, self.ny + 1, self.nz + 1), dtype=complextype)
        self.Ty = np.zeros((Material.maxpoles, self.nx + 1, self.ny, self.nz + 1), dtype=complextype)
        self.Tz = np.zeros((Material.maxpoles, self.nx + 1, self.ny + 1, self.nz), dtype=complextype)
        self.updatecoeffsdispersive = np.zeros((len(self.materials), 3 * Material.maxpoles), dtype=complextype)


def dispersion_analysis(G):
    """Analysis of numerical dispersion (Taflove et al, 2005, p112) - worse case of maximum frequency and minimum wavelength

    Args:
        G (class): Grid class instance - holds essential parameters describing the model.

    Returns:
        results (dict): Results from dispersion analysis
    """

    # Physical phase velocity error (percentage); grid sampling density; material with maximum permittivity; maximum frequency of interest
    results = {'deltavp': False, 'N': False, 'material': False, 'maxfreq': False}

    # Find maximum frequency
    maxfreqs = []
    for waveform in G.waveforms:

        if waveform.type == 'sine' or waveform.type == 'contsine':
            maxfreqs.append(4 * waveform.freq)

        elif waveform.type == 'impulse':
            pass

        else:
            # User-defined waveform
            if waveform.type == 'user':
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

            # Ensure source waveform is not being overly truncated before attempting any FFT
            if np.abs(waveformvalues[-1]) < np.abs(np.amax(waveformvalues)) / 100:
                # Calculate magnitude of frequency spectra of waveform
                power = 10 * np.log10(np.abs(np.fft.fft(waveformvalues))**2)
                freqs = np.fft.fftfreq(power.size, d=G.dt)

                # Shift powers so that frequency with maximum power is at zero decibels
                power -= np.amax(power)
                
                # Get frequency for max power
                freqmaxpower = np.where(power[1::] == np.amax(power[1::]))[0][0]

                # Set maximum frequency to a threshold drop from maximum power, ignoring DC value
                freq = np.where((np.amax(power[freqmaxpower::]) - power[freqmaxpower::]) > G.highestfreqthres)[0][0] + 1
                maxfreqs.append(freqs[freq])

            else:
                print(Fore.RED + "\nWARNING: Duration of source waveform '{}' means it does not fit within specified time window and is therefore being truncated.".format(waveform.ID) + Style.RESET_ALL)

    if maxfreqs:
        results['maxfreq'] = max(maxfreqs)

        # Find minimum wavelength (material with maximum permittivity)
        maxer = 0
        matmaxer = ''
        for x in G.materials:
            if x.se != float('inf'):
                er = x.er
                if x.deltaer:
                    er += max(x.deltaer)
                if er > maxer:
                    maxer = er
                    matmaxer = x.ID
        results['material'] = next(x for x in G.materials if x.ID == matmaxer)

        # Minimum velocity
        minvelocity = c / np.sqrt(maxer)

        # Minimum wavelength
        minwavelength = minvelocity / results['maxfreq']

        # Maximum spatial step
        if G.dimension == '3D':
            delta = max(G.dx, G.dy, G.dz)
        elif G.dimension == '2D':
            if G.nx == 1:
                delta = max(G.dy, G.dz)
            elif G.ny == 1:
                delta = max(G.dx, G.dz)
            elif G.nz == 1:
                delta = max(G.dx, G.dy)

        # Courant stability factor
        S = (c * G.dt) / delta

        # Grid sampling density
        results['N'] = minwavelength / delta

        # If there is an invalid inverse sine value set to maximum, i.e. pi/2
        try:
            tmp = np.arcsin((1 / S) * np.sin((np.pi * S) / results['N']))
        except FloatingPointError:
            tmp = np.pi / 2

        # Numerical phase velocity
        vp = np.pi / (results['N'] * tmp)

        # Physical phase velocity error (percentage)
        results['deltavp'] = (((vp * c) - c) / c) * 100

    return results


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

    else:
        Iz = G.dx * (Hx[x, y - 1, z] - Hx[x, y, z]) + G.dy * (Hy[x, y, z] - Hy[x - 1, y, z])

    return Iz
