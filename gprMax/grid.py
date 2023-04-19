# Copyright (C) 2015-2023: The University of Edinburgh
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

from colorama import init
from colorama import Fore
from colorama import Style
init()
import numpy as np
np.seterr(invalid='raise')

from gprMax.constants import c
from gprMax.constants import floattype
from gprMax.constants import complextype
from gprMax.exceptions import GeneralError
from gprMax.materials import Material
from gprMax.pml import PML
from gprMax.utilities import fft_power
from gprMax.utilities import human_size
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
        i = self.nx
        j = self.ny
        k = self.nz
        e = (i * j * (k - 1)) + (j * k * (i - 1)) + (i * k * (j - 1))
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
    """
    Holds attributes associated with the entire grid. A convenient
    way for accessing regularly used parameters.
    """

    def __init__(self):
        self.inputfilename = ''
        self.inputdirectory = ''
        self.outputdirectory = ''
        self.title = ''
        self.messages = True
        self.progressbars = self.messages
        self.memoryusage = 0

        # Get information about host machine
        self.hostinfo = None

        # CPU - OpenMP threads
        self.nthreads = 0

        # GPU
        # Threads per block - electric and magnetic field updates
        self.tpb = (256, 1, 1)

        # GPU object
        self.gpu = None

        # Copy snapshot data from GPU to CPU during simulation
        # N.B. This will happen if the requested snapshots are too large to fit
        # on the memory of the GPU. If True this will slow performance significantly
        self.snapsgpu2cpu = False

        # Threshold (dB) down from maximum power (0dB) of main frequency used
        # to calculate highest frequency for numerical dispersion analysis
        self.highestfreqthres = 40
        # Maximum allowable percentage physical phase-velocity phase error
        self.maxnumericaldisp = 2
        # Minimum grid sampling of smallest wavelength for physical wave propagation
        self.mingridsampling = 3

        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.dt = 0
        self.mode = None
        self.iterations = 0
        self.timewindow = 0

        # Ordered dictionary required so that PMLs are always updated in the
        # same order. The order itself does not matter, however, if must be the
        # same from model to model otherwise the numerical precision from adding
        # the PML corrections will be different.
        self.pmlthickness = OrderedDict((key, 10) for key in PML.boundaryIDs)
        self.cfs = []
        self.pmls = []
        self.pmlformulation = 'HORIPML'

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
        """
        Initialise an array for volumetric material IDs (solid);
            boolean arrays for specifying whether materials can have dielectric smoothing (rigid);
            and an array for cell edge IDs (ID).
        Solid and ID arrays are initialised to free_space (one);
            rigid arrays to allow dielectric smoothing (zero).
        """
        self.solid = np.ones((self.nx, self.ny, self.nz), dtype=np.uint32)
        self.rigidE = np.zeros((12, self.nx, self.ny, self.nz), dtype=np.int8)
        self.rigidH = np.zeros((6, self.nx, self.ny, self.nz), dtype=np.int8)
        self.ID = np.ones((6, self.nx + 1, self.ny + 1, self.nz + 1), dtype=np.uint32)
        self.IDlookup = {'Ex': 0, 'Ey': 1, 'Ez': 2, 'Hx': 3, 'Hy': 4, 'Hz': 5}

    def initialise_field_arrays(self):
        """Initialise arrays for the electric and magnetic field components."""
        self.Ex = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=floattype)
        self.Ey = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=floattype)
        self.Ez = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=floattype)
        self.Hx = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=floattype)
        self.Hy = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=floattype)
        self.Hz = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=floattype)

    def initialise_std_update_coeff_arrays(self):
        """Initialise arrays for storing update coefficients."""
        self.updatecoeffsE = np.zeros((len(self.materials), 5), dtype=floattype)
        self.updatecoeffsH = np.zeros((len(self.materials), 5), dtype=floattype)

    def initialise_dispersive_arrays(self):
        """Initialise arrays for storing coefficients when there are dispersive materials present."""
        self.Tx = np.zeros((Material.maxpoles, self.nx + 1, self.ny + 1, self.nz + 1), dtype=complextype)
        self.Ty = np.zeros((Material.maxpoles, self.nx + 1, self.ny + 1, self.nz + 1), dtype=complextype)
        self.Tz = np.zeros((Material.maxpoles, self.nx + 1, self.ny + 1, self.nz + 1), dtype=complextype)
        self.updatecoeffsdispersive = np.zeros((len(self.materials), 3 * Material.maxpoles), dtype=complextype)

    def memory_estimate_basic(self):
        """Estimate the amount of memory (RAM) required to run a model."""

        stdoverhead = 50e6

        solidarray = self.nx * self.ny * self.nz * np.dtype(np.uint32).itemsize

        # 12 x rigidE array components + 6 x rigidH array components
        rigidarrays = (12 + 6) * self.nx * self.ny * self.nz * np.dtype(np.int8).itemsize

        # 6 x field arrays + 6 x ID arrays
        fieldarrays = (6 + 6) * (self.nx + 1) * (self.ny + 1) * (self.nz + 1) * np.dtype(floattype).itemsize

        # PML arrays
        pmlarrays = 0
        for (k, v) in self.pmlthickness.items():
            if v > 0:
                if 'x' in k:
                    pmlarrays += ((v + 1) * self.ny * (self.nz + 1))
                    pmlarrays += ((v + 1) * (self.ny + 1) * self.nz)
                    pmlarrays += (v * self.ny * (self.nz + 1))
                    pmlarrays += (v * (self.ny + 1) * self.nz)
                elif 'y' in k:
                    pmlarrays += (self.nx * (v + 1) * (self.nz + 1))
                    pmlarrays += ((self.nx + 1) * (v + 1) * self.nz)
                    pmlarrays += ((self.nx + 1) * v * self.nz)
                    pmlarrays += (self.nx * v * (self.nz + 1))
                elif 'z' in k:
                    pmlarrays += (self.nx * (self.ny + 1) * (v + 1))
                    pmlarrays += ((self.nx + 1) * self.ny * (v + 1))
                    pmlarrays += ((self.nx + 1) * self.ny * v)
                    pmlarrays += (self.nx * (self.ny + 1) * v)

        self.memoryusage = int(stdoverhead + fieldarrays + solidarray + rigidarrays + pmlarrays)

    def memory_check(self, snapsmemsize=0):
        """Check if the required amount of memory (RAM) is available on the host and GPU if specified.

        Args:
            snapsmemsize (int): amount of memory (bytes) required to store all requested snapshots
        """

        # Check if model can be built and/or run on host
        if self.memoryusage > self.hostinfo['ram']:
            raise GeneralError('Memory (RAM) required ~{} exceeds {} detected!\n'.format(human_size(self.memoryusage), human_size(self.hostinfo['ram'], a_kilobyte_is_1024_bytes=True)))

        # Check if model can be run on specified GPU if required
        if self.gpu is not None:
            if self.memoryusage - snapsmemsize > self.gpu.totalmem:
                raise GeneralError('Memory (RAM) required ~{} exceeds {} detected on specified {} - {} GPU!\n'.format(human_size(self.memoryusage), human_size(self.gpu.totalmem, a_kilobyte_is_1024_bytes=True), self.gpu.deviceID, self.gpu.name))

            # If the required memory without the snapshots will fit on the GPU then transfer and store snaphots on host
            if snapsmemsize != 0 and self.memoryusage - snapsmemsize < self.gpu.totalmem:
                self.snapsgpu2cpu = True

    def gpu_set_blocks_per_grid(self):
        """Set the blocks per grid size used for updating the electric and magnetic field arrays on a GPU."""
        self.bpg = (int(np.ceil(((self.nx + 1) * (self.ny + 1) * (self.nz + 1)) / self.tpb[0])), 1, 1)

    def gpu_initialise_arrays(self):
        """Initialise standard field arrays on GPU."""

        import pycuda.gpuarray as gpuarray

        self.ID_gpu = gpuarray.to_gpu(self.ID)
        self.Ex_gpu = gpuarray.to_gpu(np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=floattype))
        self.Ey_gpu = gpuarray.to_gpu(np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=floattype))
        self.Ez_gpu = gpuarray.to_gpu(np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=floattype))
        self.Hx_gpu = gpuarray.to_gpu(np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=floattype))
        self.Hy_gpu = gpuarray.to_gpu(np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=floattype))
        self.Hz_gpu = gpuarray.to_gpu(np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=floattype))

    def gpu_initialise_dispersive_arrays(self):
        """Initialise dispersive material coefficient arrays on GPU."""

        import pycuda.gpuarray as gpuarray

        self.Tx_gpu = gpuarray.to_gpu(self.Tx)
        self.Ty_gpu = gpuarray.to_gpu(self.Ty)
        self.Tz_gpu = gpuarray.to_gpu(self.Tz)
        self.updatecoeffsdispersive_gpu = gpuarray.to_gpu(self.updatecoeffsdispersive)


def dispersion_analysis(G):
    """
    Analysis of numerical dispersion (Taflove et al, 2005, p112) -
        worse case of maximum frequency and minimum wavelength

    Args:
        G (class): Grid class instance - holds essential parameters describing the model.

    Returns:
        results (dict): Results from dispersion analysis
    """

    # Physical phase velocity error (percentage); grid sampling density;
    # material with maximum permittivity; maximum significant frequency; error message
    results = {'deltavp': False, 'N': False, 'material': False, 'maxfreq': [], 'error': ''}

    # Find maximum significant frequency
    if G.waveforms:
        for waveform in G.waveforms:
            if waveform.type == 'sine' or waveform.type == 'contsine':
                results['maxfreq'].append(4 * waveform.freq)

            elif waveform.type == 'impulse':
                results['error'] = 'impulse waveform used.'

            else:
                # User-defined waveform
                if waveform.type == 'user':
                    iterations = G.iterations

                # Built-in waveform
                else:
                    # Time to analyse waveform - 4*pulse_width as using entire
                    # time window can result in demanding FFT
                    waveform.calculate_coefficients()
                    iterations = round_value(4 * waveform.chi / G.dt)
                    if iterations > G.iterations:
                        iterations = G.iterations

                waveformvalues = np.zeros(G.iterations)
                for iteration in range(G.iterations):
                    waveformvalues[iteration] = waveform.calculate_value(iteration * G.dt, G.dt)

                # Ensure source waveform is not being overly truncated before attempting any FFT
                if np.abs(waveformvalues[-1]) < np.abs(np.amax(waveformvalues)) / 100:
                    # FFT
                    freqs, power = fft_power(waveformvalues, G.dt)
                    # Get frequency for max power
                    freqmaxpower = np.where(np.isclose(power, 0))[0][0]

                    # Set maximum frequency to a threshold drop from maximum power, ignoring DC value
                    try:
                        freqthres = np.where(power[freqmaxpower:] < -G.highestfreqthres)[0][0] + freqmaxpower
                        results['maxfreq'].append(freqs[freqthres])
                    except ValueError:
                        results['error'] = 'unable to calculate maximum power from waveform, most likely due to undersampling.'

                # Ignore case where someone is using a waveform with zero amplitude, i.e. on a receiver
                elif waveform.amp == 0:
                    pass

                # If waveform is truncated don't do any further analysis
                else:
                    results['error'] = 'waveform does not fit within specified time window and is therefore being truncated.'
    else:
        results['error'] = 'no waveform detected.'

    if results['maxfreq']:
        results['maxfreq'] = max(results['maxfreq'])

        # Find minimum wavelength (material with maximum permittivity)
        maxer = 0
        matmaxer = ''
        for x in G.materials:
            if x.se != float('inf'):
                er = x.er
                # If there are dispersive materials calculate the complex relative permittivity
                # at maximum frequency and take the real part
                if x.poles > 0:
                    er = x.calculate_er(results['maxfreq'])
                    er = er.real
                if er > maxer:
                    maxer = er
                    matmaxer = x.ID
        results['material'] = next(x for x in G.materials if x.ID == matmaxer)

        # Minimum velocity
        minvelocity = c / np.sqrt(maxer)

        # Minimum wavelength
        minwavelength = minvelocity / results['maxfreq']

        # Maximum spatial step
        if '3D' in G.mode:
            delta = max(G.dx, G.dy, G.dz)
        elif '2D' in G.mode:
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

        # Check grid sampling will result in physical wave propagation
        if int(np.floor(results['N'])) >= G.mingridsampling:
            # Numerical phase velocity
            vp = np.pi / (results['N'] * np.arcsin((1 / S) * np.sin((np.pi * S) / results['N'])))

            # Physical phase velocity error (percentage)
            results['deltavp'] = (((vp * c) - c) / c) * 100

        # Store rounded down value of grid sampling density
        results['N'] = int(np.floor(results['N']))

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


def Ix(x, y, z, Hx, Hy, Hz, G):
    """Calculates the x-component of current at a grid position.

    Args:
        x, y, z (float): Coordinates of position in grid.
        Hx, Hy, Hz (memory view): numpy array of magnetic field values.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    if y == 0 or z == 0:
        Ix = 0
    else:
        Ix = G.dy * (Hy[x, y, z - 1] - Hy[x, y, z]) + G.dz * (Hz[x, y, z] - Hz[x, y - 1, z])

    return Ix


def Iy(x, y, z, Hx, Hy, Hz, G):
    """Calculates the y-component of current at a grid position.

    Args:
        x, y, z (float): Coordinates of position in grid.
        Hx, Hy, Hz (memory view): numpy array of magnetic field values.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    if x == 0 or z == 0:
        Iy = 0
    else:
        Iy = G.dx * (Hx[x, y, z] - Hx[x, y, z - 1]) + G.dz * (Hz[x - 1, y, z] - Hz[x, y, z])

    return Iy


def Iz(x, y, z, Hx, Hy, Hz, G):
    """Calculates the z-component of current at a grid position.

    Args:
        x, y, z (float): Coordinates of position in grid.
        Hx, Hy, Hz (memory view): numpy array of magnetic field values.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    if x == 0 or y == 0:
        Iz = 0
    else:
        Iz = G.dx * (Hx[x, y - 1, z] - Hx[x, y, z]) + G.dy * (Hy[x, y, z] - Hy[x - 1, y, z])

    return Iz
