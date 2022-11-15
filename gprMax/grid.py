# Copyright (C) 2015-2022: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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

import decimal as d
from collections import OrderedDict

import numpy as np

import gprMax.config as config

from .pml import PML
from .utilities.utilities import fft_power, round_value

np.seterr(invalid='raise')


class FDTDGrid:
    """Holds attributes associated with entire grid. A convenient way for
        accessing regularly used parameters.
    """

    def __init__(self):
        self.title = ''
        self.name = 'main_grid'
        self.mem_use = 0

        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.dt = 0
        self.iteration = 0
        self.iterations = 0
        self.timewindow = 0

        # PML parameters - set some defaults to use if not user provided
        self.pmls = {}
        self.pmls['formulation'] = 'HORIPML'
        self.pmls['cfs'] = []
        self.pmls['slabs'] = []
        # Ordered dictionary required so that PMLs are always updated in the
        # same order. The order itself does not matter, however, if must be the
        # same from model to model otherwise the numerical precision from adding
        # the PML corrections will be different.
        self.pmls['thickness'] = OrderedDict((key, 10) for key in PML.boundaryIDs)
        
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
        self.subgrids = []

    def within_bounds(self, p):
        if p[0] < 0 or p[0] > self.nx:
            raise ValueError('x')
        if p[1] < 0 or p[1] > self.ny:
            raise ValueError('y')
        if p[2] < 0 or p[2] > self.nz:
            raise ValueError('z')

    def discretise_point(self, p):
        x = round_value(float(p[0]) / self.dx)
        y = round_value(float(p[1]) / self.dy)
        z = round_value(float(p[2]) / self.dz)
        return (x, y, z)

    def round_to_grid(self, p):
        p = self.discretise_point(p)
        p_r = (p[0] * self.dx,
               p[1] * self.dy,
               p[2] * self.dz)
        return p_r

    def within_pml(self, p):
        if (p[0] < self.pmls['thickness']['x0'] or
            p[0] > self.nx - self.pmls['thickness']['xmax'] or
            p[1] < self.pmls['thickness']['y0'] or
            p[1] > self.ny - self.pmls['thickness']['ymax'] or
            p[2] < self.pmls['thickness']['z0'] or
            p[2] > self.nz - self.pmls['thickness']['zmax']):
            return True
        else:
            return False

    def initialise_geometry_arrays(self):
        """Initialise an array for volumetric material IDs (solid);
            boolean arrays for specifying whether materials can have dielectric 
            smoothing (rigid); and an array for cell edge IDs (ID).
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
        self.Ex = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1),
                            dtype=config.sim_config.dtypes['float_or_double'])
        self.Ey = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1),
                            dtype=config.sim_config.dtypes['float_or_double'])
        self.Ez = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1),
                            dtype=config.sim_config.dtypes['float_or_double'])
        self.Hx = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1),
                            dtype=config.sim_config.dtypes['float_or_double'])
        self.Hy = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1),
                            dtype=config.sim_config.dtypes['float_or_double'])
        self.Hz = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1),
                            dtype=config.sim_config.dtypes['float_or_double'])

    def initialise_std_update_coeff_arrays(self):
        """Initialise arrays for storing update coefficients."""
        self.updatecoeffsE = np.zeros((len(self.materials), 5),
                                       dtype=config.sim_config.dtypes['float_or_double'])
        self.updatecoeffsH = np.zeros((len(self.materials), 5),
                                       dtype=config.sim_config.dtypes['float_or_double'])

    def initialise_dispersive_arrays(self):
        """Initialise field arrays when there are dispersive materials present."""
        self.Tx = np.zeros((config.get_model_config().materials['maxpoles'],
                            self.nx + 1, self.ny + 1, self.nz + 1), 
                            dtype=config.get_model_config().materials['dispersivedtype'])
        self.Ty = np.zeros((config.get_model_config().materials['maxpoles'],
                            self.nx + 1, self.ny + 1, self.nz + 1), 
                            dtype=config.get_model_config().materials['dispersivedtype'])
        self.Tz = np.zeros((config.get_model_config().materials['maxpoles'],
                            self.nx + 1, self.ny + 1, self.nz + 1), 
                            dtype=config.get_model_config().materials['dispersivedtype'])

    def initialise_dispersive_update_coeff_array(self):
        """Initialise array for storing update coefficients when there are dispersive
            materials present.
        """
        self.updatecoeffsdispersive = np.zeros((len(self.materials), 3 *
                                                config.get_model_config().materials['maxpoles']),
                                                dtype=config.get_model_config().materials['dispersivedtype'])

    def reset_fields(self):
        """Clear arrays for field components and PMLs."""
        # Clear arrays for field components
        self.initialise_field_arrays()
        if config.get_model_config().materials['maxpoles'] > 0:
            self.initialise_dispersive_arrays()

        # Clear arrays for fields in PML
        for pml in self.pmls['slabs']:
            pml.initialise_field_arrays()

    def mem_est_basic(self):
        """Estimate the amount of memory (RAM) required for grid arrays.

        Returns:
            mem_use: int of memory (bytes).
        """

        solidarray = self.nx * self.ny * self.nz * np.dtype(np.uint32).itemsize

        # 12 x rigidE array components + 6 x rigidH array components
        rigidarrays = (12 + 6) * self.nx * self.ny * self.nz * np.dtype(np.int8).itemsize

        # 6 x field arrays + 6 x ID arrays
        fieldarrays = ((6 + 6) * (self.nx + 1) * (self.ny + 1) * (self.nz + 1) * 
                       np.dtype(config.sim_config.dtypes['float_or_double']).itemsize)

        # PML arrays
        pmlarrays = 0
        for (k, v) in self.pmls['thickness'].items():
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

        mem_use = int(fieldarrays + solidarray + rigidarrays + pmlarrays)

        return mem_use

    def mem_est_dispersive(self):
        """Estimate the amount of memory (RAM) required for dispersive grid arrays.

        Returns:
            mem_use: int of memory (bytes).
        """

        mem_use = int(3 * config.get_model_config().materials['maxpoles'] *
                       (self.nx + 1) * (self.ny + 1) * (self.nz + 1) *
                       np.dtype(config.get_model_config().materials['dispersivedtype']).itemsize)
        return mem_use

    def tmx(self):
        """Add PEC boundaries to invariant direction in 2D TMx mode.
            N.B. 2D modes are a single cell slice of 3D grid.
        """
        # Ey & Ez components
        self.ID[1, 0, :, :] = 0
        self.ID[1, 1, :, :] = 0
        self.ID[2, 0, :, :] = 0
        self.ID[2, 1, :, :] = 0

    def tmy(self):
        """Add PEC boundaries to invariant direction in 2D TMy mode.
            N.B. 2D modes are a single cell slice of 3D grid.
        """
        # Ex & Ez components
        self.ID[0, :, 0, :] = 0
        self.ID[0, :, 1, :] = 0
        self.ID[2, :, 0, :] = 0
        self.ID[2, :, 1, :] = 0

    def tmz(self):
        """Add PEC boundaries to invariant direction in 2D TMz mode.
            N.B. 2D modes are a single cell slice of 3D grid.
        """
        # Ex & Ey components
        self.ID[0, :, :, 0] = 0
        self.ID[0, :, :, 1] = 0
        self.ID[1, :, :, 0] = 0
        self.ID[1, :, :, 1] = 0

    def calculate_dt(self):
        """Calculate time step at the CFL limit."""
        if config.get_model_config().mode == '2D TMx':
            self.dt = 1 / (config.sim_config.em_consts['c'] *
                           np.sqrt((1 / self.dy**2) + (1 / self.dz**2)))
        elif config.get_model_config().mode == '2D TMy':
            self.dt = 1 / (config.sim_config.em_consts['c'] *
                           np.sqrt((1 / self.dx**2) + (1 / self.dz**2)))
        elif config.get_model_config().mode == '2D TMz':
            self.dt = 1 / (config.sim_config.em_consts['c'] *
                           np.sqrt((1 / self.dx**2) + (1 / self.dy**2)))
        else:
            self.dt = 1 / (config.sim_config.em_consts['c'] *
                           np.sqrt((1 / self.dx**2) + (1 / self.dy**2) + (1 / self.dz**2)))

        # Round down time step to nearest float with precision one less than
        # hardware maximum. Avoids inadvertently exceeding the CFL due to
        # binary representation of floating point number.
        self.dt = round_value(self.dt, decimalplaces=d.getcontext().prec - 1)
    
    def get_srcs(self):
        return self.hertziandipoles + self.magneticdipoles + self.voltagesources + self.transmissionlines


class CUDAGrid(FDTDGrid):
    """Additional grid methods for solving on GPU using CUDA."""

    def __init__(self):
        super().__init__()

        # Threads per block - used for main electric/magnetic field updates
        self.tpb = (128, 1, 1)
        # Blocks per grid - used for main electric/magnetic field updates
        self.bpg = None

    def set_blocks_per_grid(self):
        """Set the blocks per grid size used for updating the electric and
            magnetic field arrays on a GPU.
        """

        self.bpg = (int(np.ceil(((self.nx + 1) * (self.ny + 1) *
                   (self.nz + 1)) / self.tpb[0])), 1, 1)

    def htod_geometry_arrays(self, queue=None):
        """Initialise an array for cell edge IDs (ID) on compute device.
        
        Args:
            queue: pyopencl queue.
        """

        if config.sim_config.general['solver'] == 'cuda':
            import pycuda.gpuarray as gpuarray
            self.ID_dev = gpuarray.to_gpu(self.ID)

        elif config.sim_config.general['solver'] == 'opencl':
            import pyopencl.array as clarray
            self.ID_dev = clarray.to_device(queue, self.ID)        

    def htod_field_arrays(self, queue=None):
        """Initialise field arrays on compute device.
        
        Args:
            queue: pyopencl queue.
        """

        if config.sim_config.general['solver'] == 'cuda':
            import pycuda.gpuarray as gpuarray
            self.Ex_dev = gpuarray.to_gpu(self.Ex)
            self.Ey_dev = gpuarray.to_gpu(self.Ey)
            self.Ez_dev = gpuarray.to_gpu(self.Ez)
            self.Hx_dev = gpuarray.to_gpu(self.Hx)
            self.Hy_dev = gpuarray.to_gpu(self.Hy)
            self.Hz_dev = gpuarray.to_gpu(self.Hz)
        elif config.sim_config.general['solver'] == 'opencl':
            import pyopencl.array as clarray
            self.Ex_dev = clarray.to_device(queue, self.Ex)
            self.Ey_dev = clarray.to_device(queue, self.Ey)
            self.Ez_dev = clarray.to_device(queue, self.Ez)
            self.Hx_dev = clarray.to_device(queue, self.Hx)
            self.Hy_dev = clarray.to_device(queue, self.Hy)
            self.Hz_dev = clarray.to_device(queue, self.Hz)

    def htod_dispersive_arrays(self, queue=None):
        """Initialise dispersive material coefficient arrays on compute device.
        
        Args:
            queue: pyopencl queue.
        """

        if config.sim_config.general['solver'] == 'cuda':
            import pycuda.gpuarray as gpuarray
            self.Tx_dev = gpuarray.to_gpu(self.Tx)
            self.Ty_dev = gpuarray.to_gpu(self.Ty)
            self.Tz_dev = gpuarray.to_gpu(self.Tz)
            self.updatecoeffsdispersive_dev = gpuarray.to_gpu(self.updatecoeffsdispersive)
        elif config.sim_config.general['solver'] == 'opencl':
            import pyopencl.array as clarray
            self.Tx_dev = clarray.to_device(queue, self.Tx)
            self.Ty_dev = clarray.to_device(queue, self.Ty)
            self.Tz_dev = clarray.to_device(queue, self.Tz)
            self.updatecoeffsdispersive_dev = clarray.to_device(queue, self.updatecoeffsdispersive)


class OpenCLGrid(CUDAGrid):
    """Additional grid methods for solving on compute device using OpenCL."""

    def __init__(self):
        super().__init__()

    def set_blocks_per_grid(self):
        pass


def dispersion_analysis(G):
    """Analysis of numerical dispersion (Taflove et al, 2005, p112) -
        worse case of maximum frequency and minimum wavelength

    Args:
        G: FDTDGrid class describing a grid in a model.

    Returns:
        results: dict of results from dispersion analysis.
    """

    # deltavp: physical phase velocity error (percentage)
    # N: grid sampling density
    # material: material with maximum permittivity
    # maxfreq: maximum significant frequency
    # error: error message
    results = {'deltavp': None, 
               'N': None, 
               'material': None, 
               'maxfreq': [], 
               'error': ''}

    # Find maximum significant frequency
    if G.waveforms:
        for waveform in G.waveforms:
            if waveform.type == 'sine' or waveform.type == 'contsine':
                results['maxfreq'].append(4 * waveform.freq)

            elif waveform.type == 'impulse':
                results['error'] = 'impulse waveform used.'

            elif waveform.type == 'user':
                results['error'] = 'user waveform detected.'

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
                        freqthres = np.where(power[freqmaxpower:] < -config.get_model_config().numdispersion['highestfreqthres'])[0][0] + freqmaxpower
                        results['maxfreq'].append(freqs[freqthres])
                    except ValueError:
                        results['error'] = ('unable to calculate maximum power ' +
                                           'from waveform, most likely due to ' +
                                           'undersampling.')

                # Ignore case where someone is using a waveform with zero amplitude, i.e. on a receiver
                elif waveform.amp == 0:
                    pass

                # If waveform is truncated don't do any further analysis
                else:
                    results['error'] = ('waveform does not fit within specified ' +
                                       'time window and is therefore being truncated.')
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
                # If there are dispersive materials calculate the complex 
                # relative permittivity at maximum frequency and take the real part
                if x.__class__.__name__ == 'DispersiveMaterial':
                    er = x.calculate_er(results['maxfreq'])
                    er = er.real
                if er > maxer:
                    maxer = er
                    matmaxer = x.ID
        results['material'] = next(x for x in G.materials if x.ID == matmaxer)

        # Minimum velocity
        minvelocity = config.c / np.sqrt(maxer)

        # Minimum wavelength
        minwavelength = minvelocity / results['maxfreq']

        # Maximum spatial step
        if '3D' in config.get_model_config().mode:
            delta = max(G.dx, G.dy, G.dz)
        elif '2D' in config.get_model_config().mode:
            if G.nx == 1:
                delta = max(G.dy, G.dz)
            elif G.ny == 1:
                delta = max(G.dx, G.dz)
            elif G.nz == 1:
                delta = max(G.dx, G.dy)

        # Courant stability factor
        S = (config.c * G.dt) / delta

        # Grid sampling density
        results['N'] = minwavelength / delta

        # Check grid sampling will result in physical wave propagation
        if int(np.floor(results['N'])) >= config.get_model_config().numdispersion['mingridsampling']:
            # Numerical phase velocity
            vp = np.pi / (results['N'] * np.arcsin((1 / S) * np.sin((np.pi * S) / results['N'])))

            # Physical phase velocity error (percentage)
            results['deltavp'] = (((vp * config.c) - config.c) / config.c) * 100

        # Store rounded down value of grid sampling density
        results['N'] = int(np.floor(results['N']))

    return results


def Ix(x, y, z, Hx, Hy, Hz, G):
    """Calculates the x-component of current at a grid position.

    Args:
        x, y, z: floats for coordinates of position in grid.
        Hx, Hy, Hz: numpy array of magnetic field values.
        G: FDTDGrid class describing a grid in a model.
    """

    if y == 0 or z == 0:
        Ix = 0
    else:
        Ix = G.dy * (Hy[x, y, z - 1] - Hy[x, y, z]) + G.dz * (Hz[x, y, z] - Hz[x, y - 1, z])

    return Ix


def Iy(x, y, z, Hx, Hy, Hz, G):
    """Calculates the y-component of current at a grid position.

    Args:
        x, y, z: floats for coordinates of position in grid.
        Hx, Hy, Hz: numpy array of magnetic field values.
        G: FDTDGrid class describing a grid in a model.
    """

    if x == 0 or z == 0:
        Iy = 0
    else:
        Iy = G.dx * (Hx[x, y, z] - Hx[x, y, z - 1]) + G.dz * (Hz[x - 1, y, z] - Hz[x, y, z])

    return Iy


def Iz(x, y, z, Hx, Hy, Hz, G):
    """Calculates the z-component of current at a grid position.

    Args:
        x, y, z: floats for coordinates of position in grid.
        Hx, Hy, Hz: numpy array of magnetic field values.
        G: FDTDGrid class describing a grid in a model.
    """

    if x == 0 or y == 0:
        Iz = 0
    else:
        Iz = G.dx * (Hx[x, y - 1, z] - Hx[x, y, z]) + G.dy * (Hy[x, y, z] - Hy[x - 1, y, z])

    return Iz
