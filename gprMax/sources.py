# Copyright (C) 2015-2023: The University of Edinburgh, United Kingdom
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

from copy import deepcopy

import numpy as np

import gprMax.config as config

from .fields_outputs import Ix, Iy, Iz
from .utilities.utilities import round_value

import scipy.constants as constants
from .cython.planeWaveModules import getIntegerForAngles, getProjections, getGridFields

class Source:
    """Super-class which describes a generic source."""

    def __init__(self):
        self.ID = None
        self.polarisation = None
        self.xcoord = None
        self.ycoord = None
        self.zcoord = None
        self.xcoordorigin = None
        self.ycoordorigin = None
        self.zcoordorigin = None
        self.start = None
        self.stop = None
        self.waveformID = None

    def calculate_waveform_values(self, G):
        """Calculates all waveform values for source for duration of simulation.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """
        # Waveform values for sources that need to be calculated on whole timesteps
        self.waveformvalues_wholedt = np.zeros((G.iterations), dtype=config.sim_config.dtypes["float_or_double"])

        # Waveform values for sources that need to be calculated on half timesteps
        self.waveformvalues_halfdt = np.zeros((G.iterations), dtype=config.sim_config.dtypes["float_or_double"])

        waveform = next(x for x in G.waveforms if x.ID == self.waveformID)

        for iteration in range(G.iterations):
            time = G.dt * iteration
            if time >= self.start and time <= self.stop:
                # Set the time of the waveform evaluation to account for any
                # delay in the start
                time -= self.start
                self.waveformvalues_wholedt[iteration] = waveform.calculate_value(time, G.dt)
                self.waveformvalues_halfdt[iteration] = waveform.calculate_value(time + 0.5 * G.dt, G.dt)


class VoltageSource(Source):
    """A voltage source can be a hard source if it's resistance is zero,
    i.e. the time variation of the specified electric field component
    is prescribed. If it's resistance is non-zero it behaves as a resistive
    voltage source.
    """

    def __init__(self):
        super().__init__()
        self.resistance = None

    def update_electric(self, iteration, updatecoeffsE, ID, Ex, Ey, Ez, G):
        """Updates electric field values for a voltage source.

        Args:
            iteration: int of current iteration (timestep).
            updatecoeffsE: memory view of array of electric field update
                            coefficients.
            ID: memory view of array of numeric IDs corresponding to materials
                in the model.
            Ex, Ey, Ez: memory view of array of electric field values.
            G: FDTDGrid class describing a grid in a model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord
            componentID = f"E{self.polarisation}"

            if self.polarisation == "x":
                if self.resistance != 0:
                    Ex[i, j, k] -= (
                        updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                        * self.waveformvalues_halfdt[iteration]
                        * (1 / (self.resistance * G.dy * G.dz))
                    )
                else:
                    Ex[i, j, k] = -1 * self.waveformvalues_wholedt[iteration] / G.dx

            elif self.polarisation == "y":
                if self.resistance != 0:
                    Ey[i, j, k] -= (
                        updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                        * self.waveformvalues_halfdt[iteration]
                        * (1 / (self.resistance * G.dx * G.dz))
                    )
                else:
                    Ey[i, j, k] = -1 * self.waveformvalues_wholedt[iteration] / G.dy

            elif self.polarisation == "z":
                if self.resistance != 0:
                    Ez[i, j, k] -= (
                        updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                        * self.waveformvalues_halfdt[iteration]
                        * (1 / (self.resistance * G.dx * G.dy))
                    )
                else:
                    Ez[i, j, k] = -1 * self.waveformvalues_wholedt[iteration] / G.dz

    def create_material(self, G):
        """Create a new material at the voltage source location that adds the
            voltage source conductivity to the underlying parameters.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        if self.resistance == 0:
            return
        i = self.xcoord
        j = self.ycoord
        k = self.zcoord

        componentID = f"E{self.polarisation}"
        requirednumID = G.ID[G.IDlookup[componentID], i, j, k]
        material = next(x for x in G.materials if x.numID == requirednumID)
        newmaterial = deepcopy(material)
        newmaterial.ID = f"{material.ID}+{self.ID}"
        newmaterial.numID = len(G.materials)
        newmaterial.averagable = False
        newmaterial.type += ",\nvoltage-source" if newmaterial.type else "voltage-source"

        # Add conductivity of voltage source to underlying conductivity
        if self.polarisation == "x":
            newmaterial.se += G.dx / (self.resistance * G.dy * G.dz)
        elif self.polarisation == "y":
            newmaterial.se += G.dy / (self.resistance * G.dx * G.dz)
        elif self.polarisation == "z":
            newmaterial.se += G.dz / (self.resistance * G.dx * G.dy)

        G.ID[G.IDlookup[componentID], i, j, k] = newmaterial.numID
        G.materials.append(newmaterial)


class HertzianDipole(Source):
    """A Hertzian dipole is an additive source (electric current density)."""

    def __init__(self):
        super().__init__()
        self.dl = None

    def update_electric(self, iteration, updatecoeffsE, ID, Ex, Ey, Ez, G):
        """Updates electric field values for a Hertzian dipole.

        Args:
            iteration: int of current iteration (timestep).
            updatecoeffsE: memory view of array of electric field update
                            coefficients.
            ID: memory view of array of numeric IDs corresponding to materials
                in the model.
            Ex, Ey, Ez: memory view of array of electric field values.
            G: FDTDGrid class describing a grid in a model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord
            componentID = f"E{self.polarisation}"
            if self.polarisation == "x":
                Ex[i, j, k] -= (
                    updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_halfdt[iteration]
                    * self.dl
                    * (1 / (G.dx * G.dy * G.dz))
                )

            elif self.polarisation == "y":
                Ey[i, j, k] -= (
                    updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_halfdt[iteration]
                    * self.dl
                    * (1 / (G.dx * G.dy * G.dz))
                )

            elif self.polarisation == "z":
                Ez[i, j, k] -= (
                    updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_halfdt[iteration]
                    * self.dl
                    * (1 / (G.dx * G.dy * G.dz))
                )


class MagneticDipole(Source):
    """A magnetic dipole is an additive source (magnetic current density)."""

    def update_magnetic(self, iteration, updatecoeffsH, ID, Hx, Hy, Hz, G):
        """Updates magnetic field values for a magnetic dipole.

        Args:
            iteration: int of current iteration (timestep).
            updatecoeffsH: memory view of array of magnetic field update
                            coefficients.
            ID: memory view of array of numeric IDs corresponding to materials
                in the model.
            Hx, Hy, Hz: memory view of array of magnetic field values.
            G: FDTDGrid class describing a grid in a model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord
            componentID = f"H{self.polarisation}"

            if self.polarisation == "x":
                Hx[i, j, k] -= (
                    updatecoeffsH[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_wholedt[iteration]
                    * (1 / (G.dx * G.dy * G.dz))
                )

            elif self.polarisation == "y":
                Hy[i, j, k] -= (
                    updatecoeffsH[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_wholedt[iteration]
                    * (1 / (G.dx * G.dy * G.dz))
                )

            elif self.polarisation == "z":
                Hz[i, j, k] -= (
                    updatecoeffsH[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_wholedt[iteration]
                    * (1 / (G.dx * G.dy * G.dz))
                )


def htod_src_arrays(sources, G, queue=None):
    """Initialise arrays on compute device for source coordinates/polarisation,
        other source information, and source waveform values.

    Args:
        sources: list of sources of one type, e.g. HertzianDipole
        G: FDTDGrid class describing a grid in a model.
        queue: pyopencl queue.

    Returns:
        srcinfo1_dev: int array of source cell coordinates and polarisation
                        information.
        srcinfo2_dev: float array of other source information, e.g. length,
                        resistance etc...
        srcwaves_dev: float array of source waveform values.
    """

    srcinfo1 = np.zeros((len(sources), 4), dtype=np.int32)
    srcinfo2 = np.zeros((len(sources)), dtype=config.sim_config.dtypes["float_or_double"])
    srcwaves = np.zeros((len(sources), G.iterations), dtype=config.sim_config.dtypes["float_or_double"])
    for i, src in enumerate(sources):
        srcinfo1[i, 0] = src.xcoord
        srcinfo1[i, 1] = src.ycoord
        srcinfo1[i, 2] = src.zcoord

        if src.polarisation == "x":
            srcinfo1[i, 3] = 0
        elif src.polarisation == "y":
            srcinfo1[i, 3] = 1
        elif src.polarisation == "z":
            srcinfo1[i, 3] = 2

        if src.__class__.__name__ == "HertzianDipole":
            srcinfo2[i] = src.dl
            srcwaves[i, :] = src.waveformvalues_halfdt
        elif src.__class__.__name__ == "VoltageSource":
            if src.resistance:
                srcinfo2[i] = src.resistance
                srcwaves[i, :] = src.waveformvalues_halfdt
            else:
                srcinfo2[i] = 0
                srcwaves[i, :] = src.waveformvalues_wholedt
            srcinfo2[i] = src.resistance
            srcwaves[i, :] = src.waveformvalues_halfdt
        elif src.__class__.__name__ == "MagneticDipole":
            srcwaves[i, :] = src.waveformvalues_wholedt

    # Copy arrays to compute device
    if config.sim_config.general["solver"] == "cuda":
        import pycuda.gpuarray as gpuarray

        srcinfo1_dev = gpuarray.to_gpu(srcinfo1)
        srcinfo2_dev = gpuarray.to_gpu(srcinfo2)
        srcwaves_dev = gpuarray.to_gpu(srcwaves)
    elif config.sim_config.general["solver"] == "opencl":
        import pyopencl.array as clarray

        srcinfo1_dev = clarray.to_device(queue, srcinfo1)
        srcinfo2_dev = clarray.to_device(queue, srcinfo2)
        srcwaves_dev = clarray.to_device(queue, srcwaves)

    return srcinfo1_dev, srcinfo2_dev, srcwaves_dev


class TransmissionLine(Source):
    """A transmission line source is a one-dimensional transmission line
    which is attached virtually to a grid cell.
    """

    def __init__(self, G):
        """
        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        super().__init__()
        self.resistance = None

        # Coefficients for ABC termination of end of the transmission line
        self.abcv0 = 0
        self.abcv1 = 0

        # Spatial step of transmission line (N.B if the magic time step is
        # used it results in instabilities for certain impedances)
        self.dl = np.sqrt(3) * config.c * G.dt

        # Number of cells in the transmission line (initially a long line to
        # calculate incident voltage and current); consider putting ABCs/PML at end
        self.nl = round_value(0.667 * G.iterations)

        # Cell position of the one-way injector excitation in the transmission line
        self.srcpos = 5

        # Cell position of where line connects to antenna/main grid
        self.antpos = 10

        self.voltage = np.zeros(self.nl, dtype=config.sim_config.dtypes["float_or_double"])
        self.current = np.zeros(self.nl, dtype=config.sim_config.dtypes["float_or_double"])
        self.Vinc = np.zeros(G.iterations, dtype=config.sim_config.dtypes["float_or_double"])
        self.Iinc = np.zeros(G.iterations, dtype=config.sim_config.dtypes["float_or_double"])
        self.Vtotal = np.zeros(G.iterations, dtype=config.sim_config.dtypes["float_or_double"])
        self.Itotal = np.zeros(G.iterations, dtype=config.sim_config.dtypes["float_or_double"])

    def calculate_incident_V_I(self, G):
        """Calculates the incident voltage and current with a long length
            transmission line not connected to the main grid
            from: http://dx.doi.org/10.1002/mop.10415

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        for iteration in range(G.iterations):
            self.Iinc[iteration] = self.current[self.antpos]
            self.Vinc[iteration] = self.voltage[self.antpos]
            self.update_current(iteration, G)
            self.update_voltage(iteration, G)

        # Shorten number of cells in the transmission line before use with main grid
        self.nl = self.antpos + 1

    def update_abc(self, G):
        """Updates absorbing boundary condition at end of the transmission line.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        h = (config.c * G.dt - self.dl) / (config.c * G.dt + self.dl)

        self.voltage[0] = h * (self.voltage[1] - self.abcv0) + self.abcv1
        self.abcv0 = self.voltage[0]
        self.abcv1 = self.voltage[1]

    def update_voltage(self, iteration, G):
        """Updates voltage values along the transmission line.

        Args:
            iteration: int of current iteration (timestep).
            G: FDTDGrid class describing a grid in a model.
        """

        # Update all the voltage values along the line
        self.voltage[1 : self.nl] -= (
            self.resistance * (config.c * G.dt / self.dl) * (self.current[1 : self.nl] - self.current[0 : self.nl - 1])
        )

        # Update the voltage at the position of the one-way injector excitation
        self.voltage[self.srcpos] += (config.c * G.dt / self.dl) * self.waveformvalues_halfdt[iteration]

        # Update ABC before updating current
        self.update_abc(G)

    def update_current(self, iteration, G):
        """Updates current values along the transmission line.

        Args:
            iteration: int of current iteration (timestep).
            G: FDTDGrid class describing a grid in a model.
        """

        # Update all the current values along the line
        self.current[0 : self.nl - 1] -= (
            (1 / self.resistance)
            * (config.c * G.dt / self.dl)
            * (self.voltage[1 : self.nl] - self.voltage[0 : self.nl - 1])
        )

        # Update the current one cell before the position of the one-way injector excitation
        self.current[self.srcpos - 1] += (
            (1 / self.resistance) * (config.c * G.dt / self.dl) * self.waveformvalues_wholedt[iteration]
        )

    def update_electric(self, iteration, updatecoeffsE, ID, Ex, Ey, Ez, G):
        """Updates electric field value in the main grid from voltage value in
            the transmission line.

        Args:
            iteration: int of current iteration (timestep).
            updatecoeffsE: memory view of array of electric field update
                            coefficients.
            ID: memory view of array of numeric IDs corresponding to materials
                in the model.
            Ex, Ey, Ez: memory view of array of electric field values.
            G: FDTDGrid class describing a grid in a model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord

            self.update_voltage(iteration, G)

            if self.polarisation == "x":
                Ex[i, j, k] = -self.voltage[self.antpos] / G.dx

            elif self.polarisation == "y":
                Ey[i, j, k] = -self.voltage[self.antpos] / G.dy

            elif self.polarisation == "z":
                Ez[i, j, k] = -self.voltage[self.antpos] / G.dz

    def update_magnetic(self, iteration, updatecoeffsH, ID, Hx, Hy, Hz, G):
        """Updates current value in transmission line from magnetic field values
            in the main grid.

        Args:
            iteration: int of current iteration (timestep).
            updatecoeffsH: memory view of array of magnetic field update
                            coefficients.
            ID: memory view of array of numeric IDs corresponding to materials
                in the model.
            Hx, Hy, Hz: memory view of array of magnetic field values.
            G: FDTDGrid class describing a grid in a model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord

            if self.polarisation == "x":
                self.current[self.antpos] = Ix(i, j, k, G.Hx, G.Hy, G.Hz, G)

            elif self.polarisation == "y":
                self.current[self.antpos] = Iy(i, j, k, G.Hx, G.Hy, G.Hz, G)

            elif self.polarisation == "z":
                self.current[self.antpos] = Iz(i, j, k, G.Hx, G.Hy, G.Hz, G)

            self.update_current(iteration, G)


class DiscretePlaneWave(Source):
    '''
    Class to implement the discrete plane wave (DPW) formulation as described in
    Tan, T.; Potter, M. (2010). 
    FDTD Discrete Planewave (FDTD-DPW) Formulation for a Perfectly Matched Source in TFSF Simulations. ,
    58(8), 0–2648. doi:10.1109/tap.2010.2050446 
    __________________________
    
    Instance variables:
    --------------------------
        m, int array           : stores the integer mappings, m_x, m_y, m_z which determine the rational angles
                                 last element stores max(m_x, m_y, m_z)
        directions, int array  : stores the directions of propagation of the DPW
        dimensions, int        : stores the number of dimensions in which the simulation is run (2D or 3D)
        time_dimension, int    : stores the time length over which the simulation is run
        E_fileds, double array : stores the electric flieds associated with the 1D DPW
        H_fields, double array : stores the magnetic fields associated with the 1D DPW
    '''
    
    
    def __init__(self, G):     
        '''
        Defines the instance variables of class DiscretePlaneWave()
        __________________________

        Input parameters:
        --------------------------
            time_dimension, int : local variable to store the time length over which the simulation is run 
            dimensions, int     : local variable to store the number of dimensions in which the simulation is run
        '''
        super().__init__()
        self.m = np.zeros(3+1, dtype=np.int32)          #+1 to store the max(m_x, m_y, m_z)
        self.directions = np.zeros(3, dtype=np.int32)
        self.time_dimension = G.iterations
        self.length = 0
        self.projections = np.zeros(3)
        self.corners = None
        self.ds = 0
        self.E_fields = None  
        self.H_fields = None
        self.E_coefficients = None
        self.H_coefficients = None
        self.dt = G.dt
        self.ppw = 10.0*G.dt
        self.real_time = 0.0
        
    def initialize1DGrid(self, G):
        '''
        Method to initialize the one dimensions grids for the DPW
        __________________________

        Input parameters:
        --------------------------
            length_dimension, int : stores the spatial length of the ggrids for the DPW
            dl, double            : stores the spatial seperation between two adjacent elements of the DPW array
            dt, double            : stores the temporal separation between two adjacent rows of the DPW array
        __________________________

        Returns:
        --------------------------
            E_fields, double array :       stores the electric field values for the DPW
                                           first index denotes the spatial dimension
                                           second index denotes the spatial position (grid cell position index)
                                           thid index denotes time (grid cell time index)
            H_fields, double array :       stores the magnetic field values for the DPW
                                           first index denotes the spatial dimension
                                           second index denotes the spatial position (grid cell position index)
                                           thid index denotes time (grid cell time index)
            E_coefficients, double array : stores the coefficients of the fields in the equation to update electric fields
            H_coefficients, double array : stores the coefficients of the fields in the equation to update magnetic fields

        '''
        self.E_fields = np.zeros((3, self.length), order='C', dtype=config.sim_config.dtypes["float_or_double"])
        self.H_fields = np.zeros((3, self.length), order='C', dtype=config.sim_config.dtypes["float_or_double"])
        
        '''
        self.E_coefficients = np.array([G.updatecoeffsE[1, 0], G.updatecoeffsE[1, 2], G.updatecoeffsE[1, 3],
                                        G.updatecoeffsE[1, 0], G.updatecoeffsE[1, 3], G.updatecoeffsE[1, 1],
                                        G.updatecoeffsE[1, 0], G.updatecoeffsE[1, 1], G.updatecoeffsE[1, 2]])     #coefficients in the update equations of the electric field
        self.H_coefficients = np.array([G.updatecoeffsH[1, 0], G.updatecoeffsH[1, 2], G.updatecoeffsH[1, 3],
                                        G.updatecoeffsH[1, 0], G.updatecoeffsH[1, 3], G.updatecoeffsH[1, 1],
                                        G.updatecoeffsH[1, 0], G.updatecoeffsH[1, 1], G.updatecoeffsH[1, 2]])     #coefficients in the update equations of the magnetic field
        '''

        self.E_coefficients = np.zeros(9, dtype=config.sim_config.dtypes["float_or_double"])
        self.H_coefficients = np.zeros(9, dtype=config.sim_config.dtypes["float_or_double"])
        impedance = config.c*config.m0   #calculate the impedance of free space 
    
        for i in range(3): #loop to calculate the coefficients for each dimension
            self.E_coefficients[3*i] = 1.0
            self.E_coefficients[3*i+1] = G.dt/(config.e0*G.dx)
            self.E_coefficients[3*i+2] = G.dt/(config.e0*G.dx)        
            
            self.H_coefficients[3*i] = 1.0
            self.H_coefficients[3*i+1] = G.dt/(config.m0*G.dx)
            self.H_coefficients[3*i+2] = G.dt/(config.m0*G.dx)
        
    
    def initDiscretePlaneWave(self, psi, phi, Delta_phi, theta, Delta_theta, G):
        '''
        Method to create a DPW, assign memory to the grids and get field values at different time and space indices
        __________________________

        Input parameters:
        --------------------------
            psi, float         : polarization angle of the incident plane wave (in radians)
            phi, float         : azimuthal angle of the incident plane wave (in radians)
            Delta_phi, float   : permissible error in the rational angle approximation to phi (in radians)
            theta, float       : polar angle of the incident plane wave (in radians)
            Delta_theta, float : permissible error in the rational angle approximation to theta (in radians)
            number, int        : number of cells in the 3D FDTD simulation
            dx, double         : separation between adjacent cells in the x direction
            dy, double         : separation between adjacent cells in the y direction
            dz, double         : separation between adjacent cells in the z direction
            dt, double         : time step for the FDTD simulation
        __________________________

        Returns:
        --------------------------
            E_fields, double array   : the electric field for the DPW as it evolves over time and space indices
            H_fields, double array   : the magnetic field for the DPW as it evolves over time and space indices
            C, double array          : stores the coefficients of the fields for the update equation of the electric fields
            D, double array          : stores the coefficients of the fields for the update equation of the magnetic fields

        '''
        self.directions, self.m = getIntegerForAngles(phi, Delta_phi, theta, Delta_theta,
                                          np.array([G.dx, G.dy, G.dz]))   #get the integers for the nearest rational angle
        #store max(m_x, m_y, m_z) in the last element of the array
        self.length = int(2*np.sum(self.m[:-1])*max(G.nx, G.ny, G.nz))                  #set an appropriate length fo the one dimensional arrays
        #the 1D grid has no ABC to terminate it, sufficiently long array prevents reflections from the back 
        # Projections for field components
        projections_h, P = getProjections(psi*180/np.pi, self.m)  #get the projection vertors for different fields
        self.projections = projections_h / np.sqrt(config.m0/config.e0) #scale the projection vector for the mangetic field
        
        if self.m[0] == 0:       #calculate dr that is needed for sourcing the 1D array
            if self.m[1] == 0:
                if self.m[2] == 0:
                    raise ValueError("not all m_i values can be zero")
                else:
                    self.ds = P[2]*G.dz/self.m[2]
            else:
                self.ds = P[1]*G.dy/self.m[1]
        else:
            self.ds = P[0]*G.dx/self.m[0]

    """
    def getSource(self, time, wavetype, dt=0.0):
    '''
    Method to get the magnitude of the source field in the direction perpendicular to the propagation of the plane wave
        __________________________

        Input parameters:
        --------------------------
            time, float      : time at which the magnitude of the source is calculated
            ppw, int         : points per wavelength for the wave source
            wavetype, string : stores the type of waveform whose magnitude should be returned
            dt, double       : the time upto which the wave should exist in a impulse delta pulse

        __________________________

        Returns:
        --------------------------
            sourceMagnitude, double : magnitude of the source for the requested indices at the current time
    '''
    # Waveforms
    if (wavetype is "gaussian"):
        return np.exp(-(np.pi*(time/self.ppw - 1.0))**2)

    elif (wavetype in ["gaussiandot", "gaussianprime"]):
        return 4 * np.pi/self.ppw * (np.pi*(time/self.ppw - 1.0)) * np.exp(-(np.pi*(time/self.ppw - 1.0))**2))

    elif (wavetype is "gaussiandotnorm"):
        return 2 * (np.pi*(time/self.ppw - 1.0)) * np.exp(-(np.pi*(time/self.ppw - 1.0))**2))

    elif (wavetype in ["gaussiandotdot", "gaussiandoubleprime"]):
        return 2 * np.pi*np.pi/(self.ppw*self.ppw) * (1.0-2.0*(np.pi*(time/self.ppw - 1.0))*(M_PI*(time/ppw - 1.0))
                ) * exp(-(M_PI*(time/<double>ppw - 1.0)) * (M_PI*(time/<double>ppw - 1.0)))
    
    elif (strcmp(wavetype, "gaussiandotdotnorm") == 0 or strcmp(wavetype, "ricker") == 0):
        return (1.0 - 2.0*(M_PI*(time/<double>ppw - 1.0)) * (M_PI*(time/<double>ppw - 1.0))
                ) * exp(-(M_PI*(time/<double>ppw - 1.0)) * (M_PI*(time/<double>ppw - 1.0)))  # define a Ricker wave source

    elif (strcmp(wavetype, "sine") == 0):
        return sin(2 * M_PI * time/<double>ppw)

    elif (strcmp(wavetype, "contsine") == 0):
        return min(0.25 * time/<double>ppw, 1) * sin(2 * M_PI * time/<double>ppw)

    elif (strcmp(wavetype, "impulse") == 0):
        if (time < dt):                         # time < dt condition required to do impulsive magnetic dipole
            return 1
        else:
            return 0
    """
    def getSource(self, time):
        '''
        Method to get the magnitude of the source field in the direction perpendicular to the propagation of the plane wave
        __________________________

        Input parameters:
        --------------------------
            time, float : time at which the magnitude of the source is calculated
            ppw, int    : points per wavelength for the wave source

        __________________________

        Returns:
        --------------------------
            sourceMagnitude, double array : magnitude of the source for the requested indices at the current time
        '''
        sourceMagnitude = 0
        if time >= 0: 
            arg = np.pi*((time) / self.ppw - 1.0)
            arg = arg * arg
            sourceMagnitude = (1.0-2.0*arg)*np.exp(-arg)  # define a Ricker wave source
        return sourceMagnitude


    def initialize_magnetic_fields_1D(self):
        for k in range(3):
            for l in range(self.m[3]):      #loop to assign the source values of magnetic field to the first few gridpoints
                self.H_fields[k, l] = self.projections[k]*self.getSource(self.real_time - (l+(
                                self.m[(k+1)%3]+self.m[(k+2)%3])*0.5)*self.ds/config.c)#, self.waveformID, self.dt)


    def update_magnetic_field_1D(self, G):
        '''
        Method to update the magnetic fields for the next time step using
        Equation 8 of DOI: 10.1109/LAWP.2009.2016851
        __________________________

        Input parameters:
        --------------------------
            n , int                      : stores the spatial length of the DPW array so that each length grid cell is updated when the method updateMagneticFields() is called
            H_coefficients, double array : stores the coefficients of the fields in the update equation for the magnetic field
            H_fields, double array       : stores the magnetic fields of the DPW till temporal index time
            E_fields, double array       : stores the electric fields of the DPW till temporal index time
            time, int                    : time index storing the current axis number which would be updated for the H_fields
        __________________________

        Returns:
        --------------------------
            H_fields, double array       : magnetic field array with the axis entry for the current time added

        '''
        self.initialize_magnetic_fields_1D()

        for i in range(3):          #loop to update each component of the magnetic field
            for j in range(self.m[-1], self.length-self.m[-1]):  #loop to update the magnetic field at each spatial index
                self.H_fields[i, j] = self.H_coefficients[3*i] * self.H_fields[i, j] + self.H_coefficients[3*i+1] * (
                    self.E_fields[(i+1)%3, j+self.m[(i+2)%3]] - self.E_fields[(i+1)%3, j]) - self.H_coefficients[3*i+2] * (
                    self.E_fields[(i+2)%3, j+self.m[(i+1)%3]] - self.E_fields[(i+2)%3, j])     #equation 8 of Tan, Potter paper
    
    def update_electric_field_1D(self, G):
        '''
        Method to update the electric fields for the next time step using
        Equation 9 of DOI: 10.1109/LAWP.2009.2016851
        __________________________

        Input parameters:
        --------------------------
            n , int                      : stores the spatial length of the DPW array so that each length grid cell is updated when the method updateMagneticFields() is called
            E_coefficients, double array : stores the coefficients of the fields in the update equation for the electric field
            H_fields, double array       : stores the magnetic fields of the DPW till temporal index time
            E_fields, double array       : stores the electric fields of the DPW till temporal index time
            time, int                    : time index storing the current axis number which would be updated for the H_fields
        __________________________

        Returns:
        --------------------------
            E_fields, double array       : electric field array with the axis entry for the current time added

        '''
        for i in range(3):  #loop to update each component of the electric field
            for j in range(self.m[-1], self.length-self.m[-1]):   #loop to update the electric field at each spatial index 
                self.E_fields[i, j] = self.E_coefficients[3*i] * self.E_fields[i, j] + self.E_coefficients[3*i+1] * (
                    self.H_fields[(i+2)%3, j] - self.H_fields[(i+2)%3, j-self.m[(i+1)%3]]) - self.E_coefficients[3*i+2] * ( 
                    self.H_fields[(i+1)%3, j] - self.H_fields[(i+1)%3, j-self.m[(i+2)%3]])  #equation 9 of Tan, Potter paper

        if(G.iteration % 10 == 0):
            np.save('./snapshots/electric_z_{}.npy'.format(G.iteration), G.Ex)

    
    def getField(self, i, j, k, array, m, component):
        buffer = 10
        return array[component, np.dot(m[:-1], np.array([i, j, k]))]


    def apply_TFSF_conditions_magnetic(self, G):

        #**** constant x faces -- scattered-field nodes ****
        i = self.corners[0]
        for j in range(self.corners[1], self.corners[4]+1):
            for k in range(self.corners[2], self.corners[5]):
                #correct Hy at firstX-1/2 by subtracting Ez_inc
                G.Hy[i-1, j, k] -= self.H_coefficients[5] * self.getField(i, j, k, self.E_fields, self.m, 2) 

        for j in range(self.corners[1], self.corners[4]):
            for k in range(self.corners[2], self.corners[5]+1):
                #correct Hz at firstX-1/2 by adding Ey_inc
                G.Hz[i-1, j, k] += self.H_coefficients[8] * self.getField(i, j, k, self.E_fields, self.m, 1)

        i = self.corners[3]
        for j in range(self.corners[1], self.corners[4]+1):
            for k in range(self.corners[2], self.corners[5]):
                #correct Hy at lastX+1/2 by adding Ez_inc
                G.Hy[i, j, k] += self.H_coefficients[5] * self.getField(i, j, k, self.E_fields, self.m, 2)    

        for j in range(self.corners[1], self.corners[4]):
            for k in range(self.corners[2], self.corners[5]+1):
                #correct Hz at lastX+1/2 by subtractinging Ey_inc
                G.Hz[i, j, k] -= self.H_coefficients[8] * self.getField(i, j, k, self.E_fields, self.m, 1)            

        #**** constant y faces -- scattered-field nodes ****
        j = self.corners[1]
        for i in range(self.corners[0], self.corners[3]+1):
            for k in range(self.corners[2], self.corners[5]):
                #correct Hx at firstY-1/2 by adding Ez_inc
                G.Hx[i, j-1, k] += self.H_coefficients[2] * self.getField(i, j, k, self.E_fields, self.m, 2)

        for i in range(self.corners[0], self.corners[3]):
            for k in range(self.corners[2], self.corners[5]+1):
                #correct Hz at firstY-1/2 by subtracting Ex_inc
                G.Hz[i, j-1, k] -= self.H_coefficients[7] * self.getField(i, j, k, self.E_fields, self.m, 0)

        j = self.corners[4]
        for i in range(self.corners[0], self.corners[3]+1):
            for k in range(self.corners[2], self.corners[5]):
                #correct Hx at lastY+1/2 by subtracting Ez_inc
                G.Hx[i, j, k] -= self.H_coefficients[2] * self.getField(i, j, k, self.E_fields, self.m, 2)

        for i in range(self.corners[0], self.corners[3]):
            for k in range(self.corners[2], self.corners[5]+1):
                #correct Hz at lastY-1/2 by adding Ex_inc
                G.Hz[i, j, k] += self.H_coefficients[7] * self.getField(i, j, k, self.E_fields, self.m, 0)

        #**** constant z faces -- scattered-field nodes ****
        k = self.corners[2]
        for i in range(self.corners[0], self.corners[3]):
            for j in range(self.corners[1], self.corners[4]+1):
                #correct Hy at firstZ-1/2 by adding Ex_inc
                G.Hy[i, j, k-1] += self.H_coefficients[5] * self.getField(i, j, k, self.E_fields, self.m, 0)

        for i in range(self.corners[0], self.corners[3]+1):
            for j in range(self.corners[1], self.corners[4]):
                #correct Hx at firstZ-1/2 by subtracting Ey_inc
                G.Hx[i, j, k-1] -= self.H_coefficients[1] * self.getField(i, j, k, self.E_fields, self.m, 1)

        k = self.corners[5]
        for i in range(self.corners[0], self.corners[3]):
            for j in range(self.corners[1], self.corners[4]+1):
                #correct Hy at firstZ-1/2 by subtracting Ex_inc
                G.Hy[i, j, k] -= self.H_coefficients[5] * self.getField(i, j, k, self.E_fields, self.m, 0)

        for i in range(self.corners[0], self.corners[3]+1):
            for j in range(self.corners[1], self.corners[4]):
                #correct Hx at lastZ+1/2 by adding Ey_inc
                G.Hx[i, j, k] += self.H_coefficients[1] * self.getField(i, j, k, self.E_fields, self.m, 1)

    


    def apply_TFSF_conditions_electric(self, G):
        #**** constant x faces -- total-field nodes ****/
        i = self.corners[0]
        for j in range(self.corners[1], self.corners[4]+1):
            for k in range(self.corners[2], self.corners[5]):
                #correct Ez at firstX face by subtracting Hy_inc
                G.Ez[i, j, k] -= self.E_coefficients[7] * self.getField(i-1, j, k, self.H_fields, self.m, 1)

        for j in range(self.corners[1], self.corners[4]):
            for k in range(self.corners[2], self.corners[5]+1):
                #correct Ey at firstX face by adding Hz_inc
                G.Ey[i, j, k] += self.E_coefficients[4] * self.getField(i-1, j, k, self.H_fields, self.m, 2)

        i = self.corners[3]
        for j in range(self.corners[1], self.corners[4]+1):
            for k in range(self.corners[2], self.corners[5]):
                #correct Ez at lastX face by adding Hy_inc
                G.Ez[i, j, k] += self.E_coefficients[7] * self.getField(i, j, k, self.H_fields, self.m, 1)

        i = self.corners[3]
        for j in range(self.corners[1], self.corners[4]):
            for k in range(self.corners[2], self.corners[5]+1):
                #correct Ey at lastX face by subtracting Hz_inc
                G.Ey[i, j, k] -= self.E_coefficients[4] * self.getField(i, j, k, self.H_fields, self.m, 2)

        #**** constant y faces -- total-field nodes ****/
        j = self.corners[1]
        for i in range(self.corners[0], self.corners[3]+1):
            for k in range(self.corners[2], self.corners[5]):
                #correct Ez at firstY face by adding Hx_inc
                G.Ez[i, j, k] += self.E_coefficients[8] * self.getField(i, j-1, k, self.H_fields, self.m, 0)

        for i in range(self.corners[0], self.corners[3]):
            for k in range(self.corners[2], self.corners[5]+1):
                #correct Ex at firstY face by subtracting Hz_inc
                G.Ex[i, j, k] -= self.E_coefficients[1] * self.getField(i, j-1, k, self.H_fields, self.m, 2)

        j = self.corners[4]
        for i in range(self.corners[0], self.corners[3]+1):
            for k in range(self.corners[2], self.corners[5]):
                #correct Ez at lastY face by subtracting Hx_inc
                G.Ez[i, j, k] -= self.E_coefficients[8] * self.getField(i, j, k, self.H_fields, self.m, 0)

        for i in range(self.corners[0], self.corners[3]):
            for k in range(self.corners[2], self.corners[5]+1):
                #correct Ex at lastY face by adding Hz_inc
                G.Ex[i, j, k] += self.E_coefficients[1] * self.getField(i, j, k, self.H_fields, self.m, 2)

        #**** constant z faces -- total-field nodes ****/
        k = self.corners[2]
        for i in range(self.corners[0], self.corners[3]+1):
            for j in range(self.corners[1], self.corners[4]):
                #correct Ey at firstZ face by subtracting Hx_inc
                G.Ey[i, j, k] -= self.E_coefficients[4] * self.getField(i, j, k-1, self.H_fields, self.m, 0)

        for i in range(self.corners[0], self.corners[3]):
            for j in range(self.corners[1], self.corners[4]+1):
                #correct Ex at firstZ face by adding Hy_inc
                G.Ex[i, j, k] += self.E_coefficients[2] * self.getField(i, j, k-1, self.H_fields, self.m, 1)

        k = self.corners[5]
        for i in range(self.corners[0], self.corners[3]+1):
            for j in range(self.corners[1], self.corners[4]):
                #correct Ey at lastZ face by adding Hx_inc
                G.Ey[i, j, k] += self.E_coefficients[4] * self.getField(i, j, k, self.H_fields, self.m, 0)

        for i in range(self.corners[0], self.corners[3]):
            for j in range(self.corners[1], self.corners[4]+1):
                #correct Ex at lastZ face by subtracting Hy_inc
                G.Ex[i, j, k] -= self.E_coefficients[2] * self.getField(i, j, k, self.H_fields, self.m, 1)
        
           


class TFSFBox():
    '''
    Class to implement a Total Field/Scattered Field(TFSF) implementation of the DPW described in
    Chapter 3: Exact Total-Field/Scattered-Field Plane-Wave Source Condition
    by Tengmeng Tan and Mike Potter
    of Steven Johnson; Ardavan Oskooi; Allen Taflove, Advances in FDTD Computational Electrodynamics: Photonics and Nanotechnology,
    Artech, 2013. (ISBN:9781608071715)
    __________________________
    
    Instance variables:
    --------------------------
        n_x, int            : stores the number of grid cells along the x axis of the TFSF box
        n_y, int            : stores the number of grid cells along the y axis of the TFSF box
        n_z, int            : stores the number of grid cells along the z axis of the TFSF box
        e_x, double array   : stores the x component of the electric field for the grid cells over the TFSF box
        e_y, double array   : stores the y component of the electric field for the grid cells over the TFSF box
        e_z, double array   : stores the z component of the electric field for the grid cells over the TFSF box
        h_x, double array   : stores the x component of the magnetic field for the grid cells over the TFSF box
        h_y, double array   : stores the y component of the magnetic field for the grid cells over the TFSF box
        h_z, double array   : stores the z component of the magnetic field for the grid cells over the TFSF box
        corners, int array  : stores the coordinates of the cornets of the total field/scattered field boundaries
        time_dimension, int : stores the time length over which the FDTD simulation is run
    '''
    def __init__(self, n_x, n_y, n_z, corners, time_duration, dimensions, noOfWaves):
        '''
        Defines the instance variables of class DiscretePlaneWave()
        __________________________
        
        Input parameters:
        --------------------------
            n_x, int            : stores the number of grid cells along the x axis of the TFSF box
            n_y, int            : stores the number of grid cells along the y axis of the TFSF box
            n_z, int            : stores the number of grid cells along the z axis of the TFSF box
            corners, int array  : stores the coordinates of the cornets of the total field/scattered field boundaries
            time_dimension, int : stores the time length over which the FDTD simulation is run
        '''
        self.n_x = n_x   #assign the instance varibales with the number of grid points along each axis
        self.n_y = n_y
        self.n_z = n_z
        #intitialise the 3D grids with n_x, n_y, n_z cells and +1 components where necessary  
        self.dimensions = dimensions
        self.fields = np.zeros((noOfWaves+1, 2*dimensions, self.n_x+1, self.n_y+1, self.n_z+1), order='C')
        self.corners = corners
        self.time_duration = time_duration
        self.noOfWaves = noOfWaves
    

    def initializeABC(self):
        # Allocate memory for ABC arrays
        face_fields = np.zeros((self.noOfWaves, 4*self.dimensions, max(self.n_x, self.n_y, self.n_z),
                                max(self.n_x, self.n_y, self.n_z)), order='C')

        abccoef = (1/np.sqrt(3.0)-1.0)/(1/np.sqrt(3.0)+1.0)

        return face_fields, abccoef
        
    def getFields(self, planeWaves, snapshot, angles, number, dx, dy, dz, dt, ppw):
        face_fields, abccoef = self.initializeABC()
        for i in range(self.noOfWaves):
            print(f"Plane Wave {i+1} :")
            s
            C, D = planeWaves[i].initializeGrid(np.array([dx, dy, dz]), dt)  #initialize the one dimensional arrays and coefficients
        getGridFields(planeWaves, C, D, snapshot, self.n_x, self.n_y, self.n_z, self.fields, self.corners,
                     self.time_duration, face_fields, abccoef, dt, self.noOfWaves, constants.c, ppw,
                     self.dimensions, './snapshots/electric', [0], []) 

