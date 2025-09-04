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

from copy import deepcopy

import numpy as np

from gprMax.constants import c
from gprMax.constants import floattype
from gprMax.grid import Ix
from gprMax.grid import Iy
from gprMax.grid import Iz
from gprMax.utilities import round_value


class Source(object):
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
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        # Waveform values on timesteps
        self.waveformvalues_wholestep = np.zeros((G.iterations), dtype=floattype)

        # Waveform values on half timesteps
        self.waveformvalues_halfstep = np.zeros((G.iterations), dtype=floattype)

        waveform = next(x for x in G.waveforms if x.ID == self.waveformID)

        for iteration in range(G.iterations):
            time = G.dt * iteration
            if time >= self.start and time <= self.stop:
                # Set the time of the waveform evaluation to account for any delay in the start
                time -= self.start
                self.waveformvalues_wholestep[iteration] = waveform.calculate_value(time, G.dt)
                self.waveformvalues_halfstep[iteration] = waveform.calculate_value(time + 0.5 * G.dt, G.dt)
                

class VoltageSource(Source):
    """A voltage source can be a hard source if it's resistance is zero, 
        i.e. the time variation of the specified electric field component is 
        prescribed. If it's resistance is non-zero it behaves as a resistive 
        voltage source."""

    def __init__(self):
        super().__init__()
        self.resistance = None

    def update_electric(self, iteration, updatecoeffsE, ID, Ex, Ey, Ez, G):
        """Updates electric field values for a voltage source.

        Args:
            iteration (int): Current iteration (timestep).
            updatecoeffsE (memory view): numpy array of electric field update coefficients.
            ID (memory view): numpy array of numeric IDs corresponding to materials in the model.
            Ex, Ey, Ez (memory view): numpy array of electric field values.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord
            componentID = 'E' + self.polarisation

            if self.polarisation == 'x':
                if self.resistance != 0:
                    Ex[i, j, k] -= (updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4] 
                                    * self.waveformvalues_wholestep[iteration] 
                                    * (1 / (self.resistance * G.dy * G.dz)))
                else:
                    Ex[i, j, k] = - self.waveformvalues_halfstep[iteration] / G.dx

            elif self.polarisation == 'y':
                if self.resistance != 0:
                    Ey[i, j, k] -= (updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4] 
                                    * self.waveformvalues_wholestep[iteration] 
                                    * (1 / (self.resistance * G.dx * G.dz)))
                else:
                    Ey[i, j, k] = - self.waveformvalues_halfstep[iteration] / G.dy

            elif self.polarisation == 'z':
                if self.resistance != 0:
                    Ez[i, j, k] -= (updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4] 
                                    * self.waveformvalues_wholestep[iteration] 
                                    * (1 / (self.resistance * G.dx * G.dy)))
                else:
                    Ez[i, j, k] = - self.waveformvalues_halfstep[iteration] / G.dz

    def create_material(self, G):
        """Create a new material at the voltage source location that adds the
            voltage source conductivity to the underlying parameters.

        Args:
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        if self.resistance != 0:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord

            componentID = 'E' + self.polarisation
            requirednumID = G.ID[G.IDlookup[componentID], i, j, k]
            material = next(x for x in G.materials if x.numID == requirednumID)
            newmaterial = deepcopy(material)
            newmaterial.ID = material.ID + '+' + self.ID
            newmaterial.numID = len(G.materials)
            newmaterial.averagable = False
            newmaterial.type += ',\nvoltage-source'

            # Add conductivity of voltage source to underlying conductivity
            if self.polarisation == 'x':
                newmaterial.se += G.dx / (self.resistance * G.dy * G.dz)
            elif self.polarisation == 'y':
                newmaterial.se += G.dy / (self.resistance * G.dx * G.dz)
            elif self.polarisation == 'z':
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
            iteration (int): Current iteration (timestep).
            updatecoeffsE (memory view): numpy array of electric field update coefficients.
            ID (memory view): numpy array of numeric IDs corresponding to materials in the model.
            Ex, Ey, Ez (memory view): numpy array of electric field values.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord
            componentID = 'E' + self.polarisation

            if self.polarisation == 'x':
                Ex[i, j, k] -= (updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4] 
                                * self.waveformvalues_wholestep[iteration] 
                                * self.dl * (1 / (G.dx * G.dy * G.dz)))

            elif self.polarisation == 'y':
                Ey[i, j, k] -= (updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4] 
                                * self.waveformvalues_wholestep[iteration] 
                                * self.dl * (1 / (G.dx * G.dy * G.dz)))

            elif self.polarisation == 'z':
                Ez[i, j, k] -= (updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4] 
                                * self.waveformvalues_wholestep[iteration] 
                                * self.dl * (1 / (G.dx * G.dy * G.dz)))


class MagneticDipole(Source):
    """A magnetic dipole is an additive source (magnetic current density)."""

    def __init__(self):
        super().__init__()

    def update_magnetic(self, iteration, updatecoeffsH, ID, Hx, Hy, Hz, G):
        """Updates magnetic field values for a magnetic dipole.

        Args:
            iteration (int): Current iteration (timestep).
            updatecoeffsH (memory view): numpy array of magnetic field update coefficients.
            ID (memory view): numpy array of numeric IDs corresponding to materials in the model.
            Hx, Hy, Hz (memory view): numpy array of magnetic field values.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord
            componentID = 'H' + self.polarisation

            if self.polarisation == 'x':
                Hx[i, j, k] -= (updatecoeffsH[ID[G.IDlookup[componentID], i, j, k], 4] 
                                * self.waveformvalues_halfstep[iteration] 
                                * (1 / (G.dx * G.dy * G.dz)))

            elif self.polarisation == 'y':
                Hy[i, j, k] -= (updatecoeffsH[ID[G.IDlookup[componentID], i, j, k], 4] 
                                * self.waveformvalues_halfstep[iteration] 
                                * (1 / (G.dx * G.dy * G.dz)))

            elif self.polarisation == 'z':
                Hz[i, j, k] -= (updatecoeffsH[ID[G.IDlookup[componentID], i, j, k], 4] 
                                * self.waveformvalues_halfstep[iteration] 
                                * (1 / (G.dx * G.dy * G.dz)))


def gpu_initialise_src_arrays(sources, G):
    """Initialise arrays on GPU for source coordinates/polarisation, 
        other source information, and source waveform values.

    Args:
        sources (list): List of sources of one class, e.g. HertzianDipoles.
        G (class): Grid class instance - holds essential parameters describing the model.

    Returns:
        srcinfo1_gpu (int): numpy array of source cell coordinates and polarisation information.
        srcinfo2_gpu (float): numpy array of other source information, e.g. length, resistance etc...
        srcwaves_gpu (float): numpy array of source waveform values.
    """

    import pycuda.gpuarray as gpuarray

    srcinfo1 = np.zeros((len(sources), 4), dtype=np.int32)
    srcinfo2 = np.zeros((len(sources)), dtype=floattype)
    srcwaves = np.zeros((len(sources), G.iterations), dtype=floattype)
    for i, src in enumerate(sources):
        srcinfo1[i, 0] = src.xcoord
        srcinfo1[i, 1] = src.ycoord
        srcinfo1[i, 2] = src.zcoord

        if src.polarisation == 'x':
            srcinfo1[i, 3] = 0
        elif src.polarisation == 'y':
            srcinfo1[i, 3] = 1
        elif src.polarisation == 'z':
            srcinfo1[i, 3] = 2

        if src.__class__.__name__ == 'HertzianDipole':
            srcinfo2[i] = src.dl
            srcwaves[i, :] = src.waveformvalues_wholestep
        elif src.__class__.__name__ == 'VoltageSource':
            if src.resistance:
                srcinfo2[i] = src.resistance
                srcwaves[i, :] = src.waveformvalues_wholestep
            else:
                srcinfo2[i] = 0
                srcwaves[i, :] = src.waveformvalues_halfstep
        elif src.__class__.__name__ == 'MagneticDipole':
            srcwaves[i, :] = src.waveformvalues_halfstep

    srcinfo1_gpu = gpuarray.to_gpu(srcinfo1)
    srcinfo2_gpu = gpuarray.to_gpu(srcinfo2)
    srcwaves_gpu = gpuarray.to_gpu(srcwaves)

    return srcinfo1_gpu, srcinfo2_gpu, srcwaves_gpu


class TransmissionLine(Source):
    """A transmission line source is a one-dimensional transmission
        line which is attached virtually to a grid cell. An example of this
        type of model can be found in: https://doi.org/10.1109/8.277228
    """

    def __init__(self, G):
        """
        Args:
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        super().__init__()
        self.resistance = None

        # Coefficients for ABC termination of end of the transmission line
        self.abcv0 = 0
        self.abcv1 = 0

        # Spatial step of transmission line (N.B if the magic time step is
        # used it results in instabilities for certain impedances)
        self.dl = np.sqrt(3) * c * G.dt

        # Number of cells in the transmission line (initially a long line to
        # calculate incident voltage and current); consider putting ABCs/PML at end
        self.nl = round_value(0.667 * G.iterations)

        # Cell position of the one-way injector excitation in the transmission line
        self.srcpos = 5

        # Cell position of where line connects to antenna/main grid
        self.antpos = 10

        # Voltage values along the line
        self.voltage = np.zeros(self.nl, dtype=floattype)

        # Current values along the line
        self.current = np.zeros(self.nl, dtype=floattype)

        # Total (incident and scattered) voltage and current
        self.Vtotal = np.zeros(G.iterations, dtype=floattype)
        self.Itotal = np.zeros(G.iterations, dtype=floattype)

    def calculate_incident_V_I(self, G):
        """
        Calculates the incident voltage and current with a long length
            transmission line, initially not connected to the main grid from: 
            http://dx.doi.org/10.1002/mop.10415. Incident voltage and current,
            are only used to calculate s-parameters after the simulation has
            run.

        Args:
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        self.Vinc = np.zeros(G.iterations, dtype=floattype)
        self.Iinc = np.zeros(G.iterations, dtype=floattype)

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
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        h = (c * G.dt - self.dl) / (c * G.dt + self.dl)
        self.voltage[0] = h * (self.voltage[1] - self.abcv0) + self.abcv1
        self.abcv0 = self.voltage[0]
        self.abcv1 = self.voltage[1]

    def update_voltage(self, iteration, G):
        """Updates voltage values along the transmission line.

        Args:
            iteration (int): Current iteration (timestep).
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        # Update all the voltage values along the line
        self.voltage[1:self.nl] -= (self.resistance * (c * G.dt / self.dl) 
                                    * (self.current[1:self.nl] - self.current[0:self.nl - 1]))

        # Update the voltage at the position of the one-way injector excitation
        self.voltage[self.srcpos] += ((c * G.dt / self.dl) 
                                       * self.waveformvalues_wholestep[iteration])

        # Update ABC before updating current
        self.update_abc(G)

    def update_current(self, iteration, G):
        """Updates current values along the transmission line.

        Args:
            iteration (int): Current iteration (timestep).
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        # Update all the current values along the line
        self.current[0:self.nl - 1] -= ((1 / self.resistance) * (c * G.dt / self.dl) 
                                        * (self.voltage[1:self.nl] - self.voltage[0:self.nl - 1]))

        # Update the current one cell before the position of the one-way injector excitation
        self.current[self.srcpos - 1] += ((1 / self.resistance) * (c * G.dt / self.dl) 
                                          * self.waveformvalues_halfstep[iteration])

    def update_electric(self, iteration, updatecoeffsE, ID, Ex, Ey, Ez, G):
        """Updates electric field value in the main grid from voltage value in the transmission line.

        Args:
            iteration (int): Current iteration (timestep).
            updatecoeffsE (memory view): numpy array of electric field update coefficients.
            ID (memory view): numpy array of numeric IDs corresponding to materials in the model.
            Ex, Ey, Ez (memory view): numpy array of electric field values.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord

            self.update_voltage(iteration, G)

            if self.polarisation == 'x':
                Ex[i, j, k] = - self.voltage[self.antpos] / G.dx

            elif self.polarisation == 'y':
                Ey[i, j, k] = - self.voltage[self.antpos] / G.dy

            elif self.polarisation == 'z':
                Ez[i, j, k] = - self.voltage[self.antpos] / G.dz

    def update_magnetic(self, iteration, updatecoeffsH, ID, Hx, Hy, Hz, G):
        """Updates current value in transmission line from magnetic field values in the main grid.

        Args:
            iteration (int): Current iteration (timestep).
            updatecoeffsH (memory view): numpy array of magnetic field update coefficients.
            ID (memory view): numpy array of numeric IDs corresponding to materials in the model.
            Hx, Hy, Hz (memory view): numpy array of magnetic field values.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord

            self.update_current(iteration, G)

            if self.polarisation == 'x':
                self.current[self.antpos] = Ix(i, j, k, G.Hx, G.Hy, G.Hz, G)

            elif self.polarisation == 'y':
                self.current[self.antpos] = Iy(i, j, k, G.Hx, G.Hy, G.Hz, G)

            elif self.polarisation == 'z':
                self.current[self.antpos] = Iz(i, j, k, G.Hx, G.Hy, G.Hz, G)        