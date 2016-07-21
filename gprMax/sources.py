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

from copy import deepcopy

import numpy as np

from gprMax.constants import c, floattype
from gprMax.grid import Ix, Iy, Iz
from gprMax.utilities import round_value


class Source(object):
    """Super-class which describes a generic source."""

    def __init__(self):
        self.ID = None
        self.polarisation = None
        self.xcoord = None
        self.ycoord = None
        self.zcoord = None
        self.xcoordbase = None
        self.ycoordbase = None
        self.zcoordbase = None
        self.start = None
        self.stop = None
        self.waveformID = None


class VoltageSource(Source):
    """The voltage source can be a hard source if it's resistance is zero, i.e. the time variation of the specified electric field component is prescribed. If it's resistance is non-zero it behaves as a resistive voltage source."""

    def __init__(self):
        super(Source, self).__init__()
        self.resistance = None

    def update_electric(self, abstime, updatecoeffsE, ID, Ex, Ey, Ez, G):
        """Updates electric field values for a voltage source.
            
        Args:
            abstime (float): Absolute time.
            updatecoeffsE (memory view): numpy array of electric field update coefficients.
            ID (memory view): numpy array of numeric IDs corresponding to materials in the model.
            Ex, Ey, Ez (memory view): numpy array of electric field values.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        if abstime >= self.start and abstime <= self.stop:
            # Set the time of the waveform evaluation to account for any delay in the start
            time = abstime - self.start
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord
            waveform = next(x for x in G.waveforms if x.ID == self.waveformID)

            if self.polarisation is 'x':
                if self.resistance != 0:
                    componentID = 'E' + self.polarisation
                    Ex[i, j, k] -= updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4] * waveform.amp * waveform.calculate_value(time, G.dt) * (1 / (self.resistance * G.dy * G.dz))
                else:
                    Ex[i, j, k] = -1 * waveform.amp * waveform.calculate_value(time, G.dt) / G.dx

            elif self.polarisation is 'y':
                if self.resistance != 0:
                    componentID = 'E' + self.polarisation
                    Ey[i, j, k] -= updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4] * waveform.amp * waveform.calculate_value(time, G.dt) * (1 / (self.resistance * G.dx * G.dz))
                else:
                    Ey[i, j, k] = -1 * waveform.amp * waveform.calculate_value(time, G.dt) / G.dy

            elif self.polarisation is 'z':
                if self.resistance != 0:
                    componentID = 'E' + self.polarisation
                    Ez[i, j, k] -= updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4] * waveform.amp * waveform.calculate_value(time, G.dt) * (1 / (self.resistance * G.dx * G.dy))
                else:
                    Ez[i, j, k] = -1 * waveform.amp * waveform.calculate_value(time, G.dt) / G.dz

    def create_material(self, G):
        """Create a new material at the voltage source location that adds the voltage source conductivity to the underlying parameters.
            
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
            newmaterial.ID = material.ID + '+VoltageSource_' + str(self.resistance)
            newmaterial.numID = len(G.materials)
            newmaterial.average = False

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
    """The Hertzian dipole is an additive source (electric current density)."""

    def __init__(self):
        super(Source, self).__init__()

    def update_electric(self, abstime, updatecoeffsE, ID, Ex, Ey, Ez, G):
        """Updates electric field values for a Hertzian dipole.
            
        Args:
            abstime (float): Absolute time.
            updatecoeffsE (memory view): numpy array of electric field update coefficients.
            ID (memory view): numpy array of numeric IDs corresponding to materials in the model.
            Ex, Ey, Ez (memory view): numpy array of electric field values.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        if abstime >= self.start and abstime <= self.stop:
            # Set the time of the waveform evaluation to account for any delay in the start
            time = abstime - self.start
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord
            waveform = next(x for x in G.waveforms if x.ID == self.waveformID)

            if self.polarisation is 'x':
                componentID = 'E' + self.polarisation
                Ex[i, j, k] -= updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4] * waveform.amp * waveform.calculate_value(time, G.dt) * (1 / (G.dy * G.dz))

            elif self.polarisation is 'y':
                componentID = 'E' + self.polarisation
                Ey[i, j, k] -= updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4] * waveform.amp * waveform.calculate_value(time, G.dt) * (1 / (G.dx * G.dz))

            elif self.polarisation is 'z':
                componentID = 'E' + self.polarisation
                Ez[i, j, k] -= updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4] * waveform.amp * waveform.calculate_value(time, G.dt) * (1 / (G.dx * G.dy))


class MagneticDipole(Source):
    """The magnetic dipole is an additive source (magnetic current density)."""

    def __init__(self):
        super(Source, self).__init__()

    def update_magnetic(self, abstime, updatecoeffsH, ID, Hx, Hy, Hz, G):
        """Updates magnetic field values for a magnetic dipole.
            
        Args:
            abstime (float): Absolute time.
            updatecoeffsH (memory view): numpy array of magnetic field update coefficients.
            ID (memory view): numpy array of numeric IDs corresponding to materials in the model.
            Hx, Hy, Hz (memory view): numpy array of magnetic field values.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        if abstime >= self.start and abstime <= self.stop:
            # Set the time of the waveform evaluation to account for any delay in the start
            time = abstime - self.start
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord
            waveform = next(x for x in G.waveforms if x.ID == self.waveformID)

            if self.polarisation is 'x':
                Hx[i, j, k] -= waveform.amp * waveform.calculate_value(time, G.dt) * (G.dt / (G.dx * G.dy * G.dz))

            elif self.polarisation is 'y':
                Hy[i, j, k] -= waveform.amp * waveform.calculate_value(time, G.dt) * (G.dt / (G.dx * G.dy * G.dz))

            elif self.polarisation is 'z':
                Hz[i, j, k] -= waveform.amp * waveform.calculate_value(time, G.dt) * (G.dt / (G.dx * G.dy * G.dz))


class TransmissionLine(Source):
    """The transmission line source is a one-dimensional transmission line which is attached virtually to a grid cell."""

    def __init__(self, G):
        """
        Args:
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        super(Source, self).__init__()
        self.resistance = None

        # Coefficients for ABC termination of end of the transmission line
        self.abcv0 = 0
        self.abcv1 = 0

        # Spatial step of transmission line (based on magic time step for dispersionless behaviour)
        self.dl = c * G.dt

        # Number of nodes in the transmission line (initially a long line to calculate incident voltage and current); consider putting ABCs/PML at end
        self.nl = round_value(0.667 * G.iterations)

        # Nodal position of the one-way injector excitation in the transmission line
        self.srcpos = 5

        # Nodal position of where line connects to antenna/main grid
        self.antpos = 10

        self.voltage = np.zeros(self.nl, dtype=floattype)
        self.current = np.zeros(self.nl, dtype=floattype)
        self.Vinc = np.zeros(G.iterations, dtype=floattype)
        self.Iinc = np.zeros(G.iterations, dtype=floattype)
        self.Vtotal = np.zeros(G.iterations, dtype=floattype)
        self.Itotal = np.zeros(G.iterations, dtype=floattype)

    def calculate_incident_V_I(self, G):
        """Calculates the incident voltage and current with a long length transmission line not connected to the main grid from: http://dx.doi.org/10.1002/mop.10415
            
        Args:
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        abstime = 0
        for timestep in range(G.iterations):
            self.Vinc[timestep] = self.voltage[self.antpos]
            self.Iinc[timestep] = self.current[self.antpos]
            self.update_voltage(abstime, G)
            abstime += 0.5 * G.dt
            self.update_current(abstime, G)
            abstime += 0.5 * G.dt

        # Shorten number of nodes in the transmission line before use with main grid
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

    def update_voltage(self, time, G):
        """Updates voltage values along the transmission line.
            
        Args:
            time (float): Absolute time.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        # Update all the voltage values along the line
        self.voltage[1:self.nl] -= self.resistance * (c * G.dt / self.dl) * (self.current[1:self.nl] - self.current[0:self.nl - 1])

        # Update the voltage at the position of the one-way injector excitation
        waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
        self.voltage[self.srcpos] += (c * G.dt / self.dl) * waveform.amp * waveform.calculate_value(time - 0.5 * G.dt, G.dt)

        # Update ABC before updating current
        self.update_abc(G)

    def update_current(self, time, G):
        """Updates current values along the transmission line.
            
        Args:
            time (float): Absolute time.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        # Update all the current values along the line
        self.current[0:self.nl - 1] -= (1 / self.resistance) * (c * G.dt / self.dl) * (self.voltage[1:self.nl] - self.voltage[0:self.nl - 1])

        # Update the current one node before the position of the one-way injector excitation
        waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
        self.current[self.srcpos - 1] += (c * G.dt / self.dl) * waveform.amp * waveform.calculate_value(time - 0.5 * G.dt, G.dt) * (1 / self.resistance)

    def update_electric(self, abstime, Ex, Ey, Ez, G):
        """Updates electric field value in the main grid from voltage value in the transmission line.
            
        Args:
            abstime (float): Absolute time.
            Ex, Ey, Ez (memory view): numpy array of electric field values.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        if abstime >= self.start and abstime <= self.stop:
            # Set the time of the waveform evaluation to account for any delay in the start
            time = abstime - self.start
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord

            self.update_voltage(time, G)

            if self.polarisation is 'x':
                Ex[i, j, k] = - self.voltage[self.antpos] / G.dx

            elif self.polarisation is 'y':
                Ey[i, j, k] = - self.voltage[self.antpos] / G.dy

            elif self.polarisation is 'z':
                Ez[i, j, k] = - self.voltage[self.antpos] / G.dz

    def update_magnetic(self, abstime, Hx, Hy, Hz, G):
        """Updates current value in transmission line from magnetic field values in the main grid.
            
        Args:
            abstime (float): Absolute time.
            Hx, Hy, Hz (memory view): numpy array of magnetic field values.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        if abstime >= self.start and abstime <= self.stop:
            # Set the time of the waveform evaluation to account for any delay in the start
            time = abstime - self.start
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord

            if self.polarisation is 'x':
                self.current[self.antpos] = Ix(i, j, k, G.Hy, G.Hz, G)

            elif self.polarisation is 'y':
                self.current[self.antpos] = Iy(i, j, k, G.Hx, G.Hz, G)

            elif self.polarisation is 'z':
                self.current[self.antpos] = Iz(i, j, k, G.Hx, G.Hy, G)

            self.update_current(time, G)
