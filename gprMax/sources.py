# Copyright (C) 2015: The University of Edinburgh
#            Authors: Craig Warren and Antonis Giannopoulos
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

from gprMax.constants import c, floattype
from gprMax.grid import Ix, Iy, Iz
from gprMax.utilities import rvalue


class VoltageSource:
    """The voltage source can be a hard source if it's resistance is zero, i.e. the time variation of the specified electric field component is prescribed. If it's resistance is non-zero it behaves as a resistive voltage source."""
    
    def __init__(self):
        self.polarisation = None
        self.positionx = None
        self.positiony = None
        self.positionz = None
        self.start = None
        self.stop = None
        self.resistance = None
        self.waveformID = None

    def update_E(self, abstime, updatecoeffsE, ID, Ex, Ey, Ez, G):
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
            i = self.positionx
            j = self.positiony
            k = self.positionz
            waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
            
            if self.polarisation is 'x':
                if self.resistance != 0:
                    Ex[i, j, k] -= updatecoeffsE[ID[0, i, j, k], 4] * waveform.amp * waveform.calculate_value(time, G.dt) * (1 / (self.resistance * G.dy * G.dz))
                else:
                    Ex[i, j, k] = -1 * waveform.amp * waveform.calculate_value(time, G.dt) / G.dx

            elif self.polarisation is 'y':
                if self.resistance != 0:
                    Ey[i, j, k] -= updatecoeffsE[ID[1, i, j, k], 4] * waveform.amp * waveform.calculate_value(time, G.dt) * (1 / (self.resistance * G.dx * G.dz))
                else:
                    Ey[i, j, k] = -1 * waveform.amp * waveform.calculate_value(time, G.dt) / G.dy

            elif self.polarisation is 'z':
                if self.resistance != 0:
                    Ez[i, j, k] -= updatecoeffsE[ID[2, i, j, k], 4] * waveform.amp * waveform.calculate_value(time, G.dt) * (1 / (self.resistance * G.dx * G.dy))
                else:
                    Ez[i, j, k] = -1 * waveform.amp * waveform.calculate_value(time, G.dt) / G.dz


class HertzianDipole:
    """The Hertzian dipole is an additive source (electric current density)."""
    
    def __init__(self):
        self.polarisation = None
        self.positionx = None
        self.positiony = None
        self.positionz = None
        self.start = None
        self.stop = None
        self.waveformID = None

    def update_E(self, abstime, updatecoeffsE, ID, Ex, Ey, Ez, G):
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
            i = self.positionx
            j = self.positiony
            k = self.positionz
            waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
            
            if self.polarisation is 'x':
                Ex[i, j, k] -= updatecoeffsE[ID[0, i, j, k], 4] * waveform.amp * waveform.calculate_value(time, G.dt) * (1 / (G.dy * G.dz))

            elif self.polarisation is 'y':
                Ey[i, j, k] -= updatecoeffsE[ID[1, i, j, k], 4] * waveform.amp * waveform.calculate_value(time, G.dt) * (1 / (G.dx * G.dz))

            elif self.polarisation is 'z':
                Ez[i, j, k] -= updatecoeffsE[ID[2, i, j, k], 4] * waveform.amp * waveform.calculate_value(time, G.dt) * (1 / (G.dx * G.dy))


class MagneticDipole:
    """The magnetic dipole is an additive source (magnetic current density)."""
    
    def __init__(self):
        self.polarisation = None
        self.positionx = None
        self.positiony = None
        self.positionz = None
        self.start = None
        self.stop = None
        self.waveformID = None

    def update_H(self, abstime, updatecoeffsH, ID, Hx, Hy, Hz, G):
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
            i = self.positionx
            j = self.positiony
            k = self.positionz
            waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
            
            if self.polarisation is 'x':
                Hx[i, j, k] -= waveform.amp  * waveform.calculate_value(time, G.dt) * (G.dt / (G.dx * G.dy * G.dz))

            elif self.polarisation is 'y':
                Hy[i, j, k] -= waveform.amp  * waveform.calculate_value(time, G.dt) * (G.dt / (G.dx * G.dy * G.dz))

            elif self.polarisation is 'z':
                Hz[i, j, k] -= waveform.amp  * waveform.calculate_value(time, G.dt) * (G.dt / (G.dx * G.dy * G.dz))


class TransmissionLine:
    """The transmission line source is a one-dimensional transmission line which is attached virtually to a grid cell."""
    
    def __init__(self, G):
        """
        Args:
            G (class): Grid class instance - holds essential parameters describing the model.
        """
        
        self.polarisation = None
        self.positionx = None
        self.positiony = None
        self.positionz = None
        self.start = None
        self.stop = None
        self.length = None
        self.resistance = None
        self.waveformID = None
        
        # Coefficients for ABC termination of end of the transmission line
        self.abcv0 = 0
        self.abcv1 = 0

        # Spatial step of transmission line
        self.dl = np.sqrt(3) * c * G.dt
        
        # Nodal position of one-way injector excitation in the transmission line
        self.source = 10
        
        # Number of nodes in the transmission line; add nodes to the length to account for position of one-way injector
        self.nl = rvalue(self.length/self.dl) + self.source

        self.voltage = np.zeros(self.nl, dtype=floattype)
        self.current = np.zeros(self.nl, dtype=floattype)
    
    def update_abc(self, G):
        """Updates absorbing boundary condition at end of the transmission line.
            
        Args:
            G (class): Grid class instance - holds essential parameters describing the model.
        """
    
        tmp = (c * G.dt - self.dl) / (c * G.dt + self.dl)
    
        self.voltage[0] = (self.voltage[1] - self.abcv0) + self.abcv1
        self.abcv0 = self.voltage[0]
        self.abcv1 = self.voltage[1]

    def update_voltage(self, time, G):
        """Updates voltage values along the transmission line.
            
        Args:
            time (float): Absolute time.
            G (class): Grid class instance - holds essential parameters describing the model.
        """
            
        waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
        
        # Update all the voltage values along the line
        for i in range(1, self.nl):
            self.voltage[i] -= self.resistance * (c * G.dt / self.dl) * (self.current[i] - self.current[i - 1])
        
        # Update the voltage at the position of the one-way injector excitation
        self.voltage[self.source] += (c * G.dt / self.dl) * waveform.amp * waveform.calculate_value(time - 0.5 * G.dt, G.dt)

        # Update ABC before updating current
        self.update_abc(G)

    def update_current(self, time, G):
        """Updates current values along the transmission line.
            
        Args:
            time (float): Absolute time.
            G (class): Grid class instance - holds essential parameters describing the model.
        """
            
        waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
        
        # Update all the current values along the line
        for i in range(0, self.nl - 1):
            self.current[i] -= (1 / self.resistance) * (c * G.dt / self.dl) * (self.voltage[i + 1] - self.voltage[i])
        
        # Update the current one node before the position of the one-way injector excitation
        self.current[self.source - 1] += (c * G.dt / self.dl) * waveform.amp * waveform.calculate_value(time - 0.5 * G.dt, G.dt) * (1 / self.resistance)

    def update_E(self, abstime, Ex, Ey, Ez, G):
        """Updates electric field value in the main grid from voltage value in the transmission line.
            
        Args:
            abstime (float): Absolute time.
            Ex, Ey, Ez (memory view): numpy array of electric field values.
            G (class): Grid class instance - holds essential parameters describing the model.
        """
        
        if abstime >= self.start and abstime <= self.stop:
            # Set the time of the waveform evaluation to account for any delay in the start
            time = abstime - self.start
            i = self.positionx
            j = self.positiony
            k = self.positionz
            
            self.update_voltage(time, G)
            
            if self.polarisation is 'x':
                Ex[i, j, k] = - self.voltage[self.nl - 1] / G.dx

            elif self.polarisation is 'y':
                Ey[i, j, k] = - self.voltage[self.nl - 1] / G.dy

            elif self.polarisation is 'z':
                Ez[i, j, k] = - self.voltage[self.nl - 1] / G.dz

    def update_H(self, abstime, Hx, Hy, Hz, G):
        """Updates current value in transmission line from magnetic field values in the main grid.
            
        Args:
            abstime (float): Absolute time.
            Hx, Hy, Hz (memory view): numpy array of magnetic field values.
            G (class): Grid class instance - holds essential parameters describing the model.
        """
        
        if abstime >= self.start and abstime <= self.stop:
            # Set the time of the waveform evaluation to account for any delay in the start
            time = abstime - self.start
            i = self.positionx
            j = self.positiony
            k = self.positionz
            
            if self.polarisation is 'x':
                self.current[self.nl - 1] = Ix(i, j, k, G.Hy, G.Hz, G)

            elif self.polarisation is 'y':
                self.current[self.nl - 1] = Iy(i, j, k, G.Hx, G.Hz, G)

            elif self.polarisation is 'z':
                self.current[self.nl - 1] = Iz(i, j, k, G.Hx, G.Hy, G)

            self.update_current(time, G)

