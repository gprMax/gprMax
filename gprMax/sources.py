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

from gprMax.utilities import rvalue


class VoltageSource:
    """Voltage sources."""
    
    def __init__(self):
        self.polarisation = None
        self.positionx = None
        self.positiony = None
        self.positionz = None
        self.start = None
        self.stop = None
        self.resistance = None
        self.waveformID = None

    def update_fields(self, abstime, timestep, updatecoeffsE, ID, Ex, Ey, Ez, G):
        """Updates electric field values for a voltage source.
            
        Args:
            abstime (float): Absolute time.
            timestep (int): Iteration number.
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
    """Hertzian dipoles, i.e. normal additive source (current density)."""
    
    def __init__(self):
        self.polarisation = None
        self.positionx = None
        self.positiony = None
        self.positionz = None
        self.start = None
        self.stop = None
        self.waveformID = None

    def update_fields(self, abstime, timestep, updatecoeffsE, ID, Ex, Ey, Ez, G):
        """Updates electric field values for a Hertzian dipole.
            
        Args:
            abstime (float): Absolute time.
            timestep (int): Iteration number.
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
    """Magnetic dipoles, i.e. current on a small loop."""
    
    def __init__(self):
        self.polarisation = None
        self.positionx = None
        self.positiony = None
        self.positionz = None
        self.start = None
        self.stop = None
        self.waveformID = None

    def update_fields(self, abstime, timestep, updatecoeffsH, ID, Hx, Hy, Hz, G):
        """Updates electric field values for a magnetic dipole.
            
        Args:
            abstime (float): Absolute time.
            timestep (int): Iteration number.
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

