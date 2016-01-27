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

from gprMax.utilities import round_value


class Waveform:
    """Definitions of waveform shapes that can be used with sources."""
    
    waveformtypes = ['gaussian', 'gaussiandot', 'gaussiandotnorm', 'gaussiandotdot', 'gaussiandotdotnorm', 'ricker', 'sine', 'contsine', 'impulse', 'user']
    
    def __init__(self):
        self.ID = None
        self.type = None
        self.amp = 1
        self.freq = 0
        self.uservalues = None

    def calculate_value(self, time, dt):
        """Calculates value of the waveform at a specific time.
            
        Args:
            time (float): Absolute time.
            dt (float): Absolute time discretisation.
            
        Returns:
            waveform (float): Calculated value for waveform.
        """
        
        # Coefficients for certain waveforms
        if self.type == 'gaussian' or self.type == 'gaussiandot' or self.type == 'gaussiandotdot':
            chi = 1 / self.freq
            zeta = 2 * np.pi * np.pi * self.freq * self.freq
            delay = time - chi
        elif self.type == 'gaussiandotnorm' or self.type == 'gaussiandotdotnorm' or self.type == 'ricker':
            chi = np.sqrt(2) / self.freq
            zeta = np.pi * np.pi * self.freq * self.freq
            delay = time - chi
    
        # Waveforms
        if self.type == 'gaussian':
            waveform = np.exp(-zeta * delay * delay)
        
        elif self.type == 'gaussiandot':
            waveform = -2 * zeta * delay * np.exp(-zeta * delay * delay)
        
        elif self.type == 'gaussiandotnorm':
            normalise = np.sqrt(np.exp(1) / (2 * zeta))
            waveform = -2 * zeta * delay * np.exp(-zeta * delay * delay) * normalise
        
        elif self.type == 'gaussiandotdot':
            waveform = 2 * zeta * (2 * zeta * delay * delay - 1) * np.exp(-zeta * delay * delay)
        
        elif self.type == 'gaussiandotdotnorm':
            normalise = 1 / (2 * zeta)
            waveform = 2 * zeta * (2 * zeta * delay * delay - 1) * np.exp(-zeta * delay * delay) * normalise

        elif self.type == 'ricker':
            normalise = 1 / (2 * zeta)
            waveform = - (2 * zeta * (2 * zeta * delay * delay - 1) * np.exp(-zeta * delay * delay)) * normalise

        elif self.type == 'sine':
            waveform = np.sin(2 * np.pi * self.freq * time)
            if time * self.freq > 1:
                waveform = 0
                
        elif self.type == 'contsine':
            rampamp = 0.25
            ramp = rampamp * time * self.freq
            if ramp > 1:
                ramp = 1
            waveform = ramp * np.sin(2 * np.pi * self.freq * time)

        elif self.type == 'impulse':
            # time < G.dt condition required to do impulsive magnetic dipole
            if time == 0 or time < dt:
                waveform = 1
            elif time >= dt:
                waveform = 0
        
        elif self.type == 'user':
            index = round_value(time / dt)
            # Check to see if there are still user specified values and if not use zero
            if index > len(self.uservalues) - 1:
                waveform = 0
            else:
                waveform = self.uservalues[index]
        
        return waveform