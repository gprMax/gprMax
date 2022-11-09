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

import numpy as np

import gprMax.config as config
from gprMax.waveforms import Waveform


def hertzian_dipole_fs(iterations, dt, dxdydz, rx):
    """Analytical solution of a z-directed Hertzian dipole in free space with a 
        Gaussian current waveform (http://dx.doi.org/10.1016/0021-9991(83)90103-1).

    Args:
        iterations: int for number of time steps.
        dt: float for time step (seconds).
        dxdydz: tuple of floats for spatial resolution (metres).
        rx: tuple of floats for coordinates of receiver position relative to 
            transmitter position (metres).

    Returns:
        fields: float array containing electric and magnetic field components.
    """

    # Waveform
    w = Waveform()
    w.type = 'gaussianprime'
    w.amp = 1
    w.freq = 1e9

    # Waveform integral
    wint = Waveform()
    wint.type = 'gaussian'
    wint.amp = w.amp
    wint.freq = w.freq

    # Waveform first derivative
    wdot = Waveform()
    wdot.type = 'gaussiandoubleprime'
    wdot.amp = w.amp
    wdot.freq = w.freq

    # Time
    time = np.linspace(0, 1, iterations)
    time *= (iterations * dt)

    # Spatial resolution
    dx = dxdydz[0]
    dy = dxdydz[1]
    dz = dxdydz[2]

    # Length of Hertzian dipole
    dl = dz

    # Coordinates of Rx relative to Tx
    x = rx[0]
    y = rx[1]
    z = rx[2]

    # Coordinates of Rx for Ex FDTD component
    Ex_x = x + 0.5 * dx
    Ex_y = y
    Ex_z = z - 0.5 * dz
    Er_x = np.sqrt((Ex_x**2 + Ex_y**2 + Ex_z**2))
    tau_Ex = Er_x / config.sim_config.em_consts['c']

    # Coordinates of Rx for Ey FDTD component
    Ey_x = x
    Ey_y = y + 0.5 * dy
    Ey_z = z - 0.5 * dz
    Er_y = np.sqrt((Ey_x**2 + Ey_y**2 + Ey_z**2))
    tau_Ey = Er_y / config.sim_config.em_consts['c']

    # Coordinates of Rx for Ez FDTD component
    Ez_x = x
    Ez_y = y
    Ez_z = z
    Er_z = np.sqrt((Ez_x**2 + Ez_y**2 + Ez_z**2))
    tau_Ez = Er_z / config.sim_config.em_consts['c']

    # Coordinates of Rx for Hx FDTD component
    Hx_x = x
    Hx_y = y + 0.5 * dy
    Hx_z = z
    Hr_x = np.sqrt((Hx_x**2 + Hx_y**2 + Hx_z**2))
    tau_Hx = Hr_x / config.sim_config.em_consts['c']

    # Coordinates of Rx for Hy FDTD component
    Hy_x = x + 0.5 * dx
    Hy_y = y
    Hy_z = z
    Hr_y = np.sqrt((Hy_x**2 + Hy_y**2 + Hy_z**2))
    tau_Hy = Hr_y / config.sim_config.em_consts['c']

    # Initialise fields
    fields = np.zeros((iterations, 6))

    # Calculate fields
    for timestep in range(iterations):

        # Calculate values for waveform, I * dl (current multiplied by dipole 
        # length) to match gprMax behaviour
        fint_Ex = wint.calculate_value((timestep * dt) - tau_Ex, dt) * dl
        f_Ex = w.calculate_value((timestep * dt) - tau_Ex, dt) * dl
        fdot_Ex = wdot.calculate_value((timestep * dt) - tau_Ex, dt) * dl

        fint_Ey = wint.calculate_value((timestep * dt) - tau_Ey, dt) * dl
        f_Ey = w.calculate_value((timestep * dt) - tau_Ey, dt) * dl
        fdot_Ey = wdot.calculate_value((timestep * dt) - tau_Ey, dt) * dl

        fint_Ez = wint.calculate_value((timestep * dt) - tau_Ez, dt) * dl
        f_Ez = w.calculate_value((timestep * dt) - tau_Ez, dt) * dl
        fdot_Ez = wdot.calculate_value((timestep * dt) - tau_Ez, dt) * dl

        f_Hx = w.calculate_value((timestep * dt) - tau_Hx, dt) * dl
        fdot_Hx = wdot.calculate_value((timestep * dt) - tau_Hx, dt) * dl

        f_Hy = w.calculate_value((timestep * dt) - tau_Hy, dt) * dl
        fdot_Hy = wdot.calculate_value((timestep * dt) - tau_Hy, dt) * dl

        # Ex
        fields[timestep, 0] = (((Ex_x * Ex_z) / (4 * np.pi * config.sim_config.em_consts['e0'] * Er_x**5)) * 
                               (3 * (fint_Ex + (tau_Ex * f_Ex)) + (tau_Ex**2 * fdot_Ex)))

        # Ey
        try:
            tmp = Ey_y / Ey_x
        except ZeroDivisionError:
            tmp = 0
        fields[timestep, 1] = (tmp * ((Ey_x * Ey_z) / (4 * np.pi * config.sim_config.em_consts['e0'] * Er_y**5)) * 
                               (3 * (fint_Ey + (tau_Ey * f_Ey)) + (tau_Ey**2 * fdot_Ey)))

        # Ez
        fields[timestep, 2] = ((1 / (4 * np.pi * config.sim_config.em_consts['e0'] * Er_z**5)) * 
                               ((2 * Ez_z**2 - (Ez_x**2 + Ez_y**2)) * (fint_Ez + (tau_Ez * f_Ez)) - 
                               (Ez_x**2 + Ez_y**2) * tau_Ez**2 * fdot_Ez))

        # Hx
        fields[timestep, 3] = - (Hx_y / (4 * np.pi * Hr_x**3)) * (f_Hx + (tau_Hx * fdot_Hx))

        # Hy
        try:
            tmp = Hy_x / Hy_y
        except ZeroDivisionError:
            tmp = 0
        fields[timestep, 4] = - tmp * (- (Hy_y / (4 * np.pi * Hr_y**3)) * (f_Hy + (tau_Hy * fdot_Hy)))

        # Hz
        fields[timestep, 5] = 0

    return fields
