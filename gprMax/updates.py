# Copyright (C) 2015-2019: The University of Edinburgh
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
from importlib import import_module
from gprMax.fields_outputs import store_outputs
import gprMax.config as config
from gprMax.cython.fields_updates_normal import update_electric
from gprMax.cython.fields_updates_normal import update_magnetic


class CPUUpdates:

    def __init__(self, G):
        self.grid = G
        self.dispersive_update_a = None
        self.dispersive_update_b = None

    def store_outputs(self):
        # Store field component values for every receiver and transmission line
        store_outputs(self.grid)

    def store_snapshots(self, iteration):
        # Store any snapshots
        for snap in self.grid.snapshots:
            if snap.time == iteration + 1:
                snap.store(self.grid)

    def update_magnetic(self):
        # Update magnetic field components
        update_magnetic(self.grid.nx,
                        self.grid.ny,
                        self.grid.nz,
                        config.hostinfo['ompthreads'],
                        self.grid.updatecoeffsH,
                        self.grid.ID,
                        self.grid.Ex,
                        self.grid.Ey,
                        self.grid.Ez,
                        self.grid.Hx,
                        self.grid.Hy,
                        self.grid.Hz)

    def update_magnetic_pml(self):
        # Update magnetic field components with the PML correction
        for pml in self.grid.pmls:
            pml.update_magnetic(self.grid)

    def update_magnetic_sources(self, iteration):
        # Update magnetic field components from sources
        for source in self.grid.transmissionlines + self.grid.magneticdipoles:
            source.update_magnetic(iteration,
                                   self.grid.updatecoeffsH,
                                   self.grid.ID,
                                   self.grid.Hx,
                                   self.grid.Hy,
                                   self.grid.Hz,
                                   self.grid)

    def update_electric_a(self):
        # Update electric field components
        # All materials are non-dispersive so do standard update
        if config.materials['maxpoles'] == 0:
            update_electric(self.grid.nx,
                            self.grid.ny,
                            self.grid.nz,
                            config.hostinfo['ompthreads'],
                            self.grid.updatecoeffsE,
                            self.grid.ID,
                            self.grid.Ex,
                            self.grid.Ey,
                            self.grid.Ez,
                            self.grid.Hx,
                            self.grid.Hy,
                            self.grid.Hz)

        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        else:
            self.dispersive_update_a(self.grid.nx,
                                     self.grid.ny,
                                     self.grid.nz,
                                     config.hostinfo['ompthreads'],
                                     config.materials['maxpoles'],
                                     self.grid.updatecoeffsE,
                                     self.grid.updatecoeffsdispersive,
                                     self.grid.ID,
                                     self.grid.Tx,
                                     self.grid.Ty,
                                     self.grid.Tz,
                                     self.grid.Ex,
                                     self.grid.Ey,
                                     self.grid.Ez,
                                     self.grid.Hx,
                                     self.grid.Hy,
                                     self.grid.Hz)

    def update_electric_pml(self):
        # Update electric field components with the PML correction
        for pml in self.grid.pmls:
            pml.update_electric(self.grid)

    def update_electric_sources(self, iteration):
        # Update electric field components from sources (update any Hertzian dipole sources last)
        for source in self.grid.voltagesources + self.grid.transmissionlines + self.grid.hertziandipoles:
            source.update_electric(iteration, self.grid.updatecoeffsE, self.grid.ID, self.grid.Ex, self.grid.Ey, self.grid.Ez, self.grid)

    def update_electric_b(self):
        # If there are any dispersive materials do 2nd part of dispersive update
        # (it is split into two parts as it requires present and updated electric
        # field values). Therefore it can only be completely updated after the
        # electric field has been updated by the PML and source updates.
        if config.materials['maxpoles'] != 0:
            self.dispersive_update_b(self.grid.nx,
                                     self.grid.ny,
                                     self.grid.nz,
                                     config.hostinfo['ompthreads'],
                                     config.materials['maxpoles'],
                                     self.grid.updatecoeffsdispersive,
                                     self.grid.ID,
                                     self.grid.Tx,
                                     self.grid.Ty,
                                     self.grid.Tz,
                                     self.grid.Ex,
                                     self.grid.Ey,
                                     self.grid.Ez)

    def adapt_dispersive_config(self, config):

        if config.materials['maxpoles'] > 1:
            poles = 'multi'

        else:
            poles = '1'

        if config.precision == 'single':
            type = 'float'

        else:
            type = 'double'

        if config.materials['dispersivedtype'] == config.dtypes['complex']:
            dispersion = 'complex'
        else:
            dispersion = 'real'

        class Props():
            pass

        props = Props()
        props.poles = poles
        props.precision = type
        props.dispersion_type = dispersion

        return props

    def set_dispersive_updates(self, props):
        """Function to set dispersive update functions based on model."""

        update_f = 'update_electric_dispersive_{}pole_{}_{}_{}'
        disp_a = update_f.format(props.poles, 'A', props.precision, props.dispersion_type)
        disp_b = update_f.format(props.poles, 'B', props.precision, props.dispersion_type)

        disp_a_f = getattr(import_module('gprMax.cython.fields_updates_dispersive'), disp_a)
        disp_b_f = getattr(import_module('gprMax.cython.fields_updates_dispersive'), disp_b)

        self.dispersive_update_a = disp_a_f
        self.dispersive_update_b = disp_b_f


class GPUUpdates:
    pass
