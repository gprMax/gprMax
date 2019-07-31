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

class CPUUpdates:

    def __init__(self, G):
        self.G = G

    def store_outputs(self, iteration):
        # Store field component values for every receiver and transmission line
        store_outputs(iteration,
                      self.G.Ex,
                      self.G.Ey,
                      self.G.Ez,
                      self.G.Hx,
                      self.G.Hy,
                      self.G.Hz,
                      self.G)

    def store_snapshots(self, iteration):
        # Store any snapshots
        for snap in self.G.snapshots:
            if snap.time == iteration + 1:
                snap.store(self.G)

    def update_magnetic(self):
        # Update magnetic field components
        update_magnetic(self.G.nx,
                        self.G.ny,
                        self.G.nz,
                        config.hostinfo['ompthreads'],
                        self.G.updatecoeffsH,
                        self.G.ID,
                        self.G.Ex,
                        self.G.Ey,
                        self.G.Ez,
                        self.G.Hx,
                        self.G.Hy,
                        self.G.Hz)

    def update_magnetic_pml(self, iteration):
        # Update magnetic field components with the PML correction
        for pml in self.G.pmls:
            pml.update_magnetic(self.G)

    def update_magnetic_sources(self, iteration):
        # Update magnetic field components from sources
        for source in self.G.transmissionlines + self.G.magneticdipoles:
            source.update_magnetic(iteration,
                                   self.G.updatecoeffsH,
                                   self.G.ID,
                                   self.G.Hx,
                                   self.G.Hy,
                                   self.G.Hz,
                                   self.G)

    def update_electric_a(self):
        # Update electric field components
        # All materials are non-dispersive so do standard update
        if Material.maxpoles == 0:
            update_electric(self.G.nx,
                            self.G.ny,
                            self.G.nz,
                            config.hostinfo['ompthreads'],
                            self.G.updatecoeffsE,
                            self.G.ID,
                            self.G.Ex,
                            self.G.Ey,
                            self.G.Ez,
                            self.G.Hx,
                            self.G.Hy,
                            self.G.Hz)

        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        elif Material.maxpoles == 1:
            update_electric_dispersive_1pole_A(self.G.nx,
                                               self.G.ny,
                                               self.G.nz,
                                               config.hostinfo['ompthreads'],
                                               self.G.updatecoeffsE,
                                               self.G.updatecoeffsdispersive,
                                               self.G.ID,
                                               self.G.Tx,
                                               self.G.Ty,
                                               self.G.Tz,
                                               self.G.Ex,
                                               self.G.Ey,
                                               self.G.Ez,
                                               self.G.Hx,
                                               self.G.Hy,
                                               self.G.Hz)
        elif Material.maxpoles > 1:
            update_electric_dispersive_multipole_A(self.G.nx,
                                                   self.G.ny,
                                                   self.G.nz,
                                                   config.hostinfo['ompthreads'],
                                                   Material.maxpoles,
                                                   self.G.updatecoeffsE,
                                                   self.G.updatecoeffsdispersive,
                                                   self.G.ID,
                                                   self.G.Tx,
                                                   self.G.Ty,
                                                   self.G.Tz,
                                                   self.G.Ex,
                                                   self.G.Ey,
                                                   self.G.Ez,
                                                   self.G.Hx,
                                                   self.G.Hy,
                                                   self.G.Hz)

    def update_electric_pml(self):
        # Update electric field components with the PML correction
        for pml in self.G.pmls:
            pml.update_electric(self.G)

    def update_electric_sources(self, iteration):
        # Update electric field components from sources (update any Hertzian dipole sources last)
        for source in self.G.voltagesources + self.G.transmissionlines + self.G.hertziandipoles:
            source.update_electric(iteration, self.G.updatecoeffsE, self.G.ID, self.G.Ex, self.G.Ey, self.G.Ez, self.G)

    def update_electric_b(self):
        # If there are any dispersive materials do 2nd part of dispersive update
        # (it is split into two parts as it requires present and updated electric
        # field values). Therefore it can only be completely updated after the
        # electric field has been updated by the PML and source updates.
        if Material.maxpoles == 1:
            update_electric_dispersive_1pole_B(self.G.nx,
                                               self.G.ny,
                                               self.G.nz,
                                               config.hostinfo['ompthreads'],
                                               self.G.updatecoeffsdispersive,
                                               self.G.ID,
                                               self.G.Tx,
                                               self.G.Ty,
                                               self.G.Tz,
                                               self.G.Ex,
                                               self.G.Ey,
                                               self.G.Ez)
        elif Material.maxpoles > 1:
            update_electric_dispersive_multipole_B(self.G.nx,
                                                   self.G.ny,
                                                   self.G.nz,
                                                   config.hostinfo['ompthreads'],
                                                   Material.maxpoles,
                                                   self.G.updatecoeffsdispersive,
                                                   self.G.ID,
                                                   self.G.Tx,
                                                   self.G.Ty,
                                                   self.G.Tz,
                                                   self.G.Ex,
                                                   self.G.Ey,
                                                   self.G.Ez)


class SubgridUpdates:
    pass

class GPUUpdates:
    pass
