# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
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

from importlib import import_module

from gprMax import config
from gprMax.cython.fields_updates_normal import update_electric as update_electric_cpu
from gprMax.cython.fields_updates_normal import update_magnetic as update_magnetic_cpu
from gprMax.fields_outputs import store_outputs as store_outputs_cpu
from gprMax.updates.updates import Updates
from gprMax.utilities.utilities import timer


class CPUUpdates(Updates):
    """Defines update functions for CPU-based solver."""

    def __init__(self, G):
        """
        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        self.grid = G

    def store_outputs(self):
        """Stores field component values for every receiver and transmission line."""
        store_outputs_cpu(self.grid)

    def store_snapshots(self, iteration):
        """Stores any snapshots.

        Args:
            iteration: int for iteration number.
        """
        for snap in self.grid.snapshots:
            if snap.time == iteration + 1:
                snap.store(self.grid)

    def update_magnetic(self):
        """Updates magnetic field components."""
        update_magnetic_cpu(
            self.grid.nx,
            self.grid.ny,
            self.grid.nz,
            config.get_model_config().ompthreads,
            self.grid.updatecoeffsH,
            self.grid.ID,
            self.grid.Ex,
            self.grid.Ey,
            self.grid.Ez,
            self.grid.Hx,
            self.grid.Hy,
            self.grid.Hz,
        )

    def update_magnetic_pml(self):
        """Updates magnetic field components with the PML correction."""
        for pml in self.grid.pmls["slabs"]:
            pml.update_magnetic()

    def update_magnetic_sources(self):
        """Updates magnetic field components from sources."""
        for source in self.grid.transmissionlines + self.grid.magneticdipoles:
            source.update_magnetic(
                self.grid.iteration,
                self.grid.updatecoeffsH,
                self.grid.ID,
                self.grid.Hx,
                self.grid.Hy,
                self.grid.Hz,
                self.grid,
            )

    def update_electric_a(self):
        """Updates electric field components."""
        # All materials are non-dispersive so do standard update.
        if config.get_model_config().materials["maxpoles"] == 0:
            update_electric_cpu(
                self.grid.nx,
                self.grid.ny,
                self.grid.nz,
                config.get_model_config().ompthreads,
                self.grid.updatecoeffsE,
                self.grid.ID,
                self.grid.Ex,
                self.grid.Ey,
                self.grid.Ez,
                self.grid.Hx,
                self.grid.Hy,
                self.grid.Hz,
            )

        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        else:
            self.dispersive_update_a(
                self.grid.nx,
                self.grid.ny,
                self.grid.nz,
                config.get_model_config().ompthreads,
                config.get_model_config().materials["maxpoles"],
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
                self.grid.Hz,
            )

    def update_electric_pml(self):
        """Updates electric field components with the PML correction."""
        for pml in self.grid.pmls["slabs"]:
            pml.update_electric()

    def update_electric_sources(self):
        """Updates electric field components from sources -
        update any Hertzian dipole sources last.
        """
        for source in self.grid.voltagesources + self.grid.transmissionlines + self.grid.hertziandipoles:
            source.update_electric(
                self.grid.iteration,
                self.grid.updatecoeffsE,
                self.grid.ID,
                self.grid.Ex,
                self.grid.Ey,
                self.grid.Ez,
                self.grid,
            )
        self.grid.iteration += 1

    def update_electric_b(self):
        """If there are any dispersive materials do 2nd part of dispersive
        update - it is split into two parts as it requires present and
        updated electric field values. Therefore it can only be completely
        updated after the electric field has been updated by the PML and
        source updates.
        """
        if config.get_model_config().materials["maxpoles"] > 0:
            self.dispersive_update_b(
                self.grid.nx,
                self.grid.ny,
                self.grid.nz,
                config.get_model_config().ompthreads,
                config.get_model_config().materials["maxpoles"],
                self.grid.updatecoeffsdispersive,
                self.grid.ID,
                self.grid.Tx,
                self.grid.Ty,
                self.grid.Tz,
                self.grid.Ex,
                self.grid.Ey,
                self.grid.Ez,
            )

    def set_dispersive_updates(self):
        """Sets dispersive update functions."""

        poles = "multi" if config.get_model_config().materials["maxpoles"] > 1 else "1"
        precision = "float" if config.sim_config.general["precision"] == "single" else "double"
        dispersion = (
            "complex"
            if config.get_model_config().materials["dispersivedtype"] == config.sim_config.dtypes["complex"]
            else "real"
        )

        update_f = "update_electric_dispersive_{}pole_{}_{}_{}"
        disp_a = update_f.format(poles, "A", precision, dispersion)
        disp_b = update_f.format(poles, "B", precision, dispersion)

        disp_a_f = getattr(import_module("gprMax.cython.fields_updates_dispersive"), disp_a)
        disp_b_f = getattr(import_module("gprMax.cython.fields_updates_dispersive"), disp_b)

        self.dispersive_update_a = disp_a_f
        self.dispersive_update_b = disp_b_f

    def time_start(self):
        """Starts timer used to calculate solving time for model."""
        self.timestart = timer()

    def calculate_solve_time(self):
        """Calculates solving time for model."""
        return timer() - self.timestart

    def finalise(self):
        pass

    def cleanup(self):
        pass
