# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley, 
#                          and Nathan Mannall
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
import numpy.typing as npt
import math
import logging

import gprMax.config as config
from gprMax.waveforms import Waveform

from .cython.plane_wave import (
    calculate1DWaveformValues,
    getSource,
    updatePlaneWave_magnetic,
    updatePlaneWave_magnetic_axial,
    updatePlaneWave_electric,
    updatePlaneWave_electric_axial,
    updatePlaneWave_electric_dispersive,
    updatePlaneWave_electric_dispersive_axial,
)
from .utilities.utilities import round_value

logger = logging.getLogger(__name__)


class Source:
    """Super-class which describes a generic source."""

    def __init__(self):
        self.ID: str
        self.polarisation = None
        self.coord = np.zeros(3, dtype=np.int32)
        self.coordorigin = np.zeros(3, dtype=np.int32)
        self.start = 0.0
        self.stop = 0.0
        self.waveformID = None
        # Waveform values for sources that need to be calculated on whole timesteps
        self.waveformvalues_wholedt = None
        # Waveform values for sources that need to be calculated on half timesteps
        self.waveformvalues_halfdt = None

    @property
    def xcoord(self) -> int:
        return self.coord[0]

    @xcoord.setter
    def xcoord(self, value: int):
        self.coord[0] = value

    @property
    def ycoord(self) -> int:
        return self.coord[1]

    @ycoord.setter
    def ycoord(self, value: int):
        self.coord[1] = value

    @property
    def zcoord(self) -> int:
        return self.coord[2]

    @zcoord.setter
    def zcoord(self, value: int):
        self.coord[2] = value

    @property
    def xcoordorigin(self) -> int:
        return self.coordorigin[0]

    @xcoordorigin.setter
    def xcoordorigin(self, value: int):
        self.coordorigin[0] = value

    @property
    def ycoordorigin(self) -> int:
        return self.coordorigin[1]

    @ycoordorigin.setter
    def ycoordorigin(self, value: int):
        self.coordorigin[1] = value

    @property
    def zcoordorigin(self) -> int:
        return self.coordorigin[2]

    @zcoordorigin.setter
    def zcoordorigin(self, value: int):
        self.coordorigin[2] = value


class VoltageSource(Source):
    """A voltage source can be a hard source if it's resistance is zero,
    i.e. the time variation of the specified electric field component
    is prescribed. If it's resistance is non-zero it behaves as a resistive
    voltage source.
    """

    def __init__(self):
        super().__init__()
        self.resistance = None

    def calculate_waveform_values(self, G):
        """Calculates all waveform values for source for duration of simulation.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        # Check if a source matches existing source in terms of waveform and
        # does not have a customised start/stop time. If so, use its
        # pre-calculated waveform values, otherwise calculate them.
        src_match = False

        if self.start == 0 and self.stop == G.timewindow:
            for src in G.voltagesources:
                if src.waveformID == self.waveformID:
                    src_match = True
                    self.waveformvalues_halfdt = src.waveformvalues_halfdt
                    self.waveformvalues_wholedt = src.waveformvalues_wholedt

        if not src_match:
            waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
            self.waveformvalues_halfdt = np.zeros(
                (G.iterations), dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.waveformvalues_wholedt = np.zeros(
                (G.iterations), dtype=config.sim_config.dtypes["float_or_double"]
            )

            for iteration in range(G.iterations):
                time = G.dt * iteration
                if time >= self.start and time <= self.stop:
                    # Set the time of the waveform evaluation to account for any
                    # delay in the start
                    time -= self.start
                    self.waveformvalues_halfdt[iteration] = waveform.calculate_value(
                        time + 0.5 * G.dt, G.dt
                    )
                    self.waveformvalues_wholedt[iteration] = waveform.calculate_value(time, G.dt)

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
                        * self.waveformvalues_wholedt[iteration]
                        * (1 / (self.resistance * G.dy * G.dz))
                    )
                else:
                    Ex[i, j, k] = -1 * self.waveformvalues_halfdt[iteration] / G.dx

            elif self.polarisation == "y":
                if self.resistance != 0:
                    Ey[i, j, k] -= (
                        updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                        * self.waveformvalues_wholedt[iteration]
                        * (1 / (self.resistance * G.dx * G.dz))
                    )
                else:
                    Ey[i, j, k] = -1 * self.waveformvalues_halfdt[iteration] / G.dy

            elif self.polarisation == "z":
                if self.resistance != 0:
                    Ez[i, j, k] -= (
                        updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                        * self.waveformvalues_wholedt[iteration]
                        * (1 / (self.resistance * G.dx * G.dy))
                    )
                else:
                    Ez[i, j, k] = -1 * self.waveformvalues_halfdt[iteration] / G.dz

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
        self.dl = 0.0

    def calculate_waveform_values(self, G):
        """Calculates all waveform values for source for duration of simulation.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        # Check if a source matches existing source in terms of waveform and
        # does not have a customised start/stop time. If so, use its
        # pre-calculated waveform values, otherwise calculate them.
        src_match = False

        if self.start == 0 and self.stop == G.timewindow:
            for src in G.hertziandipoles:
                if src.waveformID == self.waveformID:
                    src_match = True
                    self.waveformvalues_wholedt = src.waveformvalues_wholedt

        if not src_match:
            waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
            self.waveformvalues_wholedt = np.zeros(
                (G.iterations), dtype=config.sim_config.dtypes["float_or_double"]
            )

            for iteration in range(G.iterations):
                time = G.dt * iteration
                if time >= self.start and time <= self.stop:
                    # Set the time of the waveform evaluation to account for any
                    # delay in the start
                    time -= self.start
                    self.waveformvalues_wholedt[iteration] = waveform.calculate_value(
                        time, G.dt
                    )

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
                    * self.waveformvalues_wholedt[iteration]
                    * self.dl
                    * (1 / (G.dx * G.dy * G.dz))
                )

            elif self.polarisation == "y":
                Ey[i, j, k] -= (
                    updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_wholedt[iteration]
                    * self.dl
                    * (1 / (G.dx * G.dy * G.dz))
                )

            elif self.polarisation == "z":
                Ez[i, j, k] -= (
                    updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_wholedt[iteration]
                    * self.dl
                    * (1 / (G.dx * G.dy * G.dz))
                )


class MagneticDipole(Source):
    """A magnetic dipole is an additive source (magnetic current density)."""

    def calculate_waveform_values(self, G):
        """Calculates all waveform values for source for duration of simulation.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        # Check if a source matches existing source in terms of waveform and
        # does not have a customised start/stop time. If so, use its
        # pre-calculated waveform values, otherwise calculate them.
        src_match = False

        if self.start == 0 and self.stop == G.timewindow:
            for src in G.magneticdipoles:
                if src.waveformID == self.waveformID:
                    src_match = True
                    self.waveformvalues_halfdt = src.waveformvalues_halfdt

        if not src_match:
            waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
            self.waveformvalues_halfdt = np.zeros(
                (G.iterations), dtype=config.sim_config.dtypes["float_or_double"]
            )

            for iteration in range(G.iterations):
                time = G.dt * iteration
                if time >= self.start and time <= self.stop:
                    # Set the time of the waveform evaluation to account for any
                    # delay in the start
                    time -= self.start
                    self.waveformvalues_halfdt[iteration] = waveform.calculate_value(time + 0.5 * G.dt, G.dt)

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
                    * self.waveformvalues_halfdt[iteration]
                    * (1 / (G.dx * G.dy * G.dz))
                )

            elif self.polarisation == "y":
                Hy[i, j, k] -= (
                    updatecoeffsH[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_halfdt[iteration]
                    * (1 / (G.dx * G.dy * G.dz))
                )

            elif self.polarisation == "z":
                Hz[i, j, k] -= (
                    updatecoeffsH[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_halfdt[iteration]
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
    srcwaves = np.zeros(
        (len(sources), G.iterations), dtype=config.sim_config.dtypes["float_or_double"]
    )
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
            srcwaves[i, :] = src.waveformvalues_wholedt
        elif src.__class__.__name__ == "VoltageSource":
            if src.resistance:
                srcinfo2[i] = src.resistance
                srcwaves[i, :] = src.waveformvalues_wholedt
            else:
                srcinfo2[i] = 0
                srcwaves[i, :] = src.waveformvalues_halfdt
            srcinfo2[i] = src.resistance
            srcwaves[i, :] = src.waveformvalues_wholedt
        elif src.__class__.__name__ == "MagneticDipole":
            srcwaves[i, :] = src.waveformvalues_halfdt

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
    elif config.sim_config.general["solver"] == "metal":
        # Metal doesn't use a queue parameter, need to get device from config
        dev = config.get_model_config().device["dev"]
        srcinfo1_dev = dev.newBufferWithBytes_length_options_(srcinfo1.tobytes(), srcinfo1.nbytes, 0)
        srcinfo2_dev = dev.newBufferWithBytes_length_options_(srcinfo2.tobytes(), srcinfo2.nbytes, 0)
        srcwaves_dev = dev.newBufferWithBytes_length_options_(srcwaves.tobytes(), srcwaves.nbytes, 0)

    return srcinfo1_dev, srcinfo2_dev, srcwaves_dev


class TransmissionLine(Source):
    """A transmission line source is a one-dimensional transmission line
    which is attached virtually to a grid cell.
    """

    def __init__(self, iterations: int, dt: float):
        """
        Args:
            iterations: number of iterations
            dt: time step of the grid
        """

        super().__init__()
        self.resistance = None
        self.iterations = iterations

        # Coefficients for ABC termination of end of the transmission line
        self.abcv0 = 0
        self.abcv1 = 0

        # Spatial step of transmission line (N.B if the magic time step is
        # used it results in instabilities for certain impedances)
        self.dl = np.sqrt(3) * config.c * dt

        # Number of cells in the transmission line (initially a long line to
        # calculate incident voltage and current); consider putting ABCs/PML at end
        self.nl = round_value(0.667 * self.iterations)

        # Cell position of the one-way injector excitation in the transmission line
        self.srcpos = 5

        # Cell position of where line connects to antenna/main grid
        self.antpos = 10

        self.voltage = np.zeros(self.nl, dtype=config.sim_config.dtypes["float_or_double"])
        self.current = np.zeros(self.nl, dtype=config.sim_config.dtypes["float_or_double"])
        self.Vinc = np.zeros(self.iterations, dtype=config.sim_config.dtypes["float_or_double"])
        self.Iinc = np.zeros(self.iterations, dtype=config.sim_config.dtypes["float_or_double"])
        self.Vtotal = np.zeros(self.iterations, dtype=config.sim_config.dtypes["float_or_double"])
        self.Itotal = np.zeros(self.iterations, dtype=config.sim_config.dtypes["float_or_double"])

    def calculate_waveform_values(self, G):
        """Calculates all waveform values for source for duration of simulation.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        # Check if a source matches existing source in terms of waveform and
        # does not have a customised start/stop time. If so, use its
        # pre-calculated waveform values, otherwise calculate them.
        src_match = False

        if self.start == 0 and self.stop == G.timewindow:
            for src in G.transmissionlines:
                if src.waveformID == self.waveformID:
                    src_match = True
                    self.waveformvalues_wholedt = src.waveformvalues_wholedt
                    self.waveformvalues_halfdt = src.waveformvalues_halfdt

        if not src_match:
            waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
            self.waveformvalues_wholedt = np.zeros(
                (G.iterations), dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.waveformvalues_halfdt = np.zeros(
                (G.iterations), dtype=config.sim_config.dtypes["float_or_double"]
            )

            for iteration in range(G.iterations):
                time = G.dt * iteration
                if time >= self.start and time <= self.stop:
                    # Set the time of the waveform evaluation to account for any
                    # delay in the start
                    time -= self.start
                    self.waveformvalues_wholedt[iteration] = waveform.calculate_value(time, G.dt)
                    self.waveformvalues_halfdt[iteration] = waveform.calculate_value(
                        time + 0.5 * G.dt, G.dt
                    )

    def calculate_incident_V_I(self, G):
        """Calculates the incident voltage and current with a long length
            transmission line not connected to the main grid
            from: http://dx.doi.org/10.1002/mop.10415

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        for iteration in range(self.iterations):
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
            self.resistance
            * (config.c * G.dt / self.dl)
            * (self.current[1 : self.nl] - self.current[0 : self.nl - 1])
        )

        # Update the voltage at the position of the one-way injector excitation
        self.voltage[self.srcpos] += (config.c * G.dt / self.dl) * self.waveformvalues_wholedt[
            iteration
        ]

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
            (1 / self.resistance)
            * (config.c * G.dt / self.dl)
            * self.waveformvalues_halfdt[iteration]
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

    # TODO: Add type information (if can avoid circular dependency)
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
                self.current[self.antpos] = G.calculate_Ix(i, j, k)

            elif self.polarisation == "y":
                self.current[self.antpos] = G.calculate_Iy(i, j, k)

            elif self.polarisation == "z":
                self.current[self.antpos] = G.calculate_Iz(i, j, k)

            self.update_current(iteration, G)


class DiscretePlaneWave(Source):
    """Implements the discrete plane wave (DPW) formulation as described in
    Tan, T.; Potter, M. (2010). FDTD Discrete Planewave (FDTD-DPW)
    Formulation for a Perfectly Matched Source in TFSF Simulations., 58(8),
    0–2648. doi:10.1109/tap.2010.2050446

    Implements a PML terninated 1D DPW FDTD grid which is used to source
    a plane wave into a 2D or 3D FDTD grid using the total-field/scattered-field
    (TFSF) formulation.

    Origin of the DPW can be any cornere of the FDTD grid and the
    propagation direction is defined by two angles, phi and theta. The DPW
    is defined by three integers, m_x, m_y, m_z which determine the rational
    angles corresponding to the propagation direction. 
    
    """

    def __init__(self, G):
        """    
        Args:
            m: int array stores the integer mappings, m_x, m_y, m_z which
                determine the rational angles last element stores
                max(m_x, m_y, m_z).
            directions: int array stores the directions of propagation of DPW.
            dimensions: int stores the number of dimensions in which the
                        simulation is run (2D or 3D).
            time_dimension: int stores the time length over which the simulation
                            is run.
            E_fields: double array stores the electric flieds associated with
                        1D DPW.
            H_fields: double array stores the magnetic fields associated with
                        1D DPW.
            G: FDTDGrid class describing a grid in a model.
        """

        super().__init__()
        self.m = np.zeros(3 + 1, dtype=np.int32)  # +1 to store the max(m_x, m_y, m_z)
        self.origin = np.zeros(3, dtype=np.int32)
        self.origin[0] = 0
        self.origin[1] = 0
        self.origin[2] = 0
        self.length = 0
        #self.projections = np.zeros(6, dtype=config.sim_config.dtypes["float_or_double"])
        self.projections = np.zeros(6, dtype=np.float64)  # Use float64 for better precision in projections
        self.corners = None
        self.materialID = 1
        self.ds = 0
        self.speed = config.c
        self.axial = 0
        self.pml_cells = 20
        self.psi= 0.0
        self.phi= 0.0
        self.theta = 0.0
        self.max_angle_diff = 0.0
        self.actual_angles = np.zeros(2, dtype=np.float64)  # [theta, phi]
        self.angle_errors = np.zeros(2, dtype=np.float64)  # [Delta_theta, Delta_phi]
        self.total_error = 0.0
  
    def initializeDiscretePlaneWave(self, G):
        """Creates a DPW, assigns memory to the grids, and gets field values
            at different time and space indices.

        Args:
            psi: float for polarization angle of the incident plane wave.
            phi: float for azimuthal angle (radians) of the incident plane wave.
            Delta_phi: float for permissible error in the rational angle
                        (radians) approximation to phi.
            theta: float for polar angle (radians) of the incident plane wave.
            Delta_theta: float for permissible error in the rational angle
                            (radians) approximation to theta.
            G: FDTDGrid class describing a grid in a model.
            number: int for number of cells in the 3D FDTD simulation.
            dx: double for separation between adjacent cells in the x direction.
            dy: double for separation between adjacent cells in the y direction.
            dz: double for separation between adjacent cells in the z direction.
            dt: double for time step for the FDTD simulation.

        Returns:
            E_fields: double array for electric field for the DPW as it evolves
                        over time and space indices.
            H_fields: double array for magnetic field for the DPW as it evolves
                        over time and space indices.
            C: double array stores coefficients of the fields for the update
                equation of the electric fields.
            D: double array stores coefficients of the fields for the update
                equation of the magnetic fields.
        """

        # check for plane wave definition using angles and in this case m vector should be zero and needs to be calculated
        if self.m[0] == 0 and self.m[1] == 0 and self.m[2] == 0:
            # Find the integer mappings m_x, m_y, m_z for the DPW using partial fractions
            self.m[:3], self.actual_angles, self.angle_errors, self.total_error = self.find_dpw_integers_optimized(self.theta, self.phi, [G.dx, G.dy, G.dz], self.max_angle_diff)

        # check for axial propagation case where the user wants a plane wave normally incident using grid geometry assuming layered model at best.
        elif self.axial != 0:
            self.actual_angles[0] = self.theta
            self.actual_angles[1] = self.phi
            self.angle_errors[0] = 0.0
            self.angle_errors[1] = 0.0
            self.total_error = 0.0

        # check for plane wave definition using m vector and in this case angles should be zero and needs to be calculated. There is no error in this case. 
        else:
            self.actual_angles[0] = math.degrees(math.atan2(self.m[1] * G.dx, self.m[0] * G.dy))
            self.actual_angles[1] = math.degrees(math.acos(self.m[2] * G.dz / (math.sqrt((self.m[0] * G.dy) ** 2 + (self.m[1] * G.dx) ** 2 + (self.m[2] * G.dz) ** 2))))
            self.angle_errors[0] = 0.0
            self.angle_errors[1] = 0.0            
            self.total_error = 0.0

        # Get angles in radians
        self.phi_est_rad = math.radians(self.actual_angles[1])
        self.theta_est_rad = math.radians(self.actual_angles[0])
        self.psi_rad = math.radians(self.psi)

        # Calculate the direction cosines
        px = math.sin(self.theta_est_rad)*math.cos(self.phi_est_rad)
        py = math.sin(self.theta_est_rad)*math.sin(self.phi_est_rad)
        pz = math.cos(self.theta_est_rad)
        
        #Maximum of the absolute values of m_x, m_y, m_z
        self.max_m = np.max(np.abs(self.m[:3]))

        # Store the absolute value of max(m_x, m_y, m_z) in the last element of the array
        self.m[3] = self.max_m
  
        if self.m[0] < 0:
            self.origin[0] = G.nx +1
        if self.m[1] < 0:
            self.origin[1] = G.ny +1
        if self.m[2] < 0:
            self.origin[2] = G.nz +1
        
        # Calculate ds that is needed for sourcing the 1D array. This is the spatial step of the 1D DPW grid. 
        # For axial propagation this is simply the grid step in the direction of propagation. 
        # For non-axial propagation this is calculated from the grid steps and the m vector.
        if self.m[0] == 0:  
            if self.m[1] == 0:
                if self.m[2] == 0:
                    raise ValueError("DPW should not be here as not all m_i values can be zero")
                else:
                    self.ds = pz * G.dz / self.m[2]
            else:
                self.ds = py * G.dy / self.m[1]
        else:
            self.ds = px * G.dx / self.m[0]


        # get the number of 1D DPW grid PML cells from the number of 3D FDTD PML cells used for terminating the 1D grid. This is set to 20 cells by default.
        self.pml_length = np.abs(self.m[0]) * self.pml_cells + np.abs(self.m[1]) * self.pml_cells + np.abs(self.m[2]) * self.pml_cells      
        # Set few buffer FDTD cells as extra 
        self.buffercells = np.abs(self.m[0]) * self.max_m + np.abs(self.m[1]) * self.max_m + np.abs(self.m[2]) * self.max_m
        # Total length of the 1D grid
        self.length = np.abs(self.m[0]) * (G.nx + 1) + np.abs(self.m[1]) * (G.ny + 1) + np.abs(self.m[2]) * (G.nz + 1) + self.pml_length + self.buffercells
        
        #self.length = 8000  # For testing purposes, limit length to 8000 cells


        # Setup an DPW grid ID array for accessing material IDs of the main grid for axial propagation problmes only
        # Allocate memory for the 1D fields
        self.E_fields = np.zeros(
            (3, self.length),
            order="C",
            dtype=config.sim_config.dtypes["float_or_double"],
        )
        self.H_fields = np.zeros(
            (3, self.length),
            order="C",
            dtype=config.sim_config.dtypes["float_or_double"],
        )

        # Allocate memory for the 1D DPW PML integrals
        # Izjxy means correcting an E_z field due to a H_x field variation in y derivative direction array position 0
        # Izjyx means correcting an E_z field due to a H_y field variation in x derivative direction array position 1
        # Izmxy means correcting an H_z field due to an E_x field variation in y derivative direction array position 2
        # Izmyx means correcting an H_z field due to an E_y field variation in x derivative direction array position 3 
        self.Iz = np.zeros((4,self.pml_length), order="C", dtype=config.sim_config.dtypes["float_or_double"]);      
        
        # Iyjxz means correcting an E_y field due to a H_x field variation in z derivative direction array position 0
        # Iyjzx means correcting an E_y field due to a H_z field variation in x derivative direction array position 1
        # Iymxz means correcting an H_y field due to an E_x field variation in z derivative direction array position 2
        # Iymzx means correcting an H_y field due to an E_z field variation in x derivative direction array position 3
        self.Iy = np.zeros((4,self.pml_length), order="C", dtype=config.sim_config.dtypes["float_or_double"]);

        
        # Ixjyz means correcting an E_x field due to a H_y field variation in z derivative direction array position 0
        # Ixjzy means correcting an E_x field due to a H_z field variation in y derivative direction array position 1
        # Ixmyz means correcting an H_x field due to an E_y field variation in z derivative direction array position 2
        # Ixmzy means correcting an H_x field due to an E_z field variation in y derivative direction array position 3
        self.Ix = np.zeros((4,self.pml_length), order="C", dtype=config.sim_config.dtypes["float_or_double"]);


       # When no grid IDs are used Get the background material object with the matching ID and add it to the PlaneWave object
        if self.axial == 0:
            self.material = next((x for x in G.materials if x.ID == self.materialID), None)
        
            
            # Find if the source material is dispersive
            if self.material.type == "debye":
                self.dispersive = True
                self.materialZ = math.sqrt(config.m0 * self.material.mr / (config.e0 * np.real(self.material.calculate_er(self.waveform.freq))))  # Impedance in the material
                self.speed = config.c / math.sqrt(np.real(self.material.calculate_er(self.waveform.freq)) * self.material.mr)  # Speed in the material
                self.max_poles = self.material.poles
                # Allocate memory for the polarization terms for dispersive materials
                self.Px = np.zeros((self.max_poles,self.length), order="C", dtype=config.sim_config.dtypes["float_or_double"])
                self.Py = np.zeros((self.max_poles,self.length), order="C", dtype=config.sim_config.dtypes["float_or_double"])
                self.Pz = np.zeros((self.max_poles,self.length), order="C", dtype=config.sim_config.dtypes["float_or_double"])
              
            else:
                self.dispersive = False
                self.materialZ = math.sqrt(config.m0 * self.material.mr / (config.e0 * self.material.er)) # Impedance in the material
                self.speed = config.c / math.sqrt(self.material.er * self.material.mr)  # Speed in the material
             
            

            # Calculate the projections for sourcing the electric and magnetic fields
            # using double precision for better accuracy

            self.projections[0]=math.cos(self.psi_rad)*math.sin(self.phi_est_rad)-math.sin(self.psi_rad)*math.cos(self.theta_est_rad)*math.cos(self.phi_est_rad) 
            if abs(self.projections[0]) <= 1e-15:
                self.projections[0] = 0

            self.projections[1]=-math.cos(self.psi_rad)*math.cos(self.phi_est_rad)-math.sin(self.psi_rad)*math.cos(self.theta_est_rad)*math.sin(self.phi_est_rad)
            if abs(self.projections[1]) <= 1e-15:
                self.projections[1] =0 

            self.projections[2]=math.sin(self.psi_rad)*math.sin(self.theta_est_rad)
            if abs(self.projections[2]) <= 1e-15:
                self.projections[2] = 0    

            self.projections[3]=(math.sin(self.psi_rad)*math.sin(self.phi_est_rad)+math.cos(self.psi_rad)*math.cos(self.theta_est_rad)*math.cos(self.phi_est_rad))/self.materialZ
            if abs(self.projections[3]) <= 1e-15:
                self.projections[3] = 0

            self.projections[4]=(-math.sin(self.psi_rad)*math.cos(self.phi_est_rad)+math.cos(self.psi_rad)*math.cos(self.theta_est_rad)*math.sin(self.phi_est_rad))/self.materialZ
            if abs(self.projections[4]) <= 1e-15:
                self.projections[4] = 0    

            self.projections[5]=(-math.cos(self.psi_rad)*math.sin(self.theta_est_rad))/self.materialZ
            if abs(self.projections[5]) <= 1e-15:
                self.projections[5] = 0    



        # Get the waveform object with the matching ID and add it to the PlaneWave object
        self.waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
        
        if self.axial == 0:
            self._get_pml_parameters(G, self.materialZ)
         

    def grid_init(self,G):
        # Initialize the ID array for axial propagation problems only
        if self.axial != 0:
            self.ID = np.zeros((6,self.length), dtype=np.uint32)  # 6 for the 6 field components

            if self.axial == 1:  # x direction
                for idx in range(G.nx):
                    self.ID[0,idx]=G.ID[0,idx,1,1]
                    self.ID[1,idx]=G.ID[1,idx,1,1]
                    self.ID[2,idx]=G.ID[2,idx,1,1]
                    self.ID[3,idx]=G.ID[3,idx,1,1]
                    self.ID[4,idx]=G.ID[4,idx,1,1]
                    self.ID[5,idx]=G.ID[5,idx,1,1]

                for idx in range(G.nx,self.length):
                    self.ID[0,idx]=G.ID[0,G.nx-1,1,1] 
                    self.ID[1,idx]=G.ID[1,G.nx-1,1,1]
                    self.ID[2,idx]=G.ID[2,G.nx-1,1,1]
                    self.ID[3,idx]=G.ID[3,G.nx-1,1,1]
                    self.ID[4,idx]=G.ID[4,G.nx-1,1,1]
                    self.ID[5,idx]=G.ID[5,G.nx-1,1,1]

            if self.axial == 2:  # y direction
                for idx in range(G.ny):
                    self.ID[0,idx]=G.ID[0,1,idx,1]
                    self.ID[1,idx]=G.ID[1,1,idx,1]
                    self.ID[2,idx]=G.ID[2,1,idx,1]
                    self.ID[3,idx]=G.ID[3,1,idx,1]
                    self.ID[4,idx]=G.ID[4,1,idx,1]
                    self.ID[5,idx]=G.ID[5,1,idx,1]

                for idx in range(G.ny,self.length):
                    self.ID[0,idx]=G.ID[0,1,G.ny-1,1]
                    self.ID[1,idx]=G.ID[1,1,G.ny-1,1]
                    self.ID[2,idx]=G.ID[2,1,G.ny-1,1]
                    self.ID[3,idx]=G.ID[3,1,G.ny-1,1]
                    self.ID[4,idx]=G.ID[4,1,G.ny-1,1]
                    self.ID[5,idx]=G.ID[5,1,G.ny-1,1]

            if self.axial == 3:  # z direction  
                for idx in range(G.nz):
                    self.ID[0,idx]=G.ID[0,1,1,idx]
                    self.ID[1,idx]=G.ID[1,1,1,idx]
                    self.ID[2,idx]=G.ID[2,1,1,idx]
                    self.ID[3,idx]=G.ID[3,1,1,idx]
                    self.ID[4,idx]=G.ID[4,1,1,idx]
                    self.ID[5,idx]=G.ID[5,1,1,idx]

                for idx in range(G.nz,self.length):
                    self.ID[0,idx]=G.ID[0,1,1,G.nz-1]
                    self.ID[1,idx]=G.ID[1,1,1,G.nz-1]
                    self.ID[2,idx]=G.ID[2,1,1,G.nz-1]
                    self.ID[3,idx]=G.ID[3,1,1,G.nz-1]
                    self.ID[4,idx]=G.ID[4,1,1,G.nz-1]
                    self.ID[5,idx]=G.ID[5,1,1,G.nz-1]   

            
            # Get material by numeric ID of the right cell next to the origin. This material is used for calculating the speed and impedance of the DPW
            self.material = next((x for x in G.materials if x.numID == G.ID[0,2, 2, 2]), None)
            # Get material by numeric ID of the PML cell next to the last cell of the main grid. This material is used for calculating the PML parameters
            if self.axial == 1:
                self.materialPML = next((x for x in G.materials if x.numID == G.ID[0,G.nx-2,2,2]), None)
            elif self.axial == 2:
                self.materialPML = next((x for x in G.materials if x.numID == G.ID[0,2,G.ny-2,2]), None)
            elif self.axial == 3:
                self.materialPML = next((x for x in G.materials if x.numID == G.ID[0,2,2,G.nz-2]), None)    


            # Find if the source material is dispersive
            if self.material.type == "debye":
                self.materialZ = math.sqrt(config.m0 * self.material.mr / (config.e0 * np.real(self.material.calculate_er(self.waveform.freq))))  # Impedance in the material
                self.speed = config.c / math.sqrt(np.real(self.material.calculate_er(self.waveform.freq)) * self.material.mr)  # Speed in the material
               
            else:
                self.dispersive = False
                self.materialZ = math.sqrt(config.m0 * self.material.mr / (config.e0 * self.material.er)) # Impedance in the material
                self.speed = config.c / math.sqrt(self.material.er * self.material.mr)  # Speed in the material
               

            # Find if the PML material is dispersive
            if self.materialPML.type == "debye":
                self.materialPMLZ = math.sqrt(config.m0 * self.materialPML.mr / (config.e0 * np.real(self.materialPML.calculate_er(self.waveform.freq))))  # Impedance in the material
                self.PMLspeed = config.c / math.sqrt(np.real(self.materialPML.calculate_er(self.waveform.freq)) * self.materialPML.mr)  # Speed in the material

            else:
                self.dispersive = False
                self.materialPMLZ = math.sqrt(config.m0 * self.materialPML.mr / (config.e0 * self.materialPML.er)) # Impedance in the material
                self.PMLspeed = config.c / math.sqrt(self.materialPML.er * self.materialPML.mr)  # Speed in the material
               

            #find if the grid has any debye materials which can determine if we need dispersive updates even if the source material is non-dispersive
            has_debye = any(getattr(x, "type", None) == "debye" for x in G.materials)

            # If axial propagation and has any debye materials in the grid then need to use dispersive updates for all materials in the main grid        
            if has_debye:
                self.dispersive = True
                print("Axial DPW propagation: will use dispersive updates for all materials in the main grid")
                self.max_poles = 0
                for material in G.materials:
                    poles = getattr(material, "poles", 0)  # Default to 0 if 'poles' doesn't exist
                    if poles is not None:
                        self.max_poles = max(self.max_poles, poles)

            # Allocate memory for the polarization arrays for the DPW updates if dispersive materials are present in the grid
            if self.dispersive:
                self.Px = np.zeros((self.max_poles,self.length), order="C", dtype=config.sim_config.dtypes["float_or_double"])
                self.Py = np.zeros((self.max_poles,self.length), order="C", dtype=config.sim_config.dtypes["float_or_double"])
                self.Pz = np.zeros((self.max_poles,self.length), order="C", dtype=config.sim_config.dtypes["float_or_double"])

        
            # Calculate the projections for sourcing the electric and magnetic fields
            # using double precision for better accuracy

            self.projections[0]=math.cos(self.psi_rad)*math.sin(self.phi_est_rad)-math.sin(self.psi_rad)*math.cos(self.theta_est_rad)*math.cos(self.phi_est_rad) 
            if abs(self.projections[0]) <= 1e-15:
                self.projections[0] = 0

            self.projections[1]=-math.cos(self.psi_rad)*math.cos(self.phi_est_rad)-math.sin(self.psi_rad)*math.cos(self.theta_est_rad)*math.sin(self.phi_est_rad)
            if abs(self.projections[1]) <= 1e-15:
                self.projections[1] =0 

            self.projections[2]=math.sin(self.psi_rad)*math.sin(self.theta_est_rad)
            if abs(self.projections[2]) <= 1e-15:
                self.projections[2] = 0    

            self.projections[3]=(math.sin(self.psi_rad)*math.sin(self.phi_est_rad)+math.cos(self.psi_rad)*math.cos(self.theta_est_rad)*math.cos(self.phi_est_rad))/self.materialZ
            if abs(self.projections[3]) <= 1e-15:
                self.projections[3] = 0

            self.projections[4]=(-math.sin(self.psi_rad)*math.cos(self.phi_est_rad)+math.cos(self.psi_rad)*math.cos(self.theta_est_rad)*math.sin(self.phi_est_rad))/self.materialZ
            if abs(self.projections[4]) <= 1e-15:
                self.projections[4] = 0    

            self.projections[5]=(-math.cos(self.psi_rad)*math.sin(self.theta_est_rad))/self.materialZ
            if abs(self.projections[5]) <= 1e-15:
                self.projections[5] = 0    

            self._get_pml_parameters(G, self.materialPMLZ)

           
            print(f"Discrete Plane Wave has been initialized "
            + f"with field projections (Ex, Ey, Ez, Hx, Hy, Hz) = ({self.projections[0]:.4f}, {self.projections[1]:.4f}, {self.projections[2]:.4f}, {self.projections[3]:.4f}, {self.projections[4]:.4f}, {self.projections[5]:.4f})"
            + f" , grid origin = ({self.origin[0]}, {self.origin[1]}, {self.origin[2]})"
            + f" and 1D vector length = {self.length} cells.")

        else:
            pass


    def calculate_waveform_values(self, G, cythonize=True):
        """Calculates all waveform values for source for duration of simulation.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        # Waveform values for sources that need to be calculated on whole timesteps
        self.waveformvalues_wholedt = np.zeros(
            (G.iterations, 3, self.m[3]),
            dtype=config.sim_config.dtypes["float_or_double"],
        )

        # Waveform values for sources that need to be calculated on half timesteps
        self.waveformvalues_halfdt = np.zeros(
            (G.iterations, 3, self.m[3]),
            dtype=config.sim_config.dtypes["float_or_double"],
        )

        #waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
        if cythonize:
            calculate1DWaveformValues(
                self.waveformvalues_wholedt,
                self.waveformvalues_halfdt,
                G.iterations,
                self.m,
                G.dt,
                self.ds,
                self.speed,
                self.start,
                self.stop,
                self.waveform.freq,
                self.waveform.type.encode("UTF-8"),
            )
        else:
            for dimension in range(3):
                for iteration in range(G.iterations):
                    for r in range(self.m[3]):
                        time1 = (
                            G.dt * iteration
                            - (
                                r
                                + (np.abs(self.m[(dimension + 1) % 3]) + np.abs(self.m[(dimension + 2) % 3])) * 0.5
                            )
                            * self.ds
                            / self.speed
                        )
                        if time1 >= self.start and time1 <= self.stop:
                            # Set the time of the waveform evaluation to account for any
                            # delay in the start
                            time1 -= self.start
                            self.waveformvalues_wholedt[
                                iteration, dimension, r
                            ] = self.waveform.calculate_value(time1, G.dt)

                    for r in range(self.m[3]):
                        time2 = (
                            G.dt * (iteration + 0.5)
                            - (
                                r
                                + (np.abs(self.m[(dimension)]) + np.abs(self.m[(dimension)])) * 0.5
                            )
                            * self.ds
                            / self.speed
                        )
                        if time2 >= self.start and time2 <= self.stop:
                            # Set the time of the waveform evaluation to account for any
                            # delay in the start
                            time2 -= self.start
                            self.waveformvalues_halfdt[
                                iteration, dimension, r
                            ] = self.waveform.calculate_value(time2, G.dt)


    def update_plane_wave_magnetic(
        self,
        nthreads,
        updatecoeffsE,
        updatecoeffsH,
        Ex,
        Ey,
        Ez,
        Hx,
        Hy,
        Hz,
        iteration,
        G,
        cythonize=True,
        precompute=True
    ):
        if self.axial != 0:
            
            updatePlaneWave_magnetic_axial(
                self.length,
                self.pml_length,
                nthreads,
                self.H_fields,
                self.E_fields,
                self.Ix,
                self.Iy,
                self.Iz,    
                updatecoeffsE[:, :],
                updatecoeffsH[:, :],
                self.ID,
                G.ID,
                self.pml_rex,
                self.pml_rey,
                self.pml_rez,
                self.pml_rhx,
                self.pml_rhy,
                self.pml_rhz,   
                Ex,
                Ey,
                Ez,
                Hx,
                Hy,
                Hz,
                self.projections,
                self.waveformvalues_wholedt[:, :, :],
                self.waveformvalues_halfdt[:, :, :],
                self.m,
                self.origin,
                self.corners,
                precompute,
                iteration,
                G.dt,
                G.dx,
                G.dy,
                G.dz,
                self.ds,
                self.speed,
                self.start,
                self.stop,
                self.waveform.freq,
                self.waveform.type.encode("UTF-8")
            )
            
        else:        

            if cythonize:
            
                updatePlaneWave_magnetic(
                    self.length,
                    self.pml_length,
                    nthreads,
                    self.H_fields,
                    self.E_fields,
                    self.Ix,
                    self.Iy,
                    self.Iz,    
                    updatecoeffsE[self.material.numID, :],
                    updatecoeffsH[self.material.numID, :],
                    self.pml_rex,
                    self.pml_rey,
                    self.pml_rez,
                    self.pml_rhx,
                    self.pml_rhy,
                    self.pml_rhz,   
                    Ex,
                    Ey,
                    Ez,
                    Hx,
                    Hy,
                    Hz,
                    self.projections,
                    self.waveformvalues_wholedt[:, :, :],
                    self.waveformvalues_halfdt[:, :, :],
                    self.m,
                    self.origin,
                    self.corners,
                    precompute,
                    iteration,
                    G.dt,
                    G.dx,
                    G.dy,
                    G.dz,
                    self.ds,
                    self.speed,
                    self.start,
                    self.stop,
                    self.waveform.freq,
                    self.waveform.type.encode("UTF-8"),
                )
            else:
                    self.update_magnetic_field_1D(G, iteration, precompute)
                    self.apply_TFSF_conditions_magnetic(G)
            

    def update_plane_wave_electric(
        self,
        nthreads,
        updatecoeffsE,
        updatecoeffsH,
        Ex,
        Ey,
        Ez,
        Hx,
        Hy,
        Hz,
        iteration,
        G,
        cythonize=True,
        precompute=True
    ):
        
        if self.axial != 0:
            updatePlaneWave_electric_axial(
                self.length,
                self.pml_length,
                nthreads,
                self.H_fields,
                self.E_fields,
                self.Ix,
                self.Iy,
                self.Iz,    
                updatecoeffsE[:, :],
                updatecoeffsH[:, :],
                self.ID,
                G.ID,
                self.pml_rex,
                self.pml_rey,
                self.pml_rez,
                self.pml_rhx,
                self.pml_rhy,
                self.pml_rhz,   
                Ex,
                Ey,
                Ez,
                Hx,
                Hy,
                Hz,
                self.projections,
                self.waveformvalues_wholedt[:, :, :],
                self.waveformvalues_halfdt[:, :, :],
                self.m,
                self.origin,
                self.corners,
                precompute,
                iteration,
                G.dt,
                G.dx,
                G.dy,
                G.dz,
                self.ds,
                self.speed,
                self.start,
                self.stop,
                self.waveform.freq,
                self.waveform.type.encode("UTF-8"),
            )

        else:

            if cythonize:
                updatePlaneWave_electric(
                    self.length,
                    self.pml_length,
                    nthreads,
                    self.H_fields,
                    self.E_fields,
                    self.Ix,
                    self.Iy,
                    self.Iz,    
                    updatecoeffsE[self.material.numID, :],
                    updatecoeffsH[self.material.numID, :],
                    self.pml_rex,
                    self.pml_rey,
                    self.pml_rez,
                    self.pml_rhx,
                    self.pml_rhy,
                    self.pml_rhz,   
                    Ex,
                    Ey,
                    Ez,
                    Hx,
                    Hy,
                    Hz,
                    self.projections,
                    self.waveformvalues_wholedt[:, :, :],
                    self.waveformvalues_halfdt[:, :, :],
                    self.m,
                    self.origin,
                    self.corners,
                    precompute,
                    iteration,
                    G.dt,
                    G.dx,
                    G.dy,
                    G.dz,
                    self.ds,
                    self.speed,
                    self.start,
                    self.stop,
                    self.waveform.freq,
                    self.waveform.type.encode("UTF-8"),
                )
            else:
                self.update_electric_field_1D(G, iteration, precompute)
                self.apply_TFSF_conditions_electric(G)
            

    def update_plane_wave_electric_dispersive(
        self,
        nthreads,
        updatecoeffsE,
        updatecoeffsH,
        updatecoeffsdispersive,
        Ex,
        Ey,
        Ez,
        Hx,
        Hy,
        Hz,
        iteration,
        G,
        cythonize=True,
        precompute=True
    ):
        if self.axial != 0:
            updatePlaneWave_electric_dispersive_axial(
                self.length,
                self.pml_length,
                nthreads,
                self.H_fields,
                self.E_fields,
                self.Px,
                self.Py,
                self.Pz,
                self.Ix,
                self.Iy,
                self.Iz,    
                updatecoeffsE[:, :],
                updatecoeffsH[:, :],
                updatecoeffsdispersive[:, :],
                self.ID,
                G.ID,
                self.max_poles,
                self.pml_rex,
                self.pml_rey,
                self.pml_rez,
                self.pml_rhx,
                self.pml_rhy,
                self.pml_rhz,   
                Ex,
                Ey,
                Ez,
                Hx,
                Hy,
                Hz,
                self.projections,
                self.waveformvalues_wholedt[:, :, :],
                self.waveformvalues_halfdt[:, :, :],
                self.m,
                self.origin,
                self.corners,
                precompute,
                iteration,
                G.dt,
                G.dx,
                G.dy,
                G.dz,
                self.ds,
                self.speed,
                self.start,
                self.stop,
                self.waveform.freq,
                self.waveform.type.encode("UTF-8"),
            )

        else:

            if cythonize:
                updatePlaneWave_electric_dispersive(
                    self.length,
                    self.pml_length,
                    nthreads,
                    self.H_fields,
                    self.E_fields,
                    self.Px,
                    self.Py,
                    self.Pz,
                    self.Ix,
                    self.Iy,
                    self.Iz,    
                    updatecoeffsE[self.material.numID, :],
                    updatecoeffsH[self.material.numID, :],
                    updatecoeffsdispersive[self.material.numID, :],
                    self.max_poles,
                    self.pml_rex,
                    self.pml_rey,
                    self.pml_rez,
                    self.pml_rhx,
                    self.pml_rhy,
                    self.pml_rhz,   
                    Ex,
                    Ey,
                    Ez,
                    Hx,
                    Hy,
                    Hz,
                    self.projections,
                    self.waveformvalues_wholedt[:, :, :],
                    self.waveformvalues_halfdt[:, :, :],
                    self.m,
                    self.origin,
                    self.corners,
                    precompute,
                    iteration,
                    G.dt,
                    G.dx,
                    G.dy,
                    G.dz,
                    self.ds,
                    self.speed,
                    self.start,
                    self.stop,
                    self.waveform.freq,
                    self.waveform.type.encode("UTF-8"),
                )
            else:
                raise NotImplementedError("Cythonized version not available")
               

    def initialize_magnetic_fields_1D(self, G, iteration, precompute):
        if precompute:
            for dimension in range(3):
                for r in range(self.m[3]):
                    # Assign source values of magnetic field to first few gridpoints
                    self.H_fields[dimension, r] = (
                        self.projections[dimension]
                        * self.waveformvalues_wholedt[iteration, dimension, r]
                    )
                    # self.getSource(self.real_time - (j+(self.m[(i+1)%3]+self.m[(i+2)%3])*0.5)*self.ds/config.c)#, self.waveformID, G.dt)
        else:
            for dimension in range(3):
                for r in range(self.m[3]):
                    # Assign source values of magnetic field to first few gridpoints
                    self.H_fields[dimension, r] = self.projections[dimension] * getSource(
                        iteration * G.dt
                        - (r + (self.m[(dimension + 1) % 3] + self.m[(dimension + 2) % 3]) * 0.5)
                        * self.ds
                        / self.speed,
                        self.waveform.freq,
                        self.waveform.type.encode("UTF-8"),
                        G.dt,
                    )

    def initialize_electric_fields_1D(self, G, iteration, precompute):
        if precompute:
            for dimension in range(3):
                for r in range(self.m[3]):
                    # Assign source values of magnetic field to first few gridpoints
                    self.E_fields[dimension, r] = (
                        self.projections[dimension]
                        * self.waveformvalues_halfdt[iteration, dimension, r]
                    )
                    # self.getSource(self.real_time - (j+(self.m[(i+1)%3]+self.m[(i+2)%3])*0.5)*self.ds/config.c)#, self.waveformID, G.dt)
        else:
            for dimension in range(3):
                for r in range(self.m[3]):
                    # Assign source values of magnetic field to first few gridpoints
                    self.E_fields[dimension, r] = self.projections[dimension] * getSource(
                        (iteration + 0.5) * G.dt
                        - (r + np.abs(self.m[dimension]) * 0.5)
                        * self.ds
                        / self.speed,
                        self.waveform.freq,
                        self.waveform.type.encode("UTF-8"),
                        G.dt,
                    )
    

    def update_magnetic_field_1D(self, G, iteration, precompute=True):
        """Updates magnetic fields for the next time step using Equation 8 of
            DOI: 10.1109/LAWP.2009.2016851

        Args:
            n: int stores spatial length of the DPW array so that each length
                grid cell is updated when updateMagneticFields() called.
            H_coefficients: double array stores coefficients of the fields in
                            the update equation for the magnetic field.
            H_fields: double array stores magnetic fields of the DPW until
                        temporal index time.
            E_fields: double array stores electric fields of the DPW until
                        temporal index time.
            time: int time index storing current axis number which would be
                    updated for the H_fields.

        Returns:
            H_fields: double array for magnetic field with the axis entry for
                        the current time added.
        """

        self.initialize_magnetic_fields_1D(G, iteration, precompute)

        for i in range(3):  # Update each component of magnetic field
            materialH = G.ID[
                3 + i,
                (self.corners[0] + self.corners[3]) // 2,
                (self.corners[1] + self.corners[4]) // 2,
                (self.corners[2] + self.corners[5]) // 2,
            ]
            # Update magnetic field at each spatial index
            for j in range(self.m[-1], self.length - self.m[-1]):
                self.H_fields[i, j] = (
                    G.updatecoeffsH[materialH, 0] * self.H_fields[i, j]
                    + G.updatecoeffsH[materialH, (i + 2) % 3 + 1]
                    * (
                        self.E_fields[(i + 1) % 3, j + self.m[(i + 2) % 3]]
                        - self.E_fields[(i + 1) % 3, j]
                    )
                    - G.updatecoeffsH[materialH, (i + 1) % 3 + 1]
                    * (
                        self.E_fields[(i + 2) % 3, j + self.m[(i + 1) % 3]]
                        - self.E_fields[(i + 2) % 3, j]
                    )
                )  # equation 8 of Tan, Potter paper

    def update_electric_field_1D(self, G, iteration, precompute=True):
        """Updates electric fields for the next time step using Equation 9 of
            DOI: 10.1109/LAWP.2009.2016851

        Args:
            n: int stores spatial length of DPW array so that each length grid
                cell is updated when updateMagneticFields() is called.
            E_coefficients: double array stores coefficients of the fields in
                            the update equation for the electric field.
            H_fields: double array stores magnetic fields of the DPW until
                        temporal index time.
            E_fields: double array stores electric fields of the DPW until
                        temporal index time.
            time: int time index storing current axis number which would be
                    updated for the E_fields.

        Returns:
            E_fields: double array for electric field with the axis entry for
                        the current time added.

        """
        self.initialize_electric_fields_1D(G, iteration, precompute)
       

        for i in range(3):  # Update each component of electric field
            materialE = G.ID[
                i,
                (self.corners[0] + self.corners[3]) // 2,
                (self.corners[1] + self.corners[4]) // 2,
                (self.corners[2] + self.corners[5]) // 2,
            ]
            # Update electric field at each spatial index
            for j in range(self.m[-1], self.length - self.m[-1]):
                self.E_fields[i, j] = (
                    G.updatecoeffsE[materialE, 0] * self.E_fields[i, j]
                    + G.updatecoeffsE[materialE, (i + 2) % 3 + 1]
                    * (
                        self.H_fields[(i + 2) % 3, j]
                        - self.H_fields[(i + 2) % 3, j - self.m[(i + 1) % 3]]
                    )
                    - G.updatecoeffsE[materialE, (i + 1) % 3 + 1]
                    * (
                        self.H_fields[(i + 1) % 3, j]
                        - self.H_fields[(i + 1) % 3, j - self.m[(i + 2) % 3]]
                    )
                )  # equation 9 of Tan, Potter paper

    def getField(self, i, j, k, array, m, origin, component):
        return array[component, np.dot(m[:-1], np.array([i-origin[0], j-origin[1], k-origin[2]]))]

    def apply_TFSF_conditions_magnetic(self, G):
        # **** constant x faces -- scattered-field nodes ****
        i = self.corners[0]
        for j in range(self.corners[1], self.corners[4] + 1):
            for k in range(self.corners[2], self.corners[5]):
                # correct Hy at firstX-1/2 by subtracting Ez_inc
                G.Hy[i - 1, j, k] -= G.updatecoeffsH[G.ID[4, i, j, k], 1] * self.getField(
                    i, j, k, self.E_fields, self.m, self.origin, 2
                )

        for j in range(self.corners[1], self.corners[4]):
            for k in range(self.corners[2], self.corners[5] + 1):
                # correct Hz at firstX-1/2 by adding Ey_inc
                G.Hz[i - 1, j, k] += G.updatecoeffsH[G.ID[5, i, j, k], 1] * self.getField(
                    i, j, k, self.E_fields, self.m, self.origin, 1
                )

        i = self.corners[3]
        for j in range(self.corners[1], self.corners[4] + 1):
            for k in range(self.corners[2], self.corners[5]):
                # correct Hy at lastX+1/2 by adding Ez_inc
                G.Hy[i, j, k] += G.updatecoeffsH[G.ID[4, i, j, k], 1] * self.getField(
                    i, j, k, self.E_fields, self.m, self.origin, 2
                )

        for j in range(self.corners[1], self.corners[4]):
            for k in range(self.corners[2], self.corners[5] + 1):
                # correct Hz at lastX+1/2 by subtractinging Ey_inc
                G.Hz[i, j, k] -= G.updatecoeffsH[G.ID[5, i, j, k], 1] * self.getField(
                    i, j, k, self.E_fields, self.m, self.origin, 1
                )

        # **** constant y faces -- scattered-field nodes ****
        j = self.corners[1]
        for i in range(self.corners[0], self.corners[3] + 1):
            for k in range(self.corners[2], self.corners[5]):
                # correct Hx at firstY-1/2 by adding Ez_inc
                G.Hx[i, j - 1, k] += G.updatecoeffsH[G.ID[3, i, j, k], 2] * self.getField(
                    i, j, k, self.E_fields, self.m, self.origin, 2
                )

        for i in range(self.corners[0], self.corners[3]):
            for k in range(self.corners[2], self.corners[5] + 1):
                # correct Hz at firstY-1/2 by subtracting Ex_inc
                G.Hz[i, j - 1, k] -= G.updatecoeffsH[G.ID[5, i, j, k], 2] * self.getField(
                    i, j, k, self.E_fields, self.m, self.origin, 0
                )

        j = self.corners[4]
        for i in range(self.corners[0], self.corners[3] + 1):
            for k in range(self.corners[2], self.corners[5]):
                # correct Hx at lastY+1/2 by subtracting Ez_inc
                G.Hx[i, j, k] -= G.updatecoeffsH[G.ID[3, i, j, k], 2] * self.getField(
                    i, j, k, self.E_fields, self.m, self.origin, 2
                )

        for i in range(self.corners[0], self.corners[3]):
            for k in range(self.corners[2], self.corners[5] + 1):
                # correct Hz at lastY-1/2 by adding Ex_inc
                G.Hz[i, j, k] += G.updatecoeffsH[G.ID[5, i, j, k], 2] * self.getField(
                    i, j, k, self.E_fields, self.m, self.origin, 0
                )

        # **** constant z faces -- scattered-field nodes ****
        k = self.corners[2]
        for i in range(self.corners[0], self.corners[3]):
            for j in range(self.corners[1], self.corners[4] + 1):
                # correct Hy at firstZ-1/2 by adding Ex_inc
                G.Hy[i, j, k - 1] += G.updatecoeffsH[G.ID[4, i, j, k], 3] * self.getField(
                    i, j, k, self.E_fields, self.m, self.origin, 0
                )

        for i in range(self.corners[0], self.corners[3] + 1):
            for j in range(self.corners[1], self.corners[4]):
                # correct Hx at firstZ-1/2 by subtracting Ey_inc
                G.Hx[i, j, k - 1] -= G.updatecoeffsH[G.ID[3, i, j, k], 3] * self.getField(
                    i, j, k, self.E_fields, self.m, self.origin, 1
                )

        k = self.corners[5]
        for i in range(self.corners[0], self.corners[3]):
            for j in range(self.corners[1], self.corners[4] + 1):
                # correct Hy at firstZ-1/2 by subtracting Ex_inc
                G.Hy[i, j, k] -= G.updatecoeffsH[G.ID[4, i, j, k], 3] * self.getField(
                    i, j, k, self.E_fields, self.m, self.origin, 0
                )

        for i in range(self.corners[0], self.corners[3] + 1):
            for j in range(self.corners[1], self.corners[4]):
                # correct Hx at lastZ+1/2 by adding Ey_inc
                G.Hx[i, j, k] += G.updatecoeffsH[G.ID[3, i, j, k], 3] * self.getField(
                    i, j, k, self.E_fields, self.m, self.origin, 1
                )

    def apply_TFSF_conditions_electric(self, G):
        # **** constant x faces -- total-field nodes ****/
        i = self.corners[0]
        for j in range(self.corners[1], self.corners[4] + 1):
            for k in range(self.corners[2], self.corners[5]):
                # correct Ez at firstX face by subtracting Hy_inc
                G.Ez[i, j, k] -= G.updatecoeffsE[G.ID[2, i, j, k], 1] * self.getField(
                    i - 1, j, k, self.H_fields, self.m, self.origin, 1
                )

        for j in range(self.corners[1], self.corners[4]):
            for k in range(self.corners[2], self.corners[5] + 1):
                # correct Ey at firstX face by adding Hz_inc
                G.Ey[i, j, k] += G.updatecoeffsE[G.ID[1, i, j, k], 1] * self.getField(
                    i - 1, j, k, self.H_fields, self.m, self.origin, 2
                )

        i = self.corners[3]
        for j in range(self.corners[1], self.corners[4] + 1):
            for k in range(self.corners[2], self.corners[5]):
                # correct Ez at lastX face by adding Hy_inc
                G.Ez[i, j, k] += G.updatecoeffsE[G.ID[2, i, j, k], 1] * self.getField(
                    i, j, k, self.H_fields, self.m, self.origin, 1
                )

        i = self.corners[3]
        for j in range(self.corners[1], self.corners[4]):
            for k in range(self.corners[2], self.corners[5] + 1):
                # correct Ey at lastX face by subtracting Hz_inc
                G.Ey[i, j, k] -= G.updatecoeffsE[G.ID[1, i, j, k], 1] * self.getField(
                    i, j, k, self.H_fields, self.m, self.origin, 2
                )

        # **** constant y faces -- total-field nodes ****/
        j = self.corners[1]
        for i in range(self.corners[0], self.corners[3] + 1):
            for k in range(self.corners[2], self.corners[5]):
                # correct Ez at firstY face by adding Hx_inc
                G.Ez[i, j, k] += G.updatecoeffsE[G.ID[2, i, j, k], 2] * self.getField(
                    i, j - 1, k, self.H_fields, self.m, self.origin, 0
                )

        for i in range(self.corners[0], self.corners[3]):
            for k in range(self.corners[2], self.corners[5] + 1):
                # correct Ex at firstY face by subtracting Hz_inc
                G.Ex[i, j, k] -= G.updatecoeffsE[G.ID[0, i, j, k], 2] * self.getField(
                    i, j - 1, k, self.H_fields, self.m, self.origin, 2
                )

        j = self.corners[4]
        for i in range(self.corners[0], self.corners[3] + 1):
            for k in range(self.corners[2], self.corners[5]):
                # correct Ez at lastY face by subtracting Hx_inc
                G.Ez[i, j, k] -= G.updatecoeffsE[G.ID[2, i, j, k], 2] * self.getField(
                    i, j, k, self.H_fields, self.m, self.origin, 0
                )

        for i in range(self.corners[0], self.corners[3]):
            for k in range(self.corners[2], self.corners[5] + 1):
                # correct Ex at lastY face by adding Hz_inc
                G.Ex[i, j, k] += G.updatecoeffsE[G.ID[0, i, j, k], 2] * self.getField(
                    i, j, k, self.H_fields, self.m, self.origin, 2
                )

        # **** constant z faces -- total-field nodes ****/
        k = self.corners[2]
        for i in range(self.corners[0], self.corners[3] + 1):
            for j in range(self.corners[1], self.corners[4]):
                # correct Ey at firstZ face by subtracting Hx_inc
                G.Ey[i, j, k] -= G.updatecoeffsE[G.ID[1, i, j, k], 3] * self.getField(
                    i, j, k - 1, self.H_fields, self.m, self.origin, 0
                )

        for i in range(self.corners[0], self.corners[3]):
            for j in range(self.corners[1], self.corners[4] + 1):
                # correct Ex at firstZ face by adding Hy_inc
                G.Ex[i, j, k] += G.updatecoeffsE[G.ID[0, i, j, k], 3] * self.getField(
                    i, j, k - 1, self.H_fields, self.m, self.origin, 1
                )

        k = self.corners[5]
        for i in range(self.corners[0], self.corners[3] + 1):
            for j in range(self.corners[1], self.corners[4]):
                # correct Ey at lastZ face by adding Hx_inc
                G.Ey[i, j, k] += G.updatecoeffsE[G.ID[1, i, j, k], 3] * self.getField(
                    i, j, k, self.H_fields, self.m, self.origin, 0
                )

        for i in range(self.corners[0], self.corners[3]):
            for j in range(self.corners[1], self.corners[4] + 1):
                # correct Ex at lastZ face by subtracting Hy_inc
                G.Ex[i, j, k] -= G.updatecoeffsE[G.ID[0, i, j, k], 3] * self.getField(
                    i, j, k, self.H_fields, self.m, self.origin, 1
                )


    def find_dpw_integers_optimized(self, theta_deg, phi_deg, delta_xyz, max_total_error_deg):
        """
           Finds the OPTIMAL smallest integer vector (mx, my, mz) for a DPW source
           by generating all candidates and selecting the simplest valid one.
            --- Parameters ---
           theta_deg : float
               Polar angle in degrees (0 to 180) from the +Z axis.
           phi_deg : float
               Azimuthal angle in degrees (0 to 360) from the +X axis.
           delta_xyz : list or tuple
               Grid step sizes [dx, dy, dz] in your simulation units.
           max_total_error_deg : float
               Maximum acceptable TOTAL 3D angular error in degrees.

           --- Returns ---
           m_vec : numpy.ndarray
               The optimal 1x3 integer vector [mx, my, mz]. Returns None if no solution is found.
           actual_angles_deg : tuple
               The actual (theta, phi) angles of the vector, in degrees.
           errors_deg : tuple
               The geometrically correct projected error components (d_theta, d_phi), in degrees.
           total_error_deg : float
               The final total 3D angular error of the returned vector, in degrees.
        """ 

        # --- Helper Function to calculate continued fraction convergents ---
        def continued_fractions(x, n_terms=15): # Reduced to 15 to avoid flint warning
            """Computes the continued fraction convergents of a number x."""
            convergents = []
            p_prev, q_prev = 0, 1
            p_curr, q_curr = 1, 0
            xi = x
            for _ in range(n_terms):
                a = math.floor(xi)
                p_next = a * p_curr + p_prev
                q_next = a * q_curr + q_prev
                convergents.append((p_next, q_next))
                if abs(xi - a) < 1e-12: return convergents
                xi = 1 / (xi - a)
                p_prev, q_prev = p_curr, q_curr
                p_curr, q_curr = p_next, q_next
            return convergents

        #  Prepare inputs for calculations converting angles from degrees (which are easy for humans) to radians
        #  required by math functions and unpacking the grid spacing vector into individual variables.
        theta_rad = math.radians(theta_deg)
        phi_rad = math.radians(phi_deg)

        #  Convert the spherical angles (theta, phi) into a standard 3D (x,y,z)
        #  vector. This vector is the "target direction" of the plane wave. It
        #  is automatically a "unit vector" (length of 1).
        u_vec_target = np.array([
            math.sin(theta_rad) * math.cos(phi_rad),
            math.sin(theta_rad) * math.sin(phi_rad),
            math.cos(theta_rad)
        ])

        #  The algorithm works by dividing by one of the vector's components (x, y, or z).
        #  To avoid dividing by zero and to keep the math stable, we always choose
        #  the component with the LARGEST absolute value as the reference. We
        #  temporarily rearrange (permute) the vectors so this largest component
        #  is always in the 3rd position (the "z" position). Keep track of how
        #  it has been rearranged so it can be undone later.

        ref_idx = np.argmax(np.abs(u_vec_target))
        perm_order = [0, 1, 2] # Corresponds to x, y, z
        if ref_idx != 2:
            perm_order[2], perm_order[ref_idx] = perm_order[ref_idx], perm_order[2]

        u_perm = u_vec_target[perm_order]
        d_perm = np.array(delta_xyz)[perm_order]

        #  The direction of a wave on the grid depends on both the integers (mx, my, mz)
        #  AND the grid spacing (dx,dy,dz). To find the correct integers, we must
        #  pre-compensate for the grid spacing. We create two target ratios that
        #  tell us what the ideal ratios of (mx/mz) and (my/mz) should be.
        #  We pre-compensate for the grid spacing to find the correct integer ratios.
        ratio_1 = (u_perm[0] / u_perm[2]) * (d_perm[0] / d_perm[2])
        ratio_2 = (u_perm[1] / u_perm[2]) * (d_perm[1] / d_perm[2])

        #  The target ratios are decimal numbers. We need to find simple integer
        #  fractions that as close as possible to these decimals. 
        #  This is done using "continued fractions". A list of these best-guess fractions 
        #  is generated using the helper function for continued fractions (called "convergents").
        convergents1 = continued_fractions(ratio_1)
        convergents2 = continued_fractions(ratio_2)

        #  Search through the lists of "best guess" fractions to find a pair
        #  that works well and meets our error tolerance. We start with the
        #  simplest fractions first and gradually move to more complex ones.
        candidates = []
        for p1, q1 in convergents1:
            for p2, q2 in convergents2:
                common_denom = math.lcm(q1, q2)
                m_perm = np.array([
                    p1 * (common_denom // q1),
                    p2 * (common_denom // q2),
                    common_denom
                ], dtype=int)

                # Apply the crucial sign correction for the correct quadrant
                if np.sign(u_perm[2]) < 0:
                    m_perm = -m_perm

                # Undo the permutation from Step 3
                m_vec_candidate = np.zeros(3, dtype=int)
                m_vec_candidate[perm_order] = m_perm

                # Calculate the candidate's total error
                phys_vec = m_vec_candidate / np.array(delta_xyz)
                phys_vec_norm = phys_vec / (np.linalg.norm(phys_vec) or 1)
                dot_prod = np.clip(np.dot(phys_vec_norm, u_vec_target), -1.0, 1.0)
                total_error_deg = math.degrees(math.acos(dot_prod))

                # Store the candidate with its error and "size" metric. The size
                # is the largest integer component, a good measure of cost.
                size = np.max(np.abs(m_vec_candidate))
                candidates.append({
                    'm_vec': m_vec_candidate,
                    'error': total_error_deg,
                    'size': size
                 })

        # From our list, we keep only those that meet the error criteria, then
        # sort them by size to find the one with the smallest integers.
        valid_candidates = [c for c in candidates if c['error'] <= max_total_error_deg]

        if not valid_candidates:
            print("Warning: No DPW solution found within the error tolerance.")
            return None, (math.nan, math.nan), (math.nan, math.nan), math.nan

        # Sort the valid solutions by size (smallest integers first)
        valid_candidates.sort(key=lambda c: c['size'])
        best_candidate = valid_candidates[0]
        m_vec = best_candidate['m_vec']
        total_error_deg = best_candidate['error']
        max_m = best_candidate['size'] 

        # We perform the final detailed error calculation for the winning vector.
        phys_vec = m_vec / np.array(delta_xyz)
        phys_vec_norm = phys_vec / np.linalg.norm(phys_vec)

        # Define local spherical basis vectors for error projection
        u_theta = np.array([
            math.cos(theta_rad) * math.cos(phi_rad),
            math.cos(theta_rad) * math.sin(phi_rad),
            -math.sin(theta_rad)
        ])
        u_phi = np.array([-math.sin(phi_rad), math.cos(phi_rad), 0])

        # Project the 3D error vector onto the basis vectors
        diff_vec = phys_vec_norm - u_vec_target
        errors_deg = (
            math.degrees(np.dot(diff_vec, u_theta)),
            math.degrees(np.dot(diff_vec, u_phi))
        )

        # Calculate the final angles for user information
        actual_angles_deg = (
            math.degrees(math.acos(np.clip(phys_vec_norm[2], -1.0, 1.0))),
            math.degrees(math.atan2(phys_vec_norm[1], phys_vec_norm[0]))
        )

        return m_vec, actual_angles_deg, errors_deg, total_error_deg


    def _get_pml_parameters(self, G, Z):
        """
        Calculates and sets the DPW PML parameters based on RIPML formulation.
                The forumlation can handle full CFS PML parameters but these are not needed and cannot be set by the user.
                Hence only sigma_max is calculated here based on the standard formula for PMLs. The other parameters are set to values that disable their grading 
                but they can be edited here for testing purposes.
                
                This method uses NumPy vectorization for high performance.
        """
        # --- PML Configuration ---
        self.Order = 4
        self.KOrder = 1
        self.AOrder = 1
        # Sigma Max is calcualted in the same way to the main grid PMls. It must take into account the 
        # actual physical step size of the DPW grid when calcualting the step value which is not just ds.
        self.sigma_max = (0.8 * (self.Order + 1) / 
                          (Z * np.sqrt(self.m[0]**2 + self.m[1]**2 + self.m[2]**2) * self.ds))
        self.Kappa_max = 1.0  # No kappa grading for DPW as it is not needed. You can change this value for testing purposes.
        self.Alpha_max = 0.0  # No alpha grading for DPW as it is not needed. You can change this value for testing purposes.

      
        # --- Create helper arrays for vectorized calculations ---
        # 'depth' array runs from 0 to PMLSize-1  (for sigma and kappa calculations)
        depth = np.arange(self.pml_length)
        # 'i_arr' array runs from PMLSize down to 0 (for alpha calculations)
        i_arr = np.arange(self.pml_length, 0, -1)

        # --- E-Field PML Parameters (Vectorized) ---
        sEx_base = (depth + self.m[0] * 0.5) / self.pml_length
        aEx_base = (i_arr + self.m[0] * 0.5) / self.pml_length
        sEx = self.sigma_max * np.maximum(0, sEx_base)**self.Order
        kEx = 1.0 + (self.Kappa_max - 1.0) * sEx_base**self.KOrder
        aEx = self.Alpha_max * aEx_base**self.AOrder

        sEy_base = (depth + self.m[1] * 0.5) / self.pml_length
        aEy_base = (i_arr + self.m[1] * 0.5) / self.pml_length
        sEy = self.sigma_max * np.maximum(0, sEy_base)**self.Order
        kEy = 1.0 + (self.Kappa_max - 1.0) * sEy_base**self.KOrder
        aEy = self.Alpha_max * aEy_base**self.AOrder
     
        sEz_base = (depth + self.m[2] * 0.5) / self.pml_length
        aEz_base = (i_arr + self.m[2] * 0.5) / self.pml_length
        sEz = self.sigma_max * np.maximum(0, sEz_base)**self.Order
        kEz = 1.0 + (self.Kappa_max - 1.0) * sEz_base**self.KOrder
        aEz = self.Alpha_max * aEz_base**self.AOrder
     
        # --- H-Field PML Parameters (Vectorized) ---
        sHx_base = (depth + (self.m[1] + self.m[2]) * 0.5) / self.pml_length
        aHx_base = (i_arr + (self.m[1] + self.m[2]) * 0.5) / self.pml_length
        sHx = self.sigma_max * np.maximum(0, sHx_base)**self.Order
        kHx = 1.0 + (self.Kappa_max - 1.0) * sHx_base**self.KOrder
        aHx = self.Alpha_max * aHx_base**self.AOrder
     
        sHy_base = (depth + (self.m[0] + self.m[2]) * 0.5) / self.pml_length
        aHy_base = (i_arr + (self.m[0] + self.m[2]) * 0.5) / self.pml_length
        sHy = self.sigma_max * np.maximum(0, sHy_base)**self.Order
        kHy = 1.0 + (self.Kappa_max - 1.0) * sHy_base**self.KOrder
        aHy = self.Alpha_max * aHy_base**self.AOrder
     
        sHz_base = (depth + (self.m[0] + self.m[1]) * 0.5) / self.pml_length
        aHz_base = (i_arr + (self.m[0] + self.m[1]) * 0.5) / self.pml_length
        sHz = self.sigma_max * np.maximum(0, sHz_base)**self.Order
        kHz = 1.0 + (self.Kappa_max - 1.0) * sHz_base**self.KOrder 
        aHz = self.Alpha_max * aHz_base**self.AOrder
       
        # --- Final Update Coefficients (Vectorized) ---
        # Denominators for E and H field updates
        den_Ex = 2 * config.e0 * kEx + G.dt * kEx * aEx + G.dt * sEx
        den_Ey = 2 * config.e0 * kEy + G.dt * kEy * aEy + G.dt * sEy
        den_Ez = 2 * config.e0 * kEz + G.dt * kEz * aEz + G.dt * sEz
        den_Hx = 2 * config.e0 * kHx + G.dt * kHx * aHx + G.dt * sHx
        den_Hy = 2 * config.e0 * kHy + G.dt * kHy * aHy + G.dt * sHy
        den_Hz = 2 * config.e0 * kHz + G.dt * kHz * aHz + G.dt * sHz

        # RA Coefficients
        self.RAEx = (2 * config.e0 * (1 - kEx) + G.dt * aEx * (1 - kEx) - G.dt * sEx) / den_Ex
        self.RAEy = (2 * config.e0 * (1 - kEy) + G.dt * aEy * (1 - kEy) - G.dt * sEy) / den_Ey
        self.RAEz = (2 * config.e0 * (1 - kEz) + G.dt * aEz * (1 - kEz) - G.dt * sEz) / den_Ez
        self.RAHx = (2 * config.e0 * (1 - kHx) + G.dt * aHx * (1 - kHx) - G.dt * sHx) / den_Hx
        self.RAHy = (2 * config.e0 * (1 - kHy) + G.dt * aHy * (1 - kHy) - G.dt * sHy) / den_Hy
        self.RAHz = (2 * config.e0 * (1 - kHz) + G.dt * aHz * (1 - kHz) - G.dt * sHz) / den_Hz

        # RB Coefficients
        self.RBEx, self.RBEy, self.RBEz = 2 / den_Ex, 2 / den_Ey, 2 / den_Ez
        self.RBHx, self.RBHy, self.RBHz = 2 / den_Hx, 2 / den_Hy, 2 / den_Hz

        # RC Coefficients
        self.RCEx = G.dt * (kEx * aEx + sEx)
        self.RCEy = G.dt * (kEy * aEy + sEy)
        self.RCEz = G.dt * (kEz * aEz + sEz)
        self.RCHx = G.dt * (kHx * aHx + sHx)
        self.RCHy = G.dt * (kHy * aHy + sHy)
        self.RCHz = G.dt * (kHz * aHz + sHz)
        
        # RD Coefficients
        self.RDEx = G.dt * (aEx * (1 - kEx) - sEx)
        self.RDEy = G.dt * (aEy * (1 - kEy) - sEy)
        self.RDEz = G.dt * (aEz * (1 - kEz) - sEz)
        self.RDHx = G.dt * (aHx * (1 - kHx) - sHx)
        self.RDHy = G.dt * (aHy * (1 - kHy) - sHy)
        self.RDHz = G.dt * (aHz * (1 - kHz) - sHz)

        # --- Combine Coefficients into Single Matrices ---
        # Creates 2D arrays (4 rows x pml_length columns) for the PML coefficients RA row: 0, RB row: 1, RC rowe:2 and RD row: 3 for the Ex,Ey,Ez, 
        # Hz, Hy, Hz components
        self.pml_rex = np.array([self.RAEx, self.RBEx, self.RCEx, self.RDEx], order="C", dtype=config.sim_config.dtypes["float_or_double"])
        self.pml_rey = np.array([self.RAEy, self.RBEy, self.RCEy, self.RDEy], order="C", dtype=config.sim_config.dtypes["float_or_double"])
        self.pml_rez = np.array([self.RAEz, self.RBEz, self.RCEz, self.RDEz], order="C", dtype=config.sim_config.dtypes["float_or_double"])

        self.pml_rhx = np.array([self.RAHx, self.RBHx, self.RCHx, self.RDHx], order="C", dtype=config.sim_config.dtypes["float_or_double"])
        self.pml_rhy = np.array([self.RAHy, self.RBHy, self.RCHy, self.RDHy], order="C", dtype=config.sim_config.dtypes["float_or_double"])
        self.pml_rhz = np.array([self.RAHz, self.RBHz, self.RCHz, self.RDHz], order="C", dtype=config.sim_config.dtypes["float_or_double"])

        