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

import logging
from importlib import import_module
from typing import List

import numpy as np
from mpi4py import MPI

import gprMax.config as config

from .cython.pml_build import pml_average_er_mr, pml_sum_er_mr

logger = logging.getLogger(__name__)


class CFSParameter:
    """Individual CFS parameter (e.g. alpha, kappa, or sigma)."""

    # Allowable scaling profiles and directions
    scalingprofiles = {
        "constant": 0,
        "linear": 1,
        "quadratic": 2,
        "cubic": 3,
        "quartic": 4,
        "quintic": 5,
        "sextic": 6,
        "septic": 7,
        "octic": 8,
    }
    scalingdirections = ["forward", "reverse"]

    def __init__(
        self,
        ID=None,
        scaling="polynomial",
        scalingprofile=None,
        scalingdirection="forward",
        min=0,
        max=0,
    ):
        """
        Args:
            ID: string identifier for CFS parameter, can be: 'alpha', 'kappa' or
                'sigma'.
            scaling: string for type of scaling, can be: 'polynomial'.
            scalingprofile: string for type of scaling profile from
                            scalingprofiles.
            scalingdirection: string for direction of scaling profile from
                                scalingdirections.
            min: float for minimum value for parameter.
            max: float for maximum value for parameter.
        """

        self.ID = ID
        self.scaling = scaling
        self.scalingprofile = scalingprofile
        self.scalingdirection = scalingdirection
        self.min = min
        self.max = max


class CFS:
    """CFS term for PML."""

    def __init__(self):
        """
        Args:
            alpha: CFSParameter alpha parameter for CFS.
            kappa: CFSParameter kappa parameter for CFS.
            sigma: CFSParameter sigma parameter for CFS.
        """

        self.alpha = CFSParameter(ID="alpha", scalingprofile="constant")
        self.kappa = CFSParameter(ID="kappa", scalingprofile="constant", min=1, max=1)
        self.sigma = CFSParameter(ID="sigma", scalingprofile="quartic", min=0, max=None)

    def calculate_sigmamax(self, d, er, mr):
        """Calculates an optimum value for sigma max based on underlying
            material properties.

        Args:
            d: float for dx, dy, or dz in direction of PML.
            er: float for average permittivity of underlying material.
            mr: float for average permeability of underlying material.
            G: FDTDGrid class describing a grid in a model.
        """

        # Calculation of the maximum value of sigma from http://dx.doi.org/10.1109/8.546249
        m = CFSParameter.scalingprofiles[self.sigma.scalingprofile]
        self.sigma.max = (0.8 * (m + 1)) / (
            config.sim_config.em_consts["z0"] * d * np.sqrt(er * mr)
        )

    def scaling_polynomial(self, order, Evalues, Hvalues):
        """Applies the polynomial to be used for the scaling profile for
            electric and magnetic PML updates.

        Args:
            order: int of order of polynomial for scaling profile.
            Evalues: float array holding scaling profile values for
                        electric PML update.
            Hvalues: float array holding scaling profile values for
                        magnetic PML update.

        Returns:
            Evalues: float array holding scaling profile values for
                        electric PML update.
            Hvalues: float array holding scaling profile values for
                        magnetic PML update.
        """

        tmp = (
            np.linspace(0, (len(Evalues) - 1) + 0.5, num=2 * len(Evalues)) / (len(Evalues) - 1)
        ) ** order
        Evalues = tmp[0:-1:2]
        Hvalues = tmp[1::2]

        return Evalues, Hvalues

    def calculate_values(self, thickness, parameter):
        """Calculates values for electric and magnetic PML updates based on
            profile type and minimum and maximum values.

        Args:
            thickness: int of thickness of PML in cells.
            parameter: instance of CFSParameter

        Returns:
            Evalues: float array holding profile value for electric
                        PML update.
            Hvalues: float array holding profile value for magnetic
                        PML update.
        """

        # Extra cell of thickness added to allow correct scaling of electric and
        # magnetic values
        Evalues = np.zeros(thickness + 1, dtype=config.sim_config.dtypes["float_or_double"])
        Hvalues = np.zeros(thickness + 1, dtype=config.sim_config.dtypes["float_or_double"])

        if parameter.scalingprofile == "constant":
            Evalues += parameter.max
            Hvalues += parameter.max

        elif parameter.scaling == "polynomial":
            Evalues, Hvalues = self.scaling_polynomial(
                CFSParameter.scalingprofiles[parameter.scalingprofile], Evalues, Hvalues
            )
            if parameter.ID == "alpha":
                Evalues = Evalues * (self.alpha.max - self.alpha.min) + self.alpha.min
                Hvalues = Hvalues * (self.alpha.max - self.alpha.min) + self.alpha.min
            elif parameter.ID == "kappa":
                Evalues = Evalues * (self.kappa.max - self.kappa.min) + self.kappa.min
                Hvalues = Hvalues * (self.kappa.max - self.kappa.min) + self.kappa.min
            elif parameter.ID == "sigma":
                Evalues = Evalues * (self.sigma.max - self.sigma.min) + self.sigma.min
                Hvalues = Hvalues * (self.sigma.max - self.sigma.min) + self.sigma.min

        if parameter.scalingdirection == "reverse":
            Evalues = Evalues[::-1]
            Hvalues = Hvalues[::-1]
            # Magnetic values must be shifted one element to the left after
            # reversal
            Hvalues = np.roll(Hvalues, -1)

        # Extra cell of thickness not required and therefore removed after
        # scaling
        Evalues = Evalues[:-1]
        Hvalues = Hvalues[:-1]

        return Evalues, Hvalues


class PML:
    """Perfectly Matched Layer (PML) Absorbing Boundary Conditions (ABC)"""

    # Available PML formulations:
    # Higher Order RIPML (HORIPML) see: https://doi.org/10.1109/TAP.2011.2180344
    # Multipole RIPML (MRIPML) see: https://doi.org/10.1109/TAP.2018.2823864
    formulations = ["HORIPML", "MRIPML"]

    # PML slabs IDs at boundaries of domain.
    boundaryIDs = ["x0", "y0", "z0", "xmax", "ymax", "zmax"]

    # Indicates direction of increasing absorption
    # xminus, yminus, zminus - absorption increases in negative direction of
    #                           x-axis, y-axis, or z-axis
    # xplus, yplus, zplus - absorption increases in positive direction of
    #                       x-axis, y-axis, or z-axis
    directions = ["xminus", "yminus", "zminus", "xplus", "yplus", "zplus"]

    def __init__(self, G, ID: str, direction: str, xs=0, xf=0, ys=0, yf=0, zs=0, zf=0):
        """
        Args:
            G: FDTDGrid class describing a grid in a model.
            ID: string identifier for PML slab.
            direction: string for direction of increasing absorption.
            xs, xf, ys, yf, zs, zf: floats of extent of the PML slab.
        """

        self.G = G
        self.ID = ID
        self.direction = direction
        self.xs = xs
        self.xf = xf
        self.ys = ys
        self.yf = yf
        self.zs = zs
        self.zf = zf
        self.nx = xf - xs
        self.ny = yf - ys
        self.nz = zf - zs

        # Spatial discretisation and thickness
        if self.direction[0] == "x":
            self.d = self.G.dx
            self.thickness = self.nx
        elif self.direction[0] == "y":
            self.d = self.G.dy
            self.thickness = self.ny
        elif self.direction[0] == "z":
            self.d = self.G.dz
            self.thickness = self.nz

        self.CFS: List[CFS] = self.G.pmls["cfs"]
        self.check_kappamin()

        self.initialise_field_arrays()

    def check_kappamin(self):
        """Checks that the sum of all kappamin values, i.e. when a multi-pole
        PML is used, is not less than one.
        """

        kappamin = sum(cfs.kappa.min for cfs in self.CFS)
        if kappamin < 1:
            logger.exception(
                f"Sum of kappamin value(s) for PML is {kappamin} and must be greater than one."
            )
            raise ValueError

    def initialise_field_arrays(self):
        """Initialise arrays to store fields in PML."""

        if self.direction[0] == "x":
            self.EPhi1 = np.zeros(
                (len(self.CFS), self.nx + 1, self.ny, self.nz + 1),
                dtype=config.sim_config.dtypes["float_or_double"],
            )
            self.EPhi2 = np.zeros(
                (len(self.CFS), self.nx + 1, self.ny + 1, self.nz),
                dtype=config.sim_config.dtypes["float_or_double"],
            )
            self.HPhi1 = np.zeros(
                (len(self.CFS), self.nx, self.ny + 1, self.nz),
                dtype=config.sim_config.dtypes["float_or_double"],
            )
            self.HPhi2 = np.zeros(
                (len(self.CFS), self.nx, self.ny, self.nz + 1),
                dtype=config.sim_config.dtypes["float_or_double"],
            )
        elif self.direction[0] == "y":
            self.EPhi1 = np.zeros(
                (len(self.CFS), self.nx, self.ny + 1, self.nz + 1),
                dtype=config.sim_config.dtypes["float_or_double"],
            )
            self.EPhi2 = np.zeros(
                (len(self.CFS), self.nx + 1, self.ny + 1, self.nz),
                dtype=config.sim_config.dtypes["float_or_double"],
            )
            self.HPhi1 = np.zeros(
                (len(self.CFS), self.nx + 1, self.ny, self.nz),
                dtype=config.sim_config.dtypes["float_or_double"],
            )
            self.HPhi2 = np.zeros(
                (len(self.CFS), self.nx, self.ny, self.nz + 1),
                dtype=config.sim_config.dtypes["float_or_double"],
            )
        elif self.direction[0] == "z":
            self.EPhi1 = np.zeros(
                (len(self.CFS), self.nx, self.ny + 1, self.nz + 1),
                dtype=config.sim_config.dtypes["float_or_double"],
            )
            self.EPhi2 = np.zeros(
                (len(self.CFS), self.nx + 1, self.ny, self.nz + 1),
                dtype=config.sim_config.dtypes["float_or_double"],
            )
            self.HPhi1 = np.zeros(
                (len(self.CFS), self.nx + 1, self.ny, self.nz),
                dtype=config.sim_config.dtypes["float_or_double"],
            )
            self.HPhi2 = np.zeros(
                (len(self.CFS), self.nx, self.ny + 1, self.nz),
                dtype=config.sim_config.dtypes["float_or_double"],
            )

    def calculate_update_coeffs(self, er, mr):
        """Calculates electric and magnetic update coefficients for the PML.

        Args:
            er: float of average permittivity of underlying material
            mr: float of average permeability of underlying material
        """

        self.ERA = np.zeros(
            (len(self.CFS), self.thickness), dtype=config.sim_config.dtypes["float_or_double"]
        )
        self.ERB = np.zeros(
            (len(self.CFS), self.thickness), dtype=config.sim_config.dtypes["float_or_double"]
        )
        self.ERE = np.zeros(
            (len(self.CFS), self.thickness), dtype=config.sim_config.dtypes["float_or_double"]
        )
        self.ERF = np.zeros(
            (len(self.CFS), self.thickness), dtype=config.sim_config.dtypes["float_or_double"]
        )
        self.HRA = np.zeros(
            (len(self.CFS), self.thickness), dtype=config.sim_config.dtypes["float_or_double"]
        )
        self.HRB = np.zeros(
            (len(self.CFS), self.thickness), dtype=config.sim_config.dtypes["float_or_double"]
        )
        self.HRE = np.zeros(
            (len(self.CFS), self.thickness), dtype=config.sim_config.dtypes["float_or_double"]
        )
        self.HRF = np.zeros(
            (len(self.CFS), self.thickness), dtype=config.sim_config.dtypes["float_or_double"]
        )

        for x, cfs in enumerate(self.CFS):
            if not cfs.sigma.max:
                cfs.calculate_sigmamax(self.d, er, mr)
            logger.debug(
                f"PML {self.ID}: sigma.max set to {cfs.sigma.max} for {'first' if x == 0 else 'second'} order CFS parameter"
            )
            Ealpha, Halpha = cfs.calculate_values(self.thickness, cfs.alpha)
            Ekappa, Hkappa = cfs.calculate_values(self.thickness, cfs.kappa)
            Esigma, Hsigma = cfs.calculate_values(self.thickness, cfs.sigma)

            # Define different parameters depending on PML formulation
            if self.G.pmls["formulation"] == "HORIPML":
                # HORIPML electric update coefficients
                tmp = (2 * config.sim_config.em_consts["e0"] * Ekappa) + self.G.dt * (
                    Ealpha * Ekappa + Esigma
                )
                self.ERA[x, :] = (2 * config.sim_config.em_consts["e0"] + self.G.dt * Ealpha) / tmp
                self.ERB[x, :] = (2 * config.sim_config.em_consts["e0"] * Ekappa) / tmp
                self.ERE[x, :] = (
                    (2 * config.sim_config.em_consts["e0"] * Ekappa)
                    - self.G.dt * (Ealpha * Ekappa + Esigma)
                ) / tmp
                self.ERF[x, :] = (2 * Esigma * self.G.dt) / (Ekappa * tmp)

                # HORIPML magnetic update coefficients
                tmp = (2 * config.sim_config.em_consts["e0"] * Hkappa) + self.G.dt * (
                    Halpha * Hkappa + Hsigma
                )
                self.HRA[x, :] = (2 * config.sim_config.em_consts["e0"] + self.G.dt * Halpha) / tmp
                self.HRB[x, :] = (2 * config.sim_config.em_consts["e0"] * Hkappa) / tmp
                self.HRE[x, :] = (
                    (2 * config.sim_config.em_consts["e0"] * Hkappa)
                    - self.G.dt * (Halpha * Hkappa + Hsigma)
                ) / tmp
                self.HRF[x, :] = (2 * Hsigma * self.G.dt) / (Hkappa * tmp)

            elif self.G.pmls["formulation"] == "MRIPML":
                # MRIPML electric update coefficients
                tmp = 2 * config.sim_config.em_consts["e0"] + self.G.dt * Ealpha
                self.ERA[x, :] = Ekappa + (self.G.dt * Esigma) / tmp
                self.ERB[x, :] = (2 * config.sim_config.em_consts["e0"]) / tmp
                self.ERE[x, :] = (
                    (2 * config.sim_config.em_consts["e0"]) - self.G.dt * Ealpha
                ) / tmp
                self.ERF[x, :] = (2 * Esigma * self.G.dt) / tmp

                # MRIPML magnetic update coefficients
                tmp = 2 * config.sim_config.em_consts["e0"] + self.G.dt * Halpha
                self.HRA[x, :] = Hkappa + (self.G.dt * Hsigma) / tmp
                self.HRB[x, :] = (2 * config.sim_config.em_consts["e0"]) / tmp
                self.HRE[x, :] = (
                    (2 * config.sim_config.em_consts["e0"]) - self.G.dt * Halpha
                ) / tmp
                self.HRF[x, :] = (2 * Hsigma * self.G.dt) / tmp

    def update_electric(self):
        """This functions updates electric field components with the PML
        correction.
        """

        pmlmodule = "gprMax.cython.pml_updates_electric_" + self.G.pmls["formulation"]
        func = getattr(
            import_module(pmlmodule), "order" + str(len(self.CFS)) + "_" + self.direction
        )
        func(
            self.xs,
            self.xf,
            self.ys,
            self.yf,
            self.zs,
            self.zf,
            config.get_model_config().ompthreads,
            self.G.updatecoeffsE,
            self.G.ID,
            self.G.Ex,
            self.G.Ey,
            self.G.Ez,
            self.G.Hx,
            self.G.Hy,
            self.G.Hz,
            self.EPhi1,
            self.EPhi2,
            self.ERA,
            self.ERB,
            self.ERE,
            self.ERF,
            self.d,
        )

    def update_magnetic(self):
        """This functions updates magnetic field components with the PML
        correction.
        """

        pmlmodule = "gprMax.cython.pml_updates_magnetic_" + self.G.pmls["formulation"]
        func = getattr(
            import_module(pmlmodule), "order" + str(len(self.CFS)) + "_" + self.direction
        )
        func(
            self.xs,
            self.xf,
            self.ys,
            self.yf,
            self.zs,
            self.zf,
            config.get_model_config().ompthreads,
            self.G.updatecoeffsH,
            self.G.ID,
            self.G.Ex,
            self.G.Ey,
            self.G.Ez,
            self.G.Hx,
            self.G.Hy,
            self.G.Hz,
            self.HPhi1,
            self.HPhi2,
            self.HRA,
            self.HRB,
            self.HRE,
            self.HRF,
            self.d,
        )


class CUDAPML(PML):
    """Perfectly Matched Layer (PML) Absorbing Boundary Conditions (ABC) for
    solving on GPU using CUDA.
    """

    def __init__(self, *args, **kwargs):
        super(CUDAPML, self).__init__(*args, **kwargs)

    def htod_field_arrays(self):
        """Initialises PML field and coefficient arrays on GPU."""

        import pycuda.gpuarray as gpuarray

        self.ERA_dev = gpuarray.to_gpu(self.ERA)
        self.ERB_dev = gpuarray.to_gpu(self.ERB)
        self.ERE_dev = gpuarray.to_gpu(self.ERE)
        self.ERF_dev = gpuarray.to_gpu(self.ERF)
        self.HRA_dev = gpuarray.to_gpu(self.HRA)
        self.HRB_dev = gpuarray.to_gpu(self.HRB)
        self.HRE_dev = gpuarray.to_gpu(self.HRE)
        self.HRF_dev = gpuarray.to_gpu(self.HRF)
        self.EPhi1_dev = gpuarray.to_gpu(self.EPhi1)
        self.EPhi2_dev = gpuarray.to_gpu(self.EPhi2)
        self.HPhi1_dev = gpuarray.to_gpu(self.HPhi1)
        self.HPhi2_dev = gpuarray.to_gpu(self.HPhi2)

    def set_blocks_per_grid(self):
        """Sets the blocks per grid size used for updating the PML field arrays
        on a GPU."""
        self.bpg = (
            int(
                np.ceil(
                    (
                        (self.EPhi1_dev.shape[1] + 1)
                        * (self.EPhi1_dev.shape[2] + 1)
                        * (self.EPhi1_dev.shape[3] + 1)
                    )
                    / self.G.tpb[0]
                )
            ),
            1,
            1,
        )

    def update_electric(self):
        """Updates electric field components with the PML correction on the GPU."""
        self.update_electric_dev(
            np.int32(self.xs),
            np.int32(self.xf),
            np.int32(self.ys),
            np.int32(self.yf),
            np.int32(self.zs),
            np.int32(self.zf),
            np.int32(self.EPhi1_dev.shape[1]),
            np.int32(self.EPhi1_dev.shape[2]),
            np.int32(self.EPhi1_dev.shape[3]),
            np.int32(self.EPhi2_dev.shape[1]),
            np.int32(self.EPhi2_dev.shape[2]),
            np.int32(self.EPhi2_dev.shape[3]),
            np.int32(self.thickness),
            self.G.ID_dev.gpudata,
            self.G.Ex_dev.gpudata,
            self.G.Ey_dev.gpudata,
            self.G.Ez_dev.gpudata,
            self.G.Hx_dev.gpudata,
            self.G.Hy_dev.gpudata,
            self.G.Hz_dev.gpudata,
            self.EPhi1_dev.gpudata,
            self.EPhi2_dev.gpudata,
            self.ERA_dev.gpudata,
            self.ERB_dev.gpudata,
            self.ERE_dev.gpudata,
            self.ERF_dev.gpudata,
            config.sim_config.dtypes["float_or_double"](self.d),
            block=self.G.tpb,
            grid=self.bpg,
        )

    def update_magnetic(self):
        """Updates magnetic field components with the PML correction on the GPU."""
        self.update_magnetic_dev(
            np.int32(self.xs),
            np.int32(self.xf),
            np.int32(self.ys),
            np.int32(self.yf),
            np.int32(self.zs),
            np.int32(self.zf),
            np.int32(self.HPhi1_dev.shape[1]),
            np.int32(self.HPhi1_dev.shape[2]),
            np.int32(self.HPhi1_dev.shape[3]),
            np.int32(self.HPhi2_dev.shape[1]),
            np.int32(self.HPhi2_dev.shape[2]),
            np.int32(self.HPhi2_dev.shape[3]),
            np.int32(self.thickness),
            self.G.ID_dev.gpudata,
            self.G.Ex_dev.gpudata,
            self.G.Ey_dev.gpudata,
            self.G.Ez_dev.gpudata,
            self.G.Hx_dev.gpudata,
            self.G.Hy_dev.gpudata,
            self.G.Hz_dev.gpudata,
            self.HPhi1_dev.gpudata,
            self.HPhi2_dev.gpudata,
            self.HRA_dev.gpudata,
            self.HRB_dev.gpudata,
            self.HRE_dev.gpudata,
            self.HRF_dev.gpudata,
            config.sim_config.dtypes["float_or_double"](self.d),
            block=self.G.tpb,
            grid=self.bpg,
        )


class OpenCLPML(PML):
    """Perfectly Matched Layer (PML) Absorbing Boundary Conditions (ABC) for
    solving on compute device using OpenCL.
    """

    def __init__(self, *args, **kwargs):
        super(OpenCLPML, self).__init__(*args, **kwargs)

    def set_queue(self, queue):
        """Passes in pyopencl queue.

        Args:
            queue: pyopencl queue.
        """
        self.queue = queue

    def htod_field_arrays(self):
        """Initialises PML field and coefficient arrays on compute device."""

        import pyopencl.array as clarray

        self.ERA_dev = clarray.to_device(self.queue, self.ERA)
        self.ERB_dev = clarray.to_device(self.queue, self.ERB)
        self.ERE_dev = clarray.to_device(self.queue, self.ERE)
        self.ERF_dev = clarray.to_device(self.queue, self.ERF)
        self.HRA_dev = clarray.to_device(self.queue, self.HRA)
        self.HRB_dev = clarray.to_device(self.queue, self.HRB)
        self.HRE_dev = clarray.to_device(self.queue, self.HRE)
        self.HRF_dev = clarray.to_device(self.queue, self.HRF)
        self.EPhi1_dev = clarray.to_device(self.queue, self.EPhi1)
        self.EPhi2_dev = clarray.to_device(self.queue, self.EPhi2)
        self.HPhi1_dev = clarray.to_device(self.queue, self.HPhi1)
        self.HPhi2_dev = clarray.to_device(self.queue, self.HPhi2)

    def update_electric(self):
        """Updates electric field components with the PML correction on the
        compute device.
        """
        event = self.update_electric_dev(
            np.int32(self.xs),
            np.int32(self.xf),
            np.int32(self.ys),
            np.int32(self.yf),
            np.int32(self.zs),
            np.int32(self.zf),
            np.int32(self.EPhi1_dev.shape[1]),
            np.int32(self.EPhi1_dev.shape[2]),
            np.int32(self.EPhi1_dev.shape[3]),
            np.int32(self.EPhi2_dev.shape[1]),
            np.int32(self.EPhi2_dev.shape[2]),
            np.int32(self.EPhi2_dev.shape[3]),
            np.int32(self.thickness),
            self.G.ID_dev,
            self.G.Ex_dev,
            self.G.Ey_dev,
            self.G.Ez_dev,
            self.G.Hx_dev,
            self.G.Hy_dev,
            self.G.Hz_dev,
            self.EPhi1_dev,
            self.EPhi2_dev,
            self.ERA_dev,
            self.ERB_dev,
            self.ERE_dev,
            self.ERF_dev,
            config.sim_config.dtypes["float_or_double"](self.d),
        )
        event.wait()

    def update_magnetic(self):
        """Updates magnetic field components with the PML correction on the
        compute device.
        """
        event = self.update_magnetic_dev(
            np.int32(self.xs),
            np.int32(self.xf),
            np.int32(self.ys),
            np.int32(self.yf),
            np.int32(self.zs),
            np.int32(self.zf),
            np.int32(self.HPhi1_dev.shape[1]),
            np.int32(self.HPhi1_dev.shape[2]),
            np.int32(self.HPhi1_dev.shape[3]),
            np.int32(self.HPhi2_dev.shape[1]),
            np.int32(self.HPhi2_dev.shape[2]),
            np.int32(self.HPhi2_dev.shape[3]),
            np.int32(self.thickness),
            self.G.ID_dev,
            self.G.Ex_dev,
            self.G.Ey_dev,
            self.G.Ez_dev,
            self.G.Hx_dev,
            self.G.Hy_dev,
            self.G.Hz_dev,
            self.HPhi1_dev,
            self.HPhi2_dev,
            self.HRA_dev,
            self.HRB_dev,
            self.HRE_dev,
            self.HRF_dev,
            config.sim_config.dtypes["float_or_double"](self.d),
        )
        event.wait()


class MetalPML(PML):
    """Perfectly Matched Layer (PML) Absorbing Boundary Conditions (ABC) for
    solving on GPU using Apple Metal.
    """

    def __init__(self, *args, **kwargs):
        super(MetalPML, self).__init__(*args, **kwargs)

    def htod_field_arrays(self, dev=None):
        """Initialises PML field and coefficient arrays on GPU."""
        
        # Create Metal buffers for all PML arrays using device's method
        if dev is None:
            raise RuntimeError("Metal device not provided. PML arrays cannot be initialized.")
        
        # Store shapes before creating buffers (since Metal buffers don't have shape attribute)
        self.EPhi1_shape = self.EPhi1.shape
        self.EPhi2_shape = self.EPhi2.shape  
        self.HPhi1_shape = self.HPhi1.shape
        self.HPhi2_shape = self.HPhi2.shape
            
        self.ERA_dev = dev.newBufferWithBytes_length_options_(self.ERA, 
                                                                        self.ERA.nbytes, 0)
        self.ERB_dev = dev.newBufferWithBytes_length_options_(self.ERB, 
                                                                        self.ERB.nbytes, 0)
        self.ERE_dev = dev.newBufferWithBytes_length_options_(self.ERE, 
                                                                        self.ERE.nbytes, 0)
        self.ERF_dev = dev.newBufferWithBytes_length_options_(self.ERF, 
                                                                        self.ERF.nbytes, 0)
        self.HRA_dev = dev.newBufferWithBytes_length_options_(self.HRA, 
                                                                        self.HRA.nbytes, 0)
        self.HRB_dev = dev.newBufferWithBytes_length_options_(self.HRB, 
                                                                        self.HRB.nbytes, 0)
        self.HRE_dev = dev.newBufferWithBytes_length_options_(self.HRE, 
                                                                        self.HRE.nbytes, 0)
        self.HRF_dev = dev.newBufferWithBytes_length_options_(self.HRF, 
                                                                        self.HRF.nbytes, 0)
        self.EPhi1_dev = dev.newBufferWithBytes_length_options_(self.EPhi1, 
                                                                          self.EPhi1.nbytes, 0)
        self.EPhi2_dev = dev.newBufferWithBytes_length_options_(self.EPhi2, 
                                                                          self.EPhi2.nbytes, 0)
        self.HPhi1_dev = dev.newBufferWithBytes_length_options_(self.HPhi1, 
                                                                          self.HPhi1.nbytes, 0)
        self.HPhi2_dev = dev.newBufferWithBytes_length_options_(self.HPhi2, 
                                                                          self.HPhi2.nbytes, 0)

    def set_queue(self, queue):
        """Sets the command queue for the PML."""
        self.queue = queue

    def update_electric(self):
        """Updates electric field components with the PML correction on the GPU using Metal."""
        
        # Create command buffer and encoder
        cmdbuffer = self.queue.commandBuffer()
        cmpencoder = cmdbuffer.computeCommandEncoder()
        cmpencoder.setComputePipelineState_(self.psoE)
        
        # Set scalar parameters
        cmpencoder.setBytes_length_atIndex_(np.int32(self.xs).tobytes(), 4, 0)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.xf).tobytes(), 4, 1)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.ys).tobytes(), 4, 2)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.yf).tobytes(), 4, 3)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.zs).tobytes(), 4, 4)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.zf).tobytes(), 4, 5)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.EPhi1_shape[1]).tobytes(), 4, 6)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.EPhi1_shape[2]).tobytes(), 4, 7)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.EPhi1_shape[3]).tobytes(), 4, 8)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.EPhi2_shape[1]).tobytes(), 4, 9)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.EPhi2_shape[2]).tobytes(), 4, 10)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.EPhi2_shape[3]).tobytes(), 4, 11)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.thickness).tobytes(), 4, 12)
        
        # Set buffer arguments
        cmpencoder.setBuffer_offset_atIndex_(self.G.ID_dev, 0, 13)
        cmpencoder.setBuffer_offset_atIndex_(self.G.Ex_dev, 0, 14)
        cmpencoder.setBuffer_offset_atIndex_(self.G.Ey_dev, 0, 15)
        cmpencoder.setBuffer_offset_atIndex_(self.G.Ez_dev, 0, 16)
        cmpencoder.setBuffer_offset_atIndex_(self.G.Hx_dev, 0, 17)
        cmpencoder.setBuffer_offset_atIndex_(self.G.Hy_dev, 0, 18)
        cmpencoder.setBuffer_offset_atIndex_(self.G.Hz_dev, 0, 19)
        cmpencoder.setBuffer_offset_atIndex_(self.EPhi1_dev, 0, 20)
        cmpencoder.setBuffer_offset_atIndex_(self.EPhi2_dev, 0, 21)
        cmpencoder.setBuffer_offset_atIndex_(self.ERA_dev, 0, 22)
        cmpencoder.setBuffer_offset_atIndex_(self.ERB_dev, 0, 23)
        cmpencoder.setBuffer_offset_atIndex_(self.ERE_dev, 0, 24)
        cmpencoder.setBuffer_offset_atIndex_(self.ERF_dev, 0, 25)
        d_bytes = config.sim_config.dtypes["float_or_double"](self.d).tobytes()
        cmpencoder.setBytes_length_atIndex_(d_bytes, len(d_bytes), 26)
        
        # Dispatch threads using grid's thread configuration
        cmpencoder.dispatchThreads_threadsPerThreadgroup_(self.G.tptg, self.G.tgs)
        
        cmpencoder.endEncoding()
        cmdbuffer.commit()
        cmdbuffer.waitUntilCompleted()

    def update_magnetic(self):
        """Updates magnetic field components with the PML correction on the GPU using Metal."""
        
        # Create command buffer and encoder
        cmdbuffer = self.queue.commandBuffer()
        cmpencoder = cmdbuffer.computeCommandEncoder()
        cmpencoder.setComputePipelineState_(self.psoH)
        
        # Set scalar parameters
        cmpencoder.setBytes_length_atIndex_(np.int32(self.xs).tobytes(), 4, 0)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.xf).tobytes(), 4, 1)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.ys).tobytes(), 4, 2)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.yf).tobytes(), 4, 3)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.zs).tobytes(), 4, 4)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.zf).tobytes(), 4, 5)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.HPhi1_shape[1]).tobytes(), 4, 6)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.HPhi1_shape[2]).tobytes(), 4, 7)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.HPhi1_shape[3]).tobytes(), 4, 8)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.HPhi2_shape[1]).tobytes(), 4, 9)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.HPhi2_shape[2]).tobytes(), 4, 10)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.HPhi2_shape[3]).tobytes(), 4, 11)
        cmpencoder.setBytes_length_atIndex_(np.int32(self.thickness).tobytes(), 4, 12)
        
        # Set buffer arguments
        cmpencoder.setBuffer_offset_atIndex_(self.G.ID_dev, 0, 13)
        cmpencoder.setBuffer_offset_atIndex_(self.G.Ex_dev, 0, 14)
        cmpencoder.setBuffer_offset_atIndex_(self.G.Ey_dev, 0, 15)
        cmpencoder.setBuffer_offset_atIndex_(self.G.Ez_dev, 0, 16)
        cmpencoder.setBuffer_offset_atIndex_(self.G.Hx_dev, 0, 17)
        cmpencoder.setBuffer_offset_atIndex_(self.G.Hy_dev, 0, 18)
        cmpencoder.setBuffer_offset_atIndex_(self.G.Hz_dev, 0, 19)
        cmpencoder.setBuffer_offset_atIndex_(self.HPhi1_dev, 0, 20)
        cmpencoder.setBuffer_offset_atIndex_(self.HPhi2_dev, 0, 21)
        cmpencoder.setBuffer_offset_atIndex_(self.HRA_dev, 0, 22)
        cmpencoder.setBuffer_offset_atIndex_(self.HRB_dev, 0, 23)
        cmpencoder.setBuffer_offset_atIndex_(self.HRE_dev, 0, 24)
        cmpencoder.setBuffer_offset_atIndex_(self.HRF_dev, 0, 25)
        d_bytes = config.sim_config.dtypes["float_or_double"](self.d).tobytes()
        cmpencoder.setBytes_length_atIndex_(d_bytes, len(d_bytes), 26)
        
        # Dispatch threads using grid's thread configuration
        cmpencoder.dispatchThreads_threadsPerThreadgroup_(self.G.tptg, self.G.tgs)
        
        cmpencoder.endEncoding()
        cmdbuffer.commit()
        cmdbuffer.waitUntilCompleted()


class MPIPML(PML):
    comm: MPI.Cartcomm
    global_comm: MPI.Comm

    COORDINATOR_RANK = 0

    def calculate_update_coeffs(self, er: float, mr: float):
        """Calculates electric and magnetic update coefficients for the PML.

        Args:
            er: float of average permittivity of underlying material
            mr: float of average permeability of underlying material
        """
        for cfs in self.CFS:
            if not cfs.sigma.max:
                if self.global_comm.rank == self.COORDINATOR_RANK:
                    cfs.calculate_sigmamax(self.d, er, mr)
                    buffer = np.array([cfs.sigma.max])
                else:
                    buffer = np.empty(1)

                # Needs to be non-blocking because some ranks will
                # contain multiple PMLs, but the material properties for
                # a PML cannot be calculated until all ranks have
                # completed that stage. Therefore a blocking broadcast
                # would wait for ranks that are stuck calculating the
                # material properties of the PML.
                self.global_comm.Ibcast(buffer, self.COORDINATOR_RANK).Wait()
                cfs.sigma.max = buffer[0]

        super().calculate_update_coeffs(er, mr)


def print_pml_info(G):
    """Prints information about PMLs.

    Args:
        G: FDTDGrid class describing a grid in a model.
    """
    # No PML
    if all(value == 0 for value in G.pmls["thickness"].values()):
        return f"PML boundaries [{G.name}]: switched off\n"

    if all(value == G.pmls["thickness"]["x0"] for value in G.pmls["thickness"].values()):
        pmlinfo = str(G.pmls["thickness"]["x0"])
    else:
        pmlinfo = ""
        for key, value in G.pmls["thickness"].items():
            pmlinfo += f"{key}: {value}, "
        pmlinfo = pmlinfo[:-2]

    return (
        f"PML boundaries [{G.name}]: {{formulation: {G.pmls['formulation']}, "
        f"order: {len(G.pmls['cfs'])}, thickness (cells): {pmlinfo}}}\n"
    )
