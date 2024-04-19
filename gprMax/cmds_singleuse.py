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

import logging
from abc import ABC, abstractmethod

import numpy as np

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.model import Model
from gprMax.user_inputs import MainGridUserInput

from .pml import PML
from .utilities.host_info import set_omp_threads

logger = logging.getLogger(__name__)


class Properties:
    pass


class UserObjectSingle(ABC):
    """Object that can only occur a single time in a model."""

    def __init__(self, **kwargs):
        # Each single command has an order to specify the order in which
        # the commands are constructed, e.g. discretisation must be
        # created before the domain
        self.order = 0
        self.kwargs = kwargs
        self.props = Properties()
        self.autotranslate = True

        for k, v in kwargs.items():
            setattr(self.props, k, v)

    @abstractmethod
    def build(self, model: Model, uip: MainGridUserInput):
        pass

    # TODO: Check if this is actually needed
    def rotate(self, axis, angle, origin=None):
        pass


class Title(UserObjectSingle):
    """Includes a title for your model.

    Attributes:
        name: string for model title.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 1

    def build(self, G, uip):
        try:
            title = self.kwargs["name"]
            G.title = title
            logger.info(f"Model title: {G.title}")
        except KeyError:
            pass


class Discretisation(UserObjectSingle):
    """Specifies the discretization of space in the x, y, and z directions.

    Attributes:
        p1: tuple of floats to specify spatial discretisation in x, y, z direction.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 2

    def build(self, G, uip):
        try:
            G.dl = np.array(self.kwargs["p1"])
            G.dx, G.dy, G.dz = self.kwargs["p1"]
        except KeyError:
            logger.exception(f"{self.__str__()} discretisation requires a point")
            raise

        if G.dl[0] <= 0:
            logger.exception(
                f"{self.__str__()} discretisation requires the "
                f"x-direction spatial step to be greater than zero"
            )
            raise ValueError
        if G.dl[1] <= 0:
            logger.exception(
                f"{self.__str__()} discretisation requires the "
                f"y-direction spatial step to be greater than zero"
            )
            raise ValueError
        if G.dl[2] <= 0:
            logger.exception(
                f"{self.__str__()} discretisation requires the "
                f"z-direction spatial step to be greater than zero"
            )
            raise ValueError

        logger.info(f"Spatial discretisation: {G.dl[0]:g} x {G.dl[1]:g} x {G.dl[2]:g}m")


class Domain(UserObjectSingle):
    """Specifies the size of the model.

    Attributes:
        p1: tuple of floats specifying extent of model domain (x, y, z).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 3

    def build(self, G, uip):
        try:
            G.nx, G.ny, G.nz = uip.discretise_point(self.kwargs["p1"])
        except KeyError:
            logger.exception(f"{self.__str__()} please specify a point")
            raise

        if G.nx == 0 or G.ny == 0 or G.nz == 0:
            logger.exception(f"{self.__str__()} requires at least one cell in " f"every dimension")
            raise ValueError

        logger.info(
            f"Domain size: {self.kwargs['p1'][0]:g} x {self.kwargs['p1'][1]:g} x "
            + f"{self.kwargs['p1'][2]:g}m ({G.nx:d} x {G.ny:d} x {G.nz:d} = "
            + f"{(G.nx * G.ny * G.nz):g} cells)"
        )

        # Calculate time step at CFL limit; switch off appropriate PMLs for 2D
        if G.nx == 1:
            config.get_model_config().mode = "2D TMx"
            G.pmls["thickness"]["x0"] = 0
            G.pmls["thickness"]["xmax"] = 0
        elif G.ny == 1:
            config.get_model_config().mode = "2D TMy"
            G.pmls["thickness"]["y0"] = 0
            G.pmls["thickness"]["ymax"] = 0
        elif G.nz == 1:
            config.get_model_config().mode = "2D TMz"
            G.pmls["thickness"]["z0"] = 0
            G.pmls["thickness"]["zmax"] = 0
        else:
            config.get_model_config().mode = "3D"
        G.calculate_dt()

        logger.info(f"Mode: {config.get_model_config().mode}")

        # Sub-grids cannot be used with 2D models. There would typically be
        # minimal performance benefit with sub-gridding and 2D models.
        if "2D" in config.get_model_config().mode and config.sim_config.general["subgrid"]:
            logger.exception("Sub-gridding cannot be used with 2D models")
            raise ValueError

        logger.info(f"Time step (at CFL limit): {G.dt:g} secs")


class TimeStepStabilityFactor(UserObjectSingle):
    """Factor by which to reduce the time step from the CFL limit.

    Attributes:
        f: float for factor to multiply time step.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 4

    def build(self, G, uip):
        try:
            f = self.kwargs["f"]
        except KeyError:
            logger.exception(f"{self.__str__()} requires exactly one parameter")
            raise

        if f <= 0 or f > 1:
            logger.exception(
                f"{self.__str__()} requires the value of the time "
                f"step stability factor to be between zero and one"
            )
            raise ValueError

        G.dt_mod = f
        G.dt = G.dt * G.dt_mod

        logger.info(f"Time step (modified): {G.dt:g} secs")


class TimeWindow(UserObjectSingle):
    """Specifies the total required simulated time.

    Attributes:
        time: float of required simulated time in seconds.
        iterations: int of required number of iterations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 5

    def build(self, G, uip):
        # If number of iterations given
        # The +/- 1 used in calculating the number of iterations is to account for
        # the fact that the solver (iterations) loop runs from 0 to < G.iterations
        try:
            iterations = int(self.kwargs["iterations"])
            G.timewindow = (iterations - 1) * G.dt
            G.iterations = iterations
        except KeyError:
            pass

        try:
            tmp = float(self.kwargs["time"])
            if tmp > 0:
                G.timewindow = tmp
                G.iterations = int(np.ceil(tmp / G.dt)) + 1
            else:
                logger.exception(self.__str__() + " must have a value greater than zero")
                raise ValueError
        except KeyError:
            pass

        if not G.timewindow:
            logger.exception(self.__str__() + " specify a time or number of iterations")
            raise ValueError

        logger.info(f"Time window: {G.timewindow:g} secs ({G.iterations} iterations)")


class OMPThreads(UserObjectSingle):
    """Controls how many OpenMP threads (usually the number of physical CPU
        cores available) are used when running the model.

    Attributes:
        n: int for number of threads.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 6

    def build(self, G, uip):
        try:
            n = self.kwargs["n"]
        except KeyError:
            logger.exception(
                f"{self.__str__()} requires exactly one parameter "
                f"to specify the number of CPU OpenMP threads to use"
            )
            raise
        if n < 1:
            logger.exception(
                f"{self.__str__()} requires the value to be an " f"integer not less than one"
            )
            raise ValueError

        config.get_model_config().ompthreads = set_omp_threads(n)


class PMLProps(UserObjectSingle):
    """Specifies the formulation used and thickness (number of cells) of PML
        that are used on the six sides of the model domain. Current options are
        to use the Higher Order RIPML (HORIPML) - https://doi.org/10.1109/TAP.2011.2180344,
        or Multipole RIPML (MRIPML) - https://doi.org/10.1109/TAP.2018.2823864.

    Attributes:
        formulation: string specifying formulation to be used for all PMLs
                        either 'HORIPML' or 'MRIPML'.
        thickness or x0, y0, z0, xmax, ymax, zmax: ints for thickness of PML
                                                    on all 6 sides or individual
                                                    sides of the model domain.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 7

    def build(self, G, uip):
        try:
            G.pmls["formulation"] = self.kwargs["formulation"]
            if G.pmls["formulation"] not in PML.formulations:
                logger.exception(
                    self.__str__()
                    + f" requires the value to be "
                    + f"one of {' '.join(PML.formulations)}"
                )
        except KeyError:
            pass

        try:
            thickness = self.kwargs["thickness"]
            for key in G.pmls["thickness"].keys():
                G.pmls["thickness"][key] = int(thickness)

        except KeyError:
            try:
                G.pmls["thickness"]["x0"] = int(self.kwargs["x0"])
                G.pmls["thickness"]["y0"] = int(self.kwargs["y0"])
                G.pmls["thickness"]["z0"] = int(self.kwargs["z0"])
                G.pmls["thickness"]["xmax"] = int(self.kwargs["xmax"])
                G.pmls["thickness"]["ymax"] = int(self.kwargs["ymax"])
                G.pmls["thickness"]["zmax"] = int(self.kwargs["zmax"])
            except KeyError:
                logger.exception(f"{self.__str__()} requires either one or six parameter(s)")
                raise

        if (
            2 * G.pmls["thickness"]["x0"] >= G.nx
            or 2 * G.pmls["thickness"]["y0"] >= G.ny
            or 2 * G.pmls["thickness"]["z0"] >= G.nz
            or 2 * G.pmls["thickness"]["xmax"] >= G.nx
            or 2 * G.pmls["thickness"]["ymax"] >= G.ny
            or 2 * G.pmls["thickness"]["zmax"] >= G.nz
        ):
            logger.exception(f"{self.__str__()} has too many cells for the domain size")
            raise ValueError


class SrcSteps(UserObjectSingle):
    """Moves the location of all simple sources.

    Attributes:
        p1: tuple of float increments (x,y,z) to move all simple sources.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 8

    def build(self, G, uip):
        try:
            G.srcsteps = uip.discretise_point(self.kwargs["p1"])
        except KeyError:
            logger.exception(f"{self.__str__()} requires exactly three parameters")
            raise

        logger.info(
            f"Simple sources will step {G.srcsteps[0] * G.dx:g}m, "
            f"{G.srcsteps[1] * G.dy:g}m, {G.srcsteps[2] * G.dz:g}m "
            "for each model run."
        )


class RxSteps(UserObjectSingle):
    """Moves the location of all receivers.

    Attributes:
        p1: tuple of float increments (x,y,z) to move all receivers.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 9

    def build(self, G, uip):
        try:
            G.rxsteps = uip.discretise_point(self.kwargs["p1"])
        except KeyError:
            logger.exception(f"{self.__str__()} requires exactly three parameters")
            raise

        logger.info(
            f"All receivers will step {G.rxsteps[0] * G.dx:g}m, "
            f"{G.rxsteps[1] * G.dy:g}m, {G.rxsteps[2] * G.dz:g}m "
            "for each model run."
        )


class OutputDir(UserObjectSingle):
    """Controls the directory where output file(s) will be stored.

    Attributes:
        dir: string of file path to directory.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 10

    def build(self, grid, uip):
        config.get_model_config().set_output_file_path(self.kwargs["dir"])
