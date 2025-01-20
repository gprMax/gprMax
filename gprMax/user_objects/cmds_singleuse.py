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
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from gprMax import config
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.model import Model
from gprMax.pml import PML
from gprMax.user_objects.user_objects import ModelUserObject
from gprMax.utilities.host_info import set_omp_threads

logger = logging.getLogger(__name__)


class Title(ModelUserObject):
    """Title of the model.

    Attributes:
        title (str): Model title.
    """

    @property
    def order(self):
        return 1

    @property
    def hash(self):
        return "#title"

    def __init__(self, name: str):
        """Create a Title user object.

        Args:
            name: Title of the model.
        """
        super().__init__(name=name)
        self.title = name

    def build(self, model: Model):
        model.title = self.title
        logger.info(f"Model title: {model.title}")


class Discretisation(ModelUserObject):
    """Spatial discretisation of the model in the x, y, and z dimensions.

    Attributes:
        discretisation (np.array): Spatial discretisation of the model
            (x, y, z)
    """

    @property
    def order(self):
        return 2

    @property
    def hash(self):
        return "#dx_dy_dz"

    def __init__(self, p1: Tuple[float, float, float]):
        """Create a Discretisation user object.

        Args:
            p1: Spatial discretisation in the x, y, and z dimensions.
        """
        super().__init__(p1=p1)
        self.discretisation = p1

    def build(self, model: Model):
        if any(self.discretisation) <= 0:
            raise ValueError(
                f"{self} discretisation requires the spatial step to be"
                " greater than zero in all dimensions"
            )

        model.dl = np.array(self.discretisation, dtype=np.float64)
        logger.info(f"Spatial discretisation: {model.dl[0]:g} x {model.dl[1]:g} x {model.dl[2]:g}m")


class Domain(ModelUserObject):
    """Size of the model.

    Attributes:
        domain_size (tuple): Extent of the model domain (x, y, z).
    """

    @property
    def order(self):
        return 3

    @property
    def hash(self):
        return "#domain"

    def __init__(self, p1: Tuple[float, float, float]):
        """Create a Domain user object.

        Args:
            p1: Model extent in the x, y, and z dimensions.
        """
        super().__init__(p1=p1)
        self.domain_size = p1

    def build(self, model: Model):
        uip = self._create_uip(model.G)

        discretised_domain_size = uip.discretise_point(self.domain_size)

        model.set_size(discretised_domain_size)

        if model.nx == 0 or model.ny == 0 or model.nz == 0:
            raise ValueError(f"{self} requires at least one cell in every dimension")

        logger.info(
            f"Domain size: {self.domain_size[0]:g} x {self.domain_size[1]:g} x "
            + f"{self.domain_size[2]:g}m ({model.nx:d} x {model.ny:d} x {model.nz:d} = "
            + f"{(model.nx * model.ny * model.nz):g} cells)"
        )

        # Set mode and switch off appropriate PMLs for 2D models
        grid = model.G
        if model.nx == 1:
            config.get_model_config().mode = "2D TMx"
            grid.pmls["thickness"]["x0"] = 0
            grid.pmls["thickness"]["xmax"] = 0
        elif model.ny == 1:
            config.get_model_config().mode = "2D TMy"
            grid.pmls["thickness"]["y0"] = 0
            grid.pmls["thickness"]["ymax"] = 0
        elif model.nz == 1:
            config.get_model_config().mode = "2D TMz"
            grid.pmls["thickness"]["z0"] = 0
            grid.pmls["thickness"]["zmax"] = 0
        else:
            config.get_model_config().mode = "3D"

        logger.info(f"Mode: {config.get_model_config().mode}")

        # Sub-grids cannot be used with 2D models. There would typically be
        # minimal performance benefit with sub-gridding and 2D models.
        if "2D" in config.get_model_config().mode and config.sim_config.general["subgrid"]:
            raise ValueError("Sub-gridding cannot be used with 2D models")

        # Calculate time step at CFL limit
        grid.calculate_dt()

        logger.info(f"Time step (at CFL limit): {grid.dt:g} secs")


class TimeStepStabilityFactor(ModelUserObject):
    """Factor by which to reduce the time step from the CFL limit.

    Attributes:
        stability_factor (flaot): Factor to multiply time step by.
    """

    @property
    def order(self):
        return 4

    @property
    def hash(self):
        return "#time_step_stability_factor"

    def __init__(self, f: float):
        """Create a TimeStepStabilityFactor user object.

        Args:
            f: Factor to multiply the model time step by.
        """
        super().__init__(f=f)
        self.stability_factor = f

    def build(self, model: Model):
        if self.stability_factor <= 0 or self.stability_factor > 1:
            raise ValueError(
                f"{self} requires the value of the time step stability"
                " factor to be between zero and one"
            )

        model.dt_mod = self.stability_factor
        model.dt *= model.dt_mod

        logger.info(f"Time step (modified): {model.dt:g} secs")


class TimeWindow(ModelUserObject):
    """Specifies the total required simulated time.

    Either time or iterations must be specified. If both are specified,
    time takes precedence.

    Attributes:
        time: float of required simulated time in seconds.
        iterations: int of required number of iterations.
    """

    @property
    def order(self):
        return 5

    @property
    def hash(self):
        return "#time_window"

    def __init__(self, time: Optional[float] = None, iterations: Optional[int] = None):
        """Create a TimeWindow user object.

        Args:
            time: Optional simulation time in seconds. Default None.
            iterations: Optional number of iterations. Default None.
        """
        super().__init__(time=time, iterations=iterations)
        self.time = time
        self.iterations = iterations

    def build(self, model: Model):
        if self.time is not None:
            if self.time > 0:
                model.timewindow = self.time
                model.iterations = int(np.ceil(self.time / model.dt)) + 1
            else:
                raise ValueError(f"{self} must have a value greater than zero")
        elif self.iterations is not None:
            # The +/- 1 used in calculating the number of iterations is
            # to account for the fact that the solver (iterations) loop
            # runs from 0 to < G.iterations
            model.timewindow = (self.iterations - 1) * model.dt
            model.iterations = self.iterations
        else:
            raise ValueError(f"{self} specify a time or number of iterations")

        if self.time is not None and self.iterations is not None:
            logger.warning(
                f"{self.params_str()} Time and iterations were both specified, using 'time'"
            )

        logger.info(f"Time window: {model.timewindow:g} secs ({model.iterations} iterations)")


class OMPThreads(ModelUserObject):
    """Set the number of OpenMP threads to use when running the model.

    Usually this should match the number of physical CPU cores
    available.

    Attributes:
        omp_threads (int): Number of OpenMP threads.
    """

    @property
    def order(self):
        return 6

    @property
    def hash(self):
        return "#num_threads"

    def __init__(self, n: int):
        """Create an OMPThreads user object.

        Args:
            n: Number of OpenMP threads.
        """
        super().__init__(n=n)
        self.omp_threads = n

    def build(self, model: Model):
        if self.omp_threads < 1:
            raise ValueError(f"{self} requires the value to be an integer not less than one")

        config.get_model_config().ompthreads = set_omp_threads(self.omp_threads)

        logger.info(f"Simulation will use {config.get_model_config().ompthreads} OpenMP threads")


class PMLFormulation(ModelUserObject):
    """Set the formulation of the PMLs.

    Current options are to use the Higher Order RIPML (HORIPML) -
    https://doi.org/10.1109/TAP.2011.2180344, or Multipole RIPML
    (MRIPML) - https://doi.org/10.1109/TAP.2018.2823864.

    Attributes:
        formulation (str): Formulation to be used for all PMLs. Either
            'HORIPML' or 'MRIPML'.
    """

    @property
    def order(self):
        return 7

    @property
    def hash(self):
        return "#pml_formulation"

    def __init__(self, formulation: str):
        """Create a PMLFormulation user object.

        Args:
            formulation: Formulation to be used for all PMLs. Either
                'HORIPML' or 'MRIPML'.
        """
        super().__init__(formulation=formulation)
        self.formulation = formulation

    def build(self, model: Model):
        if self.formulation not in PML.formulations:
            raise ValueError(f"{self} requires the value to be one of {' '.join(PML.formulations)}")

        model.G.pmls["formulation"] = self.formulation

        logger.info(f"PML formulation set to {model.G.pmls['formulation']}")


class PMLThickness(ModelUserObject):
    """Set the thickness of the PMLs.

    The thickness can be set globally, or individually for each of the
    six sides of the model domain. Either thickness must be set, or all
    of x0, y0, z0, xmax, ymax, zmax.

    Attributes:
        thickness (int | Tuple[int]): Thickness of the PML on all 6
            sides or individual sides of the model domain.
    """

    @property
    def order(self):
        return 7

    @property
    def hash(self):
        return "#pml_cells"

    def __init__(self, thickness: Union[int, Tuple[int, int, int, int, int, int]]):
        """Create a PMLThickness user object.

        Args:
            thickness: Thickness of the PML on all 6 sides or individual
                sides of the model domain.
        """
        super().__init__(thickness=thickness)
        self.thickness = thickness

    def build(self, model: Model):
        grid = model.G

        if not (
            isinstance(self.thickness, int) or len(self.thickness) == 1 or len(self.thickness) == 6
        ):
            raise ValueError(f"{self} requires either one or six parameter(s)")

        model.G.set_pml_thickness(self.thickness)

        # Check each PML does not take up more than half the grid
        # TODO: MPI ranks not containing a PML will not throw an error
        # here.
        if (
            2 * grid.pmls["thickness"]["x0"] >= model.nx
            or 2 * grid.pmls["thickness"]["y0"] >= model.ny
            or 2 * grid.pmls["thickness"]["z0"] >= model.nz
            or 2 * grid.pmls["thickness"]["xmax"] >= model.nx
            or 2 * grid.pmls["thickness"]["ymax"] >= model.ny
            or 2 * grid.pmls["thickness"]["zmax"] >= model.nz
        ):
            raise ValueError(f"{self} has too many cells for the domain size")

        thickness = model.G.pmls["thickness"]

        logger.info(
            f"PML thickness: x0={thickness['x0']}, y0={thickness['y0']},"
            f" z0={thickness['z0']}, xmax={thickness['xmax']},"
            f" ymax={thickness['ymax']}, zmax={thickness['zmax']}"
        )


class PMLProps(ModelUserObject):
    """Specify the formulation and thickness of the PMLs.

    A PML can be set on each of the six sides of the model domain.
    Current options are to use the Higher Order RIPML (HORIPML) -
    https://doi.org/10.1109/TAP.2011.2180344, or Multipole RIPML
    (MRIPML) - https://doi.org/10.1109/TAP.2018.2823864.

    Deprecated: PMLProps is deprecated and may be removed in future
    releases of gprMax. Use the new PMLFormulation and PMLThickness
    user objects instead.

    Attributes:
        pml_formulation (PMLFormulation): User object to set the PML
            formulation.
        pml_thickness (PMLThickness): User object to set the PML
            thickness.
    """

    @property
    def order(self):
        return 7

    @property
    def hash(self):
        return "#pml_properties"

    def __init__(
        self,
        formulation: Optional[str] = None,
        thickness: Optional[int] = None,
        x0: Optional[int] = None,
        y0: Optional[int] = None,
        z0: Optional[int] = None,
        xmax: Optional[int] = None,
        ymax: Optional[int] = None,
        zmax: Optional[int] = None,
    ):
        """Create a PMLProps user object.

        If 'thickness' is set, it will take precendence over any
        individual thicknesses set. Additionally, if 'thickness' is not
        set, the individual thickness must be set for all six sides of
        the model domain.

        Deprecated: PMLProps is deprecated and may be removed in future
        releases of gprMax. Use the new PMLFormulation and PMLThickness
        user objects instead.

        Args:
            formulation (str): Formulation to be used for all PMLs. Either
                'HORIPML' or 'MRIPML'.
            thickness: Optional thickness of the PML on all 6 sides of
                the model domain. Default None.
            x0, y0, z0, xmax, ymax, zmax: Optional thickness of the PML
                on individual sides of the model domain. Default None.
        """
        super().__init__()

        logger.warning(
            "PMLProps is deprecated and may be removed in future"
            " releases of gprMax. Use the new PMLFormulation and"
            " PMLThickness user objects instead."
        )

        if formulation is not None:
            self.pml_formulation = PMLFormulation(formulation)
        else:
            self.pml_formulation = None

        if thickness is not None:
            self.pml_thickness = PMLThickness(thickness)
        elif (
            x0 is not None
            and y0 is not None
            and z0 is not None
            and xmax is not None
            and ymax is not None
            and zmax is not None
        ):
            self.pml_thickness = PMLThickness((x0, y0, z0, xmax, ymax, zmax))
        else:
            self.pml_thickness = None

        if self.pml_formulation is None and self.pml_thickness is None:
            raise ValueError(
                "Must set PML formulation or thickness. Thickness can be set by specifying all of x0, y0, z0, xmax, ymax, zmax."
            )

    def build(self, model):
        if self.pml_formulation is not None:
            self.pml_formulation.build(model)

        if self.pml_thickness is not None:
            self.pml_thickness.build(model)


class SrcSteps(ModelUserObject):
    """Move the location of all simple sources.

    Attributes:
        step_size (Tuple[float]): Increment (x, y, z) to move all
            simple sources by for each step.
    """

    @property
    def order(self):
        return 8

    @property
    def hash(self):
        return "#src_steps"

    def __init__(self, p1: Tuple[float, float, float]):
        """Create a SrcSteps user object.

        Args:
            p1: Increment (x, y, z) to move all simple sources by for
                each step.
        """
        super().__init__(p1=p1)
        self.step_size = p1

    def build(self, model: Model):
        uip = self._create_uip(model.G)
        model.srcsteps = uip.discretise_point(self.step_size)

        logger.info(
            f"Simple sources will step {model.srcsteps[0] * model.dx:g}m, "
            f"{model.srcsteps[1] * model.dy:g}m, {model.srcsteps[2] * model.dz:g}m "
            "for each model run."
        )


class RxSteps(ModelUserObject):
    """Move the location of all receivers.

    Attributes:
        step_size (Tuple[float]): Increment (x, y, z) to move all
            receivers by for each step.
    """

    @property
    def order(self):
        return 9

    @property
    def hash(self):
        return "#rx_steps"

    def __init__(self, p1: Tuple[float, float, float]):
        """Create a RxSteps user object.

        Args:
            p1: Increment (x, y, z) to move all receivers by for each
                step.
        """
        super().__init__(p1=p1)
        self.step_size = p1

    def build(self, model: Model):
        uip = self._create_uip(model.G)
        model.rxsteps = uip.discretise_point(self.step_size)

        logger.info(
            f"All receivers will step {model.rxsteps[0] * model.dx:g}m, "
            f"{model.rxsteps[1] * model.dy:g}m, {model.rxsteps[2] * model.dz:g}m "
            "for each model run."
        )


class OutputDir(ModelUserObject):
    """Set the directory where output file(s) will be stored.

    Attributes:
        output_dir (str): File path to directory.
    """

    @property
    def order(self):
        return 10

    @property
    def hash(self):
        return "#output_dir"

    def __init__(self, dir: str):
        super().__init__(dir=dir)
        self.output_dir = dir

    def build(self, model: Model):
        config.get_model_config().set_output_file_path(self.output_dir)
