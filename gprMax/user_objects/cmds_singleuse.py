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
import math
from typing import Optional, Tuple, Union

import numpy as np

from gprMax import config
from gprMax.model import Model
from gprMax.pml import PML
from gprMax.user_objects.user_objects import GridUserObject, ModelUserObject
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
        discretisation (tuple): Spatial discretisation of the model
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
        if any(d <= 0 for d in self.discretisation):
            raise ValueError(
                f"{self} discretisation requires the spatial step to be"
                " greater than zero in all dimensions"
            )

        model.dl = np.array(self.discretisation, dtype=np.float64)
        logger.info(f"Spatial discretisation: {model.dl[0]:g} x {model.dl[1]:g} x {model.dl[2]:g}m")


class DomainMode(ModelUserObject):
    """Declares the domain mode for the model: 2D TM, 2D TE, or 3D.

    Optional. Only required when the model uses `inf` on the invariant
    axis of `#domain` (and consequently on that axis in position-taking
    commands) to build a 2D model without the user having to compute the
    axis extent by hand. "3D" is accepted explicitly, even though it is
    also the default with no `#domain_mode` command present, so that a
    model can be switched between 2D and 3D by changing this command's
    argument alone, without having to remove the command.

    Attributes:
        requested_mode (str): Requested domain mode, "TM", "TE" or "3D".
    """

    @property
    def order(self):
        return 2

    @property
    def hash(self):
        return "#domain_mode"

    def __init__(self, mode: str):
        """Create a DomainMode user object.

        Args:
            mode: Requested domain mode, "TM", "TE" or "3D"
                (case-insensitive).
        """
        super().__init__(mode=mode)
        self.requested_mode = mode.upper()

    def build(self, model: Model):
        if self.requested_mode not in ("TM", "TE", "3D"):
            raise ValueError(f"{self} requires the mode to be 'TM', 'TE' or '3D'")

        config.get_model_config().requested_2d_mode = self.requested_mode
        logger.info(f"Requested domain mode: {self.requested_mode}")


class MagneticAveraging(ModelUserObject):
    """Sets the mixing rule used to combine the relative magnetic
    permeability (mu_r) and magnetic loss (sigma*) of two different
    materials when a magnetic (H) field component's Yee-cell smoothing
    needs to average across a material boundary.

    Optional. Defaults to 'harmonic'. Each H component (Hx, Hy, Hz) is
    averaged from the two neighbouring cells stacked along the component's
    own axis, i.e. normal to any interface between them. The normal
    component of B is continuous across a material interface, so the
    harmonic mean of mu_r (and, for consistency, sigma*) is the physically
    correct mixing rule for this direction - unlike the tangential 4-cell
    average used for E-field smoothing, where the arithmetic mean already
    used remains correct and is unaffected by this command.

    Versions of gprMax prior to the introduction of this command always
    used a simple arithmetic mean here instead. Set this command to
    'arithmetic' to reproduce that older behaviour exactly, e.g. to
    reproduce results generated with an older version of gprMax.

    Attributes:
        mode (str): 'harmonic' (default) or 'arithmetic'.
    """

    @property
    def order(self):
        return 2

    @property
    def hash(self):
        return "#magnetic_averaging"

    def __init__(self, mode: str = "harmonic"):
        """Create a MagneticAveraging user object.

        Args:
            mode: 'harmonic' (default) or 'arithmetic' (case-insensitive).
        """
        super().__init__(mode=mode)
        self.mode = mode.lower()

    def build(self, model: Model):
        if self.mode not in ("harmonic", "arithmetic"):
            raise ValueError(f"{self} requires the mode to be 'harmonic' or 'arithmetic'")

        config.get_model_config().magnetic_averaging_mode = self.mode
        logger.info(f"Magnetic (H-field) averaging mixing rule: {self.mode}")


class DispersiveAveraging(ModelUserObject):
    """Enable or disable arithmetic interface averaging for dispersive media.

    When enabled, Debye, Lorentz, and Drude materials participate in the four-cell
    electric-edge averaging used for nondispersive dielectric materials. The
    exact effective material retains every distinct inclusive pole and scales
    each residue by its cell weight. Ordinary dielectric smoothing remains
    controlled by the ``averaging`` argument of each volumetric object.

    This option is disabled by default because the compound material can have
    more poles than its constituents, increasing model-wide dispersive memory.

    Attributes:
        enabled (bool): Whether dispersive interface averaging is enabled.
    """

    @property
    def order(self):
        return 2

    @property
    def hash(self):
        return "#dispersive_averaging"

    def __init__(self, enabled: bool = True):
        """Create a DispersiveAveraging user object.

        Args:
            enabled: ``True`` (default) to average dispersive interfaces, or
                ``False`` to retain staircased dispersive interfaces.
        """
        super().__init__(enabled=enabled)
        self.enabled = enabled

    def build(self, model: Model):
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise TypeError(f"{self} requires enabled to be True or False")

        config.get_model_config().dispersive_averaging = bool(self.enabled)
        state = "enabled" if self.enabled else "disabled"
        logger.info(f"Dispersive material averaging: {state}")


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

    _AXIS_NAMES = ("x", "y", "z")
    _CELLS_FOR_MODE = {"TM": 1, "TE": 2}

    def _resolve_inf_axis(
        self, requested_mode: Optional[str], dl: np.ndarray
    ) -> Tuple[Tuple[float, float, float], Optional[int]]:
        """Resolve at most one `inf` axis in domain_size against the
        requested 2D mode, cross-validating the two against each other.

        Returns:
            domain_size: domain_size with any `inf` axis replaced by its
                resolved physical length.
            inf_axis: index of the resolved axis, or None if domain_size
                had no `inf` axis.
        """
        inf_axes = [i for i, v in enumerate(self.domain_size) if math.isinf(v)]

        if len(inf_axes) > 1:
            raise ValueError(
                f"{self} allows 'inf' on at most one axis of the domain size, but "
                f"found it on: {', '.join(self._AXIS_NAMES[i] for i in inf_axes)}"
            )

        if inf_axes and requested_mode not in self._CELLS_FOR_MODE:
            raise ValueError(
                f"{self} uses 'inf' on the {self._AXIS_NAMES[inf_axes[0]]} axis, "
                "which requires '#domain_mode' to be set to 'TM' or 'TE' first"
            )

        if not inf_axes and requested_mode in self._CELLS_FOR_MODE:
            raise ValueError(
                f"{self} requires 'inf' on exactly one axis of the domain size "
                f"when '#domain_mode' is set to '{requested_mode}'"
            )

        if not inf_axes:
            return self.domain_size, None

        axis = inf_axes[0]
        domain_size = list(self.domain_size)
        domain_size[axis] = self._CELLS_FOR_MODE[requested_mode] * dl[axis]
        return tuple(domain_size), axis

    def build(self, model: Model):
        requested_mode = config.get_model_config().requested_2d_mode

        # `inf` with no `#domain_mode` declared at all defaults to TM (the
        # more common case for GPR models) rather than erroring - purely
        # additive: this only changes a combination that previously always
        # raised, so no working input file's behaviour changes. An explicit
        # `#domain_mode: 3D` combined with `inf` still errors below (in
        # _resolve_inf_axis) - that combination is a genuine contradiction,
        # not a case to silently paper over.
        if requested_mode is None and any(math.isinf(v) for v in self.domain_size):
            requested_mode = "TM"
            config.get_model_config().requested_2d_mode = "TM"
            logger.info(
                f"{self} uses 'inf' with no '#domain_mode' command - defaulting to TM. "
                "Add '#domain_mode: TE' explicitly if you want TE mode instead."
            )

        domain_size, inf_axis = self._resolve_inf_axis(requested_mode, model.dl)

        uip = self._create_uip(model.G)

        discretised_domain_size = uip.discretise_static_point(domain_size)

        model.set_size(discretised_domain_size)

        # `> 0` (not `!= 0`) also rejects negative cell counts (from a
        # negative raw domain dimension, e.g. p1=(-0.04, 0.04, 0.04) -
        # discretise_static_point() does no sign/bound checking of its
        # own) and NaN (comparisons with NaN are always False, so
        # `not (x > 0)` is True for NaN, unlike `x == 0`).
        if not (model.nx > 0 and model.ny > 0 and model.nz > 0):
            raise ValueError(f"{self} requires at least one cell in every dimension")

        logger.info(
            f"Domain size: {domain_size[0]:g} x {domain_size[1]:g} x "
            + f"{domain_size[2]:g}m ({model.nx:d} x {model.ny:d} x {model.nz:d} = "
            + f"{(model.cells):g} cells)"
        )

        # Set mode and switch off appropriate PMLs for 2D models
        grid = model.G
        cells = (model.nx, model.ny, model.nz)
        singleton_axes = [i for i, c in enumerate(cells) if c == 1]

        if inf_axis is not None:
            expected_cells = self._CELLS_FOR_MODE[requested_mode]
            if cells[inf_axis] != expected_cells:
                raise ValueError(
                    f"{self} resolved the {self._AXIS_NAMES[inf_axis]} axis to "
                    f"{cells[inf_axis]} cells, expected {expected_cells} for "
                    f"{requested_mode} mode - check the spatial discretisation on "
                    "that axis"
                )

            # A 2D model must have exactly one invariant axis. If some
            # OTHER axis also resolved to a single cell, the mode is
            # ambiguous - the 2D kernels only support one invariant axis.
            other_singletons = [i for i in singleton_axes if i != inf_axis]
            if other_singletons:
                raise ValueError(
                    f"{self} resolved '{requested_mode}' mode on the "
                    f"{self._AXIS_NAMES[inf_axis]} axis, but the "
                    f"{', '.join(self._AXIS_NAMES[i] for i in other_singletons)} axis "
                    "also has only 1 cell - 2D mode requires exactly one invariant "
                    "axis; check the domain size and spatial discretisation"
                )

            axis_letter = self._AXIS_NAMES[inf_axis]
            config.get_model_config().mode = f"2D {requested_mode}{axis_letter}"
            grid.pmls["thickness"][f"{axis_letter}0"] = 0
            grid.pmls["thickness"][f"{axis_letter}max"] = 0
        elif len(singleton_axes) > 1:
            # Implicit (no '#domain_mode', no 'inf') style: picking the
            # first single-cell axis via an elif chain would silently
            # ignore that a SECOND axis is also 1 cell - ambiguous, since
            # the 2D kernels assume exactly one invariant axis.
            raise ValueError(
                f"{self} domain has more than one axis with only 1 cell "
                f"({', '.join(self._AXIS_NAMES[i] for i in singleton_axes)}) - 2D mode "
                "requires exactly one invariant axis; check the domain size and "
                "spatial discretisation, or declare '#domain_mode' with 'inf' on the "
                "intended axis"
            )
        elif len(singleton_axes) == 1:
            axis = singleton_axes[0]
            axis_letter = self._AXIS_NAMES[axis]
            config.get_model_config().mode = f"2D TM{axis_letter}"
            grid.pmls["thickness"][f"{axis_letter}0"] = 0
            grid.pmls["thickness"][f"{axis_letter}max"] = 0
        else:
            config.get_model_config().mode = "3D"

        # Purely informational nudge, zero behaviour change: a model using
        # the old-style implicit 1-cell-thick-axis convention (no
        # `#domain_mode`, no `inf`) still works exactly as before, but
        # `#domain_mode` + `inf` is the more explicit, forward-compatible
        # style (and the only one that can express TE or future
        # non-implicit-size 2D-like reductions) - only fires when
        # `#domain_mode` was never declared
        # at all, not for a model that explicitly chose "3D" and happens to
        # have a 1-cell axis by coincidence.
        if requested_mode is None and inf_axis is None and config.get_model_config().mode.startswith("2D"):
            logger.info(
                f"{self} detected a 2D model from a 1-cell-thick axis (legacy, implicit "
                "style) - consider declaring '#domain_mode: TM' explicitly with 'inf' on "
                "that axis in '#domain' instead, which also supports TE mode."
            )

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
        stability_factor (float): Factor to multiply time step by.
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
        time (float | None): Simulated time in seconds.
        iterations (int | None): Number of iterations.
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
            if self.iterations <= 0:
                raise ValueError(f"{self} must have a value greater than zero")

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


class PMLFormulation(GridUserObject):
    """Set the formulation of the PMLs.

    Current options are to use the Higher Order RIPML (HORIPML) -
    https://doi.org/10.1109/TAP.2011.2180344, or Multipole RIPML
    (MRIPML) - https://doi.org/10.1109/TAP.2018.2823864.

    Attributes:
        formulation (str): Either 'HORIPML' or 'MRIPML'.
        id (str, optional): Reusable profile identifier. If omitted, set the
            global formulation used by domain PMLs and unqualified slabs.
    """

    @property
    def order(self):
        return 18

    @property
    def hash(self):
        return "#pml_formulation"

    def __init__(self, formulation: str, id: Optional[str] = None):
        """Create a PMLFormulation user object.

        Args:
            formulation: Formulation to be used for all PMLs. Either
                'HORIPML' or 'MRIPML'.
        """
        super().__init__(formulation=formulation, id=id)
        self.formulation = formulation
        self.id = id

    def build(self, target):
        if self.formulation not in PML.formulations:
            raise ValueError(f"{self} requires the value to be one of {' '.join(PML.formulations)}")

        grid = target.G if isinstance(target, Model) else target
        if self.id is None:
            if grid.pmls["global_formulation_set"]:
                raise ValueError("Only one unnamed PML formulation can be specified.")
            grid.pmls["formulation"] = self.formulation
            grid.pmls["global_formulation_set"] = True
            logger.info(f"PML formulation set to {grid.pmls['formulation']}")
            return

        if not self.id:
            raise ValueError(f"{self} id must not be empty.")
        profile = grid.pmls["profiles"].setdefault(self.id, {"formulation": None, "cfs": []})
        if profile["formulation"] is not None:
            raise ValueError(f"PML profile '{self.id}' already has a formulation.")
        profile["formulation"] = self.formulation
        logger.info(f"PML profile '{self.id}' formulation set to {self.formulation}")


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

        # A negative thickness isn't rejected here without this check -
        # FDTDGrid._build_pmls() only constructs a slab when
        # `thickness > 0`, so a negative value would silently behave
        # like 0 (no PML on that face) instead of raising an error for a
        # nonsensical request.
        thickness_values = (
            (self.thickness,) if isinstance(self.thickness, int) else tuple(self.thickness)
        )
        if any(t < 0 for t in thickness_values):
            raise ValueError(f"{self} requires the PML thickness to be zero or greater")

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
        mode = config.get_model_config().mode
        if mode.startswith("2D") and self.step_size[("xyz".index(mode[-1]))] != 0:
            raise ValueError(
                f"{self.params_str()} cannot step sources along the invariant axis in 2D "
                f"mode ('{mode}') - only run 1 (step 0) is at the interior reference layer; "
                "any later run would move sources onto the forced-dead outer wall."
            )
        model.srcsteps = uip.discretise_static_point(self.step_size)

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
        mode = config.get_model_config().mode
        if mode.startswith("2D") and self.step_size[("xyz".index(mode[-1]))] != 0:
            raise ValueError(
                f"{self.params_str()} cannot step receivers along the invariant axis in 2D "
                f"mode ('{mode}') - only run 1 (step 0) is at the interior reference layer; "
                "any later run would move receivers onto the forced-dead outer wall."
            )
        model.rxsteps = uip.discretise_static_point(self.step_size)

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
