import logging
from typing import Tuple

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.model import Model
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.user_objects.user_objects import OutputUserObject

logger = logging.getLogger(__name__)


class GeometryView(OutputUserObject):
    """Outputs to file(s) information about the geometry (mesh) of model.

    The geometry information is saved in Visual Toolkit (VTK) formats.

    Attributes:
        p1: tuple required for lower left (x,y,z) coordinates of volume of
                geometry view in metres.
        p2: tuple required for upper right (x,y,z) coordinates of volume of
                geometry view in metres.
        dl: tuple required for spatial discretisation of geometry view in metres.
        output_tuple: string required for per-cell 'n' (normal) or per-cell-edge
                        'f' (fine) geometry views.
        filename: string required for filename where geometry view will be
                    stored in the same directory as input file.
    """

    @property
    def order(self):
        return 17

    @property
    def hash(self):
        return "#geometry_view"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def geometry_view_constructor(self, output_type):
        """Selects appropriate class for geometry view dependent on geometry
        view type, i.e. normal or fine.
        """

        if output_type == "n":
            from gprMax.geometry_outputs import GeometryViewVoxels as GeometryViewUser
        else:
            from gprMax.geometry_outputs import GeometryViewLines as GeometryViewUser

        return GeometryViewUser

    def build(self, model: Model, grid: FDTDGrid):
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            dl = self.kwargs["dl"]
            output_type = self.kwargs["output_type"].lower()
            filename = self.kwargs["filename"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires exactly eleven parameters.")
            raise

        GeometryViewUser = self.geometry_view_constructor(output_type)

        uip = self._create_uip(grid)
        try:
            p3 = uip.round_to_grid_static_point(p1)
            p4 = uip.round_to_grid_static_point(p2)
            p1, p2 = uip.check_box_points(p1, p2, self.params_str())
        except ValueError:
            logger.exception(f"{self.params_str()} point is outside the domain.")
            raise
        xs, ys, zs = p1
        xf, yf, zf = p2

        dx, dy, dz = uip.discretise_static_point(dl)

        if dx < 0 or dy < 0 or dz < 0:
            logger.exception(f"{self.params_str()} the step size should not be less than zero.")
            raise ValueError
        if dx > grid.nx or dy > grid.ny or dz > grid.nz:
            logger.exception(
                f"{self.params_str()} the step size should be less than the domain size."
            )
            raise ValueError
        if dx < 1 or dy < 1 or dz < 1:
            logger.exception(
                f"{self.params_str()} the step size should not be less than the spatial discretisation."
            )
            raise ValueError
        if output_type not in ["n", "f"]:
            logger.exception(
                f"{self.params_str()} requires type to be either n (normal) or f (fine)."
            )
            raise ValueError
        if output_type == "f" and (
            dx * grid.dx != grid.dx or dy * grid.dy != grid.dy or dz * grid.dz != grid.dz
        ):
            logger.exception(
                f"{self.params_str()} requires the spatial "
                "discretisation for the geometry view to be the "
                "same as the model for geometry view of "
                "type f (fine)"
            )
            raise ValueError

        g = GeometryViewUser(xs, ys, zs, xf, yf, zf, dx, dy, dz, filename, grid)

        logger.info(
            f"{self.grid_name(grid)}Geometry view from {p3[0]:g}m, "
            f"{p3[1]:g}m, {p3[2]:g}m, to {p4[0]:g}m, {p4[1]:g}m, "
            f"{p4[2]:g}m, discretisation {dx * grid.dx:g}m, "
            f"{dy * grid.dy:g}m, {dz * grid.dz:g}m, with filename "
            f"base {g.filename} created."
        )

        model.geometryviews.append(g)


class GeometryObjectsWrite(OutputUserObject):
    """Writes geometry generated in a model to file which can be imported into
        other models.

    Attributes:
        p1: tuple required for lower left (x,y,z) coordinates of volume of
                output in metres.
        p2: tuple required for upper right (x,y,z) coordinates of volume of
                output in metres.
        filename: string required for filename where output will be stored in
                    the same directory as input file.
    """

    @property
    def order(self):
        return 18

    @property
    def hash(self):
        return "#geometry_objects_write"

    def __init__(
        self, p1: Tuple[float, float, float], p2: Tuple[float, float, float], filename: str
    ):
        super().__init__(p1=p1, p2=p2, filename=filename)
        self.lower_bound = p1
        self.upper_bound = p2
        self.basefilename = filename

    def build(self, model: Model, grid: FDTDGrid):
        if isinstance(grid, SubGridBaseGrid):
            raise ValueError(f"{self.params_str()} do not add geometry objects to subgrids.")

        uip = self._create_uip(grid)

        discretised_lower_bound, discretised_upper_bound = uip.check_output_object_bounds(
            self.lower_bound, self.upper_bound, self.params_str()
        )

        g = model.add_geometry_object(
            grid, discretised_lower_bound, discretised_upper_bound, self.basefilename
        )

        if g is not None:
            p1 = uip.round_to_grid_static_point(self.lower_bound)
            p2 = uip.round_to_grid_static_point(self.upper_bound)

            logger.info(
                f"Geometry objects in the volume from {p1[0]:g}m,"
                f" {p1[1]:g}m, {p1[2]:g}m, to {p2[0]:g}m, {p2[1]:g}m,"
                f" {p2[2]:g}m, will be written to {g.filename_hdf5},"
                f" with materials written to {g.filename_materials}"
            )
