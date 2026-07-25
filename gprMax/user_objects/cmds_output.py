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
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.model import Model
from gprMax.ntff.evaluator import spherical_observation_points
from gprMax.ntff.frequency_domain import validate_nyquist_frequencies
from gprMax.ntff.interface import (
    CARTESIAN_OUTPUTS,
    FAR_METRICS,
    SPHERICAL_OUTPUTS,
    TIME_ORIGINS,
    WINDOWS,
    KSIRFarFieldRequestSpec,
    KSIRFrequencyRequestSpec,
    KSIRFrequencyTransformSpec,
    KSIRSurfaceSpec,
    KSIRTimeRequestSpec,
    component_dependencies,
    surface_reference_origin,
    validate_identifier,
)
from gprMax.snapshots import Snapshot as SnapshotUser
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.user_objects.user_objects import OutputUserObject
from gprMax.utilities.utilities import round_int

logger = logging.getLogger(__name__)


class Snapshot(OutputUserObject):
    """Obtains information about the electromagnetic fields within a volume
        of the model at a given time instant.

    Attributes:
        p1: tuple required to specify lower left (x,y,z) coordinates of volume
                of snapshot in metres.
        p2: tuple required to specify upper right (x,y,z) coordinates of volume
                of snapshot in metres.
        dl: tuple require to specify spatial discretisation of the snapshot
                in metres.
        filename: string required for name of the file to store snapshot.
        time/iterations: either a float for time or an int for iterations
                            must be specified for point in time at which the
                            snapshot will be taken.
        fileext: optional string to indicate type for snapshot file, either
                            '.vtkhdf' (default) or '.h5'
        outputs: optional list of outputs for receiver. It can be any
                    selection from Ex, Ey, Ez, Hx, Hy, or Hz.
    """

    @property
    def order(self):
        return 9

    @property
    def hash(self):
        return "#snapshot"

    def __init__(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        dl: Tuple[float, float, float],
        filename: str,
        fileext: Optional[str] = None,
        iterations: Optional[int] = None,
        time: Optional[float] = None,
        outputs: Optional[List[str]] = None,
    ):
        super().__init__(
            p1=p1,
            p2=p2,
            dl=dl,
            filename=filename,
            fileext=fileext,
            iterations=iterations,
            time=time,
            outputs=outputs,
        )
        self.lower_bound = p1
        self.upper_bound = p2
        self.dl = dl
        self.filename = filename
        self.file_extension = fileext
        self.iterations = iterations
        self.time = time
        self.outputs = outputs

    def _calculate_upper_bound(
        self, start: npt.NDArray[np.int32], step: npt.NDArray[np.int32], size: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.int32]:
        return start + step * np.ceil(size / step)

    def build(self, model: Model, grid: FDTDGrid):
        if isinstance(grid, SubGridBaseGrid):
            raise ValueError(f"{self.params_str()} do not add snapshots to subgrids.")

        uip = self._create_uip(grid)
        self.lower_bound = uip.resolve_inf_point(self.lower_bound, role="lower")
        self.upper_bound = uip.resolve_inf_point(self.upper_bound, role="upper")
        discretised_lower_bound, discretised_upper_bound = uip.check_output_object_bounds(
            self.lower_bound, self.upper_bound, self.params_str()
        )
        discretised_dl = uip.discretise_static_point(self.dl)

        snapshot_size = discretised_upper_bound - discretised_lower_bound

        # If p2 does not line up with the set discretisation, the actual
        # maximum element accessed in the grid will be this upper bound.
        upper_bound = self._calculate_upper_bound(
            discretised_lower_bound, discretised_dl, snapshot_size
        )

        # Each coordinate may need a different method to correct p2.
        # Therefore, this check needs to be repeated after each
        # correction has been applied.
        while any(discretised_upper_bound < upper_bound):
            try:
                grid.within_bounds(upper_bound)
                upper_bound_within_grid = True
            except ValueError:
                upper_bound_within_grid = False

            # Ideally extend p2 up to the correct upper bound. This will
            # not change the snapshot output.
            if upper_bound_within_grid:
                discretised_upper_bound = upper_bound
                upper_bound_continuous = discretised_upper_bound * grid.dl
                logger.warning(
                    f"{self.params_str()} upper bound not aligned with discretisation. Updating 'p2'"
                    f" to {upper_bound_continuous}"
                )
            # If the snapshot size cannot be increased, the
            # discretisation may need reducing. E.g. for snapshots of 2D
            # models.
            elif any(discretised_dl > snapshot_size):
                discretised_dl = np.where(
                    discretised_dl > snapshot_size, snapshot_size, discretised_dl
                )
                upper_bound = self._calculate_upper_bound(
                    discretised_lower_bound, discretised_dl, snapshot_size
                )
                dl_continuous = discretised_dl * grid.dl
                logger.warning(
                    f"{self.params_str()} current bounds and discretisation would go outside"
                    f" domain. As discretisation is larger than the snapshot size in at least one"
                    f" dimension, limiting 'dl' to {dl_continuous}"
                )
            # Otherwise, limit p2 to the discretisation step below the
            # current snapshot size. This will reduce the size of the
            # snapshot by 1 in the effected dimension(s), but avoid out
            # of memory access.
            else:
                discretised_upper_bound = np.where(
                    discretised_upper_bound < upper_bound,
                    upper_bound - discretised_dl,
                    discretised_upper_bound,
                )
                snapshot_size = discretised_upper_bound - discretised_lower_bound
                upper_bound = self._calculate_upper_bound(
                    discretised_lower_bound, discretised_dl, snapshot_size
                )
                upper_bound_continuous = discretised_upper_bound * grid.dl
                logger.warning(
                    f"{self.params_str()} current bounds and discretisation would go outside"
                    f" domain. Limiting 'p2' to {upper_bound_continuous}"
                )

                # Raise error to prevent an infinite loop. This is here
                # as a precaution, it shouldn't be needed.
                if any(discretised_upper_bound < upper_bound):
                    raise ValueError(f"{self.params_str()} invalid snapshot.")

        if any(discretised_dl < 0):
            raise ValueError(f"{self.params_str()} the step size should not be less than zero.")
        if any(discretised_dl < 1):
            raise ValueError(
                f"{self.params_str()} the step size should not be less than the spatial discretisation."
            )

        if self.iterations is not None and self.time is not None:
            logger.warning(
                f"{self.params_str()} Time and iterations were both specified, using 'iterations'"
            )

        # If number of iterations given
        if self.iterations is not None:
            if self.iterations <= 0 or self.iterations > grid.iterations:
                raise ValueError(f"{self.params_str()} time value is not valid.")

        # If time value given
        elif self.time is not None:
            if self.time > 0:
                self.iterations = round_int((self.time / grid.dt)) + 1
            else:
                raise ValueError(f"{self.params_str()} time value must be greater than zero.")

        # No iteration or time value given
        else:
            raise ValueError(f"{self} specify a time or number of iterations")

        if self.file_extension is None:
            self.file_extension = SnapshotUser.fileexts[0]
        elif self.file_extension not in SnapshotUser.fileexts:
            raise ValueError(
                f"'{self.file_extension}' is not a valid format for a snapshot file."
                f" Valid options are: {' '.join(SnapshotUser.fileexts)}."
            )

        if self.outputs is None:
            outputs = dict.fromkeys(SnapshotUser.allowableoutputs, True)
        else:
            outputs = dict.fromkeys(SnapshotUser.allowableoutputs, False)
            # Check and set output names
            for output in self.outputs:
                if output not in SnapshotUser.allowableoutputs.keys():
                    raise ValueError(
                        f"{self.params_str()} contains an output type that is not"
                        " allowable. Allowable outputs in current context are "
                        f"{', '.join(SnapshotUser.allowableoutputs.keys())}."
                    )
                else:
                    outputs[output] = True

        snapshot = model.add_snapshot(
            grid,
            discretised_lower_bound,
            discretised_upper_bound,
            discretised_dl,
            self.iterations,
            self.filename,
            self.file_extension,
            outputs,
        )

        if snapshot is not None:
            p1 = uip.round_to_grid_static_point(self.lower_bound)
            p2 = uip.round_to_grid_static_point(self.upper_bound)
            dl = uip.round_to_grid_static_point(self.dl)

            logger.info(
                f"{self.grid_name(grid)}Snapshot from"
                f" {p1[0]:g}m, {p1[1]:g}m, {p1[2]:g}m, to"
                f" {p2[0]:g}m, {p2[1]:g}m, {p2[2]:g}m, discretisation"
                f" {dl[0]:g}m, {dl[1]:g}m, {dl[2]:g}m, at"
                f" {snapshot.time * grid.dt:g} secs with field outputs"
                f" {', '.join([k for k, v in outputs.items() if v])} "
                f" and filename {snapshot.filename} will be created."
            )


def _check_ksir_interface_context(user_object, grid):
    if isinstance(grid, SubGridBaseGrid) or config.sim_config.general["subgrid"]:
        raise ValueError(f"{user_object.params_str()} does not support subgrids.")
    if config.sim_config.mpi:
        raise ValueError(f"{user_object.params_str()} does not yet support MPI.")
    if config.sim_config.general["solver"] != "cpu":
        raise ValueError(f"{user_object.params_str()} currently supports only the CPU solver.")
    if config.sim_config.args.geometry_fixed:
        raise ValueError(f"{user_object.params_str()} does not support geometry-fixed runs.")
    if config.get_model_config().mode != "3D":
        raise ValueError(f"{user_object.params_str()} currently supports only 3-D models.")


def _ksir_points(values, real_dtype, name="points"):
    points = np.asarray(values, dtype=real_dtype)
    if points.ndim == 1:
        points = points[np.newaxis, :]
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(f"{name} must have shape (npoints, 3) and be finite")
    points = np.ascontiguousarray(points)
    points.setflags(write=False)
    return points


def _ksir_spherical_coordinates(radius, theta, phi, real_dtype):
    try:
        radius_values, theta_values, phi_values = np.broadcast_arrays(
            np.asarray(radius, dtype=real_dtype),
            np.asarray(theta, dtype=real_dtype),
            np.asarray(phi, dtype=real_dtype),
        )
    except ValueError as exc:
        raise ValueError("KSIR spherical coordinates must be broadcast-compatible") from exc
    if radius_values.size == 0:
        raise ValueError("KSIR spherical coordinates must not be empty")
    if np.any(radius_values <= 0):
        raise ValueError("KSIR spherical radii must be positive")
    if np.any(theta_values < 0) or np.any(theta_values > 180):
        raise ValueError("KSIR spherical theta must lie between 0 and 180 degrees")
    return _ksir_points(
        np.stack((radius_values, theta_values, phi_values), axis=-1).reshape(-1, 3),
        real_dtype,
        "spherical coordinates",
    )


def _ksir_output_id(items, requested, prefix):
    if requested is not None:
        validate_identifier("KSIR output ID", requested)
        return requested
    used = {item.output_id for item in items}
    index = 1
    while f"{prefix}{index}" in used:
        index += 1
    return f"{prefix}{index}"


def _ksir_outputs(requested, defaults, allowed, params):
    outputs = tuple(defaults if requested is None else requested)
    if not outputs:
        raise ValueError(f"{params} must request at least one output")
    if len(set(outputs)) != len(outputs):
        raise ValueError(f"{params} outputs must not contain duplicates")
    unknown = set(outputs) - set(allowed)
    if unknown:
        raise ValueError(
            f"{params} unknown outputs {sorted(unknown)}; allowed outputs are "
            f"{', '.join(allowed)}"
        )
    component_dependencies(outputs)
    return outputs


def _ksir_array_points(p1, p2, dl, real_dtype):
    lower = np.asarray(p1, dtype=real_dtype)
    upper = np.asarray(p2, dtype=real_dtype)
    step = np.asarray(dl, dtype=real_dtype)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or step.shape != (3,)
        or not np.all(np.isfinite((lower, upper, step)))
        or np.any(upper < lower)
        or np.any(step < 0)
        or np.any((upper > lower) & (step == 0))
    ):
        raise ValueError(
            "KSIR array bounds require finite p1 <= p2 and a positive step on every varying axis"
        )
    axes = []
    tolerance = max(1e-9, 16 * np.finfo(np.dtype(real_dtype)).eps)
    for start, stop, increment in zip(lower, upper, step):
        if start == stop:
            axes.append(np.asarray((start,), dtype=real_dtype))
            continue
        quotient = (stop - start) / increment
        count = int(np.rint(quotient))
        if not np.isclose(quotient, count, rtol=tolerance, atol=tolerance):
            raise ValueError("KSIR array bounds must be an integer number of steps apart")
        axes.append(start + increment * np.arange(count + 1, dtype=real_dtype))
    mesh = np.meshgrid(*axes, indexing="ij")
    return _ksir_points(np.stack(mesh, axis=-1).reshape(-1, 3), real_dtype)


class KSIRSurface(OutputUserObject):
    """Define a reusable Yee-aligned KSIR integration surface."""

    @property
    def order(self):
        return 10

    @property
    def hash(self):
        return "#ksir_surface"

    def __init__(self, p1, p2, id: str, origin=None):
        super().__init__(p1=p1, p2=p2, id=id, origin=origin)
        self.lower_bound = p1
        self.upper_bound = p2
        self.ID = id
        self.origin = origin

    def build(self, model: Model, grid: FDTDGrid):
        _check_ksir_interface_context(self, grid)
        validate_identifier("KSIR surface ID", self.ID)
        if self.ID in grid.ksir_surface_specs:
            raise ValueError(f"{self.params_str()} surface ID is already in use")
        uip = self._create_uip(grid)
        self.lower_bound = uip.resolve_inf_point(self.lower_bound, role="lower")
        self.upper_bound = uip.resolve_inf_point(self.upper_bound, role="upper")
        lower, upper = uip.check_output_object_bounds(
            self.lower_bound, self.upper_bound, self.params_str()
        )
        origin = None
        if self.origin is not None:
            values = np.asarray(self.origin, dtype=config.sim_config.dtypes["float_or_double"])
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{self.params_str()} origin must contain 3 finite values")
            origin = tuple(float(value) for value in values)
        grid.ksir_surface_specs[self.ID] = KSIRSurfaceSpec(
            self.ID,
            tuple(int(value) for value in lower),
            tuple(int(value) for value in upper),
            origin,
        )
        logger.info(
            f"{self.grid_name(grid)}KSIR integration surface {self.ID!r} from "
            f"{tuple(self.lower_bound)}m to {tuple(self.upper_bound)}m registered."
        )


class KSIRFrequencyTransform(OutputUserObject):
    """Declare frequencies and a window for one reusable KSIR surface."""

    @property
    def order(self):
        return 11

    @property
    def hash(self):
        return "#ksir_frequency"

    def __init__(
        self,
        surface_id: str,
        id: str,
        frequencies,
        window: str = "rectangular",
        save_surface_dft: bool = True,
        plane_wave_index: Optional[int] = None,
    ):
        # Only values representable by the positional hash command belong in
        # kwargs (UserObject.__str__ uses them for command translation).
        super().__init__(
            surface_id=surface_id,
            id=id,
            frequencies=frequencies,
            window=window,
        )
        self.surface_id = surface_id
        self.ID = id
        self.frequencies = frequencies
        self.window = window
        self.save_surface_dft = save_surface_dft
        self.plane_wave_index = plane_wave_index
        self._compiled_outputs = None

    @property
    def surface_data(self):
        if self._compiled_outputs is None:
            raise RuntimeError("KSIR frequency transform has not been compiled")
        return self._compiled_outputs.transform_monitor(self.ID).surface_data

    def build(self, model: Model, grid: FDTDGrid):
        _check_ksir_interface_context(self, grid)
        validate_identifier("KSIR surface ID", self.surface_id)
        validate_identifier("KSIR transform ID", self.ID)
        if self.surface_id not in grid.ksir_surface_specs:
            raise ValueError(f"{self.params_str()} refers to unknown surface {self.surface_id!r}")
        if self.ID in grid.ksir_transform_specs:
            raise ValueError(f"{self.params_str()} transform ID is already in use")
        if self.plane_wave_index is not None and (
            not isinstance(self.plane_wave_index, (int, np.integer)) or self.plane_wave_index < 0
        ):
            raise ValueError(f"{self.params_str()} plane_wave_index must be a non-negative integer")
        values = np.asarray(self.frequencies, dtype=config.sim_config.dtypes["float_or_double"])
        if (
            values.ndim == 0
            or values.ndim > 1
            or values.size == 0
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
            or len(np.unique(values)) != values.size
        ):
            raise ValueError(
                f"{self.params_str()} frequencies must be unique, finite, non-negative values"
            )
        try:
            validate_nyquist_frequencies(self.frequencies, grid.dt)
        except ValueError as exc:
            raise ValueError(f"{self.params_str()} {exc}") from exc
        window = self.window.lower()
        if window not in WINDOWS:
            raise ValueError(f"{self.params_str()} window must be {' or '.join(WINDOWS)}")
        spec = KSIRFrequencyTransformSpec(
            self.surface_id,
            self.ID,
            tuple(float(value) for value in values),
            window,
            bool(self.save_surface_dft),
            self.plane_wave_index,
        )
        grid.ksir_transform_specs[self.ID] = spec
        grid.ksir_transform_owners[self.ID] = self
        logger.info(
            f"{self.grid_name(grid)}KSIR frequency transform {self.ID!r} on "
            f"surface {self.surface_id!r}, {values.size} frequency/frequencies, "
            f"{window} window registered."
        )


class _KSIRRequest(OutputUserObject):
    def __init__(self):
        self._compiled_outputs = None
        self._request_key = None

    @property
    def result(self):
        if self._compiled_outputs is None or self._request_key is None:
            raise RuntimeError("KSIR output has not been compiled")
        return self._compiled_outputs.result_for(self._request_key)

    def _register_owner(self, grid, key):
        self._request_key = key
        grid.ksir_request_owners[key] = self


class KSIRTimeRx(_KSIRRequest):
    """Request exact advanced-time KSIR fields at Cartesian point(s)."""

    @property
    def order(self):
        return 12

    @property
    def hash(self):
        return "#ksir_time_rx"

    def __init__(
        self,
        position,
        surface_id: str,
        id: Optional[str] = None,
        outputs=None,
        time_origin: str = "simulation",
    ):
        super().__init__()
        self.kwargs = dict(
            position=position,
            surface_id=surface_id,
            id=id,
            outputs=outputs,
            time_origin=time_origin,
        )
        self.position = position
        self.surface_id = surface_id
        self.ID = id
        self.outputs = outputs
        self.time_origin = time_origin

    def _points(self, grid):
        return _ksir_points(self.position, config.sim_config.dtypes["float_or_double"], "position")

    def build(self, model: Model, grid: FDTDGrid):
        _check_ksir_interface_context(self, grid)
        if self.surface_id not in grid.ksir_surface_specs:
            raise ValueError(f"{self.params_str()} refers to unknown surface {self.surface_id!r}")
        if self.time_origin not in TIME_ORIGINS:
            raise ValueError(f"{self.params_str()} time origin must be {' or '.join(TIME_ORIGINS)}")
        outputs = _ksir_outputs(
            self.outputs, CARTESIAN_OUTPUTS, CARTESIAN_OUTPUTS, self.params_str()
        )
        related = [item for item in grid.ksir_time_requests if item.surface_id == self.surface_id]
        output_id = _ksir_output_id(related, self.ID, "rx")
        if any(item.output_id == output_id for item in related):
            raise ValueError(f"{self.params_str()} output ID {output_id!r} is already in use")
        key = f"time:{self.surface_id}:{output_id}"
        spec = KSIRTimeRequestSpec(
            key,
            self.surface_id,
            output_id,
            self._points(grid),
            outputs,
            self.time_origin,
            "cartesian",
        )
        grid.ksir_time_requests.append(spec)
        self.ID = output_id
        self._register_owner(grid, key)


class KSIRTimeRxArray(KSIRTimeRx):
    """Request exact time-domain fields on a Cartesian point array."""

    @property
    def hash(self):
        return "#ksir_time_rx_array"

    def __init__(self, p1, p2, dl, surface_id, id=None, outputs=None, time_origin="simulation"):
        super().__init__(p1, surface_id, id=id, outputs=outputs, time_origin=time_origin)
        self.p1, self.p2, self.dl = p1, p2, dl
        self.kwargs = dict(
            p1=p1,
            p2=p2,
            dl=dl,
            surface_id=surface_id,
            id=id,
            outputs=outputs,
            time_origin=time_origin,
        )

    def _points(self, grid):
        return _ksir_array_points(
            self.p1, self.p2, self.dl, config.sim_config.dtypes["float_or_double"]
        )


class KSIRTimeRxSpherical(KSIRTimeRx):
    """Request exact advanced-time KSIR fields at spherical coordinate(s)."""

    @property
    def hash(self):
        return "#ksir_time_rx_spherical"

    def __init__(self, r, theta, phi, surface_id, id=None, outputs=None, time_origin="simulation"):
        super().__init__(
            (r, theta, phi), surface_id, id=id, outputs=outputs, time_origin=time_origin
        )
        self.r, self.theta, self.phi = r, theta, phi
        self.kwargs = dict(
            r=r,
            theta=theta,
            phi=phi,
            surface_id=surface_id,
            id=id,
            outputs=outputs,
            time_origin=time_origin,
        )

    def build(self, model: Model, grid: FDTDGrid):
        _check_ksir_interface_context(self, grid)
        if self.surface_id not in grid.ksir_surface_specs:
            raise ValueError(f"{self.params_str()} refers to unknown surface {self.surface_id!r}")
        if self.time_origin not in TIME_ORIGINS:
            raise ValueError(f"{self.params_str()} time origin must be {' or '.join(TIME_ORIGINS)}")
        outputs = _ksir_outputs(
            self.outputs, SPHERICAL_OUTPUTS, SPHERICAL_OUTPUTS, self.params_str()
        )
        dtype = config.sim_config.dtypes["float_or_double"]
        spherical = _ksir_spherical_coordinates(self.r, self.theta, self.phi, dtype)
        surface = grid.ksir_surface_specs[self.surface_id]
        origin = surface_reference_origin(surface, grid, dtype)
        points = _ksir_points(
            spherical_observation_points(
                origin, spherical[:, 0], spherical[:, 1], spherical[:, 2], degrees=True
            ),
            dtype,
        )
        related = [item for item in grid.ksir_time_requests if item.surface_id == self.surface_id]
        output_id = _ksir_output_id(related, self.ID, "rx")
        if any(item.output_id == output_id for item in related):
            raise ValueError(f"{self.params_str()} output ID {output_id!r} is already in use")
        key = f"time:{self.surface_id}:{output_id}"
        spec = KSIRTimeRequestSpec(
            key,
            self.surface_id,
            output_id,
            points,
            outputs,
            self.time_origin,
            "spherical",
            spherical,
        )
        grid.ksir_time_requests.append(spec)
        self.ID = output_id
        self._register_owner(grid, key)


class KSIRFrequencyRx(_KSIRRequest):
    """Request exact finite-distance frequency-domain Cartesian fields."""

    @property
    def order(self):
        return 13

    @property
    def hash(self):
        return "#ksir_frequency_rx"

    def __init__(self, position, transform_id, id=None, outputs=None):
        super().__init__()
        self.kwargs = dict(position=position, transform_id=transform_id, id=id, outputs=outputs)
        self.position = position
        self.transform_id = transform_id
        self.ID = id
        self.outputs = outputs

    def _points(self, grid):
        return _ksir_points(self.position, config.sim_config.dtypes["float_or_double"], "position")

    def build(self, model: Model, grid: FDTDGrid):
        _check_ksir_interface_context(self, grid)
        if self.transform_id not in grid.ksir_transform_specs:
            raise ValueError(
                f"{self.params_str()} refers to unknown transform {self.transform_id!r}"
            )
        outputs = _ksir_outputs(
            self.outputs, CARTESIAN_OUTPUTS, CARTESIAN_OUTPUTS, self.params_str()
        )
        related = [
            item for item in grid.ksir_frequency_requests if item.transform_id == self.transform_id
        ]
        output_id = _ksir_output_id(related, self.ID, "rx")
        if any(item.output_id == output_id for item in related):
            raise ValueError(f"{self.params_str()} output ID {output_id!r} is already in use")
        key = f"frequency:{self.transform_id}:{output_id}"
        spec = KSIRFrequencyRequestSpec(
            key,
            self.transform_id,
            output_id,
            self._points(grid),
            outputs,
            "cartesian",
        )
        grid.ksir_frequency_requests.append(spec)
        self.ID = output_id
        self._register_owner(grid, key)


class KSIRFrequencyRxArray(KSIRFrequencyRx):
    """Request exact frequency-domain fields on a Cartesian point array."""

    @property
    def hash(self):
        return "#ksir_frequency_rx_array"

    def __init__(self, p1, p2, dl, transform_id, id=None, outputs=None):
        super().__init__(p1, transform_id, id=id, outputs=outputs)
        self.p1, self.p2, self.dl = p1, p2, dl
        self.kwargs = dict(p1=p1, p2=p2, dl=dl, transform_id=transform_id, id=id, outputs=outputs)

    def _points(self, grid):
        return _ksir_array_points(
            self.p1, self.p2, self.dl, config.sim_config.dtypes["float_or_double"]
        )


class KSIRFrequencyRxSpherical(KSIRFrequencyRx):
    """Request exact finite-distance frequency-domain spherical fields."""

    @property
    def hash(self):
        return "#ksir_frequency_rx_spherical"

    def __init__(self, r, theta, phi, transform_id, id=None, outputs=None):
        super().__init__((r, theta, phi), transform_id, id=id, outputs=outputs)
        self.r, self.theta, self.phi = r, theta, phi
        self.kwargs = dict(
            r=r, theta=theta, phi=phi, transform_id=transform_id, id=id, outputs=outputs
        )

    def build(self, model: Model, grid: FDTDGrid):
        _check_ksir_interface_context(self, grid)
        if self.transform_id not in grid.ksir_transform_specs:
            raise ValueError(
                f"{self.params_str()} refers to unknown transform {self.transform_id!r}"
            )
        outputs = _ksir_outputs(
            self.outputs, SPHERICAL_OUTPUTS, SPHERICAL_OUTPUTS, self.params_str()
        )
        dtype = config.sim_config.dtypes["float_or_double"]
        spherical = _ksir_spherical_coordinates(self.r, self.theta, self.phi, dtype)
        transform = grid.ksir_transform_specs[self.transform_id]
        surface = grid.ksir_surface_specs[transform.surface_id]
        origin = surface_reference_origin(surface, grid, dtype)
        points = _ksir_points(
            spherical_observation_points(
                origin, spherical[:, 0], spherical[:, 1], spherical[:, 2], degrees=True
            ),
            dtype,
        )
        related = [
            item for item in grid.ksir_frequency_requests if item.transform_id == self.transform_id
        ]
        output_id = _ksir_output_id(related, self.ID, "rx")
        if any(item.output_id == output_id for item in related):
            raise ValueError(f"{self.params_str()} output ID {output_id!r} is already in use")
        key = f"frequency:{self.transform_id}:{output_id}"
        spec = KSIRFrequencyRequestSpec(
            key,
            self.transform_id,
            output_id,
            points,
            outputs,
            "spherical",
            spherical,
        )
        grid.ksir_frequency_requests.append(spec)
        self.ID = output_id
        self._register_owner(grid, key)


class KSIRFarField(_KSIRRequest):
    """Request range-normalized far fields in paired spherical directions."""

    @property
    def order(self):
        return 14

    @property
    def hash(self):
        return "#ksir_far_field"

    def __init__(self, theta, phi, transform_id, id=None, outputs=None):
        super().__init__()
        self.kwargs = dict(theta=theta, phi=phi, transform_id=transform_id, id=id, outputs=outputs)
        self.theta = theta
        self.phi = phi
        self.transform_id = transform_id
        self.ID = id
        self.outputs = outputs

    def _angles(self, dtype):
        theta, phi = np.broadcast_arrays(
            np.asarray(self.theta, dtype=dtype), np.asarray(self.phi, dtype=dtype)
        )
        if theta.size == 0 or not np.all(np.isfinite(theta)) or not np.all(np.isfinite(phi)):
            raise ValueError(f"{self.params_str()} angles must be finite and non-empty")
        if np.any(theta < 0) or np.any(theta > 180):
            raise ValueError(f"{self.params_str()} theta must lie between 0 and 180 degrees")
        return np.ascontiguousarray(theta.ravel()), np.ascontiguousarray(phi.ravel())

    def build(self, model: Model, grid: FDTDGrid):
        _check_ksir_interface_context(self, grid)
        if self.transform_id not in grid.ksir_transform_specs:
            raise ValueError(
                f"{self.params_str()} refers to unknown transform {self.transform_id!r}"
            )
        allowed = SPHERICAL_OUTPUTS + CARTESIAN_OUTPUTS + FAR_METRICS
        outputs = _ksir_outputs(self.outputs, ("Etheta", "Ephi"), allowed, self.params_str())
        theta, phi = self._angles(config.sim_config.dtypes["float_or_double"])
        theta.setflags(write=False)
        phi.setflags(write=False)
        related = [
            item for item in grid.ksir_far_field_requests if item.transform_id == self.transform_id
        ]
        output_id = _ksir_output_id(related, self.ID, "ff")
        if any(item.output_id == output_id for item in related):
            raise ValueError(f"{self.params_str()} output ID {output_id!r} is already in use")
        key = f"far:{self.transform_id}:{output_id}"
        grid.ksir_far_field_requests.append(
            KSIRFarFieldRequestSpec(key, self.transform_id, output_id, theta, phi, outputs)
        )
        self.ID = output_id
        self._register_owner(grid, key)


class KSIRFarFieldArray(KSIRFarField):
    """Request a theta/phi product grid of range-normalized far fields."""

    @property
    def hash(self):
        return "#ksir_far_field_array"

    def __init__(
        self,
        theta_start,
        theta_stop,
        theta_step,
        phi_start,
        phi_stop,
        phi_step,
        transform_id,
        id=None,
        outputs=None,
    ):
        self.theta_range = (theta_start, theta_stop, theta_step)
        self.phi_range = (phi_start, phi_stop, phi_step)
        super().__init__(theta_start, phi_start, transform_id, id=id, outputs=outputs)
        self.kwargs = dict(
            theta_start=theta_start,
            theta_stop=theta_stop,
            theta_step=theta_step,
            phi_start=phi_start,
            phi_stop=phi_stop,
            phi_step=phi_step,
            transform_id=transform_id,
            id=id,
            outputs=outputs,
        )

    @staticmethod
    def _axis(values, dtype):
        start, stop, step = (float(value) for value in values)
        if not np.all(np.isfinite((start, stop, step))) or stop < start or step <= 0:
            raise ValueError("KSIR far-field array requires start <= stop and step > 0")
        quotient = (stop - start) / step
        count = int(np.rint(quotient))
        tolerance = max(1e-9, 16 * np.finfo(np.dtype(dtype)).eps)
        if not np.isclose(quotient, count, rtol=tolerance, atol=tolerance):
            raise ValueError("KSIR far-field array range must be an integer number of steps")
        return start + step * np.arange(count + 1, dtype=dtype)

    def _angles(self, dtype):
        theta = self._axis(self.theta_range, dtype)
        phi = self._axis(self.phi_range, dtype)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        self.theta, self.phi = theta_grid.ravel(), phi_grid.ravel()
        return super()._angles(dtype)


class GeometryView(OutputUserObject):
    """Outputs to file(s) information about the geometry (mesh) of model.

    The geometry information is saved in Visual Toolkit (VTK) formats.

    Attributes:
        p1: tuple required for lower left (x,y,z) coordinates of volume of
                geometry view in metres.
        p2: tuple required for upper right (x,y,z) coordinates of volume of
                geometry view in metres.
        dl: tuple required for spatial discretisation of geometry view in metres.
        output_type: string required for per-cell 'n' (normal) or per-cell-edge
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

    def __init__(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        dl: Tuple[float, float, float],
        output_type: str,
        filename: str,
    ):
        super().__init__(p1=p1, p2=p2, dl=dl, filename=filename, output_type=output_type)
        self.lower_bound = p1
        self.upper_bound = p2
        self.dl = dl
        self.filename = filename
        self.output_type = output_type

    def build(self, model: Model, grid: FDTDGrid):
        uip = self._create_uip(grid)
        self.lower_bound = uip.resolve_inf_point(self.lower_bound, role="lower")
        self.upper_bound = uip.resolve_inf_point(self.upper_bound, role="upper")
        discretised_lower_bound, discretised_upper_bound = uip.check_output_object_bounds(
            self.lower_bound, self.upper_bound, self.params_str()
        )
        discretised_dl = uip.discretise_static_point(self.dl)

        if any(discretised_dl < 0):
            raise ValueError(f"{self.params_str()} the step size should not be less than zero.")
        if any(discretised_dl > grid.size):
            raise ValueError(
                f"{self.params_str()} the step size should be less than the domain size."
            )
        if any(discretised_dl < 1):
            raise ValueError(
                f"{self.params_str()} the step size should not be less than the spatial"
                " discretisation."
            )
        if self.output_type == "f" and any(discretised_dl != 1):
            raise ValueError(
                f"{self.params_str()} requires the spatial discretisation for the geometry view to"
                " be the same as the model for geometry view of type f (fine)."
            )

        if self.output_type == "n":
            g = model.add_geometry_view_voxels(
                grid,
                discretised_lower_bound,
                discretised_upper_bound,
                discretised_dl,
                self.filename,
            )
        elif self.output_type == "f":
            g = model.add_geometry_view_lines(
                grid,
                discretised_lower_bound,
                discretised_upper_bound,
                self.filename,
            )
        else:
            raise ValueError(
                f"{self.params_str()} requires type to be either n (normal) or f (fine)."
            )

        if g is not None:
            p1 = uip.round_to_grid_static_point(self.lower_bound)
            p2 = uip.round_to_grid_static_point(self.upper_bound)
            dl = discretised_dl * grid.dl

            logger.info(
                f"{self.grid_name(grid)}Geometry view from"
                f" {p1[0]:g}m, {p1[1]:g}m, {p1[2]:g}m,"
                f" to {p2[0]:g}m, {p2[1]:g}m, {p2[2]:g}m,"
                f" discretisation {dl[0]:g}m, {dl[1]:g}m, {dl[2]:g}m,"
                f" with filename base {g.filenamebase} created."
            )


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
        self.lower_bound = uip.resolve_inf_point(self.lower_bound, role="lower")
        self.upper_bound = uip.resolve_inf_point(self.upper_bound, role="upper")

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
