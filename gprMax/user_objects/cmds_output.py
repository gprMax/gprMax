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
    KSIRAntennaPortsSpec,
    KSIRFarFieldRequestSpec,
    KSIRFrequencyRequestSpec,
    KSIRFrequencyTransformSpec,
    KSIRTimeRequestSpec,
    NTFFFrequencyTransformSpec,
    NTFFSurfaceSpec,
    NTFFTimeFarFieldRequestSpec,
    component_dependencies,
    surface_reference_origin,
    validate_identifier,
)
from gprMax.ports import (
    DEFAULT_MINIMUM_WAVELENGTH_CELLS,
    RationalNetworkPortOutput,
    RxPortOverride,
    VoltageSourcePortMonitor,
    validate_spectrum_limit,
)
from gprMax.receivers import Rx as RxUser
from gprMax.snapshots import Snapshot as SnapshotUser
from gprMax.sources import MagneticFrillSource
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.user_objects.user_objects import OutputUserObject
from gprMax.utilities.utilities import round_int

logger = logging.getLogger(__name__)


def _reserve_mpi_port_output_id(
    grid: FDTDGrid, requested: Optional[str], owner: OutputUserObject
) -> str:
    """Reserve one output ID consistently while every MPI rank parses the scene."""

    used = grid.mpi_port_output_ids
    if requested is None:
        index = 1
        while f"port{index}" in used:
            index += 1
        output_id = f"port{index}"
    else:
        validate_identifier("port output ID", requested)
        output_id = requested
    if output_id in used:
        raise ValueError(f"port output ID {output_id!r} is already in use.")
    used.append(output_id)
    grid.mpi_port_output_owners[output_id] = owner
    return output_id


class RxPort(OutputUserObject):
    """Calculate S11 and input impedance at a one-edge voltage source.

    A finite-resistance source uses its Thevenin wave separation. A zero-
    resistance hard source uses the sampled gap voltage and an Ampere-loop
    terminal current. The voltage source owns the wave-reference impedance.

    Attributes:
        p1: Position of the voltage source in metres.
        id: Optional HDF5 port identifier.
        spectrum_limit: Minimum cells per shortest material wavelength
            (default 10), or ``"nyquist"`` for the full research spectrum.
    """

    @property
    def order(self):
        return 16

    @property
    def hash(self):
        return "#rx_port"

    def __init__(
        self,
        p1: Tuple[float, float, float],
        id: Optional[str] = None,
        spectrum_limit=DEFAULT_MINIMUM_WAVELENGTH_CELLS,
    ):
        spectrum_limit = validate_spectrum_limit(spectrum_limit)
        if id is None and spectrum_limit != DEFAULT_MINIMUM_WAVELENGTH_CELLS:
            raise ValueError(
                "RxPort requires an ID before a non-default spectrum_limit so "
                "the object has an unambiguous positional hash representation"
            )
        kwargs = dict(p1=p1, id=id)
        if spectrum_limit != DEFAULT_MINIMUM_WAVELENGTH_CELLS:
            kwargs["spectrum_limit"] = spectrum_limit
        super().__init__(**kwargs)
        self.point = p1
        self.ID = id
        self.spectrum_limit = spectrum_limit
        self._monitor = None
        # Set instead of _monitor when this RxPort ends up paired with a
        # MagneticFrillSource - its port_output object does not exist yet at
        # build() time (see build() below), so `result` must resolve it
        # lazily rather than caching a reference that doesn't exist yet.
        self._frill_source = None

    @property
    def result(self):
        if self._monitor is not None:
            if self._monitor.result is None:
                raise RuntimeError("RxPort result is not available until the model has solved")
            return self._monitor.result
        if self._frill_source is not None:
            port_output = getattr(self._frill_source, "port_output", None)
            if port_output is None or port_output.result is None:
                raise RuntimeError("RxPort result is not available until the model has solved")
            return port_output.result
        raise RuntimeError("RxPort result is not available until the model has solved")

    def _validate_context(self, grid):
        if config.sim_config.args.geometry_fixed and getattr(
            config.sim_config.study, "type", None
        ) not in ("port", "source"):
            raise ValueError(
                f"{self.params_str()} does not support geometry-fixed runs outside a "
                "PortStudy or SourceStudy."
            )
        if config.get_model_config().mode != "3D":
            raise ValueError(f"{self.params_str()} currently supports only 3-D models.")
        if config.sim_config.general["solver"] not in ("cpu", "cuda", "opencl", "metal"):
            raise ValueError(f"{self.params_str()} supports CPU, CUDA, OpenCL, and Metal solvers.")

    def build(self, model: Model, grid: FDTDGrid):
        self._validate_context(grid)
        mpi_output_id = (
            _reserve_mpi_port_output_id(grid, self.ID, self) if config.sim_config.mpi else None
        )
        uip = self._create_uip(grid)
        self.point = uip.resolve_inf_point(self.point)
        point_within_grid, coord = uip.check_src_rx_point(self.point, self.params_str())
        if not point_within_grid:
            return

        voltage_candidates = [
            source for source in grid.voltagesources if np.array_equal(source.coord, coord)
        ]
        frill_candidates = [
            source for source in grid.magneticfrillsources if np.array_equal(source.coord, coord)
        ]
        candidates = voltage_candidates + frill_candidates
        if len(candidates) != 1:
            raise ValueError(
                f"{self.params_str()} requires exactly one voltage source or "
                "magnetic frill source at grid position "
                f"{tuple(int(value) for value in coord)}; found {len(candidates)}"
            )
        source = candidates[0]
        if grid.within_pml(coord):
            raise ValueError(f"{self.params_str()} cannot be placed inside a PML.")

        if isinstance(source, MagneticFrillSource):
            self._build_for_frill_source(grid, source, coord, uip)
            return

        if not np.isfinite(source.resistance) or source.resistance < 0:
            raise ValueError(
                f"{self.params_str()} requires a finite, non-negative voltage-source resistance"
            )
        if any(monitor.source is source for monitor in grid.port_monitors):
            raise ValueError(f"{self.params_str()} source already has an RxPort output.")

        if self.ID is None:
            if mpi_output_id is not None:
                output_id = mpi_output_id
            else:
                used = {monitor.output_id for monitor in grid.port_monitors}
                index = 1
                while f"port{index}" in used:
                    index += 1
                output_id = f"port{index}"
        else:
            if mpi_output_id is not None:
                output_id = mpi_output_id
            else:
                validate_identifier("RxPort ID", self.ID)
                output_id = self.ID
        if any(monitor.output_id == output_id for monitor in grid.port_monitors):
            raise ValueError(f"{self.params_str()} output ID is already in use.")

        if (
            self.spectrum_limit != "nyquist"
            and self.spectrum_limit < DEFAULT_MINIMUM_WAVELENGTH_CELLS
        ):
            logger.warning(
                f"{self.params_str()} requests only {self.spectrum_limit:g} cells "
                "per shortest material wavelength; values below 10 may have "
                "significant spatial-dispersion error."
            )

        receiver = RxUser()
        receiver.ID = f"_rx_port_{output_id}"
        receiver.coord = np.asarray(coord, dtype=np.int32).copy()
        receiver.coordorigin = receiver.coord.copy()
        real_dtype = config.sim_config.dtypes["float_or_double"]
        receiver.outputs[f"E{source.polarisation}"] = np.zeros(grid.iterations, dtype=real_dtype)
        receiver.internal = True
        receiver.source_bound = True
        receiver.port_id = output_id
        grid.add_receiver(receiver)

        if source.resistance == 0:
            transverse_axes = {
                "x": (1, 2),
                "y": (0, 2),
                "z": (0, 1),
            }[source.polarisation]
            if any(coord[axis] == 0 for axis in transverse_axes):
                raise ValueError(
                    f"{self.params_str()} cannot calculate a hard-source current "
                    "loop on a domain-minimum transverse boundary"
                )
            # CPU and device solvers store the identical requested-only
            # Ampere-loop component at the magnetic half step.
            receiver.outputs[f"I{source.polarisation}"] = np.zeros(
                grid.iterations, dtype=real_dtype
            )

        monitor = VoltageSourcePortMonitor(
            output_id,
            source,
            receiver,
            self.spectrum_limit,
            owner=self,
        )
        grid.port_monitors.append(monitor)
        self._monitor = monitor
        position = uip.round_to_grid_static_point(self.point)
        logger.info(
            f"RxPort {output_id!r} bound to the "
            f"{source.polarisation}-polarised voltage source at "
            f"{position[0]:g}m, {position[1]:g}m, {position[2]:g}m, "
            f"reference impedance {source.reference_impedance:g} Ohms."
        )

    def _build_for_frill_source(self, grid: FDTDGrid, source, coord, uip):
        """Bind to a MagneticFrillSource's always-on automatic port output.

        Unlike a voltage source, a magnetic frill source's S11/Zin/Yin output
        is calculated automatically regardless of #rx_port (see
        prepare_magnetic_frill_ports() in gprMax/ports.py) - so this does not
        create a second, independent monitor. It can only override that
        output's spectrum_limit, via a deferred marker consumed once the
        real port_output object is constructed (it does not exist yet at
        this point in the build sequence - see gprMax/model.py).

        RxPort.__init__ requires an id whenever spectrum_limit is
        non-default (for an unambiguous positional hash representation) -
        that id is accepted here but not used to name anything: the
        automatic output already has a fixed 'frillN' identifier regardless
        of what #rx_port is called.
        """
        if getattr(source, "_rx_port_override", None) is not None:
            raise ValueError(f"{self.params_str()} source already has an RxPort output.")

        if (
            self.spectrum_limit != "nyquist"
            and self.spectrum_limit < DEFAULT_MINIMUM_WAVELENGTH_CELLS
        ):
            logger.warning(
                f"{self.params_str()} requests only {self.spectrum_limit:g} cells "
                "per shortest material wavelength; values below 10 may have "
                "significant spatial-dispersion error."
            )

        source._rx_port_override = RxPortOverride(spectrum_limit=self.spectrum_limit, owner=self)
        self._frill_source = source
        position = uip.round_to_grid_static_point(self.point)
        logger.info(
            f"RxPort spectrum_limit override bound to the magnetic frill "
            f"source at {position[0]:g}m, {position[1]:g}m, {position[2]:g}m."
        )


class NetworkPort(OutputUserObject):
    """Request port quantities for a rational-network terminal."""

    @property
    def order(self):
        return 16

    @property
    def hash(self):
        return "#network_port"

    def __init__(
        self,
        terminal_id: str,
        reference_impedance: float = 50.0,
        spectrum_limit=DEFAULT_MINIMUM_WAVELENGTH_CELLS,
    ):
        spectrum_limit = validate_spectrum_limit(spectrum_limit)
        super().__init__(
            terminal_id=terminal_id,
            reference_impedance=reference_impedance,
            spectrum_limit=spectrum_limit,
        )
        self.terminal_id = terminal_id
        self.reference_impedance = float(reference_impedance)
        self.spectrum_limit = spectrum_limit
        self._monitor = None

    @property
    def result(self):
        if self._monitor is None or self._monitor.result is None:
            raise RuntimeError("NetworkPort result is not available until the model has solved")
        return self._monitor.result

    def build(self, model: Model, grid: FDTDGrid):
        from gprMax.studies import SourceStudy

        if config.sim_config.args.geometry_fixed and not isinstance(
            config.sim_config.study, SourceStudy
        ):
            raise ValueError(f"{self.params_str()} does not support geometry-fixed runs.")
        if config.get_model_config().mode != "3D":
            raise ValueError(f"{self.params_str()} currently supports only 3-D models.")
        validate_identifier("NetworkPort terminal ID", self.terminal_id)
        if self.terminal_id not in grid.networkterminal_specs:
            raise ValueError(
                f"{self.params_str()} there is no network terminal with ID {self.terminal_id!r}."
            )
        if not np.isfinite(self.reference_impedance) or self.reference_impedance <= 0:
            raise ValueError(
                f"{self.params_str()} reference impedance must be finite and positive."
            )
        if (
            self.spectrum_limit != "nyquist"
            and self.spectrum_limit < DEFAULT_MINIMUM_WAVELENGTH_CELLS
        ):
            logger.warning(
                f"{self.params_str()} requests only {self.spectrum_limit:g} cells "
                "per shortest material wavelength; values below 10 may have "
                "significant spatial-dispersion error."
            )
        if config.sim_config.mpi:
            _reserve_mpi_port_output_id(grid, self.terminal_id, self)
        elif any(monitor.output_id == self.terminal_id for monitor in grid.port_monitors):
            raise ValueError(f"{self.params_str()} output ID is already in use.")
        terminal = next(
            (item for item in grid.networkterminals if item.ID == self.terminal_id), None
        )
        if terminal is None:
            # MPI point objects are instantiated only on their owning rank.
            if config.sim_config.mpi:
                return
            raise RuntimeError(f"{self.params_str()} terminal definition was not instantiated")
        if terminal.output is not None:
            raise ValueError(f"{self.params_str()} terminal already has a NetworkPort output.")

        monitor = RationalNetworkPortOutput(
            self.terminal_id,
            terminal,
            self.reference_impedance,
            self.spectrum_limit,
            owner=self,
        )
        terminal.output = monitor
        grid.port_monitors.append(monitor)
        self._monitor = monitor
        logger.info(
            self.grid_name(grid) + f"NetworkPort {self.terminal_id!r}: reference impedance "
            f"{self.reference_impedance:g} Ohms."
        )


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

    A snapshot added to an HSG subgrid uses that grid's spatial and temporal
    discretisation. Its output origin is expressed in global model
    coordinates.
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
    if isinstance(grid, SubGridBaseGrid):
        raise ValueError(
            f"{user_object.params_str()} must be defined on the main grid; "
            "its closed surface may enclose complete subgrids."
        )
    if config.sim_config.general["solver"] not in ("cpu", "cuda", "opencl", "metal"):
        raise ValueError(
            f"{user_object.params_str()} supports CPU, CUDA, OpenCL, and Metal solvers."
        )
    if config.sim_config.args.geometry_fixed:
        from gprMax.studies import PlaneWaveStudy, SourceStudy

        if not isinstance(config.sim_config.study, (PlaneWaveStudy, SourceStudy)):
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


class NTFFSurface(OutputUserObject):
    """Define a reusable Yee-aligned field-transformation surface."""

    @property
    def order(self):
        return 10

    @property
    def hash(self):
        return "#ntff_surface"

    def __init__(self, p1, p2, id: str, origin=None, omit_faces=()):
        if omit_faces is None:
            omit_faces = ()
        elif isinstance(omit_faces, str):
            omit_faces = (omit_faces,)
        else:
            omit_faces = tuple(omit_faces)
        super().__init__(p1=p1, p2=p2, id=id, origin=origin, omit_faces=omit_faces)
        self.lower_bound = p1
        self.upper_bound = p2
        self.ID = id
        self.origin = origin
        self.omit_faces = tuple(str(face).lower() for face in omit_faces)

    def build(self, model: Model, grid: FDTDGrid):
        _check_ksir_interface_context(self, grid)
        validate_identifier("NTFF surface ID", self.ID)
        if self.ID in grid.ntff_surface_specs:
            raise ValueError(f"{self.params_str()} surface ID is already in use")
        valid_faces = ("x0", "xmax", "y0", "ymax", "z0", "zmax")
        unknown_faces = set(self.omit_faces) - set(valid_faces)
        if unknown_faces:
            raise ValueError(
                f"{self.params_str()} omit_faces contains unknown faces "
                f"{sorted(unknown_faces)}; valid faces are {valid_faces}"
            )
        if len(set(self.omit_faces)) != len(self.omit_faces):
            raise ValueError(f"{self.params_str()} omit_faces must not contain duplicates")
        if len(self.omit_faces) == len(valid_faces):
            raise ValueError(f"{self.params_str()} must leave at least one active NTFF face")
        self.omit_faces = tuple(face for face in valid_faces if face in self.omit_faces)
        uip = self._create_uip(grid)
        self.lower_bound = uip.resolve_inf_point(self.lower_bound, role="lower")
        self.upper_bound = uip.resolve_inf_point(self.upper_bound, role="upper")
        lower, upper = uip.check_output_object_bounds(
            self.lower_bound, self.upper_bound, self.params_str()
        )
        # Main-grid user coordinates are translated into each rank's local
        # frame during MPI parsing. NTFF surfaces are global objects which
        # are partitioned only after the Yee grid has been built, so retain
        # one identical global definition on every rank.
        if hasattr(grid, "global_size"):
            lower = grid.local_to_global_coordinate(lower)
            upper = grid.local_to_global_coordinate(upper)
        origin = None
        if self.origin is not None:
            values = np.asarray(self.origin, dtype=config.sim_config.dtypes["float_or_double"])
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{self.params_str()} origin must contain 3 finite values")
            origin = tuple(float(value) for value in values)
        grid.ntff_surface_specs[self.ID] = NTFFSurfaceSpec(
            self.ID,
            tuple(int(value) for value in lower),
            tuple(int(value) for value in upper),
            origin,
            self.omit_faces,
        )
        logger.info(
            f"{self.grid_name(grid)}NTFF integration surface {self.ID!r} from "
            f"{tuple(self.lower_bound)}m to {tuple(self.upper_bound)}m registered"
            + (
                "."
                if not self.omit_faces
                else (
                    f" as an open Huygens surface using "
                    f"{len(valid_faces) - len(self.omit_faces)} active face(s) and "
                    f"omitting {self.omit_faces}."
                )
            )
        )


class KSIRFrequencyTransform(OutputUserObject):
    """Declare frequencies and a window for one reusable NTFF surface."""

    @property
    def order(self):
        return 11

    @property
    def hash(self):
        return "#ksir_frequency"

    transform_specs_attr = "ksir_transform_specs"
    transform_owners_attr = "ksir_transform_owners"
    formulation_label = "KSIR"
    spec_class = KSIRFrequencyTransformSpec

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
            raise RuntimeError("NTFF frequency transform has not been compiled")
        return self._compiled_outputs.transform_monitor(self.ID).surface_data

    def build(self, model: Model, grid: FDTDGrid):
        _check_ksir_interface_context(self, grid)
        validate_identifier("NTFF surface ID", self.surface_id)
        validate_identifier("NTFF transform ID", self.ID)
        if self.surface_id not in grid.ntff_surface_specs:
            raise ValueError(f"{self.params_str()} refers to unknown surface {self.surface_id!r}")
        if self.ID in grid.ksir_transform_specs or self.ID in grid.ntff_transform_specs:
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
        spec = self.spec_class(
            self.surface_id,
            self.ID,
            tuple(float(value) for value in values),
            window,
            bool(self.save_surface_dft),
            self.plane_wave_index,
        )
        getattr(grid, self.transform_specs_attr)[self.ID] = spec
        getattr(grid, self.transform_owners_attr)[self.ID] = self
        logger.info(
            f"{self.grid_name(grid)}{self.formulation_label} frequency transform {self.ID!r} on "
            f"surface {self.surface_id!r}, {values.size} frequency/frequencies, "
            f"{window} window registered."
        )


class NTFFFrequencyTransform(KSIRFrequencyTransform):
    """Declare a conventional equivalent-current frequency transform."""

    @property
    def hash(self):
        return "#ntff_frequency"

    transform_specs_attr = "ntff_transform_specs"
    transform_owners_attr = "ntff_transform_owners"
    formulation_label = "Equivalent-current NTFF"
    spec_class = NTFFFrequencyTransformSpec


class _NTFFRequest(OutputUserObject):
    request_owners_attr = "ksir_request_owners"

    def __init__(self):
        self._compiled_outputs = None
        self._request_key = None

    @property
    def result(self):
        if self._compiled_outputs is None or self._request_key is None:
            raise RuntimeError("NTFF output has not been compiled")
        return self._compiled_outputs.result_for(self._request_key)

    def _register_owner(self, grid, key):
        self._request_key = key
        getattr(grid, self.request_owners_attr)[key] = self


class KSIRTimeRx(_NTFFRequest):
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
        if self.surface_id not in grid.ntff_surface_specs:
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
        if self.surface_id not in grid.ntff_surface_specs:
            raise ValueError(f"{self.params_str()} refers to unknown surface {self.surface_id!r}")
        if self.time_origin not in TIME_ORIGINS:
            raise ValueError(f"{self.params_str()} time origin must be {' or '.join(TIME_ORIGINS)}")
        outputs = _ksir_outputs(
            self.outputs, SPHERICAL_OUTPUTS, SPHERICAL_OUTPUTS, self.params_str()
        )
        dtype = config.sim_config.dtypes["float_or_double"]
        spherical = _ksir_spherical_coordinates(self.r, self.theta, self.phi, dtype)
        surface = grid.ntff_surface_specs[self.surface_id]
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


class KSIRFrequencyRx(_NTFFRequest):
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
        surface = grid.ntff_surface_specs[transform.surface_id]
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


class KSIRFarField(_NTFFRequest):
    """Request range-normalized far fields in paired spherical directions.

    In addition to Cartesian and spherical field components, ``outputs`` may
    contain radiation intensity, directivity, gain, realized gain, radiation
    or total efficiency, and RCS. Directivity and efficiency use an internal
    full-sphere quadrature even when only a cut is requested. Gain and
    efficiency require a :class:`KSIRAntennaPorts` association.
    """

    @property
    def order(self):
        return 14

    @property
    def hash(self):
        return "#ksir_far_field"

    transform_specs_attr = "ksir_transform_specs"
    far_requests_attr = "ksir_far_field_requests"

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
        transform_specs = getattr(grid, self.transform_specs_attr)
        if self.transform_id not in transform_specs:
            raise ValueError(
                f"{self.params_str()} refers to unknown transform {self.transform_id!r}"
            )
        allowed = SPHERICAL_OUTPUTS + CARTESIAN_OUTPUTS + FAR_METRICS
        outputs = _ksir_outputs(self.outputs, ("Etheta", "Ephi"), allowed, self.params_str())
        theta, phi = self._angles(config.sim_config.dtypes["float_or_double"])
        theta.setflags(write=False)
        phi.setflags(write=False)
        related = [
            item
            for item in getattr(grid, self.far_requests_attr)
            if item.transform_id == self.transform_id
        ]
        output_id = _ksir_output_id(related, self.ID, "ff")
        if any(item.output_id == output_id for item in related):
            raise ValueError(f"{self.params_str()} output ID {output_id!r} is already in use")
        key = f"far:{self.transform_id}:{output_id}"
        getattr(grid, self.far_requests_attr).append(
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


class NTFFFarField(KSIRFarField):
    """Request conventional equivalent-current range-normalized far fields."""

    @property
    def hash(self):
        return "#ntff_far_field"

    transform_specs_attr = "ntff_transform_specs"
    far_requests_attr = "ntff_far_field_requests"
    request_owners_attr = "ntff_request_owners"


class NTFFFarFieldArray(KSIRFarFieldArray):
    """Request an equivalent-current theta/phi far-field product grid."""

    @property
    def hash(self):
        return "#ntff_far_field_array"

    transform_specs_attr = "ntff_transform_specs"
    far_requests_attr = "ntff_far_field_requests"
    request_owners_attr = "ntff_request_owners"


class NTFFTimeFarField(KSIRFarField):
    """Request 1997 equivalent-current time-domain far fields."""

    @property
    def hash(self):
        return "#ntff_time_far_field"

    request_owners_attr = "ntff_request_owners"

    def __init__(self, theta, phi, surface_id, id=None, outputs=None):
        super().__init__(theta, phi, surface_id, id=id, outputs=outputs)
        self.surface_id = surface_id
        self.kwargs = dict(
            theta=theta,
            phi=phi,
            surface_id=surface_id,
            id=id,
            outputs=outputs,
        )

    def build(self, model: Model, grid: FDTDGrid):
        _check_ksir_interface_context(self, grid)
        if self.surface_id not in grid.ntff_surface_specs:
            raise ValueError(f"{self.params_str()} refers to unknown surface {self.surface_id!r}")
        outputs = _ksir_outputs(
            self.outputs,
            ("Etheta", "Ephi"),
            CARTESIAN_OUTPUTS + SPHERICAL_OUTPUTS,
            self.params_str(),
        )
        theta, phi = self._angles(config.sim_config.dtypes["float_or_double"])
        theta.setflags(write=False)
        phi.setflags(write=False)
        related = [
            item for item in grid.ntff_time_far_field_requests if item.surface_id == self.surface_id
        ]
        output_id = _ksir_output_id(related, self.ID, "ff")
        if any(item.output_id == output_id for item in related):
            raise ValueError(f"{self.params_str()} output ID {output_id!r} is already in use")
        key = f"time_far:{self.surface_id}:{output_id}"
        grid.ntff_time_far_field_requests.append(
            NTFFTimeFarFieldRequestSpec(key, self.surface_id, output_id, theta, phi, outputs)
        )
        self.ID = output_id
        self._register_owner(grid, key)


class NTFFTimeFarFieldArray(NTFFTimeFarField):
    """Request a theta/phi grid of 1997 time-domain far fields."""

    @property
    def hash(self):
        return "#ntff_time_far_field_array"

    def __init__(
        self,
        theta_start,
        theta_stop,
        theta_step,
        phi_start,
        phi_stop,
        phi_step,
        surface_id,
        id=None,
        outputs=None,
    ):
        self.theta_range = (theta_start, theta_stop, theta_step)
        self.phi_range = (phi_start, phi_stop, phi_step)
        super().__init__(theta_start, phi_start, surface_id, id=id, outputs=outputs)
        self.kwargs = dict(
            theta_start=theta_start,
            theta_stop=theta_stop,
            theta_step=theta_step,
            phi_start=phi_start,
            phi_stop=phi_stop,
            phi_step=phi_step,
            surface_id=surface_id,
            id=id,
            outputs=outputs,
        )

    def _angles(self, dtype):
        theta = KSIRFarFieldArray._axis(self.theta_range, dtype)
        phi = KSIRFarFieldArray._axis(self.phi_range, dtype)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        self.theta, self.phi = theta_grid.ravel(), phi_grid.ravel()
        return KSIRFarField._angles(self, dtype)


class KSIRAntennaPorts(OutputUserObject):
    """Associate all physical antenna ports with one KSIR transform.

    Args:
        transform_id: ID of a rectangular-window
            :class:`KSIRFrequencyTransform`.
        port_ids: IDs of every physical antenna port. Voltage-source IDs come
            from coincident :class:`RxPort` objects. Transmission-line and
            magnetic-frill sources use their automatic ``tlN`` and ``frillN``
            IDs. A port on a subgrid is referenced as
            ``<subgrid ID>/<local port ID>``. Eigenmode sources are not
            compatible with the Ramahi/KSIR formulation.

    The complete set is required for an unambiguous coherent accepted-power
    balance. A source with zero waveform amplitude is still a terminated port
    and must be included.
    """

    @property
    def order(self):
        # RxPort objects use order 16. Building afterwards makes their public
        # IDs available irrespective of input-file or Scene insertion order.
        return 17

    @property
    def hash(self):
        return "#ksir_antenna_ports"

    transform_specs_attr = "ksir_transform_specs"
    antenna_specs_attr = "ksir_antenna_port_specs"
    formulation_label = "KSIR"

    def __init__(self, transform_id, port_ids):
        super().__init__(transform_id=transform_id, port_ids=port_ids)
        self.transform_id = transform_id
        self.port_ids = tuple(port_ids)

    def build(self, model: Model, grid: FDTDGrid):
        _check_ksir_interface_context(self, grid)
        if self.transform_id not in getattr(grid, self.transform_specs_attr):
            raise ValueError(
                f"{self.params_str()} refers to unknown transform {self.transform_id!r}"
            )
        if not self.port_ids:
            raise ValueError(f"{self.params_str()} requires at least one port ID")
        for port_id in self.port_ids:
            if not isinstance(port_id, str):
                validate_identifier("antenna port reference component", port_id)
                continue
            parts = port_id.split("/")
            if len(parts) not in (1, 2):
                raise ValueError(
                    "antenna port reference must be a main-grid port ID or "
                    "'<subgrid ID>/<local port ID>'"
                )
            for part in parts:
                validate_identifier("antenna port reference component", part)
        if len(set(self.port_ids)) != len(self.port_ids):
            raise ValueError(f"{self.params_str()} port IDs must not contain duplicates")
        antenna_specs = getattr(grid, self.antenna_specs_attr)
        if self.transform_id in antenna_specs:
            raise ValueError(
                f"{self.formulation_label} transform {self.transform_id!r} already has "
                "an antenna-port group"
            )

        # Subgrid objects are built after main-grid output commands. Resolve
        # and validate the complete cross-grid namespace during NTFF
        # compilation, after every subgrid child has been registered.
        antenna_specs[self.transform_id] = KSIRAntennaPortsSpec(
            self.transform_id,
            self.port_ids,
        )


class NTFFAntennaPorts(KSIRAntennaPorts):
    """Associate all physical antenna ports with an equivalent-current transform.

    In addition to conventional terminal ports, eigenmode sources use ``portN``
    for their explicit port index and eigenmode receivers use their configured
    ID. The transform frequencies must exactly match every associated modal
    port's direct-DFT bins.
    """

    @property
    def hash(self):
        return "#ntff_antenna_ports"

    transform_specs_attr = "ntff_transform_specs"
    antenna_specs_attr = "ntff_antenna_port_specs"
    formulation_label = "Equivalent-current NTFF"


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
