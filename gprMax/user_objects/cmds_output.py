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
from gprMax.mode2d import mode2d_geometry
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
    NTFFLayeredBackgroundSpec,
    NTFFLayeredFrequencyTransformSpec,
    NTFFSurfaceSpec,
    NTFFTimeFarFieldRequestSpec,
    component_dependencies,
    surface_reference_origin,
    validate_identifier,
)
from gprMax.ports import (
    DEFAULT_INCIDENT_FLOOR_DB,
    DEFAULT_MINIMUM_WAVELENGTH_CELLS,
    RationalNetworkPortOutput,
    validate_spectrum_limit,
)
from gprMax.sar import RadiometrySpec, SARSpec
from gprMax.snapshots import Snapshot as SnapshotUser
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.user_objects.user_objects import OutputUserObject
from gprMax.utilities.utilities import round_int

logger = logging.getLogger(__name__)


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


class SAR(OutputUserObject):
    """Request frequency-domain SAR over one or more geometry tags.

    The electric fields are transformed on the fly at the requested
    frequencies. Results are normalised to ``target_amplitude`` using the
    spectrum of exactly one active source associated with ``waveform_id``, or
    to incident/accepted power from ``port_id``. Port-power normalisation does
    not require a waveform ID because the physical port result supplies its
    own spectral support and validity.

    Args:
        frequencies: Strictly increasing positive frequencies in Hz.
        waveform_id: Waveform attached to a source-normalised excitation.
            Required for ``"waveform"`` and ``"current_moment"``; optional
            and unused for port-power normalisation.
        tags: Geometry tag string or iterable of tag strings.
        id: Unique HDF5 output identifier.
        target_amplitude: Peak phasor amplitude represented by the normalised
            source spectrum.
        spectrum_limit: Minimum cells per shortest material wavelength
            (default 10), or ``"nyquist"`` for explicit research output.
        source_floor_db: Frequencies below this relative source-spectrum level
            are stored as invalid/NaN.
        window: ``"rectangular"`` or ``"hann"``.
        averaging_masses: Optional spatial averaging masses in kg. The
            default empty tuple writes local cell SAR only. For example,
            ``(0.001, 0.01)`` requests the standard 1 g and 10 g results.
        normalisation: ``"waveform"`` (default), ``"current_moment"`` for a
            3-D Hertzian dipole, ``"incident_flux"`` for a discrete plane
            wave, ``"incident_power"``, or ``"accepted_power"``.
        port_id: Required for either power normalisation.
        target_power: Required positive power for either power normalisation;
            watts in 3-D and watts per metre in 2-D.
        target_flux: Required positive incident plane-wave power flux in
            W/m2 for ``"incident_flux"`` normalisation.
    """

    @property
    def order(self):
        return 16

    @property
    def hash(self):
        return "#sar"

    def __init__(
        self,
        frequencies,
        waveform_id: str | None = None,
        tags=None,
        id: str = "sar1",
        target_amplitude: float = 1.0,
        spectrum_limit=DEFAULT_MINIMUM_WAVELENGTH_CELLS,
        source_floor_db: float = DEFAULT_INCIDENT_FLOOR_DB,
        window: str = "rectangular",
        averaging_masses=(),
        normalisation: str = "waveform",
        port_id: str | None = None,
        target_power: float | None = None,
        target_flux: float | None = None,
    ):
        if tags is None:
            tags = ()
        elif isinstance(tags, str):
            tags = (tags,)
        else:
            tags = tuple(tags)
        super().__init__(
            frequencies=frequencies,
            waveform_id=waveform_id,
            tags=tags,
            id=id,
            target_amplitude=target_amplitude,
            spectrum_limit=spectrum_limit,
            source_floor_db=source_floor_db,
            window=window,
            averaging_masses=averaging_masses,
            normalisation=normalisation,
            port_id=port_id,
            target_power=target_power,
            target_flux=target_flux,
        )
        self.frequencies = frequencies
        self.waveform_id = None if waveform_id is None else str(waveform_id)
        self.tags = tags
        self.ID = str(id)
        self.target_amplitude = target_amplitude
        self.spectrum_limit = spectrum_limit
        self.source_floor_db = source_floor_db
        self.window = window
        self.averaging_masses = tuple(averaging_masses)
        self.normalisation = str(normalisation)
        self.port_id = port_id
        self.target_power = target_power
        self.target_flux = target_flux
        self._monitor = None

    @property
    def result(self):
        if self._monitor is None or self._monitor.result is None:
            raise RuntimeError("SAR result is not available until the model has solved")
        return self._monitor.result

    def build(self, model: Model, grid: FDTDGrid):
        if config.sim_config.general["solver"] not in ("cpu", "cuda", "opencl", "metal"):
            raise ValueError(f"{self.params_str()} supports CPU, CUDA, OpenCL, and Metal solvers")
        geometry = mode2d_geometry(config.get_model_config().mode)
        if geometry is not None and self.averaging_masses:
            raise ValueError(
                f"{self.params_str()} spatial mass averaging is not yet available in 2-D; "
                "omit averaging_masses to request local and tag-average SAR"
            )
        validate_identifier("SAR output ID", self.ID)
        if any(spec.output_id == self.ID for spec in grid.sar_specs):
            raise ValueError(f"{self.params_str()} output ID is already in use")
        frequencies = np.asarray(self.frequencies, dtype=np.float64)
        if frequencies.ndim == 0:
            frequencies = frequencies.reshape(1)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError(f"{self.params_str()} frequencies must be a non-empty 1-D array")
        spectrum_limit = validate_spectrum_limit(self.spectrum_limit)
        grid.sar_specs.append(
            SARSpec(
                output_id=self.ID,
                frequencies=tuple(float(value) for value in frequencies),
                tags=tuple(str(tag) for tag in self.tags),
                waveform_id=self.waveform_id,
                target_amplitude=float(self.target_amplitude),
                spectrum_limit=spectrum_limit,
                source_floor_db=float(self.source_floor_db),
                window=str(self.window),
                averaging_masses=tuple(float(value) for value in self.averaging_masses),
                normalisation=self.normalisation,
                port_id=self.port_id,
                target_power=self.target_power,
                target_flux=self.target_flux,
                owner=self,
            )
        )
        logger.info(
            f"{self.grid_name(grid)}SAR output {self.ID!r} registered for "
            f"{len(frequencies)} frequency/frequencies and tag(s) {self.tags}"
        )


class Radiometry(OutputUserObject):
    """Request absorbed-power and radiometric weighting over geometry tags.

    The field collection, source normalisation, material-loss formulation,
    PML exclusion, and mesh-validity policy are identical to :class:`SAR`.
    Density is not required and no mass-based quantity is calculated.

    ``normalisation`` may be ``"waveform"``, ``"current_moment"`` for a
    3-D Hertzian dipole, ``"incident_flux"`` for a discrete plane wave, or
    ``"incident_power"``/``"accepted_power"`` for a physical port. The
    output includes absorbed-power density and its normalised weighting:
    absorbed-power fraction density for port power, absorption cross-section
    density for plane-wave flux, or absorbed power per squared native source
    amplitude for a portless local source.

    Args:
        frequencies: Strictly increasing positive frequencies in Hz.
        tags: Geometry tag string or iterable of tag strings.
        waveform_id: Required for waveform, current-moment, and incident-flux
            normalisation; unused for port-power normalisation.
        id: Unique HDF5 output identifier.
        target_amplitude: Positive native source amplitude for waveform or
            current-moment normalisation.
        spectrum_limit: Minimum cells per shortest material wavelength, or
            ``"nyquist"`` for explicit research output.
        source_floor_db: Source-support validity threshold in dB.
        window: ``"rectangular"`` or ``"hann"``.
        normalisation: One of the modes described above.
        port_id: Required for incident- or accepted-power normalisation.
        target_power: Required positive W (3-D) or W/m (2-D) for port power.
        target_flux: Required positive W/m2 for plane-wave incident flux.
    """

    @property
    def order(self):
        return 16

    @property
    def hash(self):
        return "#radiometry"

    def __init__(
        self,
        frequencies,
        tags=None,
        waveform_id: str | None = None,
        id: str = "radiometry1",
        target_amplitude: float = 1.0,
        spectrum_limit=DEFAULT_MINIMUM_WAVELENGTH_CELLS,
        source_floor_db: float = DEFAULT_INCIDENT_FLOOR_DB,
        window: str = "rectangular",
        normalisation: str = "waveform",
        port_id: str | None = None,
        target_power: float | None = None,
        target_flux: float | None = None,
    ):
        if tags is None:
            tags = ()
        elif isinstance(tags, str):
            tags = (tags,)
        else:
            tags = tuple(tags)
        super().__init__(
            frequencies=frequencies,
            tags=tags,
            waveform_id=waveform_id,
            id=id,
            target_amplitude=target_amplitude,
            spectrum_limit=spectrum_limit,
            source_floor_db=source_floor_db,
            window=window,
            normalisation=normalisation,
            port_id=port_id,
            target_power=target_power,
            target_flux=target_flux,
        )
        self.frequencies = frequencies
        self.tags = tags
        self.waveform_id = None if waveform_id is None else str(waveform_id)
        self.ID = str(id)
        self.target_amplitude = target_amplitude
        self.spectrum_limit = spectrum_limit
        self.source_floor_db = source_floor_db
        self.window = window
        self.normalisation = str(normalisation)
        self.port_id = port_id
        self.target_power = target_power
        self.target_flux = target_flux
        self._monitor = None

    @property
    def result(self):
        if self._monitor is None or self._monitor.result is None:
            raise RuntimeError("Radiometry result is not available until the model has solved")
        return self._monitor.result

    def build(self, model: Model, grid: FDTDGrid):
        if config.sim_config.general["solver"] not in ("cpu", "cuda", "opencl", "metal"):
            raise ValueError(f"{self.params_str()} supports CPU, CUDA, OpenCL, and Metal solvers")
        validate_identifier("radiometry output ID", self.ID)
        if any(spec.output_id == self.ID for spec in grid.radiometry_specs):
            raise ValueError(f"{self.params_str()} output ID is already in use")
        frequencies = np.asarray(self.frequencies, dtype=np.float64)
        if frequencies.ndim == 0:
            frequencies = frequencies.reshape(1)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError(f"{self.params_str()} frequencies must be a non-empty 1-D array")
        grid.radiometry_specs.append(
            RadiometrySpec(
                output_id=self.ID,
                frequencies=tuple(float(value) for value in frequencies),
                tags=tuple(str(tag) for tag in self.tags),
                waveform_id=self.waveform_id,
                target_amplitude=float(self.target_amplitude),
                spectrum_limit=validate_spectrum_limit(self.spectrum_limit),
                source_floor_db=float(self.source_floor_db),
                window=str(self.window),
                normalisation=self.normalisation,
                port_id=self.port_id,
                target_power=self.target_power,
                target_flux=self.target_flux,
                owner=self,
            )
        )
        logger.info(
            f"{self.grid_name(grid)}Radiometry output {self.ID!r} registered for "
            f"{len(frequencies)} frequency/frequencies and tag(s) {self.tags}"
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
        geometry = mode2d_geometry(config.get_model_config().mode)
        if geometry is not None:
            axis = geometry.invariant_axis
            live = geometry.live_index
            if discretised_lower_bound[axis] > live or discretised_upper_bound[axis] <= live:
                axis_name = geometry.invariant_axis_name
                raise ValueError(
                    f"{self.params_str()} in {geometry.mode} mode must include the live "
                    f"{axis_name}-index {live}"
                )
            # A 2-D snapshot is one physical field plane. TM uses index zero;
            # TE uses the central live plane at index one. The second TE cell
            # exists only to preserve the three-dimensional Yee staggering.
            discretised_lower_bound[axis] = live
            discretised_upper_bound[axis] = live + 1
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
            p1 = discretised_lower_bound * grid.dl
            p2 = discretised_upper_bound * grid.dl
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
        from gprMax.studies import EigenmodeStudy, PlaneWaveStudy, SourceStudy

        if not isinstance(
            config.sim_config.study,
            (EigenmodeStudy, PlaneWaveStudy, SourceStudy),
        ):
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


class NTFFLayeredBackground(OutputUserObject):
    """Declare a planar material stack for layered equivalent-current NTFF."""

    @property
    def order(self):
        return 10

    @property
    def hash(self):
        return "#ntff_layered_background"

    def __init__(self, id: str, axis: str, materials, interfaces=()):
        super().__init__(id=id, axis=axis, materials=materials, interfaces=interfaces)
        self.ID = id
        self.axis = str(axis).lower()
        self.material_ids = tuple(str(value) for value in materials)
        self.interfaces = tuple(float(value) for value in interfaces)

    def __str__(self):
        sequence = []
        for index, material_id in enumerate(self.material_ids):
            sequence.append(material_id)
            if index < len(self.interfaces):
                sequence.append(str(self.interfaces[index]))
        return f"{self.hash}: {' '.join((self.ID, self.axis, *sequence))}"

    def build(self, model: Model, grid: FDTDGrid):
        _check_ksir_interface_context(self, grid)
        validate_identifier("NTFF layered background ID", self.ID)
        if self.ID in grid.ntff_layered_background_specs:
            raise ValueError(f"{self.params_str()} background ID is already in use")
        if self.axis not in ("x", "y", "z"):
            raise ValueError(f"{self.params_str()} axis must be x, y, or z")
        if len(self.material_ids) < 1 or len(self.interfaces) != len(self.material_ids) - 1:
            raise ValueError(
                f"{self.params_str()} requires exactly one fewer interface than materials"
            )
        values = np.asarray(self.interfaces, dtype=config.sim_config.dtypes["float_or_double"])
        if values.size and (not np.all(np.isfinite(values)) or not np.all(np.diff(values) < 0)):
            raise ValueError(
                f"{self.params_str()} interfaces must be finite and strictly descending "
                f"from the positive to the negative {self.axis}-axis"
            )
        available = {material.ID for material in grid.materials}
        missing = set(self.material_ids) - available
        if missing:
            raise ValueError(f"{self.params_str()} refers to unknown materials {sorted(missing)}")
        grid.ntff_layered_background_specs[self.ID] = NTFFLayeredBackgroundSpec(
            self.ID,
            self.axis,
            tuple(float(value) for value in values),
            self.material_ids,
        )
        logger.info(
            f"{self.grid_name(grid)}Planar-layered NTFF background {self.ID!r} registered "
            f"with {len(self.material_ids)} material layer(s) normal to {self.axis}."
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
        if (
            self.ID in grid.ksir_transform_specs
            or self.ID in grid.ntff_transform_specs
            or self.ID in grid.ntff_layered_transform_specs
        ):
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
        spec = self._make_spec(
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

    def _make_spec(
        self,
        surface_id,
        transform_id,
        frequencies,
        window,
        save_surface_dft,
        plane_wave_index,
    ):
        return self.spec_class(
            surface_id,
            transform_id,
            frequencies,
            window,
            save_surface_dft,
            plane_wave_index,
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


class NTFFLayeredFrequencyTransform(NTFFFrequencyTransform):
    """Declare an equivalent-current transform for a planar layered background."""

    @property
    def hash(self):
        return "#ntff_layered_frequency"

    transform_specs_attr = "ntff_layered_transform_specs"
    transform_owners_attr = "ntff_layered_transform_owners"
    formulation_label = "Planar-layered equivalent-current NTFF"
    spec_class = NTFFLayeredFrequencyTransformSpec

    def __init__(
        self,
        surface_id: str,
        id: str,
        background_id: str,
        frequencies,
        window: str = "rectangular",
        save_surface_dft: bool = True,
        plane_wave_index: Optional[int] = None,
    ):
        super().__init__(
            surface_id,
            id,
            frequencies,
            window,
            save_surface_dft,
            plane_wave_index,
        )
        self.kwargs = dict(
            surface_id=surface_id,
            id=id,
            background_id=background_id,
            frequencies=frequencies,
            window=window,
        )
        self.background_id = background_id

    def build(self, model: Model, grid: FDTDGrid):
        validate_identifier("NTFF layered background ID", self.background_id)
        if self.background_id not in grid.ntff_layered_background_specs:
            raise ValueError(
                f"{self.params_str()} refers to unknown layered background "
                f"{self.background_id!r}"
            )
        values = np.asarray(self.frequencies)
        if np.any(values <= 0):
            raise ValueError(f"{self.params_str()} layered frequencies must be strictly positive")
        super().build(model, grid)

    def _make_spec(
        self,
        surface_id,
        transform_id,
        frequencies,
        window,
        save_surface_dft,
        plane_wave_index,
    ):
        return self.spec_class(
            surface_id=surface_id,
            transform_id=transform_id,
            frequencies=frequencies,
            window=window,
            save_surface_dft=save_surface_dft,
            plane_wave_index=plane_wave_index,
            background_id=self.background_id,
        )


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
        transform_specs = dict(getattr(grid, self.transform_specs_attr))
        if self.transform_specs_attr == "ntff_transform_specs":
            transform_specs.update(grid.ntff_layered_transform_specs)
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
    """Request equivalent-current range-normalized far fields.

    A planar-layered transform additionally accepts the grouped outputs
    ``"exterior_power"``, ``"exterior_maximum"``, and
    ``"exterior_efficiency"``.  They store positive- and negative-stack-axis
    radiation summaries without retaining the temporary full-sphere fields.
    Exterior efficiency requires an :class:`NTFFAntennaPorts` association;
    exterior power and conventional directivity do not.
    """

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
            directly from the source (explicit or automatic). Transmission-line and
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
        # Port definitions are finalised after source objects have reserved
        # their public IDs.
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
        transform_specs = dict(getattr(grid, self.transform_specs_attr))
        if self.transform_specs_attr == "ntff_transform_specs":
            transform_specs.update(grid.ntff_layered_transform_specs)
        if self.transform_id not in transform_specs:
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
    ID. Every transform frequency must be present in each associated modal
    port's direct-DFT bins; the modal DFT grid may contain additional bins.
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
                f" with materials written to {g.filename_materials}. Read it back using "
                f"material database name '{g.filename_materials.stem}'."
            )
