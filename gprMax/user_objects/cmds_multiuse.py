# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

import inspect
import logging
import math
import operator
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import interpolate

import gprMax.config as config
from gprMax.eigenmode_config import (
    EigenmodeBandpassWaveform,
    EigenmodeBandSpec,
    EigenmodePortSpec,
    VirtualWaveguideSpec,
)
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.impedance_surfaces import SurfaceImpedanceModel, is_reserved_impedance_id
from gprMax.material_database import build_material_from_spec, load_material_spec
from gprMax.materials import CrimMixture as CrimMixtureUser
from gprMax.materials import DispersiveMaterial as DispersiveMaterialUser
from gprMax.materials import ListMaterial as ListMaterialUser
from gprMax.materials import Material as MaterialUser
from gprMax.materials import PeplinskiSoil as PeplinskiSoilUser
from gprMax.materials import RangeMaterial as RangeMaterialUser
from gprMax.materials import validate_drude_pole, validate_lorentz_pole
from gprMax.network_ports import RationalNetworkModel, RationalNetworkTerminal
from gprMax.pml import CFS, CFSParameter, InternalPMLSpec
from gprMax.receivers import Rx as RxUser
from gprMax.sources import DiscretePlaneWave as DiscretePlaneWaveUser
from gprMax.sources import EigenmodeReceiver as EigenmodeReceiverUser
from gprMax.sources import EigenmodeSource as EigenmodeSourceUser
from gprMax.sources import HertzianDipole as HertzianDipoleUser
from gprMax.sources import MagneticDipole as MagneticDipoleUser
from gprMax.sources import MagneticFrillSource as MagneticFrillSourceUser
from gprMax.sources import TransmissionLine as TransmissionLineUser
from gprMax.sources import VoltageSource as VoltageSourceUser
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.surface_impedance_plotting import (
    plot_good_conductor_surface_impedance_fit,
    surface_impedance_fit_plot_path,
)
from gprMax.surface_impedance_presets import (
    DEFAULT_METAL_FIT_ORDER,
    DEFAULT_METAL_FIT_TOLERANCE,
    fit_conductivity_surface_impedance,
    fit_metal_surface_impedance,
)
from gprMax.user_objects.user_objects import GridUserObject
from gprMax.waveforms import Waveform as WaveformUser

logger = logging.getLogger(__name__)

# Keep this module importable before ``gprMax.ports``: that module imports the
# NTFF package, whose public interface in turn imports the port evaluators.
# The local value is the public default used by the port layer.
DEFAULT_PORT_SPECTRUM_LIMIT = 10.0


def _reserve_voltage_port_output_id(
    grid: FDTDGrid, requested: Optional[str], owner: GridUserObject
) -> str:
    """Reserve a voltage-port ID, including consistently on every MPI rank."""

    mpi_registry = bool(getattr(config.sim_config, "mpi", None)) or not hasattr(
        grid, "port_monitors"
    )
    if mpi_registry:
        used = grid.mpi_port_output_ids
    else:
        used = [monitor.output_id for monitor in grid.port_monitors]
    if requested is None:
        index = 1
        while f"port{index}" in used:
            index += 1
        output_id = f"port{index}"
    else:
        from gprMax.ntff.interface import validate_identifier

        validate_identifier("voltage-source port ID", requested)
        output_id = requested
    if output_id in used:
        raise ValueError(f"port output ID {output_id!r} is already in use.")
    if mpi_registry:
        used.append(output_id)
        grid.mpi_port_output_owners[output_id] = owner
    return output_id


def _hard_source_current_loop_available(
    polarisation: str, resistance: float, coord: npt.NDArray[np.int32]
) -> bool:
    """Return whether the complete transverse Ampere loop is in the grid."""

    polarisation = str(polarisation).lower()
    if resistance != 0 or polarisation not in ("x", "y", "z"):
        # Parameter validation reports an invalid polarisation later. Avoid
        # replacing that useful public error with an internal dictionary key
        # failure while deciding whether to reserve an MPI port ID.
        return True
    transverse_axes = {
        "x": (1, 2),
        "y": (0, 2),
        "z": (0, 1),
    }[polarisation]
    return not any(coord[axis] == 0 for axis in transverse_axes)


def _require_complete_source_time_window(user_object, start, stop):
    """Require conventional source start/stop limits to be supplied together."""

    if (start is None) != (stop is None):
        raise ValueError(
            f"{user_object.params_str()} start and stop times must be supplied together"
        )
    if start is not None and (not np.isfinite(start) or not np.isfinite(stop)):
        raise ValueError(f"{user_object.params_str()} start and stop times must be finite")


def _validate_dpw_precompute(user_object, precompute):
    """Reject the non-functional on-the-fly DPW source option explicitly."""

    if not isinstance(precompute, (bool, np.bool_)):
        raise ValueError(f"{user_object.params_str()} precompute must be a boolean")
    if not precompute:
        raise ValueError(
            f"{user_object.params_str()} precompute=False is not currently supported; "
            "plane-wave source histories must be precomputed"
        )


def _configure_dpw_time_window(user_object, source, grid):
    """Validate and apply an optional DPW start/stop pair."""

    start = user_object.kwargs.get("start")
    stop = user_object.kwargs.get("stop")
    _require_complete_source_time_window(user_object, start, stop)
    if start is None:
        source.start = 0
        source.stop = grid.timewindow
        return " "
    if start < 0 or stop < 0:
        raise ValueError(f"{user_object.params_str()} start and stop times must not be negative")
    if stop <= start:
        raise ValueError(f"{user_object.params_str()} source stop time must exceed its start time")
    source.start = start
    source.stop = min(stop, grid.timewindow)
    return f" start time {start:g} secs, finish time {stop:g} secs "


class ExcitationFile(GridUserObject):
    """Specify file containing amplitude values of custom waveforms.

    The file should be an ASCII file, and the custom waveform shapes can
    be used with sources in the model.

    Attributes:
        filepath (str | PathLike): Excitation file path.
        kind (int | str | None): Optional interpolation kind passed to
            scipy.interpolate.interp1d.
        fill_value (float | str | None): Optional float value or
            'extrapolate' passed to scipy.interpolate.interp1d.
    """

    @property
    def order(self):
        return 1

    @property
    def hash(self):
        return "#excitation_file"

    def __init__(
        self,
        filepath: Union[str, PathLike],
        kind: Optional[Union[int, str]] = None,
        fill_value: Optional[Union[float, str]] = None,
    ):
        """Create an ExcitationFile user object.

        Args:
            filepath: Excitation file path.
            kind: Optional interpolation kind passed to
                scipy.interpolate.interp1d. Default None.
            fill_value: Optional float value or 'extrapolate' passed to
                scipy.interpolate.interp1d. Default None.
        """
        super().__init__(filepath=filepath, kind=kind, fill_value=fill_value)
        self.filepath = filepath
        self.kind = kind
        self.fill_value = fill_value

    def build(self, grid: FDTDGrid):
        # See if file exists at specified path and if not try input file directory
        excitationfile = Path(self.filepath)
        if not excitationfile.exists():
            excitationfile = Path(config.sim_config.input_file_path.parent, excitationfile)
        if not excitationfile.is_file():
            raise FileNotFoundError(f"Excitation file {excitationfile} does not exist")

        if (self.kind is None) != (self.fill_value is None):
            raise ValueError(f"{self} requires either one or three parameter(s)")

        logger.info(self.grid_name(grid) + f"Excitation file: {excitationfile}")

        # Get waveform names
        waveformIDs = np.loadtxt(excitationfile, max_rows=1, dtype=str, ndmin=1)

        # Read all waveform values into an array
        waveformvalues = np.loadtxt(
            excitationfile,
            skiprows=1,
            dtype=config.sim_config.dtypes["float_or_double"],
            ndmin=2,
        )
        if waveformvalues.shape[0] < 2:
            raise ValueError(f"{self} requires at least two waveform samples")
        if waveformvalues.shape[1] != waveformIDs.size:
            raise ValueError(
                f"{self} header declares {waveformIDs.size} columns but the data has "
                f"{waveformvalues.shape[1]}"
            )
        if not np.all(np.isfinite(waveformvalues)):
            raise ValueError(f"{self} waveform data must contain only finite values")

        # Time array (if specified) for interpolation, otherwise use simulation time
        if waveformIDs[0].lower() == "time":
            if waveformIDs.size < 2:
                raise ValueError(f"{self} must define at least one waveform after the time column")
            waveformIDs = waveformIDs[1:]
            waveformtime = waveformvalues[:, 0]
            waveformvalues = waveformvalues[:, 1:]
            if np.any(np.diff(waveformtime) <= 0):
                raise ValueError(f"{self} time values must be strictly increasing")
            timestr = "user-defined time array"
        else:
            waveformtime = np.arange(grid.iterations, dtype=np.float64) * grid.dt
            timestr = "simulation time array"

        for i, waveformID in enumerate(waveformIDs):
            if any(x.ID == waveformID for x in grid.waveforms):
                raise ValueError(f"Waveform with ID {waveformID} already exists")
            w = WaveformUser()
            w.ID = waveformID
            w.type = "user"

            # Select correct column of waveform values depending on array shape
            singlewaveformvalues = waveformvalues[:, i]

            # Truncate waveform array if it is longer than time array
            if len(singlewaveformvalues) > len(waveformtime):
                singlewaveformvalues = singlewaveformvalues[: len(waveformtime)]
            # Zero-pad end of waveform array if it is shorter than time array
            elif len(singlewaveformvalues) < len(waveformtime):
                singlewaveformvalues = np.pad(
                    singlewaveformvalues,
                    (0, len(waveformtime) - len(singlewaveformvalues)),
                    "constant",
                    constant_values=0,
                )

            # Interpolate waveform values
            if self.kind is None and self.fill_value is None:
                w.userfunc = interpolate.interp1d(waveformtime, singlewaveformvalues)
            else:
                w.userfunc = interpolate.interp1d(
                    waveformtime, singlewaveformvalues, kind=self.kind, fill_value=self.fill_value
                )

            logger.info(
                self.grid_name(grid) + f"User waveform {w.ID} created using {timestr} and, if "
                f"required, interpolation parameters (kind: {self.kind}, "
                f"fill value: {self.fill_value})."
            )

            grid.waveforms.append(w)


class Waveform(GridUserObject):
    """Create waveform to use with sources in the model.

    Attributes:
        wave_type (str): Waveform type. Can should be one of 'gaussian',
            'gaussiandot', 'gaussiandotnorm', 'gaussiandotdot',
            'gaussiandotdotnorm', 'ricker', 'gaussianprime',
            'gaussiandoubleprime', 'sine', 'contsine'.
        amp (float): Factor to scale the maximum amplitude of the
            waveform by. (For a #hertzian_dipole the units will be Amps,
            for a #voltage_source or #transmission_line the units will
            be Volts).
        freq: Centre frequency (Hz) of the waveform. In the case of the
            Gaussian waveform it is related to the pulse width.
        id (str): Identifier of the waveform.
        user_values: Optional 1D array of amplitude values to use with
            user waveform.
        user_time: Optional 1D array of time values to use with user
            waveform.
        kind (int | str | None): Optional string or int, see
            scipy.interpolate.interp1d - https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy-interpolate-interp1d
        fill_value: Optional array or 'extrapolate', see
            scipy.interpolate.interp1d - https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy-interpolate-interp1d
    """

    @property
    def order(self):
        return 2

    @property
    def hash(self):
        return "#waveform"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            wavetype = self.kwargs["wave_type"].lower()
        except KeyError:
            logger.exception(
                f"{self.params_str()} must have one of the following types {','.join(WaveformUser.types)}."
            )
            raise
        if wavetype not in WaveformUser.types:
            logger.exception(
                f"{self.params_str()} must have one of the following types {','.join(WaveformUser.types)}."
            )
            raise ValueError

        if wavetype != "user":
            try:
                amp = self.kwargs["amp"]
                freq = self.kwargs["freq"]
                ID = self.kwargs["id"]
            except KeyError:
                logger.exception(
                    self.params_str() + (" builtin waveforms require exactly four parameters.")
                )
                raise
            if not np.isfinite(amp):
                raise ValueError(f"{self.params_str()} amplitude scaling must be finite.")
            if not np.isfinite(freq) or freq <= 0:
                message = (
                    self.params_str()
                    + " requires a finite excitation frequency greater than zero."
                )
                logger.error(message)
                raise ValueError(message)
            if any(x.ID == ID for x in grid.waveforms):
                logger.exception(self.params_str() + (f" with ID {ID} already exists."))
                raise ValueError

            w = WaveformUser()
            w.ID = ID
            w.type = wavetype
            w.amp = amp
            w.freq = freq

            logger.info(
                self.grid_name(grid)
                + (
                    f"Waveform {w.ID} of type "
                    f"{w.type} with maximum amplitude scaling {w.amp:g}, "
                    f"frequency {w.freq:g}Hz created."
                )
            )

        else:
            try:
                ID = self.kwargs["id"]
            except KeyError:
                logger.exception(
                    self.params_str()
                    + (
                        " a user-defined waveform requires an 'id' and either "
                        "'user_func' or 'user_values'."
                    )
                )
                raise

            if any(x.ID == ID for x in grid.waveforms):
                logger.exception(self.params_str() + (f" with ID {ID} already exists."))
                raise ValueError

            if "user_func" in self.kwargs and "user_values" in self.kwargs:
                msg = (
                    self.params_str() + " a user-defined waveform requires exactly one of "
                    "'user_func' or 'user_values', not both."
                )
                logger.exception(msg)
                raise ValueError(msg)

            w = WaveformUser()
            w.ID = ID
            w.type = wavetype

            if "user_func" in self.kwargs:
                userfunc = self.kwargs["user_func"]
                if not callable(userfunc):
                    msg = self.params_str() + " 'user_func' must be a callable."
                    logger.exception(msg)
                    raise ValueError(msg)
                # Smoke-test at build time so a broken signature or a return
                # type fails fast here, before any more expensive grid
                # build work runs, rather than deep inside the per-source
                # waveform precompute loop.
                try:
                    sample = float(userfunc(0.0))
                except Exception as err:
                    msg = (
                        self.params_str() + " 'user_func' must accept a single float (time in "
                        "seconds) and return a numeric value."
                    )
                    logger.exception(msg)
                    raise ValueError(msg) from err
                if not np.isfinite(sample):
                    raise ValueError(
                        self.params_str() + " 'user_func' must return finite numeric values."
                    )
                w.userfunc = userfunc

                logger.info(
                    self.grid_name(grid)
                    + (f"Waveform {w.ID} using a user-supplied function created.")
                )

            elif "user_values" in self.kwargs:
                try:
                    uservalues = np.asarray(self.kwargs["user_values"], dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        self.params_str() + " 'user_values' must be a real numeric array."
                    ) from exc
                if uservalues.ndim != 1 or uservalues.size < 2:
                    raise ValueError(
                        self.params_str() + " 'user_values' must be a one-dimensional "
                        "array containing at least two samples."
                    )
                if not np.all(np.isfinite(uservalues)):
                    raise ValueError(
                        self.params_str() + " 'user_values' must contain only finite values."
                    )
                fullargspec = inspect.getfullargspec(interpolate.interp1d)
                kwargs = dict(zip(reversed(fullargspec.args), reversed(fullargspec.defaults)))

                if "user_time" in self.kwargs:
                    try:
                        waveformtime = np.asarray(self.kwargs["user_time"], dtype=float)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            self.params_str() + " 'user_time' must be a real numeric array."
                        ) from exc
                else:
                    waveformtime = np.arange(0, grid.timewindow + grid.dt, grid.dt)

                if waveformtime.ndim != 1 or waveformtime.size != uservalues.size:
                    raise ValueError(
                        self.params_str() + " 'user_time' and 'user_values' must be "
                        "one-dimensional arrays of the same length."
                    )
                if not np.all(np.isfinite(waveformtime)) or np.any(np.diff(waveformtime) <= 0):
                    raise ValueError(
                        self.params_str() + " 'user_time' must contain finite, strictly "
                        "increasing values."
                    )

                # Set args for interpolation if given by user
                if "kind" in self.kwargs:
                    kwargs["kind"] = self.kwargs["kind"]
                if "fill_value" in self.kwargs:
                    kwargs["fill_value"] = self.kwargs["fill_value"]

                w.userfunc = interpolate.interp1d(waveformtime, uservalues, **kwargs)

                logger.info(
                    self.grid_name(grid) + (f"Waveform {w.ID} that is user-defined created.")
                )

            else:
                msg = (
                    self.params_str() + " a user-defined waveform requires either 'user_func' "
                    "or 'user_values'."
                )
                logger.exception(msg)
                raise ValueError(msg)

        grid.waveforms.append(w)


class RationalNetwork(GridUserObject):
    """Define a reusable rational one-port admittance.

    Args:
        id: Unique network-model identifier.
        conductance: Direct conductance :math:`G` in siemens.
        capacitance: Direct capacitance :math:`C` in farads.
        poles: Sequence of poles :math:`p_m` in radians per second.
        residues: Sequence of residues :math:`r_m` paired with ``poles``.
        allow_active: Permit coefficients which do not satisfy the normal
            passive-network checks. The default is ``False``.
    """

    @property
    def order(self):
        return 2

    @property
    def hash(self):
        return "#rational_network"

    def __init__(
        self,
        id: str,
        conductance: float = 0.0,
        capacitance: float = 0.0,
        poles=(),
        residues=(),
        allow_active: bool = False,
    ):
        super().__init__(
            id=id,
            conductance=conductance,
            capacitance=capacitance,
            poles=tuple(poles),
            residues=tuple(residues),
            allow_active=allow_active,
        )
        self.ID = id
        self.conductance = conductance
        self.capacitance = capacitance
        self.poles = tuple(poles)
        self.residues = tuple(residues)
        self.allow_active = allow_active

    def build(self, grid: FDTDGrid):
        if self.ID in grid.rationalnetworkmodels:
            raise ValueError(f"{self.params_str()} network model ID is already in use")
        model = RationalNetworkModel(
            self.ID,
            self.conductance,
            self.capacitance,
            self.poles,
            self.residues,
            self.allow_active,
        )
        grid.rationalnetworkmodels[self.ID] = model
        logger.info(
            self.grid_name(grid) + f"Rational network {self.ID!r}: G={model.conductance:g} S, "
            f"C={model.capacitance:g} F, {len(model.poles)} pole(s)."
        )


class SurfaceImpedance(GridUserObject):
    """Define a reusable scalar passive surface impedance.

    Assign its ``id`` directly as the ``material_id`` of a closed,
    cell-occupying geometry object. Sheet, edge, zero-thickness, and
    directional ``material_ids`` assignments are unsupported.

    Select exactly one of a constant ``resistance``, a named bulk-metal
    ``preset``, or a user-supplied bulk-metal ``conductivity``. Fitted sources
    require ``fit_frequency_range=(fmin, fmax)`` and are converted internally
    to a passive Foster realization. ``fit_order='auto'`` tests increasing
    actual runtime pole counts and chooses the first deterministic local fit
    independently certified to reach ``fit_tolerance``. An integer asks for
    exactly that many Foster poles. The constant-resistance form is an
    idealized broadband boundary rather than a complete physical material
    model and emits a warning when built.
    """

    @property
    def order(self):
        return 2

    @property
    def hash(self):
        return "#surface_impedance"

    def __str__(self) -> str:
        """Return a valid hash-command representation of this object."""

        if self.resistance is not None:
            return f"{self.hash}: {self.ID} resistance {self.resistance:g}"

        source = (
            f"preset {self.preset}"
            if self.preset is not None
            else f"conductivity {self.conductivity:g}"
        )
        return (
            f"{self.hash}: {self.ID} {source} {self.fit_fmin_hz:g} "
            f"{self.fit_fmax_hz:g} {self.fit_order} {self.fit_tolerance:g} "
            f"{'y' if self.plot_fit else 'n'}"
        )

    def __init__(
        self,
        id: str,
        resistance: Optional[float] = None,
        *,
        preset: Optional[str] = None,
        conductivity: Optional[float] = None,
        fit_frequency_range: Optional[Tuple[float, float]] = None,
        fit_order: Union[str, int] = DEFAULT_METAL_FIT_ORDER,
        fit_tolerance: float = DEFAULT_METAL_FIT_TOLERANCE,
        plot_fit: bool = False,
    ):
        supplied_sources = sum(value is not None for value in (resistance, preset, conductivity))
        if supplied_sources != 1:
            raise ValueError(
                "SurfaceImpedance requires exactly one of resistance, preset, or conductivity"
            )
        if not isinstance(plot_fit, (bool, np.bool_)):
            raise ValueError("SurfaceImpedance plot_fit must be a boolean")

        fitted = None
        if resistance is not None:
            if fit_frequency_range is not None:
                raise ValueError(
                    "SurfaceImpedance resistance is frequency independent and does not "
                    "accept fit_frequency_range"
                )
            if fit_order != DEFAULT_METAL_FIT_ORDER:
                raise ValueError(
                    "SurfaceImpedance resistance is frequency independent and does not "
                    "accept fit_order"
                )
            if float(fit_tolerance) != DEFAULT_METAL_FIT_TOLERANCE:
                raise ValueError(
                    "SurfaceImpedance resistance is frequency independent and does not "
                    "accept fit_tolerance"
                )
            if plot_fit:
                raise ValueError(
                    "SurfaceImpedance resistance is frequency independent and has no fit to plot"
                )
            direct = float(resistance)
            if not np.isfinite(direct) or direct <= 0:
                raise ValueError("SurfaceImpedance resistance must be finite and positive")
            A = B = C = ()
            fit_fmin_hz = 0.0
            fit_fmax_hz = np.inf
            canonical_preset = None
            fitted_conductivity = None
            provenance = None
        else:
            if fit_frequency_range is None:
                raise ValueError(
                    "SurfaceImpedance preset and conductivity fits require "
                    "fit_frequency_range=(fmin, fmax)"
                )
            try:
                fit_fmin_hz, fit_fmax_hz = fit_frequency_range
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "SurfaceImpedance fit_frequency_range must contain exactly " "two frequencies"
                ) from exc
            if preset is not None:
                fitted = fit_metal_surface_impedance(
                    preset,
                    fit_fmin_hz,
                    fit_fmax_hz,
                    fit_order,
                    fit_tolerance,
                )
                canonical_preset = fitted.preset.key
                provenance = fitted.preset.source
            else:
                fitted = fit_conductivity_surface_impedance(
                    conductivity,
                    fit_fmin_hz,
                    fit_fmax_hz,
                    fit_order,
                    fit_tolerance,
                )
                canonical_preset = None
                provenance = "user-specified bulk conductivity"
            A, B, C, direct = fitted.A, fitted.B, fitted.C, fitted.D
            fit_fmin_hz, fit_fmax_hz = fitted.fmin_hz, fitted.fmax_hz
            fitted_conductivity = fitted.conductivity_s_per_m

        super().__init__(
            id=id,
            resistance=resistance,
            preset=canonical_preset,
            conductivity=fitted_conductivity,
            fit_frequency_range=(fit_fmin_hz, fit_fmax_hz) if fitted else None,
            fit_order=fit_order,
            fit_tolerance=fit_tolerance,
            plot_fit=bool(plot_fit),
        )
        self.ID = id
        self.resistance = direct if resistance is not None else None
        self.A = A
        self.B = B
        self.C = C
        self.D = direct
        self.fit_fmin_hz = fit_fmin_hz
        self.fit_fmax_hz = fit_fmax_hz
        self.fit_frequency_range = (fit_fmin_hz, fit_fmax_hz) if fitted is not None else None
        self.allow_active = False
        self.preset = canonical_preset
        self.conductivity = fitted_conductivity
        self.fit_order = fitted.requested_order if fitted is not None else None
        self.fit_pole_count = fitted.selected_pole_count if fitted is not None else 0
        self.fit_tolerance = fitted.tolerance if fitted is not None else None
        self.plot_fit = bool(plot_fit)
        self.provenance = provenance
        self.fit_max_relative_error = fitted.max_relative_error if fitted is not None else None
        self.fit_rms_relative_error = fitted.rms_relative_error if fitted is not None else None
        self.fit_meets_tolerance = fitted.meets_tolerance if fitted is not None else None
        self.fit_attempts = fitted.attempts if fitted is not None else ()
        self.fit_result = fitted

    def build(self, grid: FDTDGrid):
        if is_reserved_impedance_id(self.ID):
            raise ValueError(
                f"{self.params_str()} surface-impedance ID uses the reserved "
                f"prefix '__impedance_'"
            )
        if self.ID in grid.surface_impedance_models:
            raise ValueError(f"{self.params_str()} surface-impedance model ID is already in use")
        if any(material.ID == self.ID for material in grid.materials):
            raise ValueError(
                f"{self.params_str()} surface-impedance ID conflicts with an existing material ID"
            )
        model = SurfaceImpedanceModel(
            ID=self.ID,
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            fit_fmin_hz=self.fit_fmin_hz,
            fit_fmax_hz=self.fit_fmax_hz,
            allow_active=self.allow_active,
            preset=self.preset,
            provenance=self.provenance,
            conductivity_s_per_m=self.conductivity,
            fit_requested_order=self.fit_order,
            fit_pole_count=self.fit_pole_count if self.fit_result is not None else None,
            fit_tolerance=self.fit_tolerance,
            fit_max_relative_error=self.fit_max_relative_error,
            fit_rms_relative_error=self.fit_rms_relative_error,
            fit_method="passive-foster-bvls-v2" if self.fit_result is not None else None,
            plot_fit_in_full_run=self.plot_fit,
        )

        geometry_only = bool(config.sim_config is not None and config.sim_config.geometry_only)
        coordinator = not hasattr(grid, "is_coordinator") or grid.is_coordinator()
        if self.resistance is not None and coordinator:
            logger.warning(
                self.grid_name(grid)
                + f"Surface impedance {self.ID!r} is frequency independent and purely real. "
                "This is an idealized boundary condition, not a complete physical material "
                "model. Real conductor surfaces are dispersive and generally have a reactive "
                "component. Use a fitted metal preset or conductivity model for physically "
                "representative conductor loss."
            )
        if self.fit_result is not None and coordinator and (geometry_only or self.plot_fit):
            target = (
                f"{self.preset} preset"
                if self.preset is not None
                else f"conductivity {self.conductivity:g} S/m"
            )
            output_path = surface_impedance_fit_plot_path(
                config.get_model_config().output_file_path,
                self.ID,
            )
            plot_good_conductor_surface_impedance_fit(
                model=model,
                conductivity_s_per_m=self.conductivity,
                target_label=target,
                output_path=output_path,
                requested_order=self.fit_order,
                selected_pole_count=self.fit_pole_count,
                fit_tolerance=self.fit_tolerance,
            )
            logger.info(
                self.grid_name(grid)
                + f"Surface impedance {self.ID!r} fit plot written to {output_path}."
            )
        grid.surface_impedance_models[self.ID] = model
        description = f"D={model.D:g} Ohm, order {model.order}"
        if self.fit_order is not None:
            target = (
                f"{model.preset} preset"
                if model.preset is not None
                else f"conductivity {self.conductivity:g} S/m"
            )
            description += (
                f", {target} over {model.fit_fmin_hz:g}--"
                f"{model.fit_fmax_hz:g} Hz, maximum fit error "
                f"{100 * model.fit_max_relative_error:.3g}%"
            )
            if not self.fit_meets_tolerance:
                logger.warning(
                    self.grid_name(grid)
                    + f"Surface impedance {self.ID!r} explicit {self.fit_pole_count}-pole "
                    "fit missed tolerance "
                    f"{self.fit_tolerance:g}; maximum relative error is "
                    f"{self.fit_max_relative_error:g}."
                )
        logger.info(self.grid_name(grid) + f"Surface impedance {self.ID!r}: {description}.")


class NetworkTerminal(GridUserObject):
    """Connect a rational network model to one electric Yee edge.

    Args:
        p1: Physical ``(x, y, z)`` position of the electric edge.
        polarisation: Electric-edge direction, ``x``, ``y``, or ``z``.
        network_id: Identifier of a preceding :class:`RationalNetwork`.
        id: Unique terminal and HDF5 port identifier.
    """

    @property
    def order(self):
        return 6

    @property
    def hash(self):
        return "#network_terminal"

    def __init__(self, p1: Tuple[float, float, float], polarisation: str, network_id: str, id: str):
        super().__init__(polarisation=polarisation, p1=p1, network_id=network_id, id=id)
        self.point = p1
        self.polarisation = polarisation
        self.network_id = network_id
        self.ID = id

    def build(self, grid: FDTDGrid):
        if config.get_model_config().mode != "3D":
            raise ValueError(f"{self.params_str()} currently supports only 3-D models")
        from gprMax.studies import SourceStudy

        if config.sim_config.args.geometry_fixed and not isinstance(
            config.sim_config.study, SourceStudy
        ):
            raise ValueError(f"{self.params_str()} does not yet support geometry-fixed runs")
        self.polarisation = self.polarisation.lower()
        if self.polarisation not in ("x", "y", "z"):
            raise ValueError(f"{self.params_str()} polarisation must be x, y, or z")
        if not self.ID or "/" in self.ID or "\x00" in self.ID:
            raise ValueError(f"{self.params_str()} ID must be a non-empty HDF5 path component")
        if self.ID in grid.networkterminal_specs:
            raise ValueError(f"{self.params_str()} terminal ID is already in use")
        try:
            model = grid.rationalnetworkmodels[self.network_id]
        except KeyError as exc:
            raise ValueError(
                f"{self.params_str()} there is no rational network with ID {self.network_id!r}"
            ) from exc

        uip = self._create_uip(grid)
        self.point = uip.resolve_inf_point(self.point)
        grid.networkterminal_specs[self.ID] = {
            "polarisation": self.polarisation,
            "network_id": self.network_id,
            "point": tuple(float(value) for value in self.point),
        }
        point_within_grid, coord = uip.check_src_rx_point(self.point, self.params_str())
        if not point_within_grid:
            return
        if any(
            terminal.polarisation == self.polarisation and np.array_equal(terminal.coord, coord)
            for terminal in grid.networkterminals
        ):
            raise ValueError(
                f"{self.params_str()} another network terminal uses the same electric edge"
            )

        terminal = RationalNetworkTerminal(self.ID, model, coord, self.polarisation)
        grid.networkterminals.append(terminal)
        position = uip.round_to_grid_static_point(self.point)
        logger.info(
            self.grid_name(grid) + f"Network terminal {self.ID!r} using {self.network_id!r}, "
            f"{self.polarisation}-polarised at {position[0]:g}m, "
            f"{position[1]:g}m, {position[2]:g}m."
        )


class NetworkExcitation(GridUserObject):
    """Apply a Thevenin open-circuit waveform to a network terminal.

    Args:
        terminal_id: Identifier of a preceding :class:`NetworkTerminal`.
        waveform_id: Identifier of the driving waveform.
        start: Optional source start time in seconds.
        stop: Optional source stop time in seconds.
    """

    @property
    def order(self):
        return 7

    @property
    def hash(self):
        return "#network_excitation"

    def __init__(
        self,
        terminal_id: str,
        waveform_id: str,
        start: Optional[float] = None,
        stop: Optional[float] = None,
    ):
        super().__init__(terminal_id=terminal_id, waveform_id=waveform_id, start=start, stop=stop)
        self.terminal_id = terminal_id
        self.waveform_id = waveform_id
        self.start = start
        self.stop = stop

    def build(self, grid: FDTDGrid):
        if self.terminal_id not in grid.networkterminal_specs:
            raise ValueError(
                f"{self.params_str()} there is no network terminal with ID {self.terminal_id!r}"
            )
        if self.terminal_id in grid.networkexcitation_specs:
            raise ValueError(
                f"{self.params_str()} network terminal {self.terminal_id!r} "
                "already has an excitation"
            )
        if not any(waveform.ID == self.waveform_id for waveform in grid.waveforms):
            raise ValueError(
                f"{self.params_str()} there is no waveform with ID {self.waveform_id!r}"
            )
        grid.networkexcitation_specs[self.terminal_id] = {
            "waveform_id": self.waveform_id,
            "start": self.start,
            "stop": self.stop,
        }
        terminal = next(
            (item for item in grid.networkterminals if item.ID == self.terminal_id), None
        )
        if terminal is None:
            # In an MPI model the active terminal exists only on its owning
            # rank. The replicated specification above is sufficient on all
            # other ranks.
            if config.sim_config.mpi:
                return
            raise RuntimeError(f"{self.params_str()} terminal definition was not instantiated")
        terminal.set_excitation(self.waveform_id, self.start, self.stop)
        terminal.study_id = getattr(self, "_study_id", None)
        logger.info(
            self.grid_name(grid) + f"Network terminal {self.terminal_id!r} excited by waveform "
            f"{self.waveform_id!r}."
        )


class VoltageSource(GridUserObject):
    """Specifies a voltage source at an electric field location.

    In a 3-D model the source also owns an automatic terminal monitor. A hard
    source on a domain-minimum transverse boundary remains a valid source, but
    cannot provide that port output because its complete Ampere loop lies
    partly outside the grid.

    Attributes:
        polarisation: string required for polarisation of the source x, y, z.
        p1: tuple required for position of source x, y, z.
        resistance: float required for internal resistance (Ohms) of
                        voltage source.
        waveform_id: string required for identifier of waveform used with source.
        start: float optional to delay start time (secs) of source.
        stop: float optional to time (secs) to remove source.
        id: optional HDF5 port identifier. If omitted, ``portN`` is assigned.
        spectrum_limit: minimum cells per shortest material wavelength
            (default 10), or ``"nyquist"`` for the full research spectrum.
        reference_impedance: float optional wave-reference impedance (Ohms)
            for a hard source. The default is 50 Ohms. This final argument
            is not accepted for a finite-resistance source, whose physical
            resistance is also its wave-reference impedance.
    """

    @property
    def order(self):
        return 3

    @property
    def hash(self):
        return "#voltage_source"

    def __init__(
        self,
        p1: Tuple[float, float, float],
        polarisation: str,
        resistance: float,
        waveform_id: str,
        start: Optional[float] = None,
        stop: Optional[float] = None,
        id: Optional[str] = None,
        spectrum_limit=DEFAULT_PORT_SPECTRUM_LIMIT,
        reference_impedance: Optional[float] = None,
    ):
        from gprMax.ports import validate_spectrum_limit

        spectrum_limit = validate_spectrum_limit(spectrum_limit)
        kwargs = dict(
            polarisation=polarisation,
            p1=p1,
            resistance=resistance,
            waveform_id=waveform_id,
            start=start,
            stop=stop,
        )
        if id is not None or spectrum_limit != DEFAULT_PORT_SPECTRUM_LIMIT:
            kwargs["id"] = id
            kwargs["spectrum_limit"] = spectrum_limit
        if reference_impedance is not None:
            kwargs["reference_impedance"] = reference_impedance
        super().__init__(**kwargs)

        self.point = p1
        self.polarisation = polarisation
        self.resistance = resistance
        self.waveform_id = waveform_id
        self.start = start
        self.stop = stop
        self.id = id
        self.spectrum_limit = spectrum_limit
        self.reference_impedance = reference_impedance
        self._source = None
        self._monitor = None
        self._port_output_id = None

    @property
    def result(self):
        """Return the automatic port result after the model has solved."""

        if self._monitor is None or self._monitor.result is None:
            raise RuntimeError(
                "VoltageSource port result is not available until a supported "
                "3-D model has solved"
            )
        return self._monitor.result

    def _validate_parameters(
        self, grid: FDTDGrid, discretised_point: Optional[npt.NDArray[np.int32]] = None
    ):
        # Check polarity
        self.polarisation = self.polarisation.lower()
        if self.polarisation not in ("x", "y", "z"):
            raise ValueError(f"{self.params_str()} polarisation must be x, y, or z.")
        mode = config.get_model_config().mode
        if mode.startswith("2D"):
            invariant_letter = mode[-1]
            other_axes = [a for a in "xyz" if a != invariant_letter]
            if "TM" in mode and self.polarisation != invariant_letter:
                # E survives along the invariant axis in TM (e.g. Ez for
                # TMz) - the two tangential components are forced pec.
                raise ValueError(
                    f"{self.params_str()} polarisation must be {invariant_letter} in {mode} mode."
                )
            elif "TE" in mode and self.polarisation == invariant_letter:
                # E survives perpendicular to the invariant axis in TE
                # (e.g. Ex, Ey for TEz) - the own-axis component is
                # forced pec, same rule as HertzianDipole.
                raise ValueError(
                    f"{self.params_str()} polarisation must be {other_axes[0]} or "
                    f"{other_axes[1]} in {mode} mode."
                )

            # Once polarisation is valid, the surviving E component is
            # only ever computed by the interior update loop at one
            # specific index on the invariant axis: index 0 for TM (the
            # domain is 1 cell thick) or index 1 for TE (the interior
            # layer - the two outer walls, index 0 and 2, are forced
            # pec/pmc and never read;
            # `inf` already resolves to the correct index, but an
            # explicit coordinate might not. A voltage source landing on
            # the wrong index would be positioned on a dead cell: for a
            # resistive source this is merely ineffective
            # (create_material() would still run, but nothing reads the
            # resulting field there); for a hard (resistance=0) source
            # it's worse, since a hard source directly overwrites the
            # field value every iteration regardless of what
            # material/ID is present, bypassing the se=inf-style
            # protection a resistive source's material lookup would
            # otherwise get.
            if discretised_point is not None:
                invariant_axis = "xyz".index(invariant_letter)
                required_index = 0 if "TM" in mode else 1
                if discretised_point[invariant_axis] != required_index:
                    raise ValueError(
                        f"{self.params_str()} in {mode} mode, a voltage source must be "
                        f"positioned at index {required_index} on the invariant axis "
                        f"('{invariant_letter}') - it resolved to index "
                        f"{discretised_point[invariant_axis]}, which is never read by "
                        "the update loops and would be a dead source."
                    )

        # Check resistance
        if not np.isfinite(self.resistance) or self.resistance < 0:
            raise ValueError(
                f"{self.params_str()} requires a finite source resistance of zero or greater."
            )
        if self.reference_impedance is not None:
            self.reference_impedance = float(self.reference_impedance)
            if not np.isfinite(self.reference_impedance) or self.reference_impedance <= 0:
                raise ValueError(
                    f"{self.params_str()} reference impedance must be finite and positive."
                )
            if self.resistance > 0:
                raise ValueError(
                    f"{self.params_str()} reference impedance is only valid for a "
                    "zero-resistance hard source; a finite source uses its resistance."
                )
        if self.id is not None:
            from gprMax.ntff.interface import validate_identifier

            validate_identifier("voltage-source port ID", self.id)
        if self.spectrum_limit != "nyquist" and self.spectrum_limit < DEFAULT_PORT_SPECTRUM_LIMIT:
            logger.warning(
                f"{self.params_str()} requests only {self.spectrum_limit:g} cells "
                "per shortest material wavelength; values below 10 may have "
                "significant spatial-dispersion error."
            )

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == self.waveform_id for x in grid.waveforms):
            raise ValueError(
                f"{self.params_str()} there is no waveform with the identifier {self.waveform_id}."
            )

        # Check start and stop
        _require_complete_source_time_window(self, self.start, self.stop)
        if self.start is not None and self.stop is not None:
            if self.start < 0:
                raise ValueError(
                    f"{self.params_str()} delay of the initiation of the source should not be less"
                    " than zero."
                )
            if self.stop < 0:
                raise ValueError(
                    f"{self.params_str()} time to remove the source should not be less than zero."
                )
            if self.stop - self.start <= 0:
                raise ValueError(
                    f"{self.params_str()} duration of the source should not be zero or less."
                )

    def _create_voltage_source(
        self, grid: FDTDGrid, coord: npt.NDArray[np.int32]
    ) -> VoltageSourceUser:
        voltage_source = VoltageSourceUser()
        voltage_source.polarisation = self.polarisation
        voltage_source.coord = coord
        voltage_source.coordorigin = coord.copy()
        uip = self._create_uip(grid)
        x, y, z = uip.discretise_static_point(self.point)
        voltage_source.ID = f"{voltage_source.__class__.__name__}({x},{y},{z})"
        voltage_source.study_id = getattr(self, "_study_id", None)
        voltage_source.resistance = self.resistance
        voltage_source.reference_impedance = (
            float(self.reference_impedance)
            if self.reference_impedance is not None
            else (50.0 if self.resistance == 0 else float(self.resistance))
        )
        voltage_source.port_id = getattr(self, "_port_output_id", None)
        voltage_source.spectrum_limit = self.spectrum_limit
        voltage_source.waveformID = self.waveform_id

        if self.start is None or self.stop is None:
            voltage_source.start = 0
            voltage_source.stop = grid.timewindow
        else:
            voltage_source.start = self.start
            voltage_source.stop = min(self.stop, grid.timewindow)

        voltage_source.calculate_waveform_values(grid)

        return voltage_source

    def _create_port_monitor(
        self,
        grid: FDTDGrid,
        voltage_source: VoltageSourceUser,
        coord: npt.NDArray[np.int32],
        output_id: str,
    ) -> None:
        """Attach the source-owned terminal sampler and spectral monitor."""

        from gprMax.ports import VoltageSourcePortMonitor

        if grid.within_pml(coord):
            raise ValueError(f"{self.params_str()} cannot be placed inside a PML.")
        loop_coordinate = (
            grid.local_to_global_coordinate(coord)
            if getattr(grid, "is_distributed", False) is True
            else coord
        )
        if not _hard_source_current_loop_available(
            voltage_source.polarisation, voltage_source.resistance, loop_coordinate
        ):
            raise RuntimeError(
                "internal error: attempted to create a hard-source port without "
                "a complete transverse Ampere loop"
            )

        receiver = RxUser()
        receiver.ID = f"_voltage_port_{output_id}"
        receiver.coord = np.asarray(coord, dtype=np.int32).copy()
        receiver.coordorigin = receiver.coord.copy()
        real_dtype = config.sim_config.dtypes["float_or_double"]
        receiver.outputs[f"E{voltage_source.polarisation}"] = np.zeros(
            grid.iterations, dtype=real_dtype
        )
        receiver.internal = True
        receiver.source_bound = True
        receiver.port_id = output_id
        grid.add_receiver(receiver)

        if voltage_source.resistance == 0:
            receiver.outputs[f"I{voltage_source.polarisation}"] = np.zeros(
                grid.iterations, dtype=real_dtype
            )

        monitor = VoltageSourcePortMonitor(
            output_id,
            voltage_source,
            receiver,
            self.spectrum_limit,
            owner=self,
        )
        grid.port_monitors.append(monitor)
        voltage_source.port_output = monitor
        self._monitor = monitor

    def _log(self, grid: FDTDGrid, voltage_source: VoltageSourceUser, x: float, y: float, z: float):
        if self.start is None or self.stop is None:
            startstop = " "
        else:
            startstop = f" start time {voltage_source.start:g} secs, finish time {voltage_source.stop:g} secs "

        logger.info(
            f"{self.grid_name(grid)}Voltage source with polarity"
            f" {voltage_source.polarisation} at {x:g}m, {y:g}m, {z:g}m,"
            f" resistance {voltage_source.resistance:.1f} Ohms,"
            f"{startstop}using waveform {voltage_source.waveformID}"
            f" created."
        )

    def build(self, grid: FDTDGrid):
        # A Scene may be built again against a new grid. Do not carry runtime
        # source/monitor ownership or a previously assigned automatic ID into
        # the new build.
        self._source = None
        self._monitor = None
        self._port_output_id = None
        # Check the position of the voltage source
        uip = self._create_uip(grid)
        self.point = uip.resolve_inf_point(self.point)
        point_within_grid, discretised_point = uip.check_src_rx_point(self.point, self.params_str())

        # Every MPI rank parses the same scene. Reserve the public ID before
        # reducing the point object to its owning rank so automatic IDs remain
        # globally deterministic.
        global_discretised_point = uip.discretise_static_point(self.point)
        mpi_port_supported = _hard_source_current_loop_available(
            self.polarisation, self.resistance, global_discretised_point
        )
        if config.sim_config.mpi and config.get_model_config().mode == "3D" and mpi_port_supported:
            self._port_output_id = _reserve_voltage_port_output_id(grid, self.id, self)

        if point_within_grid:
            self._validate_parameters(grid, discretised_point)
            voltage_source = self._create_voltage_source(grid, discretised_point)
            grid.add_source(voltage_source)
            self._source = voltage_source
            if config.get_model_config().mode == "3D":
                # A local coordinate of zero on an internal MPI rank boundary
                # is not the global domain minimum: its negative halo carries
                # the magnetic samples required by the current loop.
                port_supported = _hard_source_current_loop_available(
                    voltage_source.polarisation,
                    voltage_source.resistance,
                    global_discretised_point,
                )
                if not port_supported:
                    logger.warning(
                        f"{self.params_str()} is a valid hard voltage source, but its "
                        "automatic port output is disabled because a complete transverse "
                        "Ampere loop does not fit at the domain-minimum boundary."
                    )
                else:
                    if not config.sim_config.mpi:
                        self._port_output_id = _reserve_voltage_port_output_id(grid, self.id, self)
                        voltage_source.port_id = self._port_output_id
                    self._create_port_monitor(
                        grid,
                        voltage_source,
                        discretised_point,
                        self._port_output_id,
                    )
            position = uip.round_to_grid_static_point(self.point)
            self._log(grid, voltage_source, *position)


class HertzianDipole(GridUserObject):
    """Specifies a current density term at an electric field location.

    The simplest excitation, often referred to as an additive or soft source.

    Attributes:
        polarisation: string required for polarisation of the source x, y, z.
        p1: tuple required for position of source x, y, z.
        waveform_id: string required for identifier of waveform used with source.
        start: float optional to delay start time (secs) of source.
        stop: float optional to time (secs) to remove source.
    """

    @property
    def order(self):
        return 4

    @property
    def hash(self):
        return "#hertzian_dipole"

    def __init__(
        self,
        p1: Tuple[float, float, float],
        polarisation: str,
        waveform_id: str,
        start: Optional[float] = None,
        stop: Optional[float] = None,
    ):
        super().__init__(
            polarisation=polarisation, p1=p1, waveform_id=waveform_id, start=start, stop=stop
        )

        self.point = p1
        self.polarisation = polarisation.lower()
        self.waveform_id = waveform_id
        self.start = start
        self.stop = stop

    def _validate_parameters(
        self, grid: FDTDGrid, discretised_point: Optional[npt.NDArray[np.int32]] = None
    ):
        # Check polarity
        self.polarisation = self.polarisation.lower()
        if self.polarisation not in ("x", "y", "z"):
            raise ValueError(f"{self.params_str()} polarisation must be x, y, or z.")
        mode = config.get_model_config().mode
        if mode.startswith("2D"):
            invariant_letter = mode[-1]
            other_axes = [a for a in "xyz" if a != invariant_letter]
            if "TM" in mode and self.polarisation != invariant_letter:
                # E survives along the invariant axis in TM (e.g. Ez for
                # TMz) - the two tangential components are forced pec.
                raise ValueError(
                    f"{self.params_str()} polarisation must be {invariant_letter} in {mode} mode."
                )
            elif "TE" in mode and self.polarisation == invariant_letter:
                # E survives perpendicular to the invariant axis in TE
                # (e.g. Ex, Ey for TEz) - the own-axis component is
                # forced pec.
                raise ValueError(
                    f"{self.params_str()} polarisation must be {other_axes[0]} or "
                    f"{other_axes[1]} in {mode} mode."
                )

            # Once polarisation is valid, the surviving E component is
            # only ever computed by the interior update loop at one
            # specific index on the invariant axis: index 0 for TM
            # (the domain is 1 cell thick, own-axis E offset gives a
            # single valid position) or index 1 for TE (the interior
            # layer - the two outer walls, index 0 and 2, are forced
            # pec/pmc and never read). See VoltageSource for the TM
            # case's full reasoning; the TE case was verified directly
            # from tex()/tey()/tez()'s outer-wall forcing.
            if discretised_point is not None:
                invariant_axis = "xyz".index(invariant_letter)
                required_index = 0 if "TM" in mode else 1
                if discretised_point[invariant_axis] != required_index:
                    raise ValueError(
                        f"{self.params_str()} in {mode} mode, a hertzian dipole must be "
                        f"positioned at index {required_index} on the invariant axis "
                        f"('{invariant_letter}') - it resolved to index "
                        f"{discretised_point[invariant_axis]}, which is never read by "
                        "the update loops and would be a dead source."
                    )

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == self.waveform_id for x in grid.waveforms):
            raise ValueError(
                f"{self.params_str()} there is no waveform with the identifier {self.waveform_id}."
            )

        # Check start and stop
        _require_complete_source_time_window(self, self.start, self.stop)
        if self.start is not None and self.stop is not None:
            if self.start < 0:
                raise ValueError(
                    f"{self.params_str()} delay of the initiation of the source should not be less"
                    " than zero."
                )
            if self.stop < 0:
                raise ValueError(
                    f"{self.params_str()} time to remove the source should not be less than zero."
                )
            if self.stop - self.start <= 0:
                raise ValueError(
                    f"{self.params_str()} duration of the source should not be zero or less."
                )

    def _create_hertzian_dipole(
        self, grid: FDTDGrid, coord: npt.NDArray[np.int32]
    ) -> HertzianDipoleUser:
        h = HertzianDipoleUser()
        h.polarisation = self.polarisation

        # Set length of dipole to grid size in polarisation direction
        if h.polarisation == "x":
            h.dl = grid.dx
        elif h.polarisation == "y":
            h.dl = grid.dy
        elif h.polarisation == "z":
            h.dl = grid.dz

        h.coord = coord
        h.coordorigin = coord
        uip = self._create_uip(grid)
        x, y, z = uip.discretise_static_point(self.point)
        h.ID = f"{h.__class__.__name__}({x},{y},{z})"
        h.study_id = getattr(self, "_study_id", None)
        h.waveformID = self.waveform_id

        if self.start is None or self.stop is None:
            h.start = 0
            h.stop = grid.timewindow
        else:
            h.start = self.start
            h.stop = min(self.stop, grid.timewindow)

        h.calculate_waveform_values(grid)

        return h

    def _log(
        self, grid: FDTDGrid, hertzian_dipole: HertzianDipoleUser, x: float, y: float, z: float
    ):
        if self.start is None or self.stop is None:
            startstop = " "
        else:
            startstop = f" start time {hertzian_dipole.start:g} secs, finish time {hertzian_dipole.stop:g} secs "

        if config.get_model_config().mode == "2D":
            logger.info(
                f"{self.grid_name(grid)}Hertzian dipole is a line source"
                f" in 2D with polarity {hertzian_dipole.polarisation}"
                f" at {x:g}m, {y:g}m, {z:g}m,{startstop}using"
                f" waveform {hertzian_dipole.waveformID} created."
            )
        else:
            logger.info(
                f"{self.grid_name(grid)}Hertzian dipole with polarity"
                f" {hertzian_dipole.polarisation} at {x:g}m, {y:g}m,"
                f" {z:g}m,{startstop}using waveform"
                f" {hertzian_dipole.waveformID} created."
            )

    def build(self, grid: FDTDGrid):
        # Check the position of the hertzian dipole
        uip = self._create_uip(grid)
        self.point = uip.resolve_inf_point(self.point)
        point_within_grid, discretised_point = uip.check_src_rx_point(self.point, self.params_str())

        if point_within_grid:
            self._validate_parameters(grid, discretised_point)
            hertzian_dipole = self._create_hertzian_dipole(grid, discretised_point)
            grid.add_source(hertzian_dipole)
            position = uip.round_to_grid_static_point(self.point)
            self._log(grid, hertzian_dipole, *position)


class MagneticDipole(GridUserObject):
    """Simulates an infinitesimal magnetic dipole.

    Often referred to as an additive or soft source.

    Attributes:
        polarisation: string required for polarisation of the source x, y, z.
        p1: tuple required for position of source x, y, z.
        waveform_id: string required for identifier of waveform used with source.
        start: float optional to delay start time (secs) of source.
        stop: float optional to time (secs) to remove source.
    """

    @property
    def order(self):
        return 5

    @property
    def hash(self):
        return "#magnetic_dipole"

    def __init__(
        self,
        p1: Tuple[float, float, float],
        polarisation: str,
        waveform_id: str,
        start: Optional[float] = None,
        stop: Optional[float] = None,
    ):
        super().__init__(
            polarisation=polarisation, p1=p1, waveform_id=waveform_id, start=start, stop=stop
        )

        self.point = p1
        self.polarisation = polarisation.lower()
        self.waveform_id = waveform_id
        self.start = start
        self.stop = stop

    def build(self, grid: FDTDGrid):
        # Check the position of the magnetic dipole
        uip = self._create_uip(grid)
        self.point = uip.resolve_inf_point(self.point)
        point_within_grid, discretised_point = uip.check_src_rx_point(self.point, self.params_str())

        if point_within_grid:
            self._validate_parameters(grid, discretised_point)
            magnetic_dipole = self._create_magnetic_dipole(grid, discretised_point)
            grid.add_source(magnetic_dipole)
            position = uip.round_to_grid_static_point(self.point)
            self._log(grid, magnetic_dipole, *position)

    def _validate_parameters(
        self, grid: FDTDGrid, discretised_point: Optional[npt.NDArray[np.int32]] = None
    ):
        # Check polarity
        self.polarisation = self.polarisation.lower()
        if self.polarisation not in ("x", "y", "z"):
            raise ValueError(f"{self.params_str()} polarisation must be x, y, or z.")
        # A magnetic dipole's polarisation requirement is the dual of an
        # electric source's: in TM, H survives PERPENDICULAR to the
        # invariant axis (e.g. Hx, Hy for TMz - Hz is never updated); in
        # TE, H survives ALONG the invariant axis (e.g. Hz for TEz - Hx,
        # Hy are forced pmc by tex()/tey()/tez()).
        mode = config.get_model_config().mode
        if mode.startswith("2D"):
            invariant_letter = mode[-1]
            other_axes = [a for a in "xyz" if a != invariant_letter]
            if "TM" in mode and self.polarisation == invariant_letter:
                raise ValueError(
                    f"{self.params_str()} polarisation must be {other_axes[0]} or "
                    f"{other_axes[1]} in {mode} mode."
                )
            elif "TE" in mode and self.polarisation != invariant_letter:
                raise ValueError(
                    f"{self.params_str()} polarisation must be {invariant_letter} in {mode} mode."
                )

            # Once polarisation is valid, the surviving H component is
            # only ever computed at one specific index on the invariant
            # axis: index 0 for TM (confirmed directly from the update
            # loops - fields_updates_normal.pyx's 2D Hx/Hy branches gate
            # on `k in range(0, nz)` = {0} for TMz, analogously for
            # TMx/TMy) or index 1 for TE (the interior layer - index 0
            # and 2 are the outer walls, forced pmc by
            # tex()/tey()/tez()'s defensive forcing on the own-axis H
            # survivor). Index 1 (TM) / 0,2 (TE) exist in the padded
            # array but are dead.
            if discretised_point is not None:
                invariant_axis = "xyz".index(invariant_letter)
                required_index = 0 if "TM" in mode else 1
                if discretised_point[invariant_axis] != required_index:
                    raise ValueError(
                        f"{self.params_str()} in {mode} mode, a magnetic dipole must be "
                        f"positioned at index {required_index} on the invariant axis "
                        f"('{invariant_letter}') - it resolved to index "
                        f"{discretised_point[invariant_axis]}, which is never read by "
                        "the update loops and would be a dead source."
                    )

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == self.waveform_id for x in grid.waveforms):
            raise ValueError(
                f"{self.params_str()} there is no waveform with the identifier {self.waveform_id}."
            )

        # Check start and stop
        _require_complete_source_time_window(self, self.start, self.stop)
        if self.start is not None and self.stop is not None:
            if self.start < 0:
                raise ValueError(
                    f"{self.params_str()} delay of the initiation of the source should not be less"
                    " than zero."
                )
            if self.stop < 0:
                raise ValueError(
                    f"{self.params_str()} time to remove the source should not be less than zero."
                )
            if self.stop - self.start <= 0:
                raise ValueError(
                    f"{self.params_str()} duration of the source should not be zero or less."
                )

    def _create_magnetic_dipole(
        self, grid: FDTDGrid, coord: npt.NDArray[np.int32]
    ) -> MagneticDipoleUser:
        m = MagneticDipoleUser()
        m.polarisation = self.polarisation
        m.coord = coord
        m.coordorigin = coord
        uip = self._create_uip(grid)
        x, y, z = uip.discretise_static_point(self.point)
        m.ID = f"{m.__class__.__name__}({x},{y},{z})"
        m.study_id = getattr(self, "_study_id", None)
        m.waveformID = self.waveform_id

        if self.start is None or self.stop is None:
            m.start = 0
            m.stop = grid.timewindow
        else:
            m.start = self.start
            m.stop = min(self.stop, grid.timewindow)

        m.calculate_waveform_values(grid)

        return m

    def _log(self, grid: FDTDGrid, m: MagneticDipoleUser, x: float, y: float, z: float):
        if self.start is None or self.stop is None:
            startstop = " "
        else:
            startstop = f" start time {m.start:g} secs, finish time {m.stop:g} secs "

        logger.info(
            f"{self.grid_name(grid)}Magnetic dipole with polarity"
            f" {m.polarisation} at {x:g}m, {y:g}m, {z:g}m,"
            f"{startstop}using waveform {m.waveformID} created."
        )


class TransmissionLine(GridUserObject):
    """Specifies a one-dimensional transmission line model at an electric
        field location. The source is supported by the CPU and CUDA solvers.

    Attributes:
        polarisation: string required for polarisation of the source x, y, z.
        p1: tuple required for position of source x, y, z.
        resistance: float required for internal resistance (Ohms) of
                        voltage source.
        waveform_id: string required for identifier of waveform used with source.
        start: float optional to delay start time (secs) of source.
        stop: float optional to time (secs) to remove source.
    """

    @property
    def order(self):
        return 6

    @property
    def hash(self):
        return "#transmission_line"

    def __init__(
        self,
        p1: Tuple[float, float, float],
        polarisation: str,
        resistance: float,
        waveform_id: str,
        start: Optional[float] = None,
        stop: Optional[float] = None,
    ):
        super().__init__(
            polarisation=polarisation,
            p1=p1,
            resistance=resistance,
            waveform_id=waveform_id,
            start=start,
            stop=stop,
        )

        self.point = p1
        self.polarisation = polarisation
        self.resistance = resistance
        self.waveform_id = waveform_id
        self.start = start
        self.stop = stop

    def build(self, grid: FDTDGrid):
        # Check the position of the voltage source
        uip = self._create_uip(grid)
        self.point = uip.resolve_inf_point(self.point)
        point_within_grid, discretised_point = uip.check_src_rx_point(self.point, self.params_str())

        if point_within_grid:
            self._validate_parameters(grid)
            transmission_line = self._create_transmission_line(grid, discretised_point)
            grid.add_source(transmission_line)
            position = uip.round_to_grid_static_point(self.point)
            self._log(grid, transmission_line, *position)

    def _validate_parameters(self, grid: FDTDGrid):
        # A transmission line is a 3D-only source: its internal 1D line
        # model uses a "magic time step" (TransmissionLineUser.dl =
        # sqrt(3) * c * dt) derived from the 3D Courant condition. In 2D
        # mode, calculate_dt() uses the 2-axis CFL formula instead, which
        # breaks that relationship and risks the numerical instability
        # the magic-time-step approach is already known to be sensitive
        # to even in 3D.
        if config.get_model_config().mode.startswith("2D"):
            raise ValueError(
                f"{self.params_str()} cannot be used in 2D mode - its internal "
                "line model assumes a time step derived from the 3D Courant "
                "condition, which does not match the time step used in a 2D "
                "(TM/TE) model. Consider using a #voltage_source instead."
            )

        # Check polarity
        self.polarisation = self.polarisation.lower()
        if self.polarisation not in ("x", "y", "z"):
            raise ValueError(f"{self.params_str()} polarisation must be x, y, or z.")

        # Check resistance
        if (
            not np.isfinite(self.resistance)
            or self.resistance <= 0
            or self.resistance >= config.sim_config.em_consts["z0"]
        ):
            raise ValueError(
                f"{self.params_str()} requires a finite resistance "
                "greater than zero and less than the impedance "
                "of free space, i.e. 376.73 Ohms."
            )

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == self.waveform_id for x in grid.waveforms):
            raise ValueError(
                f"{self.params_str()} there is no waveform with the identifier {self.waveform_id}."
            )

        # Check start and stop
        _require_complete_source_time_window(self, self.start, self.stop)
        if self.start is not None and self.stop is not None:
            if self.start < 0:
                raise ValueError(
                    f"{self.params_str()} delay of the initiation of the source should not be less"
                    " than zero."
                )
            if self.stop < 0:
                raise ValueError(
                    f"{self.params_str()} time to remove the source should not be less than zero."
                )
            if self.stop - self.start <= 0:
                raise ValueError(
                    f"{self.params_str()} duration of the source should not be zero or less."
                )

    def _create_transmission_line(
        self, grid: FDTDGrid, coord: npt.NDArray[np.int32]
    ) -> TransmissionLineUser:
        t = TransmissionLineUser(grid.iterations, grid.dt)
        t.polarisation = self.polarisation
        t.coord = coord
        uip = self._create_uip(grid)
        x, y, z = uip.discretise_static_point(self.point)
        t.ID = f"{t.__class__.__name__}({x},{y},{z})"
        t.study_id = getattr(self, "_study_id", None)
        t.resistance = self.resistance
        t.waveformID = self.waveform_id

        if self.start is None or self.stop is None:
            t.start = 0
            t.stop = grid.timewindow
        else:
            t.start = self.start
            t.stop = min(self.stop, grid.timewindow)

        t.calculate_waveform_values(grid)
        t.calculate_incident_V_I(grid)

        return t

    def _log(self, grid: FDTDGrid, t: TransmissionLineUser, x: float, y: float, z: float):
        if self.start is None or self.stop is None:
            startstop = " "
        else:
            startstop = f" start time {t.start:g} secs, finish time {t.stop:g} secs "

        logger.info(
            f"{self.grid_name(grid)}Transmission line with polarity"
            f" {t.polarisation} at {x:g}m, {y:g}m, {z:g}m,"
            f" resistance {t.resistance:.1f} Ohms,{startstop} using"
            f" waveform {t.waveformID} created."
        )


class MagneticFrillSource(GridUserObject):
    """Specifies a magnetic-frill (equivalent-feed) source at an electric
        field location, for an antenna fed through a PEC ground plane by a
        coaxial line. Complements #transmission_line: a different,
        well-established formulation (not a variant of the two-wire line -
        no 1D line, no ABC, no magic timestep). The corrected Hyun feed-cell
        formulation is supported by the CPU, CUDA, OpenCL, and Metal solvers,
        and by the CPU subgrid updater.

    Attributes:
        polarisation: string required for polarisation of the source - x, y,
                        or z (the antenna axis the source drives current
                        along, following the same electrical sign convention
                        as gprMax's Ix/Iy/Iz current output).
        p1: tuple required for position of source x, y, z.
        zcoax: float required for the coax's characteristic impedance
            (Ohms), calculated from its physical radii and filler. The inner
            radius used by the discrete feed cell is inferred from a
            co-located ThinWire with the same polarisation.
        waveform_id: string required for identifier of waveform used with
                        source.
        start: float optional delay before the incident waveform starts (secs).
        stop: float optional time at which the incident waveform stops (secs).
            The coaxial terminal relation remains active afterwards.
        spectrum_limit: minimum cells per shortest material wavelength
            (default 10), or ``"nyquist"`` for the full research spectrum.
    """

    @property
    def order(self):
        return 7

    @property
    def hash(self):
        return "#magnetic_frill_source"

    def __init__(
        self,
        p1: Tuple[float, float, float],
        polarisation: str,
        zcoax: float,
        waveform_id: str,
        start: Optional[float] = None,
        stop: Optional[float] = None,
        spectrum_limit=DEFAULT_PORT_SPECTRUM_LIMIT,
    ):
        from gprMax.ports import validate_spectrum_limit

        spectrum_limit = validate_spectrum_limit(spectrum_limit)
        kwargs = dict(
            polarisation=polarisation,
            p1=p1,
            zcoax=zcoax,
            waveform_id=waveform_id,
            start=start,
            stop=stop,
        )
        if spectrum_limit != DEFAULT_PORT_SPECTRUM_LIMIT:
            kwargs["spectrum_limit"] = spectrum_limit
        super().__init__(**kwargs)

        self.point = p1
        self.polarisation = polarisation
        self.zcoax = zcoax
        self.waveform_id = waveform_id
        self.start = start
        self.stop = stop
        self.spectrum_limit = spectrum_limit

    def build(self, grid: FDTDGrid):
        uip = self._create_uip(grid)
        self.point = uip.resolve_inf_point(self.point)
        point_within_grid, discretised_point = uip.check_src_rx_point(self.point, self.params_str())

        self._validate_parameters(grid)
        if config.sim_config.mpi:
            global_coord = np.asarray(uip.discretise_static_point(self.point), dtype=np.int32)
            global_index = len(grid.magneticfrill_specs) + 1
            grid.magneticfrill_specs.append(
                {
                    "coord": tuple(int(value) for value in global_coord),
                    "polarisation": self.polarisation,
                    "index": global_index,
                }
            )
            frill_source = self._create_magnetic_frill_source(grid, discretised_point)
            frill_source.mpi_global_coord = global_coord
            frill_source.mpi_global_index = global_index
            frill_source.mpi_primary_rank = int(grid.get_rank_from_coordinate(global_coord))
            frill_source.mpi_primary = bool(point_within_grid)
            grid.add_source(frill_source)
            if point_within_grid:
                position = uip.round_to_grid_static_point(self.point)
                self._log(grid, frill_source, *position)
        elif point_within_grid:
            frill_source = self._create_magnetic_frill_source(grid, discretised_point)
            grid.add_source(frill_source)
            position = uip.round_to_grid_static_point(self.point)
            self._log(grid, frill_source, *position)

    def _validate_parameters(self, grid: FDTDGrid):
        # 2D mode rejected outright - the feed point's four surrounding H
        # components have no meaningful reduction to a 2D TM/TE invariant-
        # axis model at all.
        if config.get_model_config().mode.startswith("2D"):
            raise ValueError(f"{self.params_str()} cannot be used in 2D mode.")

        # Check polarity - x, y, or z, matching #transmission_line/
        # calculate_Ix/Iy/Iz's own axis convention.
        self.polarisation = self.polarisation.lower()
        if self.polarisation not in ("x", "y", "z"):
            raise ValueError(f"{self.params_str()} polarisation must be x, y, or z.")

        # Check the user-supplied characteristic impedance used by the
        # terminal load relation and automatic port output. A zero or
        # negative value is not a physical passive coax reference impedance.
        if not np.isfinite(self.zcoax) or self.zcoax <= 0:
            raise ValueError(f"{self.params_str()} requires a finite zcoax > 0.")

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == self.waveform_id for x in grid.waveforms):
            raise ValueError(
                f"{self.params_str()} there is no waveform with the identifier {self.waveform_id}."
            )

        # Check start and stop
        _require_complete_source_time_window(self, self.start, self.stop)
        if self.start is not None and self.stop is not None:
            if self.start < 0:
                raise ValueError(
                    f"{self.params_str()} delay of the initiation of the source should not be less"
                    " than zero."
                )
            if self.stop < 0:
                raise ValueError(
                    f"{self.params_str()} time to remove the source should not be less than zero."
                )
            if self.stop - self.start <= 0:
                raise ValueError(
                    f"{self.params_str()} duration of the source should not be zero or less."
                )

    def _create_magnetic_frill_source(
        self, grid: FDTDGrid, coord: npt.NDArray[np.int32]
    ) -> MagneticFrillSourceUser:
        f = MagneticFrillSourceUser(grid.iterations, grid.dt)
        f.polarisation = self.polarisation
        f.coord = coord
        uip = self._create_uip(grid)
        x, y, z = uip.discretise_static_point(self.point)
        f.ID = f"{f.__class__.__name__}({x},{y},{z})"
        f.study_id = getattr(self, "_study_id", None)
        f.Z0 = self.zcoax
        f.waveformID = self.waveform_id
        f.spectrum_limit = self.spectrum_limit

        if self.start is None or self.stop is None:
            f.start = 0
            f.stop = grid.timewindow
        else:
            f.start = self.start
            f.stop = min(self.stop, grid.timewindow)

        f.calculate_waveform_values(grid)

        return f

    def _log(self, grid: FDTDGrid, f: MagneticFrillSourceUser, x: float, y: float, z: float):
        if self.start is None or self.stop is None:
            startstop = " "
        else:
            startstop = f" start time {f.start:g} secs, finish time {f.stop:g} secs "

        logger.info(
            f"{self.grid_name(grid)}Magnetic frill source with polarity"
            f" {f.polarisation} at {x:g}m, {y:g}m, {z:g}m,"
            f" Z0 {f.Z0:.1f} Ohms,{startstop} using"
            f" waveform {f.waveformID} created."
        )


def _dpw_tfsf_corners(uip, p1, p2, params_str):
    """Discretises and validates the TFSF box corners for a discrete plane
    wave, shared by all three #plane_wave_* builders.

    Resolves `inf` coordinates first (the recommended idiom for the
    invariant axis in 2D, matching other box-like commands; a clear error
    is raised if `inf` is used in 3D). Then, in a 2D mode, the
    mode-determined invariant-axis extent is imposed: (0, 1) for TM and
    the degenerate (1, 1) for TE - the live interior layer; the full 0..2
    TE slab must NOT be used, as its perpendicular in-plane face loops
    would write TFSF corrections into the wall-forced layers at 0 and 2.
    The invariant extent is not a user degree of freedom: explicitly-typed
    coordinates that disagree are overridden with a warning (use `inf` to
    avoid it); `inf`-typed coordinates are overridden silently - that is
    their meaning.

    Also enforces start < stop on the free (in-plane) axes - all three
    axes in 3D.

    Returns:
        start, stop: discretised corner index arrays.
    """
    if isinstance(uip.grid, SubGridBaseGrid):
        raise ValueError(
            f"{params_str} must be defined on the main grid; its TFSF box "
            "may strictly enclose complete subgrids."
        )

    mode = config.get_model_config().mode
    is_2d = mode.startswith("2D")
    if is_2d:
        inv = "xyz".index(mode[-1])
        p1_explicit = not math.isinf(p1[inv])
        p2_explicit = not math.isinf(p2[inv])
        p1 = uip.resolve_inf_point(p1, role="lower")
        p2 = uip.resolve_inf_point(p2, role="upper")

    _, start = uip.check_src_rx_point(p1, params_str)
    _, stop = uip.check_src_rx_point(p2, params_str)

    # MPI user input is translated to each rank's local coordinates for
    # geometry construction. A DPW is instead replicated on every rank, so
    # retain one authoritative global TFSF box and derive rank-local injection
    # coordinates later when the auxiliary grid is initialised.
    if config.sim_config.mpi:
        start = uip.grid.local_to_global_coordinate(start)
        stop = uip.grid.local_to_global_coordinate(stop)

    if is_2d:
        forced = (0, 1) if "TM" in mode else (1, 1)
        overridden_explicit = (p1_explicit and start[inv] != forced[0]) or (
            p2_explicit and stop[inv] != forced[1]
        )
        if overridden_explicit:
            logger.warning(
                f"{params_str} the TFSF box extent on the invariant "
                f"{'xyz'[inv]}-axis is fixed by the {mode} mode (indices "
                f"{forced[0]} to {forced[1]}) and is not adjustable - the "
                f"specified coordinates have been overridden. Use 'inf' for "
                f"the {'xyz'[inv]}-coordinates of the TFSF box corners to "
                f"avoid this warning."
            )
        start[inv], stop[inv] = forced

    for ax in range(3):
        if is_2d and ax == inv:
            continue
        if start[ax] >= stop[ax]:
            logger.exception(
                f"{params_str} the lower TFSF box corner must be strictly "
                f"less than the upper corner on the {'xyz'[ax]}-axis "
                f"(got indices {start[ax]} and {stop[ax]})."
            )
            raise ValueError

    return start, stop


class DiscretePlaneWaveAngles(GridUserObject):
    """
    Specifies a plane wave implemented using the discrete plane wave formulation.
    If the background material is not specified it will default to free space.
    The wave will propagate in the direction given by the angles theta and phi
    with a polarisation given by the angle psi.

    Attributes:
        theta: float required for propagation angle (degrees) of wave.
        phi: float required for propagation angle (degrees) of wave.
        psi: float required for polarisation of wave.
        max_angle_diff: float optional for tolerance of maximum acceptable angular difference between the
                        desired direction of the wavevector and the estimated direction of it (degrees).
                        Default is 3 arc minutes (0.05 degrees).
        p1: tuple required for the lower left position (x, y, z) of the total
            field, scattered field (TFSF) box.
        p2: tuple required for the upper right position (x, y, z) of the total
            field, scattered field (TFSF) box.
        waveform_id: string required for identifier of waveform used with source.
        material_id: string optional of material identifier to use as the
                        background material in the TFSF box.
        start: float optional to delay start time (secs) of source.
        stop: float optional to time (secs) to remove source.
        precompute: boolean optional. If ``True`` (default), precompute the
            auxiliary plane-wave source history before time stepping.
            ``False`` is reserved for a future on-the-fly implementation and
            is currently rejected.
    """

    @property
    def order(self):
        return 19

    @property
    def hash(self):
        return "#plane_wave_angles"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            theta = self.kwargs["theta"]
            phi = self.kwargs["phi"]
            psi = self.kwargs["psi"]
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            waveform_id = self.kwargs["waveform_id"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires at least ten parameters.")
            raise

        try:
            max_angle_diff = self.kwargs["max_angle_diff"]
        except KeyError:
            # 1 arcminute (0.017) is tighter than the rounding error already
            # introduced by writing theta/phi to just 1 decimal place - the
            # normal way angles are specified - so a "nice" direction like
            # (1, 2, 3) (theta=36.699.., phi=63.435..) rounds to 36.7/63.4,
            # whose true angular error (~0.021 deg) then falls just outside
            # a 0.017 deg tolerance. That silently rejects the small,
            # obviously-intended integer vector and falls through to a far
            # larger, far more expensive one satisfying the tolerance by
            # chance, with no indication a much simpler solution existed.
            # Default to 3 arc minutes (0.05 deg), which comfortably
            # recovers 1-decimal-place "nice" vectors (verified against
            # several, up to size 8) while still being a small fraction of
            # a degree - well beyond what matters for practical FDTD
            # propagation-direction accuracy.
            max_angle_diff = 0.05

        try:
            material_id = self.kwargs["material_id"]
        except KeyError:
            # set defaule to free space
            material_id = "free_space"

        try:
            precompute = self.kwargs["precompute"]
        except KeyError:
            precompute = True
        _validate_dpw_precompute(self, precompute)

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            logger.exception(
                f"{self.params_str()} there is no waveform " + f"with the identifier {waveform_id}."
            )
            raise ValueError

        # Check if there is a materialID in the materials list
        if not any(x.ID == material_id for x in grid.materials):
            logger.exception(
                f"{self.params_str()} there is no material " + f"with the identifier {material_id}."
            )
            raise ValueError

        # Check angles
        if not np.isfinite(theta) or theta < 0 or theta > 180:
            logger.exception(
                f"{self.params_str()} Polar angle theta must be between 0 and 180 degrees."
            )
            raise ValueError

        if not np.isfinite(phi) or phi < 0 or phi > 360:
            logger.exception(
                f"{self.params_str()} Azimuthal angle phi must be between 0 and 360 degrees."
            )
            raise ValueError

        if not np.isfinite(psi) or psi < 0 or psi > 360:
            logger.exception(
                f"{self.params_str()} Polarisation angle psi must be between 0 and 360 degrees."
            )
            raise ValueError
        if not np.isfinite(max_angle_diff) or max_angle_diff <= 0:
            raise ValueError(f"{self.params_str()} max_angle_diff must be finite and positive")

        uip = self._create_uip(grid)
        start, stop = _dpw_tfsf_corners(uip, p1, p2, self.params_str())

        DPW = DiscretePlaneWaveUser(grid)
        DPW.corners = np.array([*start, *stop], dtype=np.int32)
        DPW.phi = phi
        DPW.theta = theta
        DPW.psi = psi
        DPW.max_angle_diff = max_angle_diff
        DPW.waveformID = waveform_id
        DPW.materialID = material_id
        DPW.m = np.zeros(3 + 1, dtype=np.int32)
        DPW.axial = 0

        startstop = _configure_dpw_time_window(self, DPW, grid)

        DPW.initializeDiscretePlaneWave(grid)

        if precompute:
            DPW.calculate_waveform_values(grid)

        logger.info(
            f"{self.grid_name(grid)}Discrete Plane Wave within the TFSF Box "
            + f"spanning from {p1} m to {p2} m, incident in the direction "
            + f"theta {theta} degrees and phi {phi} degrees "
            + startstop
            + f"using waveform {DPW.waveformID} created."
        )

        logger.info(
            f"{self.grid_name(grid)}Discrete Plane Wave has been discretized "
            + "and angles have been approximated to the nearest rational angles "
            + f"with total angular error : {DPW.total_error:.5f}° The chosen rational integers are "
            + f"[m_x, m_y, m_z] = {DPW.m[:3]}. The approximated angles are: "
            + f"Phi: {DPW.actual_angles[1]:.3f} and Theta: {DPW.actual_angles[0]:.3f} "
            + f"and error_Theta = {DPW.angle_errors[0]:.4f}°, error_Phi = {DPW.angle_errors[1]:.4f}°"
        )

        logger.info(
            f"{self.grid_name(grid)}Discrete Plane Wave has been initialized "
            + f"with field projections (Ex, Ey, Ez, Hx, Hy, Hz) = ({DPW.projections[0]:.4f}, {DPW.projections[1]:.4f}, {DPW.projections[2]:.4f}, {DPW.projections[3]:.4f}, {DPW.projections[4]:.4f}, {DPW.projections[5]:.4f})"
            + f" , grid origin = ({DPW.origin[0]}, {DPW.origin[1]}, {DPW.origin[2]})"
            + f" and 1D vector length = {DPW.length} cells."
        )

        grid.discreteplanewaves.append(DPW)


class DiscretePlaneWaveVector(GridUserObject):
    """
    Specifies a plane wave implemented using the discrete plane wave formulation.
    If the background material is not specified it will default to free space.
    The wave will propagate in the direction given by the vector m_vec
    with a polarisation given by the angle psi. The vector m_vec should be of integer values
    and the direction of propagation will be in the direction of that vector.


    Attributes:
        m_vec: tuple required of the three integer componets specifying
            the direction of the the plane wave.
        psi: float required for the polarisation of wave.
        p1: tuple required for the lower left position (x, y, z) of the total
            field, scattered field (TFSF) box.
        p2: tuple required for the upper right position (x, y, z) of the total
            field, scattered field (TFSF) box.
        waveform_id: string required for identifier of waveform used with source.
        material_id: string optional of material identifier to use as the
                        background material in the TFSF box.
        start: float optional to delay start time (secs) of source.
        stop: float optional to time (secs) to remove source.
        precompute: boolean optional. If ``True`` (default), precompute the
            auxiliary plane-wave source history before time stepping.
            ``False`` is reserved for a future on-the-fly implementation and
            is currently rejected.
    """

    @property
    def order(self):
        return 22

    @property
    def hash(self):
        return "#plane_wave_vector"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            m_vec = self.kwargs["m_vec"]
            psi = self.kwargs["psi"]
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            waveform_id = self.kwargs["waveform_id"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires at least eleven parameters.")
            raise

        try:
            material_id = self.kwargs["material_id"]
        except KeyError:
            # set defaule to free space
            material_id = "free_space"

        try:
            precompute = self.kwargs["precompute"]
        except KeyError:
            precompute = True
        _validate_dpw_precompute(self, precompute)

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            logger.exception(
                f"{self.params_str()} there is no waveform " + f"with the identifier {waveform_id}."
            )
            raise ValueError

        # Check if there is a materialID in the materials list
        if not any(x.ID == material_id for x in grid.materials):
            logger.exception(
                f"{self.params_str()} there is no material " + f"with the identifier {material_id}."
            )
            raise ValueError

        # Check angle

        if not np.isfinite(psi) or psi < 0 or psi > 360:
            logger.exception(
                f"{self.params_str()} Polarisation angle psi must be between 0 and 360 degrees."
            )
            raise ValueError

        m_values = np.asarray(m_vec)
        if (
            m_values.shape != (3,)
            or not np.all(np.isfinite(m_values))
            or not np.all(m_values == np.rint(m_values))
            or not np.any(m_values)
        ):
            raise ValueError(
                f"{self.params_str()} m_vec must contain three finite integers and not be zero"
            )
        m_vec = tuple(int(value) for value in m_values)

        uip = self._create_uip(grid)
        start, stop = _dpw_tfsf_corners(uip, p1, p2, self.params_str())

        DPW = DiscretePlaneWaveUser(grid)
        DPW.corners = np.array([*start, *stop], dtype=np.int32)
        # Log-only angles (the authoritative computation lives in
        # initializeDiscretePlaneWave's m-vector branch): the physical
        # propagation direction is the wavefront normal (m_x/dx, m_y/dy,
        # m_z/dz) - the same convention find_dpw_integers_optimized uses.
        _phys = np.array(m_vec, dtype=np.float64) / np.array([grid.dx, grid.dy, grid.dz])
        _phys /= np.linalg.norm(_phys)
        DPW.theta = math.degrees(math.acos(np.clip(_phys[2], -1.0, 1.0)))
        DPW.phi = math.degrees(math.atan2(_phys[1], _phys[0]))
        DPW.psi = psi
        DPW.max_angle_diff = 0
        DPW.waveformID = waveform_id
        DPW.materialID = material_id
        # m is a 4-element array: the user's [m_x, m_y, m_z] plus a 4th slot
        # that initializeDiscretePlaneWave() fills with max(|m_x|, |m_y|, |m_z|)
        DPW.m = np.zeros(3 + 1, dtype=np.int32)
        DPW.m[:3] = np.array(m_vec, dtype=np.int32)
        DPW.axial = 0

        startstop = _configure_dpw_time_window(self, DPW, grid)

        DPW.initializeDiscretePlaneWave(grid)

        if precompute:
            DPW.calculate_waveform_values(grid)

        logger.info(
            f"{self.grid_name(grid)}Discrete Plane Wave within the TFSF Box "
            + f"spanning from {p1} m to {p2} m, incident in the direction "
            + f"theta {DPW.theta} degrees and phi {DPW.phi} degrees "
            + startstop
            + f"using waveform {DPW.waveformID} created."
        )

        logger.info(
            f"{self.grid_name(grid)}Discrete Plane Wave has been discretized "
            + "using user specified integers that are "
            + f"[m_x, m_y, m_z] = {DPW.m[:3]}. The approximated angles are: "
            + f"Phi: {DPW.actual_angles[1]:.3f} and Theta: {DPW.actual_angles[0]:.3f} "
            + f"and error_Theta = {DPW.angle_errors[0]:.4f}°, error_Phi = {DPW.angle_errors[1]:.4f}°"
        )

        logger.info(
            f"{self.grid_name(grid)}Discrete Plane Wave has been initialized "
            + f"with field projections (Ex, Ey, Ez, Hx, Hy, Hz) = ({DPW.projections[0]:.4f}, {DPW.projections[1]:.4f}, {DPW.projections[2]:.4f}, {DPW.projections[3]:.4f}, {DPW.projections[4]:.4f}, {DPW.projections[5]:.4f})"
            + f" , grid origin = ({DPW.origin[0]}, {DPW.origin[1]}, {DPW.origin[2]})"
            + f" and 1D vector length = {DPW.length} cells."
        )

        grid.discreteplanewaves.append(DPW)


class DiscretePlaneWaveAxial(GridUserObject):
    """
    Specifies an axial plane wave implemented using the discrete plane wave formulation.
    No background material is specified as materials are copied directly
    by the existing grid values along the axial direction of propagation.
    The wave will propagate in one of the directions of the Cartesian axes of the grid
    (x, y or z) with a polarisation given by the angle psi.


    Attributes:
        axis: a single character string required for defining the propagation axis of the wave x, y or z.
        psi: float required for polarisation of wave.
        p1: tuple required for the lower left position (x, y, z) of the total
            field, scattered field (TFSF) box.
        p2: tuple required for the upper right position (x, y, z) of the total
            field, scattered field (TFSF) box.
        waveform_id: string required for identifier of waveform used with source.
        start: float optional to delay start time (secs) of source.
        stop: float optional to time (secs) to remove source.
        precompute: boolean optional. If ``True`` (default), precompute the
            auxiliary plane-wave source history before time stepping.
            ``False`` is reserved for a future on-the-fly implementation and
            is currently rejected.
    """

    @property
    def order(self):
        return 20

    @property
    def hash(self):
        return "#plane_wave_axial"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            axis = self.kwargs["axis"]
            psi = self.kwargs["psi"]
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            waveform_id = self.kwargs["waveform_id"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires at least 9 parameters.")
            raise

        try:
            precompute = self.kwargs["precompute"]
        except KeyError:
            precompute = True
        _validate_dpw_precompute(self, precompute)

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            logger.exception(
                f"{self.params_str()} there is no waveform " + f"with the identifier {waveform_id}."
            )
            raise ValueError

        # Check polarisation angle
        if not np.isfinite(psi) or psi < 0 or psi > 360:
            logger.exception(
                f"{self.params_str()} Polarisation angle psi must be between 0 and 360 degrees."
            )
            raise ValueError

        uip = self._create_uip(grid)
        start, stop = _dpw_tfsf_corners(uip, p1, p2, self.params_str())

        DPW = DiscretePlaneWaveUser(grid)
        DPW.corners = np.array([*start, *stop], dtype=np.int32)
        DPW.waveformID = waveform_id
        DPW.m = np.zeros(3 + 1, dtype=np.int32)
        DPW.axis = axis.lower()
        DPW.psi = psi
        if axis.lower() == "x":
            DPW.axial = 1
            DPW.m[0] = 1
            DPW.m[1] = 0
            DPW.m[2] = 0
            DPW.theta = 90.0
            DPW.phi = 0.0
        elif axis.lower() == "y":
            DPW.axial = 2
            DPW.m[0] = 0
            DPW.m[1] = 1
            DPW.m[2] = 0
            DPW.theta = 90.0
            DPW.phi = 90.0
        elif axis.lower() == "z":
            DPW.axial = 3
            DPW.m[0] = 0
            DPW.m[1] = 0
            DPW.m[2] = 1
            DPW.theta = 0.0
            DPW.phi = 0.0
        else:
            logger.exception(f"{self.params_str()} DPW Axis must be x, y, or z.")
            raise ValueError

        # In a 2D model the propagation axis must be in-plane.
        mode = config.get_model_config().mode
        if mode.startswith("2D") and axis.lower() == mode[-1]:
            logger.exception(
                f"{self.params_str()} in {mode} mode the propagation axis must be "
                f"in-plane; '{axis.lower()}' is the invariant axis."
            )
            raise ValueError

        startstop = _configure_dpw_time_window(self, DPW, grid)

        DPW.initializeDiscretePlaneWave(grid)

        if precompute:
            DPW.calculate_waveform_values(grid)

        logger.info(
            f"{self.grid_name(grid)}Discrete Plane Wave within the TFSF Box "
            + f"spanning from {p1} m to {p2} m, normally propagating in the {DPW.axis.lower()} direction "
            + f" with theta {DPW.theta} degrees, phi {DPW.phi} degrees and polarisation angle psi {DPW.psi} degrees "
            + startstop
            + f"using waveform {DPW.waveformID} created."
        )

        grid.discreteplanewaves.append(DPW)


class EigenmodeBand(GridUserObject):
    """Define the frequency band and optional extra DFT bins shared by all ports.

    ``points`` selects equally spaced output frequencies from ``fmin`` to
    ``fmax``, including both endpoints. ``frequencies`` adds specific values
    within that range. The combined list is sorted from low to high, with
    repeated values included only once.

    Attributes:
        id: unique identifier for the shared eigenmode band.
        fmin: lower direct-DFT frequency in Hz.
        fmax: upper direct-DFT frequency in Hz.
        points: number of equally spaced output frequencies, including both endpoints.
        frequencies: optional additional direct-DFT frequencies in Hz. Values
            must lie between fmin and fmax inclusive and are added to the
            equally spaced frequencies selected by points.
        transition: positive transition width in Hz for an automatically
            generated bandpass waveform, or ``"auto"`` (default).
        spectral_threshold: relative spectral threshold used to determine
            significant excitation support. The default is ``1e-3``.
    """

    @property
    def order(self):
        return 20

    @property
    def hash(self):
        return "#eigenmode_band"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        if grid.eigenmodeband is not None:
            raise ValueError("Exactly one EigenmodeBand may be defined per grid.")
        try:
            band_id = str(self.kwargs["id"])
            fmin = float(self.kwargs["fmin"])
            fmax = float(self.kwargs["fmax"])
            points = int(self.kwargs["points"])
        except KeyError:
            logger.exception(f"{self.params_str()} requires id, fmin, fmax, and points.")
            raise
        if not band_id or any(character.isspace() for character in band_id):
            raise ValueError(
                f"{self.params_str()} id must be a non-empty token without whitespace."
            )
        _validate_eigenmode_dft(self.params_str(), fmin, fmax, points)
        requested_frequencies = self.kwargs.get("frequencies")
        if requested_frequencies is None:
            requested_frequencies = ()
        elif np.isscalar(requested_frequencies):
            requested_frequencies = (float(requested_frequencies),)
        else:
            requested_frequencies = tuple(float(value) for value in requested_frequencies)
        if any(
            not np.isfinite(value) or value < fmin or value > fmax
            for value in requested_frequencies
        ):
            raise ValueError(
                f"{self.params_str()} additional DFT frequencies must be finite and "
                "lie inside the inclusive band limits."
            )
        threshold = float(self.kwargs.get("spectral_threshold", 1e-3))
        if not 0 < threshold < 1:
            raise ValueError(
                f"{self.params_str()} spectral_threshold must be between zero and one."
            )
        transition = self.kwargs.get("transition", "auto")
        if transition != "auto":
            transition = float(transition)
            if not np.isfinite(transition) or transition <= 0:
                raise ValueError(f"{self.params_str()} transition must be positive or auto.")
        grid.eigenmodeband = EigenmodeBandSpec(
            id=band_id,
            fmin=fmin,
            fmax=fmax,
            points=points,
            frequencies=requested_frequencies,
            transition=transition,
            spectral_threshold=threshold,
        )
        dft_points = len(grid.eigenmodeband.dft_frequencies)
        extra_description = (
            f", including {dft_points - points} additional requested bin(s) after deduplication"
            if requested_frequencies
            else ""
        )
        logger.info(
            f"{self.grid_name(grid)}Eigenmode band {band_id!r}, frequencies "
            f"{fmin:g} to {fmax:g} Hz with {dft_points} common DFT point(s)"
            f"{extra_description}, created."
        )


class EigenmodePort(GridUserObject):
    """Define a modal port plane with its own anchor-frequency policy.

    Attributes:
        port: positive, one-based port number.
        p1: first physical ``(x, y, z)`` corner of the port plane.
        p2: opposite physical ``(x, y, z)`` corner of the port plane.
        direction: launch/monitor direction normal to the plane, ``+`` or
            ``-``.
        modes: positive integer mode count or an increasing sequence of
            one-based mode indices.
        anchors: ``"auto"`` (default), one frequency, or an increasing
            sequence of modal-solve anchor frequencies in Hz.
        plot_fields: optionally force or suppress modal-field plots. ``None``
            retains the geometry-only default.
    """

    @property
    def order(self):
        return 21

    @property
    def hash(self):
        return "#eigenmode_port"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        if grid.eigenmodeband is None:
            raise ValueError(f"{self.params_str()} requires one preceding EigenmodeBand.")
        try:
            port = int(self.kwargs["port"])
            p1 = tuple(float(value) for value in self.kwargs["p1"])
            p2 = tuple(float(value) for value in self.kwargs["p2"])
            direction = str(self.kwargs["direction"])
            modes_arg = self.kwargs["modes"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires port, p1, p2, direction, and modes.")
            raise
        if port < 1:
            raise ValueError(f"{self.params_str()} port must be one or greater.")
        if len(p1) != 3 or len(p2) != 3:
            raise ValueError(f"{self.params_str()} p1 and p2 must each contain three coordinates.")
        if direction not in ("+", "-"):
            raise ValueError(f"{self.params_str()} direction must be + or -.")
        if np.isscalar(modes_arg):
            mode_count = int(modes_arg)
            modes = tuple(range(1, mode_count + 1))
        else:
            modes = tuple(int(value) for value in modes_arg)
        if not modes or any(mode < 1 for mode in modes):
            raise ValueError(f"{self.params_str()} modes must contain positive one-based indices.")
        if modes != tuple(sorted(set(modes))):
            raise ValueError(f"{self.params_str()} modes must be unique and strictly increasing.")
        anchors_arg = self.kwargs.get("anchors", "auto")
        if isinstance(anchors_arg, str):
            if anchors_arg.lower() != "auto":
                raise ValueError(f"{self.params_str()} anchors must be auto or frequencies.")
            anchors = "auto"
        elif np.isscalar(anchors_arg):
            anchors = (float(anchors_arg),)
        else:
            anchors = tuple(float(value) for value in anchors_arg)
        if anchors != "auto":
            if not anchors:
                raise ValueError(f"{self.params_str()} requires at least one explicit anchor.")
            if any(not np.isfinite(value) or value <= 0 for value in anchors):
                raise ValueError(f"{self.params_str()} anchors must be finite and positive.")
            if any(anchors[index] >= anchors[index + 1] for index in range(len(anchors) - 1)):
                raise ValueError(
                    f"{self.params_str()} anchors must be unique and strictly increasing."
                )
        if port in grid.eigenmodeportdefs:
            raise ValueError(f"Eigenmode port {port} is already defined.")

        domain_mode = config.get_model_config().mode
        invariant_axis = "xyz".index(domain_mode[-1]) if domain_mode.startswith("2D") else None
        equal_axes = [
            axis
            for axis in range(3)
            if axis != invariant_axis and p1[axis] == p2[axis] and np.isfinite(p1[axis])
        ]
        if len(equal_axes) != 1:
            raise ValueError(
                f"{self.params_str()} must have exactly one finite matching coordinate "
                "pair, which defines the port normal."
            )
        normal_axis = equal_axes[0]
        transverse_axes = tuple(axis for axis in range(3) if axis != normal_axis)
        plot_fields = self.kwargs.get("plot_fields")
        if plot_fields is not None and not isinstance(plot_fields, (bool, np.bool_)):
            raise ValueError(f"{self.params_str()} plot_fields must be True, False, or None.")
        grid.eigenmodeportdefs[port] = EigenmodePortSpec(
            port=port,
            p1=p1,
            p2=p2,
            normal="xyz"[normal_axis],
            direction=direction,
            normal_axis=normal_axis,
            transverse_axes=transverse_axes,
            invariant_axis=invariant_axis,
            modes=modes,
            anchors=anchors,
            plot_fields=None if plot_fields is None else bool(plot_fields),
        )
        axis_name = "xyz"[normal_axis]
        logger.info(
            f"{self.grid_name(grid)}Eigenmode port {port}, normal {axis_name}{direction}, "
            f"monitoring modes {modes}, with anchors {anchors}, created."
        )


class VirtualWaveguide(GridUserObject):
    """Terminate an eigenmode port with a bidirectionally coupled FDTD guide.

    The guide repeats the material cross-section at the referenced
    :class:`EigenmodePort`, couples both E and H fields at its aperture, and
    terminates its remote end with a PML. It can contain the active modal
    source or act as a passive matched receiver.

    Args:
        port: One-based eigenmode-port number.
        length_cells: Total virtual-guide length in cells. Default 30.
        pml_cells: PML thickness at the remote end. Default 12.
        source_clearance_cells: Cells between an active internal source plane
            and the PML. Default 6.
        pml_profile: Optional reusable PML profile ID.
    """

    @property
    def order(self):
        return 22

    @property
    def hash(self):
        return "#virtual_waveguide"

    def __init__(
        self,
        port,
        length_cells=30,
        pml_cells=12,
        source_clearance_cells=6,
        pml_profile=None,
    ):
        super().__init__(
            port=port,
            length_cells=length_cells,
            pml_cells=pml_cells,
            source_clearance_cells=source_clearance_cells,
            pml_profile=pml_profile,
        )

    def build(self, grid: FDTDGrid):
        def integer_parameter(name):
            try:
                value = operator.index(self.kwargs[name])
            except TypeError as exc:
                raise ValueError(f"{self.params_str()} {name} must be an integer.") from exc
            if isinstance(self.kwargs[name], (bool, np.bool_)):
                raise ValueError(f"{self.params_str()} {name} must be an integer.")
            return value

        port = integer_parameter("port")
        length_cells = integer_parameter("length_cells")
        pml_cells = integer_parameter("pml_cells")
        clearance = integer_parameter("source_clearance_cells")
        profile_id = self.kwargs.get("pml_profile")
        if port not in grid.eigenmodeportdefs:
            raise ValueError(f"{self.params_str()} references unknown eigenmode port {port}.")
        if port in grid.virtual_waveguide_specs:
            raise ValueError(f"Eigenmode port {port} already has a virtual waveguide.")
        if length_cells < 1 or pml_cells < 1 or clearance < 1:
            raise ValueError(f"{self.params_str()} cell counts must all be positive integers.")
        if pml_cells < 2:
            raise ValueError(f"{self.params_str()} pml_cells must be at least 2.")
        minimum_length = pml_cells + clearance + 3
        if length_cells < minimum_length:
            raise ValueError(
                f"{self.params_str()} length_cells must be at least pml_cells + "
                f"source_clearance_cells + 3 ({minimum_length} for this request)."
            )
        if profile_id is not None:
            profile_id = str(profile_id)
            if not profile_id:
                raise ValueError(f"{self.params_str()} pml_profile must not be empty.")
            if profile_id not in grid.pmls["profiles"]:
                raise ValueError(
                    f"{self.params_str()} refers to unknown PML profile {profile_id!r}."
                )

        grid.virtual_waveguide_specs[port] = VirtualWaveguideSpec(
            port=port,
            length_cells=length_cells,
            pml_cells=pml_cells,
            source_clearance_cells=clearance,
            profile_id=profile_id,
        )
        logger.info(
            f"{self.grid_name(grid)}Virtual waveguide requested for eigenmode "
            f"port {port}: length {length_cells}, PML {pml_cells}, source "
            f"clearance {clearance} cell(s)"
            + (f", PML profile {profile_id!r}." if profile_id else ".")
        )


class EigenmodeExcitation(GridUserObject):
    """Attach one active modal drive to a defined port.

    Several excitations may share the same base waveform and drive different
    port/mode channels. Each physical port is solved and monitored only once;
    its additional drives reuse the cached modal anchor bank.

    Args:
        port: One-based ``EigenmodePort`` number.
        mode: One of the modes monitored by that port.
        waveform: Shared waveform ID, or ``'auto'`` for the bandpass pulse.
        amplitude: Non-zero modal amplitude scale. Mutually exclusive with
            ``power``.
        power: Optional relative incident-power scale; applies amplitude
            ``sqrt(power)``.
        phase_deg: Constant spectral phase in degrees.
        delay_s: True time delay in seconds, applied as
            ``exp(-1j * 2*pi*f*delay_s)``.
        plot_waveform: Write or suppress this drive's waveform/DFT plot. The
            default writes it only for geometry-only runs.
    """

    @property
    def order(self):
        return 23

    @property
    def hash(self):
        return "#eigenmode_excitation"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _runtime_plane_kwargs(port, band):
        transverse_p1 = tuple(port.p1[axis] for axis in port.transverse_axes)
        transverse_p2 = tuple(port.p2[axis] for axis in port.transverse_axes)
        lower = tuple(min(first, second) for first, second in zip(transverse_p1, transverse_p2))
        upper = tuple(max(first, second) for first, second in zip(transverse_p1, transverse_p2))
        kwargs = {
            "normal": port.normal,
            "direction": port.direction,
            "p1": lower,
            "p2": upper,
            "w": port.p1[port.normal_axis],
            "port_index": port.port,
            "dft_start": band.fmin,
            "dft_stop": band.fmax,
            "dft_points": band.points,
            "dft_frequencies": band.dft_frequencies,
            "plot_fields": port.plot_fields,
        }
        if len(port.resolved_anchors) == 1:
            kwargs["frequency"] = port.resolved_anchors[0]
        else:
            kwargs["frequencies"] = port.resolved_anchors
        return kwargs

    def build(self, grid: FDTDGrid):
        if grid.eigenmodeband is None:
            raise ValueError(f"{self.params_str()} requires one EigenmodeBand.")
        try:
            source_port_number = int(self.kwargs["port"])
            excitation_mode = int(self.kwargs["mode"])
        except KeyError:
            logger.exception(f"{self.params_str()} requires port and mode.")
            raise
        if not grid.eigenmodeportdefs:
            raise ValueError(f"{self.params_str()} requires at least one EigenmodePort.")
        if source_port_number not in grid.eigenmodeportdefs:
            raise ValueError(
                f"{self.params_str()} references unknown eigenmode port {source_port_number}."
            )
        source_port = grid.eigenmodeportdefs[source_port_number]
        if excitation_mode not in source_port.modes:
            raise ValueError(
                f"{self.params_str()} mode {excitation_mode} is not monitored by port "
                f"{source_port_number}; available modes are {source_port.modes}."
            )
        band = grid.eigenmodeband
        waveform_arg = self.kwargs.get("waveform", "auto")
        amplitude_given = "amplitude" in self.kwargs
        power_given = "power" in self.kwargs
        if amplitude_given and power_given:
            raise ValueError(f"{self.params_str()} accepts amplitude or power, not both.")
        if power_given:
            power = float(self.kwargs["power"])
            if not np.isfinite(power) or power <= 0:
                raise ValueError(f"{self.params_str()} power must be finite and positive.")
            amplitude = float(np.sqrt(power))
        else:
            amplitude = float(self.kwargs.get("amplitude", 1.0))
            if not np.isfinite(amplitude) or amplitude == 0:
                raise ValueError(
                    f"{self.params_str()} amplitude must be finite and non-zero. "
                    "Omit the excitation to leave a channel passive."
                )
            power = amplitude**2
        phase_deg = float(self.kwargs.get("phase_deg", 0.0))
        delay_s = float(self.kwargs.get("delay_s", 0.0))
        if not np.isfinite(phase_deg):
            raise ValueError(f"{self.params_str()} phase_deg must be finite.")
        if not np.isfinite(delay_s):
            raise ValueError(f"{self.params_str()} delay_s must be finite.")
        plot_waveform = self.kwargs.get("plot_waveform")
        if plot_waveform is not None and not isinstance(plot_waveform, (bool, np.bool_)):
            raise ValueError(f"{self.params_str()} plot_waveform must be True, False, or None.")
        if plot_waveform is not None:
            plot_waveform = bool(plot_waveform)
        generated_waveform = isinstance(waveform_arg, str) and waveform_arg.lower() == "auto"
        if generated_waveform:
            waveform_id = f"{band.id}_auto_bandpass"
            matches = [waveform for waveform in grid.waveforms if waveform.ID == waveform_id]
            if matches:
                waveform = matches[0]
                if not isinstance(waveform, EigenmodeBandpassWaveform):
                    raise ValueError(
                        f"Generated eigenmode waveform ID {waveform_id!r} is already in use."
                    )
            else:
                # Drive scaling is deliberately separate from the common base
                # waveform, allowing all array channels to share one spectrum.
                waveform = EigenmodeBandpassWaveform(
                    band_id=band.id,
                    fmin=band.fmin,
                    fmax=band.fmax,
                    amplitude=1.0,
                    dt=grid.dt,
                    sample_count=int(grid.iterations),
                    spectral_threshold=band.spectral_threshold,
                    transition=band.transition,
                )
                grid.waveforms.append(waveform)
        else:
            waveform_id = str(waveform_arg)
            matches = [waveform for waveform in grid.waveforms if waveform.ID == waveform_id]
            if not matches:
                raise ValueError(
                    f"{self.params_str()} references unknown waveform {waveform_id!r}."
                )
            waveform = matches[0]
        if grid.eigenmodeexcitations:
            base_waveform = grid.eigenmodeexcitations[0].waveform
            if waveform.ID != base_waveform.ID:
                raise ValueError(
                    "Simultaneous EigenmodeExcitation objects must share one base "
                    f"waveform; got {base_waveform.ID!r} and {waveform.ID!r}."
                )
        else:
            band.resolve_spectrum(grid, waveform, generated_waveform=generated_waveform)

        channel = (source_port_number, excitation_mode)
        existing_channels = {
            (excitation.port_index, excitation.mode_index)
            for excitation in grid.eigenmodeexcitations
        }
        if channel in existing_channels:
            raise ValueError(
                f"Eigenmode port {source_port_number}, mode {excitation_mode} is already driven."
            )
        self.port_index = source_port_number
        self.mode_index = excitation_mode
        self.waveform = waveform
        self.amplitude = amplitude
        self.power = power
        self.phase_deg = phase_deg
        self.delay_s = delay_s
        self.plot_waveform = plot_waveform
        grid.eigenmodeexcitations.append(self)
        if grid.eigenmodeexcitation is None:
            grid.eigenmodeexcitation = self
        logger.info(
            f"{self.grid_name(grid)}Eigenmode excitation created on port "
            f"{source_port_number}, mode {excitation_mode}, using waveform {waveform.ID!r}, "
            f"amplitude {amplitude:g}, phase {phase_deg:g} degrees, and delay {delay_s:g} s."
        )


def build_eigenmode_runtime_ports(grid):
    """Build one modal runtime owner per physical port after all drives exist."""

    band = grid.eigenmodeband
    drives_by_port = {}
    for excitation in grid.eigenmodeexcitations:
        drives_by_port.setdefault(excitation.port_index, []).append(excitation)
    reusable_study = bool(
        len(grid.eigenmodeexcitations) == 1
        and getattr(grid.eigenmodeexcitations[0], "_reusable_study", False)
    )
    for port_number in sorted(grid.eigenmodeportdefs):
        port = grid.eigenmodeportdefs[port_number]
        drives = drives_by_port.get(port_number, [])
        port.resolve_anchors(band, is_source=(bool(drives) or reusable_study))
        common = EigenmodeExcitation._runtime_plane_kwargs(port, band)
        if drives:
            first = drives[0]
            common.update(
                {
                    "mode_index": first.mode_index,
                    "mode_count": max(port.modes),
                    "waveform_id": first.waveform.ID,
                    "spectral_threshold": band.spectral_threshold,
                }
            )
            _EigenmodeSourceBuilder(**common).build(grid)
            runtime = grid.eigenmodesources[-1]
            runtime.mode_indices = port.modes
            runtime.plot_waveform = first.plot_waveform
            runtime.drive_specs = tuple(drives)
            runtime.set_drive_parameters(first)
        else:
            common.update({"mode_count": max(port.modes), "id": f"port{port.port}"})
            _EigenmodeReceiverBuilder(**common).build(grid)
            runtime = grid.eigenmodereceivers[-1]
            runtime.mode_indices = port.modes
            runtime.spectral_threshold = band.spectral_threshold
            runtime.drive_specs = ()
        runtime.anchor_policy = port.anchor_policy
        runtime.requested_anchor_policy = port.anchor_policy
        runtime.resolved_anchor_policy = port.anchor_policy
        runtime.fallback_frequency = 0.5 * (band.fmin + band.fmax)


def _validate_eigenmode_dft(label, start, stop, points):
    if not np.isfinite(start) or not np.isfinite(stop) or start <= 0 or stop < start:
        raise ValueError(f"{label} DFT frequencies must satisfy 0 < start <= stop.")
    if points < 1:
        raise ValueError(f"{label} DFT points must be at least one.")
    if points == 1 and stop != start:
        raise ValueError(f"{label} a one-point DFT requires equal start and stop.")
    if points > 1 and stop == start:
        raise ValueError(f"{label} a multi-point DFT requires stop greater than start.")


def _validate_eigenmode_dft_frequencies(label, frequencies):
    values = np.asarray(frequencies, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{label} DFT frequencies must be one-dimensional and non-empty.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{label} DFT frequencies must be finite and positive.")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{label} DFT frequencies must be unique and strictly increasing.")


def build_passive_virtual_eigenmode_ports(grid):
    """Build monitor runtimes when every modal port is passive and virtual."""

    band = grid.eigenmodeband
    band.significant_range = (band.fmin, band.fmax)
    band.representative_frequency = 0.5 * (band.fmin + band.fmax)
    for port_number in sorted(grid.eigenmodeportdefs):
        port = grid.eigenmodeportdefs[port_number]
        port.resolve_anchors(band, is_source=False)
        common = EigenmodeExcitation._runtime_plane_kwargs(port, band)
        common.update({"mode_count": max(port.modes), "id": f"port{port.port}"})
        _EigenmodeReceiverBuilder(**common).build(grid)
        runtime = grid.eigenmodereceivers[-1]
        runtime.mode_indices = port.modes
        runtime.anchor_policy = port.anchor_policy
        runtime.requested_anchor_policy = port.anchor_policy
        runtime.resolved_anchor_policy = port.anchor_policy
        runtime.fallback_frequency = band.representative_frequency


def _discretise_eigenmode_plane(
    user_object,
    grid,
    normal_axis,
    transverse_axes,
    full_lower,
    full_upper,
):
    """Map one global-coordinate modal plane onto its owning grid.

    Main-grid coordinates have a zero local origin. HSG coordinates are still
    supplied in the global model frame, so ``SubgridUserInput`` must translate
    them past the fine grid's auxiliary padding. A modal plane is deliberately
    more restricted than ordinary subgrid geometry: every staggered component
    used by the FDFD slice, TF/SF correction, and modal monitor must remain
    strictly inside the HSG working region and away from its coupling surface.
    """

    uip = user_object._create_uip(grid)
    if hasattr(grid, "global_size"):
        mode = config.get_model_config().mode

        def resolve_global_inf(point, role):
            resolved = list(point)
            for axis, value in enumerate(point):
                if not math.isinf(value):
                    continue
                if not mode.startswith("2D"):
                    raise ValueError(
                        f"{user_object.params_str()} 'inf' coordinates require a 2D model."
                    )
                extent = float(grid.global_size[axis] * grid.dl[axis])
                resolved[axis] = 0.0 if role == "lower" else extent
            return tuple(resolved)

        full_lower = np.asarray(resolve_global_inf(full_lower, "lower"), dtype=np.float64)
        full_upper = np.asarray(resolve_global_inf(full_upper, "upper"), dtype=np.float64)
        lower = np.asarray(uip.discretise_static_point(tuple(full_lower)), dtype=np.int32)
        upper = np.asarray(uip.discretise_static_point(tuple(full_upper)), dtype=np.int32)
    else:
        full_lower = np.asarray(
            uip.resolve_inf_point(tuple(full_lower), role="lower"),
            dtype=np.float64,
        )
        full_upper = np.asarray(
            uip.resolve_inf_point(tuple(full_upper), role="upper"),
            dtype=np.float64,
        )
        lower = np.asarray(uip.discretise_point(tuple(full_lower)), dtype=np.int32)
        upper = np.asarray(uip.discretise_point(tuple(full_upper)), dtype=np.int32)
    plane_index = int(lower[normal_axis])

    if int(upper[normal_axis]) != plane_index:
        raise ValueError(
            f"{user_object.params_str()} modal-plane normal coordinates must map "
            "to the same grid plane."
        )

    if isinstance(grid, SubGridBaseGrid):
        inner = np.asarray(
            (
                grid.n_boundary_cells_x,
                grid.n_boundary_cells_y,
                grid.n_boundary_cells_z,
            ),
            dtype=np.int32,
        )
        outer = np.asarray(grid.size, dtype=np.int32) - inner

        # E is sampled/updated on plane_index and H on plane_index or
        # plane_index - 1. Transverse staggered components include both range
        # endpoints. Keep this complete stencil off the HSG coupling surface.
        stencil_lower = lower.copy()
        stencil_upper = upper.copy()
        stencil_lower[normal_axis] = plane_index - 1
        stencil_upper[normal_axis] = plane_index
        if np.any(stencil_lower <= inner) or np.any(stencil_upper >= outer):
            working_lower = tuple(grid.local_to_global(inner))
            working_upper = tuple(grid.local_to_global(outer))
            raise ValueError(
                f"{user_object.params_str()} eigenmode plane and its adjacent "
                "Yee stencil must lie strictly inside the subgrid working region "
                f"from {working_lower} m to {working_upper} m; it cannot touch "
                "the HSG coupling surface or enter the auxiliary/PML region."
            )

    return full_lower, full_upper, lower, upper, plane_index


def _configure_eigenmode_runtime_coordinates(
    runtime,
    grid,
    lower,
    upper,
    plane_index,
    transverse_axes,
):
    """Store authoritative global port geometry and rank-local coordinates."""

    global_start = np.asarray(lower[transverse_axes], dtype=np.int32)
    global_stop = np.asarray(upper[transverse_axes], dtype=np.int32)
    runtime.global_transverse_start = global_start
    runtime.global_transverse_stop = global_stop
    runtime.global_plane_index = int(plane_index)

    if hasattr(grid, "global_size"):
        offset = np.asarray(grid.lower_extent, dtype=np.int32)
        runtime.transverse_start = global_start - offset[transverse_axes]
        runtime.transverse_stop = global_stop - offset[transverse_axes]
        runtime.plane_index = int(plane_index - offset[runtime.normal_axis])
        runtime.tfsf_owned_lower = np.asarray(grid.negative_halo_offset, dtype=np.int32)
        runtime.tfsf_owned_upper = np.asarray(grid.size, dtype=np.int32)
        runtime.mpi_coordinator = bool(grid.is_coordinator())
    else:
        runtime.transverse_start = global_start.copy()
        runtime.transverse_stop = global_stop.copy()
        runtime.plane_index = int(plane_index)
        runtime.tfsf_owned_lower = np.zeros(3, dtype=np.int32)
        runtime.tfsf_owned_upper = np.asarray(grid.size + 1, dtype=np.int32)
        runtime.mpi_coordinator = True


class _EigenmodeSourceBuilder(GridUserObject):
    """Internal builder for the active plane defined by an EigenmodePort."""

    @property
    def order(self):
        return 21

    @property
    def hash(self):
        return "#eigenmode_port"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            normal = self.kwargs["normal"]
            direction = self.kwargs["direction"]
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            w = self.kwargs["w"]
            mode_index = int(self.kwargs["mode_index"])
            mode_count = int(self.kwargs.get("mode_count", mode_index))
            port_index = int(self.kwargs["port_index"])
            waveform_id = self.kwargs["waveform_id"]
            dft_start = float(self.kwargs["dft_start"])
            dft_stop = float(self.kwargs["dft_stop"])
            dft_points = int(self.kwargs["dft_points"])
        except KeyError:
            logger.exception(
                f"{self.params_str()} requires normal, direction, p1, p2, w, mode_index, "
                "port_index, frequency or frequencies, waveform_id, dft_start, "
                "dft_stop, and dft_points."
            )
            raise

        frequency = self.kwargs.get("frequency")
        frequencies_arg = self.kwargs.get("frequencies")
        if frequency is not None and frequencies_arg is not None:
            raise ValueError(
                f"{self.params_str()} accepts either frequency or frequencies, not both."
            )
        if frequencies_arg is None:
            if frequency is None:
                raise ValueError(f"{self.params_str()} requires frequency or frequencies.")
            if np.isscalar(frequency):
                frequencies = (float(frequency),)
            else:
                frequencies = tuple(float(value) for value in frequency)
        else:
            frequencies = tuple(float(value) for value in frequencies_arg)
        spectral_threshold = float(self.kwargs.get("spectral_threshold", 1e-3))
        dft_frequencies = tuple(
            float(value)
            for value in self.kwargs.get(
                "dft_frequencies",
                np.linspace(dft_start, dft_stop, dft_points),
            )
        )
        plot_fields = self.kwargs.get("plot_fields")
        if plot_fields is not None and not isinstance(plot_fields, (bool, np.bool_)):
            raise ValueError(f"{self.params_str()} plot_fields must be True, False, or None.")
        if plot_fields is not None:
            plot_fields = bool(plot_fields)

        if normal not in ["x", "y", "z"]:
            logger.exception(f"{self.params_str()} normal must be x, y, or z.")
            raise ValueError

        if direction not in ["+", "-"]:
            logger.exception(f"{self.params_str()} direction must be + or -.")
            raise ValueError

        if mode_index < 1:
            logger.exception(f"{self.params_str()} mode_index must be one or greater.")
            raise ValueError
        if mode_count < mode_index:
            raise ValueError(
                f"{self.params_str()} mode_count must be at least the excited mode_index "
                f"({mode_index})."
            )
        if port_index < 1:
            raise ValueError(f"{self.params_str()} port_index must be one or greater.")

        _validate_eigenmode_dft(self.params_str(), dft_start, dft_stop, dft_points)
        _validate_eigenmode_dft_frequencies(self.params_str(), dft_frequencies)

        if not frequencies:
            raise ValueError(f"{self.params_str()} requires at least one frequency.")
        if any(not np.isfinite(value) or value <= 0 for value in frequencies):
            raise ValueError(
                f"{self.params_str()} frequencies must be finite and greater than zero."
            )
        if any(
            frequencies[index] >= frequencies[index + 1] for index in range(len(frequencies) - 1)
        ):
            raise ValueError(
                f"{self.params_str()} frequencies must be unique and strictly increasing."
            )
        if not 0 < spectral_threshold < 1:
            raise ValueError(
                f"{self.params_str()} spectral_threshold must be between zero and one."
            )

        if not any(x.ID == waveform_id for x in grid.waveforms):
            logger.exception(
                f"{self.params_str()} there is no waveform with the identifier {waveform_id}."
            )
            raise ValueError

        axis_map = {"x": 0, "y": 1, "z": 2}
        normal_axis = axis_map[normal]
        transverse_axes = [axis for axis in range(3) if axis != normal_axis]
        mode = config.get_model_config().mode
        invariant_axis = "xyz".index(mode[-1]) if mode.startswith("2D") else None
        if invariant_axis is not None and normal_axis == invariant_axis:
            logger.exception(
                f"{self.params_str()} in {mode} mode the source normal must be "
                "in-plane and cannot equal the invariant axis."
            )
            raise ValueError

        full_lower = np.zeros(3, dtype=np.float64)
        full_upper = np.zeros(3, dtype=np.float64)
        full_lower[normal_axis] = w
        full_upper[normal_axis] = w
        full_lower[transverse_axes] = p1
        full_upper[transverse_axes] = p2
        full_lower, full_upper, lower, upper, plane_index = _discretise_eigenmode_plane(
            self,
            grid,
            normal_axis,
            transverse_axes,
            full_lower,
            full_upper,
        )
        p1 = tuple(full_lower[transverse_axes])
        p2 = tuple(full_upper[transverse_axes])
        w = float(full_lower[normal_axis])

        domain_size = np.asarray(getattr(grid, "global_size", grid.size), dtype=np.int32)
        if plane_index < 0 or plane_index > domain_size[normal_axis]:
            logger.exception(
                f"{self.params_str()} normal source plane coordinate is outside the grid."
            )
            raise ValueError

        if direction == "+" and plane_index < 1:
            raise ValueError(
                "A positive-direction eigenmode source must be at least "
                "one cell inside the lower domain boundary."
            )

        if np.any(lower[transverse_axes] < 0) or np.any(
            upper[transverse_axes] > domain_size[transverse_axes]
        ):
            logger.exception(f"{self.params_str()} transverse source bounds are outside the grid.")
            raise ValueError

        if np.any(lower[transverse_axes] >= upper[transverse_axes]):
            logger.exception(
                f"{self.params_str()} lower transverse coordinates must be less than upper transverse coordinates."
            )
            raise ValueError

        if invariant_axis is not None and (
            lower[invariant_axis] != 0 or upper[invariant_axis] != domain_size[invariant_axis]
        ):
            logger.exception(
                f"{self.params_str()} in {mode} mode must span the complete "
                "invariant-axis thickness; use inf for both invariant coordinates."
            )
            raise ValueError

        source = EigenmodeSourceUser(grid)
        source.normal = normal
        source.direction = direction
        source.normal_axis = normal_axis
        source.transverse_axes = tuple(transverse_axes)
        source.invariant_axis = invariant_axis
        source.physical_transverse_axis = (
            next(axis for axis in transverse_axes if axis != invariant_axis)
            if invariant_axis is not None
            else None
        )
        if mode.startswith("2D TM"):
            source.domain_polarization = "TM"
        elif mode.startswith("2D TE"):
            source.domain_polarization = "TE"
        _configure_eigenmode_runtime_coordinates(
            source,
            grid,
            lower,
            upper,
            plane_index,
            transverse_axes,
        )
        source.mode_index = mode_index
        source.mode_count = mode_count
        source.port_index = port_index
        source.port_id = f"port{port_index}"
        source.frequency = frequencies[0]
        source.frequencies = frequencies
        source.spectral_threshold = spectral_threshold
        source.plot_fields = plot_fields
        source.dft_start = dft_start
        source.dft_stop = dft_stop
        source.dft_points = dft_points
        source.dft_frequencies = dft_frequencies
        source.waveformID = waveform_id
        source.waveform = next(x for x in grid.waveforms if x.ID == waveform_id)
        source.start = 0
        source.stop = grid.timewindow

        frequency_description = (
            f"frequency {frequencies[0]:g} Hz"
            if len(frequencies) == 1
            else "anchor frequencies " + ", ".join(f"{value:g}" for value in frequencies) + " Hz"
        )
        logger.info(
            f"{self.grid_name(grid)}Eigenmode source with normal {normal}{direction}, "
            f"transverse bounds {p1} m to {p2} m, normal coordinate {w:g} m, "
            f"exciting mode {mode_index}, monitoring {mode_count} mode(s) at port {port_index}, "
            f"{frequency_description}, using waveform {waveform_id} created."
        )

        grid.eigenmodesources.append(source)


class _EigenmodeReceiverBuilder(GridUserObject):
    """Internal builder for a passive plane defined by an EigenmodePort."""

    @property
    def order(self):
        return 22

    @property
    def hash(self):
        return "#eigenmode_port"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            normal = self.kwargs["normal"]
            direction = self.kwargs["direction"]
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            w = self.kwargs["w"]
            mode_count = int(self.kwargs["mode_count"])
            port_index = int(self.kwargs["port_index"])
            port_id = self.kwargs["id"]
            dft_start = float(self.kwargs["dft_start"])
            dft_stop = float(self.kwargs["dft_stop"])
            dft_points = int(self.kwargs["dft_points"])
        except KeyError:
            logger.exception(
                f"{self.params_str()} requires normal, direction, p1, p2, w, mode_count, "
                "port_index, frequency or frequencies, id, dft_start, dft_stop, "
                "and dft_points."
            )
            raise

        frequency = self.kwargs.get("frequency")
        frequencies_arg = self.kwargs.get("frequencies")
        if frequency is not None and frequencies_arg is not None:
            raise ValueError(
                f"{self.params_str()} accepts either frequency or frequencies, not both."
            )
        values = frequencies_arg if frequencies_arg is not None else frequency
        if values is None:
            raise ValueError(f"{self.params_str()} requires frequency or frequencies.")
        frequencies = (
            (float(values),) if np.isscalar(values) else tuple(float(value) for value in values)
        )
        plot_fields = self.kwargs.get("plot_fields")
        dft_frequencies = tuple(
            float(value)
            for value in self.kwargs.get(
                "dft_frequencies",
                np.linspace(dft_start, dft_stop, dft_points),
            )
        )
        if plot_fields is not None and not isinstance(plot_fields, (bool, np.bool_)):
            raise ValueError(f"{self.params_str()} plot_fields must be True, False, or None.")

        if normal not in ("x", "y", "z") or direction not in ("+", "-"):
            raise ValueError(f"{self.params_str()} requires normal x/y/z and direction +/-.")
        if mode_count < 1:
            raise ValueError(f"{self.params_str()} mode_count must be one or greater.")
        if port_index < 1:
            raise ValueError(f"{self.params_str()} port_index must be one or greater.")
        mode_indices = tuple(range(1, mode_count + 1))
        if not frequencies or any(not np.isfinite(value) or value <= 0 for value in frequencies):
            raise ValueError(f"{self.params_str()} frequencies must be finite and positive.")
        if any(
            frequencies[index] >= frequencies[index + 1] for index in range(len(frequencies) - 1)
        ):
            raise ValueError(
                f"{self.params_str()} frequencies must be unique and strictly increasing."
            )
        _validate_eigenmode_dft(self.params_str(), dft_start, dft_stop, dft_points)
        _validate_eigenmode_dft_frequencies(self.params_str(), dft_frequencies)

        axis_map = {"x": 0, "y": 1, "z": 2}
        normal_axis = axis_map[normal]
        transverse_axes = [axis for axis in range(3) if axis != normal_axis]
        domain_mode = config.get_model_config().mode
        invariant_axis = "xyz".index(domain_mode[-1]) if domain_mode.startswith("2D") else None
        if invariant_axis is not None and normal_axis == invariant_axis:
            raise ValueError(
                f"{self.params_str()} in {domain_mode} mode must have an in-plane normal."
            )

        full_lower = np.zeros(3, dtype=np.float64)
        full_upper = np.zeros(3, dtype=np.float64)
        full_lower[normal_axis] = w
        full_upper[normal_axis] = w
        full_lower[transverse_axes] = p1
        full_upper[transverse_axes] = p2
        full_lower, full_upper, lower, upper, plane_index = _discretise_eigenmode_plane(
            self,
            grid,
            normal_axis,
            transverse_axes,
            full_lower,
            full_upper,
        )
        p1 = tuple(full_lower[transverse_axes])
        p2 = tuple(full_upper[transverse_axes])
        w = float(full_lower[normal_axis])

        domain_size = np.asarray(getattr(grid, "global_size", grid.size), dtype=np.int32)
        if plane_index < 0 or plane_index > domain_size[normal_axis]:
            raise ValueError(f"{self.params_str()} receiver plane is outside the grid.")
        if direction == "+" and plane_index < 1:
            raise ValueError(
                f"{self.params_str()} positive-direction receiver needs a lower H plane."
            )
        if np.any(lower[transverse_axes] < 0) or np.any(
            upper[transverse_axes] > domain_size[transverse_axes]
        ):
            raise ValueError(f"{self.params_str()} transverse bounds are outside the grid.")
        if np.any(lower[transverse_axes] >= upper[transverse_axes]):
            raise ValueError(f"{self.params_str()} lower bounds must be less than upper bounds.")
        if invariant_axis is not None and (
            lower[invariant_axis] != 0 or upper[invariant_axis] != domain_size[invariant_axis]
        ):
            raise ValueError(
                f"{self.params_str()} in {domain_mode} mode must span the invariant axis."
            )

        if port_index not in grid.virtual_waveguide_specs and not isinstance(grid, SubGridBaseGrid):
            axis_name = "xyz"[normal_axis]
            face = f"{axis_name}0" if direction == "+" else f"{axis_name}max"
            pml_thickness = grid.pmls["thickness"][face]
            if hasattr(grid, "global_size"):
                # Interior MPI ranks deliberately have zero thickness on
                # their local faces. Recover the physical boundary value
                # before comparing a globally indexed modal plane.
                pml_thickness = max(grid.comm.allgather(pml_thickness))
            adjacent_plane = (
                pml_thickness if direction == "+" else domain_size[normal_axis] - pml_thickness
            )
            report_warning = not hasattr(grid, "is_coordinator") or grid.is_coordinator()
            if pml_thickness == 0 and report_warning:
                logger.warning(
                    f"Eigenmode receiver {port_id!r} is not next to a PML because the "
                    f"{face} face has zero PML thickness. Reflections beyond the port "
                    "can contaminate its S-parameters."
                )
            elif plane_index != adjacent_plane and report_warning:
                logger.warning(
                    f"Eigenmode receiver {port_id!r} is at plane {plane_index}, not next to the "
                    f"{face} PML interface at plane {adjacent_plane}. Reflections beyond the port "
                    "can contaminate its S-parameters."
                )

        receiver = EigenmodeReceiverUser(grid)
        receiver.normal = normal
        receiver.direction = direction
        receiver.normal_axis = normal_axis
        receiver.transverse_axes = tuple(transverse_axes)
        receiver.invariant_axis = invariant_axis
        receiver.physical_transverse_axis = (
            next(axis for axis in transverse_axes if axis != invariant_axis)
            if invariant_axis is not None
            else None
        )
        if domain_mode.startswith("2D TM"):
            receiver.domain_polarization = "TM"
        elif domain_mode.startswith("2D TE"):
            receiver.domain_polarization = "TE"
        _configure_eigenmode_runtime_coordinates(
            receiver,
            grid,
            lower,
            upper,
            plane_index,
            transverse_axes,
        )
        receiver.mode_count = mode_count
        receiver.mode_indices = mode_indices
        receiver.frequency = frequencies[0]
        receiver.frequencies = frequencies
        receiver.plot_fields = None if plot_fields is None else bool(plot_fields)
        receiver.port_index = port_index
        receiver.port_id = port_id
        receiver.dft_start = dft_start
        receiver.dft_stop = dft_stop
        receiver.dft_points = dft_points
        receiver.dft_frequencies = dft_frequencies
        grid.eigenmodereceivers.append(receiver)
        logger.info(
            f"{self.grid_name(grid)}Eigenmode receiver {port_id!r}, normal {normal}{direction}, "
            f"monitoring modes 1-{mode_count} at port {port_index}, "
            f"transverse bounds {p1} m to {p2} m, normal coordinate {w:g} m created."
        )


class Rx(GridUserObject):
    """Specifies output points in the model.

    These are locations where the values of the electric and magnetic field
    components over the numberof iterations of the model will be saved to file.

    Attributes:
        p1: tuple required for position of receiver x, y, z.
        id: optional string used as identifier for receiver.
        outputs: optional list of outputs for receiver. It can be any
                    selection from Ex, Ey, Ez, Hx, Hy, Hz, Ix, Iy, or Iz.
    """

    @property
    def order(self):
        return 7

    @property
    def hash(self):
        return "#rx"

    def __init__(
        self,
        p1: Tuple[float, float, float],
        id: Optional[str] = None,
        outputs: Optional[List[str]] = None,
    ):
        super().__init__(p1=p1, id=id, outputs=outputs)
        # TODO: Can this be removed?
        self.constructor = RxUser

        self.point = p1
        self.id = id
        self.outputs = outputs

    def _create_receiver(self, grid: FDTDGrid, coord: npt.NDArray[np.int32]) -> RxUser:
        r = RxUser()
        r.coord = coord
        r.coordorigin = coord

        if self.id is None:
            uip = self._create_uip(grid)
            x, y, z = uip.discretise_static_point(self.point)
            r.ID = f"{r.__class__.__name__}({x},{y},{z})"
        else:
            r.ID = self.id
        r.study_id = getattr(self, "_study_id", None)

        if self.outputs is None:
            self.outputs = RxUser.defaultoutputs

        self.outputs.sort()
        # Get allowable outputs
        if config.sim_config.general["solver"] in ["cuda", "opencl", "metal"]:
            allowableoutputs = RxUser.allowableoutputs_dev
        else:
            allowableoutputs = RxUser.allowableoutputs
        # Check and add field output names
        for field in self.outputs:
            if field in allowableoutputs:
                r.outputs[field] = np.zeros(
                    grid.iterations, dtype=config.sim_config.dtypes["float_or_double"]
                )
            else:
                raise ValueError(
                    f"{self.params_str()} contains an output "
                    f"type that is not allowable. Allowable "
                    f"outputs in current context are "
                    f"{allowableoutputs}."
                )

        return r

    def build(self, grid: FDTDGrid):
        # Check position of the receiver
        uip = self._create_uip(grid)
        self.point = uip.resolve_inf_point(self.point)
        point_within_grid, discretised_point = uip.check_src_rx_point(self.point, self.params_str())

        if point_within_grid:
            receiver = self._create_receiver(grid, discretised_point)
            grid.add_receiver(receiver)

            x, y, z = uip.round_to_grid_static_point(self.point)
            logger.info(
                f"{self.grid_name(grid)}Receiver at {x:g}m,"
                f" {y:g}m, {z:g}m with output component(s)"
                f" {', '.join(receiver.outputs)} created."
            )


class RxArray(GridUserObject):
    """Defines multiple output points in the model.

    Attributes:
        p1: tuple required for position of first receiver x, y, z.
        p2: tuple required for position of last receiver x, y, z.
        dl: tuple required for receiver spacing dx, dy, dz.
    """

    @property
    def order(self):
        return 8

    @property
    def hash(self):
        return "#rx_array"

    def __init__(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        dl: Tuple[float, float, float],
    ):
        super().__init__(p1=p1, p2=p2, dl=dl)

        self.lower_point = p1
        self.upper_point = p2
        self.dl = dl

    def build(self, grid: FDTDGrid):
        uip = self._create_uip(grid)

        # p1/p2 are each a full 3D point (not a Box-style corner pair) that
        # together define an array's extent along the two real (transverse)
        # axes and its step count. On the invariant axis of an active 2D
        # model, `inf` should behave like a single source/Rx point (sign-
        # based, redirecting to the mode's interior reference layer) rather
        # than the range-style lower/upper rule - a stepped invariant-axis
        # range would otherwise place receivers on the forced-dead wall
        # positions in TE mode. The other two axes use the ordinary lower/
        # upper positional rule to define the array's real extent.
        mode = config.get_model_config().mode
        invariant_axis = "xyz".index(mode[-1]) if mode.startswith("2D") else None

        lower_ranged = uip.resolve_inf_point(self.lower_point, role="lower")
        upper_ranged = uip.resolve_inf_point(self.upper_point, role="upper")
        if invariant_axis is not None:
            lower_single = uip.resolve_inf_point(self.lower_point, role=None)
            upper_single = uip.resolve_inf_point(self.upper_point, role=None)
            self.lower_point = tuple(
                lower_single[a] if a == invariant_axis else lower_ranged[a] for a in range(3)
            )
            self.upper_point = tuple(
                upper_single[a] if a == invariant_axis else upper_ranged[a] for a in range(3)
            )
        else:
            self.lower_point = lower_ranged
            self.upper_point = upper_ranged

        _, discretised_lower_point = uip.check_src_rx_point(
            self.lower_point, self.params_str(), "lower"
        )
        _, discretised_upper_point = uip.check_src_rx_point(
            self.upper_point, self.params_str(), "upper"
        )
        discretised_dl = uip.discretise_static_point(self.dl)

        if any(discretised_lower_point > discretised_upper_point):
            raise ValueError(
                f"{self.params_str()} the lower coordinates should be less than the upper coordinates."
            )
        if any(discretised_dl < 0):
            raise ValueError(f"{self.params_str()} the step size should not be less than zero.")

        discretised_dl = np.where(discretised_dl == 0, 1, discretised_dl)

        if any(discretised_dl < 1):
            raise ValueError(
                f"{self.params_str()} the step size should not be less than the spatial discretisation."
            )

        xs, ys, zs = uip.round_to_grid_static_point(self.lower_point)
        xf, yf, zf = uip.round_to_grid_static_point(self.upper_point)
        # Use discretised_dl (already corrected to a minimum of 1 cell on
        # any axis given dl=0, a common "single row along this axis"
        # pattern) rather than re-deriving from the raw self.dl, which
        # still contains the uncorrected 0 - previously caused
        # np.arange()'s internal division to divide by zero below.
        # grid.dl is whichever grid this was built against (main grid or a
        # subgrid, each with their own .dl - see SubGridBase.
        # set_discretisation()), matching what round_to_grid_static_point()
        # itself uses internally (self.grid.dl, where uip.grid is grid).
        dx, dy, dz = discretised_dl * grid.dl

        logger.info(
            f"{self.grid_name(grid)}Receiver array"
            f" {xs:g}m, {ys:g}m, {zs:g}m, to"
            f" {xf:g}m, {yf:g}m, {zf:g}m with steps"
            f" {dx:g}m, {dy:g}m, {dz:g}m"
        )

        for x in np.arange(xs, xf + grid.dx, dx):
            for y in np.arange(ys, yf + grid.dy, dy):
                for z in np.arange(zs, zf + grid.dz, dz):
                    receiver = Rx((x, y, z))
                    receiver.build(grid)


class Material(GridUserObject):
    """Specifies a material in the model described by a set of constitutive
        parameters.

    Attributes:
        er: float required for the relative electric permittivity.
        se: float required for the electric conductivity (Siemens/metre).
        mr: float required for the relative magnetic permeability.
        sm: float required for the magnetic loss.
        id: string used as identifier for material.
    """

    @property
    def order(self):
        return 10

    @property
    def hash(self):
        return "#material"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            er = self.kwargs["er"]
            se = self.kwargs["se"]
            mr = self.kwargs["mr"]
            sm = self.kwargs["sm"]
            material_id = self.kwargs["id"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires exactly five parameters.")
            raise

        try:
            er = float(er)
            se = float(se)
            mr = float(mr)
            sm = float(sm)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.params_str()} electromagnetic properties must be numeric."
            ) from exc

        if not np.isfinite(er) or er < 1:
            logger.exception(
                f"{self.params_str()} requires a finite value of one or greater for static (DC) permittivity."
            )
            raise ValueError
        # Positive infinity is the documented representation of a PEC.
        if np.isnan(se) or se < 0:
            logger.exception(
                f"{self.params_str()} requires a non-negative value for electric conductivity."
            )
            raise ValueError
        if not np.isfinite(mr) or mr < 1:
            logger.exception(
                f"{self.params_str()} requires a finite value of one or greater for magnetic permeability."
            )
            raise ValueError
        # Positive infinity is the documented representation of a PMC.
        if np.isnan(sm) or sm < 0:
            logger.exception(f"{self.params_str()} requires a non-negative value for magnetic loss.")
            raise ValueError

        if (
            is_reserved_impedance_id(material_id)
            or any(x.ID == material_id for x in grid.materials)
            or material_id in getattr(grid, "surface_impedance_models", {})
        ):
            logger.exception(f"{self.params_str()} with ID {material_id} already exists")
            raise ValueError

        # Create a new instance of the Material class material
        # (start index after pec & free_space)
        m = MaterialUser(len(grid.materials), material_id)
        m.se = se
        m.mr = mr
        m.sm = sm

        # Perfect electric and magnetic conductors cannot participate in
        # dielectric smoothing at a material interface.
        if m.se == float("inf") or m.sm == float("inf"):
            m.averagable = False

        m.er = er
        logger.info(
            f"{self.grid_name(grid)}Material {m.ID} with eps_r={m.er:g}, "
            f"sigma={m.se:g} S/m; mu_r={m.mr:g}, sigma*={m.sm:g} Ohm/m "
            f"created."
        )

        grid.materials.append(m)


class MaterialFromDatabase(GridUserObject):
    """Create a material from a versioned JSON material database.

    Official databases are installed with gprMax. A user database is a JSON
    file named ``<database>.json`` beside the input file, or in the current
    working directory for a direct Python API model.

    Attributes:
        database: official or local database name.
        material: material entry key in the database.
        id: optional local material ID; defaults to ``material``. This can
            provide a model-specific name, avoid a collision, or match the
            material name expected by imported geometry.
    """

    @property
    def order(self):
        return 10

    @property
    def hash(self):
        return "#material_from_database"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            database = self.kwargs["database"]
            material_key = self.kwargs["material"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires a database and material key")
            raise
        material_id = self.kwargs.get("id") or material_key

        # Hash-command models resolve local databases beside the input file;
        # direct API models deliberately use the execution directory even if
        # their output file is placed somewhere else.
        inputfile = getattr(config.sim_config.args, "inputfile", None)
        search_directory = config.sim_config.input_file_path.parent if inputfile else Path.cwd()
        spec = load_material_spec(
            database,
            material_key,
            search_directory=search_directory,
        )
        created = build_material_from_spec(grid, spec, material_id)
        logger.info(
            f"{self.grid_name(grid)}Material {created.ID} created from "
            f"{spec.source.database_id}:{spec.key} "
            f"(database version {spec.source.database_version}, "
            f"entry SHA-256 {spec.entry_sha256[:12]}...)."
        )


class MaterialDensity(GridUserObject):
    """Assign a physical mass density to one or more existing materials.

    Density is cell-centred metadata used by derived quantities such as SAR;
    it does not alter electromagnetic update coefficients or material
    averaging.

    Attributes:
        density: Finite mass density greater than zero in kg/m3.
        material_ids: Material identifiers to which the density is assigned.
    """

    @property
    def order(self):
        # Run after every dispersion modifier. Those commands replace the
        # original Material instance while retaining its numeric ID.
        return 14

    @property
    def hash(self):
        return "#material_density"

    def __init__(self, *, density, material_ids):
        super().__init__(density=density, material_ids=material_ids)

    def build(self, grid: FDTDGrid):
        try:
            density = float(self.kwargs["density"])
            raw_material_ids = self.kwargs["material_ids"]
            material_ids = (
                [raw_material_ids] if isinstance(raw_material_ids, str) else list(raw_material_ids)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.params_str()} requires a density and at least one material identifier"
            ) from exc

        if not np.isfinite(density) or density <= 0:
            raise ValueError(
                f"{self.params_str()} density must be finite and greater than zero kg/m3"
            )
        if not material_ids or any(not isinstance(item, str) or not item for item in material_ids):
            raise ValueError(
                f"{self.params_str()} requires at least one non-empty material identifier"
            )
        if len(set(material_ids)) != len(material_ids):
            raise ValueError(f"{self.params_str()} material identifiers must not be duplicated")

        by_id = {material.ID: material for material in grid.materials}
        missing = [material_id for material_id in material_ids if material_id not in by_id]
        if missing:
            raise ValueError(f"{self.params_str()} material(s) {missing} do not exist")

        for material_id in material_ids:
            by_id[material_id].mass_density = density

        logger.info(
            f"{self.grid_name(grid)}Mass density {density:g} kg/m3 assigned to material(s) "
            f"{', '.join(material_ids)}."
        )


def _reject_perfect_conductor_dispersion(user_object, materials, formulation):
    """Reject electric dispersion on ideal electric or magnetic conductors."""

    invalid = []
    for material in materials:
        conductor_types = []
        if material.is_pec:
            conductor_types.append("PEC")
        if material.is_pmc:
            conductor_types.append("PMC")
        if conductor_types:
            invalid.append(f"{material.ID!r} ({'/'.join(conductor_types)})")

    if invalid:
        message = (
            f"{user_object.params_str()} cannot add {formulation} electric dispersion to "
            f"perfect conductor material(s): {', '.join(invalid)}"
        )
        logger.error(message)
        raise ValueError(message)


def _validate_dispersion_definition(user_object, poles, material_ids, **pole_parameters):
    """Validate the common structure of an API dispersive-material request."""

    try:
        poles = operator.index(poles)
    except TypeError as exc:
        raise ValueError(f"{user_object.params_str()} requires an integer number of poles") from exc

    if poles <= 0:
        raise ValueError(f"{user_object.params_str()} requires a positive number of poles")
    if not material_ids:
        raise ValueError(f"{user_object.params_str()} requires at least one material identifier")

    for name, values in pole_parameters.items():
        if len(values) != poles:
            raise ValueError(
                f"{user_object.params_str()} requires exactly {poles} {name} value(s), "
                "one for each pole"
            )

    return poles


class AddDebyeDispersion(GridUserObject):
    """Adds dispersive properties to already defined Material based on a
    multi-pole Debye formulation.

    Perfect electric and magnetic conductors cannot be assigned electric
    dispersion, including custom materials with infinite electric or magnetic
    conductivity.

    Attributes:
        poles: float required for number of Debye poles.
        er_delta: tuple required for difference between zero-frequency relative
                    permittivity and relative permittivity at infinite frequency
                    for each pole.
        tau: tuple required for relaxation time (secs) for each pole.
        material_ids: list required of material ids to apply dispersive
                        properties.
    """

    @property
    def order(self):
        return 11

    @property
    def hash(self):
        return "#add_dispersion_debye"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            poles = self.kwargs["poles"]
            er_delta = self.kwargs["er_delta"]
            tau = self.kwargs["tau"]
            material_ids = self.kwargs["material_ids"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires at least four parameters.")
            raise

        poles = _validate_dispersion_definition(
            self,
            poles,
            material_ids,
            er_delta=er_delta,
            tau=tau,
        )

        # Look up requested materials in existing list of material instances
        materials = [y for x in material_ids for y in grid.materials if y.ID == x]

        if len(materials) != len(material_ids):
            found_ids = {material.ID for material in materials}
            notfound = [material_id for material_id in material_ids if material_id not in found_ids]
            message = f"{self.params_str()} material(s) {notfound} do not exist"
            logger.error(message)
            raise ValueError(message)

        _reject_perfect_conductor_dispersion(self, materials, "Debye")

        for material in materials:
            disp_material = DispersiveMaterialUser(material.numID, material.ID)
            disp_material.er = material.er
            disp_material.se = material.se
            disp_material.mr = material.mr
            disp_material.sm = material.sm
            disp_material.mass_density = material.mass_density
            disp_material.type = "debye"
            disp_material.poles = poles
            disp_material.averagable = config.get_model_config().dispersive_averaging
            for i in range(poles):
                if not np.isfinite(er_delta[i]) or er_delta[i] <= 0:
                    raise ValueError(
                        f"{self.params_str()} requires finite, positive "
                        f"relative-permittivity differences (invalid pole {i + 1})"
                    )
                if not np.isfinite(tau[i]) or tau[i] <= 0:
                    raise ValueError(
                        f"{self.params_str()} requires finite, positive relaxation times "
                        f"(invalid pole {i + 1})"
                    )
                logger.debug("Not checking if relaxation times are greater than time-step.")
                disp_material.deltaer.append(er_delta[i])
                disp_material.tau.append(tau[i])
            if disp_material.poles > config.get_model_config().materials["maxpoles"]:
                config.get_model_config().materials["maxpoles"] = disp_material.poles

            # Replace original material with newly created DispersiveMaterial
            grid.materials = [
                disp_material if mat.numID == material.numID else mat for mat in grid.materials
            ]

            logger.info(
                f"{self.grid_name(grid)}Debye disperion added to {disp_material.ID} "
                f"with delta_eps_r={', '.join(f'{deltaer:4.2f}' for deltaer in disp_material.deltaer)}, "
                f"and tau={', '.join(f'{tau:4.3e}' for tau in disp_material.tau)} secs created."
            )


class AddLorentzDispersion(GridUserObject):
    """Adds dispersive properties to already defined Material based on a
    multi-pole Lorentz formulation.

    Perfect electric and magnetic conductors cannot be assigned electric
    dispersion, including custom materials with infinite electric or magnetic
    conductivity.

    Attributes:
        poles: float required for number of Lorentz poles.
        er_delta: tuple required for difference between zero-frequency relative
                    permittivity and relative permittivity at infinite frequency
                    for each pole.
        omega: tuple required for resonance frequency (Hz) for each pole.
        delta: tuple required for damping coefficient (per second) for each pole.
        material_ids: list required of material ids to apply dispersive
                        properties.
    """

    @property
    def order(self):
        return 12

    @property
    def hash(self):
        return "#add_dispersion_lorentz"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            poles = self.kwargs["poles"]
            er_delta = self.kwargs["er_delta"]
            omega = self.kwargs["omega"]
            delta = self.kwargs["delta"]
            material_ids = self.kwargs["material_ids"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires at least five parameters.")
            raise

        poles = _validate_dispersion_definition(
            self,
            poles,
            material_ids,
            er_delta=er_delta,
            omega=omega,
            delta=delta,
        )

        # Look up requested materials in existing list of material instances
        materials = [y for x in material_ids for y in grid.materials if y.ID == x]

        if len(materials) != len(material_ids):
            found_ids = {material.ID for material in materials}
            notfound = [material_id for material_id in material_ids if material_id not in found_ids]
            message = f"{self.params_str()} material(s) {notfound} do not exist"
            logger.error(message)
            raise ValueError(message)

        _reject_perfect_conductor_dispersion(self, materials, "Lorentz")

        for material in materials:
            disp_material = DispersiveMaterialUser(material.numID, material.ID)
            disp_material.er = material.er
            disp_material.se = material.se
            disp_material.mr = material.mr
            disp_material.sm = material.sm
            disp_material.mass_density = material.mass_density
            disp_material.type = "lorentz"
            disp_material.poles = poles
            disp_material.averagable = config.get_model_config().dispersive_averaging
            for i in range(poles):
                if not np.isfinite(er_delta[i]) or er_delta[i] <= 0:
                    raise ValueError(
                        f"{self.params_str()} requires finite, positive "
                        f"relative-permittivity differences (invalid pole {i + 1})"
                    )
                try:
                    validate_lorentz_pole(omega[i], delta[i], grid.dt)
                except ValueError as exc:
                    raise ValueError(f"{self.params_str()}: {exc}") from exc
                disp_material.deltaer.append(er_delta[i])
                disp_material.tau.append(omega[i])
                disp_material.alpha.append(delta[i])
            if disp_material.poles > config.get_model_config().materials["maxpoles"]:
                config.get_model_config().materials["maxpoles"] = disp_material.poles

            # Replace original material with newly created DispersiveMaterial
            grid.materials = [
                disp_material if mat.numID == material.numID else mat for mat in grid.materials
            ]

            logger.info(
                f"{self.grid_name(grid)}Lorentz disperion added to {disp_material.ID} "
                f"with delta_eps_r={', '.join(f'{deltaer:4.2f}' for deltaer in disp_material.deltaer)}, "
                f"omega={', '.join(f'{omega:4.3e}' for omega in disp_material.tau)} Hertz, "
                f"and delta={', '.join(f'{delta:4.3e}' for delta in disp_material.alpha)} per second, created."
            )


class AddDrudeDispersion(GridUserObject):
    """Adds dispersive properties to already defined Material based on a
    multi-pole Drude formulation.

    Perfect electric and magnetic conductors cannot be assigned electric
    dispersion, including custom materials with infinite electric or magnetic
    conductivity.

    Attributes:
        poles: float required for number of Drude poles.
        omega: tuple required for plasma frequency (Hz) for each pole.
        alpha: tuple required for inverse of relaxation time (per second) for each pole.
        material_ids: list required of material ids to apply dispersive
                        properties.
    """

    @property
    def order(self):
        return 13

    @property
    def hash(self):
        return "#add_dispersion_drude"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            poles = self.kwargs["poles"]
            omega = self.kwargs["omega"]
            alpha = self.kwargs["alpha"]
            material_ids = self.kwargs["material_ids"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires at least four parameters.")
            raise

        poles = _validate_dispersion_definition(
            self,
            poles,
            material_ids,
            omega=omega,
            alpha=alpha,
        )

        # Look up requested materials in existing list of material instances
        materials = [y for x in material_ids for y in grid.materials if y.ID == x]

        if len(materials) != len(material_ids):
            found_ids = {material.ID for material in materials}
            notfound = [material_id for material_id in material_ids if material_id not in found_ids]
            message = f"{self.params_str()} material(s) {notfound} do not exist"
            logger.error(message)
            raise ValueError(message)

        _reject_perfect_conductor_dispersion(self, materials, "Drude")

        for material in materials:
            disp_material = DispersiveMaterialUser(material.numID, material.ID)
            disp_material.er = material.er
            disp_material.se = material.se
            disp_material.mr = material.mr
            disp_material.sm = material.sm
            disp_material.mass_density = material.mass_density
            disp_material.type = "drude"
            disp_material.poles = poles
            disp_material.averagable = config.get_model_config().dispersive_averaging
            for i in range(poles):
                try:
                    validate_drude_pole(omega[i], alpha[i], grid.dt)
                except ValueError as exc:
                    raise ValueError(f"{self.params_str()}: {exc}") from exc
                disp_material.tau.append(omega[i])
                disp_material.alpha.append(alpha[i])
            if disp_material.poles > config.get_model_config().materials["maxpoles"]:
                config.get_model_config().materials["maxpoles"] = disp_material.poles

            # Replace original material with newly created DispersiveMaterial
            grid.materials = [
                disp_material if mat.numID == material.numID else mat for mat in grid.materials
            ]

            logger.info(
                f"{self.grid_name(grid)}Drude disperion added to {disp_material.ID} "
                f"with omega={', '.join(f'{omega:4.3e}' for omega in disp_material.tau)} Hertz, "
                f"and alpha={', '.join(f'{alpha:4.3e}' for alpha in disp_material.alpha)} per second created."
            )


class SoilPeplinski(GridUserObject):
    """Mixing model for soils proposed by Peplinski et al.
        (http://dx.doi.org/10.1109/36.387598)

    Attributes:
        sand_fraction: float required for sand fraction of soil.
        clay_fraction: float required for clay of soil.
        bulk_density: float required for bulk density of soil (gm/cm^3).
        sand_density: float required for density of sand particles in soil (gm/cm^3).
        water_fraction_lower: float required for lower boundary of volumetric
                                water fraction of the soil.
        water_fraction_upper: float required for upper boundary of volumetric
                                water fraction of the soil.
        id: string used as identifier for soil.
    """

    @property
    def order(self):
        return 14

    @property
    def hash(self):
        return "#soil_peplinski"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            sand_fraction = self.kwargs["sand_fraction"]
            clay_fraction = self.kwargs["clay_fraction"]
            bulk_density = self.kwargs["bulk_density"]
            sand_density = self.kwargs["sand_density"]
            water_fraction_lower = self.kwargs["water_fraction_lower"]
            water_fraction_upper = self.kwargs["water_fraction_upper"]
            ID = self.kwargs["id"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires at exactly seven parameters.")
            raise

        values = (
            sand_fraction,
            clay_fraction,
            bulk_density,
            sand_density,
            water_fraction_lower,
            water_fraction_upper,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{self.params_str()} soil parameters must all be finite")

        # sand_fraction/clay_fraction are physical fractions - values
        # above 1 were previously unvalidated.
        if not (0 <= sand_fraction <= 1):
            logger.exception(
                f"{self.params_str()} requires the sand fraction to be between 0 and 1."
            )
            raise ValueError
        if not (0 <= clay_fraction <= 1):
            logger.exception(
                f"{self.params_str()} requires the clay fraction to be between 0 and 1."
            )
            raise ValueError
        if bulk_density < 0:
            logger.exception(f"{self.params_str()} requires a positive value for the bulk density.")
            raise ValueError
        # sand_density is a divisor in PeplinskiSoil.calculate_properties()
        # (materials.py) - zero would raise a ZeroDivisionError there,
        # not here, at build time.
        if sand_density <= 0:
            logger.exception(
                f"{self.params_str()} requires a value greater than zero for the sand particle "
                "density."
            )
            raise ValueError
        if water_fraction_lower < 0:
            logger.exception(
                f"{self.params_str()} requires a positive value for the lower limit of the water volumetric "
                "fraction."
            )
            raise ValueError
        # water_fraction_upper is also a divisor (via the bin midpoints
        # calculate_properties() generates between the lower/upper
        # limits) - a reversed or all-zero range would produce a zero (or
        # negative-width) bin range and risk division by zero.
        if water_fraction_upper <= 0:
            logger.exception(
                f"{self.params_str()} requires a value greater than zero for the upper limit of "
                "the water volumetric fraction."
            )
            raise ValueError
        if water_fraction_lower > water_fraction_upper:
            logger.exception(
                f"{self.params_str()} requires the lower limit of the water volumetric fraction "
                "to be less than or equal to the upper limit."
            )
            raise ValueError
        if any(x.ID == ID for x in grid.mixingmodels):
            logger.exception(f"{self.params_str()} with ID {ID} already exists")
            raise ValueError

        # Create a new instance of the Material class material
        # (start index after pec & free_space)
        s = PeplinskiSoilUser(
            ID,
            sand_fraction,
            clay_fraction,
            bulk_density,
            sand_density,
            (water_fraction_lower, water_fraction_upper),
        )

        logger.info(
            f"{self.grid_name(grid)}Mixing model (Peplinski) used to "
            f"create {s.ID} with sand fraction {s.S:g}, clay fraction "
            f"{s.C:g}, bulk density {s.rb:g}g/cm3, sand particle "
            f"density {s.rs:g}g/cm3, and water volumetric fraction "
            f"{s.mu[0]:g} to {s.mu[1]:g} created."
        )

        grid.mixingmodels.append(s)


class MaterialRange(GridUserObject):
    """Creates varying material properties for stochastic models.

    Attributes:
        er_lower: float required for lower relative permittivity value.
        er_upper: float required for upper relative permittivity value.
        sigma_lower: float required for lower conductivity value.
        sigma_upper: float required for upper conductivity value.
        mr_lower: float required for lower relative magnetic permeability value.
        mr_upper: float required for upper relative magnetic permeability value.
        ro_lower: float required for lower magnetic loss value.
        ro_upper: float required for upper magnetic loss value.
        id: string used as identifier for this variable material.
    """

    @property
    def order(self):
        return 15

    @property
    def hash(self):
        return "#material_range"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            er_lower = self.kwargs["er_lower"]
            er_upper = self.kwargs["er_upper"]
            sigma_lower = self.kwargs["sigma_lower"]
            sigma_upper = self.kwargs["sigma_upper"]
            mr_lower = self.kwargs["mr_lower"]
            mr_upper = self.kwargs["mr_upper"]
            ro_lower = self.kwargs["ro_lower"]
            ro_upper = self.kwargs["ro_upper"]
            ID = self.kwargs["id"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires at exactly nine parameters.")
            raise

        values = (
            er_lower,
            er_upper,
            sigma_lower,
            sigma_upper,
            mr_lower,
            mr_upper,
            ro_lower,
            ro_upper,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{self.params_str()} material-range limits must all be finite")

        if er_lower < 1:
            logger.exception(
                f"{self.params_str()} requires a value greater or equal to 1 "
                "for the lower range of relative permittivity."
            )
            raise ValueError
        if mr_lower < 1:
            logger.exception(
                f"{self.params_str()} requires a value greater or equal to 1 "
                "for the lower range of relative magnetic permeability."
            )
            raise ValueError
        if sigma_lower < 0:
            logger.exception(
                f"{self.params_str()} requires a positive value for the lower limit of conductivity."
            )
            raise ValueError
        if ro_lower < 0:
            logger.exception(
                f"{self.params_str()} requires a positive value for the lower range magnetic loss."
            )
            raise ValueError
        if er_upper < 1:
            logger.exception(
                f"{self.params_str()} requires a value greater or equal to 1"
                "for the upper range of relative permittivity."
            )
            raise ValueError
        if mr_upper < 1:
            logger.exception(
                f"{self.params_str()} requires a value greater or equal to 1"
                "for the upper range of relative magnetic permeability"
            )
            raise ValueError
        if sigma_upper < 0:
            logger.exception(
                f"{self.params_str()} requires a positive value for the upper range of conductivity."
            )
            raise ValueError
        if ro_upper < 0:
            logger.exception(
                f"{self.params_str()} requires a positive value for the upper range of magnetic loss."
            )
            raise ValueError
        ranges = (
            ("relative permittivity", er_lower, er_upper),
            ("electric conductivity", sigma_lower, sigma_upper),
            ("relative magnetic permeability", mr_lower, mr_upper),
            ("magnetic loss", ro_lower, ro_upper),
        )
        for label, lower, upper in ranges:
            if lower > upper:
                raise ValueError(
                    f"{self.params_str()} lower {label} limit must not exceed its upper limit"
                )
        if any(x.ID == ID for x in grid.mixingmodels):
            logger.exception(f"{self.params_str()} with ID {ID} already exists")
            raise ValueError

        s = RangeMaterialUser(
            ID,
            (er_lower, er_upper),
            (sigma_lower, sigma_upper),
            (mr_lower, mr_upper),
            (ro_lower, ro_upper),
        )

        logger.info(
            f"{self.grid_name(grid)}Material properties used to "
            f"create {s.ID} with range(s) {s.er[0]:g} to {s.er[1]:g}, relative permittivity "
            f"{s.sig[0]:g} to {s.sig[1]:g}, S/m conductivity, {s.mu[0]:g} to {s.mu[1]:g} relative magnetic permeability "
            f"{s.ro[0]:g} to {s.ro[1]:g} Ohm/m magnetic loss, created"
        )

        grid.mixingmodels.append(s)


class MaterialList(GridUserObject):
    """Creates varying material properties for stochastic models.

    Attributes:
        list_of_materials: list of material IDs
        id: string used as identifier for this variable material.
    """

    @property
    def order(self):
        return 15

    @property
    def hash(self):
        return "#material_list"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            list_of_materials = self.kwargs["list_of_materials"]
            ID = self.kwargs["id"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires at at least 2 parameters.")
            raise
        if any(x.ID == ID for x in grid.mixingmodels):
            logger.exception(f"{self.params_str()} with ID {ID} already exists")
            raise ValueError

        s = ListMaterialUser(ID, list_of_materials)

        logger.info(
            f"{self.grid_name(grid)}A list of materials used to create {s.ID} that includes {s.mat}, created"
        )

        grid.mixingmodels.append(s)


class MaterialCrim(GridUserObject):
    """Mixing model based on the Complex Refractive Index Model (CRIM),
    combining a fixed-fraction non-dispersive matrix material with a
    single-pole Debye dispersive material (e.g. water or brine); the
    remaining volume fraction is assumed to be air.

    Attributes:
        matrix_id: string for ID of an existing non-dispersive material used
                    for the fixed-fraction matrix (solid) phase.
        matrix_fraction: float required for fixed volumetric fraction of the
                            matrix phase.
        dispersive_id: string for ID of an existing single-pole Debye
                        material used for the dispersive phase.
        fraction_lower: float required for lower boundary of the volumetric
                            fraction of the dispersive phase.
        fraction_upper: float required for upper boundary of the volumetric
                            fraction of the dispersive phase.
        f_min: float required for lower bound of the frequency range (Hz)
                used to fit the CRIM mixing curve.
        f_max: float required for upper bound of the frequency range (Hz)
                used to fit the CRIM mixing curve.
        a: float required for the CRIM shape factor.
        id: string used as identifier for this CRIM mixing model.
    """

    @property
    def order(self):
        return 15

    @property
    def hash(self):
        return "#material_crim"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            matrix_id = self.kwargs["matrix_id"]
            matrix_fraction = float(self.kwargs["matrix_fraction"])
            dispersive_id = self.kwargs["dispersive_id"]
            fraction_lower = float(self.kwargs["fraction_lower"])
            fraction_upper = float(self.kwargs["fraction_upper"])
            f_min = float(self.kwargs["f_min"])
            f_max = float(self.kwargs["f_max"])
            a = float(self.kwargs["a"])
            ID = self.kwargs["id"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.params_str()} requires exactly nine parameters with numeric fractions, "
                "frequencies, and shape factor"
            ) from exc

        if not np.all(
            np.isfinite([matrix_fraction, fraction_lower, fraction_upper, f_min, f_max, a])
        ):
            raise ValueError(f"{self.params_str()} requires all numeric parameters to be finite")

        if not (0 <= matrix_fraction <= 1):
            logger.exception(
                f"{self.params_str()} requires the matrix fraction to be between 0 and 1."
            )
            raise ValueError
        if fraction_lower < 0:
            logger.exception(
                f"{self.params_str()} requires a positive value for the lower limit of the "
                "dispersive volumetric fraction."
            )
            raise ValueError
        if fraction_upper <= 0:
            logger.exception(
                f"{self.params_str()} requires a value greater than zero for the upper limit of "
                "the dispersive volumetric fraction."
            )
            raise ValueError
        if fraction_lower > fraction_upper:
            logger.exception(
                f"{self.params_str()} requires the lower limit of the dispersive volumetric "
                "fraction to be less than or equal to the upper limit."
            )
            raise ValueError
        if matrix_fraction + fraction_upper > 1:
            logger.exception(
                f"{self.params_str()} requires the matrix fraction plus the upper limit of the "
                "dispersive volumetric fraction to not exceed 1."
            )
            raise ValueError
        if f_min <= 0 or f_max <= 0:
            logger.exception(f"{self.params_str()} requires positive values for f_min and f_max.")
            raise ValueError
        if f_min >= f_max:
            logger.exception(f"{self.params_str()} requires f_min to be less than f_max.")
            raise ValueError
        if a <= 0:
            logger.exception(f"{self.params_str()} requires a positive value for the shape factor.")
            raise ValueError
        if any(x.ID == ID for x in grid.mixingmodels):
            logger.exception(f"{self.params_str()} with ID {ID} already exists")
            raise ValueError

        s = CrimMixtureUser(
            ID,
            matrix_id,
            matrix_fraction,
            dispersive_id,
            fraction_lower,
            fraction_upper,
            f_min,
            f_max,
            a,
        )

        logger.info(
            f"{self.grid_name(grid)}Mixing model (CRIM) used to create {s.ID} with matrix "
            f"material {matrix_id!r} (fraction {matrix_fraction:g}), dispersive material "
            f"{dispersive_id!r} (fraction {fraction_lower:g} to {fraction_upper:g}), shape "
            f"factor {a:g}, created."
        )

        grid.mixingmodels.append(s)


class PMLCFS(GridUserObject):
    """Controls parameters that are used to build each order of PML. Default
        values are set in pml.py

    Attributes:
        alphascalingprofile: string required for type of scaling to use for
                                CFS alpha parameter.
        alphascalingdirection: string required for direction of scaling to use
                                for CFS alpha parameter.
        alphamin: float required for minimum value for the CFS alpha parameter.
        alphamax: float required for maximum value for the CFS alpha parameter.
        kappascalingprofile: string required for type of scaling to use for
                                CFS kappa parameter.
        kappascalingdirection: string required for direction of scaling to use
                                for CFS kappa parameter.
        kappamin: float required for minimum value for the CFS kappa parameter.
        kappamax: float required for maximum value for the CFS kappa parameter.
        sigmascalingprofile: string required for type of scaling to use for
                                CFS sigma parameter.
        sigmascalingdirection: string required for direction of scaling to use
                                for CFS sigma parameter.
        sigmamin: float required for minimum value for the CFS sigma parameter.
        sigmamax: float required for maximum value for the CFS sigma parameter.
        profile_id: optional reusable PML-profile identifier. If omitted, the
            parameters modify the global domain-PML configuration.
    """

    @property
    def order(self):
        return 19

    @property
    def hash(self):
        return "#pml_cfs"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            alphascalingprofile = self.kwargs["alphascalingprofile"]
            alphascalingdirection = self.kwargs["alphascalingdirection"]
            alphamin = self.kwargs["alphamin"]
            alphamax = self.kwargs["alphamax"]
            kappascalingprofile = self.kwargs["kappascalingprofile"]
            kappascalingdirection = self.kwargs["kappascalingdirection"]
            kappamin = self.kwargs["kappamin"]
            kappamax = self.kwargs["kappamax"]
            sigmascalingprofile = self.kwargs["sigmascalingprofile"]
            sigmascalingdirection = self.kwargs["sigmascalingdirection"]
            sigmamin = self.kwargs["sigmamin"]
            sigmamax = self.kwargs["sigmamax"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires exactly twelve parameters.")
            raise

        if (
            alphascalingprofile not in CFSParameter.scalingprofiles.keys()
            or kappascalingprofile not in CFSParameter.scalingprofiles.keys()
            or sigmascalingprofile not in CFSParameter.scalingprofiles.keys()
        ):
            logger.exception(
                f"{self.params_str()} must have scaling type {','.join(CFSParameter.scalingprofiles.keys())}"
            )
            raise ValueError
        if (
            alphascalingdirection not in CFSParameter.scalingdirections
            or kappascalingdirection not in CFSParameter.scalingdirections
            or sigmascalingdirection not in CFSParameter.scalingdirections
        ):
            logger.exception(
                f"{self.params_str()} must have scaling type {','.join(CFSParameter.scalingdirections)}"
            )
            raise ValueError

        if isinstance(sigmamax, str) and sigmamax.lower() == "none":
            sigmamax = None

        if (
            float(alphamin) < 0
            or float(alphamax) < 0
            or float(kappamin) < 0
            or float(kappamax) < 0
            or float(sigmamin) < 0
            or (sigmamax is not None and float(sigmamax) < 0)
        ):
            message = (
                f"{self.params_str()} minimum and maximum scaling values must be zero or greater."
            )
            logger.error(message)
            raise ValueError(message)

        cfsalpha = CFSParameter()
        cfsalpha.ID = "alpha"
        cfsalpha.scalingprofile = alphascalingprofile
        cfsalpha.scalingdirection = alphascalingdirection
        cfsalpha.min = float(alphamin)
        cfsalpha.max = float(alphamax)
        cfskappa = CFSParameter()
        cfskappa.ID = "kappa"
        cfskappa.scalingprofile = kappascalingprofile
        cfskappa.scalingdirection = kappascalingdirection
        cfskappa.min = float(kappamin)
        cfskappa.max = float(kappamax)
        cfssigma = CFSParameter()
        cfssigma.ID = "sigma"
        cfssigma.scalingprofile = sigmascalingprofile
        cfssigma.scalingdirection = sigmascalingdirection
        cfssigma.min = float(sigmamin)
        if sigmamax is not None:
            sigmamax = float(sigmamax)
        cfssigma.max = sigmamax
        cfs = CFS()
        cfs.alpha = cfsalpha
        cfs.kappa = cfskappa
        cfs.sigma = cfssigma

        profile_id = self.kwargs.get("profile_id")
        destination = (
            "global PML configuration" if profile_id is None else f"PML profile '{profile_id}'"
        )
        logger.info(
            f"{destination} CFS parameters: alpha (scaling: {cfsalpha.scalingprofile}, "
            f"scaling direction: {cfsalpha.scalingdirection}, min: "
            f"{cfsalpha.min:g}, max: {cfsalpha.max:g}), kappa (scaling: "
            f"{cfskappa.scalingprofile}, scaling direction: "
            f"{cfskappa.scalingdirection}, min: {cfskappa.min:g}, max: "
            f"{cfskappa.max:g}), sigma (scaling: {cfssigma.scalingprofile}, "
            f"scaling direction: {cfssigma.scalingdirection}, min: "
            f"{cfssigma.min:g}, max: {cfssigma.max}) created."
        )

        if profile_id is None:
            terms = grid.pmls["cfs"]
        else:
            if not profile_id:
                raise ValueError(f"{self.params_str()} profile_id must not be empty.")
            profile = grid.pmls["profiles"].setdefault(profile_id, {"formulation": None, "cfs": []})
            terms = profile["cfs"]
        terms.append(cfs)

        if len(terms) > 2:
            logger.exception(
                f"{self.params_str()} can only be used up to two times, for up to a 2nd order PML."
            )
            raise ValueError


class PMLSlab(GridUserObject):
    """Place an experimental one-axis RIPML slab inside a grid.

    ``p1`` and ``p2`` bound a rectangular region. ``maximum_face`` selects the
    local face at which complex stretching is greatest; the profile reduces
    to zero towards the opposite face. For example, ``x0`` has its maximum at
    ``p1.x`` and opens towards increasing x. Unlike a domain-boundary PML, the
    slab is a correction applied to the existing geometry. By default, gprMax
    creates PEC plates on its four transverse faces and maximum-stretch face;
    faces coincident with the model boundary are omitted. The zero-stretch
    entrance remains open. Set ``build_pec=False`` to provide a custom or
    deliberately open enclosure. Exposed faces are then reported as warnings;
    incomplete enclosures have no stability guarantee. A slab added to an HSG
    subgrid must lie wholly within its working region. Domain-decomposed MPI
    CPU models may partition a slab normally or transversely; every rank stores
    only its local PML history while retaining the complete global CFS profile.

    Attributes:
        p1: lower physical ``(x, y, z)`` corner of the slab.
        p2: upper physical ``(x, y, z)`` corner of the slab.
        maximum_face: face at maximum stretching: ``x0``, ``xmax``, ``y0``,
            ``ymax``, ``z0``, or ``zmax``.
        profile_id: optional reusable PML profile. If omitted, the global PML
            formulation and CFS parameters are used.
        build_pec: create the five enclosing PEC plates. The default is
            ``True``.
        id: optional unique slab identifier. If omitted, gprMax generates one.
    """

    FACE_TO_DIRECTION = {
        "x0": "xminus",
        "xmax": "xplus",
        "y0": "yminus",
        "ymax": "yplus",
        "z0": "zminus",
        "zmax": "zplus",
    }

    @property
    def order(self):
        # Register before sources/receivers so their PML-position warning also
        # recognises user-positioned slabs.
        return 0

    @property
    def hash(self):
        return "#pml_slab"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            maximum_face = self.kwargs["maximum_face"].lower()
        except KeyError:
            logger.exception(f"{self.params_str()} requires p1, p2, and maximum_face.")
            raise

        profile_id = self.kwargs.get("profile_id")
        build_pec = self.kwargs.get("build_pec", True)
        ID = self.kwargs.get("id")
        if ID is None:
            number = len(grid.pmls["internal_specs"]) + 1
            ID = f"internal_pml_{number}"
            while any(spec.ID == ID for spec in grid.pmls["internal_specs"]):
                number += 1
                ID = f"internal_pml_{number}"

        if config.get_model_config().mode.startswith("2D"):
            raise ValueError(f"{self.params_str()} cannot currently be used in 2D mode.")
        if maximum_face not in self.FACE_TO_DIRECTION:
            raise ValueError(
                f"{self.params_str()} maximum_face must be one of "
                f"{', '.join(self.FACE_TO_DIRECTION)}."
            )
        if any(spec.ID == ID for spec in grid.pmls["internal_specs"]):
            raise ValueError(f"{self.params_str()} id '{ID}' is already in use.")
        if not isinstance(build_pec, (bool, np.bool_)):
            raise TypeError(f"{self.params_str()} build_pec must be a boolean.")

        uip = self._create_uip(grid)
        within_grid, lower, upper = uip.check_box_points(p1, p2, self.__str__())
        if getattr(grid, "is_distributed", False) is True:
            # Every rank retains the same immutable global declaration. Local
            # clipping is deferred until PML construction, where the global
            # CFS depth can be preserved across rank boundaries.
            lower = uip.discretise_static_point(p1)
            upper = uip.discretise_static_point(p2)
        elif not within_grid:
            return
        if not np.all(upper > lower):
            raise ValueError(f"{self.params_str()} must have non-zero extent on all three axes.")
        if isinstance(grid, SubGridBaseGrid):
            inner = np.array(
                [grid.n_boundary_cells_x, grid.n_boundary_cells_y, grid.n_boundary_cells_z]
            )
            outer = np.asarray(grid.size) - inner
            if np.any(lower < inner) or np.any(upper > outer):
                raise ValueError(
                    f"{self.params_str()} must lie wholly inside the subgrid working region; "
                    "it cannot overlap the HSG coupling or auxiliary PML regions."
                )

        axis = "xyz".index(maximum_face[0])
        if upper[axis] - lower[axis] < 2:
            raise ValueError(
                f"{self.params_str()} must be at least two cells thick along its absorption axis."
            )

        spec = InternalPMLSpec(
            ID=ID,
            maximum_face=maximum_face,
            direction=self.FACE_TO_DIRECTION[maximum_face],
            xs=int(lower[0]),
            xf=int(upper[0]),
            ys=int(lower[1]),
            yf=int(upper[1]),
            zs=int(lower[2]),
            zf=int(upper[2]),
            profile_id=profile_id,
            build_pec=bool(build_pec),
        )
        grid.pmls["internal_specs"].append(spec)
        grid.pmls["internal_registry"][ID] = {
            "spec": spec,
            "classification": "unvalidated",
            "profile_id": profile_id,
            "build_pec": bool(build_pec),
            "generated_pec_faces": 0,
        }
        if isinstance(grid, SubGridBaseGrid):
            continuous_lower = grid.local_to_global(lower)
            continuous_upper = grid.local_to_global(upper)
        else:
            continuous_lower = uip.discrete_to_continuous(lower)
            continuous_upper = uip.discrete_to_continuous(upper)
        logger.info(
            f"{self.grid_name(grid)}Internal PML slab '{ID}' from "
            f"{tuple(continuous_lower)}m to "
            f"{tuple(continuous_upper)}m, with maximum "
            f"stretching on its {maximum_face} face"
            + (
                f", using profile '{profile_id}'"
                if profile_id
                else ", using the global PML configuration"
            )
            + (
                ", with an automatic PEC enclosure"
                if build_pec
                else ", without an automatic PEC enclosure"
            )
            + ", registered."
        )


class SymmetryBoundary(GridUserObject):
    """Sets a PEC or PMC symmetry boundary condition on a model-domain face.

    The selected boundary replaces the PML on that face. A PEC boundary
    forces the tangential electric-field component IDs to the built-in PEC
    material during grid construction. A PMC boundary uses an image-theory
    ghost-node update for the on-wall tangential electric fields.

    PEC and PMC boundaries, including PMC boundaries in dispersive models, are
    supported by the CPU, CUDA, OpenCL, Metal, and domain-decomposed MPI CPU
    solvers. Symmetry boundaries are not supported in 2D mode or on a subgrid,
    although they may be used on the main grid of a model that contains
    subgrids.

    Attributes:
        face: One of ``x0``, ``y0``, ``z0``, ``xmax``, ``ymax``, or ``zmax``.
        type: Either ``pec`` or ``pmc``.
    """

    VALID_FACES = ("x0", "y0", "z0", "xmax", "ymax", "zmax")
    VALID_TYPES = ("pec", "pmc")

    @property
    def order(self):
        # Build before sources and receivers so their PML-position checks see
        # the symmetry face's disabled PML thickness.
        return 0

    @property
    def hash(self):
        return "#symmetry_boundary"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            face = self.kwargs["face"]
            boundary_type = self.kwargs["type"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires a face and a boundary type.")
            raise

        if face not in self.VALID_FACES:
            logger.exception(
                f"{self.params_str()} face must be one of {', '.join(self.VALID_FACES)}."
            )
            raise ValueError

        if boundary_type not in self.VALID_TYPES:
            logger.exception(
                f"{self.params_str()} type must be one of {', '.join(self.VALID_TYPES)}."
            )
            raise ValueError

        if config.get_model_config().mode.startswith("2D"):
            logger.exception(f"{self.params_str()} cannot currently be used in 2D mode.")
            raise ValueError

        if isinstance(grid, SubGridBaseGrid):
            logger.exception(
                f"{self.params_str()} cannot be used on a subgrid. It may still be "
                "used on the main grid of a model that contains subgrids."
            )
            raise ValueError

        if face in grid.symmetry_boundaries:
            logger.exception(
                f"{self.params_str()} a symmetry boundary has already been set on face '{face}'."
            )
            raise ValueError

        grid.symmetry_boundaries[face] = boundary_type
        overridden_thickness = grid.pmls["thickness"][face]
        grid.pmls["thickness"][face] = 0

        logger.info(
            f"Symmetry boundary ({boundary_type}) set on face '{face}'"
            + (
                f"; PML thickness on that face (was {overridden_thickness}) disabled."
                if overridden_thickness
                else "; PML on that face disabled."
            )
        )


"""
TODO: Can this be removed?
class Subgrid(UserObjectMulti):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.warning(
            "Subgrid user object is deprecated and may be removed in"
            " future releases of gprMax. Use the SubGridHSG user object"
            " instead."
        )
        self.children_multiple = []
        self.children_geometry = []

    def add(self, node):
        if isinstance(node, UserObjectMulti):
            self.children_multiple.append(node)
        elif isinstance(node, UserObjectGeometry):
            self.children_geometry.append(node)
        else:
            logger.exception("This object is unknown to gprMax.")
            raise ValueError
"""
