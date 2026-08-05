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

import inspect
import logging
import math
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import interpolate

import gprMax.config as config
from gprMax.eigenmode_config import (
    EigenmodeBandSpec,
    EigenmodeBandpassWaveform,
    EigenmodePortSpec,
)
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import DispersiveMaterial as DispersiveMaterialUser
from gprMax.materials import ListMaterial as ListMaterialUser
from gprMax.materials import Material as MaterialUser
from gprMax.materials import PeplinskiSoil as PeplinskiSoilUser
from gprMax.materials import RangeMaterial as RangeMaterialUser
from gprMax.pml import CFS, CFSParameter
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
from gprMax.user_objects.cmds_geometry.cmds_geometry import (
    rotate_2point_object,
    rotate_polarisation,
)
from gprMax.user_objects.rotatable import RotatableMixin
from gprMax.user_objects.user_objects import GridUserObject
from gprMax.waveforms import Waveform as WaveformUser

logger = logging.getLogger(__name__)


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

        logger.info(self.grid_name(grid) + f"Excitation file: {excitationfile}")

        # Get waveform names
        waveformIDs = np.loadtxt(excitationfile, max_rows=1, dtype=str)

        # Read all waveform values into an array
        waveformvalues = np.loadtxt(excitationfile, skiprows=1, dtype=config.sim_config.dtypes["float_or_double"])

        # Time array (if specified) for interpolation, otherwise use simulation time
        if waveformIDs[0].lower() == "time":
            waveformIDs = waveformIDs[1:]
            waveformtime = waveformvalues[:, 0]
            waveformvalues = waveformvalues[:, 1:]
            timestr = "user-defined time array"
        else:
            waveformtime = np.arange(0, grid.timewindow + grid.dt, grid.dt)
            timestr = "simulation time array"

        for i, waveformID in enumerate(waveformIDs):
            if any(x.ID == waveformID for x in grid.waveforms):
                raise ValueError(f"Waveform with ID {waveformID} already exists")
            w = WaveformUser()
            w.ID = waveformID
            w.type = "user"

            # Select correct column of waveform values depending on array shape
            singlewaveformvalues = waveformvalues[:] if len(waveformvalues.shape) == 1 else waveformvalues[:, i]

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
            elif self.kind is not None and self.fill_value is not None:
                w.userfunc = interpolate.interp1d(
                    waveformtime, singlewaveformvalues, kind=self.kind, fill_value=self.fill_value
                )
            else:
                raise ValueError(f"{self} requires either one or three parameter(s)")

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
                logger.exception(self.params_str() + (" builtin waveforms require exactly four parameters."))
                raise
            if freq <= 0:
                logger.exception(
                    self.params_str() + (" requires an excitation " "frequency value of greater than zero.")
                )
                raise ValueError
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
                    + (" a user-defined waveform requires an 'id' and either " "'user_func' or 'user_values'.")
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
                    float(userfunc(0.0))
                except Exception as err:
                    msg = (
                        self.params_str() + " 'user_func' must accept a single float (time in "
                        "seconds) and return a numeric value."
                    )
                    logger.exception(msg)
                    raise ValueError(msg) from err
                w.userfunc = userfunc

                logger.info(self.grid_name(grid) + (f"Waveform {w.ID} using a user-supplied function created."))

            elif "user_values" in self.kwargs:
                uservalues = self.kwargs["user_values"]
                fullargspec = inspect.getfullargspec(interpolate.interp1d)
                kwargs = dict(zip(reversed(fullargspec.args), reversed(fullargspec.defaults)))

                if "user_time" in self.kwargs:
                    waveformtime = self.kwargs["user_time"]
                else:
                    waveformtime = np.arange(0, grid.timewindow + grid.dt, grid.dt)

                # Set args for interpolation if given by user
                if "kind" in self.kwargs:
                    kwargs["kind"] = self.kwargs["kind"]
                if "fill_value" in self.kwargs:
                    kwargs["fill_value"] = self.kwargs["fill_value"]

                w.userfunc = interpolate.interp1d(waveformtime, uservalues, **kwargs)

                logger.info(self.grid_name(grid) + (f"Waveform {w.ID} that is user-defined created."))

            else:
                msg = self.params_str() + " a user-defined waveform requires either 'user_func' " "or 'user_values'."
                logger.exception(msg)
                raise ValueError(msg)

        grid.waveforms.append(w)


class VoltageSource(RotatableMixin, GridUserObject):
    """Specifies a voltage source at an electric field location.

    Attributes:
        polarisation: string required for polarisation of the source x, y, z.
        p1: tuple required for position of source x, y, z.
        resistance: float required for internal resistance (Ohms) of
                        voltage source.
        waveform_id: string required for identifier of waveform used with source.
        start: float optional to delay start time (secs) of source.
        stop: float optional to time (secs) to remove source.
        reference_impedance: float optional wave-reference impedance (Ohms)
            for a hard source. The default is 50 Ohms. For a finite-
            resistance source it must equal the source resistance.
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
        reference_impedance: Optional[float] = None,
    ):
        super().__init__(
            polarisation=polarisation,
            p1=p1,
            resistance=resistance,
            waveform_id=waveform_id,
            start=start,
            stop=stop,
            reference_impedance=reference_impedance,
        )

        self.point = p1
        self.polarisation = polarisation
        self.resistance = resistance
        self.waveform_id = waveform_id
        self.start = start
        self.stop = stop
        self.reference_impedance = reference_impedance

    def _do_rotate(self, grid: FDTDGrid):
        """Performs rotation."""
        rot_pol_pts, self.polarisation = rotate_polarisation(self.point, self.polarisation, self.axis, self.angle, grid)
        rot_pts = rotate_2point_object(rot_pol_pts, self.axis, self.angle, self.origin)
        self.point = tuple(rot_pts[0, :])

    def _validate_parameters(self, grid: FDTDGrid, discretised_point: Optional[npt.NDArray[np.int32]] = None):
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
                raise ValueError(f"{self.params_str()} polarisation must be {invariant_letter} in {mode} mode.")
            elif "TE" in mode and self.polarisation == invariant_letter:
                # E survives perpendicular to the invariant axis in TE
                # (e.g. Ex, Ey for TEz) - the own-axis component is
                # forced pec, same rule as HertzianDipole.
                raise ValueError(
                    f"{self.params_str()} polarisation must be {other_axes[0]} or " f"{other_axes[1]} in {mode} mode."
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
        if self.resistance < 0:
            raise ValueError(f"{self.params_str()} requires a source resistance of zero or greater.")
        if self.reference_impedance is not None:
            self.reference_impedance = float(self.reference_impedance)
            if not np.isfinite(self.reference_impedance) or self.reference_impedance <= 0:
                raise ValueError(f"{self.params_str()} reference impedance must be finite and positive.")
            if self.resistance > 0 and not np.isclose(self.reference_impedance, self.resistance):
                raise ValueError(f"{self.params_str()} reference impedance must equal the finite " "source resistance.")

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == self.waveform_id for x in grid.waveforms):
            raise ValueError(f"{self.params_str()} there is no waveform with the identifier {self.waveform_id}.")

        # Check start and stop
        if self.start is not None and self.stop is not None:
            if self.start < 0:
                raise ValueError(
                    f"{self.params_str()} delay of the initiation of the source should not be less" " than zero."
                )
            if self.stop < 0:
                raise ValueError(f"{self.params_str()} time to remove the source should not be less than zero.")
            if self.stop - self.start <= 0:
                raise ValueError(f"{self.params_str()} duration of the source should not be zero or less.")

    def _create_voltage_source(self, grid: FDTDGrid, coord: npt.NDArray[np.int32]) -> VoltageSourceUser:
        voltage_source = VoltageSourceUser()
        voltage_source.polarisation = self.polarisation
        voltage_source.coord = coord
        uip = self._create_uip(grid)
        x, y, z = uip.discretise_static_point(self.point)
        voltage_source.ID = f"{voltage_source.__class__.__name__}({x},{y},{z})"
        voltage_source.resistance = self.resistance
        voltage_source.reference_impedance = (
            float(self.reference_impedance)
            if self.reference_impedance is not None
            else (50.0 if self.resistance == 0 else float(self.resistance))
        )
        voltage_source.waveformID = self.waveform_id

        if self.start is None or self.stop is None:
            voltage_source.start = 0
            voltage_source.stop = grid.timewindow
        else:
            voltage_source.start = self.start
            voltage_source.stop = min(self.stop, grid.timewindow)

        voltage_source.calculate_waveform_values(grid)

        return voltage_source

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
        if self.do_rotate:
            self._do_rotate(grid)

        # Check the position of the voltage source
        uip = self._create_uip(grid)
        self.point = uip.resolve_inf_point(self.point)
        point_within_grid, discretised_point = uip.check_src_rx_point(self.point, self.params_str())

        if point_within_grid:
            self._validate_parameters(grid, discretised_point)
            voltage_source = self._create_voltage_source(grid, discretised_point)
            grid.add_source(voltage_source)
            position = uip.round_to_grid_static_point(self.point)
            self._log(grid, voltage_source, *position)


class HertzianDipole(RotatableMixin, GridUserObject):
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
        super().__init__(polarisation=polarisation, p1=p1, waveform_id=waveform_id, start=start, stop=stop)

        self.point = p1
        self.polarisation = polarisation.lower()
        self.waveform_id = waveform_id
        self.start = start
        self.stop = stop

    def _do_rotate(self, grid: FDTDGrid):
        """Performs rotation."""
        rot_pol_pts, self.polarisation = rotate_polarisation(self.point, self.polarisation, self.axis, self.angle, grid)
        rot_pts = rotate_2point_object(rot_pol_pts, self.axis, self.angle, self.origin)
        self.point = tuple(rot_pts[0, :])

    def _validate_parameters(self, grid: FDTDGrid, discretised_point: Optional[npt.NDArray[np.int32]] = None):
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
                raise ValueError(f"{self.params_str()} polarisation must be {invariant_letter} in {mode} mode.")
            elif "TE" in mode and self.polarisation == invariant_letter:
                # E survives perpendicular to the invariant axis in TE
                # (e.g. Ex, Ey for TEz) - the own-axis component is
                # forced pec.
                raise ValueError(
                    f"{self.params_str()} polarisation must be {other_axes[0]} or " f"{other_axes[1]} in {mode} mode."
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
            raise ValueError(f"{self.params_str()} there is no waveform with the identifier {self.waveform_id}.")

        # Check start and stop
        if self.start is not None and self.stop is not None:
            if self.start < 0:
                raise ValueError(
                    f"{self.params_str()} delay of the initiation of the source should not be less" " than zero."
                )
            if self.stop < 0:
                raise ValueError(f"{self.params_str()} time to remove the source should not be less than zero.")
            if self.stop - self.start <= 0:
                raise ValueError(f"{self.params_str()} duration of the source should not be zero or less.")

    def _create_hertzian_dipole(self, grid: FDTDGrid, coord: npt.NDArray[np.int32]) -> HertzianDipoleUser:
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
        h.waveformID = self.waveform_id

        if self.start is None or self.stop is None:
            h.start = 0
            h.stop = grid.timewindow
        else:
            h.start = self.start
            h.stop = min(self.stop, grid.timewindow)

        h.calculate_waveform_values(grid)

        return h

    def _log(self, grid: FDTDGrid, hertzian_dipole: HertzianDipoleUser, x: float, y: float, z: float):
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
        if self.do_rotate:
            self._do_rotate(grid)

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


class MagneticDipole(RotatableMixin, GridUserObject):
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
        super().__init__(polarisation=polarisation, p1=p1, waveform_id=waveform_id, start=start, stop=stop)

        self.point = p1
        self.polarisation = polarisation.lower()
        self.waveform_id = waveform_id
        self.start = start
        self.stop = stop

    def build(self, grid: FDTDGrid):
        if self.do_rotate:
            self._do_rotate(grid)

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

    def _do_rotate(self, grid: FDTDGrid):
        """Performs rotation."""
        rot_pol_pts, self.polarisation = rotate_polarisation(self.point, self.polarisation, self.axis, self.angle, grid)
        rot_pts = rotate_2point_object(rot_pol_pts, self.axis, self.angle, self.origin)
        self.point = tuple(rot_pts[0, :])

    def _validate_parameters(self, grid: FDTDGrid, discretised_point: Optional[npt.NDArray[np.int32]] = None):
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
                    f"{self.params_str()} polarisation must be {other_axes[0]} or " f"{other_axes[1]} in {mode} mode."
                )
            elif "TE" in mode and self.polarisation != invariant_letter:
                raise ValueError(f"{self.params_str()} polarisation must be {invariant_letter} in {mode} mode.")

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
            raise ValueError(f"{self.params_str()} there is no waveform with the identifier {self.waveform_id}.")

        # Check start and stop
        if self.start is not None and self.stop is not None:
            if self.start < 0:
                raise ValueError(
                    f"{self.params_str()} delay of the initiation of the source should not be less" " than zero."
                )
            if self.stop < 0:
                raise ValueError(f"{self.params_str()} time to remove the source should not be less than zero.")
            if self.stop - self.start <= 0:
                raise ValueError(f"{self.params_str()} duration of the source should not be zero or less.")

    def _create_magnetic_dipole(self, grid: FDTDGrid, coord: npt.NDArray[np.int32]) -> MagneticDipoleUser:
        m = MagneticDipoleUser()
        m.polarisation = self.polarisation
        m.coord = coord
        m.coordorigin = coord
        uip = self._create_uip(grid)
        x, y, z = uip.discretise_static_point(self.point)
        m.ID = f"{m.__class__.__name__}({x},{y},{z})"
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


class TransmissionLine(RotatableMixin, GridUserObject):
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

    def _do_rotate(self, grid: FDTDGrid):
        """Performs rotation."""
        rot_pol_pts, self.polarisation = rotate_polarisation(self.point, self.polarisation, self.axis, self.angle, grid)
        rot_pts = rotate_2point_object(rot_pol_pts, self.axis, self.angle, self.origin)
        self.point = tuple(rot_pts[0, :])

    def build(self, grid: FDTDGrid):
        if self.do_rotate:
            self._do_rotate(grid)

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
        # CUDA has a device-resident transmission-line update. OpenCL and
        # Metal use the same kernel template but do not yet have their host
        # buffer/launch lifecycle enabled and verified.
        if config.sim_config.general["solver"] in ["opencl", "metal"]:
            raise ValueError(
                f"{self.params_str()} cannot currently be used "
                "with the OpenCL or Metal-based solver. Consider "
                "using a #voltage_source instead."
            )

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
        if self.resistance <= 0 or self.resistance >= config.sim_config.em_consts["z0"]:
            raise ValueError(
                f"{self.params_str()} requires a resistance "
                "greater than zero and less than the impedance "
                "of free space, i.e. 376.73 Ohms."
            )

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == self.waveform_id for x in grid.waveforms):
            raise ValueError(f"{self.params_str()} there is no waveform with the identifier {self.waveform_id}.")

        # Check start and stop
        if self.start is not None and self.stop is not None:
            if self.start < 0:
                raise ValueError(
                    f"{self.params_str()} delay of the initiation of the source should not be less" " than zero."
                )
            if self.stop < 0:
                raise ValueError(f"{self.params_str()} time to remove the source should not be less than zero.")
            if self.stop - self.start <= 0:
                raise ValueError(f"{self.params_str()} duration of the source should not be zero or less.")

    def _create_transmission_line(self, grid: FDTDGrid, coord: npt.NDArray[np.int32]) -> TransmissionLineUser:
        t = TransmissionLineUser(grid.iterations, grid.dt)
        t.polarisation = self.polarisation
        t.coord = coord
        uip = self._create_uip(grid)
        x, y, z = uip.discretise_static_point(self.point)
        t.ID = f"{t.__class__.__name__}({x},{y},{z})"
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


class MagneticFrillSource(RotatableMixin, GridUserObject):
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
    ):
        super().__init__(
            polarisation=polarisation,
            p1=p1,
            zcoax=zcoax,
            waveform_id=waveform_id,
            start=start,
            stop=stop,
        )

        self.point = p1
        self.polarisation = polarisation
        self.zcoax = zcoax
        self.waveform_id = waveform_id
        self.start = start
        self.stop = stop

    def _do_rotate(self, grid: FDTDGrid):
        """Performs rotation."""
        rot_pol_pts, self.polarisation = rotate_polarisation(self.point, self.polarisation, self.axis, self.angle, grid)
        rot_pts = rotate_2point_object(rot_pol_pts, self.axis, self.angle, self.origin)
        self.point = tuple(rot_pts[0, :])

    def build(self, grid: FDTDGrid):
        if self.do_rotate:
            self._do_rotate(grid)

        uip = self._create_uip(grid)
        self.point = uip.resolve_inf_point(self.point)
        point_within_grid, discretised_point = uip.check_src_rx_point(self.point, self.params_str())

        if point_within_grid:
            self._validate_parameters(grid)
            frill_source = self._create_magnetic_frill_source(grid, discretised_point)
            grid.add_source(frill_source)
            position = uip.round_to_grid_static_point(self.point)
            self._log(grid, frill_source, *position)

    def _validate_parameters(self, grid: FDTDGrid):
        # MPI is rejected because a feed stencil cannot be split across rank
        # boundaries. A subgrid is safe: the frill, attached thin wire,
        # material rows, field stencil, and time histories all belong to the
        # same fine grid and are advanced by its CPU updater.
        if config.sim_config.mpi:
            raise ValueError(f"{self.params_str()} does not support MPI.")

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
            raise ValueError(f"{self.params_str()} there is no waveform with the identifier {self.waveform_id}.")

        # Check start and stop
        if self.start is not None and self.stop is not None:
            if self.start < 0:
                raise ValueError(
                    f"{self.params_str()} delay of the initiation of the source should not be less" " than zero."
                )
            if self.stop < 0:
                raise ValueError(f"{self.params_str()} time to remove the source should not be less than zero.")
            if self.stop - self.start <= 0:
                raise ValueError(f"{self.params_str()} duration of the source should not be zero or less.")

    def _create_magnetic_frill_source(self, grid: FDTDGrid, coord: npt.NDArray[np.int32]) -> MagneticFrillSourceUser:
        f = MagneticFrillSourceUser(grid.iterations, grid.dt)
        f.polarisation = self.polarisation
        f.coord = coord
        uip = self._create_uip(grid)
        x, y, z = uip.discretise_static_point(self.point)
        f.ID = f"{f.__class__.__name__}({x},{y},{z})"
        f.Z0 = self.zcoax
        f.waveformID = self.waveform_id

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


def _reject_discrete_plane_wave_mpi(params_str):
    """Reject TFSF corrections that are not MPI-decomposition aware."""

    if config.sim_config.mpi:
        raise ValueError(f"{params_str} cannot currently be used with MPI.")


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
            f"{params_str} must be defined on the main grid; its TFSF box " "may strictly enclose complete subgrids."
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

    if is_2d:
        forced = (0, 1) if "TM" in mode else (1, 1)
        overridden_explicit = (p1_explicit and start[inv] != forced[0]) or (p2_explicit and stop[inv] != forced[1])
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

        _reject_discrete_plane_wave_mpi(self.params_str())
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

        # Warn about using a discrete plane wave on GPU
        if config.sim_config.general["solver"] in ["opencl", "metal"]:
            logger.exception(
                f"{self.params_str()} cannot currently be used " + "with the OpenCL or Apple Metal-based solver. "
            )
            raise ValueError

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            logger.exception(f"{self.params_str()} there is no waveform " + f"with the identifier {waveform_id}.")
            raise ValueError

        # Check if there is a materialID in the materials list
        if not any(x.ID == material_id for x in grid.materials):
            logger.exception(f"{self.params_str()} there is no material " + f"with the identifier {material_id}.")
            raise ValueError

        # Check angles
        if theta < 0 or theta > 180:
            logger.exception(f"{self.params_str()} Polar angle theta must be between 0 and 180 degrees.")
            raise ValueError

        if phi < 0 or phi > 360:
            logger.exception(f"{self.params_str()} Azimuthal angle phi must be between 0 and 360 degrees.")
            raise ValueError

        if psi < 0 or psi > 360:
            logger.exception(f"{self.params_str()} Polarisation angle psi must be between 0 and 360 degrees.")
            raise ValueError

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

        try:
            # Check source start & source remove time parameters
            start = self.kwargs["start"]
            stop = self.kwargs["stop"]
            if start < 0:
                logger.exception(
                    self.params_str() + (" delay of the initiation " "of the source should not " "be less than zero.")
                )
                raise ValueError
            if stop < 0:
                logger.exception(self.params_str() + (" time to remove the source should not be less than zero."))
                raise ValueError
            if stop - start <= 0:
                logger.exception(self.params_str() + (" duration of the source should not be zero or less."))
                raise ValueError
            DPW.start = start
            DPW.stop = min(stop, grid.timewindow)
            startstop = f" start time {start:g} secs, finish time {stop:g} secs "
        except KeyError:
            DPW.start = 0
            DPW.stop = grid.timewindow
            startstop = " "

        DPW.initializeDiscretePlaneWave(grid)

        precompute = True
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

        _reject_discrete_plane_wave_mpi(self.params_str())

        try:
            material_id = self.kwargs["material_id"]
        except KeyError:
            # set defaule to free space
            material_id = "free_space"

        try:
            precompute = self.kwargs["precompute"]
        except KeyError:
            precompute = True

        # Warn about using a discrete plane wave on GPU
        if config.sim_config.general["solver"] in ["opencl", "metal"]:
            logger.exception(
                f"{self.params_str()} cannot currently be used " + "with the OpenCL or Apple Metal-based solver. "
            )
            raise ValueError

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            logger.exception(f"{self.params_str()} there is no waveform " + f"with the identifier {waveform_id}.")
            raise ValueError

        # Check if there is a materialID in the materials list
        if not any(x.ID == material_id for x in grid.materials):
            logger.exception(f"{self.params_str()} there is no material " + f"with the identifier {material_id}.")
            raise ValueError

        # Check angle

        if psi < 0 or psi > 360:
            logger.exception(f"{self.params_str()} Polarisation angle psi must be between 0 and 360 degrees.")
            raise ValueError

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

        try:
            # Check source start & source remove time parameters
            start = self.kwargs["start"]
            stop = self.kwargs["stop"]
            if start < 0:
                logger.exception(
                    self.params_str() + (" delay of the initiation " "of the source should not " "be less than zero.")
                )
                raise ValueError
            if stop < 0:
                logger.exception(self.params_str() + (" time to remove the source should not be less than zero."))
                raise ValueError
            if stop - start <= 0:
                logger.exception(self.params_str() + (" duration of the source should not be zero or less."))
                raise ValueError
            DPW.start = start
            DPW.stop = min(stop, grid.timewindow)
            startstop = f" start time {start:g} secs, finish time {stop:g} secs "
        except KeyError:
            DPW.start = 0
            DPW.stop = grid.timewindow
            startstop = " "

        DPW.initializeDiscretePlaneWave(grid)

        precompute = True
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

        _reject_discrete_plane_wave_mpi(self.params_str())

        try:
            precompute = self.kwargs["precompute"]
        except KeyError:
            precompute = True

        # Warn about using a discrete plane wave on GPU
        if config.sim_config.general["solver"] in ["opencl", "metal"]:
            logger.exception(
                f"{self.params_str()} cannot currently be used " + "with the OpenCL or Apple Metal-based solver. "
            )
            raise ValueError

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            logger.exception(f"{self.params_str()} there is no waveform " + f"with the identifier {waveform_id}.")
            raise ValueError

        # Check polarisation angle
        if psi < 0 or psi > 360:
            logger.exception(f"{self.params_str()} Polarisation angle psi must be between 0 and 360 degrees.")
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

        try:
            # Check source start & source remove time parameters
            start = self.kwargs["start"]
            stop = self.kwargs["stop"]
            if start < 0:
                logger.exception(
                    self.params_str() + (" delay of the initiation " "of the source should not " "be less than zero.")
                )
                raise ValueError
            if stop < 0:
                logger.exception(self.params_str() + (" time to remove the source should not be less than zero."))
                raise ValueError
            if stop - start <= 0:
                logger.exception(self.params_str() + (" duration of the source should not be zero or less."))
                raise ValueError
            DPW.start = start
            DPW.stop = min(stop, grid.timewindow)
            startstop = f" start time {start:g} secs, finish time {stop:g} secs "
        except KeyError:
            DPW.start = 0
            DPW.stop = grid.timewindow
            startstop = " "

        DPW.initializeDiscretePlaneWave(grid)

        precompute = True
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
    '''Define the single frequency band shared by all eigenmode ports.'''

    @property
    def order(self):
        return 20

    @property
    def hash(self):
        return '#eigenmode_band'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        if isinstance(grid, SubGridBaseGrid):
            raise ValueError(f'{self.params_str()} currently supports only the main grid.')
        if grid.eigenmodeband is not None:
            raise ValueError('Exactly one EigenmodeBand may be defined per grid.')
        try:
            band_id = str(self.kwargs['id'])
            fmin = float(self.kwargs['fmin'])
            fmax = float(self.kwargs['fmax'])
            points = int(self.kwargs['points'])
        except KeyError:
            logger.exception(f'{self.params_str()} requires id, fmin, fmax, and points.')
            raise
        if not band_id or any(character.isspace() for character in band_id):
            raise ValueError(f'{self.params_str()} id must be a non-empty token without whitespace.')
        _validate_eigenmode_dft(self.params_str(), fmin, fmax, points)
        threshold = float(self.kwargs.get('spectral_threshold', 1e-3))
        if not 0 < threshold < 1:
            raise ValueError(f'{self.params_str()} spectral_threshold must be between zero and one.')
        transition = self.kwargs.get('transition', 'auto')
        if transition != 'auto':
            transition = float(transition)
            if not np.isfinite(transition) or transition <= 0:
                raise ValueError(f'{self.params_str()} transition must be positive or auto.')
        grid.eigenmodeband = EigenmodeBandSpec(
            id=band_id,
            fmin=fmin,
            fmax=fmax,
            points=points,
            transition=transition,
            spectral_threshold=threshold,
        )
        logger.info(
            f'{self.grid_name(grid)}Eigenmode band {band_id!r}, frequencies '
            f'{fmin:g} to {fmax:g} Hz with {points} common DFT point(s), created.'
        )


class EigenmodePort(GridUserObject):
    '''Define a modal port plane with its own anchor-frequency policy.'''

    @property
    def order(self):
        return 21

    @property
    def hash(self):
        return '#eigenmode_port'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        if isinstance(grid, SubGridBaseGrid):
            raise ValueError(f'{self.params_str()} currently supports only the main grid.')
        if grid.eigenmodeband is None:
            raise ValueError(f'{self.params_str()} requires one preceding EigenmodeBand.')
        if config.sim_config.general['solver'] in ('cuda', 'opencl', 'metal'):
            raise ValueError(
                f'{self.params_str()} cannot currently be used with CUDA, OpenCL, or Metal.'
            )
        if config.sim_config.mpi:
            raise ValueError(f'{self.params_str()} cannot currently be used with MPI.')
        try:
            port = int(self.kwargs['port'])
            p1 = tuple(float(value) for value in self.kwargs['p1'])
            p2 = tuple(float(value) for value in self.kwargs['p2'])
            direction = str(self.kwargs['direction'])
            modes_arg = self.kwargs['modes']
        except KeyError:
            logger.exception(f'{self.params_str()} requires port, p1, p2, direction, and modes.')
            raise
        if port < 1:
            raise ValueError(f'{self.params_str()} port must be one or greater.')
        if len(p1) != 3 or len(p2) != 3:
            raise ValueError(f'{self.params_str()} p1 and p2 must each contain three coordinates.')
        if direction not in ('+', '-'):
            raise ValueError(f'{self.params_str()} direction must be + or -.')
        if np.isscalar(modes_arg):
            mode_count = int(modes_arg)
            modes = tuple(range(1, mode_count + 1))
        else:
            modes = tuple(int(value) for value in modes_arg)
        if not modes or any(mode < 1 for mode in modes):
            raise ValueError(f'{self.params_str()} modes must contain positive one-based indices.')
        if modes != tuple(sorted(set(modes))):
            raise ValueError(f'{self.params_str()} modes must be unique and strictly increasing.')
        anchors_arg = self.kwargs.get('anchors', 'auto')
        if isinstance(anchors_arg, str):
            if anchors_arg.lower() != 'auto':
                raise ValueError(f'{self.params_str()} anchors must be auto or frequencies.')
            anchors = 'auto'
        elif np.isscalar(anchors_arg):
            anchors = (float(anchors_arg),)
        else:
            anchors = tuple(float(value) for value in anchors_arg)
        if anchors != 'auto':
            if not anchors:
                raise ValueError(f'{self.params_str()} requires at least one explicit anchor.')
            if any(not np.isfinite(value) or value <= 0 for value in anchors):
                raise ValueError(f'{self.params_str()} anchors must be finite and positive.')
            if any(anchors[index] >= anchors[index + 1] for index in range(len(anchors) - 1)):
                raise ValueError(f'{self.params_str()} anchors must be unique and strictly increasing.')
        if port in grid.eigenmodeportdefs:
            raise ValueError(f'Eigenmode port {port} is already defined.')

        domain_mode = config.get_model_config().mode
        invariant_axis = 'xyz'.index(domain_mode[-1]) if domain_mode.startswith('2D') else None
        equal_axes = [
            axis
            for axis in range(3)
            if axis != invariant_axis and p1[axis] == p2[axis] and np.isfinite(p1[axis])
        ]
        if len(equal_axes) != 1:
            raise ValueError(
                f'{self.params_str()} must have exactly one finite matching coordinate '
                'pair, which defines the port normal.'
            )
        normal_axis = equal_axes[0]
        transverse_axes = tuple(axis for axis in range(3) if axis != normal_axis)
        plot_fields = self.kwargs.get('plot_fields')
        if plot_fields is not None and not isinstance(plot_fields, (bool, np.bool_)):
            raise ValueError(f'{self.params_str()} plot_fields must be True, False, or None.')
        grid.eigenmodeportdefs[port] = EigenmodePortSpec(
            port=port,
            p1=p1,
            p2=p2,
            normal='xyz'[normal_axis],
            direction=direction,
            normal_axis=normal_axis,
            transverse_axes=transverse_axes,
            invariant_axis=invariant_axis,
            modes=modes,
            anchors=anchors,
            plot_fields=None if plot_fields is None else bool(plot_fields),
        )
        axis_name = 'xyz'[normal_axis]
        logger.info(
            f'{self.grid_name(grid)}Eigenmode port {port}, normal {axis_name}{direction}, '
            f'monitoring modes {modes}, with anchors {anchors}, created.'
        )


class EigenmodeExcitation(GridUserObject):
    '''Attach the single active modal excitation to a defined port.'''

    @property
    def order(self):
        return 22

    @property
    def hash(self):
        return '#eigenmode_excitation'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _runtime_plane_kwargs(port, band):
        transverse_p1 = tuple(port.p1[axis] for axis in port.transverse_axes)
        transverse_p2 = tuple(port.p2[axis] for axis in port.transverse_axes)
        lower = tuple(min(first, second) for first, second in zip(transverse_p1, transverse_p2))
        upper = tuple(max(first, second) for first, second in zip(transverse_p1, transverse_p2))
        kwargs = {
            'normal': port.normal,
            'direction': port.direction,
            'p1': lower,
            'p2': upper,
            'w': port.p1[port.normal_axis],
            'port_index': port.port,
            'dft_start': band.fmin,
            'dft_stop': band.fmax,
            'dft_points': band.points,
            'plot_fields': port.plot_fields,
        }
        if len(port.resolved_anchors) == 1:
            kwargs['frequency'] = port.resolved_anchors[0]
        else:
            kwargs['frequencies'] = port.resolved_anchors
        return kwargs

    def build(self, grid: FDTDGrid):
        if isinstance(grid, SubGridBaseGrid):
            raise ValueError(f'{self.params_str()} currently supports only the main grid.')
        if grid.eigenmodeexcitation is not None:
            raise ValueError('Only one EigenmodeExcitation may be defined per grid.')
        if grid.eigenmodeband is None:
            raise ValueError(f'{self.params_str()} requires one EigenmodeBand.')
        try:
            source_port_number = int(self.kwargs['port'])
            excitation_mode = int(self.kwargs['mode'])
        except KeyError:
            logger.exception(f'{self.params_str()} requires port and mode.')
            raise
        if not grid.eigenmodeportdefs:
            raise ValueError(f'{self.params_str()} requires at least one EigenmodePort.')
        if source_port_number not in grid.eigenmodeportdefs:
            raise ValueError(f'{self.params_str()} references unknown eigenmode port {source_port_number}.')
        source_port = grid.eigenmodeportdefs[source_port_number]
        if excitation_mode not in source_port.modes:
            raise ValueError(
                f'{self.params_str()} mode {excitation_mode} is not monitored by port '
                f'{source_port_number}; available modes are {source_port.modes}.'
            )
        band = grid.eigenmodeband
        waveform_arg = self.kwargs.get('waveform', 'auto')
        amplitude = float(self.kwargs.get('amplitude', 1.0))
        plot_waveform = self.kwargs.get('plot_waveform')
        if plot_waveform is not None and not isinstance(plot_waveform, (bool, np.bool_)):
            raise ValueError(
                f'{self.params_str()} plot_waveform must be True, False, or None.'
            )
        if plot_waveform is not None:
            plot_waveform = bool(plot_waveform)
        generated_waveform = isinstance(waveform_arg, str) and waveform_arg.lower() == 'auto'
        if generated_waveform:
            waveform = EigenmodeBandpassWaveform(
                band_id=band.id,
                fmin=band.fmin,
                fmax=band.fmax,
                amplitude=amplitude,
                dt=grid.dt,
                sample_count=int(grid.iterations),
                spectral_threshold=band.spectral_threshold,
                transition=band.transition,
            )
            if any(existing.ID == waveform.ID for existing in grid.waveforms):
                raise ValueError(f'Generated eigenmode waveform ID {waveform.ID!r} is already in use.')
            grid.waveforms.append(waveform)
        else:
            if amplitude != 1.0:
                raise ValueError('EigenmodeExcitation amplitude is only used with waveform=\'auto\'.')
            waveform_id = str(waveform_arg)
            matches = [waveform for waveform in grid.waveforms if waveform.ID == waveform_id]
            if not matches:
                raise ValueError(f'{self.params_str()} references unknown waveform {waveform_id!r}.')
            waveform = matches[0]
        band.resolve_spectrum(grid, waveform, generated_waveform=generated_waveform)

        for port_number in sorted(grid.eigenmodeportdefs):
            port = grid.eigenmodeportdefs[port_number]
            is_source = port_number == source_port_number
            port.resolve_anchors(band, is_source=is_source)
            common = self._runtime_plane_kwargs(port, band)
            if is_source:
                common.update(
                    {
                        'mode_index': excitation_mode,
                        'mode_count': max(port.modes),
                        'waveform_id': waveform.ID,
                        'spectral_threshold': band.spectral_threshold,
                    }
                )
                _EigenmodeSourceBuilder(**common).build(grid)
                runtime = grid.eigenmodesources[-1]
                runtime.mode_indices = port.modes
                runtime.plot_waveform = plot_waveform
            else:
                common.update({'mode_count': max(port.modes), 'id': f'port{port.port}'})
                _EigenmodeReceiverBuilder(**common).build(grid)
                runtime = grid.eigenmodereceivers[-1]
                runtime.mode_indices = port.modes
            runtime.anchor_policy = port.anchor_policy
            runtime.requested_anchor_policy = port.anchor_policy
            runtime.resolved_anchor_policy = port.anchor_policy
            runtime.fallback_frequency = 0.5 * (band.fmin + band.fmax)
        grid.eigenmodeexcitation = self
        logger.info(
            f'{self.grid_name(grid)}Eigenmode excitation created on port '
            f'{source_port_number}, mode {excitation_mode}, using waveform {waveform.ID!r}.'
        )


def _validate_eigenmode_dft(label, start, stop, points):
    if not np.isfinite(start) or not np.isfinite(stop) or start <= 0 or stop < start:
        raise ValueError(f"{label} DFT frequencies must satisfy 0 < start <= stop.")
    if points < 1:
        raise ValueError(f"{label} DFT points must be at least one.")
    if points == 1 and stop != start:
        raise ValueError(f"{label} a one-point DFT requires equal start and stop.")
    if points > 1 and stop == start:
        raise ValueError(f"{label} a multi-point DFT requires stop greater than start.")


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
        if isinstance(grid, SubGridBaseGrid):
            raise ValueError(f"{self.params_str()} currently supports only the main grid.")

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
            raise ValueError(f"{self.params_str()} accepts either frequency or frequencies, not both.")
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
        plot_fields = self.kwargs.get("plot_fields")
        if plot_fields is not None and not isinstance(plot_fields, (bool, np.bool_)):
            raise ValueError(f"{self.params_str()} plot_fields must be True, False, or None.")
        if plot_fields is not None:
            plot_fields = bool(plot_fields)

        if config.sim_config.general["solver"] in ["cuda", "opencl", "metal"]:
            logger.exception(
                f"{self.params_str()} cannot currently be used with the CUDA, OpenCL, or Apple Metal solver."
            )
            raise ValueError

        # MPI decomposes the grid across ranks, but the source plane's
        # bounds/plane-index validation below and the FDFD cross-section
        # extraction in EigenmodeSource.grid_init() both assume a single,
        # whole grid with globally-valid coordinates. Neither is currently
        # rank-aware, so reject MPI outright until that support is added.
        if config.sim_config.mpi:
            raise ValueError(f"{self.params_str()} cannot currently be used with MPI.")

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
                f"{self.params_str()} mode_count must be at least the excited mode_index " f"({mode_index})."
            )
        if port_index < 1:
            raise ValueError(f"{self.params_str()} port_index must be one or greater.")

        _validate_eigenmode_dft(self.params_str(), dft_start, dft_stop, dft_points)

        if not frequencies:
            raise ValueError(f"{self.params_str()} requires at least one frequency.")
        if any(not np.isfinite(value) or value <= 0 for value in frequencies):
            raise ValueError(f"{self.params_str()} frequencies must be finite and greater than zero.")
        if any(frequencies[index] >= frequencies[index + 1] for index in range(len(frequencies) - 1)):
            raise ValueError(f"{self.params_str()} frequencies must be unique and strictly increasing.")
        if not 0 < spectral_threshold < 1:
            raise ValueError(f"{self.params_str()} spectral_threshold must be between zero and one.")

        if not any(x.ID == waveform_id for x in grid.waveforms):
            logger.exception(f"{self.params_str()} there is no waveform with the identifier {waveform_id}.")
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
        uip = self._create_uip(grid)
        full_lower = np.asarray(
            uip.resolve_inf_point(tuple(full_lower), role="lower"),
            dtype=np.float64,
        )
        full_upper = np.asarray(
            uip.resolve_inf_point(tuple(full_upper), role="upper"),
            dtype=np.float64,
        )
        p1 = tuple(full_lower[transverse_axes])
        p2 = tuple(full_upper[transverse_axes])
        w = float(full_lower[normal_axis])

        lower = np.zeros(3, dtype=np.int32)
        upper = np.zeros(3, dtype=np.int32)
        plane_index = int(round(w / grid.dl[normal_axis]))
        lower[transverse_axes[0]] = int(round(p1[0] / grid.dl[transverse_axes[0]]))
        lower[transverse_axes[1]] = int(round(p1[1] / grid.dl[transverse_axes[1]]))
        upper[transverse_axes[0]] = int(round(p2[0] / grid.dl[transverse_axes[0]]))
        upper[transverse_axes[1]] = int(round(p2[1] / grid.dl[transverse_axes[1]]))

        if plane_index < 0 or plane_index > grid.size[normal_axis]:
            logger.exception(f"{self.params_str()} normal source plane coordinate is outside the grid.")
            raise ValueError

        if direction == "+" and plane_index < 1:
            raise ValueError(
                "A positive-direction eigenmode source must be at least " "one cell inside the lower domain boundary."
            )

        if np.any(lower[transverse_axes] < 0) or np.any(upper[transverse_axes] > grid.size[transverse_axes]):
            logger.exception(f"{self.params_str()} transverse source bounds are outside the grid.")
            raise ValueError

        if np.any(lower[transverse_axes] >= upper[transverse_axes]):
            logger.exception(
                f"{self.params_str()} lower transverse coordinates must be less than upper transverse coordinates."
            )
            raise ValueError

        if invariant_axis is not None and (
            lower[invariant_axis] != 0 or upper[invariant_axis] != grid.size[invariant_axis]
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
            next(axis for axis in transverse_axes if axis != invariant_axis) if invariant_axis is not None else None
        )
        if mode.startswith("2D TM"):
            source.domain_polarization = "TM"
        elif mode.startswith("2D TE"):
            source.domain_polarization = "TE"
        source.transverse_start = lower[transverse_axes].copy()
        source.transverse_stop = upper[transverse_axes].copy()
        source.plane_index = plane_index
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
        if isinstance(grid, SubGridBaseGrid):
            raise ValueError(f"{self.params_str()} currently supports only the main grid.")

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
            raise ValueError(f"{self.params_str()} accepts either frequency or frequencies, not both.")
        values = frequencies_arg if frequencies_arg is not None else frequency
        if values is None:
            raise ValueError(f"{self.params_str()} requires frequency or frequencies.")
        frequencies = (float(values),) if np.isscalar(values) else tuple(float(value) for value in values)
        plot_fields = self.kwargs.get("plot_fields")
        if plot_fields is not None and not isinstance(plot_fields, (bool, np.bool_)):
            raise ValueError(f"{self.params_str()} plot_fields must be True, False, or None.")

        if config.sim_config.general["solver"] in ["cuda", "opencl", "metal"]:
            raise ValueError(
                f"{self.params_str()} cannot currently be used with the CUDA, OpenCL, or Apple Metal solver."
            )
        if config.sim_config.mpi:
            raise ValueError(f"{self.params_str()} cannot currently be used with MPI.")
        if normal not in ("x", "y", "z") or direction not in ("+", "-"):
            raise ValueError(f"{self.params_str()} requires normal x/y/z and direction +/-.")
        if mode_count < 1:
            raise ValueError(f"{self.params_str()} mode_count must be one or greater.")
        if port_index < 1:
            raise ValueError(f"{self.params_str()} port_index must be one or greater.")
        mode_indices = tuple(range(1, mode_count + 1))
        if not frequencies or any(not np.isfinite(value) or value <= 0 for value in frequencies):
            raise ValueError(f"{self.params_str()} frequencies must be finite and positive.")
        if any(frequencies[index] >= frequencies[index + 1] for index in range(len(frequencies) - 1)):
            raise ValueError(f"{self.params_str()} frequencies must be unique and strictly increasing.")
        _validate_eigenmode_dft(self.params_str(), dft_start, dft_stop, dft_points)

        axis_map = {"x": 0, "y": 1, "z": 2}
        normal_axis = axis_map[normal]
        transverse_axes = [axis for axis in range(3) if axis != normal_axis]
        domain_mode = config.get_model_config().mode
        invariant_axis = "xyz".index(domain_mode[-1]) if domain_mode.startswith("2D") else None
        if invariant_axis is not None and normal_axis == invariant_axis:
            raise ValueError(f"{self.params_str()} in {domain_mode} mode must have an in-plane normal.")

        full_lower = np.zeros(3, dtype=np.float64)
        full_upper = np.zeros(3, dtype=np.float64)
        full_lower[normal_axis] = w
        full_upper[normal_axis] = w
        full_lower[transverse_axes] = p1
        full_upper[transverse_axes] = p2
        uip = self._create_uip(grid)
        full_lower = np.asarray(uip.resolve_inf_point(tuple(full_lower), role="lower"), dtype=np.float64)
        full_upper = np.asarray(uip.resolve_inf_point(tuple(full_upper), role="upper"), dtype=np.float64)
        p1 = tuple(full_lower[transverse_axes])
        p2 = tuple(full_upper[transverse_axes])
        w = float(full_lower[normal_axis])
        lower = np.zeros(3, dtype=np.int32)
        upper = np.zeros(3, dtype=np.int32)
        plane_index = int(round(w / grid.dl[normal_axis]))
        for position, axis in enumerate(transverse_axes):
            lower[axis] = int(round(p1[position] / grid.dl[axis]))
            upper[axis] = int(round(p2[position] / grid.dl[axis]))

        if plane_index < 0 or plane_index > grid.size[normal_axis]:
            raise ValueError(f"{self.params_str()} receiver plane is outside the grid.")
        if direction == "+" and plane_index < 1:
            raise ValueError(f"{self.params_str()} positive-direction receiver needs a lower H plane.")
        if np.any(lower[transverse_axes] < 0) or np.any(upper[transverse_axes] > grid.size[transverse_axes]):
            raise ValueError(f"{self.params_str()} transverse bounds are outside the grid.")
        if np.any(lower[transverse_axes] >= upper[transverse_axes]):
            raise ValueError(f"{self.params_str()} lower bounds must be less than upper bounds.")
        if invariant_axis is not None and (
            lower[invariant_axis] != 0 or upper[invariant_axis] != grid.size[invariant_axis]
        ):
            raise ValueError(f"{self.params_str()} in {domain_mode} mode must span the invariant axis.")

        axis_name = "xyz"[normal_axis]
        face = f"{axis_name}0" if direction == "+" else f"{axis_name}max"
        pml_thickness = grid.pmls["thickness"][face]
        adjacent_plane = pml_thickness if direction == "+" else grid.size[normal_axis] - pml_thickness
        if pml_thickness == 0:
            logger.warning(
                f"Eigenmode receiver {port_id!r} is not next to a PML because the "
                f"{face} face has zero PML thickness. Reflections beyond the port "
                "can contaminate its S-parameters."
            )
        elif plane_index != adjacent_plane:
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
            next(axis for axis in transverse_axes if axis != invariant_axis) if invariant_axis is not None else None
        )
        if domain_mode.startswith("2D TM"):
            receiver.domain_polarization = "TM"
        elif domain_mode.startswith("2D TE"):
            receiver.domain_polarization = "TE"
        receiver.transverse_start = lower[transverse_axes].copy()
        receiver.transverse_stop = upper[transverse_axes].copy()
        receiver.plane_index = plane_index
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
        grid.eigenmodereceivers.append(receiver)
        logger.info(
            f"{self.grid_name(grid)}Eigenmode receiver {port_id!r}, normal {normal}{direction}, "
            f"monitoring modes 1-{mode_count} at port {port_index}, "
            f"transverse bounds {p1} m to {p2} m, normal coordinate {w:g} m created."
        )


class Rx(RotatableMixin, GridUserObject):
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

    def _do_rotate(self, grid: FDTDGrid):
        """Performs rotation."""
        new_pt = self.point + grid.dl
        pts = np.array([self.point, new_pt])
        rot_pts = rotate_2point_object(pts, self.axis, self.angle, self.origin)
        self.point = tuple(rot_pts[0, :])

        # TODO: Why does this need resetting if rotate the receiver?
        # If specific field components are specified, set to output all components
        if self.outputs is not None:
            self.outputs = None
            self.kwargs.pop("outputs", None)

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
                r.outputs[field] = np.zeros(grid.iterations, dtype=config.sim_config.dtypes["float_or_double"])
            else:
                raise ValueError(
                    f"{self.params_str()} contains an output "
                    f"type that is not allowable. Allowable "
                    f"outputs in current context are "
                    f"{allowableoutputs}."
                )

        return r

    def build(self, grid: FDTDGrid):
        if self.do_rotate:
            self._do_rotate(grid)

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
            self.lower_point = tuple(lower_single[a] if a == invariant_axis else lower_ranged[a] for a in range(3))
            self.upper_point = tuple(upper_single[a] if a == invariant_axis else upper_ranged[a] for a in range(3))
        else:
            self.lower_point = lower_ranged
            self.upper_point = upper_ranged

        _, discretised_lower_point = uip.check_src_rx_point(self.lower_point, self.params_str(), "lower")
        _, discretised_upper_point = uip.check_src_rx_point(self.upper_point, self.params_str(), "upper")
        discretised_dl = uip.discretise_static_point(self.dl)

        if any(discretised_lower_point > discretised_upper_point):
            raise ValueError(f"{self.params_str()} the lower coordinates should be less than the upper coordinates.")
        if any(discretised_dl < 0):
            raise ValueError(f"{self.params_str()} the step size should not be less than zero.")

        discretised_dl = np.where(discretised_dl == 0, 1, discretised_dl)

        if any(discretised_dl < 1):
            raise ValueError(f"{self.params_str()} the step size should not be less than the spatial discretisation.")

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

        if er < 1:
            logger.exception(
                f"{self.params_str()} requires a positive value of one or greater for static (DC) permittivity."
            )
            raise ValueError
        if se != "inf":
            se = float(se)
            if se < 0:
                logger.exception(f"{self.params_str()} requires a positive value for electric conductivity.")
                raise ValueError
        else:
            se = float("inf")
        if mr < 1:
            logger.exception(
                f"{self.params_str()} requires a positive value of one or greater for magnetic permeability."
            )
            raise ValueError
        if sm < 0:
            logger.exception(f"{self.params_str()} requires a positive value for magnetic loss.")
            raise ValueError

        if any(x.ID == material_id for x in grid.materials):
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


class AddDebyeDispersion(GridUserObject):
    """Adds dispersive properties to already defined Material based on a
        multi-pole Debye formulation.

    Attributes:
        poles: float required for number of Debye poles.
        er_delta: tuple required for difference between zero-frequency relative
                    permittivity and relative permittivity at infinite frequency
                    for each pole.
        tau: tuple required for relaxation time (secs) for each pole.
        material_ids: list required of material ids to apply disperive
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

        if poles < 0:
            logger.exception(f"{self.params_str()} requires a positive value for number of poles.")
            raise ValueError

        # Look up requested materials in existing list of material instances
        materials = [y for x in material_ids for y in grid.materials if y.ID == x]

        if len(materials) != len(material_ids):
            notfound = [x for x in material_ids if x not in materials]
            logger.exception(f"{self.params_str()} material(s) {notfound} do not exist")
            raise ValueError

        for material in materials:
            disp_material = DispersiveMaterialUser(material.numID, material.ID)
            disp_material.er = material.er
            disp_material.se = material.se
            disp_material.mr = material.mr
            disp_material.sm = material.sm
            disp_material.type = "debye"
            disp_material.poles = poles
            disp_material.averagable = False
            for i in range(poles):
                if tau[i] > 0:
                    logger.debug("Not checking if relaxation times are " "greater than time-step.")
                    disp_material.deltaer.append(er_delta[i])
                    disp_material.tau.append(tau[i])
                else:
                    logger.exception(f"{self.params_str()} requires positive values for the permittivity difference.")
                    raise ValueError
            if disp_material.poles > config.get_model_config().materials["maxpoles"]:
                config.get_model_config().materials["maxpoles"] = disp_material.poles

            # Replace original material with newly created DispersiveMaterial
            grid.materials = [disp_material if mat.numID == material.numID else mat for mat in grid.materials]

            logger.info(
                f"{self.grid_name(grid)}Debye disperion added to {disp_material.ID} "
                f"with delta_eps_r={', '.join(f'{deltaer:4.2f}' for deltaer in disp_material.deltaer)}, "
                f"and tau={', '.join(f'{tau:4.3e}' for tau in disp_material.tau)} secs created."
            )


class AddLorentzDispersion(GridUserObject):
    """Adds dispersive properties to already defined Material based on a
        multi-pole Lorentz formulation.

    Attributes:
        poles: float required for number of Lorentz poles.
        er_delta: tuple required for difference between zero-frequency relative
                    permittivity and relative permittivity at infinite frequency
                    for each pole.
        omega: tuple required for frequency (Hz) for each pole.
        delta: tuple required for damping coefficient (Hz) for each pole.
        material_ids: list required of material ids to apply disperive
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

        if poles < 0:
            logger.exception(f"{self.params_str()} requires a positive value for number of poles.")
            raise ValueError

        # Look up requested materials in existing list of material instances
        materials = [y for x in material_ids for y in grid.materials if y.ID == x]

        if len(materials) != len(material_ids):
            notfound = [x for x in material_ids if x not in materials]
            logger.exception(f"{self.params_str()} material(s) {notfound} do not exist")
            raise ValueError

        for material in materials:
            disp_material = DispersiveMaterialUser(material.numID, material.ID)
            disp_material.er = material.er
            disp_material.se = material.se
            disp_material.mr = material.mr
            disp_material.sm = material.sm
            disp_material.type = "lorentz"
            disp_material.poles = poles
            disp_material.averagable = False
            for i in range(poles):
                if (
                    er_delta[i] > 0
                    and omega[i] < (2.0 * np.pi) / grid.dt
                    and delta[i] < 1.0 / grid.dt
                    and omega[i] != delta[i]
                ):
                    disp_material.deltaer.append(er_delta[i])
                    disp_material.tau.append(omega[i])
                    disp_material.alpha.append(delta[i])
                else:
                    logger.exception(
                        f"{self.params_str()} requires positive "
                        "values for the permittivity difference "
                        "and frequencies, and associated times "
                        "that are greater than the inverse of the time step for "
                        "the model."
                    )
                    raise ValueError
            if disp_material.poles > config.get_model_config().materials["maxpoles"]:
                config.get_model_config().materials["maxpoles"] = disp_material.poles

            # Replace original material with newly created DispersiveMaterial
            grid.materials = [disp_material if mat.numID == material.numID else mat for mat in grid.materials]

            logger.info(
                f"{self.grid_name(grid)}Lorentz disperion added to {disp_material.ID} "
                f"with delta_eps_r={', '.join(f'{deltaer:4.2f}' for deltaer in disp_material.deltaer)}, "
                f"omega={', '.join(f'{omega:4.3e}' for omega in disp_material.tau)} Hertz, "
                f"and delta={', '.join(f'{delta:4.3e}' for delta in disp_material.alpha)} Hertz, created."
            )


class AddDrudeDispersion(GridUserObject):
    """Adds dispersive properties to already defined Material based on a
        multi-pole Drude formulation.

    Attributes:
        poles: float required for number of Drude poles.
        omega: tuple required for frequency (Hz) for each pole.
        alpha: tuple required for inverse of relaxation time (secs) for each pole.
        material_ids: list required of material ids to apply disperive
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

        if poles < 0:
            logger.exception(f"{self.params_str()} requires a positive value for number of poles.")
            raise ValueError

        # Look up requested materials in existing list of material instances
        materials = [y for x in material_ids for y in grid.materials if y.ID == x]

        if len(materials) != len(material_ids):
            notfound = [x for x in material_ids if x not in materials]
            logger.exception(f"{self.params_str()} material(s) {notfound} do not exist.")
            raise ValueError

        for material in materials:
            disp_material = DispersiveMaterialUser(material.numID, material.ID)
            disp_material.er = material.er
            disp_material.se = material.se
            disp_material.mr = material.mr
            disp_material.sm = material.sm
            disp_material.type = "drude"
            disp_material.poles = poles
            disp_material.averagable = False
            for i in range(poles):
                if omega[i] < (2.0 * np.pi) / grid.dt and alpha[i] < 1.0 / grid.dt:
                    disp_material.tau.append(omega[i])
                    disp_material.alpha.append(alpha[i])
                else:
                    logger.exception(
                        f"{self.params_str()} requires positive "
                        + "values for the frequencies, and "
                        + "associated times that are greater than "
                        + "the inverse of time step for the model."
                    )
                    raise ValueError
            if disp_material.poles > config.get_model_config().materials["maxpoles"]:
                config.get_model_config().materials["maxpoles"] = disp_material.poles

            # Replace original material with newly created DispersiveMaterial
            grid.materials = [disp_material if mat.numID == material.numID else mat for mat in grid.materials]

            logger.info(
                f"{self.grid_name(grid)}Drude disperion added to {disp_material.ID} "
                f"with omega={', '.join(f'{omega:4.3e}' for omega in disp_material.tau)} Hertz, "
                f"and alpha={', '.join(f'{alpha:4.3e}' for alpha in disp_material.alpha)} Hertz created."
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

        # sand_fraction/clay_fraction are physical fractions - values
        # above 1 were previously unvalidated.
        if not (0 <= sand_fraction <= 1):
            logger.exception(f"{self.params_str()} requires the sand fraction to be between 0 and 1.")
            raise ValueError
        if not (0 <= clay_fraction <= 1):
            logger.exception(f"{self.params_str()} requires the clay fraction to be between 0 and 1.")
            raise ValueError
        if bulk_density < 0:
            logger.exception(f"{self.params_str()} requires a positive value for the bulk density.")
            raise ValueError
        # sand_density is a divisor in PeplinskiSoil.calculate_properties()
        # (materials.py) - zero would raise a ZeroDivisionError there,
        # not here, at build time.
        if sand_density <= 0:
            logger.exception(
                f"{self.params_str()} requires a value greater than zero for the sand particle " "density."
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
            logger.exception(f"{self.params_str()} requires a positive value for the lower limit of conductivity.")
            raise ValueError
        if ro_lower < 0:
            logger.exception(f"{self.params_str()} requires a positive value for the lower range magnetic loss.")
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
            logger.exception(f"{self.params_str()} requires a positive value for the upper range of conductivity.")
            raise ValueError
        if ro_upper < 0:
            logger.exception(f"{self.params_str()} requires a positive value for the upper range of magnetic loss.")
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
        return "#material_range"

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

        logger.info(f"{self.grid_name(grid)}A list of materials used to create {s.ID} that includes {s.mat}, created")

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
            logger.exception(f"{self.params_str()} must have scaling type {','.join(CFSParameter.scalingdirections)}")
            raise ValueError
        if (
            float(alphamin) < 0
            or float(alphamax) < 0
            or float(kappamin) < 0
            or float(kappamax) < 0
            or float(sigmamin) < 0
        ):
            logger.exception(f"{self.params_str()} minimum and maximum scaling values must be greater than zero.")
            raise ValueError

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
        if sigmamax == "None":
            sigmamax = None
        if sigmamax is not None:
            sigmamax = float(sigmamax)
        cfssigma.max = sigmamax
        cfs = CFS()
        cfs.alpha = cfsalpha
        cfs.kappa = cfskappa
        cfs.sigma = cfssigma

        logger.info(
            f"PML CFS parameters: alpha (scaling: {cfsalpha.scalingprofile}, "
            f"scaling direction: {cfsalpha.scalingdirection}, min: "
            f"{cfsalpha.min:g}, max: {cfsalpha.max:g}), kappa (scaling: "
            f"{cfskappa.scalingprofile}, scaling direction: "
            f"{cfskappa.scalingdirection}, min: {cfskappa.min:g}, max: "
            f"{cfskappa.max:g}), sigma (scaling: {cfssigma.scalingprofile}, "
            f"scaling direction: {cfssigma.scalingdirection}, min: "
            f"{cfssigma.min:g}, max: {cfssigma.max}) created."
        )

        grid.pmls["cfs"].append(cfs)

        if len(grid.pmls["cfs"]) > 2:
            logger.exception(f"{self.params_str()} can only be used up to two times, for up to a 2nd order PML.")
            raise ValueError


class SymmetryBoundary(GridUserObject):
    """Sets a PEC or PMC symmetry boundary condition on a model-domain face.

    The selected boundary replaces the PML on that face. A PEC boundary
    forces the tangential electric-field component IDs to the built-in PEC
    material during grid construction. A PMC boundary uses an image-theory
    ghost-node update for the on-wall tangential electric fields.

    Nondispersive PMC boundaries are supported by the CPU, CUDA, OpenCL, and
    Metal solvers. Dispersive PMC boundaries are currently CPU-only.
    Symmetry boundaries are not supported in 2D mode, with MPI, or on a
    subgrid, although they may be used on the main grid of a model that
    contains subgrids.

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
            logger.exception(f"{self.params_str()} face must be one of {', '.join(self.VALID_FACES)}.")
            raise ValueError

        if boundary_type not in self.VALID_TYPES:
            logger.exception(f"{self.params_str()} type must be one of {', '.join(self.VALID_TYPES)}.")
            raise ValueError

        if config.sim_config.mpi:
            logger.exception(f"{self.params_str()} cannot currently be used with MPI.")
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
            logger.exception(f"{self.params_str()} a symmetry boundary has already been set on face '{face}'.")
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
