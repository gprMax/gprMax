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
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import interpolate

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import DispersiveMaterial as DispersiveMaterialUser
from gprMax.materials import ListMaterial as ListMaterialUser
from gprMax.materials import Material as MaterialUser
from gprMax.materials import PeplinskiSoil as PeplinskiSoilUser
from gprMax.materials import RangeMaterial as RangeMaterialUser
from gprMax.pml import CFS, CFSParameter
from gprMax.receivers import Rx as RxUser
from gprMax.sources import DiscretePlaneWave as DiscretePlaneWaveUser
from gprMax.sources import HertzianDipole as HertzianDipoleUser
from gprMax.sources import MagneticDipole as MagneticDipoleUser
from gprMax.sources import TransmissionLine as TransmissionLineUser
from gprMax.sources import VoltageSource as VoltageSourceUser
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
        waveformvalues = np.loadtxt(
            excitationfile, skiprows=1, dtype=config.sim_config.dtypes["float_or_double"]
        )

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
            singlewaveformvalues = (
                waveformvalues[:] if len(waveformvalues.shape) == 1 else waveformvalues[:, i]
            )

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
                logger.exception(
                    self.params_str() + (" builtin waveforms require exactly four parameters.")
                )
                raise
            if freq <= 0:
                logger.exception(
                    self.params_str()
                    + (" requires an excitation " "frequency value of greater than zero.")
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
                uservalues = self.kwargs["user_values"]
                ID = self.kwargs["id"]
                fullargspec = inspect.getfullargspec(interpolate.interp1d)
                kwargs = dict(zip(reversed(fullargspec.args), reversed(fullargspec.defaults)))
            except KeyError:
                logger.exception(
                    self.params_str()
                    + (" a user-defined waveform requires at least two parameters.")
                )
                raise

            if "user_time" in self.kwargs:
                waveformtime = self.kwargs["user_time"]
            else:
                waveformtime = np.arange(0, grid.timewindow + grid.dt, grid.dt)

            # Set args for interpolation if given by user
            if "kind" in self.kwargs:
                kwargs["kind"] = self.kwargs["kind"]
            if "fill_value" in self.kwargs:
                kwargs["fill_value"] = self.kwargs["fill_value"]

            if any(x.ID == ID for x in grid.waveforms):
                logger.exception(self.params_str() + (f" with ID {ID} already exists."))
                raise ValueError

            w = WaveformUser()
            w.ID = ID
            w.type = wavetype
            w.userfunc = interpolate.interp1d(waveformtime, uservalues, **kwargs)

            logger.info(self.grid_name(grid) + (f"Waveform {w.ID} that is user-defined created."))

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
    ):
        super().__init__(
            p1=p1,
            polarisation=polarisation,
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
        rot_pol_pts, self.polarisation = rotate_polarisation(
            self.point, self.polarisation, self.axis, self.angle, grid
        )
        rot_pts = rotate_2point_object(rot_pol_pts, self.axis, self.angle, self.origin)
        self.point = tuple(rot_pts[0, :])

    def _validate_parameters(self, grid: FDTDGrid):
        # Check polarity
        self.polarisation = self.polarisation.lower()
        if self.polarisation not in ("x", "y", "z"):
            raise ValueError(f"{self.params_str()} polarisation must be x, y, or z.")
        if "2D TMx" in config.get_model_config().mode and self.polarisation in ["y", "z"]:
            raise ValueError(f"{self.params_str()} polarisation must be x in 2D TMx mode.")
        elif "2D TMy" in config.get_model_config().mode and self.polarisation in ["x", "z"]:
            raise ValueError(f"{self.params_str()} polarisation must be y in 2D TMy mode.")
        elif "2D TMz" in config.get_model_config().mode and self.polarisation in ["x", "y"]:
            raise ValueError(f"{self.params_str()} polarisation must be z in 2D TMz mode.")

        # Check resistance
        if self.resistance < 0:
            raise ValueError(
                f"{self.params_str()} requires a source resistance of zero or greater."
            )

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == self.waveform_id for x in grid.waveforms):
            raise ValueError(
                f"{self.params_str()} there is no waveform with the identifier {self.waveform_id}."
            )

        # Check start and stop
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
        uip = self._create_uip(grid)
        x, y, z = uip.discretise_static_point(self.point)
        voltage_source.ID = f"{voltage_source.__class__.__name__}({x},{y},{z})"
        voltage_source.resistance = self.resistance
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
        point_within_grid, discretised_point = uip.check_src_rx_point(self.point, self.params_str())

        if point_within_grid:
            self._validate_parameters(grid)
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
        super().__init__(
            p1=p1, polarisation=polarisation, waveform_id=waveform_id, start=start, stop=stop
        )

        self.point = p1
        self.polarisation = polarisation.lower()
        self.waveform_id = waveform_id
        self.start = start
        self.stop = stop

    def _do_rotate(self, grid: FDTDGrid):
        """Performs rotation."""
        rot_pol_pts, self.polarisation = rotate_polarisation(
            self.point, self.polarisation, self.axis, self.angle, grid
        )
        rot_pts = rotate_2point_object(rot_pol_pts, self.axis, self.angle, self.origin)
        self.point = tuple(rot_pts[0, :])

    def _validate_parameters(self, grid: FDTDGrid):
        # Check polarity
        self.polarisation = self.polarisation.lower()
        if self.polarisation not in ("x", "y", "z"):
            raise ValueError(f"{self.params_str()} polarisation must be x, y, or z.")
        if "2D TMx" in config.get_model_config().mode and self.polarisation in ["y", "z"]:
            raise ValueError(f"{self.params_str()} polarisation must be x in 2D TMx mode.")
        elif "2D TMy" in config.get_model_config().mode and self.polarisation in ["x", "z"]:
            raise ValueError(f"{self.params_str()} polarisation must be y in 2D TMy mode.")
        elif "2D TMz" in config.get_model_config().mode and self.polarisation in ["x", "y"]:
            raise ValueError(f"{self.params_str()} polarisation must be z in 2D TMz mode.")

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == self.waveform_id for x in grid.waveforms):
            raise ValueError(
                f"{self.params_str()} there is no waveform with the identifier {self.waveform_id}."
            )

        # Check start and stop
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
        if self.do_rotate:
            self._do_rotate(grid)

        # Check the position of the hertzian dipole
        uip = self._create_uip(grid)
        point_within_grid, discretised_point = uip.check_src_rx_point(self.point, self.params_str())

        if point_within_grid:
            self._validate_parameters(grid)
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
        super().__init__(
            p1=p1, polarisation=polarisation, waveform_id=waveform_id, start=start, stop=stop
        )

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
        point_within_grid, discretised_point = uip.check_src_rx_point(self.point, self.params_str())

        if point_within_grid:
            self._validate_parameters(grid)
            magnetic_dipole = self._create_magnetic_dipole(grid, discretised_point)
            grid.add_source(magnetic_dipole)
            position = uip.round_to_grid_static_point(self.point)
            self._log(grid, magnetic_dipole, *position)

    def _do_rotate(self, grid: FDTDGrid):
        """Performs rotation."""
        rot_pol_pts, self.polarisation = rotate_polarisation(
            self.point, self.polarisation, self.axis, self.angle, grid
        )
        rot_pts = rotate_2point_object(rot_pol_pts, self.axis, self.angle, self.origin)
        self.point = tuple(rot_pts[0, :])

    def _validate_parameters(self, grid: FDTDGrid):
        # Check polarity
        self.polarisation = self.polarisation.lower()
        if self.polarisation not in ("x", "y", "z"):
            raise ValueError(f"{self.params_str()} polarisation must be x, y, or z.")
        if "2D TMx" in config.get_model_config().mode and self.polarisation in ["y", "z"]:
            raise ValueError(f"{self.params_str()} polarisation must be x in 2D TMx mode.")
        elif "2D TMy" in config.get_model_config().mode and self.polarisation in ["x", "z"]:
            raise ValueError(f"{self.params_str()} polarisation must be y in 2D TMy mode.")
        elif "2D TMz" in config.get_model_config().mode and self.polarisation in ["x", "y"]:
            raise ValueError(f"{self.params_str()} polarisation must be z in 2D TMz mode.")

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == self.waveform_id for x in grid.waveforms):
            raise ValueError(
                f"{self.params_str()} there is no waveform with the identifier {self.waveform_id}."
            )

        # Check start and stop
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
        field location.

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
            p1=p1,
            polarisation=polarisation,
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
        rot_pol_pts, self.polarisation = rotate_polarisation(
            self.point, self.polarisation, self.axis, self.angle, grid
        )
        rot_pts = rotate_2point_object(rot_pol_pts, self.axis, self.angle, self.origin)
        self.point = tuple(rot_pts[0, :])

    def build(self, grid: FDTDGrid):
        if self.do_rotate:
            self._do_rotate(grid)

        # Check the position of the voltage source
        uip = self._create_uip(grid)
        point_within_grid, discretised_point = uip.check_src_rx_point(self.point, self.params_str())

        if point_within_grid:
            self._validate_parameters(grid)
            transmission_line = self._create_transmission_line(grid, discretised_point)
            grid.add_source(transmission_line)
            position = uip.round_to_grid_static_point(self.point)
            self._log(grid, transmission_line, *position)

    def _validate_parameters(self, grid: FDTDGrid):
        # Warn about using a transmission line on GPU
        if config.sim_config.general["solver"] in ["cuda", "opencl"]:
            raise ValueError(
                f"{self.params_str()} cannot currently be used "
                "with the CUDA or OpenCL-based solver. Consider "
                "using a #voltage_source instead."
            )

        # Check polarity
        self.polarisation = self.polarisation.lower()
        if self.polarisation not in ("x", "y", "z"):
            raise ValueError(f"{self.params_str()} polarisation must be x, y, or z.")
        if "2D TMx" in config.get_model_config().mode and self.polarisation in ["y", "z"]:
            raise ValueError(f"{self.params_str()} polarisation must be x in 2D TMx mode.")
        elif "2D TMy" in config.get_model_config().mode and self.polarisation in ["x", "z"]:
            raise ValueError(f"{self.params_str()} polarisation must be y in 2D TMy mode.")
        elif "2D TMz" in config.get_model_config().mode and self.polarisation in ["x", "y"]:
            raise ValueError(f"{self.params_str()} polarisation must be z in 2D TMz mode.")

        # Check resistance
        if self.resistance <= 0 or self.resistance >= config.sim_config.em_consts["z0"]:
            raise ValueError(
                f"{self.params_str()} requires a resistance "
                "greater than zero and less than the impedance "
                "of free space, i.e. 376.73 Ohms."
            )

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == self.waveform_id for x in grid.waveforms):
            raise ValueError(
                f"{self.params_str()} there is no waveform with the identifier {self.waveform_id}."
            )

        # Check start and stop
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


class DiscretePlaneWave(GridUserObject):
    """
    Specifies a plane wave implemented using the discrete plane wave formulation.

    Attributes:
        theta: float required for propagation angle (degrees) of wave.
        phi: float required for propagation angle (degrees) of wave.
        psi: float required for polarisation of wave.
        delta_theta: float optional for tolerance of theta angle to nearest
                        rational angle.
        delta_phi: float optional for tolerance to phi angle to nearest
                        rational angle.
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
        return "#discrete_plane_wave"

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
            dtheta = self.kwargs["delta_theta"]
            dphi = self.kwargs["delta_phi"]
        except KeyError:
            dtheta = 1.0
            dphi = 1.0

        # Warn about using a discrete plane wave on GPU
        if config.sim_config.general["solver"] in ["cuda", "opencl"]:
            logger.exception(
                f"{self.params_str()} cannot currently be used "
                + "with the CUDA or OpenCL-based solver. "
            )
            raise ValueError

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            logger.exception(
                f"{self.params_str()} there is no waveform " + f"with the identifier {waveform_id}."
            )
            raise ValueError

        if theta > 180:
            theta -= np.floor(theta / 180) * 180.0
        if phi > 360:
            phi -= np.floor(phi / 360) * 360.0
        if psi > 360:
            psi -= np.floor(psi / 360) * 360.0

        uip = self._create_uip(grid)
        _, start = uip.check_src_rx_point(p1, self.params_str())
        _, stop = uip.check_src_rx_point(p2, self.params_str())

        DPW = DiscretePlaneWaveUser(grid)
        DPW.corners = np.array([*start, *stop], dtype=np.int32)
        DPW.waveformID = waveform_id
        DPW.initializeDiscretePlaneWave(psi, phi, dphi, theta, dtheta, grid)

        try:
            DPW.material_ID = self.kwargs["material_id"]
        except KeyError:
            DPW.material_ID = 1

        try:
            # Check source start & source remove time parameters
            start = self.kwargs["start"]
            stop = self.kwargs["stop"]
            if start < 0:
                logger.exception(
                    self.params_str()
                    + (" delay of the initiation " "of the source should not " "be less than zero.")
                )
                raise ValueError
            if stop < 0:
                logger.exception(
                    self.params_str() + (" time to remove the source should not be less than zero.")
                )
                raise ValueError
            if stop - start <= 0:
                logger.exception(
                    self.params_str() + (" duration of the source should not be zero or less.")
                )
                raise ValueError
            DPW.start = start
            DPW.stop = min(stop, grid.timewindow)
            startstop = f" start time {t.start:g} secs, finish time {t.stop:g} secs "
        except KeyError:
            DPW.start = 0
            DPW.stop = grid.timewindow
            startstop = " "

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
        phi_approx = np.arctan2(DPW.m[1] / grid.dy, DPW.m[0] / grid.dx) * 180 / np.pi
        theta_approx = (
            np.arctan2(
                np.sqrt(
                    (DPW.m[0] / grid.dx) * (DPW.m[0] / grid.dx)
                    + (DPW.m[1] / grid.dy) * (DPW.m[1] / grid.dy)
                ),
                DPW.m[2] / grid.dz,
            )
            * 180
            / np.pi
        )
        logger.info(
            f"{self.grid_name(grid)}Discrete Plane Wave has been discretized "
            + "the angles have been approximated to the nearest rational angles "
            + "with some small tolerance levels. The chosen rational integers are "
            + f"[m_x, m_y, m_z] : {DPW.m[:3]}. The approximated angles are: "
            + f"Phi: {phi_approx:.3f} and Theta: {theta_approx:.3f}"
        )

        grid.discreteplanewaves.append(DPW)


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
        if config.sim_config.general["solver"] in ["cuda", "opencl"]:
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
        if self.do_rotate:
            self._do_rotate(grid)

        # Check position of the receiver
        uip = self._create_uip(grid)
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
        _, discretised_lower_point = uip.check_src_rx_point(
            self.lower_point, self.params_str(), "lower"
        )
        _, discretised_upper_point = uip.check_src_rx_point(
            self.lower_point, self.params_str(), "upper"
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
        dx, dy, dz = uip.round_to_grid_static_point(self.dl)

        logger.info(
            f"{self.grid_name(grid)}Receiver array"
            f" {xs:g}m, {ys:g}m, {zs:g}m, to"
            f" {xf:g}m, {yf:g}m, {zf:g}m with steps"
            f" {dx:g}m, {dy:g}m, {dz:g}m"
        )

        for x in range(xs, xf + grid.dx, dx):
            for y in range(ys, yf + grid.dy, dy):
                for z in range(zs, zf + grid.dz, dz):
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
                logger.exception(
                    f"{self.params_str()} requires a positive value for electric conductivity."
                )
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

        # Set material averaging to False if infinite conductivity, i.e. pec
        if m.se == float("inf"):
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
                    logger.exception(
                        f"{self.params_str()} requires positive values for the permittivity difference."
                    )
                    raise ValueError
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
                if er_delta[i] > 0 and omega[i] > grid.dt and delta[i] > grid.dt:
                    disp_material.deltaer.append(er_delta[i])
                    disp_material.tau.append(omega[i])
                    disp_material.alpha.append(delta[i])
                else:
                    logger.exception(
                        f"{self.params_str()} requires positive "
                        "values for the permittivity difference "
                        "and frequencies, and associated times "
                        "that are greater than the time step for "
                        "the model."
                    )
                    raise ValueError
            if disp_material.poles > config.get_model_config().materials["maxpoles"]:
                config.get_model_config().materials["maxpoles"] = disp_material.poles

            # Replace original material with newly created DispersiveMaterial
            grid.materials = [
                disp_material if mat.numID == material.numID else mat for mat in grid.materials
            ]

            logger.info(
                f"{self.grid_name(grid)}Lorentz disperion added to {disp_material.ID} "
                f"with delta_eps_r={', '.join(f'{deltaer:4.2f}' for deltaer in disp_material.deltaer)}, "
                f"omega={', '.join(f'{omega:4.3e}' for omega in disp_material.tau)} secs, "
                f"and delta={', '.join(f'{delta:4.3e}' for delta in disp_material.alpha)} created."
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
                if omega[i] > 0 and alpha[i] > grid.dt:
                    disp_material.tau.append(omega[i])
                    disp_material.alpha.append(alpha[i])
                else:
                    logger.exception(
                        f"{self.params_str()} requires positive "
                        + "values for the frequencies, and "
                        + "associated times that are greater than "
                        + "the time step for the model."
                    )
                    raise ValueError
            if disp_material.poles > config.get_model_config().materials["maxpoles"]:
                config.get_model_config().materials["maxpoles"] = disp_material.poles

            # Replace original material with newly created DispersiveMaterial
            grid.materials = [
                disp_material if mat.numID == material.numID else mat for mat in grid.materials
            ]

            logger.info(
                f"{self.grid_name(grid)}Drude disperion added to {disp_material.ID} "
                f"with omega={', '.join(f'{omega:4.3e}' for omega in disp_material.tau)} secs, "
                f"and alpha={', '.join(f'{alpha:4.3e}' for alpha in disp_material.alpha)} secs created."
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

        if sand_fraction < 0:
            logger.exception(
                f"{self.params_str()} requires a positive value for the sand fraction."
            )
            raise ValueError
        if clay_fraction < 0:
            logger.exception(
                f"{self.params_str()} requires a positive value for the clay fraction."
            )
            raise ValueError
        if bulk_density < 0:
            logger.exception(f"{self.params_str()} requires a positive value for the bulk density.")
            raise ValueError
        if sand_density < 0:
            logger.exception(
                f"{self.params_str()} requires a positive value for the sand particle density."
            )
            raise ValueError
        if water_fraction_lower < 0:
            logger.exception(
                f"{self.params_str()} requires a positive value for the lower limit of the water volumetric "
                "fraction."
            )
            raise ValueError
        if water_fraction_upper < 0:
            logger.exception(
                f"{self.params_str()} requires a positive value for the upper limit of the water volumetric "
                "fraction."
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

        logger.info(
            f"{self.grid_name(grid)}A list of materials used to create {s.ID} that includes {s.mat}, created"
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
        if (
            float(alphamin) < 0
            or float(alphamax) < 0
            or float(kappamin) < 0
            or float(kappamax) < 0
            or float(sigmamin) < 0
        ):
            logger.exception(
                f"{self.params_str()} minimum and maximum scaling values must be greater than zero."
            )
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
            logger.exception(
                f"{self.params_str()} can only be used up to two times, for up to a 2nd order PML."
            )
            raise ValueError


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
