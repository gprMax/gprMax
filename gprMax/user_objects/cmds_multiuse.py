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

import inspect
import logging
from os import PathLike
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import interpolate

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.materials import DispersiveMaterial as DispersiveMaterialUser
from gprMax.materials import ListMaterial as ListMaterialUser
from gprMax.materials import Material as MaterialUser
from gprMax.materials import PeplinskiSoil as PeplinskiSoilUser
from gprMax.materials import RangeMaterial as RangeMaterialUser
from gprMax.model import Model
from gprMax.pml import CFS, CFSParameter
from gprMax.receivers import Rx as RxUser
from gprMax.snapshots import MPISnapshot as MPISnapshotUser
from gprMax.snapshots import Snapshot as SnapshotUser
from gprMax.sources import HertzianDipole as HertzianDipoleUser
from gprMax.sources import MagneticDipole as MagneticDipoleUser
from gprMax.sources import TransmissionLine as TransmissionLineUser
from gprMax.sources import VoltageSource as VoltageSourceUser
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.user_objects.cmds_geometry.cmds_geometry import (
    rotate_2point_object,
    rotate_polarisation,
)
from gprMax.user_objects.rotatable import RotatableMixin
from gprMax.user_objects.user_objects import GridUserObject
from gprMax.utilities.utilities import round_value
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
        voltage_source.ID = (
            voltage_source.__class__.__name__
            + "("
            + str(voltage_source.xcoord)
            + ","
            + str(voltage_source.ycoord)
            + ","
            + str(voltage_source.zcoord)
            + ")"
        )
        voltage_source.resistance = self.resistance
        voltage_source.waveform = grid.get_waveform_by_id(self.waveform_id)

        if self.start is None or self.stop is None:
            voltage_source.start = 0
            voltage_source.stop = grid.timewindow
        else:
            voltage_source.start = self.start
            voltage_source.stop = min(self.stop, grid.timewindow)

        voltage_source.calculate_waveform_values(grid.iterations, grid.dt)

        return voltage_source

    def _log(self, grid: FDTDGrid, voltage_source: VoltageSourceUser):
        if self.start is None or self.stop is None:
            startstop = " "
        else:
            startstop = f" start time {voltage_source.start:g} secs, finish time {voltage_source.stop:g} secs "

        uip = self._create_uip(grid)
        p = uip.discretised_to_continuous(voltage_source.coord)

        logger.info(
            f"{self.grid_name(grid)}Voltage source with polarity"
            f" {voltage_source.polarisation} at {p[0]:g}m, {p[1]:g}m,"
            f" {p[2]:g}m, resistance {voltage_source.resistance:.1f}"
            f" Ohms,{startstop}using waveform {voltage_source.waveform.ID}"
            f" created."
        )

    def build(self, grid: FDTDGrid):
        if self.do_rotate:
            self._do_rotate(grid)

        # Check the position of the voltage source
        uip = self._create_uip(grid)
        discretised_point = uip.discretise_point(self.point)

        if uip.check_src_rx_point(discretised_point, self.params_str()):
            self._validate_parameters(grid)
            voltage_source = self._create_voltage_source(grid, discretised_point)
            grid.voltagesources.append(voltage_source)
            self._log(grid, voltage_source)


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
        h.ID = f"{h.__class__.__name__}({str(h.xcoord)},{str(h.ycoord)},{str(h.zcoord)})"
        h.waveform = grid.get_waveform_by_id(self.waveform_id)

        if self.start is None or self.stop is None:
            h.start = 0
            h.stop = grid.timewindow
        else:
            h.start = self.start
            h.stop = min(self.stop, grid.timewindow)

        h.calculate_waveform_values(grid.iterations, grid.dt)

        return h

    def _log(self, grid: FDTDGrid, hertzian_dipole: HertzianDipoleUser):
        if self.start is None or self.stop is None:
            startstop = " "
        else:
            startstop = f" start time {hertzian_dipole.start:g} secs, finish time {hertzian_dipole.stop:g} secs "

        uip = self._create_uip(grid)
        p = uip.discretised_to_continuous(hertzian_dipole.coord)

        if config.get_model_config().mode == "2D":
            logger.info(
                f"{self.grid_name(grid)}Hertzian dipole is a line source"
                f" in 2D with polarity {hertzian_dipole.polarisation} at"
                f" {p[0]:g}m, {p[1]:g}m, {p[2]:g}m,{startstop}using"
                f" waveform {hertzian_dipole.waveform.ID} created."
            )
        else:
            logger.info(
                f"{self.grid_name(grid)}Hertzian dipole with polarity"
                f" {hertzian_dipole.polarisation} at {p[0]:g}m,"
                f" {p[1]:g}m, {p[2]:g}m,{startstop} using"
                f" waveform {hertzian_dipole.waveform.ID} created."
            )

    def build(self, grid: FDTDGrid):
        if self.do_rotate:
            self._do_rotate(grid)

        # Check the position of the hertzian dipole
        uip = self._create_uip(grid)
        discretised_point = uip.discretise_point(self.point)

        if uip.check_src_rx_point(discretised_point, self.params_str()):
            self._validate_parameters(grid)
            hertzian_dipole = self._create_hertzian_dipole(grid, discretised_point)
            grid.hertziandipoles.append(hertzian_dipole)
            self._log(grid, hertzian_dipole)


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
        discretised_point = uip.discretise_point(self.point)

        if uip.check_src_rx_point(discretised_point, self.params_str()):
            self._validate_parameters(grid)
            magnetic_dipole = self._create_magnetic_dipole(grid, discretised_point)
            grid.magneticdipoles.append(magnetic_dipole)
            self._log(grid, magnetic_dipole)

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
        m.ID = (
            m.__class__.__name__
            + "("
            + str(m.xcoord)
            + ","
            + str(m.ycoord)
            + ","
            + str(m.zcoord)
            + ")"
        )
        m.waveform = grid.get_waveform_by_id(self.waveform_id)

        if self.start is None or self.stop is None:
            m.start = 0
            m.stop = grid.timewindow
        else:
            m.start = self.start
            m.stop = min(self.stop, grid.timewindow)

        m.calculate_waveform_values(grid.iterations, grid.dt)

        return m

    def _log(self, grid: FDTDGrid, m: MagneticDipoleUser):
        if self.start is None or self.stop is None:
            startstop = " "
        else:
            startstop = f" start time {m.start:g} secs, finish time {m.stop:g} secs "

        uip = self._create_uip(grid)
        p = uip.discretised_to_continuous(m.coord)

        logger.info(
            f"{self.grid_name(grid)}Magnetic dipole with polarity"
            f" {m.polarisation} at {p[0]:g}m, {p[1]:g}m, {p[2]:g}m,"
            f"{startstop}using waveform {m.waveform.ID} created."
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do_rotate(self, grid: FDTDGrid):
        """Performs rotation."""
        rot_pol_pts, self.kwargs["polarisation"] = rotate_polarisation(
            self.kwargs["p1"], self.kwargs["polarisation"], self.axis, self.angle, grid
        )
        rot_pts = rotate_2point_object(rot_pol_pts, self.axis, self.angle, self.origin)
        self.kwargs["p1"] = tuple(rot_pts[0, :])

    def build(self, grid: FDTDGrid):
        try:
            polarisation = self.kwargs["polarisation"].lower()
            p1 = self.kwargs["p1"]
            waveform_id = self.kwargs["waveform_id"]
            resistance = self.kwargs["resistance"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires at least six parameters.")
            raise

        if self.do_rotate:
            self._do_rotate(grid)

        # Warn about using a transmission line on GPU
        if config.sim_config.general["solver"] in ["cuda", "opencl"]:
            logger.exception(
                f"{self.params_str()} cannot currently be used "
                "with the CUDA or OpenCL-based solver. Consider "
                "using a #voltage_source instead."
            )
            raise ValueError

        # Check polarity & position parameters
        if polarisation not in ("x", "y", "z"):
            logger.exception(self.params_str() + (" polarisation must be " "x, y, or z."))
            raise ValueError
        if "2D TMx" in config.get_model_config().mode and polarisation in [
            "y",
            "z",
        ]:
            logger.exception(self.params_str() + (" polarisation must be x in " "2D TMx mode."))
            raise ValueError
        elif "2D TMy" in config.get_model_config().mode and polarisation in [
            "x",
            "z",
        ]:
            logger.exception(self.params_str() + (" polarisation must be y in " "2D TMy mode."))
            raise ValueError
        elif "2D TMz" in config.get_model_config().mode and polarisation in [
            "x",
            "y",
        ]:
            logger.exception(self.params_str() + (" polarisation must be z in " "2D TMz mode."))
            raise ValueError

        uip = self._create_uip(grid)
        xcoord, ycoord, zcoord = uip.check_src_rx_point(p1, self.params_str())
        p2 = uip.round_to_grid_static_point(p1)

        if resistance <= 0 or resistance >= config.sim_config.em_consts["z0"]:
            logger.exception(
                f"{self.params_str()} requires a resistance "
                "greater than zero and less than the impedance "
                "of free space, i.e. 376.73 Ohms."
            )
            raise ValueError

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            logger.exception(
                f"{self.params_str()} there is no waveform with the identifier {waveform_id}."
            )
            raise ValueError

        t = TransmissionLineUser(grid.iterations, grid.dt)
        t.polarisation = polarisation
        t.xcoord = xcoord
        t.ycoord = ycoord
        t.zcoord = zcoord
        t.ID = (
            t.__class__.__name__
            + "("
            + str(t.xcoord)
            + ","
            + str(t.ycoord)
            + ","
            + str(t.zcoord)
            + ")"
        )
        t.resistance = resistance
        t.waveform = grid.get_waveform_by_id(waveform_id)

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
                    self.params_str()
                    + (" time to remove the " "source should not be " "less than zero.")
                )
                raise ValueError
            if stop - start <= 0:
                logger.exception(
                    self.params_str()
                    + (" duration of the source " "should not be zero or " "less.")
                )
                raise ValueError
            t.start = start
            t.stop = min(stop, grid.timewindow)
            startstop = f" start time {t.start:g} secs, finish time {t.stop:g} secs "
        except KeyError:
            t.start = 0
            t.stop = grid.timewindow
            startstop = " "

        t.calculate_waveform_values(grid.iterations, grid.dt)
        t.calculate_incident_V_I(grid)

        logger.info(
            f"{self.grid_name(grid)}Transmission line with polarity "
            + f"{t.polarisation} at {p2[0]:g}m, {p2[1]:g}m, "
            + f"{p2[2]:g}m, resistance {t.resistance:.1f} Ohms,"
            + startstop
            + f"using waveform {t.waveform.ID} created."
        )

        grid.transmissionlines.append(t)


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Can this be removed?
        self.constructor = RxUser

    def _do_rotate(self, grid: FDTDGrid):
        """Performs rotation."""
        new_pt = (
            self.kwargs["p1"][0] + grid.dx,
            self.kwargs["p1"][1] + grid.dy,
            self.kwargs["p1"][2] + grid.dz,
        )
        pts = np.array([self.kwargs["p1"], new_pt])
        rot_pts = rotate_2point_object(pts, self.axis, self.angle, self.origin)
        self.kwargs["p1"] = tuple(rot_pts[0, :])

        # If specific field components are specified, set to output all components
        try:
            self.kwargs["id"]
            self.kwargs["outputs"]
            rxargs = dict(self.kwargs)
            del rxargs["outputs"]
            self.kwargs = rxargs
        except KeyError:
            pass

    def build(self, grid: FDTDGrid):
        try:
            p1 = self.kwargs["p1"]
        except KeyError:
            logger.exception(self.params_str())
            raise

        if self.do_rotate:
            self._do_rotate(grid)

        uip = self._create_uip(grid)
        p = uip.check_src_rx_point(p1, self.params_str())
        p2 = uip.round_to_grid_static_point(p1)

        r = self.constructor()
        r.xcoord, r.ycoord, r.zcoord = p
        r.xcoordorigin, r.ycoordorigin, r.zcoordorigin = p

        try:
            r.ID = self.kwargs["id"]
            outputs = self.kwargs["outputs"]
        except KeyError:
            # If no ID or outputs are specified, use default
            r.ID = f"{r.__class__.__name__}({str(r.xcoord)},{str(r.ycoord)},{str(r.zcoord)})"
            for key in RxUser.defaultoutputs:
                r.outputs[key] = np.zeros(
                    grid.iterations, dtype=config.sim_config.dtypes["float_or_double"]
                )
        else:
            outputs.sort()
            # Get allowable outputs
            if config.sim_config.general["solver"] in ["cuda", "opencl"]:
                allowableoutputs = RxUser.allowableoutputs_dev
            else:
                allowableoutputs = RxUser.allowableoutputs
            # Check and add field output names
            for field in outputs:
                if field in allowableoutputs:
                    r.outputs[field] = np.zeros(
                        grid.iterations, dtype=config.sim_config.dtypes["float_or_double"]
                    )
                else:
                    logger.exception(
                        f"{self.params_str()} contains an output "
                        f"type that is not allowable. Allowable "
                        f"outputs in current context are "
                        f"{allowableoutputs}."
                    )
                    raise ValueError

        logger.info(
            f"{self.grid_name(grid)}Receiver at {p2[0]:g}m, {p2[1]:g}m, "
            f"{p2[2]:g}m with output component(s) "
            f"{', '.join(r.outputs)} created."
        )

        grid.rxs.append(r)

        return r


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            dl = self.kwargs["dl"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires exactly 9 parameters")
            raise

        uip = self._create_uip(grid)
        xs, ys, zs = uip.check_src_rx_point(p1, self.params_str(), "lower")
        xf, yf, zf = uip.check_src_rx_point(p2, self.params_str(), "upper")
        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)
        dx, dy, dz = uip.discretise_point(dl)

        if xs > xf or ys > yf or zs > zf:
            logger.exception(
                f"{self.params_str()} the lower coordinates should be less than the upper coordinates."
            )
            raise ValueError
        if dx < 0 or dy < 0 or dz < 0:
            logger.exception(f"{self.params_str()} the step size should not be less than zero.")
            raise ValueError
        if dx < 1:
            if dx == 0:
                dx = 1
            else:
                logger.exception(
                    f"{self.params_str()} the step size should not be less than the spatial discretisation."
                )
                raise ValueError
        if dy < 1:
            if dy == 0:
                dy = 1
            else:
                logger.exception(
                    f"{self.params_str()} the step size should not be less than the spatial discretisation."
                )
                raise ValueError
        if dz < 1:
            if dz == 0:
                dz = 1
            else:
                logger.exception(
                    f"{self.params_str()} the step size should not be less than the spatial discretisation."
                )
                raise ValueError

        logger.info(
            f"{self.grid_name(grid)}Receiver array "
            f"{p3[0]:g}m, {p3[1]:g}m, {p3[2]:g}m, to "
            f"{p4[0]:g}m, {p4[1]:g}m, {p4[2]:g}m with steps "
            f"{dx * grid.dx:g}m, {dy * grid.dy:g}m, {dz * grid.dz:g}m"
        )

        for x in range(xs, xf + 1, dx):
            for y in range(ys, yf + 1, dy):
                for z in range(zs, zf + 1, dz):
                    r = RxUser()
                    r.xcoord = x
                    r.ycoord = y
                    r.zcoord = z
                    r.xcoordorigin = x
                    r.ycoordorigin = y
                    r.zcoordorigin = z
                    # Point relative to main grid
                    p5 = np.array([x, y, z])
                    p5 = uip.discretised_to_continuous(p5)
                    p5 = uip.round_to_grid_static_point(p5)
                    r.ID = f"{r.__class__.__name__}({str(x)},{str(y)},{str(z)})"
                    for key in RxUser.defaultoutputs:
                        r.outputs[key] = np.zeros(
                            grid.iterations, dtype=config.sim_config.dtypes["float_or_double"]
                        )
                    logger.info(
                        f"  Receiver at {p5[0]:g}m, {p5[1]:g}m, "
                        f"{p5[2]:g}m with output component(s) "
                        f"{', '.join(r.outputs)} created."
                    )
                    grid.rxs.append(r)


class Snapshot(GridUserObject):
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
                            '.vti' (default) or '.h5'
        outputs: optional list of outputs for receiver. It can be any
                    selection from Ex, Ey, Ez, Hx, Hy, or Hz.
    """

    # TODO: Make this an output user object
    @property
    def order(self):
        return 9

    @property
    def hash(self):
        return "#snapshot"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calculate_upper_bound(
        self, start: npt.NDArray, step: npt.NDArray, size: npt.NDArray
    ) -> npt.NDArray:
        # upper_bound = p2 + dl - ((snapshot_size - 1) % dl) - 1
        return start + step * np.ceil(size / step)

    def build(self, grid: FDTDGrid):
        if isinstance(grid, SubGridBaseGrid):
            logger.exception(f"{self.params_str()} do not add snapshots to subgrids.")
            raise ValueError
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            dl = self.kwargs["dl"]
            filename = self.kwargs["filename"]
        except KeyError:
            logger.exception(f"{self.params_str()} requires exactly 11 parameters.")
            raise

        uip = self._create_uip(grid)
        dl = np.array(uip.discretise_static_point(dl))

        try:
            p1, p2 = uip.check_box_points(p1, p2, self.params_str())
        except ValueError:
            logger.exception(f"{self.params_str()} point is outside the domain.")
            raise

        p1 = np.array(p1)
        p2 = np.array(p2)
        snapshot_size = p2 - p1

        # If p2 does not line up with the set discretisation, the actual
        # maximum element accessed in the grid will be this upper bound.
        upper_bound = self._calculate_upper_bound(p1, dl, snapshot_size)

        # Each coordinate may need a different method to correct p2.
        # Therefore, this check needs to be repeated after each
        # correction has been applied.
        while any(p2 < upper_bound):
            # Ideally extend p2 up to the correct upper bound. This will
            # not change the snapshot output.
            if uip.point_within_bounds(upper_bound, "", "", ignore_error=True):
                p2 = upper_bound
                p2_continuous = uip.discretised_to_continuous(p2)
                logger.warning(
                    f"{self.params_str()} upper bound not aligned with discretisation. Updating 'p2'"
                    f" to {p2_continuous}"
                )
            # If the snapshot size cannot be increased, the
            # discretisation may need reducing. E.g. for snapshots of 2D
            # models.
            elif any(dl > snapshot_size):
                dl = np.where(dl > snapshot_size, snapshot_size, dl)
                upper_bound = self._calculate_upper_bound(p1, dl, snapshot_size)
                dl_continuous = uip.discretised_to_continuous(dl)
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
                p2 = np.where(uip.grid_upper_bound() < upper_bound, p2 - (snapshot_size % dl), p2)
                snapshot_size = p2 - p1
                upper_bound = self._calculate_upper_bound(p1, dl, snapshot_size)
                p2_continuous = uip.discretised_to_continuous(p2)
                logger.warning(
                    f"{self.params_str()} current bounds and discretisation would go outside"
                    f" domain. Limiting 'p2' to {p2_continuous}"
                )

        if any(dl < 0):
            logger.exception(f"{self.params_str()} the step size should not be less than zero.")
            raise ValueError
        if any(dl < 1):
            logger.exception(
                f"{self.params_str()} the step size should not be less than the spatial discretisation."
            )
            raise ValueError

        # If number of iterations given
        try:
            iterations = self.kwargs["iterations"]
        # If real floating point value given
        except KeyError:
            try:
                time = self.kwargs["time"]
            except KeyError:
                logger.exception(f"{self.params_str()} requires exactly 5 parameters.")
                raise
            if time > 0:
                iterations = round_value((time / grid.dt)) + 1
            else:
                logger.exception(f"{self.params_str()} time value must be greater than zero.")
                raise ValueError

        if iterations <= 0 or iterations > grid.iterations:
            logger.exception(f"{self.params_str()} time value is not valid.")
            raise ValueError

        try:
            fileext = self.kwargs["fileext"]
            if fileext not in SnapshotUser.fileexts:
                logger.exception(
                    f"'{fileext}' is not a valid format for a "
                    "snapshot file. Valid options are: "
                    f"{' '.join(SnapshotUser.fileexts)}."
                )
                raise ValueError
        except KeyError:
            fileext = SnapshotUser.fileexts[0]

        if isinstance(grid, MPIGrid) and fileext != ".h5":
            logger.exception(
                f"{self.params_str()} currently only '.h5' snapshots are compatible with MPI."
            )
            raise ValueError

        try:
            tmp = self.kwargs["outputs"]
            outputs = dict.fromkeys(SnapshotUser.allowableoutputs, False)
            # Check and set output names
            for output in tmp:
                if output not in SnapshotUser.allowableoutputs.keys():
                    logger.exception(
                        f"{self.params_str()} contains an output "
                        f"type that is not allowable. Allowable "
                        f"outputs in current context are "
                        f"{', '.join(SnapshotUser.allowableoutputs.keys())}."
                    )
                    raise ValueError
                else:
                    outputs[output] = True
        except KeyError:
            # If outputs are not specified, use default
            outputs = dict.fromkeys(SnapshotUser.allowableoutputs, True)

        if isinstance(grid, MPIGrid):
            snapshot_type = MPISnapshotUser
        else:
            snapshot_type = SnapshotUser

        xs, ys, zs = p1
        xf, yf, zf = p2
        dx, dy, dz = dl

        s = snapshot_type(
            xs,
            ys,
            zs,
            xf,
            yf,
            zf,
            dx,
            dy,
            dz,
            iterations,
            filename,
            fileext=fileext,
            outputs=outputs,
            grid_dl=grid.dl,
            grid_dt=grid.dt,
        )

        logger.info(
            f"Snapshot from {xs * grid.dx:g}m, {ys * grid.dy:g}m, {zs * grid.dz:g}m, to "
            f"{xf * grid.dx:g}m, {yf * grid.dy:g}m, {zf * grid.dz:g}m, discretisation "
            f"{dx * grid.dx:g}m, {dy * grid.dy:g}m, {dz * grid.dz:g}m, "
            f"at {s.time * grid.dt:g} secs with field outputs "
            f"{', '.join([k for k, v in outputs.items() if v])} and "
            f"filename {s.filename}{s.fileext} will be created."
        )

        grid.snapshots.append(s)


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
                f"and gamma={', '.join(f'{delta:4.3e}' for delta in disp_material.alpha)} created."
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
                f"and gamma={', '.join(f'{alpha:4.3e}' for alpha in disp_material.alpha)} secs created."
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
