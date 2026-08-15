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
from pathlib import Path

import h5py
import numpy as np

from gprMax.grid.fdtd_grid import FDTDGrid

from ._version import __version__

logger = logging.getLogger(__name__)


SOURCE_EXCITATION_SCHEMA_VERSION = 1
RECEIVER_TIMING_SCHEMA_VERSION = 1


def _waveform_metadata(grid, waveform_id):
    """Return serialisable metadata for a waveform used by a source."""

    waveform = next((item for item in grid.waveforms if item.ID == waveform_id), None)
    if waveform is None:
        return {}
    metadata = {
        "WaveformID": str(waveform_id or ""),
        "WaveformType": str(waveform.type or ""),
        "WaveformAmplitude": float(waveform.amp),
    }
    if waveform.freq is not None and np.isfinite(waveform.freq):
        metadata["WaveformFrequency"] = float(waveform.freq)
    return metadata


def _create_sample_dataset(group, name, samples):
    """Write a compressed one-dimensional source history."""

    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError(f"source excitation dataset {name!r} must be one-dimensional")
    kwargs = {"shuffle": True, "compression": "gzip"} if values.size else {}
    dataset = group.create_dataset(name, data=values, **kwargs)
    dataset.attrs["SampleCount"] = values.size
    return dataset


def _write_source_excitation(group, source, grid):
    """Write the exact scalar excitation history consumed by a local source.

    The time offset refers to the physical Yee time of sample zero. Source
    arrays contain one extra value for update look-ahead; only the values
    consumed by the model's ``iterations`` updates are persisted.
    """

    source_type = type(source).__name__
    iterations = int(grid.iterations)
    dt = float(grid.dt)
    excitation = group.create_group("excitation")
    excitation.attrs["SchemaVersion"] = SOURCE_EXCITATION_SCHEMA_VERSION
    excitation.attrs["SampleInterval"] = dt
    excitation.attrs["SourceType"] = source_type
    excitation.attrs["SourceStartTime"] = float(source.start)
    excitation.attrs["SourceStopTime"] = float(source.stop)
    excitation.attrs["Polarisation"] = str(source.polarisation or "")
    for key, value in _waveform_metadata(grid, source.waveformID).items():
        excitation.attrs[key] = value

    if source_type == "VoltageSource":
        if source.resistance == 0:
            samples = source.waveformvalues_wholedt[:iterations]
            time_offset = dt
            evaluation_time_offset = 0.0
            quantity = "imposed_gap_voltage"
            lattice = "electric"
        else:
            samples = source.waveformvalues_halfdt[:iterations]
            time_offset = 0.5 * dt
            evaluation_time_offset = 0.5 * dt
            quantity = "generator_voltage"
            lattice = "electric_half_step"
        _create_sample_dataset(excitation, "samples", samples)
        excitation.attrs["TimeSampleOffset"] = time_offset
        excitation.attrs["WaveformEvaluationTimeOffset"] = evaluation_time_offset
        excitation.attrs["DrivingQuantity"] = quantity
        excitation.attrs["Units"] = "V"
        excitation.attrs["SpatialScale"] = 1.0
        excitation.attrs["UpdateLattice"] = lattice
        excitation.attrs["Resistance"] = float(source.resistance)
        return

    if source_type == "HertzianDipole":
        _create_sample_dataset(excitation, "samples", source.waveformvalues_halfdt[:iterations])
        excitation.attrs["TimeSampleOffset"] = 0.5 * dt
        excitation.attrs["WaveformEvaluationTimeOffset"] = 0.5 * dt
        excitation.attrs["DrivingQuantity"] = "electric_current"
        excitation.attrs["Units"] = "A"
        excitation.attrs["SpatialScale"] = float(source.dl)
        excitation.attrs["SpatialScaleQuantity"] = "dipole_length"
        excitation.attrs["SpatialScaleUnits"] = "m"
        excitation.attrs["UpdateLattice"] = "electric_half_step"
        return

    if source_type == "MagneticDipole":
        _create_sample_dataset(excitation, "samples", source.waveformvalues_wholedt[:iterations])
        excitation.attrs["TimeSampleOffset"] = 0.0
        excitation.attrs["WaveformEvaluationTimeOffset"] = 0.0
        excitation.attrs["DrivingQuantity"] = "magnetic_dipole_waveform"
        excitation.attrs["Units"] = ""
        excitation.attrs["SpatialScale"] = 1.0
        excitation.attrs["UpdateLattice"] = "magnetic_whole_step"
        return

    if source_type == "RationalNetworkTerminal":
        _create_sample_dataset(excitation, "samples", source.waveform_half[:iterations])
        excitation.attrs["TimeSampleOffset"] = 0.5 * dt
        excitation.attrs["WaveformEvaluationTimeOffset"] = 0.5 * dt
        excitation.attrs["DrivingQuantity"] = "generator_voltage"
        excitation.attrs["Units"] = "V"
        excitation.attrs["SpatialScale"] = 1.0
        excitation.attrs["UpdateLattice"] = "terminal_half_step"
        return

    # Keep the schema extensible without silently claiming a time convention
    # for a source family that has not been mapped explicitly.
    excitation.attrs["Available"] = False


def _write_dual_lattice_source_excitation(group, source, grid, source_type):
    """Write whole- and half-step histories for a transmission-line source."""

    iterations = int(grid.iterations)
    excitation = group.create_group("excitation")
    excitation.attrs["SchemaVersion"] = SOURCE_EXCITATION_SCHEMA_VERSION
    excitation.attrs["SampleInterval"] = float(grid.dt)
    excitation.attrs["SourceType"] = source_type
    excitation.attrs["SourceStartTime"] = float(source.start)
    excitation.attrs["SourceStopTime"] = float(source.stop)
    excitation.attrs["Polarisation"] = str(source.polarisation or "")
    excitation.attrs["DrivingQuantity"] = "transmission_line_injector_waveform"
    excitation.attrs["Units"] = "V"
    excitation.attrs["UpdateLattice"] = "dual"
    excitation.attrs["TimeSampleOffset"] = 0.0
    excitation.attrs["WaveformEvaluationTimeOffset"] = 0.0
    excitation.attrs["SpatialScale"] = 1.0
    for key, value in _waveform_metadata(grid, source.waveformID).items():
        excitation.attrs[key] = value
    whole = _create_sample_dataset(
        excitation, "samples_whole", source.waveformvalues_wholedt[:iterations]
    )
    half = _create_sample_dataset(
        excitation, "samples_half", source.waveformvalues_halfdt[:iterations]
    )
    whole.attrs["TimeSampleOffset"] = 0.0
    half.attrs["TimeSampleOffset"] = 0.5 * float(grid.dt)
    # The whole-step generator voltage is the scalar reference waveform for
    # post-processing. It is a hard link, so no source samples are duplicated.
    excitation["samples"] = whole


def _write_frill_source_excitation(group, source, grid):
    """Write the generator waveform consumed by a magnetic-frill source."""

    iterations = int(grid.iterations)
    excitation = group.create_group("excitation")
    excitation.attrs["SchemaVersion"] = SOURCE_EXCITATION_SCHEMA_VERSION
    excitation.attrs["SampleInterval"] = float(grid.dt)
    excitation.attrs["TimeSampleOffset"] = 0.0
    excitation.attrs["WaveformEvaluationTimeOffset"] = 0.0
    excitation.attrs["SourceType"] = type(source).__name__
    excitation.attrs["SourceStartTime"] = float(source.start)
    excitation.attrs["SourceStopTime"] = float(source.stop)
    excitation.attrs["Polarisation"] = str(source.polarisation or "")
    excitation.attrs["DrivingQuantity"] = "generator_voltage"
    excitation.attrs["Units"] = "V"
    excitation.attrs["SpatialScale"] = 1.0
    excitation.attrs["UpdateLattice"] = "magnetic_whole_step"
    for key, value in _waveform_metadata(grid, source.waveformID).items():
        excitation.attrs[key] = value
    _create_sample_dataset(excitation, "samples", source.waveformvalues_wholedt[:iterations])


def _receiver_time_offset(component, dt):
    """Return the physical Yee-time offset for a receiver component."""

    if component.startswith("E"):
        return 0.0
    if component.startswith(("H", "I")):
        return -0.5 * float(dt)
    raise ValueError(f"unknown receiver component {component!r}")


def store_outputs(G: FDTDGrid, iteration: int):
    """Stores field component values for every receiver and transmission line.

    Args:
        G: FDTDGrid class describing a grid in a model.
    """

    # Assign iteration and fields to local variables
    Ex, Ey, Ez, Hx, Hy, Hz = G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz

    for rx in G.rxs:
        for output in rx.outputs:
            # Store electric or magnetic field components
            if "I" not in output:
                field = locals()[output]
                rx.outputs[output][iteration] = field[rx.xcoord, rx.ycoord, rx.zcoord]
            # Store current component
            else:
                func = globals()[output]
                rx.outputs[output][iteration] = func(rx.xcoord, rx.ycoord, rx.zcoord, Hx, Hy, Hz, G)

    for tl in G.transmissionlines:
        tl.Vtotal[iteration] = tl.voltage[tl.antpos]
        tl.Itotal[iteration] = tl.current[tl.antpos]


# TODO: Add type information for grid (without a circular dependency)
def write_hdf5_outputfile(outputfile: Path, title: str, model):
    """Writes an output file in HDF5 (.h5) format.

    Args:
        outputfile: string of the name of the output file.
        G: FDTDGrid class describing a grid in a model.
    """
    # Create output file and write top-level meta data, meta data for main grid,
    # and any outputs in the main grid
    with h5py.File(outputfile, "w") as f:
        f.attrs["gprMax"] = __version__
        f.attrs["Title"] = title
        f.attrs["Iterations"] = model.iterations
        f.attrs["srcsteps"] = model.srcsteps
        f.attrs["rxsteps"] = model.rxsteps
        write_hd5_data(f, model.G)

        # Write meta data and data for any subgrids
        sg_rxs = [True for sg in model.subgrids if sg.rxs]
        sg_tls = [True for sg in model.subgrids if sg.transmissionlines]
        sg_frills = [True for sg in model.subgrids if sg.magneticfrillsources]
        sg_ports = [True for sg in model.subgrids if sg.port_monitors]
        sg_eigenmode_ports = [True for sg in model.subgrids if sg.eigenmodeports]
        if sg_rxs or sg_tls or sg_frills or sg_ports or sg_eigenmode_ports:
            for sg in model.subgrids:
                grp = f.create_group(f"/subgrids/{sg.name}")
                write_hd5_data(grp, sg, is_subgrid=True)

    logger.basic("")
    logger.basic(f"Written output file: {outputfile.name}\n")


def _global_position(grid, coordx: int, coordy: int, coordz: int, is_subgrid: bool):
    """Converts a grid-local (x, y, z) index to a physical position in the
    global/main-grid coordinate frame.

    For the main grid, local index 0 coincides with the global origin so no
    translation is needed. For a subgrid, SubGridBaseGrid.local_to_global
    reverses the subgrid's boundary padding and i0/j0/k0 placement offset.
    """
    if is_subgrid:
        return tuple(grid.local_to_global((coordx, coordy, coordz)))
    return (coordx * grid.dx, coordy * grid.dy, coordz * grid.dz)


def write_hd5_data(basegrp, grid, is_subgrid=False):
    """Writes grid meta data and data to HDF5 group.

    Args:
        basegrp: dict of HDF5 group.
        grid: FDTDGrid class describing a grid in a model.
        is_subgrid: boolean for grid instance the main grid or a subgrid.
    """

    # Write meta data for grid
    basegrp.attrs["nx_ny_nz"] = (grid.nx, grid.ny, grid.nz)
    basegrp.attrs["dx_dy_dz"] = (grid.dx, grid.dy, grid.dz)
    basegrp.attrs["dt"] = grid.dt
    basegrp.attrs["SourceExcitationSchemaVersion"] = SOURCE_EXCITATION_SCHEMA_VERSION
    basegrp.attrs["ReceiverTimingSchemaVersion"] = RECEIVER_TIMING_SCHEMA_VERSION
    excited_network_terminals = [
        terminal for terminal in getattr(grid, "networkterminals", ()) if terminal.excited
    ]
    nsrc = len(
        grid.voltagesources
        + grid.hertziandipoles
        + grid.magneticdipoles
        + grid.transmissionlines
        + grid.magneticfrillsources
    ) + len(excited_network_terminals)
    basegrp.attrs["nsrc"] = nsrc
    public_rxs = [rx for rx in grid.rxs if not getattr(rx, "internal", False)]
    basegrp.attrs["nrx"] = len(public_rxs)
    basegrp.attrs["nports"] = len(getattr(grid, "port_monitors", ()))
    basegrp.attrs["neigenmodeports"] = len(getattr(grid, "eigenmodeports", ()))

    if is_subgrid:
        # Write additional meta data about subgrid
        basegrp.attrs["Iterations"] = grid.iterations
        basegrp.attrs["srcsteps"] = grid.srcsteps
        basegrp.attrs["rxsteps"] = grid.rxsteps
        basegrp.attrs["is_os_sep"] = grid.is_os_sep
        basegrp.attrs["pml_separation"] = grid.pml_separation
        basegrp.attrs["subgrid_pml_thickness"] = grid.pmls["thickness"]["x0"]
        basegrp.attrs["filter"] = grid.filter
        basegrp.attrs["ratio"] = grid.ratio
        basegrp.attrs["interpolation"] = grid.interpolation

    # Create group for sources (except transmission lines); add type and positional data attributes
    srclist = (
        grid.voltagesources
        + grid.hertziandipoles
        + grid.magneticdipoles
        + excited_network_terminals
    )
    for srcindex, src in enumerate(srclist):
        grp = basegrp.create_group(f"srcs/src{str(srcindex + 1)}")
        grp.attrs["Type"] = type(src).__name__
        grp.attrs["ID"] = str(src.ID)
        grp.attrs["GridPosition"] = np.asarray(src.coord, dtype=np.int32)
        grp.attrs["Polarisation"] = str(src.polarisation or "")
        grp.attrs["Position"] = _global_position(
            grid, src.xcoord, src.ycoord, src.zcoord, is_subgrid
        )
        _write_source_excitation(grp, src, grid)

    # Create group for transmission lines; add positional data, line resistance and
    # line discretisation attributes; write arrays for line voltages and currents
    for tlindex, tl in enumerate(grid.transmissionlines):
        grp = basegrp.create_group("tls/tl" + str(tlindex + 1))
        grp.attrs["Position"] = _global_position(grid, tl.xcoord, tl.ycoord, tl.zcoord, is_subgrid)
        grp.attrs["Resistance"] = tl.resistance
        grp.attrs["dl"] = tl.dl
        grp.attrs["ID"] = str(tl.ID)
        grp.attrs["GridPosition"] = np.asarray(tl.coord, dtype=np.int32)
        # Save incident voltage and current
        grp["Vinc"] = tl.Vinc
        grp["Iinc"] = tl.Iinc
        # Save total voltage and current
        basegrp["tls/tl" + str(tlindex + 1) + "/Vtotal"] = tl.Vtotal
        basegrp["tls/tl" + str(tlindex + 1) + "/Itotal"] = tl.Itotal
        _write_dual_lattice_source_excitation(grp, tl, grid, type(tl).__name__)
        port_output = getattr(tl, "port_output", None)
        if port_output is not None:
            port_output.write_hdf5(grp)

    # Create group for magnetic frill sources; add positional data, Z0, and
    # resolved symmetry-plane adjacency attributes; write arrays for the
    # incident/total voltage and total current histories. Unlike
    # transmission lines, Vtotal/Itot are written directly by
    # MagneticFrillSource.update_magnetic() every iteration, so no separate
    # store_outputs() copy step is needed here.
    for frillindex, frill in enumerate(grid.magneticfrillsources):
        grp = basegrp.create_group("frills/frill" + str(frillindex + 1))
        grp.attrs["Position"] = _global_position(
            grid, frill.xcoord, frill.ycoord, frill.zcoord, is_subgrid
        )
        grp.attrs["Polarisation"] = frill.polarisation
        grp.attrs["Z0"] = frill.Z0
        grp.attrs["InnerConductorRadius"] = frill.inner_radius
        grp.attrs["CurrentTimeApproximation"] = "average"
        grp.attrs["FeedSelfAdmittance"] = frill._G_coeff
        grp.attrs["ID"] = str(frill.ID)
        grp.attrs["GridPosition"] = np.asarray(frill.coord, dtype=np.int32)
        # The two faces transverse to Polarisation, in the fixed order used
        # throughout MagneticFrillSource.finalise_setup()/update_magnetic()
        # (see that method's docstring): x -> (z0, y0); y -> (z0, x0);
        # z -> (x0, y0).
        mirror_faces = {"x": ("z0", "y0"), "y": ("z0", "x0"), "z": ("x0", "y0")}
        face1, face2 = mirror_faces[frill.polarisation]
        grp.attrs["Mirror1Face"] = face1
        grp.attrs["Mirror2Face"] = face2
        grp.attrs["Mirror1"] = bool(frill._mirror1)
        grp.attrs["Mirror2"] = bool(frill._mirror2)
        # Save incident and total voltage, and total current
        grp["Vinc"] = frill.Vinc
        grp["Vtotal"] = frill.Vtotal
        grp["Itot"] = frill.Itot
        _write_frill_source_excitation(grp, frill, grid)
        port_output = getattr(frill, "port_output", None)
        if port_output is not None:
            port_output.write_hdf5(grp)

    # Ensure public receiver output order is consistent without mutating the
    # solver's receiver order. Device transfer maps receiver pages by this
    # original order and internal port monitors are intentionally omitted from
    # the public /rxs namespace.
    public_rxs.sort(key=lambda rx: rx.ID)

    # Create group, add positional data and write field component arrays for receivers
    for rxindex, rx in enumerate(public_rxs):
        grp = basegrp.create_group("rxs/rx" + str(rxindex + 1))
        if rx.ID:
            grp.attrs["Name"] = rx.ID
        grp.attrs["Position"] = _global_position(grid, rx.xcoord, rx.ycoord, rx.zcoord, is_subgrid)
        grp.attrs["GridPosition"] = np.asarray(rx.coord, dtype=np.int32)

        for output in rx.outputs:
            dataset = grp.create_dataset(output, data=rx.outputs[output])
            dataset.attrs["SampleInterval"] = float(grid.dt)
            dataset.attrs["TimeSampleOffset"] = _receiver_time_offset(output, grid.dt)
            dataset.attrs["Quantity"] = output

    # Managed reusable outputs own their grouped schema. Standalone monitors
    # may write themselves when they are not owned by a grouped writer.
    for monitor in getattr(grid, "ntff_monitors", ()):
        write_ntff = getattr(monitor, "write_hdf5", None)
        if write_ntff is not None and not getattr(monitor, "managed_output", False):
            write_ntff(basegrp)
    for writer in getattr(grid, "ntff_output_writers", ()):
        writer.write_hdf5(basegrp)

    for port in getattr(grid, "port_monitors", ()):
        port.write_hdf5(basegrp)

    for port in getattr(grid, "eigenmodeports", ()):
        port.write_hdf5(basegrp)


def Ix(x, y, z, Hx, Hy, Hz, G):
    """Calculates the x-component of current at a grid position.

    Args:
        x, y, z: floats for coordinates of position in grid.
        Hx, Hy, Hz: numpy array of magnetic field values.
        G: FDTDGrid class describing a grid in a model.
    """

    if y == 0 or z == 0:
        Ix = 0
    else:
        Ix = G.dy * (Hy[x, y, z - 1] - Hy[x, y, z]) + G.dz * (Hz[x, y, z] - Hz[x, y - 1, z])

    return Ix


def Iy(x, y, z, Hx, Hy, Hz, G):
    """Calculates the y-component of current at a grid position.

    Args:
        x, y, z: floats for coordinates of position in grid.
        Hx, Hy, Hz: numpy array of magnetic field values.
        G: FDTDGrid class describing a grid in a model.
    """

    if x == 0 or z == 0:
        Iy = 0
    else:
        Iy = G.dx * (Hx[x, y, z] - Hx[x, y, z - 1]) + G.dz * (Hz[x - 1, y, z] - Hz[x, y, z])

    return Iy


def Iz(x, y, z, Hx, Hy, Hz, G):
    """Calculates the z-component of current at a grid position.

    Args:
        x, y, z: floats for coordinates of position in grid.
        Hx, Hy, Hz: numpy array of magnetic field values.
        G: FDTDGrid class describing a grid in a model.
    """

    if x == 0 or y == 0:
        Iz = 0
    else:
        Iz = G.dx * (Hx[x, y - 1, z] - Hx[x, y, z]) + G.dy * (Hy[x, y, z] - Hy[x - 1, y, z])

    return Iz
