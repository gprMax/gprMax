# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
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

import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np

from gprMax.utilities.utilities import natural_keys


TIME_DOMAIN_RECEIVER_OUTPUTS = frozenset(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Ix", "Iy", "Iz"))
TIME_DOMAIN_VOLTAGE_OUTPUTS = frozenset(("Vtotal",))
VOLTAGE_TRACE_PARENTS = ("ports", "tls", "frills")
MERGED_OUTPUT_SCHEMA_VERSION = 1


def _grid_group(output, grid_path="/"):
    grid_path = str(grid_path).strip("/")
    return output[grid_path] if grid_path else output


def get_output_data(filename, rxnumber, rxcomponent, grid_path="/", trace_group=None):
    """Gets B-scan output data from a model.

    Args:
        filename: string of tilename (including path) of output file.
        rxnumber: int of receiver output number.
        rxcomponent: string of receiver output field/current component.
        grid_path: optional main-grid or subgrid HDF5 path.
        trace_group: optional voltage-trace group, e.g. ``ports/receive``.

    Returns:
        outputdata: array of A-scans, i.e. B-scan data.
        dt: float of temporal resolution of the model.
    """

    # Open output file and read some attributes
    with h5py.File(filename, "r") as f:
        grid = _grid_group(f, grid_path)
        dt = grid.attrs["dt"]
        if trace_group is None:
            nrx = int(grid.attrs["nrx"])
            if nrx == 0:
                raise ValueError(f"No receivers found in {filename}")
            if rxnumber < 1 or rxnumber > nrx:
                raise ValueError(f"Receiver {rxnumber} is outside the valid range 1-{nrx}")
            path = f"rxs/rx{rxnumber}"
            allowed = TIME_DOMAIN_RECEIVER_OUTPUTS
            description = f"receiver {rxnumber}"
        else:
            path = str(trace_group).strip("/")
            allowed = TIME_DOMAIN_VOLTAGE_OUTPUTS
            description = f"trace group {path}"
            if path not in grid or not isinstance(grid[path], h5py.Group):
                raise ValueError(f"Trace group {path!r} was not found in {filename}")

        if rxcomponent not in allowed:
            raise ValueError(f"{rxcomponent!r} is not a supported real time-domain output for {description}")
        availableoutputs = list(grid[path].keys())

        # Check if requested output is in file
        if rxcomponent not in availableoutputs:
            raise ValueError(
                f"{rxcomponent} output requested to plot, but the "
                + f"available output for {description} is "
                + f"{', '.join(availableoutputs)}"
            )

        dataset = grid[f"{path}/{rxcomponent}"]
        outputdata = np.asarray(dataset)
        if outputdata.ndim not in (1, 2) or np.iscomplexobj(outputdata):
            raise ValueError(f"{path}/{rxcomponent} is not a real time-domain A-scan or B-scan")

    return outputdata, dt


def _default_merged_filename(outputfiles):
    paths = [Path(filename).resolve() for filename in outputfiles]
    parent = Path(os.path.commonpath([str(path.parent) for path in paths]))
    prefix = os.path.commonprefix([path.stem for path in paths]).rstrip("_- ")
    return parent / f"{prefix or 'output'}_merged.h5"


def _output_grid_paths(output):
    paths = [""]
    if "subgrids" in output:
        paths.extend(f"subgrids/{name}" for name in output["subgrids"])
    return paths


def _values_equal(first, second):
    return np.array_equal(np.asarray(first), np.asarray(second))


def _validate_position(group, attribute, description):
    """Validate optional position metadata written by current gprMax files."""

    if attribute not in group.attrs:
        return
    position = np.asarray(group.attrs[attribute], dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"Invalid {attribute} metadata for {description}")


def _validate_receiver_dataset(reference, candidate, path, filename, iterations, dt):
    """Validate that an HDF5 object is a compatible time-domain A-scan."""

    output = path.rsplit("/", maxsplit=1)[-1]
    if output not in TIME_DOMAIN_RECEIVER_OUTPUTS:
        raise ValueError(
            f"{path} is not a supported time-domain receiver component; "
            "frequency-domain outputs cannot be merged into a B-scan"
        )
    if not isinstance(reference, h5py.Dataset) or not isinstance(candidate, h5py.Dataset):
        raise ValueError(f"{path} in {filename} is not a receiver dataset")

    expected = (iterations,)
    if candidate.shape != expected:
        qualifier = "original one-dimensional A-scan" if candidate.ndim != 1 else str(expected)
        raise ValueError(f"{path} in {filename} has shape {candidate.shape}; expected {qualifier}")
    if not np.issubdtype(candidate.dtype, np.number) or np.issubdtype(candidate.dtype, np.complexfloating):
        raise ValueError(
            f"{path} in {filename} has dtype {candidate.dtype}; "
            "B-scan traces must contain real numeric time-domain samples"
        )
    if candidate.dtype != reference.dtype:
        raise ValueError(f"Inconsistent dtype for {path} in {filename}: " f"{candidate.dtype} != {reference.dtype}")

    if set(candidate.attrs) != set(reference.attrs):
        raise ValueError(f"Receiver metadata differs for {path} in {filename}")
    for name, value in reference.attrs.items():
        if not _values_equal(value, candidate.attrs[name]):
            raise ValueError(f"Receiver metadata {name} differs for {path} in {filename}")

    if "SampleInterval" in candidate.attrs:
        sample_interval = float(candidate.attrs["SampleInterval"])
        if not np.isfinite(sample_interval) or not np.isclose(sample_interval, dt, rtol=1e-12, atol=0.0):
            raise ValueError(f"Invalid SampleInterval metadata for {path} in {filename}")
    if "TimeSampleOffset" in candidate.attrs and not np.isfinite(float(candidate.attrs["TimeSampleOffset"])):
        raise ValueError(f"Invalid TimeSampleOffset metadata for {path} in {filename}")
    if "Quantity" in candidate.attrs:
        quantity = candidate.attrs["Quantity"]
        if isinstance(quantity, bytes):
            quantity = quantity.decode(errors="replace")
        if str(quantity) != output:
            raise ValueError(f"Invalid Quantity metadata for {path} in {filename}")


def _voltage_trace_paths(grid):
    """Return groups containing authoritative time-domain terminal voltage."""

    paths = []
    for parent_name in VOLTAGE_TRACE_PARENTS:
        if parent_name not in grid:
            continue
        parent = grid[parent_name]
        for name in sorted(parent, key=natural_keys):
            group = parent[name]
            if isinstance(group, h5py.Group) and "Vtotal" in group:
                paths.append(f"{parent_name}/{name}")
    return paths


def _voltage_time_offset(group, parent_name):
    attribute = {
        "ports": "TimeSampleOffset",
        "tls": "TimeVoltageOffset",
        "frills": "TimeOffset",
    }[parent_name]
    value = float(group.attrs.get(attribute, 0.0))
    if not np.isfinite(value):
        raise ValueError(f"Invalid {attribute} metadata for {group.name}")
    return value


def _voltage_sample_count(group, parent_name, iterations, filename):
    """Return the physical voltage-history length for one source type."""

    dataset = group["Vtotal"]
    path = dataset.name
    if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1:
        raise ValueError(f"{path} in {filename} is not a one-dimensional voltage trace")
    if not np.issubdtype(dataset.dtype, np.number) or np.issubdtype(dataset.dtype, np.complexfloating):
        raise ValueError(f"{path} in {filename} is not a real numeric voltage trace")

    if parent_name == "ports":
        valid_lengths = {iterations, max(0, iterations - 1)}
        if dataset.size not in valid_lengths:
            raise ValueError(
                f"{path} in {filename} has {dataset.size} samples; expected " f"{iterations} or {iterations - 1}"
            )
        sample_count = dataset.size
        if "time" in group and group["time"].shape != (sample_count,):
            raise ValueError(f"{group.name}/time in {filename} is inconsistent with Vtotal")
    elif parent_name == "tls":
        if dataset.size != iterations:
            raise ValueError(f"{path} in {filename} has {dataset.size} samples; expected {iterations}")
        sample_count = iterations
        if "time_voltage" in group and group["time_voltage"].shape != (sample_count,):
            raise ValueError(f"{group.name}/time_voltage in {filename} is inconsistent with Vtotal")
    else:
        if dataset.size not in (iterations, iterations + 1):
            raise ValueError(
                f"{path} in {filename} has {dataset.size} samples; expected " f"{iterations} or {iterations + 1}"
            )
        # A magnetic-frill source stores one extra endpoint in its raw arrays;
        # the port transform and physical output use the first Iterations values.
        sample_count = iterations
        if "time" in group and group["time"].shape != (sample_count,):
            raise ValueError(f"{group.name}/time in {filename} is inconsistent with Vtotal")
    return sample_count


def _validate_voltage_time_axis(group, parent_name, sample_count, dt, filename):
    time_name = {
        "ports": "time",
        "tls": "time_voltage",
        "frills": "time",
    }[parent_name]
    if time_name not in group:
        return
    dataset = group[time_name]
    if not isinstance(dataset, h5py.Dataset) or dataset.shape != (sample_count,):
        raise ValueError(f"{group.name}/{time_name} in {filename} has an invalid shape")
    values = np.asarray(dataset)
    if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"{group.name}/{time_name} in {filename} is not a real time axis")
    expected = _voltage_time_offset(group, parent_name) + np.arange(sample_count) * dt
    if not np.allclose(values, expected, rtol=1e-6, atol=1e-30):
        raise ValueError(
            f"{group.name}/{time_name} in {filename} is inconsistent with the " "grid time step and voltage offset"
        )


def _validate_voltage_traces(reference, candidate, grid_path, filename):
    reference_paths = set(_voltage_trace_paths(reference))
    candidate_paths = set(_voltage_trace_paths(candidate))
    if candidate_paths != reference_paths:
        raise ValueError(f"Time-domain voltage outputs differ in grid {grid_path or '/'} of {filename}")

    iterations = int(reference.attrs["Iterations"])
    dt = float(reference.attrs["dt"])
    for path in reference_paths:
        parent_name = path.split("/", maxsplit=1)[0]
        reference_count = _voltage_sample_count(reference[path], parent_name, iterations, reference.file.filename)
        candidate_count = _voltage_sample_count(candidate[path], parent_name, iterations, filename)
        if candidate_count != reference_count:
            raise ValueError(f"Inconsistent voltage trace length for {path} in {filename}")
        _validate_voltage_time_axis(
            reference[path],
            parent_name,
            reference_count,
            dt,
            reference.file.filename,
        )
        _validate_voltage_time_axis(candidate[path], parent_name, candidate_count, dt, filename)
        if candidate[f"{path}/Vtotal"].dtype != reference[f"{path}/Vtotal"].dtype:
            raise ValueError(f"Inconsistent dtype for {path}/Vtotal in {filename}")
        if not np.isclose(
            _voltage_time_offset(candidate[path], parent_name),
            _voltage_time_offset(reference[path], parent_name),
            rtol=1e-12,
            atol=1e-30,
        ):
            raise ValueError(f"Inconsistent voltage time offset for {path} in {filename}")
        for attribute in (
            "Name",
            "ID",
            "StudyID",
            "SourceType",
            "PortMode",
            "Polarisation",
            "ReferenceImpedance",
        ):
            if (attribute in reference[path].attrs) != (attribute in candidate[path].attrs):
                raise ValueError(f"Voltage-port metadata {attribute} differs for {path}")
            if attribute in reference[path].attrs and not _values_equal(
                reference[path].attrs[attribute], candidate[path].attrs[attribute]
            ):
                raise ValueError(f"Voltage-port metadata {attribute} differs for {path}")


def _validate_grid(reference, candidate, grid_path, filename):
    for attribute in ("Iterations", "nrx", "dt"):
        if attribute not in reference.attrs or attribute not in candidate.attrs:
            raise ValueError(f"Missing {attribute} metadata for grid {grid_path or '/'}")
        if not _values_equal(reference.attrs[attribute], candidate.attrs[attribute]):
            raise ValueError(f"Inconsistent {attribute} for grid {grid_path or '/'} in {filename}")

    nrx = int(reference.attrs["nrx"])
    for rx in range(1, nrx + 1):
        rxpath = f"rxs/rx{rx}"
        if rxpath not in reference:
            raise ValueError(f"Missing {rxpath} in reference grid {grid_path or '/'}")
        if rxpath not in candidate:
            raise ValueError(f"Missing {rxpath} in grid {grid_path or '/'} of {filename}")
        if set(reference[rxpath].keys()) != set(candidate[rxpath].keys()):
            raise ValueError(f"Receiver outputs differ for {rxpath} in grid {grid_path or '/'} of {filename}")
        for attribute in ("Name", "StudyID"):
            if (attribute in reference[rxpath].attrs) != (attribute in candidate[rxpath].attrs):
                raise ValueError(
                    f"Receiver metadata {attribute} differs for {rxpath} " f"in grid {grid_path or '/'} of {filename}"
                )
            if attribute in reference[rxpath].attrs and not _values_equal(
                reference[rxpath].attrs[attribute], candidate[rxpath].attrs[attribute]
            ):
                raise ValueError(
                    f"Receiver metadata {attribute} differs for {rxpath} " f"in grid {grid_path or '/'} of {filename}"
                )
        _validate_position(
            candidate[rxpath],
            "Position",
            f"{grid_path or '/'}/{rxpath} in {filename}",
        )
        _validate_position(
            candidate[rxpath],
            "GridPosition",
            f"{grid_path or '/'}/{rxpath} in {filename}",
        )
        for output, dataset in reference[rxpath].items():
            _validate_receiver_dataset(
                dataset,
                candidate[f"{rxpath}/{output}"],
                f"{grid_path or '/'}/{rxpath}/{output}",
                filename,
                int(reference.attrs["Iterations"]),
                float(reference.attrs["dt"]),
            )

    _validate_voltage_traces(reference, candidate, grid_path, filename)

    reference_paths = set(_position_group_paths(reference))
    candidate_paths = set(_position_group_paths(candidate))
    if candidate_paths != reference_paths:
        raise ValueError(
            f"Position-bearing source/receiver structure differs in " f"grid {grid_path or '/'} of {filename}"
        )
    for path in reference_paths:
        _validate_position(candidate[path], "Position", f"{grid_path or '/'}/{path} in {filename}")
        if ("GridPosition" in reference[path].attrs) != ("GridPosition" in candidate[path].attrs):
            raise ValueError(f"GridPosition metadata differs for {grid_path or '/'}/{path} in {filename}")
        _validate_position(
            candidate[path],
            "GridPosition",
            f"{grid_path or '/'}/{path} in {filename}",
        )


def _position_group_paths(grid):
    """Return position-bearing source and receiver groups in stable order."""

    paths = []
    for parent_name in ("srcs", "ports", "tls", "frills", "rxs"):
        if parent_name not in grid:
            continue
        parent = grid[parent_name]
        for name in sorted(parent, key=natural_keys):
            group = parent[name]
            if isinstance(group, h5py.Group) and "Position" in group.attrs:
                paths.append(f"{parent_name}/{name}")
    return paths


def _create_trace_metadata(destination_grid, reference_grid, outputfiles):
    """Create per-trace acquisition metadata alongside a merged B-scan."""

    metadata = destination_grid.create_group("trace_metadata")
    metadata.attrs["SchemaVersion"] = MERGED_OUTPUT_SCHEMA_VERSION
    metadata.attrs["NumberOfTraces"] = len(outputfiles)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    metadata.create_dataset(
        "InputFiles",
        data=np.asarray([path.name for path in outputfiles], dtype=object),
        dtype=string_dtype,
    )

    for path in _position_group_paths(reference_grid):
        group = metadata.require_group(path)
        group.create_dataset("Position", shape=(len(outputfiles), 3), dtype=np.float64)
        if "GridPosition" in reference_grid[path].attrs:
            group.create_dataset("GridPosition", shape=(len(outputfiles), 3), dtype=np.int64)


def _write_trace_metadata(destination_grid, source_grid, index, filename):
    """Write acquisition positions for one B-scan column."""

    metadata = destination_grid["trace_metadata"]
    expected_paths = set()
    for parent_name in ("srcs", "ports", "tls", "frills", "rxs"):
        if parent_name in metadata:
            expected_paths.update(
                f"{parent_name}/{name}"
                for name in metadata[parent_name]
                if isinstance(metadata[f"{parent_name}/{name}"], h5py.Group)
            )
    if set(_position_group_paths(source_grid)) != expected_paths:
        raise ValueError(f"Position-bearing source/receiver structure differs in {filename}")

    for path in expected_paths:
        source = source_grid[path]
        _validate_position(source, "Position", f"{path} in {filename}")
        if "Position" not in source.attrs:
            raise ValueError(f"Missing Position metadata for {path} in {filename}")
        metadata[f"{path}/Position"][index] = np.asarray(source.attrs["Position"])

        grid_position_path = f"{path}/GridPosition"
        if grid_position_path in metadata:
            if "GridPosition" not in source.attrs:
                raise ValueError(f"Missing GridPosition metadata for {path} in {filename}")
            _validate_position(source, "GridPosition", f"{path} in {filename}")
            metadata[grid_position_path][index] = np.asarray(source.attrs["GridPosition"])
        elif "GridPosition" in source.attrs:
            raise ValueError(f"Unexpected GridPosition metadata for {path} in {filename}")


def _create_voltage_outputs(destination_grid, reference_grid, outputfiles):
    """Create B-scan matrices for source-terminal total voltages."""

    iterations = int(reference_grid.attrs["Iterations"])
    dt = float(reference_grid.attrs["dt"])
    metadata_attributes = (
        "Name",
        "ID",
        "StudyID",
        "Position",
        "GridPosition",
        "SourceType",
        "PortMode",
        "Polarisation",
        "ReferenceImpedance",
        "Resistance",
        "Z0",
        "dl",
        "CellLength",
        "WaveformID",
        "NetworkModelID",
    )
    for path in _voltage_trace_paths(reference_grid):
        parent_name = path.split("/", maxsplit=1)[0]
        source_group = reference_grid[path]
        destination_group = destination_grid.require_group(path)
        for name in metadata_attributes:
            if name in source_group.attrs:
                destination_group.attrs[name] = source_group.attrs[name]
        destination_group.attrs["MergedTimeDomainOnly"] = True
        sample_count = _voltage_sample_count(source_group, parent_name, iterations, reference_grid.file.filename)
        dataset = destination_group.create_dataset(
            "Vtotal",
            shape=(sample_count, len(outputfiles)),
            dtype=source_group["Vtotal"].dtype,
        )
        dataset.attrs["SampleInterval"] = dt
        dataset.attrs["TimeSampleOffset"] = _voltage_time_offset(source_group, parent_name)
        dataset.attrs["Quantity"] = "Vtotal"
        dataset.attrs["Units"] = "V"
        dataset.attrs["SourcePath"] = path


def _write_voltage_outputs(destination_grid, source_grid, index):
    """Write one model run into every voltage B-scan matrix."""

    for path in _voltage_trace_paths(source_grid):
        destination = destination_grid[f"{path}/Vtotal"]
        destination[:, index] = source_grid[f"{path}/Vtotal"][: destination.shape[0]]


def merge_files(outputfiles, merged_outputfile=None, removefiles=False):
    """Merges traces (A-scans) from multiple output files into one new file,
        then optionally removes the series of output files.

    Args:
        outputfiles: list of output files to be merged.
        removefiles: boolean flag to remove individual output files after merge.
        merged_outputfile: string or Path object of location to save the merged
        output. If not specified a default location is used.
    """

    outputfiles = [Path(filename) for filename in outputfiles]
    if not outputfiles:
        raise ValueError("No output files were supplied for merging")
    if any(not filename.is_file() for filename in outputfiles):
        missing = [str(filename) for filename in outputfiles if not filename.is_file()]
        raise FileNotFoundError("Output file(s) not found: " + ", ".join(missing))

    merged_outputfile = (
        Path(merged_outputfile) if merged_outputfile is not None else _default_merged_filename(outputfiles)
    )
    if merged_outputfile.resolve() in {filename.resolve() for filename in outputfiles}:
        raise ValueError("Merged output file must not overwrite an input file")

    with h5py.File(outputfiles[0], "r") as reference:
        grid_paths = _output_grid_paths(reference)
        if not any(
            int(_grid_group(reference, path).attrs.get("nrx", 0)) or _voltage_trace_paths(_grid_group(reference, path))
            for path in grid_paths
        ):
            raise ValueError(
                "No receiver or terminal-voltage A-scan outputs were found; this "
                "utility merges real time-domain traces into B-scans and does not "
                "merge frequency-domain, NTFF, SAR, or study outputs"
            )

        # Validate every input before creating/truncating the destination.
        for outputfile in outputfiles:
            with h5py.File(outputfile, "r") as source:
                if _output_grid_paths(source) != grid_paths:
                    raise ValueError(f"Grid structure differs in {outputfile}")
                for grid_path in grid_paths:
                    _validate_grid(
                        _grid_group(reference, grid_path),
                        _grid_group(source, grid_path),
                        grid_path,
                        outputfile,
                    )

        with h5py.File(merged_outputfile, "w") as merged:
            for name, value in reference.attrs.items():
                merged.attrs[name] = value

            for grid_path in grid_paths:
                source_grid = _grid_group(reference, grid_path)
                destination_grid = merged.require_group(grid_path) if grid_path else merged
                for name, value in source_grid.attrs.items():
                    destination_grid.attrs[name] = value
                destination_grid.attrs["MergedOutput"] = True
                destination_grid.attrs["MergedOutputSchemaVersion"] = MERGED_OUTPUT_SCHEMA_VERSION
                destination_grid.attrs["MergedContent"] = "real_time_domain_receivers_and_terminal_voltages"
                destination_grid.attrs["ntraces"] = len(outputfiles)

                nrx = int(source_grid.attrs["nrx"])
                for rx in range(1, nrx + 1):
                    rxpath = f"rxs/rx{rx}"
                    source_rx = source_grid[rxpath]
                    destination_rx = destination_grid.require_group(rxpath)
                    for name, value in source_rx.attrs.items():
                        destination_rx.attrs[name] = value
                    for output, dataset in source_rx.items():
                        merged_dataset = destination_rx.create_dataset(
                            output,
                            shape=(dataset.shape[0], len(outputfiles)),
                            dtype=dataset.dtype,
                        )
                        for name, value in dataset.attrs.items():
                            merged_dataset.attrs[name] = value
                _create_voltage_outputs(destination_grid, source_grid, outputfiles)
                _create_trace_metadata(destination_grid, source_grid, outputfiles)

            for index, outputfile in enumerate(outputfiles):
                with h5py.File(outputfile, "r") as source:
                    for grid_path in grid_paths:
                        source_grid = _grid_group(source, grid_path)
                        destination_grid = _grid_group(merged, grid_path)
                        _write_trace_metadata(destination_grid, source_grid, index, outputfile)
                        _write_voltage_outputs(destination_grid, source_grid, index)
                        for rx in range(1, int(source_grid.attrs["nrx"]) + 1):
                            rxpath = f"rxs/rx{rx}"
                            for output in source_grid[rxpath]:
                                destination_grid[f"{rxpath}/{output}"][:, index] = source_grid[f"{rxpath}/{output}"][:]

    if removefiles:
        for outputfile in outputfiles:
            outputfile.unlink()

    return merged_outputfile


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Merges traces (A-scans) from multiple "
        + "output files into one new file, then "
        + "optionally removes the series of output files.",
        usage="python -m toolboxes.Utilities.outputfiles_merge basefilename",
    )
    parser.add_argument("basefilename", help="base name of output file series including path")
    parser.add_argument(
        "-o",
        "--output-file",
        default=None,
        type=str,
        required=False,
        help="location to save merged file",
    )
    parser.add_argument(
        "--remove-files",
        action="store_true",
        default=False,
        help="flag to remove individual output files after merge",
    )
    args = parser.parse_args()

    files = glob.glob(args.basefilename + "*.h5")
    requested_output = Path(args.output_file).resolve() if args.output_file else None
    outputfiles = [
        filename
        for filename in files
        if "_merged" not in Path(filename).stem
        and (requested_output is None or Path(filename).resolve() != requested_output)
    ]
    outputfiles.sort(key=natural_keys)
    if not outputfiles:
        parser.error(f"No output files match {args.basefilename}*.h5")
    merge_files(outputfiles, merged_outputfile=args.output_file, removefiles=args.remove_files)
