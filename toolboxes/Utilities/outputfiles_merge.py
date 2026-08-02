# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
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

import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np

from gprMax.utilities.utilities import natural_keys


def _grid_group(output, grid_path="/"):
    grid_path = str(grid_path).strip("/")
    return output[grid_path] if grid_path else output


def get_output_data(filename, rxnumber, rxcomponent, grid_path="/"):
    """Gets B-scan output data from a model.

    Args:
        filename: string of tilename (including path) of output file.
        rxnumber: int of receiver output number.
        rxcomponent: string of receiver output field/current component.

    Returns:
        outputdata: array of A-scans, i.e. B-scan data.
        dt: float of temporal resolution of the model.
    """

    # Open output file and read some attributes
    with h5py.File(filename, "r") as f:
        grid = _grid_group(f, grid_path)
        nrx = int(grid.attrs["nrx"])
        dt = grid.attrs["dt"]

        # Check there are any receivers
        if nrx == 0:
            raise ValueError(f"No receivers found in {filename}")

        if rxnumber < 1 or rxnumber > nrx:
            raise ValueError(f"Receiver {rxnumber} is outside the valid range 1-{nrx}")

        path = f"rxs/rx{rxnumber}"
        availableoutputs = list(grid[path].keys())

        # Check if requested output is in file
        if rxcomponent not in availableoutputs:
            raise ValueError(
                f"{rxcomponent} output requested to plot, but the "
                + f"available output for receiver {rxnumber} is "
                + f"{', '.join(availableoutputs)}"
            )

        outputdata = np.asarray(grid[f"{path}/{rxcomponent}"])

    return outputdata, dt


def _default_merged_filename(outputfiles):
    paths = [Path(filename).resolve() for filename in outputfiles]
    parent = Path(os.path.commonpath([str(path.parent) for path in paths]))
    prefix = os.path.commonprefix([path.stem for path in paths]).rstrip("_- ")
    return parent / f"{prefix or 'output'}_merged.h5"


def _receiver_grid_paths(output):
    paths = [""]
    if "subgrids" in output:
        paths.extend(f"subgrids/{name}" for name in output["subgrids"])
    return paths


def _values_equal(first, second):
    return np.array_equal(np.asarray(first), np.asarray(second))


def _validate_grid(reference, candidate, grid_path, filename):
    for attribute in ("Iterations", "nrx", "dt"):
        if attribute not in reference.attrs or attribute not in candidate.attrs:
            raise ValueError(f"Missing {attribute} metadata for grid {grid_path or '/'}")
        if not _values_equal(reference.attrs[attribute], candidate.attrs[attribute]):
            raise ValueError(f"Inconsistent {attribute} for grid {grid_path or '/'} in {filename}")

    nrx = int(reference.attrs["nrx"])
    for rx in range(1, nrx + 1):
        rxpath = f"rxs/rx{rx}"
        if rxpath not in candidate:
            raise ValueError(f"Missing {rxpath} in grid {grid_path or '/'} of {filename}")
        if set(reference[rxpath].keys()) != set(candidate[rxpath].keys()):
            raise ValueError(
                f"Receiver outputs differ for {rxpath} in grid {grid_path or '/'} of {filename}"
            )
        for output in reference[rxpath]:
            expected = (int(reference.attrs["Iterations"]),)
            if candidate[f"{rxpath}/{output}"].shape != expected:
                raise ValueError(
                    f"{grid_path or '/'}/{rxpath}/{output} in {filename} has shape "
                    f"{candidate[f'{rxpath}/{output}'].shape}; expected {expected}"
                )


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
        Path(merged_outputfile)
        if merged_outputfile is not None
        else _default_merged_filename(outputfiles)
    )
    if merged_outputfile.resolve() in {filename.resolve() for filename in outputfiles}:
        raise ValueError("Merged output file must not overwrite an input file")

    with h5py.File(outputfiles[0], "r") as reference:
        grid_paths = _receiver_grid_paths(reference)

        # Validate every input before creating/truncating the destination.
        for outputfile in outputfiles:
            with h5py.File(outputfile, "r") as source:
                if _receiver_grid_paths(source) != grid_paths:
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

                nrx = int(source_grid.attrs["nrx"])
                for rx in range(1, nrx + 1):
                    rxpath = f"rxs/rx{rx}"
                    source_rx = source_grid[rxpath]
                    destination_rx = destination_grid.require_group(rxpath)
                    for name, value in source_rx.attrs.items():
                        destination_rx.attrs[name] = value
                    for output, dataset in source_rx.items():
                        destination_rx.create_dataset(
                            output,
                            shape=(dataset.shape[0], len(outputfiles)),
                            dtype=dataset.dtype,
                        )

            for index, outputfile in enumerate(outputfiles):
                with h5py.File(outputfile, "r") as source:
                    for grid_path in grid_paths:
                        source_grid = _grid_group(source, grid_path)
                        destination_grid = _grid_group(merged, grid_path)
                        for rx in range(1, int(source_grid.attrs["nrx"]) + 1):
                            rxpath = f"rxs/rx{rx}"
                            for output in source_grid[rxpath]:
                                destination_grid[f"{rxpath}/{output}"][:, index] = source_grid[
                                    f"{rxpath}/{output}"
                                ][:]

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
    outputfiles = [
        filename for filename in files if "_merged" not in filename and args.output_file != filename
    ]
    outputfiles.sort(key=natural_keys)
    if not outputfiles:
        parser.error(f"No output files match {args.basefilename}*.h5")
    merge_files(outputfiles, merged_outputfile=args.output_file, removefiles=args.remove_files)
