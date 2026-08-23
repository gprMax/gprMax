"""Validate spatial-average SAR against the IEC/IEEE 62704-1 SAR Star.

The standard's supplemental archive is intentionally not vendored. Download
``62704-1_supplemental_files.zip`` from the IEC supporting documents and run::

    python -m testing.validation.validate_sar_star \
        /path/to/62704-1_supplemental_files.zip

Only the uniformly sampled 1 g and 10 g reference reports are supported. The
production gprMax averaging routine is compared with the official voxel flags,
averaging masses, volumes, orientations, and spatial-average SAR values.
The complete 281-cubed cases are long-running manual release validations.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np

from gprMax.sar_averaging import spatial_average_sar

REFERENCE_NAMES = {
    0.001: "sar_star_uniform_ref_01g_2016_03_18_V3.txt",
    0.01: "sar_star_uniform_ref_10g_2016_03_18_V3.txt",
}
SPACING = (0.001, 0.001, 0.001)
DENSITIES = (2000.0, 1100.0)  # core, outer layer [kg/m3]
MAXIMUM_SAR_RELATIVE_DIFFERENCE = 2e-6


def _extract_reference(archive: Path, name: str, directory: Path) -> Path:
    """Extract one named reference report from the supplemental archive."""

    with zipfile.ZipFile(archive) as source:
        member = next((item for item in source.namelist() if item.endswith(name)), None)
        if member is None:
            raise FileNotFoundError(f"{name!r} is not present in {archive}")
        destination = directory / name
        with source.open(member) as input_file, destination.open("wb") as output_file:
            while chunk := input_file.read(8 * 1024 * 1024):
                output_file.write(chunk)
    return destination


def _grid_shape(reference: Path) -> tuple[int, int, int]:
    """Read the three grid-subdivision vectors from the report header."""

    coordinates = []
    with reference.open() as source:
        for line_number, line in enumerate(source, start=1):
            if 4 <= line_number <= 6:
                coordinates.append(
                    np.fromstring(line.removeprefix("%").strip().rstrip(","), sep=",")
                )
            if line_number == 6:
                break
    if len(coordinates) != 3 or any(axis.size == 0 for axis in coordinates):
        raise ValueError(f"Could not read the grid vectors from {reference}")
    return tuple(int(axis.size) for axis in coordinates)


def _load_rows(reference: Path):
    """Load and consolidate the report's occasional duplicate directions."""

    rows = np.loadtxt(reference, comments="%", dtype=np.float64)
    indices = rows[:, :3].astype(np.int32)
    starts = np.flatnonzero(np.r_[True, np.any(indices[1:] != indices[:-1], axis=1)])
    groups = np.cumsum(np.r_[True, np.any(indices[1:] != indices[:-1], axis=1)]) - 1
    ends = np.r_[starts[1:], rows.shape[0]]
    maximum_sar = np.maximum.reduceat(rows[:, 8], starts)
    candidate = rows[:, 8] == maximum_sar[groups]
    selected = np.full(starts.size, rows.shape[0], dtype=np.int64)
    locations = np.flatnonzero(candidate)
    np.minimum.at(selected, groups[candidate], locations)
    consolidated = rows[selected].copy()
    ambiguous = [
        (group, rows[start:end, 4:7].copy())
        for group, (start, end) in enumerate(zip(starts, ends))
        if end - start > 1
    ]
    return consolidated, ambiguous, int(rows.shape[0] - selected.size)


def _density_and_local_sar(shape, rows):
    """Reconstruct the two-material SAR Star volume defined by the standard."""

    density = np.full(shape, np.nan, dtype=np.float64)
    local_sar = np.full(shape, np.nan, dtype=np.float64)
    indices = rows[:, :3].astype(np.int32)
    cells = tuple(indices[:, axis] for axis in range(3))
    density[cells] = DENSITIES[1]
    local_sar[cells] = rows[:, 7]

    # Cell centres used by the standard's uniformly sampled reference model.
    axes = [(-0.14 + 0.0005) + np.arange(size) * 0.001 for size in shape]
    x, y, z = np.meshgrid(*axes, indexing="ij", sparse=True)
    embedded_cube = (np.abs(x) <= 0.007) & (np.abs(y) <= 0.007) & (np.abs(z) <= 0.007)
    enclosing_cube = (
        (np.abs(x) <= 0.04)
        & (np.abs(y) <= 0.04)
        & (np.abs(z) <= 0.04)
        & ~((np.abs(x) <= 0.012) & (np.abs(y) <= 0.012) & (np.abs(z) <= 0.012))
    )
    peg_x = (np.abs(x) >= 0.04) & (np.abs(x) <= 0.085) & (y**2 + z**2 <= 0.01**2)
    peg_y = (np.abs(y) >= 0.04) & (np.abs(y) <= 0.085) & (x**2 + z**2 <= 0.01**2)
    peg_z = (np.abs(z) >= 0.04) & (np.abs(z) <= 0.085) & (x**2 + y**2 <= 0.01**2)
    core = (embedded_cube | enclosing_cube | peg_x | peg_y | peg_z) & np.isfinite(density)
    density[core] = DENSITIES[0]
    return density, local_sar


def _relative_error(actual, expected):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    nonzero = expected != 0
    error = np.zeros_like(expected, dtype=np.float64)
    error[nonzero] = np.abs(actual[nonzero] / expected[nonzero] - 1)
    error[~nonzero] = np.abs(actual[~nonzero] - expected[~nonzero])
    return error


def validate_reference(reference: Path, target_mass: float):
    """Run one official uniform-grid SAR Star validation."""

    shape = _grid_shape(reference)
    rows, ambiguous, duplicate_count = _load_rows(reference)
    density, local_sar = _density_and_local_sar(shape, rows)
    indices = rows[:, :3].astype(np.int32)
    cells = tuple(indices[:, axis] for axis in range(3))

    start = perf_counter()
    result = spatial_average_sar(density, local_sar, SPACING, target_mass)
    runtime = perf_counter() - start

    actual_status = result.status[cells]
    expected_status = np.abs(rows[:, 3]).astype(np.uint8)
    status_mismatches = int(np.count_nonzero(actual_status != expected_status))

    # The official file carries one row per acceptable Step-2 orientation for
    # ambiguous cells. Match mass/volume to the direction selected by gprMax,
    # while the consolidated SAR remains the largest reference value.
    actual_orientation = result.orientation[cells]
    ambiguous_mask = np.zeros(rows.shape[0], dtype=bool)
    for group, alternatives in ambiguous:
        ambiguous_mask[group] = True
        matched = alternatives[alternatives[:, 2] == actual_orientation[group]]
        if matched.size:
            rows[group, 4:7] = matched[0]

    defined_cube = expected_status != 2
    mass_error = _relative_error(
        result.averaging_mass[cells][defined_cube] * 1000, rows[defined_cube, 4]
    )
    volume_error = _relative_error(
        result.averaging_volume[cells][defined_cube] * 1e9,
        rows[defined_cube, 5],
    )
    sar_error = _relative_error(result.sar[cells], rows[:, 8])
    sar_signed_deviation = result.sar[cells] / rows[:, 8] - 1
    official_sar_failures = int(np.count_nonzero(sar_signed_deviation[ambiguous_mask] < -0.2))

    # For ambiguous Step-2 cells the reference contains several acceptable
    # directions; duplicate consolidation selected the maximum reference SAR.
    # Orientation is therefore reported for unambiguous rows only.
    unambiguous = rows[:, 3] >= 0
    orientation_mismatches = int(
        np.count_nonzero(actual_orientation[unambiguous] != rows[unambiguous, 6])
    )
    background_valid = int(np.count_nonzero(result.status[~np.isfinite(density)] != 0))

    report = {
        "target_mass_g": 1000 * target_mass,
        "grid_shape": shape,
        "tissue_voxels": int(rows.shape[0]),
        "consolidated_duplicate_rows": duplicate_count,
        "runtime_seconds": runtime,
        "status_mismatches": status_mismatches,
        "background_status_mismatches": background_valid,
        "orientation_mismatches_unambiguous": orientation_mismatches,
        "official_ambiguous_sar_failures": official_sar_failures,
        "maximum_mass_relative_error": float(np.nanmax(mass_error)),
        "maximum_volume_relative_error": float(np.nanmax(volume_error)),
        "maximum_sar_relative_error": float(np.nanmax(sar_error)),
        "mean_sar_relative_error": float(np.nanmean(sar_error)),
        "official_tolerances": {
            "status_mismatches": 0,
            "mass_relative": 2e-6,
            "volume_relative": 2e-6,
            "ambiguous_sar_lower_relative": 0.2,
        },
        "additional_validation_tolerances": {
            "maximum_sar_relative": MAXIMUM_SAR_RELATIVE_DIFFERENCE,
        },
    }
    if status_mismatches or background_valid or orientation_mismatches:
        raise AssertionError(f"SAR Star status validation failed: {report}")
    if np.nanmax(mass_error) > 2e-6 or np.nanmax(volume_error) > 2e-6:
        raise AssertionError(f"SAR Star cube validation failed: {report}")
    if official_sar_failures:
        raise AssertionError(f"SAR Star spatial-average validation failed: {report}")
    if np.nanmax(sar_error) > MAXIMUM_SAR_RELATIVE_DIFFERENCE:
        raise AssertionError(f"SAR Star SAR-value validation failed: {report}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("sar_star_validation.json"))
    parser.add_argument(
        "--masses",
        choices=("1g", "10g", "both"),
        default="both",
        help="run only one averaging mass while developing, or both (default)",
    )
    args = parser.parse_args()
    report = {}
    requested = {
        "1g": (0.001,),
        "10g": (0.01,),
        "both": tuple(REFERENCE_NAMES),
    }[args.masses]
    with tempfile.TemporaryDirectory(prefix="gprmax-sar-star-") as directory:
        directory = Path(directory)
        for target_mass in requested:
            name = REFERENCE_NAMES[target_mass]
            reference = _extract_reference(args.archive, name, directory)
            report[f"{1000 * target_mass:g}g"] = validate_reference(reference, target_mass)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
