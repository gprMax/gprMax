"""Compare gprMax 1 g/10 g averaging with the independent STASIS implementation.

The reference repository is not vendored. Clone and build
https://github.com/umbertozanovello/IEC-IEEE-62704-1-spatial-average-SAR,
then pass its directory with ``--reference``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from gprMax.sar_averaging import spatial_average_sar

MAXIMUM_RELATIVE_SAR_DIFFERENCE = 1e-4


def _load_reference(root: Path):
    module_path = root / "spatialAverageSAR.py"
    spec = importlib.util.spec_from_file_location("independent_spatial_sar", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    step1 = next(iter(root.glob("**/libAvgSARStep1.so")), None)
    step2 = next(iter(root.glob("**/libAvgSARStep2.so")), None)
    if step1 is None or step2 is None:
        raise FileNotFoundError("build libAvgSARStep1.so and libAvgSARStep2.so first")
    return module.spatialAverageSAR, step1, step2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("sar_spatial_average_validation.json"))
    args = parser.parse_args()

    reference, step1, step2 = _load_reference(args.reference)
    shape = (16, 16, 16)
    spacing = 1.7e-3
    density = np.full(shape, 987.0)
    density[8:, :, :] = 1123.0
    local_sar = np.indices(shape).sum(axis=0).astype(np.float64) + 1
    voxel_mass = density * spacing**3
    report = {}
    failures = []
    for target in (0.001, 0.01):
        actual = spatial_average_sar(density, local_sar, (spacing,) * 3, target)
        expected, expected_status = reference(
            voxel_mass,
            local_sar,
            target,
            str(step1),
            str(step2),
        )
        common = np.isfinite(actual.sar) & (expected != 0)
        difference = np.abs(actual.sar[common] - expected[common])
        status_equal = np.array_equal(actual.status, expected_status.astype(np.uint8))
        maximum_relative_difference = float(
            np.max(difference / np.abs(expected[common]), initial=0.0)
        )
        passed = bool(
            np.any(common)
            and status_equal
            and maximum_relative_difference <= MAXIMUM_RELATIVE_SAR_DIFFERENCE
        )
        report[f"{1000 * target:g}g"] = {
            "compared_cells": int(np.count_nonzero(common)),
            "maximum_absolute_sar_difference_W_per_kg": float(np.max(difference, initial=0.0)),
            "maximum_relative_sar_difference": maximum_relative_difference,
            "status_arrays_equal": bool(status_equal),
            "maximum_allowed_relative_sar_difference": (MAXIMUM_RELATIVE_SAR_DIFFERENCE),
            "status": "PASS" if passed else "FAIL",
        }
        if not passed:
            failures.append(f"{1000 * target:g} g")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("Spatial-average SAR validation failed for " + ", ".join(failures))


if __name__ == "__main__":
    main()
