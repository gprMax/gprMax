from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


MIN_BEND_S21_IMPROVEMENT_DB = 2.0


def _series(path: Path, destination_port: int, destination_mode: int) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if int(row["destination_port"]) == destination_port
            and int(row["destination_mode"]) == destination_mode
            and bool(int(row["valid"]))
        ]
    if not rows:
        raise AssertionError(f"Missing valid port {destination_port}, mode {destination_mode} data in {path}")
    return np.asarray([float(row["S_magnitude_db"]) for row in rows])


def _paths_below(root: Path, family: str) -> list[Path]:
    return [path for path in root.rglob("*_sparameters.csv") if family in path.parts]


def validate_straight(root: Path) -> list[str]:
    messages = []
    for path in _paths_below(root, "straight_waveguide"):
        s11 = _series(path, 1, 1)
        s21 = _series(path, 2, 1)
        if np.max(s11) >= -20:
            raise AssertionError(f"{path}: expected S11 below -20 dB, got {np.max(s11):.3f} dB")
        if np.max(np.abs(s21)) >= 0.75:
            raise AssertionError(
                f"{path}: expected fundamental S21 within 0.75 dB of 0 dB, "
                f"got range {np.min(s21):.3f} to {np.max(s21):.3f} dB"
            )
        label = path.parent.relative_to(root)
        messages.append(
            f"{label}: mean S21={np.mean(s21):.3f} dB, "
            f"max S11={np.max(s11):.3f} dB"
        )
    return messages


def validate_grid_spacing(root: Path) -> list[str]:
    paths = _paths_below(root, 'grid_spacing')
    messages = []
    for path in paths:
        s21 = _series(path, 2, 1)
        minimum = float(np.min(s21))
        maximum = float(np.max(s21))
        maximum_absolute = float(np.max(np.abs(s21)))
        half_peak_to_peak = 0.5 * float(np.ptp(s21))
        message = (
            '%s: S21 range=%.3f to %.3f dB, max absolute=%.3f dB, half peak-to-peak=%.3f dB'
            % (path.parent.name, minimum, maximum, maximum_absolute, half_peak_to_peak)
        )
        messages.append(message)
    return messages


def validate_bends(root: Path) -> list[str]:
    messages = []
    paths = _paths_below(root, "bending_waveguide")
    by_polarisation: dict[str, dict[str, Path]] = {}
    for path in paths:
        polarisation = next(
            (part for part in path.parts if part in {"2d_tm", "2d_te"}),
            None,
        )
        if polarisation is not None:
            by_polarisation.setdefault(polarisation, {})[path.parent.name] = path

    required = {"small_bend", "medium_bend", "large_bend"}
    for polarisation, cases in sorted(by_polarisation.items()):
        if not required <= cases.keys():
            continue
        mean_s21 = {bend: float(np.mean(_series(cases[bend], 2, 1))) for bend in required}
        if not (mean_s21["small_bend"] < mean_s21["medium_bend"] < mean_s21["large_bend"]):
            raise AssertionError(
                f"{polarisation}: expected mean fundamental S21 to improve "
                "monotonically with bend radius; got small="
                f"{mean_s21['small_bend']:.3f}, medium="
                f"{mean_s21['medium_bend']:.3f}, large="
                f"{mean_s21['large_bend']:.3f} dB"
            )
        improvement = mean_s21["large_bend"] - mean_s21["small_bend"]
        if improvement < MIN_BEND_S21_IMPROVEMENT_DB:
            raise AssertionError(
                f"{polarisation}: expected at least "
                f"{MIN_BEND_S21_IMPROVEMENT_DB:g} dB mean fundamental S21 "
                f"improvement from the small to large bend, got "
                f"{improvement:.3f} dB"
            )
        messages.append(
            f"{polarisation} curved bends: mean S21 small="
            f"{mean_s21['small_bend']:.3f}, medium="
            f"{mean_s21['medium_bend']:.3f}, large="
            f"{mean_s21['large_bend']:.3f} dB; improvement="
            f"{improvement:.3f} dB"
        )
    return messages


def validate_loss(root: Path) -> list[str]:
    paths = _paths_below(root, "loss_comparison")
    by_name = {path.parent.name: path for path in paths}
    if not {"lossy", "nonlossy"} <= by_name.keys():
        return []
    lossy = np.mean(_series(by_name["lossy"], 2, 1))
    nonlossy = np.mean(_series(by_name["nonlossy"], 2, 1))
    if lossy >= nonlossy - 3:
        raise AssertionError(
            f"Expected lossy S21 to be at least 3 dB lower; got {lossy:.3f} " f"versus {nonlossy:.3f} dB"
        )
    return [f"loss comparison: lossy={lossy:.3f} dB, non-lossy={nonlossy:.3f} dB"]


def validate_source_profiles(root: Path) -> list[str]:
    paths = _paths_below(root, "broadband_vs_single_frequency")
    by_name = {path.parent.name: path for path in paths}
    if not {"broadband", "single_frequency"} <= by_name.keys():
        return []
    broadband = _series(by_name["broadband"], 2, 1)
    single = _series(by_name["single_frequency"], 2, 1)
    broadband_error = float(np.mean(np.abs(broadband)))
    single_error = float(np.mean(np.abs(single)))
    if broadband_error >= single_error:
        raise AssertionError(
            "Expected multi-anchor S21 to remain closer to 0 dB than the "
            f"single-profile result; got {broadband_error:.4f} versus {single_error:.4f} dB"
        )
    return [
        f"source-profile comparison: broadband mean |S21|={broadband_error:.4f} dB, "
        f"single-profile={single_error:.4f} dB"
    ]


def validate_tree(root: Path) -> list[str]:
    root = root.resolve()
    messages = []
    for validator in (
        validate_straight,
        validate_grid_spacing,
        validate_bends,
        validate_loss,
        validate_source_profiles,
    ):
        messages.extend(validator(root))
    if not messages:
        raise FileNotFoundError(f"No complete S-parameter expectation set found below {root}")
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate eigenmode regression S-parameter expectations.")
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    for message in validate_tree(parser.parse_args().root):
        print(f"PASS: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
