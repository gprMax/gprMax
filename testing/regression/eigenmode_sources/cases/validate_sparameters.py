from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

MIN_BEND_S21_IMPROVEMENT_DB = 2.0
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
PARTIAL_CUTOFF_GUIDE_WIDTH_M = 0.006
PARTIAL_CUTOFF_TE10_HZ = SPEED_OF_LIGHT_M_PER_S / (2.0 * PARTIAL_CUTOFF_GUIDE_WIDTH_M)
PARTIAL_CUTOFF_START_HZ = 24.25e9
PARTIAL_CUTOFF_STOP_HZ = 34.75e9
PARTIAL_CUTOFF_DFT_POINTS = 100
PARTIAL_CUTOFF_INVALID_POINTS = 7
PARTIAL_CUTOFF_FREQUENCY_ATOL_HZ = 5e3


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
        raise AssertionError(
            f"Missing valid port {destination_port}, mode {destination_mode} data in {path}"
        )
    return np.asarray([float(row["S_magnitude_db"]) for row in rows])


def _paths_below(root: Path, family: str) -> list[Path]:
    return [path for path in root.rglob("*_sparameters.csv") if family in path.parts]


def validate_straight(root: Path) -> list[str]:
    messages = []
    for path in _paths_below(root, "straight_waveguide"):
        if "rectangular_waveguide_partial_cutoff" in path.parts:
            continue
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
            f"{label}: mean S21={np.mean(s21):.3f} dB, " f"max S11={np.max(s11):.3f} dB"
        )
    return messages


def _modal_rows(path: Path, destination_port: int, destination_mode: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if int(row["source_port"]) == 1
            and int(row["source_mode"]) == 1
            and int(row["destination_port"]) == destination_port
            and int(row["destination_mode"]) == destination_mode
        ]
    rows.sort(key=lambda row: float(row["frequency_hz"]))
    if not rows:
        raise AssertionError(f"Missing S{destination_port}1 mode-1 data in {path}")
    return rows


def validate_partial_cutoff(root: Path) -> list[str]:
    messages = []
    numeric_s_columns = (
        "S_real",
        "S_imag",
        "S_magnitude",
        "S_magnitude_db",
        "S_phase_deg",
        "coefficient_magnitude_squared",
    )
    for path in _paths_below(root, "rectangular_waveguide_partial_cutoff"):
        series = {}
        frequency_series = {}
        propagating_frequency_series = {}
        expected_frequencies = np.linspace(
            PARTIAL_CUTOFF_START_HZ,
            PARTIAL_CUTOFF_STOP_HZ,
            PARTIAL_CUTOFF_DFT_POINTS,
        )
        expected_valid = expected_frequencies > PARTIAL_CUTOFF_TE10_HZ
        for destination_port in (1, 2):
            rows = _modal_rows(path, destination_port, 1)
            frequencies = np.asarray([float(row["frequency_hz"]) for row in rows])
            valid = np.asarray([bool(int(row["valid"])) for row in rows])
            if frequencies.shape != expected_frequencies.shape or not np.allclose(
                frequencies,
                expected_frequencies,
                rtol=0.0,
                atol=PARTIAL_CUTOFF_FREQUENCY_ATOL_HZ,
            ):
                raise AssertionError(
                    f"{path}: S{destination_port}1 must contain the complete "
                    f"{PARTIAL_CUTOFF_DFT_POINTS}-point partial-cutoff DFT grid"
                )
            if not np.array_equal(valid, expected_valid):
                unexpected = frequencies[valid != expected_valid] * 1e-9
                raise AssertionError(
                    f"{path}: S{destination_port}1 cutoff validity differs at "
                    f"{unexpected.tolist()} GHz"
                )
            invalid_rows = [row for row, is_valid in zip(rows, valid) if not is_valid]
            expected_invalid_count = PARTIAL_CUTOFF_INVALID_POINTS
            if (
                len(invalid_rows) != expected_invalid_count
                or np.count_nonzero(valid) != PARTIAL_CUTOFF_DFT_POINTS - expected_invalid_count
            ):
                raise AssertionError(
                    f"{path}: expected {expected_invalid_count} cutoff and "
                    f"{PARTIAL_CUTOFF_DFT_POINTS - expected_invalid_count} propagating bins "
                    f"for S{destination_port}1, got {len(invalid_rows)} and "
                    f"{np.count_nonzero(valid)}"
                )
            if any(
                not np.isnan(float(row[column]))
                for row in invalid_rows
                for column in numeric_s_columns
            ):
                raise AssertionError(
                    f"{path}: invalid S{destination_port}1 cutoff rows must contain NaN values"
                )
            valid_rows = [row for row, is_valid in zip(rows, valid) if is_valid]
            if any(
                not np.isfinite(float(row[column]))
                for row in valid_rows
                for column in numeric_s_columns
            ):
                raise AssertionError(
                    f"{path}: valid S{destination_port}1 propagating rows must be finite"
                )
            series[destination_port] = np.asarray(
                [float(row["S_magnitude_db"]) for row in valid_rows]
            )
            frequency_series[destination_port] = frequencies
            propagating_frequency_series[destination_port] = frequencies[valid]

        s11 = series[1]
        s21 = series[2]
        if not np.array_equal(frequency_series[1], frequency_series[2]):
            raise AssertionError(f"{path}: S11 and S21 DFT frequency grids differ")
        if not np.array_equal(propagating_frequency_series[1], propagating_frequency_series[2]):
            raise AssertionError(f"{path}: S11 and S21 propagating frequency grids differ")
        propagating_frequencies = propagating_frequency_series[1]
        settled = propagating_frequencies >= PARTIAL_CUTOFF_TE10_HZ + 0.2e9
        if not np.any(settled):
            raise AssertionError(f"{path}: missing settled propagating bins above cutoff")
        if np.max(s11[settled]) >= -20:
            raise AssertionError(
                f"{path}: expected settled propagating-bin S11 below -20 dB, "
                f"got {np.max(s11[settled]):.3f} dB"
            )
        if np.max(np.abs(s21[settled])) >= 0.1:
            raise AssertionError(
                f"{path}: expected settled propagating-bin S21 within 0.1 dB of 0 dB, "
                f"got range {np.min(s21[settled]):.3f} to {np.max(s21[settled]):.3f} dB"
            )
        if np.max(np.abs(s21)) >= 0.25:
            raise AssertionError(
                f"{path}: expected all propagating-bin S21 within 0.25 dB of 0 dB, "
                f"got range {np.min(s21):.3f} to {np.max(s21):.3f} dB"
            )
        label = path.parent.relative_to(root)
        messages.append(
            f"{label}: {expected_invalid_count} cutoff bins invalid; propagating S21 range="
            f"{np.min(s21):.3f} to {np.max(s21):.3f} dB, "
            f"max settled S11={np.max(s11[settled]):.3f} dB"
        )
    return messages


def validate_grid_spacing(root: Path) -> list[str]:
    paths = _paths_below(root, "grid_spacing")
    messages = []
    for path in paths:
        s21 = _series(path, 2, 1)
        minimum = float(np.min(s21))
        maximum = float(np.max(s21))
        maximum_absolute = float(np.max(np.abs(s21)))
        half_peak_to_peak = 0.5 * float(np.ptp(s21))
        message = (
            "%s: S21 range=%.3f to %.3f dB, max absolute=%.3f dB, half peak-to-peak=%.3f dB"
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
            f"Expected lossy S21 to be at least 3 dB lower; got {lossy:.3f} "
            f"versus {nonlossy:.3f} dB"
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
        validate_partial_cutoff,
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
    parser = argparse.ArgumentParser(
        description="Validate eigenmode regression S-parameter expectations."
    )
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    for message in validate_tree(parser.parse_args().root):
        print(f"PASS: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
