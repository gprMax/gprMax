"""Record MATLAB mesh and optional FDTD time-window sensitivity checks."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from plot_vivaldi_comparison import complex_column, interpolate_valid, read_table
from vivaldi_antenna_gprmax import PATTERN_FREQUENCIES, RESULTS


def check(results, time_dir=None):
    first = read_table(results / "vivaldi_matlab_port.csv")
    refined = results / "matlab_mesh6mm"
    second = read_table(refined / "vivaldi_matlab_port.csv")
    np.testing.assert_array_equal(first["frequency_hz"], second["frequency_hz"])
    dz = complex_column(first, "Zin") - complex_column(second, "Zin")
    ds = complex_column(first, "S11") - complex_column(second, "S11")
    report = {
        "matlab_mesh_8mm_vs_6mm": {
            "complex_s11_max_absolute_difference": float(np.max(abs(ds))),
            "impedance_relative_l2_difference": float(
                np.linalg.norm(dz) / np.linalg.norm(complex_column(second, "Zin"))
            ),
            "reference_8mm": json.loads((results / "vivaldi_matlab_summary.json").read_text()),
            "reference_6mm": json.loads((refined / "vivaldi_matlab_summary.json").read_text()),
        }
    }
    if time_dir is not None:
        baseline = read_table(results / "vivaldi_coarse_port.csv")
        requested = baseline["frequency_hz"]
        time_check = {}
        with h5py.File(time_dir / "vivaldi_coarse.h5", "r") as output:
            port = output["ports/feed"]
            for name in ("Zin", "S11"):
                values = interpolate_valid(
                    port["frequency"][:],
                    port[name][:],
                    port[f"valid_{name}"][:].astype(bool),
                    requested,
                )
                difference = values - complex_column(baseline, name)
                time_check[f"{name}_max_absolute_difference"] = float(np.max(abs(difference)))
            with h5py.File(results / "vivaldi_coarse.h5", "r") as shorter:
                path = "ntff/surface/frequency/spectrum/far_field"
                for plane in ("xy", "xz"):
                    key = f"{path}/{plane}/fields/directivity_dbi"
                    first_d = np.asarray(shorter[key], dtype=float)
                    second_d = np.asarray(output[key], dtype=float)
                    selected = first_d > first_d.max(axis=1, keepdims=True) - 20
                    time_check[f"{plane}_max_directivity_difference_db_above_minus20"] = float(
                        np.max(abs(first_d - second_d)[selected])
                    )
            voltage = port["Vtotal"][:]
            time_check["long_run_voltage_tail_fraction"] = float(
                np.max(abs(voltage[-len(voltage) // 10 :])) / np.max(abs(voltage))
            )
        time_check["long_run"] = json.loads((time_dir / "vivaldi_coarse.json").read_text())
        time_check["baseline_run"] = json.loads((results / "vivaldi_coarse.json").read_text())
        report["coarse_time_window"] = time_check
    report["pattern_frequency_hz"] = list(PATTERN_FREQUENCIES)
    (results / "vivaldi_sensitivity_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--time-dir", type=Path, help="directory containing a longer coarse HDF5 run")
    args = parser.parse_args()
    check(args.results_dir, args.time_dir)
