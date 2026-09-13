"""Export native gprMax outputs and compare them with saved MATLAB MoM data."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vivaldi_antenna_gprmax import ANGLE, MESHES, PATTERN_FREQUENCIES, RESULTS

FAR = "ntff/surface/frequency/spectrum/far_field"


def read_table(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def write_table(path, names, columns):
    np.savetxt(path, np.column_stack(columns), delimiter=",", header=",".join(names), comments="")


def interpolate_valid(frequency, values, valid, requested):
    """Interpolate complex values without crossing invalid bins or extrapolating."""
    right = np.searchsorted(frequency, requested).clip(1, len(frequency) - 1)
    left = right - 1
    if np.any(requested < frequency[0]) or np.any(requested > frequency[-1]) or not np.all(valid[left] & valid[right]):
        raise ValueError("The requested comparison band contains invalid port samples")
    return np.interp(requested, frequency, values.real) + 1j * np.interp(requested, frequency, values.imag)


def export_result(results, mesh):
    reference = read_table(results / "vivaldi_matlab_port.csv")
    requested = reference["frequency_hz"]
    summary = {}
    with h5py.File(results / f"vivaldi_{mesh}.h5", "r") as output:
        np.testing.assert_allclose(
            output["ntff/surface/frequency/spectrum/frequencies"][:],
            PATTERN_FREQUENCIES,
            rtol=1e-7,
            atol=0,
        )
        port = output["ports/feed"]
        if float(port.attrs["ReferenceImpedance"]) != 50:
            raise ValueError("The comparison requires a 50-ohm port reference")
        frequency = np.asarray(port["frequency"], dtype=float)
        spectra = []
        for name in ("Zin", "S11"):
            values = np.asarray(port[name], dtype=complex)
            valid = np.asarray(port[f"valid_{name}"], dtype=bool) & np.isfinite(values)
            spectra.append(interpolate_valid(frequency, values, valid, requested))
        z, s = spectra
        write_table(
            results / f"vivaldi_{mesh}_port.csv",
            ("frequency_hz", "Zin_real_ohm", "Zin_imag_ohm", "S11_real", "S11_imag"),
            (requested, z.real, z.imag, s.real, s.imag),
        )
        voltage = np.asarray(port["Vtotal"], dtype=float)
        summary["voltage_tail_peak_fraction"] = float(
            np.max(np.abs(voltage[-len(voltage) // 10 :])) / np.max(np.abs(voltage))
        )
        summary["independent_frequency_resolution_hz"] = float(port.attrs["IndependentFrequencyResolution"])
        summary["dt_s"] = float(output.attrs["dt"])
        full = output[FAR + "/full"]
        fields = full["fields"]
        summary["pattern_frequency_hz"] = list(PATTERN_FREQUENCIES)
        for name in ("directivity_dbi", "gain_dbi", "realized_gain_dbi"):
            values = np.asarray(fields[name], dtype=float)
            if values.shape[0] != len(PATTERN_FREQUENCIES) or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid full-sphere {name}")
            summary[f"peak_{name}"] = np.max(values, axis=1).tolist()
        for name in ("radiation_efficiency", "total_efficiency"):
            summary[name] = np.asarray(fields[name], dtype=float).ravel().tolist()
        for index, f in enumerate(PATTERN_FREQUENCIES):
            columns = [ANGLE]
            names = ["angle_deg"]
            for plane in ("xy", "xz"):
                group = output[FAR + f"/{plane}"]
                theta = np.radians(np.asarray(group["theta"]))
                phi = np.radians(np.asarray(group["phi"]))
                directions = np.column_stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))
                expected = np.column_stack((np.cos(np.radians(ANGLE)), np.sin(np.radians(ANGLE)), np.zeros(ANGLE.size)))
                if plane == "xz":
                    expected = expected[:, (0, 2, 1)]
                if not np.allclose(directions, expected, atol=5e-7, rtol=0):
                    raise ValueError(f"Unexpected angular convention for {plane}")
                for name in ("directivity_dbi", "gain_dbi", "realized_gain_dbi"):
                    columns.append(np.asarray(group[f"fields/{name}"][index], dtype=float))
                    names.append(f"{plane}_{name}")
            write_table(results / f"vivaldi_{mesh}_pattern_{f/1e9:g}GHz.csv", names, columns)
    (results / f"vivaldi_{mesh}_antenna_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def complex_column(table, name):
    suffix = "_ohm" if name == "Zin" else ""
    return table[f"{name}_real{suffix}"] + 1j * table[f"{name}_imag{suffix}"]


def compare_and_plot(results, meshes):
    reference = read_table(results / "vivaldi_matlab_port.csv")
    matlab_summary = json.loads((results / "vivaldi_matlab_summary.json").read_text())
    results_by_mesh = {mesh: read_table(results / f"vivaldi_{mesh}_port.csv") for mesh in meshes}
    styles = {
        "coarse": dict(color="0.55", linestyle="--"),
        "fine": dict(color="0.25", linestyle=":"),
        "finer": dict(color="black", linestyle="-"),
    }
    report = {
        "comparison_type": "independent numerical solvers; neither is ground truth",
        "matlab": matlab_summary,
        "meshes": {},
    }
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    f = reference["frequency_hz"] / 1e9
    zref, sref = complex_column(reference, "Zin"), complex_column(reference, "S11")
    for mesh, table in results_by_mesh.items():
        if not np.array_equal(table["frequency_hz"], reference["frequency_hz"]):
            raise ValueError("Port frequency grids differ")
        z, s = complex_column(table, "Zin"), complex_column(table, "S11")
        for ax, values in zip(axes, (20 * np.log10(abs(s)), z.real, z.imag)):
            ax.plot(f, values, label=f"gprMax {mesh}", **styles[mesh])
        report["meshes"][mesh] = dict(
            complex_s11_max_absolute_difference=float(np.max(abs(s - sref))),
            complex_s11_rms_absolute_difference=float(np.sqrt(np.mean(abs(s - sref) ** 2))),
            impedance_relative_l2_difference=float(np.linalg.norm(z - zref) / np.linalg.norm(zref)),
            antenna_metrics=json.loads((results / f"vivaldi_{mesh}_antenna_metrics.json").read_text()),
            run=json.loads((results / f"vivaldi_{mesh}.json").read_text()),
            patterns={},
        )
    for ax, values, label in zip(
        axes,
        (20 * np.log10(abs(sref)), zref.real, zref.imag),
        ("S11 (dB)", "Re Zin (ohm)", "Im Zin (ohm)"),
    ):
        ax.plot(f, values, "o", color="tab:purple", mfc="white", label="MATLAB MoM", markersize=5)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Frequency (GHz)")
    fig.suptitle("PEC Vivaldi — native 50-ohm port outputs")
    fig.tight_layout()
    fig.savefig(results / "vivaldi_port_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10, 13), subplot_kw={"projection": "polar"})
    floor = -25
    for row, frequency in enumerate(PATTERN_FREQUENCIES):
        ref = read_table(results / f"vivaldi_matlab_pattern_{frequency/1e9:g}GHz.csv")
        for col, plane in enumerate(("xy", "xz")):
            ax = axes[row, col]
            reference_dbi = ref[f"{plane}_directivity_dbi"]
            for mesh in meshes:
                table = read_table(results / f"vivaldi_{mesh}_pattern_{frequency/1e9:g}GHz.csv")
                if not np.array_equal(table["angle_deg"], ref["angle_deg"]):
                    raise ValueError("Principal-plane angle grids differ")
                dbi = table[f"{plane}_directivity_dbi"]
                ax.plot(
                    np.radians(ANGLE),
                    np.maximum(dbi, floor) - floor,
                    label=f"gprMax {mesh}",
                    **styles[mesh],
                )
                selected = reference_dbi > reference_dbi.max() - 20
                report["meshes"][mesh]["patterns"][f"{frequency/1e9:g}GHz_{plane}"] = dict(
                    rms_absolute_directivity_difference_db=float(
                        np.sqrt(np.mean((dbi[selected] - reference_dbi[selected]) ** 2))
                    ),
                    comparison_region="MATLAB cut above its peak minus 20 dB; no fitted normalization",
                )
            ax.plot(
                np.radians(ANGLE[::5]),
                np.maximum(reference_dbi[::5], floor) - floor,
                "o",
                color="tab:purple",
                mfc="white",
                markersize=3,
                label="MATLAB MoM",
            )
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylim(0, 35)
            ax.set_yticks([5, 15, 25, 35], ["−20", "−10", "0", "10"])
            ax.set_title(
                f"{frequency/1e9:g} GHz — {plane.upper()} ({'E' if col == 0 else 'H'} plane)",
                pad=18,
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.962))
    fig.suptitle("Absolute directivity (dBi), full circles; 0° = +x\nValues below −25 dBi clipped for display only")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(results / "vivaldi_pattern_comparison.png", dpi=180)
    plt.close(fig)
    (results / "vivaldi_comparison_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--meshes", nargs="+", choices=tuple(MESHES), default=list(MESHES))
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="replot retained CSV/JSON without solver HDF5 files",
    )
    args = parser.parse_args()
    if not args.no_export:
        for mesh in args.meshes:
            export_result(args.results_dir, mesh)
    compare_and_plot(args.results_dir, args.meshes)


if __name__ == "__main__":
    main()
