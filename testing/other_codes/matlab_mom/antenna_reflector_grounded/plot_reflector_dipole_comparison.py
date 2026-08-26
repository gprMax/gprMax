"""Compare grounded strip-dipole results from gprMax and MATLAB."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CASES = ("lambda_8", "lambda_4")
PLANES = ("e_plane", "h_plane")
FLOOR_DB = -40.0


def _read_csv(path: Path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def _complex_field(values, component):
    return np.asarray(values[f"{component}_real"] + 1j * values[f"{component}_imag"])


def _complex_impedance(values):
    return np.asarray(values["Zin_real_ohm"] + 1j * values["Zin_imag_ohm"])


def _normalise(values):
    return values - np.nanmax(values)


def _case_label(case):
    return {"lambda_8": r"$\lambda_0/8$", "lambda_4": r"$\lambda_0/4$"}[case]


def _pattern_error(reference, candidate):
    reference = _normalise(reference)
    candidate = _normalise(candidate)
    valid = (reference >= FLOOR_DB) & (candidate >= FLOOR_DB)
    difference = candidate[valid] - reference[valid]
    return {
        "rms_difference_db": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_difference_db": float(np.max(np.abs(difference))),
        "retained_points": int(np.count_nonzero(valid)),
    }


def _complex_field_error(matlab_parts, gprmax_parts):
    matlab = np.concatenate(matlab_parts)
    gprmax = np.concatenate(gprmax_parts)
    # One complex scale per physical case removes the arbitrary feed/range
    # convention but preserves both components and both principal-plane levels.
    scale = np.vdot(matlab, gprmax) / np.vdot(matlab, matlab)
    fitted = scale * matlab
    peak = max(float(np.max(np.abs(fitted))), np.finfo(float).tiny)
    retained = np.maximum(np.abs(fitted), np.abs(gprmax)) >= peak * 10 ** (FLOOR_DB / 20)
    error = np.abs(gprmax[retained] - fitted[retained]) / peak
    return {
        "least_squares_scale_real": float(scale.real),
        "least_squares_scale_imag": float(scale.imag),
        "rms_error_peak_normalised": float(np.sqrt(np.mean(error**2))),
        "maximum_error_peak_normalised": float(np.max(error)),
        "retained_complex_samples": int(np.count_nonzero(retained)),
    }


def _interpolate_complex(source_frequency, values, target_frequency):
    return np.interp(target_frequency, source_frequency, values.real) + 1j * np.interp(
        target_frequency, source_frequency, values.imag
    )


def main() -> None:
    matlab_summary = json.loads((RESULTS / "reflector_dipole_matlab_summary.json").read_text())
    gprmax_summary = json.loads((RESULTS / "reflector_dipole_gprmax_summary.json").read_text())
    metrics = {"cases": {}}

    fig, axes = plt.subplots(
        len(CASES),
        len(PLANES),
        figsize=(10, 8),
        subplot_kw={"projection": "polar"},
    )
    for row, case in enumerate(CASES):
        case_metrics = {"patterns": {}}
        matlab_field_parts = []
        gprmax_field_parts = []
        for column, plane in enumerate(PLANES):
            matlab_values = _read_csv(RESULTS / f"reflector_dipole_matlab_{case}_{plane}.csv")
            gprmax_values = _read_csv(RESULTS / f"reflector_dipole_gprmax_{case}_{plane}.csv")
            np.testing.assert_allclose(matlab_values["theta_deg"], gprmax_values["theta_deg"], rtol=0, atol=1e-12)
            matlab_directivity = np.asarray(matlab_values["directivity_dbi"])
            gprmax_directivity = np.asarray(gprmax_values["directivity_dbi"])
            case_metrics["patterns"][plane] = _pattern_error(matlab_directivity, gprmax_directivity)
            for component in ("Etheta", "Ephi"):
                matlab_field_parts.append(_complex_field(matlab_values, component))
                gprmax_field_parts.append(_complex_field(gprmax_values, component))

            axis = axes[row, column]
            theta = np.deg2rad(matlab_values["theta_deg"])
            axis.plot(theta, _normalise(matlab_directivity), "k-", label="MATLAB MoM")
            axis.plot(
                theta,
                _normalise(gprmax_directivity),
                "ko",
                markerfacecolor="white",
                markersize=3,
                markevery=3,
                label="gprMax FDTD",
            )
            axis.set_theta_zero_location("N")
            axis.set_theta_direction(-1)
            axis.set_thetamin(0)
            axis.set_thetamax(90)
            axis.set_rlim(FLOOR_DB, 0)
            axis.set_rticks((-40, -30, -20, -10, 0))
            axis.grid(True, alpha=0.3)
            axis.set_title(f"{_case_label(case)}, {plane.replace('_', ' ')}")
        case_metrics["complex_fields"] = _complex_field_error(matlab_field_parts, gprmax_field_parts)

        matlab_case = matlab_summary["cases"][case]
        gprmax_case = gprmax_summary["cases"][case]
        case_metrics["maximum_directivity"] = {
            "matlab_dbi": matlab_case["maximum_directivity_dbi"],
            "gprmax_dbi": gprmax_case["maximum_directivity_dbi"],
            "difference_db": gprmax_case["maximum_directivity_dbi"] - matlab_case["maximum_directivity_dbi"],
        }
        case_metrics["radiation_efficiency"] = {
            "matlab": matlab_case["radiation_efficiency"],
            "gprmax": gprmax_case["radiation_efficiency"],
            "difference": gprmax_case["radiation_efficiency"] - matlab_case["radiation_efficiency"],
        }
        metrics["cases"][case] = case_metrics

    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(-0.18, 1.18), fontsize=8)
    fig.suptitle("Strip dipole above PEC at 1 GHz: normalised directivity", y=0.995)
    fig.tight_layout()
    pattern_path = RESULTS / "reflector_dipole_pattern_comparison.png"
    fig.savefig(pattern_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, (magnitude_axis, impedance_axis) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    styles = {
        ("lambda_8", "matlab"): ("k", "-"),
        ("lambda_8", "gprmax"): ("k", "o"),
        ("lambda_4", "matlab"): ("0.45", "--"),
        ("lambda_4", "gprmax"): ("0.45", "s"),
    }
    for case in CASES:
        matlab_port = _read_csv(RESULTS / f"reflector_dipole_matlab_{case}_port.csv")
        gprmax_port = _read_csv(RESULTS / f"reflector_dipole_gprmax_{case}_port.csv")
        frequency = np.asarray(matlab_port["frequency_hz"])
        np.testing.assert_allclose(frequency, gprmax_port["frequency_hz"], rtol=0, atol=1e-6)
        for solver, values in (("matlab", matlab_port), ("gprmax", gprmax_port)):
            colour, style = styles[(case, solver)]
            label = f"{solver}, {_case_label(case)}"
            if style in ("o", "s"):
                magnitude_axis.plot(
                    frequency / 1e9,
                    values["S11_magnitude_db"],
                    style,
                    color=colour,
                    markerfacecolor="white",
                    markersize=4,
                    label=label,
                )
            else:
                magnitude_axis.plot(frequency / 1e9, values["S11_magnitude_db"], style, color=colour, label=label)
        matlab_z = _complex_impedance(matlab_port)
        gprmax_z = _complex_impedance(gprmax_port)
        impedance_axis.plot(frequency / 1e9, matlab_z.real, "k-" if case == "lambda_8" else "k--")
        impedance_axis.plot(
            frequency / 1e9,
            gprmax_z.real,
            "ko" if case == "lambda_8" else "ks",
            markerfacecolor="white",
            markersize=3,
        )
        impedance_axis.plot(frequency / 1e9, matlab_z.imag, color="0.55", linestyle="-")
        impedance_axis.plot(
            frequency / 1e9,
            gprmax_z.imag,
            "^",
            color="0.55",
            markerfacecolor="white",
            markersize=3,
        )
        delta_s11 = np.asarray(gprmax_port["S11_magnitude_db"] - matlab_port["S11_magnitude_db"])
        pattern_index = int(np.argmin(np.abs(frequency - 1e9)))
        metrics["cases"][case]["s11"] = {
            "magnitude_rms_difference_db": float(np.sqrt(np.mean(delta_s11**2))),
            "magnitude_maximum_absolute_difference_db": float(np.max(np.abs(delta_s11))),
            "magnitude_difference_at_1ghz_db": float(delta_s11[pattern_index]),
        }
        delta_z = gprmax_z - matlab_z
        metrics["cases"][case]["impedance"] = {
            "complex_rms_difference_ohm": float(np.sqrt(np.mean(np.abs(delta_z) ** 2))),
            "complex_maximum_difference_ohm": float(np.max(np.abs(delta_z))),
            "complex_difference_at_1ghz_ohm": {
                "real": float(delta_z[pattern_index].real),
                "imag": float(delta_z[pattern_index].imag),
            },
        }

    magnitude_axis.set_ylabel(r"$|S_{11}|$ [dB]")
    magnitude_axis.grid(True, alpha=0.3)
    magnitude_axis.legend(fontsize=8, ncol=2)
    impedance_axis.set_ylabel(r"$Z_{\mathrm{in}}$ real/imaginary [$\Omega$]")
    impedance_axis.set_xlabel("frequency [GHz]")
    impedance_axis.grid(True, alpha=0.3)
    impedance_axis.text(0.01, 0.96, "black: real; grey: imaginary", transform=impedance_axis.transAxes, va="top")
    fig.tight_layout()
    port_path = RESULTS / "reflector_dipole_port_comparison.png"
    fig.savefig(port_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    (RESULTS / "reflector_dipole_comparison_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {pattern_path}")
    print(f"Saved {port_path}")


if __name__ == "__main__":
    main()
