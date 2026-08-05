"""Export and plot the patch sweep and the companion 2.45 GHz far field."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
S_PARAMETER_DIR = HERE / "S-parameter"
FARFIELD_DIR = HERE / "Farfield"
SPARAMETERS = S_PARAMETER_DIR / "patch_antenna_sparameters.csv"
CST_S1P = S_PARAMETER_DIR / "patch_cst.s1p"
CST_FAR_FIELD = FARFIELD_DIR / "patch_ff_cst.txt"
OUTPUT_H5 = FARFIELD_DIR / "patch_antenna_farfield.h5"
FAR_FIELD_GROUP = "ntff/patch_surface/frequency/patch_farfield_band/far_field/full_sphere"
DESIGN_FREQUENCY_HZ = 2.45e9


def read_s11() -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    with SPARAMETERS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (
                int(row["destination_port"]) == 1
                and int(row["destination_mode"]) == 1
                and bool(int(row["valid"]))
            ):
                rows.append((float(row["frequency_hz"]), float(row["S_magnitude_db"])))
    if not rows:
        raise ValueError(f"No valid port-1 mode-1 S11 samples in {SPARAMETERS}")
    data = np.asarray(sorted(rows), dtype=np.float64)
    return data[:, 0], data[:, 1]


def read_cst_s11() -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    for line in CST_S1P.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("!", "#")):
            continue
        frequency_ghz, magnitude, _phase_deg = map(float, stripped.split()[:3])
        rows.append((frequency_ghz * 1e9, 20.0 * np.log10(magnitude)))
    if not rows:
        raise ValueError(f"No S11 samples in {CST_S1P}")
    data = np.asarray(rows, dtype=np.float64)
    return data[:, 0], data[:, 1]


def read_far_field() -> dict[str, np.ndarray]:
    with h5py.File(OUTPUT_H5, "r") as output:
        group = output[FAR_FIELD_GROUP]
        fields = group["fields"]
        frequencies = np.asarray(group.parent.parent["frequencies"], dtype=np.float64)
        frequency_index = int(np.argmin(np.abs(frequencies - DESIGN_FREQUENCY_HZ)))
        if not np.isclose(frequencies[frequency_index], DESIGN_FREQUENCY_HZ):
            raise ValueError(f"No 2.45 GHz far-field bin in {OUTPUT_H5}: {frequencies}")
        return {
            "theta": np.asarray(group["theta"], dtype=np.float64),
            "phi": np.asarray(group["phi"], dtype=np.float64),
            "Etheta": np.asarray(fields["Etheta"])[frequency_index],
            "Ephi": np.asarray(fields["Ephi"])[frequency_index],
            "radiation_intensity": np.asarray(fields["radiation_intensity"], dtype=np.float64)[frequency_index],
            "directivity_dbi": np.asarray(fields["directivity_dbi"], dtype=np.float64)[frequency_index],
            "gain_dbi": np.asarray(fields["gain_dbi"], dtype=np.float64)[frequency_index],
            "realized_gain_dbi": np.asarray(fields["realized_gain_dbi"], dtype=np.float64)[frequency_index],
            "radiation_efficiency": np.asarray(fields["radiation_efficiency"], dtype=np.float64)[frequency_index],
            "total_efficiency": np.asarray(fields["total_efficiency"], dtype=np.float64)[frequency_index],
        }


def read_cst_far_field() -> dict[str, np.ndarray]:
    data = np.loadtxt(CST_FAR_FIELD, skiprows=2)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Unexpected CST far-field format in {CST_FAR_FIELD}")
    return {
        "theta": data[:, 0],
        "phi": data[:, 1],
        "directivity_dbi": data[:, 2],
    }


def export_far_field(far_field: dict[str, np.ndarray]) -> Path:
    path = FARFIELD_DIR / "patch_farfield_2p45GHz_1deg.csv"
    etheta = far_field["Etheta"]
    ephi = far_field["Ephi"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "theta_deg",
                "phi_deg",
                "Etheta_real_V",
                "Etheta_imag_V",
                "Ephi_real_V",
                "Ephi_imag_V",
                "radiation_intensity_W_per_sr",
                "directivity_dbi",
                "gain_dbi",
                "realized_gain_dbi",
            )
        )
        for index in range(far_field["theta"].size):
            writer.writerow(
                (
                    far_field["theta"][index],
                    far_field["phi"][index],
                    etheta[index].real,
                    etheta[index].imag,
                    ephi[index].real,
                    ephi[index].imag,
                    far_field["radiation_intensity"][index],
                    far_field["directivity_dbi"][index],
                    far_field["gain_dbi"][index],
                    far_field["realized_gain_dbi"][index],
                )
            )
    return path


def plot_s11(
    frequency: np.ndarray,
    s11_db: np.ndarray,
    cst_frequency: np.ndarray,
    cst_s11_db: np.ndarray,
) -> tuple[float, float, float, float]:
    minimum = int(np.nanargmin(s11_db))
    minimum_frequency = float(frequency[minimum])
    minimum_s11 = float(s11_db[minimum])
    cst_minimum = int(np.nanargmin(cst_s11_db))
    cst_minimum_frequency = float(cst_frequency[cst_minimum])
    cst_minimum_s11 = float(cst_s11_db[cst_minimum])
    figure, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    axis.plot(frequency * 1e-9, s11_db, linewidth=2, label="gprMax")
    axis.plot(cst_frequency * 1e-9, cst_s11_db, linewidth=2, label="CST")
    axis.plot(minimum_frequency * 1e-9, minimum_s11, "o", label=f"gprMax min: {minimum_frequency * 1e-9:.3f} GHz, {minimum_s11:.1f} dB")
    axis.plot(cst_minimum_frequency * 1e-9, cst_minimum_s11, "o", label=f"CST min: {cst_minimum_frequency * 1e-9:.3f} GHz, {cst_minimum_s11:.1f} dB")
    axis.set(xlabel="Frequency (GHz)", ylabel="S11 magnitude (dB)", title="SAB-derived patch antenna input reflection")
    axis.set_xlim(1.6, 3.2)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(S_PARAMETER_DIR / "patch_s11.png", dpi=180)
    plt.close(figure)
    return minimum_frequency, minimum_s11, cst_minimum_frequency, cst_minimum_s11


def plot_far_field(
    far_field: dict[str, np.ndarray], cst_far_field: dict[str, np.ndarray]
) -> tuple[float, float, float, float, float, float]:
    theta_axis = np.unique(far_field["theta"])
    phi_axis = np.unique(far_field["phi"])
    directivity = far_field["directivity_dbi"].reshape(theta_axis.size, phi_axis.size)
    peak_index = int(np.nanargmax(directivity))
    peak_theta_index, peak_phi_index = np.unravel_index(peak_index, directivity.shape)
    peak = float(directivity[peak_theta_index, peak_phi_index])
    peak_theta = float(theta_axis[peak_theta_index])
    peak_phi = float(phi_axis[peak_phi_index])

    cst_theta_axis = np.unique(cst_far_field["theta"])
    cst_phi_axis = np.unique(cst_far_field["phi"])
    cst_directivity = cst_far_field["directivity_dbi"].reshape(
        cst_phi_axis.size, cst_theta_axis.size
    ).T
    cst_peak_index = int(np.nanargmax(cst_directivity))
    cst_peak_theta_index, cst_peak_phi_index = np.unravel_index(
        cst_peak_index, cst_directivity.shape
    )
    cst_peak = float(cst_directivity[cst_peak_theta_index, cst_peak_phi_index])
    cst_peak_theta = float(cst_theta_axis[cst_peak_theta_index])
    cst_peak_phi = float(cst_phi_axis[cst_peak_phi_index])

    def full_plane_cut(
        theta: np.ndarray,
        phi: np.ndarray,
        values: np.ndarray,
        positive_phi: float,
        negative_phi: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        positive_index = int(np.argmin(np.abs(phi - positive_phi)))
        negative_index = int(np.argmin(np.abs(phi - negative_phi)))
        angles = np.concatenate((theta, 360.0 - theta[-2:0:-1]))
        full_values = np.concatenate(
            (values[:, positive_index], values[-2:0:-1, negative_index])
        )
        return angles, full_values

    figure, axes = plt.subplots(
        1,
        2,
        subplot_kw={"projection": "polar"},
        figsize=(12.0, 6.2),
        constrained_layout=True,
    )
    floor_dbi = -30.0
    planes = (
        (axes[0], "XZ plane", 0.0, 180.0),
        (axes[1], "YZ plane", 90.0, 270.0),
    )
    for axis, plane_name, positive_phi, negative_phi in planes:
        angles, gprmax_cut = full_plane_cut(
            theta_axis, phi_axis, directivity, positive_phi, negative_phi
        )
        cst_angles, cst_cut = full_plane_cut(
            cst_theta_axis,
            cst_phi_axis,
            cst_directivity,
            positive_phi,
            negative_phi,
        )
        axis.plot(
            np.deg2rad(angles),
            np.clip(gprmax_cut, floor_dbi, None) - floor_dbi,
            linewidth=2,
            label="gprMax",
        )
        axis.plot(
            np.deg2rad(cst_angles),
            np.clip(cst_cut, floor_dbi, None) - floor_dbi,
            linewidth=2,
            linestyle="--",
            label="CST",
        )
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_ylim(0, 40)
        axis.set_yticks((10, 20, 30, 40), labels=("-20", "-10", "0", "10"))
        axis.set_rlabel_position(135)
        axis.set_title(
            f"{plane_name}\nphi={positive_phi:.0f} / {negative_phi:.0f} deg"
        )
        axis.legend(loc="lower right")
    figure.suptitle(
        f"2.45 GHz directivity (dBi; below {floor_dbi:.0f} dBi clipped)\n"
        f"gprMax peak {peak:.2f} dBi; CST peak {cst_peak:.2f} dBi"
    )
    figure.savefig(FARFIELD_DIR / "patch_farfield_2p45GHz.png", dpi=180)
    plt.close(figure)
    return peak, peak_theta, peak_phi, cst_peak, cst_peak_theta, cst_peak_phi


def main() -> None:
    S_PARAMETER_DIR.mkdir(parents=True, exist_ok=True)
    FARFIELD_DIR.mkdir(parents=True, exist_ok=True)
    frequency, s11_db = read_s11()
    cst_frequency, cst_s11_db = read_cst_s11()
    far_field = read_far_field()
    cst_far_field = read_cst_far_field()
    minimum_frequency, minimum_s11, cst_minimum_frequency, cst_minimum_s11 = plot_s11(
        frequency, s11_db, cst_frequency, cst_s11_db
    )
    (
        peak_directivity,
        peak_theta,
        peak_phi,
        cst_peak_directivity,
        cst_peak_theta,
        cst_peak_phi,
    ) = plot_far_field(far_field, cst_far_field)
    csv_path = export_far_field(far_field)

    summary = {
        "s11_minimum_frequency_hz": minimum_frequency,
        "s11_minimum_db": minimum_s11,
        "cst_s11_minimum_frequency_hz": cst_minimum_frequency,
        "cst_s11_minimum_db": cst_minimum_s11,
        "minimum_frequency_difference_hz": minimum_frequency - cst_minimum_frequency,
        "minimum_frequency_difference_percent": 100.0 * (minimum_frequency - cst_minimum_frequency) / cst_minimum_frequency,
        "minimum_depth_difference_db": minimum_s11 - cst_minimum_s11,
        "far_field_frequency_hz": DESIGN_FREQUENCY_HZ,
        "theta_step_deg": 1.0,
        "phi_step_deg": 1.0,
        "far_field_samples": int(far_field["theta"].size),
        "peak_directivity_dbi": peak_directivity,
        "peak_theta_deg": peak_theta,
        "peak_phi_deg": peak_phi,
        "cst_peak_directivity_dbi": cst_peak_directivity,
        "cst_peak_theta_deg": cst_peak_theta,
        "cst_peak_phi_deg": cst_peak_phi,
        "peak_directivity_difference_db": peak_directivity - cst_peak_directivity,
        "peak_theta_difference_deg": peak_theta - cst_peak_theta,
        "peak_phi_difference_deg": peak_phi - cst_peak_phi,
        "radiation_efficiency": float(np.ravel(far_field["radiation_efficiency"])[0]),
        "total_efficiency": float(np.ravel(far_field["total_efficiency"])[0]),
        "far_field_csv": csv_path.name,
    }
    summary_path = HERE / "patch_results_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
