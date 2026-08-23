"""Run the triangular bow-tie antenna comparison model."""

import argparse
import csv
import json
import logging
from pathlib import Path

import h5py
import numpy as np

import gprMax

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_STEM = "bowtie_antenna_gprmax"

DL = 1e-3
DOMAIN = (0.200, 0.200, 0.100)
TIME_WINDOW = 80e-9
PML_CELLS = 12

FEED_X = 0.100
FEED_Y = 0.100
FEED_Z = 0.050
FEED_GAP = DL
WING_LENGTH = 50e-3
OUTER_WIDTH = 100e-3
OUTER_LENGTH = 2 * WING_LENGTH + FEED_GAP
CENTRE = (FEED_X + 0.5 * FEED_GAP, FEED_Y, FEED_Z)
LEFT_X = FEED_X - WING_LENGTH
RIGHT_X = FEED_X + FEED_GAP + WING_LENGTH
LOWER_Y = FEED_Y - OUTER_WIDTH / 2
UPPER_Y = FEED_Y + OUTER_WIDTH / 2

MATLAB_FLARE_ANGLE = 2 * np.rad2deg(np.arctan(OUTER_WIDTH / OUTER_LENGTH))
PORT_REFERENCE_IMPEDANCE = 50.0
SOURCE_AMPLITUDE = 1.0
SOURCE_FREQUENCY = 1e9
PATTERN_FREQUENCY = 0.82e9
S11_FREQUENCY_MIN = 0.45e9
S11_FREQUENCY_MAX = 1.20e9
S11_INCIDENT_FLOOR_DB = -60.0
S11_FFT_ZERO_PADDING = 8

KSIR_P1 = (0.020, 0.020, 0.020)
KSIR_P2 = (0.180, 0.180, 0.080)


def build_scene():
    """Build the planar bow tie, feed-current contour, and KSIR outputs."""

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="Triangular bow-tie antenna: MATLAB comparison"))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.PMLThickness(thickness=PML_CELLS))
    scene.add(
        gprMax.Waveform(
            wave_type="gaussian",
            amp=SOURCE_AMPLITUDE,
            freq=SOURCE_FREQUENCY,
            id="pulse",
        )
    )

    # Keep the x-directed driven edge strictly between the two triangle apex
    # nodes. This corrects the legacy example, whose right wing began on the
    # source's first node and could assign PEC to the driven Yee edge.
    scene.add(
        gprMax.Triangle(
            p1=(FEED_X, FEED_Y, FEED_Z),
            p2=(LEFT_X, LOWER_Y, FEED_Z),
            p3=(LEFT_X, UPPER_Y, FEED_Z),
            thickness=0,
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Triangle(
            p1=(FEED_X + FEED_GAP, FEED_Y, FEED_Z),
            p2=(RIGHT_X, LOWER_Y, FEED_Z),
            p3=(RIGHT_X, UPPER_Y, FEED_Z),
            thickness=0,
            material_id="pec",
        )
    )
    # The triangle rasteriser deliberately tests cell centres. At this acute
    # apex its first retained x edge therefore begins one cell beyond p1.
    # Join that edge to the source's right-hand node explicitly; otherwise the
    # right wing is separated from the driven edge by one free-space cell.
    scene.add(
        gprMax.Edge(
            p1=(FEED_X + FEED_GAP, FEED_Y, FEED_Z),
            p2=(FEED_X + 2 * FEED_GAP, FEED_Y, FEED_Z),
            material_id="pec",
        )
    )

    feed_position = (FEED_X, FEED_Y, FEED_Z)
    scene.add(
        gprMax.VoltageSource(
            p1=feed_position,
            polarisation="x",
            resistance=PORT_REFERENCE_IMPEDANCE,
            waveform_id="pulse",
        )
    )

    # Reconstruct native Ix on device backends from its four H samples:
    # Ix = dy [Hy(i,j,k-1) - Hy(i,j,k)]
    #    + dz [Hz(i,j,k) - Hz(i,j-1,k)].
    scene.add(
        gprMax.Rx(
            p1=feed_position,
            id="feed_port",
            outputs=["Ex", "Hy", "Hz"],
        )
    )
    scene.add(
        gprMax.Rx(
            p1=(FEED_X, FEED_Y, FEED_Z - DL),
            id="feed_hy_zminus",
            outputs=["Hy"],
        )
    )
    scene.add(
        gprMax.Rx(
            p1=(FEED_X, FEED_Y - DL, FEED_Z),
            id="feed_hz_yminus",
            outputs=["Hz"],
        )
    )

    surface = gprMax.NTFFSurface(
        p1=KSIR_P1,
        p2=KSIR_P2,
        id="bowtie_surface",
        origin=CENTRE,
    )
    transform = gprMax.KSIRFrequencyTransform(
        surface_id="bowtie_surface",
        id="bowtie_spectrum",
        frequencies=(PATTERN_FREQUENCY,),
        save_surface_dft=False,
    )
    angle = np.arange(-180.0, 180.0 + 2.0, 2.0)
    xz_cut = gprMax.KSIRFarField(
        theta=np.abs(angle),
        phi=np.where(angle < 0, 180.0, 0.0),
        transform_id="bowtie_spectrum",
        id="xz_plane",
        outputs=("Etheta", "Ephi"),
    )
    xy_cut = gprMax.KSIRFarField(
        theta=np.full(angle.shape, 90.0),
        phi=angle,
        transform_id="bowtie_spectrum",
        id="xy_plane",
        outputs=("Ephi", "Etheta"),
    )
    for item in (surface, transform, xz_cut, xy_cut):
        scene.add(item)

    scene.add(
        gprMax.GeometryView(
            p1=(LEFT_X - 2 * DL, LOWER_Y - 2 * DL, FEED_Z - 2 * DL),
            p2=(RIGHT_X + 2 * DL, UPPER_Y + 2 * DL, FEED_Z + 2 * DL),
            dl=(DL, DL, DL),
            filename="bowtie_antenna_geometry",
            output_type="f",
        )
    )
    return scene, angle


def _gaussian_source_voltage(time):
    """Evaluate the unit-amplitude gprMax Gaussian source waveform."""

    delay = time - 1 / SOURCE_FREQUENCY
    return SOURCE_AMPLITUDE * np.exp(-2 * np.pi**2 * SOURCE_FREQUENCY**2 * delay**2)


def _read_receiver_output(h5, receiver_name, output_name):
    """Return one named receiver output from an open gprMax HDF5 file."""

    for receiver in h5["rxs"].values():
        name = receiver.attrs.get("Name", "")
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        if name == receiver_name:
            if output_name not in receiver:
                raise RuntimeError(f"Receiver {receiver_name!r} has no {output_name!r} output")
            return np.asarray(receiver[output_name], dtype=np.float64)
    raise RuntimeError(f"The HDF5 output has no receiver named {receiver_name!r}")


def calculate_port_hdf5(h5_path):
    """Calculate input impedance and two independent S11 estimates."""

    with h5py.File(h5_path, "r") as h5:
        dt = float(h5.attrs["dt"])
        dx, dy, dz = (float(value) for value in h5.attrs["dx_dy_dz"])
        edge_field = _read_receiver_output(h5, "feed_port", "Ex")
        hy = _read_receiver_output(h5, "feed_port", "Hy")
        hz = _read_receiver_output(h5, "feed_port", "Hz")
        hy_zminus = _read_receiver_output(h5, "feed_hy_zminus", "Hy")
        hz_yminus = _read_receiver_output(h5, "feed_hz_yminus", "Hz")

    magnetic_histories = (hy, hz, hy_zminus, hz_yminus)
    if edge_field.ndim != 1 or not np.all(np.isfinite(edge_field)):
        raise ValueError("The feed-port Ex history is invalid")
    if any(values.shape != edge_field.shape for values in magnetic_histories):
        raise ValueError("The feed-port E and H histories have inconsistent shapes")
    if not all(np.all(np.isfinite(values)) for values in magnetic_histories):
        raise ValueError("A feed-current magnetic-field history is invalid")

    edge_voltage = -dx * edge_field
    edge_current = dy * (hy_zminus - hy) + dz * (hz - hz_yminus)
    total_voltage = 0.5 * (edge_voltage[:-1] + edge_voltage[1:])
    total_current = edge_current[1:]
    half_step_time = (np.arange(total_voltage.size) + 0.5) * dt
    generator_voltage = _gaussian_source_voltage(half_step_time)
    incident_voltage = 0.5 * generator_voltage
    reflected_voltage = total_voltage - incident_voltage

    n_fft = S11_FFT_ZERO_PADDING * total_voltage.size
    frequency = np.fft.rfftfreq(n_fft, d=dt)
    total_spectrum = dt * np.fft.rfft(total_voltage, n=n_fft)
    current_spectrum = dt * np.fft.rfft(total_current, n=n_fft)
    incident_spectrum = dt * np.fft.rfft(incident_voltage, n=n_fft)
    reflected_spectrum = dt * np.fft.rfft(reflected_voltage, n=n_fft)
    incident_magnitude = np.abs(incident_spectrum)
    relative_db = 20 * np.log10(
        np.maximum(
            incident_magnitude / np.max(incident_magnitude),
            np.finfo(float).tiny,
        )
    )
    selected = (
        (frequency >= S11_FREQUENCY_MIN)
        & (frequency <= S11_FREQUENCY_MAX)
        & (relative_db >= S11_INCIDENT_FLOOR_DB)
    )
    if not np.any(selected):
        raise RuntimeError("No port frequencies remain after filtering")

    selected_current = current_spectrum[selected]
    current_floor = np.finfo(float).eps * np.max(np.abs(current_spectrum))
    if np.any(np.abs(selected_current) <= current_floor):
        raise RuntimeError("The feed current is too small for a stable impedance")
    input_impedance = total_spectrum[selected] / selected_current
    impedance_s11 = (input_impedance - PORT_REFERENCE_IMPEDANCE) / (
        input_impedance + PORT_REFERENCE_IMPEDANCE
    )

    return {
        "frequency": frequency[selected],
        "incident_relative_db": relative_db[selected],
        "total_spectrum": total_spectrum[selected],
        "current_spectrum": selected_current,
        "input_impedance": input_impedance,
        "incident_spectrum": incident_spectrum[selected],
        "reflected_spectrum": reflected_spectrum[selected],
        "voltage_s11": reflected_spectrum[selected] / incident_spectrum[selected],
        "impedance_s11": impedance_s11,
    }


def write_port_from_hdf5(h5_path):
    """Write voltage, current, impedance, and both S11 estimates to CSV."""

    result = calculate_port_hdf5(h5_path)
    output = RESULTS_DIR / f"{OUTPUT_STEM}_s11.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "frequency_hz",
                "incident_relative_db",
                "total_voltage_spectrum_real",
                "total_voltage_spectrum_imag",
                "total_current_spectrum_real",
                "total_current_spectrum_imag",
                "input_impedance_real_ohm",
                "input_impedance_imag_ohm",
                "incident_voltage_spectrum_real",
                "incident_voltage_spectrum_imag",
                "reflected_voltage_spectrum_real",
                "reflected_voltage_spectrum_imag",
                "s11_real",
                "s11_imag",
                "s11_magnitude_db",
                "s11_phase_deg",
                "return_loss_db",
                "impedance_s11_real",
                "impedance_s11_imag",
                "impedance_s11_magnitude_db",
                "impedance_s11_phase_deg",
            )
        )
        for index, frequency in enumerate(result["frequency"]):
            total = result["total_spectrum"][index]
            current = result["current_spectrum"][index]
            impedance = result["input_impedance"][index]
            incident = result["incident_spectrum"][index]
            reflected = result["reflected_spectrum"][index]
            s11 = result["voltage_s11"][index]
            impedance_s11 = result["impedance_s11"][index]
            magnitude_db = 20 * np.log10(max(abs(s11), np.finfo(float).tiny))
            impedance_magnitude_db = 20 * np.log10(max(abs(impedance_s11), np.finfo(float).tiny))
            writer.writerow(
                (
                    frequency,
                    result["incident_relative_db"][index],
                    total.real,
                    total.imag,
                    current.real,
                    current.imag,
                    impedance.real,
                    impedance.imag,
                    incident.real,
                    incident.imag,
                    reflected.real,
                    reflected.imag,
                    s11.real,
                    s11.imag,
                    magnitude_db,
                    np.angle(s11, deg=True),
                    -magnitude_db,
                    impedance_s11.real,
                    impedance_s11.imag,
                    impedance_magnitude_db,
                    np.angle(impedance_s11, deg=True),
                )
            )
    return output


def _read_complex_field(h5, path, number_of_angles):
    """Read one single-frequency KSIR field and validate its shape."""

    values = np.asarray(h5[path])
    if values.shape != (1, number_of_angles):
        raise ValueError(
            f"HDF5 dataset {path!r} has shape {values.shape}; " f"expected {(1, number_of_angles)}"
        )
    values = values[0]
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        raise ValueError(f"HDF5 dataset {path!r} contains non-finite values")
    return values


def read_pattern_hdf5(angle, h5_path):
    """Read and validate the persisted KSIR pattern cuts."""

    root = "ntff/bowtie_surface/frequency/bowtie_spectrum"
    with h5py.File(h5_path, "r") as h5:
        frequency = np.asarray(h5[f"{root}/frequencies"])
        if frequency.shape != (1,) or not np.allclose(frequency, (PATTERN_FREQUENCY,), rtol=1e-12):
            raise ValueError("The stored KSIR frequency is not the requested value")

        xz_root = f"{root}/far_field/xz_plane"
        xy_root = f"{root}/far_field/xy_plane"
        expected = {
            f"{xz_root}/theta": np.abs(angle),
            f"{xz_root}/phi": np.where(angle < 0, 180.0, 0.0),
            f"{xy_root}/theta": np.full(angle.shape, 90.0),
            f"{xy_root}/phi": angle,
        }
        for path, requested in expected.items():
            values = np.asarray(h5[path])
            if values.shape != requested.shape or not np.allclose(
                values, requested, rtol=1e-7, atol=1e-7
            ):
                raise ValueError(f"Stored KSIR coordinates do not match {path!r}")

        return (
            _read_complex_field(h5, f"{xz_root}/fields/Etheta", angle.size),
            _read_complex_field(h5, f"{xz_root}/fields/Ephi", angle.size),
            _read_complex_field(h5, f"{xy_root}/fields/Ephi", angle.size),
            _read_complex_field(h5, f"{xy_root}/fields/Etheta", angle.size),
        )


def write_pattern_from_hdf5(angle, h5_path):
    """Write normalised complex KSIR principal-plane fields to CSV."""

    xz_co, xz_cross, xy_co, xy_cross = read_pattern_hdf5(angle, h5_path)
    peak = max(np.max(np.abs(xz_co)), np.max(np.abs(xy_co)))
    if not np.isfinite(peak) or peak <= 0:
        raise RuntimeError("The KSIR result has no finite non-zero co-polar field")

    output = RESULTS_DIR / f"{OUTPUT_STEM}_pattern.csv"
    floor = np.finfo(float).tiny
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "angle_deg",
                "xz_co_real",
                "xz_co_imag",
                "xz_co_normalized_db",
                "xz_cross_normalized_db",
                "xy_co_real",
                "xy_co_imag",
                "xy_co_normalized_db",
                "xy_cross_normalized_db",
            )
        )
        for index, value in enumerate(angle):
            writer.writerow(
                (
                    value,
                    xz_co[index].real,
                    xz_co[index].imag,
                    20 * np.log10(max(abs(xz_co[index]) / peak, floor)),
                    20 * np.log10(max(abs(xz_cross[index]) / peak, floor)),
                    xy_co[index].real,
                    xy_co[index].imag,
                    20 * np.log10(max(abs(xy_co[index]) / peak, floor)),
                    20 * np.log10(max(abs(xy_cross[index]) / peak, floor)),
                )
            )

    metadata = {
        "discretisation_m": (DL, DL, DL),
        "domain_m": DOMAIN,
        "time_window_s": TIME_WINDOW,
        "outer_length_m": OUTER_LENGTH,
        "outer_width_m": OUTER_WIDTH,
        "wing_length_m": WING_LENGTH,
        "feed_gap_m": FEED_GAP,
        "matlab_flare_angle_deg": MATLAB_FLARE_ANGLE,
        "port_reference_impedance_ohm": PORT_REFERENCE_IMPEDANCE,
        "pattern_frequency_hz": PATTERN_FREQUENCY,
        "s11_fft_zero_padding_factor": S11_FFT_ZERO_PADDING,
        "s11_independent_frequency_resolution_hz": 1 / TIME_WINDOW,
        "s11_frequency_range_hz": (S11_FREQUENCY_MIN, S11_FREQUENCY_MAX),
        "ksir_surface_m": {"p1": KSIR_P1, "p2": KSIR_P2},
        "postprocessing_source": h5_path.name,
    }
    (RESULTS_DIR / f"{OUTPUT_STEM}_model.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for CPU")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--geometry-only",
        action="store_true",
        help="build only and write the ParaView geometry",
    )
    mode.add_argument(
        "--postprocess-only",
        action="store_true",
        help="regenerate CSV files from an existing HDF5 output",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_base = RESULTS_DIR / OUTPUT_STEM
    angle = np.arange(-180.0, 180.0 + 2.0, 2.0)
    if not args.postprocess_only:
        scene, angle = build_scene()
        options = {}
        if args.gpu is not None:
            options["gpu"] = [args.gpu]
            options["gpu_precision"] = "single"
        gprMax.run(
            scenes=[scene],
            outputfile=output_base,
            geometry_only=args.geometry_only,
            hide_progress_bars=False,
            log_level=logging.INFO,
            **options,
        )
    if not args.geometry_only:
        h5_path = output_base.with_suffix(".h5")
        pattern_output = write_pattern_from_hdf5(angle, h5_path)
        port_output = write_port_from_hdf5(h5_path)
        print(f"Checked KSIR far fields in {h5_path}")
        print(f"Saved HDF5-derived pattern to {pattern_output}")
        print(f"Saved HDF5-derived impedance and S11 to {port_output}")


if __name__ == "__main__":
    main()
