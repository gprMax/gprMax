"""Run the gprMax half-wave Yee-edge dipole comparison model."""

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
OUTPUT_STEM = "dipole_antenna_gprmax"

DL = 1e-3
DOMAIN = (0.250, 0.250, 0.250)
TIME_WINDOW = 100e-9
PML_CELLS = 12

WIRE_LENGTH = 151e-3
WIRE_EFFECTIVE_RADIUS = 0.23 * DL
WIRE_X = 0.125
WIRE_Y = 0.125
LOWER_END_Z = 0.050
FEED_Z = 0.125
UPPER_START_Z = FEED_Z + DL
UPPER_END_Z = 0.201
ORIGIN = (WIRE_X, WIRE_Y, FEED_Z + 0.5 * DL)

PORT_REFERENCE_IMPEDANCE = 73.0
SOURCE_AMPLITUDE = 1.0
SOURCE_FREQUENCY = 1e9
PATTERN_FREQUENCY = 0.95e9
S11_FREQUENCY_MIN = 0.55e9
S11_FREQUENCY_MAX = 1.35e9
S11_INCIDENT_FLOOR_DB = -60.0
S11_FFT_ZERO_PADDING = 8

KSIR_P1 = (0.025, 0.025, 0.025)
KSIR_P2 = (0.225, 0.225, 0.225)


def build_scene():
    """Build the centred thin-wire dipole and its reusable KSIR outputs."""

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="Half-wave Yee-edge dipole: MATLAB comparison"))
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

    # The two 75 mm PEC arms are separated by one z-directed Yee edge. The
    # source occupies that 1 mm feed gap, giving a symmetric 151 mm outer span.
    scene.add(
        gprMax.Edge(
            p1=(WIRE_X, WIRE_Y, LOWER_END_Z),
            p2=(WIRE_X, WIRE_Y, FEED_Z),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Edge(
            p1=(WIRE_X, WIRE_Y, UPPER_START_Z),
            p2=(WIRE_X, WIRE_Y, UPPER_END_Z),
            material_id="pec",
        )
    )
    feed_position = (WIRE_X, WIRE_Y, FEED_Z)
    scene.add(
        gprMax.VoltageSource(
            p1=feed_position,
            polarisation="z",
            resistance=PORT_REFERENCE_IMPEDANCE,
            waveform_id="pulse",
        )
    )
    scene.add(gprMax.RxPort(p1=feed_position, id="dipole_feed"))
    # Iz is the Ampere contour integral around this z-directed Yee edge. The
    # CPU solver can store Iz directly, but device receivers currently expose
    # only E and H. These three receivers save the four H samples needed to
    # reconstruct exactly the same quantity on CUDA, OpenCL, and Metal:
    #
    # Iz = dx [Hx(i,j-1,k) - Hx(i,j,k)]
    #    + dy [Hy(i,j,k) - Hy(i-1,j,k)].
    scene.add(
        gprMax.Rx(
            p1=feed_position,
            id="feed_port",
            outputs=["Ez", "Hx", "Hy"],
        )
    )
    scene.add(
        gprMax.Rx(
            p1=(WIRE_X, WIRE_Y - DL, FEED_Z),
            id="feed_hx_yminus",
            outputs=["Hx"],
        )
    )
    scene.add(
        gprMax.Rx(
            p1=(WIRE_X - DL, WIRE_Y, FEED_Z),
            id="feed_hy_xminus",
            outputs=["Hy"],
        )
    )

    surface = gprMax.NTFFSurface(
        p1=KSIR_P1,
        p2=KSIR_P2,
        id="dipole_surface",
        origin=ORIGIN,
    )
    transform = gprMax.KSIRFrequencyTransform(
        surface_id="dipole_surface",
        id="dipole_spectrum",
        frequencies=(PATTERN_FREQUENCY,),
        save_surface_dft=False,
    )
    antenna_ports = gprMax.KSIRAntennaPorts(
        "dipole_spectrum",
        ("dipole_feed",),
    )
    angle = np.arange(-180.0, 180.0 + 2.0, 2.0)
    xz_cut = gprMax.KSIRFarField(
        theta=np.abs(angle),
        phi=np.where(angle < 0, 180.0, 0.0),
        transform_id="dipole_spectrum",
        id="xz_plane",
        outputs=("Etheta", "Ephi"),
    )
    xy_cut = gprMax.KSIRFarField(
        theta=np.full(angle.shape, 90.0),
        phi=angle,
        transform_id="dipole_spectrum",
        id="xy_plane",
        outputs=("Etheta", "Ephi"),
    )
    metrics_3d = gprMax.KSIRFarFieldArray(
        theta_start=0,
        theta_stop=180,
        theta_step=2,
        phi_start=0,
        phi_stop=358,
        phi_step=2,
        transform_id="dipole_spectrum",
        id="metrics_3d",
        outputs=(
            "radiation_intensity",
            "directivity",
            "directivity_dbi",
            "gain",
            "gain_dbi",
            "realized_gain",
            "realized_gain_dbi",
            "radiation_efficiency",
            "total_efficiency",
        ),
    )
    for item in (surface, transform, antenna_ports, xz_cut, xy_cut, metrics_3d):
        scene.add(item)

    scene.add(
        gprMax.GeometryView(
            p1=(WIRE_X - 3 * DL, WIRE_Y - 3 * DL, LOWER_END_Z - 3 * DL),
            p2=(WIRE_X + 3 * DL, WIRE_Y + 3 * DL, UPPER_END_Z + 3 * DL),
            dl=(DL, DL, DL),
            filename="dipole_antenna_geometry",
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


def calculate_s11_hdf5(h5_path):
    """Calculate port impedance and S11 from persisted source-edge fields."""

    with h5py.File(h5_path, "r") as h5:
        dt = float(h5.attrs["dt"])
        dx, dy, dz = (float(value) for value in h5.attrs["dx_dy_dz"])
        edge_field = _read_receiver_output(h5, "feed_port", "Ez")
        hx = _read_receiver_output(h5, "feed_port", "Hx")
        hy = _read_receiver_output(h5, "feed_port", "Hy")
        hx_yminus = _read_receiver_output(h5, "feed_hx_yminus", "Hx")
        hy_xminus = _read_receiver_output(h5, "feed_hy_xminus", "Hy")

    if edge_field.ndim != 1 or not np.all(np.isfinite(edge_field)):
        raise ValueError("The feed-port Ez history is invalid")
    magnetic_histories = (hx, hy, hx_yminus, hy_xminus)
    if any(values.shape != edge_field.shape for values in magnetic_histories):
        raise ValueError("The feed-port E and H histories have inconsistent shapes")
    if not all(np.all(np.isfinite(values)) for values in magnetic_histories):
        raise ValueError("A feed-current magnetic-field history is invalid")

    edge_voltage = -dz * edge_field
    edge_current = dx * (hx_yminus - hx) + dy * (hy - hy_xminus)

    # At receiver sample n, E is at n*dt while H (and hence the Ampere current)
    # is at (n-1/2)*dt. Centre adjacent voltage samples on the H time grid and
    # discard the all-zero initial H sample.
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
        raise RuntimeError("No S11 frequencies remain after filtering")

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
        "s11": reflected_spectrum[selected] / incident_spectrum[selected],
        "impedance_s11": impedance_s11,
    }


def write_s11_from_hdf5(h5_path):
    """Write the complex HDF5-derived S11 spectrum to CSV."""

    result = calculate_s11_hdf5(h5_path)
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
            s11 = result["s11"][index]
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
    """Read and validate the persisted KSIR elevation and azimuth cuts."""

    root = "ntff/dipole_surface/frequency/dipole_spectrum"
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
            _read_complex_field(h5, f"{xy_root}/fields/Etheta", angle.size),
            _read_complex_field(h5, f"{xy_root}/fields/Ephi", angle.size),
        )


def write_pattern_from_hdf5(angle, h5_path):
    """Write normalised complex KSIR elevation and azimuth fields to CSV."""

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
        "wire_outer_length_m": WIRE_LENGTH,
        "wire_arm_length_m": 75e-3,
        "feed_gap_m": DL,
        "fdtd_equivalent_wire_radius_m": WIRE_EFFECTIVE_RADIUS,
        "fdtd_equivalent_radius_cells": 0.23,
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
        s11_output = write_s11_from_hdf5(h5_path)
        print(f"Checked KSIR far fields in {h5_path}")
        print(f"Saved HDF5-derived pattern to {pattern_output}")
        print(f"Saved HDF5-derived complex S11 to {s11_output}")


if __name__ == "__main__":
    main()
