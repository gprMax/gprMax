"""Run an in-phase two-element dipole-array comparison model."""

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
OUTPUT_STEM = "dipole_array_gprmax"

DL = 1e-3
DOMAIN = (0.250, 0.250, 0.250)
TIME_WINDOW = 60e-9
PML_CELLS = 12

ELEMENT_LENGTH = 75e-3
ELEMENT_SPACING = 80e-3
ELEMENT_X = (0.085, 0.165)
WIRE_Y = 0.125
LOWER_END_Z = 0.088
FEED_Z = 0.125
UPPER_START_Z = FEED_Z + DL
UPPER_END_Z = 0.163
ARRAY_ORIGIN = (0.125, WIRE_Y, FEED_Z + 0.5 * DL)
WIRE_EFFECTIVE_RADIUS = 0.23 * DL

PORT_REFERENCE_IMPEDANCE = 50.0
SOURCE_AMPLITUDE = 1.0
SOURCE_FREQUENCY = 2e9
PATTERN_FREQUENCY = 1.9e9
PORT_FREQUENCY_MIN = 1.30e9
PORT_FREQUENCY_MAX = 2.50e9
INCIDENT_FLOOR_DB = -60.0
FFT_ZERO_PADDING = 8

KSIR_P1 = (0.025, 0.025, 0.025)
KSIR_P2 = (0.225, 0.225, 0.225)


def _add_port_receivers(scene, port_number, x):
    """Add the E edge and four H samples needed for one native Iz contour."""

    feed = (x, WIRE_Y, FEED_Z)
    scene.add(
        gprMax.Rx(
            p1=feed,
            id=f"port_{port_number}",
            outputs=["Ez", "Hx", "Hy"],
        )
    )
    scene.add(
        gprMax.Rx(
            p1=(x, WIRE_Y - DL, FEED_Z),
            id=f"port_{port_number}_hx_yminus",
            outputs=["Hx"],
        )
    )
    scene.add(
        gprMax.Rx(
            p1=(x - DL, WIRE_Y, FEED_Z),
            id=f"port_{port_number}_hy_xminus",
            outputs=["Hy"],
        )
    )


def build_scene():
    """Build two equally driven, z-directed thin-wire dipoles and KSIR cuts."""

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="In-phase two-element dipole array: MATLAB comparison"))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.PMLThickness(thickness=PML_CELLS))
    scene.add(
        gprMax.Waveform(
            wave_type="gaussian",
            amp=SOURCE_AMPLITUDE,
            freq=SOURCE_FREQUENCY,
            id="in_phase_pulse",
        )
    )

    for port_number, x in enumerate(ELEMENT_X, start=1):
        scene.add(
            gprMax.Edge(
                p1=(x, WIRE_Y, LOWER_END_Z),
                p2=(x, WIRE_Y, FEED_Z),
                material_id="pec",
            )
        )
        scene.add(
            gprMax.Edge(
                p1=(x, WIRE_Y, UPPER_START_Z),
                p2=(x, WIRE_Y, UPPER_END_Z),
                material_id="pec",
            )
        )
        scene.add(
            gprMax.VoltageSource(
                p1=(x, WIRE_Y, FEED_Z),
                polarisation="z",
                resistance=PORT_REFERENCE_IMPEDANCE,
                waveform_id="in_phase_pulse",
            )
        )
        _add_port_receivers(scene, port_number, x)

    surface = gprMax.NTFFSurface(
        p1=KSIR_P1,
        p2=KSIR_P2,
        id="array_surface",
        origin=ARRAY_ORIGIN,
    )
    transform = gprMax.KSIRFrequencyTransform(
        surface_id="array_surface",
        id="array_spectrum",
        frequencies=(PATTERN_FREQUENCY,),
        save_surface_dft=False,
    )
    angle = np.arange(-180.0, 180.0 + 2.0, 2.0)
    xz_cut = gprMax.KSIRFarField(
        theta=np.abs(angle),
        phi=np.where(angle < 0, 180.0, 0.0),
        transform_id="array_spectrum",
        id="xz_plane",
        outputs=("Etheta", "Ephi"),
    )
    yz_cut = gprMax.KSIRFarField(
        theta=np.abs(angle),
        phi=np.where(angle < 0, 270.0, 90.0),
        transform_id="array_spectrum",
        id="yz_plane",
        outputs=("Etheta", "Ephi"),
    )
    for item in (surface, transform, xz_cut, yz_cut):
        scene.add(item)

    scene.add(
        gprMax.GeometryView(
            p1=(ELEMENT_X[0] - 4 * DL, WIRE_Y - 4 * DL, LOWER_END_Z - 4 * DL),
            p2=(ELEMENT_X[1] + 4 * DL, WIRE_Y + 4 * DL, UPPER_END_Z + 4 * DL),
            dl=(DL, DL, DL),
            filename="dipole_array_geometry",
            output_type="f",
        )
    )
    return scene, angle


def _gaussian_source_voltage(time):
    """Evaluate the unit-amplitude gprMax Gaussian generator waveform."""

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


def _read_port_histories(h5, port_number, dx, dy, dz):
    """Read and collocate one port's voltage and Ampere-contour current."""

    root = f"port_{port_number}"
    ez = _read_receiver_output(h5, root, "Ez")
    hx = _read_receiver_output(h5, root, "Hx")
    hy = _read_receiver_output(h5, root, "Hy")
    hx_yminus = _read_receiver_output(h5, f"{root}_hx_yminus", "Hx")
    hy_xminus = _read_receiver_output(h5, f"{root}_hy_xminus", "Hy")
    histories = (ez, hx, hy, hx_yminus, hy_xminus)
    if ez.ndim != 1 or any(values.shape != ez.shape for values in histories):
        raise ValueError(f"Port {port_number} histories have inconsistent shapes")
    if not all(np.all(np.isfinite(values)) for values in histories):
        raise ValueError(f"Port {port_number} history contains non-finite values")

    edge_voltage = -dz * ez
    edge_current = dx * (hx_yminus - hx) + dy * (hy - hy_xminus)
    return 0.5 * (edge_voltage[:-1] + edge_voltage[1:]), edge_current[1:]


def calculate_ports_hdf5(h5_path):
    """Calculate active impedances and reflection coefficients for both ports."""

    with h5py.File(h5_path, "r") as h5:
        dt = float(h5.attrs["dt"])
        dx, dy, dz = (float(value) for value in h5.attrs["dx_dy_dz"])
        histories = tuple(_read_port_histories(h5, port, dx, dy, dz) for port in (1, 2))

    sample_count = histories[0][0].size
    if any(values.size != sample_count for pair in histories for values in pair):
        raise ValueError("The two ports have inconsistent history lengths")
    half_step_time = (np.arange(sample_count) + 0.5) * dt
    incident_voltage = 0.5 * _gaussian_source_voltage(half_step_time)
    n_fft = FFT_ZERO_PADDING * sample_count
    frequency = np.fft.rfftfreq(n_fft, d=dt)
    incident_spectrum = dt * np.fft.rfft(incident_voltage, n=n_fft)
    incident_magnitude = np.abs(incident_spectrum)
    relative_db = 20 * np.log10(
        np.maximum(
            incident_magnitude / np.max(incident_magnitude),
            np.finfo(float).tiny,
        )
    )
    selected = (
        (frequency >= PORT_FREQUENCY_MIN)
        & (frequency <= PORT_FREQUENCY_MAX)
        & (relative_db >= INCIDENT_FLOOR_DB)
    )
    if not np.any(selected):
        raise RuntimeError("No active-port frequencies remain after filtering")

    ports = []
    for voltage, current in histories:
        voltage_spectrum = dt * np.fft.rfft(voltage, n=n_fft)
        current_spectrum = dt * np.fft.rfft(current, n=n_fft)
        selected_current = current_spectrum[selected]
        current_floor = np.finfo(float).eps * np.max(np.abs(current_spectrum))
        if np.any(np.abs(selected_current) <= current_floor):
            raise RuntimeError("A port current is too small for stable impedance")
        impedance = voltage_spectrum[selected] / selected_current
        reflected = voltage_spectrum[selected] - incident_spectrum[selected]
        ports.append(
            {
                "voltage": voltage_spectrum[selected],
                "current": selected_current,
                "impedance": impedance,
                "voltage_gamma": reflected / incident_spectrum[selected],
                "impedance_gamma": (impedance - PORT_REFERENCE_IMPEDANCE)
                / (impedance + PORT_REFERENCE_IMPEDANCE),
            }
        )

    return {
        "frequency": frequency[selected],
        "incident_relative_db": relative_db[selected],
        "incident_spectrum": incident_spectrum[selected],
        "ports": ports,
    }


def write_ports_from_hdf5(h5_path):
    """Write both active ports and their symmetry averages to CSV."""

    result = calculate_ports_hdf5(h5_path)
    output = RESULTS_DIR / f"{OUTPUT_STEM}_active_ports.csv"
    headings = ["frequency_hz", "incident_relative_db"]
    per_port = (
        "voltage_spectrum_real",
        "voltage_spectrum_imag",
        "current_spectrum_real",
        "current_spectrum_imag",
        "active_impedance_real_ohm",
        "active_impedance_imag_ohm",
        "voltage_gamma_real",
        "voltage_gamma_imag",
        "impedance_gamma_real",
        "impedance_gamma_imag",
    )
    for port in (1, 2):
        headings.extend(f"port_{port}_{name}" for name in per_port)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headings)
        for index, frequency in enumerate(result["frequency"]):
            row = [frequency, result["incident_relative_db"][index]]
            for port in result["ports"]:
                voltage = port["voltage"][index]
                current = port["current"][index]
                impedance = port["impedance"][index]
                voltage_gamma = port["voltage_gamma"][index]
                impedance_gamma = port["impedance_gamma"][index]
                row.extend(
                    (
                        voltage.real,
                        voltage.imag,
                        current.real,
                        current.imag,
                        impedance.real,
                        impedance.imag,
                        voltage_gamma.real,
                        voltage_gamma.imag,
                        impedance_gamma.real,
                        impedance_gamma.imag,
                    )
                )
            writer.writerow(row)
    return output


def _read_complex_field(h5, path, number_of_angles):
    """Read and validate a single-frequency KSIR field."""

    values = np.asarray(h5[path])
    if values.shape != (1, number_of_angles):
        raise ValueError(f"HDF5 dataset {path!r} has unexpected shape {values.shape}")
    values = values[0]
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        raise ValueError(f"HDF5 dataset {path!r} contains non-finite values")
    return values


def write_pattern_from_hdf5(angle, h5_path):
    """Read persisted KSIR cuts and write normalised complex fields to CSV."""

    root = "ntff/array_surface/frequency/array_spectrum"
    with h5py.File(h5_path, "r") as h5:
        frequency = np.asarray(h5[f"{root}/frequencies"])
        if frequency.shape != (1,) or not np.allclose(frequency, (PATTERN_FREQUENCY,), rtol=1e-12):
            raise ValueError("The stored KSIR frequency is not the requested value")
        expected = {
            f"{root}/far_field/xz_plane/theta": np.abs(angle),
            f"{root}/far_field/xz_plane/phi": np.where(angle < 0, 180.0, 0.0),
            f"{root}/far_field/yz_plane/theta": np.abs(angle),
            f"{root}/far_field/yz_plane/phi": np.where(angle < 0, 270.0, 90.0),
        }
        for path, requested in expected.items():
            values = np.asarray(h5[path])
            if values.shape != requested.shape or not np.allclose(
                values, requested, rtol=1e-7, atol=1e-7
            ):
                raise ValueError(f"Stored KSIR coordinates do not match {path!r}")
        xz_co = _read_complex_field(h5, f"{root}/far_field/xz_plane/fields/Etheta", angle.size)
        xz_cross = _read_complex_field(h5, f"{root}/far_field/xz_plane/fields/Ephi", angle.size)
        yz_co = _read_complex_field(h5, f"{root}/far_field/yz_plane/fields/Etheta", angle.size)
        yz_cross = _read_complex_field(h5, f"{root}/far_field/yz_plane/fields/Ephi", angle.size)

    peak = max(np.max(np.abs(xz_co)), np.max(np.abs(yz_co)))
    if not np.isfinite(peak) or peak <= 0:
        raise RuntimeError("The KSIR result has no finite non-zero co-polar field")
    floor = np.finfo(float).tiny
    output = RESULTS_DIR / f"{OUTPUT_STEM}_pattern.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "angle_deg",
                "xz_co_real",
                "xz_co_imag",
                "xz_co_normalized_db",
                "xz_cross_normalized_db",
                "yz_co_real",
                "yz_co_imag",
                "yz_co_normalized_db",
                "yz_cross_normalized_db",
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
                    yz_co[index].real,
                    yz_co[index].imag,
                    20 * np.log10(max(abs(yz_co[index]) / peak, floor)),
                    20 * np.log10(max(abs(yz_cross[index]) / peak, floor)),
                )
            )

    metadata = {
        "discretisation_m": (DL, DL, DL),
        "domain_m": DOMAIN,
        "time_window_s": TIME_WINDOW,
        "element_length_m": ELEMENT_LENGTH,
        "element_spacing_m": ELEMENT_SPACING,
        "wire_effective_radius_m": WIRE_EFFECTIVE_RADIUS,
        "port_reference_impedance_ohm": PORT_REFERENCE_IMPEDANCE,
        "excitation": "equal-amplitude, zero-relative-phase voltage sources",
        "pattern_frequency_hz": PATTERN_FREQUENCY,
        "port_fft_zero_padding_factor": FFT_ZERO_PADDING,
        "port_independent_frequency_resolution_hz": 1 / TIME_WINDOW,
        "port_frequency_range_hz": (PORT_FREQUENCY_MIN, PORT_FREQUENCY_MAX),
        "ksir_surface_m": {"p1": KSIR_P1, "p2": KSIR_P2},
        "postprocessing_source": h5_path.name,
    }
    (RESULTS_DIR / f"{OUTPUT_STEM}_model.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main():
    """Run the model or rebuild its CSV products from the persisted HDF5 file."""

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
        port_output = write_ports_from_hdf5(h5_path)
        print(f"Checked KSIR far fields in {h5_path}")
        print(f"Saved HDF5-derived pattern to {pattern_output}")
        print(f"Saved HDF5-derived active ports to {port_output}")


if __name__ == "__main__":
    main()
