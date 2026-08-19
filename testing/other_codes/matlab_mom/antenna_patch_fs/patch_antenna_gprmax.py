"""Run the gprMax model of the Antenna Toolbox rectangular patch case."""

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
FREQUENCY = 2.37e9
SOURCE_AMPLITUDE = 1.0
TIME_WINDOW = 60e-9
S11_FREQUENCY_MIN = 1.5e9
S11_FREQUENCY_MAX = 3.2e9
S11_INCIDENT_FLOOR_DB = -60.0
S11_FFT_ZERO_PADDING = 8

SUBSTRATE_HEIGHT = 1.57e-3
STANDARD_DX = 0.5e-3
STANDARD_DY = 0.5e-3
STANDARD_DZ = SUBSTRATE_HEIGHT / 3
DX = STANDARD_DX
DY = STANDARD_DY
DZ = STANDARD_DZ
DL = (DX, DY, DZ)

DOMAIN = (0.120, 0.100, 120 * STANDARD_DZ)
CENTRE = (DOMAIN[0] / 2, DOMAIN[1] / 2)

GROUND_LENGTH = 80e-3
GROUND_WIDTH = 60e-3
PATCH_LENGTH = 40e-3
PATCH_WIDTH = 30e-3
FEED_OFFSET = 5.5e-3
DISTRIBUTED_FEED_EDGE_OFFSETS = (-0.5e-3, 0.0, 0.5e-3)
PORT_REFERENCE_IMPEDANCE = 50.0
FRILL_WIRE_RADIUS = 0.23e-3

GROUND_Z = 40 * STANDARD_DZ
PATCH_Z = GROUND_Z + SUBSTRATE_HEIGHT
KSIR_P1 = (12.5e-3, 12.5e-3, 25 * STANDARD_DZ)
KSIR_P2 = (107.5e-3, 87.5e-3, 75 * STANDARD_DZ)


def _configure_mesh(mesh_mode):
    """Set the active discretisation while preserving physical dimensions."""

    global DX, DY, DZ, DL
    refine_xy = mesh_mode == "fine-xyz"
    refine_z = mesh_mode in ("fine-z", "fine-xyz")
    DX = STANDARD_DX / (2 if refine_xy else 1)
    DY = STANDARD_DY / (2 if refine_xy else 1)
    DZ = SUBSTRATE_HEIGHT / (6 if refine_z else 3)
    DL = (DX, DY, DZ)


def _rectangle_bounds(length, width):
    """Return centred x-y bounds for a rectangular antenna layer."""

    return (
        CENTRE[0] - length / 2,
        CENTRE[1] - width / 2,
        CENTRE[0] + length / 2,
        CENTRE[1] + width / 2,
    )


def _feed_edges(feed_mode):
    """Return identifiers and positions for the selected feed."""

    feed_x = CENTRE[0] + FEED_OFFSET
    feed_y = CENTRE[1]
    if feed_mode == "series":
        source_cells = round(SUBSTRATE_HEIGHT / DZ)
        return tuple(
            (
                f"feed_port_series_{z_index}",
                (feed_x, feed_y, GROUND_Z + z_index * DZ),
            )
            for z_index in range(source_cells)
        )
    offsets = DISTRIBUTED_FEED_EDGE_OFFSETS if feed_mode == "distributed" else (0.0,)
    return tuple(
        (
            f"feed_port_{x_index}_{y_index}",
            (feed_x + x_offset, feed_y + y_offset, GROUND_Z),
        )
        for x_index, x_offset in enumerate(offsets)
        for y_index, y_offset in enumerate(offsets)
    )


def _feed_source_resistance(feed_mode):
    """Return each source resistance for a 50 Ohm equivalent port."""

    if feed_mode == "series":
        return PORT_REFERENCE_IMPEDANCE / len(_feed_edges(feed_mode))
    return PORT_REFERENCE_IMPEDANCE * len(_feed_edges(feed_mode))


def _feed_waveform_amplitude(feed_mode):
    """Return each generator voltage while retaining a 1 V modal source."""

    if feed_mode == "series":
        return SOURCE_AMPLITUDE / len(_feed_edges(feed_mode))
    return SOURCE_AMPLITUDE


def _output_stem(
    feed_mode,
    mesh_mode="standard",
    conductor_mode="plate",
    patch_trim_cells=0,
    board_trim_cells=0,
):
    """Return a stable output stem while retaining the original default names."""

    if feed_mode == "distributed":
        stem = "patch_antenna_gprmax"
    else:
        stem = f"patch_antenna_gprmax_{feed_mode}_feed"
    if mesh_mode != "standard":
        stem += f"_{mesh_mode.replace('-', '_')}"
    if conductor_mode == "box":
        stem += "_pec_box"
    if patch_trim_cells:
        stem += f"_patch_trim_{patch_trim_cells}"
    if board_trim_cells:
        stem += f"_board_trim_{board_trim_cells}"
    return stem


def build_scene(
    feed_mode="distributed",
    mesh_mode="standard",
    conductor_mode="plate",
    patch_trim_cells=0,
    board_trim_cells=0,
):
    """Build the patch, its KSIR principal-plane requests, and fine view."""

    scene = gprMax.Scene()
    scene.add(
        gprMax.Title(
            name=f"Rectangular patch antenna: MATLAB comparison "
            f"({feed_mode} feed, {mesh_mode} mesh, {conductor_mode} conductors, "
            f"{patch_trim_cells} patch and {board_trim_cells} board cell(s) "
            f"trimmed per end)"
        )
    )
    scene.add(gprMax.Discretisation(p1=DL))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    pml_thickness = {
        "standard": 12,
        "fine-z": (12, 12, 24, 12, 12, 24),
        "fine-xyz": 24,
    }[mesh_mode]
    scene.add(gprMax.PMLThickness(thickness=pml_thickness))

    scene.add(gprMax.Material(er=2.33, se=0, mr=1, sm=0, id="substrate"))
    scene.add(
        gprMax.Waveform(
            wave_type="gaussian",
            amp=_feed_waveform_amplitude(feed_mode),
            freq=FREQUENCY,
            id="pulse",
        )
    )

    ground_length = GROUND_LENGTH - 2 * board_trim_cells * DX
    ground_width = GROUND_WIDTH - 2 * board_trim_cells * DY
    if ground_length <= 0 or ground_width <= 0:
        raise ValueError("Board trimming must leave positive ground dimensions")
    gx0, gy0, gx1, gy1 = _rectangle_bounds(ground_length, ground_width)
    patch_length = PATCH_LENGTH - 2 * patch_trim_cells * DX
    if patch_length <= 0:
        raise ValueError("Patch trimming must leave a positive patch length")
    px0, py0, px1, py1 = _rectangle_bounds(patch_length, PATCH_WIDTH)

    # The dielectric must be built before the conductors so PEC remains the
    # final material assignment. For the one-cell-thick comparison, the ground
    # extends down and the patch extends up. Their substrate-facing surfaces
    # therefore stay at the same coordinates as the zero-thickness plates.
    scene.add(
        gprMax.Box(
            p1=(gx0, gy0, GROUND_Z),
            p2=(gx1, gy1, PATCH_Z),
            material_id="substrate",
        )
    )
    if conductor_mode == "plate":
        scene.add(
            gprMax.Plate(
                p1=(gx0, gy0, GROUND_Z),
                p2=(gx1, gy1, GROUND_Z),
                material_id="pec",
            )
        )
        scene.add(
            gprMax.Plate(
                p1=(px0, py0, PATCH_Z),
                p2=(px1, py1, PATCH_Z),
                material_id="pec",
            )
        )
    else:
        scene.add(
            gprMax.Box(
                p1=(gx0, gy0, GROUND_Z - DZ),
                p2=(gx1, gy1, GROUND_Z),
                material_id="pec",
            )
        )
        scene.add(
            gprMax.Box(
                p1=(px0, py0, PATCH_Z),
                p2=(px1, py1, PATCH_Z + DZ),
                material_id="pec",
            )
        )

    feed_x = CENTRE[0] + FEED_OFFSET
    feed_y = CENTRE[1]
    # A 1 mm square PEC post matches the Antenna Toolbox feed geometry. The
    # bottom substrate cell is left as a gap and driven by the voltage source.
    # The magnetic-frill case instead uses a physical subcell thin wire over
    # the full substrate height, including the feed edge at the ground plane.
    if feed_mode in ("distributed", "single"):
        scene.add(
            gprMax.Box(
                p1=(feed_x - 0.5e-3, feed_y - 0.5e-3, GROUND_Z + DZ),
                p2=(feed_x + 0.5e-3, feed_y + 0.5e-3, PATCH_Z),
                material_id="pec",
            )
        )
    # The distributed feed uses nine equal 450 Ohm sources in parallel. The
    # simpler alternative uses one central 50 Ohm source. The series experiment
    # replaces the PEC post with three 16.67 Ohm, one-third-voltage sources.
    # All represent a 50 Ohm modal port and retain total-field receiver(s).
    if feed_mode == "frill":
        if FRILL_WIRE_RADIUS >= 0.5 * min(DX, DY):
            raise ValueError(
                "The magnetic-frill thin-wire radius must be smaller than "
                "half the transverse cell size"
            )
        feed_position = (feed_x, feed_y, GROUND_Z)
        scene.add(
            gprMax.ThinWire(
                p1=feed_position,
                p2=(feed_x, feed_y, PATCH_Z),
                radius=FRILL_WIRE_RADIUS,
            )
        )
        scene.add(
            gprMax.MagneticFrillSource(
                p1=feed_position,
                polarisation="z",
                zcoax=PORT_REFERENCE_IMPEDANCE,
                waveform_id="pulse",
            )
        )
    else:
        source_resistance = _feed_source_resistance(feed_mode)
        for receiver_id, position in _feed_edges(feed_mode):
            scene.add(
                gprMax.VoltageSource(
                    p1=position,
                    polarisation="z",
                    resistance=source_resistance,
                    waveform_id="pulse",
                    id=receiver_id,
                )
            )
            # Ez on the driven edge is the total port field. Keeping these as
            # ordinary receivers also exercises the persisted HDF5 output path.
            scene.add(gprMax.Rx(p1=position, id=receiver_id, outputs=["Ez"]))

    surface = gprMax.NTFFSurface(
        p1=KSIR_P1,
        p2=KSIR_P2,
        id="patch_surface",
        origin=(CENTRE[0], CENTRE[1], PATCH_Z),
    )
    transform = gprMax.KSIRFrequencyTransform(
        surface_id="patch_surface",
        id="patch_spectrum",
        frequencies=(FREQUENCY,),
        save_surface_dft=False,
    )

    angle = np.arange(-180.0, 180.0 + 2.0, 2.0)
    theta = np.abs(angle)
    phi_xz = np.where(angle < 0, 180.0, 0.0)
    phi_yz = np.where(angle < 0, -90.0, 90.0)
    xz_cut = gprMax.KSIRFarField(
        theta=theta,
        phi=phi_xz,
        transform_id="patch_spectrum",
        id="xz_plane",
        outputs=("Etheta", "Ephi"),
    )
    yz_cut = gprMax.KSIRFarField(
        theta=theta,
        phi=phi_yz,
        transform_id="patch_spectrum",
        id="yz_plane",
        outputs=("Etheta", "Ephi"),
    )
    for item in (surface, transform, xz_cut, yz_cut):
        scene.add(item)

    # This deliberately uses the model discretisation. The resulting VTKHDF
    # is an UnstructuredGrid with one line cell for every Yee edge.
    scene.add(
        gprMax.GeometryView(
            p1=(gx0 - 1e-3, gy0 - 1e-3, GROUND_Z - DZ),
            p2=(gx1 + 1e-3, gy1 + 1e-3, PATCH_Z + DZ),
            dl=DL,
            filename=_output_stem(
                feed_mode,
                mesh_mode,
                conductor_mode,
                patch_trim_cells,
                board_trim_cells,
            ).replace("gprmax", "geometry", 1),
            output_type="f",
        )
    )
    return scene, angle


def _validate_hdf5_vector(h5, path, expected):
    """Read a vector and check its shape, values, and finiteness."""

    values = np.asarray(h5[path])
    expected = np.asarray(expected)
    if values.shape != expected.shape:
        raise ValueError(
            f"HDF5 dataset {path!r} has shape {values.shape}; expected {expected.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"HDF5 dataset {path!r} contains non-finite values")
    if not np.allclose(values, expected, rtol=1e-6, atol=1e-6):
        maximum_error = float(np.max(np.abs(values - expected)))
        raise ValueError(
            f"HDF5 dataset {path!r} does not match the requested values "
            f"(maximum absolute error {maximum_error:g})"
        )
    return values


def _read_hdf5_field(h5, path, number_of_angles):
    """Read and validate one single-frequency complex far-field component."""

    values = np.asarray(h5[path])
    expected_shape = (1, number_of_angles)
    if values.shape != expected_shape:
        raise ValueError(
            f"HDF5 dataset {path!r} has shape {values.shape}; expected {expected_shape}"
        )
    values = values[0]
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        raise ValueError(f"HDF5 dataset {path!r} contains non-finite complex values")
    return values


def _gaussian_source_voltage(time):
    """Evaluate the unit-amplitude gprMax Gaussian source waveform."""

    delay = time - 1 / FREQUENCY
    return SOURCE_AMPLITUDE * np.exp(-2 * np.pi**2 * FREQUENCY**2 * delay**2)


def _read_feed_edge_fields(h5, feed_mode):
    """Read all named feed-edge Ez histories, independent of rx ordering."""

    expected_ids = {receiver_id for receiver_id, _ in _feed_edges(feed_mode)}
    fields = {}
    try:
        receiver_group = h5["rxs"]
    except KeyError as exc:
        raise RuntimeError(
            "The HDF5 output has no feed receivers; rerun the updated patch model"
        ) from exc

    for receiver in receiver_group.values():
        name = receiver.attrs.get("Name", "")
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        if name in expected_ids:
            values = np.asarray(receiver["Ez"], dtype=np.float64)
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise ValueError(f"Receiver {name!r} has an invalid Ez history")
            fields[name] = values

    missing = sorted(expected_ids - fields.keys())
    if missing:
        raise RuntimeError(f"Missing feed receiver(s) in HDF5 output: {', '.join(missing)}")
    lengths = {values.size for values in fields.values()}
    if len(lengths) != 1:
        raise ValueError("Feed receiver histories do not all have the same length")
    return np.stack([fields[name] for name in sorted(fields)])


def calculate_frill_s11_hdf5(h5_path):
    """Calculate a zero-padded frill spectrum and verify the native HDF5 port."""

    with h5py.File(h5_path, "r") as h5:
        dt = float(h5.attrs["dt"])
        group = h5["frills/frill1"]
        nsamples = int(np.asarray(group["time"]).size)
        incident_voltage = np.asarray(group["Vinc"][:nsamples], dtype=np.float64)
        total_voltage = np.asarray(group["Vtotal"][:nsamples], dtype=np.float64)
        native_frequency = np.asarray(group["frequency"], dtype=np.float64)
        native_s11 = np.asarray(group["S11"], dtype=np.complex128)
        native_valid = np.asarray(group["valid_S11"], dtype=bool)

    if incident_voltage.shape != total_voltage.shape or incident_voltage.size < 2:
        raise ValueError("The magnetic-frill voltage histories are inconsistent")
    if not np.all(np.isfinite(incident_voltage)) or not np.all(np.isfinite(total_voltage)):
        raise ValueError("The magnetic-frill voltage histories are not finite")

    # Reproduce the automatic native-bin result before using the same histories
    # with zero padding. This checks the HDF5 terminal output independently of
    # the plotting grid used below.
    checked_frequency_full = np.fft.rfftfreq(nsamples, d=dt)
    incident_native = dt * np.fft.rfft(incident_voltage)
    reflected_native = dt * np.fft.rfft(total_voltage - incident_voltage)
    with np.errstate(divide="ignore", invalid="ignore"):
        checked_s11_full = reflected_native / incident_native
    stored_bins = native_frequency.size
    checked_frequency = checked_frequency_full[:stored_bins]
    checked_s11 = checked_s11_full[:stored_bins]
    if not np.allclose(checked_frequency, native_frequency, rtol=2e-6, atol=1e-3):
        raise ValueError("The stored magnetic-frill frequency grid is inconsistent")
    check = native_valid & np.isfinite(checked_s11)
    if not np.any(check) or not np.allclose(
        checked_s11[check], native_s11[check], rtol=2e-5, atol=2e-6
    ):
        raise ValueError("The stored magnetic-frill S11 failed independent verification")

    n_fft = S11_FFT_ZERO_PADDING * nsamples
    frequencies = np.fft.rfftfreq(n_fft, d=dt)
    total_spectrum = dt * np.fft.rfft(total_voltage, n=n_fft)
    incident_spectrum = dt * np.fft.rfft(incident_voltage, n=n_fft)
    reflected_spectrum = total_spectrum - incident_spectrum
    incident_magnitude = np.abs(incident_spectrum)
    incident_relative_db = 20 * np.log10(
        np.maximum(
            incident_magnitude / np.max(incident_magnitude),
            np.finfo(float).tiny,
        )
    )
    selected = (
        (frequencies >= S11_FREQUENCY_MIN)
        & (frequencies <= S11_FREQUENCY_MAX)
        & (incident_relative_db >= S11_INCIDENT_FLOOR_DB)
    )
    if not np.any(selected):
        raise RuntimeError("No frill frequencies remain after the spectrum filter")

    s11 = reflected_spectrum[selected] / incident_spectrum[selected]
    return {
        "frequency": frequencies[selected],
        "incident_relative_db": incident_relative_db[selected],
        "total_spectrum": total_spectrum[selected],
        "incident_spectrum": incident_spectrum[selected],
        "reflected_spectrum": reflected_spectrum[selected],
        "s11": s11,
        "edge_voltage_rms_nonuniformity": np.zeros(s11.shape, dtype=np.float64),
    }


def calculate_s11_hdf5(h5_path, feed_mode="distributed"):
    """Calculate the voltage-source port S11 from persisted feed-edge Ez."""

    if feed_mode == "frill":
        return calculate_frill_s11_hdf5(h5_path)

    with h5py.File(h5_path, "r") as h5:
        dt = float(h5.attrs["dt"])
        dz = float(h5.attrs["dx_dy_dz"][2])
        edge_fields = _read_feed_edge_fields(h5, feed_mode)

    # Electric fields are stored at integer time steps, while the resistive
    # source is injected at n+1/2. Averaging adjacent samples places the total
    # edge voltage at the same time as the Thevenin generator waveform.
    total_edge_voltage = -dz * edge_fields
    centred_edge_voltage = 0.5 * (total_edge_voltage[:, :-1] + total_edge_voltage[:, 1:])
    if feed_mode == "series":
        total_voltage = np.sum(centred_edge_voltage, axis=0)
    else:
        total_voltage = np.mean(centred_edge_voltage, axis=0)
    half_step_time = (np.arange(total_voltage.size) + 0.5) * dt

    # Identical Thevenin sources in parallel retain their generator voltage;
    # their resistances combine to the requested 50 Ohm reference impedance.
    generator_voltage = _gaussian_source_voltage(half_step_time)
    incident_voltage = 0.5 * generator_voltage
    reflected_voltage = total_voltage - incident_voltage

    # Zero padding samples the engineering-convention DTFT more finely around
    # the narrow patch resonance. It interpolates the transform but does not
    # change the independent 1 / TIME_WINDOW spectral resolution.
    n_fft = S11_FFT_ZERO_PADDING * total_voltage.size
    frequencies = np.fft.rfftfreq(n_fft, d=dt)
    edge_spectra = dt * np.fft.rfft(centred_edge_voltage, n=n_fft, axis=1)
    total_spectrum = dt * np.fft.rfft(total_voltage, n=n_fft)
    incident_spectrum = dt * np.fft.rfft(incident_voltage, n=n_fft)
    reflected_spectrum = dt * np.fft.rfft(reflected_voltage, n=n_fft)
    incident_magnitude = np.abs(incident_spectrum)
    incident_relative_db = 20 * np.log10(
        np.maximum(incident_magnitude / np.max(incident_magnitude), np.finfo(float).tiny)
    )
    selected = (
        (frequencies >= S11_FREQUENCY_MIN)
        & (frequencies <= S11_FREQUENCY_MAX)
        & (incident_relative_db >= S11_INCIDENT_FLOOR_DB)
    )
    if not np.any(selected):
        raise RuntimeError("No S11 frequencies remain after the incident-spectrum filter")

    s11 = reflected_spectrum[selected] / incident_spectrum[selected]
    expected_edge_spectrum = total_spectrum[selected]
    if feed_mode == "series":
        expected_edge_spectrum = expected_edge_spectrum / edge_spectra.shape[0]
    edge_deviation = edge_spectra[:, selected] - expected_edge_spectrum
    edge_nonuniformity = np.sqrt(np.mean(np.abs(edge_deviation) ** 2, axis=0)) / np.maximum(
        np.abs(expected_edge_spectrum), np.finfo(float).tiny
    )
    return {
        "frequency": frequencies[selected],
        "incident_relative_db": incident_relative_db[selected],
        "total_spectrum": total_spectrum[selected],
        "incident_spectrum": incident_spectrum[selected],
        "reflected_spectrum": reflected_spectrum[selected],
        "s11": s11,
        "edge_voltage_rms_nonuniformity": edge_nonuniformity,
    }


def write_s11_from_hdf5(
    h5_path,
    feed_mode="distributed",
    mesh_mode="standard",
    conductor_mode="plate",
    patch_trim_cells=0,
    board_trim_cells=0,
):
    """Write the complex HDF5-derived S11 spectrum to a portable CSV file."""

    result = calculate_s11_hdf5(h5_path, feed_mode)
    output = RESULTS_DIR / (
        f"{_output_stem(feed_mode, mesh_mode, conductor_mode, patch_trim_cells, board_trim_cells)}_s11.csv"
    )
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "frequency_hz",
                "incident_relative_db",
                "total_voltage_spectrum_real",
                "total_voltage_spectrum_imag",
                "incident_voltage_spectrum_real",
                "incident_voltage_spectrum_imag",
                "reflected_voltage_spectrum_real",
                "reflected_voltage_spectrum_imag",
                "s11_real",
                "s11_imag",
                "s11_magnitude_db",
                "s11_phase_deg",
                "return_loss_db",
                "edge_voltage_rms_nonuniformity",
            )
        )
        for index, frequency in enumerate(result["frequency"]):
            total = result["total_spectrum"][index]
            incident = result["incident_spectrum"][index]
            reflected = result["reflected_spectrum"][index]
            s11 = result["s11"][index]
            magnitude_db = 20 * np.log10(max(abs(s11), np.finfo(float).tiny))
            writer.writerow(
                (
                    frequency,
                    result["incident_relative_db"][index],
                    total.real,
                    total.imag,
                    incident.real,
                    incident.imag,
                    reflected.real,
                    reflected.imag,
                    s11.real,
                    s11.imag,
                    magnitude_db,
                    np.angle(s11, deg=True),
                    -magnitude_db,
                    result["edge_voltage_rms_nonuniformity"][index],
                )
            )
    return output


def read_pattern_hdf5(angle, h5_path):
    """Read and validate the persisted KSIR principal-plane far fields."""

    theta = np.abs(angle)
    phi_xz = np.where(angle < 0, 180.0, 0.0)
    phi_yz = np.where(angle < 0, -90.0, 90.0)
    transform_path = "ntff/patch_surface/frequency/patch_spectrum"
    xz_path = f"{transform_path}/far_field/xz_plane"
    yz_path = f"{transform_path}/far_field/yz_plane"

    try:
        with h5py.File(h5_path, "r") as h5:
            _validate_hdf5_vector(h5, f"{transform_path}/frequencies", np.asarray((FREQUENCY,)))
            _validate_hdf5_vector(h5, f"{xz_path}/theta", theta)
            _validate_hdf5_vector(h5, f"{xz_path}/phi", phi_xz)
            _validate_hdf5_vector(h5, f"{yz_path}/theta", theta)
            _validate_hdf5_vector(h5, f"{yz_path}/phi", phi_yz)
            xz_co = _read_hdf5_field(h5, f"{xz_path}/fields/Etheta", angle.size)
            xz_cross = _read_hdf5_field(h5, f"{xz_path}/fields/Ephi", angle.size)
            yz_co = _read_hdf5_field(h5, f"{yz_path}/fields/Ephi", angle.size)
            yz_cross = _read_hdf5_field(h5, f"{yz_path}/fields/Etheta", angle.size)
    except OSError as exc:
        raise RuntimeError(f"Could not read gprMax HDF5 output {h5_path}") from exc
    except KeyError as exc:
        raise RuntimeError(f"Required KSIR dataset is missing from {h5_path}: {exc}") from exc

    return xz_co, xz_cross, yz_co, yz_cross


def write_pattern_from_hdf5(
    angle,
    h5_path,
    feed_mode="distributed",
    mesh_mode="standard",
    conductor_mode="plate",
    patch_trim_cells=0,
    board_trim_cells=0,
):
    """Validate HDF5 far fields and write them to a portable CSV file."""

    xz_co, xz_cross, yz_co, yz_cross = read_pattern_hdf5(angle, h5_path)
    peak = max(np.max(np.abs(xz_co)), np.max(np.abs(yz_co)))
    if not np.isfinite(peak) or peak <= 0:
        raise RuntimeError("The gprMax far-field result has no finite non-zero co-polar field")

    output = RESULTS_DIR / (
        f"{_output_stem(feed_mode, mesh_mode, conductor_mode, patch_trim_cells, board_trim_cells)}_pattern.csv"
    )
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
        floor = np.finfo(float).tiny
        for i, value in enumerate(angle):
            writer.writerow(
                (
                    value,
                    xz_co[i].real,
                    xz_co[i].imag,
                    20 * np.log10(max(abs(xz_co[i]) / peak, floor)),
                    20 * np.log10(max(abs(xz_cross[i]) / peak, floor)),
                    yz_co[i].real,
                    yz_co[i].imag,
                    20 * np.log10(max(abs(yz_co[i]) / peak, floor)),
                    20 * np.log10(max(abs(yz_cross[i]) / peak, floor)),
                )
            )

    metadata = {
        "frequency_hz": FREQUENCY,
        "discretisation_m": DL,
        "mesh_mode": mesh_mode,
        "domain_m": DOMAIN,
        "time_window_s": TIME_WINDOW,
        "substrate_relative_permittivity": 2.33,
        "substrate_height_m": SUBSTRATE_HEIGHT,
        "patch_length_m": PATCH_LENGTH - 2 * patch_trim_cells * DX,
        "nominal_patch_length_m": PATCH_LENGTH,
        "patch_trim_cells_per_x_end": patch_trim_cells,
        "patch_width_m": PATCH_WIDTH,
        "ground_length_m": GROUND_LENGTH - 2 * board_trim_cells * DX,
        "ground_width_m": GROUND_WIDTH - 2 * board_trim_cells * DY,
        "nominal_ground_length_m": GROUND_LENGTH,
        "nominal_ground_width_m": GROUND_WIDTH,
        "board_trim_cells_per_edge": board_trim_cells,
        "feed_offset_m": FEED_OFFSET,
        "feed_side_m": 1e-3,
        "feed_mode": feed_mode,
        "conductor_mode": conductor_mode,
        "conductor_thickness_m": 0 if conductor_mode == "plate" else DZ,
        "feed_excitation": (
            "nine equal 450 Ohm voltage sources in parallel"
            if feed_mode == "distributed"
            else (
                "one central 50 Ohm voltage source"
                if feed_mode == "single"
                else (
                    "three one-third-voltage, 16.67 Ohm sources in series"
                    if feed_mode == "series"
                    else "50 Ohm magnetic frill on a physical subcell thin wire"
                )
            )
        ),
        "feed_source_count": 1 if feed_mode == "frill" else len(_feed_edges(feed_mode)),
        "feed_source_resistance_ohm": (
            None if feed_mode == "frill" else _feed_source_resistance(feed_mode)
        ),
        "feed_source_waveform_amplitude_v": _feed_waveform_amplitude(feed_mode),
        "feed_equivalent_resistance_ohm": PORT_REFERENCE_IMPEDANCE,
        "feed_thin_wire_radius_m": FRILL_WIRE_RADIUS if feed_mode == "frill" else None,
        "s11_incident_voltage": (
            "magnetic-frill Vinc history"
            if feed_mode == "frill"
            else "one half of the Thevenin generator waveform"
        ),
        "s11_fft_zero_padding_factor": S11_FFT_ZERO_PADDING,
        "s11_independent_frequency_resolution_hz": 1 / TIME_WINDOW,
        "s11_frequency_range_hz": (S11_FREQUENCY_MIN, S11_FREQUENCY_MAX),
        "pattern_postprocessing_source": h5_path.name,
        "ksir_surface_m": {
            "p1": KSIR_P1,
            "p2": KSIR_P2,
        },
    }
    model_output = RESULTS_DIR / (
        f"{_output_stem(feed_mode, mesh_mode, conductor_mode, patch_trim_cells, board_trim_cells)}_model.json"
    )
    model_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit to use the CPU solver")
    parser.add_argument(
        "--feed",
        choices=("distributed", "single", "series", "frill"),
        default="distributed",
        help="feed implementation; default: distributed",
    )
    parser.add_argument(
        "--mesh",
        choices=("standard", "fine-z", "fine-xyz"),
        default="standard",
        help=(
            "mesh resolution; fine-z refines only z, while fine-xyz also uses "
            "0.25 mm cells in x and y"
        ),
    )
    parser.add_argument(
        "--conductor",
        choices=("plate", "box"),
        default="plate",
        help=(
            "PEC representation; box uses one-cell thickness outside the "
            "substrate while preserving both substrate-facing surfaces"
        ),
    )
    parser.add_argument(
        "--patch-trim-cells",
        type=int,
        choices=range(0, 11),
        default=0,
        metavar="N",
        help="remove N x-directed mesh cells from each end of the patch",
    )
    parser.add_argument(
        "--board-trim-cells",
        type=int,
        choices=range(0, 11),
        default=0,
        metavar="N",
        help="remove N mesh cells from every x-y edge of the ground and dielectric",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--geometry-only",
        action="store_true",
        help="only build the model and write the fine ParaView geometry",
    )
    mode.add_argument(
        "--postprocess-only",
        action="store_true",
        help="validate the existing HDF5 output and regenerate the pattern CSV",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _configure_mesh(args.mesh)
    angle = np.arange(-180.0, 180.0 + 2.0, 2.0)
    output_base = RESULTS_DIR / _output_stem(
        args.feed,
        args.mesh,
        args.conductor,
        args.patch_trim_cells,
        args.board_trim_cells,
    )
    if not args.postprocess_only:
        scene, angle = build_scene(
            args.feed,
            args.mesh,
            args.conductor,
            args.patch_trim_cells,
            args.board_trim_cells,
        )
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
        output = write_pattern_from_hdf5(
            angle,
            h5_path,
            args.feed,
            args.mesh,
            args.conductor,
            args.patch_trim_cells,
            args.board_trim_cells,
        )
        s11_output = write_s11_from_hdf5(
            h5_path,
            args.feed,
            args.mesh,
            args.conductor,
            args.patch_trim_cells,
            args.board_trim_cells,
        )
        print(f"Checked KSIR far fields in {h5_path}")
        print(f"Saved HDF5-derived principal-plane pattern to {output}")
        print(f"Saved HDF5-derived complex S11 to {s11_output}")


if __name__ == "__main__":
    main()
