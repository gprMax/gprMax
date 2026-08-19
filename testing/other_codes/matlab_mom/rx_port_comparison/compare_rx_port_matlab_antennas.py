"""Compare source-owned voltage-port output with MATLAB Antenna Toolbox models."""

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gprMax

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
MATLAB_RESULTS_ROOT = SCRIPT_DIR.parent

BOWTIE_DL = 1e-3
DIPOLE_DL = 2e-3
MONOPOLE_DL = 1e-3
PATCH_DX = 0.5e-3
PATCH_DY = 0.5e-3
PATCH_DZ = 1.57e-3 / 3
PML_CELLS = 12
OMP_THREADS = 20


@dataclass(frozen=True)
class AntennaCase:
    """Fixed model and MATLAB-comparison metadata for one antenna."""

    name: str
    label: str
    reference_impedance: float
    frequency_min: float
    frequency_max: float
    matlab_csv: Path
    discretisation: tuple
    time_window: float
    source_frequency: float
    model_characteristic_length: float
    matlab_characteristic_length: float

    @property
    def output_base(self):
        return RESULTS_DIR / f"rx_port_{self.name}_gprmax"


CASES = {
    "dipole": AntennaCase(
        name="dipole",
        label="151 mm cylindrical dipole",
        reference_impedance=73.0,
        frequency_min=0.55e9,
        frequency_max=1.35e9,
        matlab_csv=(
            MATLAB_RESULTS_ROOT / "antenna_dipole_fs" / "results" / "dipole_antenna_matlab_s11.csv"
        ),
        discretisation=(DIPOLE_DL,) * 3,
        time_window=6e-9,
        source_frequency=1e9,
        model_characteristic_length=0.150,
        matlab_characteristic_length=0.151,
    ),
    "bowtie": AntennaCase(
        name="bowtie",
        label="101 mm triangular bow-tie",
        reference_impedance=50.0,
        frequency_min=0.45e9,
        frequency_max=1.20e9,
        matlab_csv=(
            MATLAB_RESULTS_ROOT / "antenna_bowtie_fs" / "results" / "bowtie_antenna_matlab_s11.csv"
        ),
        discretisation=(BOWTIE_DL,) * 3,
        time_window=6e-9,
        source_frequency=1e9,
        model_characteristic_length=0.101,
        matlab_characteristic_length=0.101,
    ),
    "monopole": AntennaCase(
        name="monopole",
        label="79 mm finite-ground monopole",
        reference_impedance=36.5,
        frequency_min=0.55e9,
        frequency_max=1.30e9,
        matlab_csv=(
            MATLAB_RESULTS_ROOT
            / "antenna_monopole_fs"
            / "results"
            / "monopole_antenna_matlab_s11.csv"
        ),
        discretisation=(MONOPOLE_DL,) * 3,
        time_window=8e-9,
        source_frequency=1e9,
        model_characteristic_length=0.079,
        matlab_characteristic_length=0.079,
    ),
    "patch": AntennaCase(
        name="patch",
        label="40 by 30 mm substrate-backed patch",
        reference_impedance=50.0,
        frequency_min=1.50e9,
        frequency_max=3.20e9,
        matlab_csv=(
            MATLAB_RESULTS_ROOT / "antenna_patch_fs" / "results" / "patch_antenna_matlab_s11.csv"
        ),
        discretisation=(PATCH_DX, PATCH_DY, PATCH_DZ),
        time_window=15e-9,
        source_frequency=2.37e9,
        model_characteristic_length=0.040,
        matlab_characteristic_length=0.040,
    ),
}


def _common_scene(case, title, domain):
    """Create common model controls for one antenna case."""

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=title))
    scene.add(gprMax.Discretisation(p1=case.discretisation))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.TimeWindow(time=case.time_window))
    scene.add(gprMax.PMLThickness(thickness=PML_CELLS))
    scene.add(gprMax.OMPThreads(OMP_THREADS))
    scene.add(
        gprMax.Waveform(
            wave_type="gaussian",
            amp=1,
            freq=case.source_frequency,
            id="pulse",
        )
    )
    return scene


def build_dipole_scene():
    """Build the original 151 mm one-edge-feed dipole geometry."""

    dl = DIPOLE_DL
    case = CASES["dipole"]
    scene = _common_scene(
        case,
        "Voltage-port MATLAB comparison: 150 mm grid dipole",
        (0.180, 0.180, 0.220),
    )
    wire_x = 0.090
    wire_y = 0.090
    feed_z = 0.108
    lower_end_z = 0.034
    upper_start_z = feed_z + dl
    upper_end_z = 0.184
    feed = (wire_x, wire_y, feed_z)

    scene.add(
        gprMax.Edge(
            p1=(wire_x, wire_y, lower_end_z),
            p2=(wire_x, wire_y, feed_z),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Edge(
            p1=(wire_x, wire_y, upper_start_z),
            p2=(wire_x, wire_y, upper_end_z),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.VoltageSource(
            p1=feed,
            polarisation="z",
            resistance=case.reference_impedance,
            waveform_id="pulse",
            id="feed",
        )
    )
    return scene


def build_bowtie_scene():
    """Build the original 101 by 100 mm triangular bow-tie geometry."""

    dl = BOWTIE_DL
    case = CASES["bowtie"]
    scene = _common_scene(
        case,
        "Voltage-port MATLAB comparison: triangular bow-tie",
        (0.160, 0.160, 0.080),
    )
    feed_x = 0.080
    feed_y = 0.080
    feed_z = 0.040
    wing_length = 50e-3
    lower_y = feed_y - 50e-3
    upper_y = feed_y + 50e-3
    left_x = feed_x - wing_length
    right_apex_x = feed_x + dl
    right_x = right_apex_x + wing_length
    feed = (feed_x, feed_y, feed_z)

    scene.add(
        gprMax.Triangle(
            p1=feed,
            p2=(left_x, lower_y, feed_z),
            p3=(left_x, upper_y, feed_z),
            thickness=0,
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Triangle(
            p1=(right_apex_x, feed_y, feed_z),
            p2=(right_x, lower_y, feed_z),
            p3=(right_x, upper_y, feed_z),
            thickness=0,
            material_id="pec",
        )
    )
    # Join both triangle rasterisations explicitly to the feed nodes. At an
    # acute apex the cell-centre test can otherwise leave the driven edge
    # electrically open after translating the geometry on the grid.
    scene.add(
        gprMax.Edge(
            p1=(feed_x - dl, feed_y, feed_z),
            p2=(feed_x, feed_y, feed_z),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Edge(
            p1=(right_apex_x, feed_y, feed_z),
            p2=(right_apex_x + dl, feed_y, feed_z),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.VoltageSource(
            p1=feed,
            polarisation="x",
            resistance=case.reference_impedance,
            waveform_id="pulse",
            id="feed",
        )
    )
    return scene


def build_monopole_scene():
    """Build the MATLAB finite-ground quarter-wave monopole geometry."""

    case = CASES["monopole"]
    dl = MONOPOLE_DL
    scene = _common_scene(
        case,
        "Voltage-port MATLAB comparison: finite-ground monopole",
        (0.250, 0.250, 0.250),
    )
    wire_x = 0.125
    wire_y = 0.125
    ground_z = 0.075
    feed = (wire_x, wire_y, ground_z)
    scene.add(
        gprMax.Plate(
            p1=(0.045, 0.045, ground_z),
            p2=(0.205, 0.205, ground_z),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Edge(
            p1=(wire_x, wire_y, ground_z + dl),
            p2=(wire_x, wire_y, ground_z + 0.079),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.VoltageSource(
            p1=feed,
            polarisation="z",
            resistance=case.reference_impedance,
            waveform_id="pulse",
            id="feed",
        )
    )
    return scene


def build_patch_scene():
    """Build the single-edge-feed MATLAB substrate-backed patch geometry."""

    case = CASES["patch"]
    domain = (0.120, 0.100, 120 * PATCH_DZ)
    scene = _common_scene(
        case,
        "Voltage-port MATLAB comparison: single-feed rectangular patch",
        domain,
    )
    centre_x = domain[0] / 2
    centre_y = domain[1] / 2
    ground_z = 40 * PATCH_DZ
    patch_z = ground_z + 1.57e-3
    ground_p1 = (centre_x - 40e-3, centre_y - 30e-3, ground_z)
    ground_p2 = (centre_x + 40e-3, centre_y + 30e-3, patch_z)
    patch_p1 = (centre_x - 20e-3, centre_y - 15e-3, patch_z)
    patch_p2 = (centre_x + 20e-3, centre_y + 15e-3, patch_z)
    feed_x = centre_x + 5.5e-3
    feed_y = centre_y
    feed = (feed_x, feed_y, ground_z)

    scene.add(gprMax.Material(er=2.33, se=0, mr=1, sm=0, id="substrate"))
    scene.add(gprMax.Box(p1=ground_p1, p2=ground_p2, material_id="substrate"))
    scene.add(
        gprMax.Plate(
            p1=ground_p1,
            p2=(ground_p2[0], ground_p2[1], ground_z),
            material_id="pec",
        )
    )
    scene.add(gprMax.Plate(p1=patch_p1, p2=patch_p2, material_id="pec"))
    scene.add(
        gprMax.Box(
            p1=(feed_x - 0.5e-3, feed_y - 0.5e-3, ground_z + PATCH_DZ),
            p2=(feed_x + 0.5e-3, feed_y + 0.5e-3, patch_z),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.VoltageSource(
            p1=feed,
            polarisation="z",
            resistance=case.reference_impedance,
            waveform_id="pulse",
            id="feed",
        )
    )
    return scene


BUILDERS = {
    "dipole": build_dipole_scene,
    "bowtie": build_bowtie_scene,
    "monopole": build_monopole_scene,
    "patch": build_patch_scene,
}


def run_case(case, gpu=None):
    """Run one model using the production source-bound port output."""

    scene = BUILDERS[case.name]()
    options = {}
    if gpu is not None:
        options["gpu"] = [gpu]
        options["gpu_precision"] = "single"
    gprMax.run(
        scenes=[scene],
        outputfile=case.output_base,
        hide_progress_bars=True,
        log_level=logging.INFO,
        cpu_precision="single",
        **options,
    )


def _complex_columns(data, real_name, imaginary_name):
    return np.asarray(data[real_name] + 1j * data[imaginary_name])


def _interpolate_complex(x, source_x, source_values):
    """Linearly interpolate real and imaginary parts on an in-range axis."""

    return np.interp(x, source_x, source_values.real) + 1j * np.interp(
        x, source_x, source_values.imag
    )


def read_comparison(case):
    """Read production HDF5 port data and interpolate the MATLAB result."""

    h5_path = case.output_base.with_suffix(".h5")
    if not h5_path.is_file():
        raise FileNotFoundError(f"Run the {case.name} model first: {h5_path}")
    if not case.matlab_csv.is_file():
        raise FileNotFoundError(f"MATLAB comparison data is missing: {case.matlab_csv}")

    with h5py.File(h5_path, "r") as output:
        port = output["ports/feed"]
        frequency = np.asarray(port["frequency"], dtype=np.float64)
        s11 = np.asarray(port["S11"], dtype=np.complex128)
        s11_source = np.asarray(port["S11_source"], dtype=np.complex128)
        zin = np.asarray(port["Zin"], dtype=np.complex128)
        zin_source = np.asarray(port["Zin_source"], dtype=np.complex128)
        valid = np.asarray(port["valid_S11"], dtype=bool)
        valid_zin = np.asarray(port["valid_Zin"], dtype=bool)
        incident_relative_db = np.asarray(port["incident_relative_dB"], dtype=np.float64)
        tail_relative_db = float(port.attrs["TailRelativeLevelDB"])
        mesh_frequency_limit = float(port.attrs["MeshFrequencyLimit"])
        frequency_resolution = float(port.attrs["IndependentFrequencyResolution"])
        gap_capacitance = float(port.attrs["GapCapacitance"])

    selected = (
        valid
        & valid_zin
        & np.isfinite(s11)
        & np.isfinite(zin)
        & (frequency >= case.frequency_min)
        & (frequency <= case.frequency_max)
    )
    if np.count_nonzero(selected) < 5:
        raise RuntimeError(f"{case.name} has too few valid S11 bins in its comparison band")

    matlab = np.genfromtxt(case.matlab_csv, delimiter=",", names=True)
    matlab_frequency = np.asarray(matlab["frequency_hz"], dtype=np.float64)
    matlab_s11_all = _complex_columns(matlab, "s11_real", "s11_imag")
    matlab_zin_all = _complex_columns(
        matlab,
        "input_impedance_real_ohm",
        "input_impedance_imag_ohm",
    )

    frequency = frequency[selected]
    if frequency[0] < matlab_frequency[0] or frequency[-1] > matlab_frequency[-1]:
        raise RuntimeError(f"{case.name} port frequencies exceed its MATLAB comparison range")

    return {
        "case": case,
        "h5_path": h5_path,
        "frequency": frequency,
        "s11": s11[selected],
        "s11_source": s11_source[selected],
        "zin": zin[selected],
        "zin_source": zin_source[selected],
        "incident_relative_db": incident_relative_db[selected],
        "matlab_s11": _interpolate_complex(frequency, matlab_frequency, matlab_s11_all),
        "matlab_zin": _interpolate_complex(frequency, matlab_frequency, matlab_zin_all),
        "tail_relative_db": tail_relative_db,
        "mesh_frequency_limit": mesh_frequency_limit,
        "frequency_resolution": frequency_resolution,
        "gap_capacitance": gap_capacitance,
        "discretisation": case.discretisation,
    }


def _magnitude_db(values):
    return 20 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))


def _aligned_unwrapped_phase(reference, values):
    """Unwrap phase and choose the equivalent 360-degree reference branch."""

    reference_phase = np.rad2deg(np.unwrap(np.angle(reference)))
    phase = np.rad2deg(np.unwrap(np.angle(values)))
    centre = phase.size // 2
    phase += 360 * np.round((reference_phase[centre] - phase[centre]) / 360)
    return reference_phase, phase


def _minimum(frequency, s11):
    magnitude = _magnitude_db(s11)
    index = int(np.argmin(magnitude))
    return {
        "frequency_hz": float(frequency[index]),
        "magnitude_db": float(magnitude[index]),
    }


def calculate_metrics(result):
    """Calculate direct complex, magnitude, phase, and impedance errors."""

    matlab_phase, gprmax_phase = _aligned_unwrapped_phase(result["matlab_s11"], result["s11"])
    matlab_s11_db = _magnitude_db(result["matlab_s11"])
    gprmax_s11_db = _magnitude_db(result["s11"])
    source_s11_db = _magnitude_db(result["s11_source"])
    return {
        "frequency_points": int(result["frequency"].size),
        "frequency_range_hz": [
            float(result["frequency"][0]),
            float(result["frequency"][-1]),
        ],
        "frequency_resolution_hz": result["frequency_resolution"],
        "mesh_frequency_limit_hz": result["mesh_frequency_limit"],
        "tail_relative_level_db": result["tail_relative_db"],
        "gap_capacitance_f": result["gap_capacitance"],
        "discretisation_m": list(result["discretisation"]),
        "model_characteristic_length_m": result["case"].model_characteristic_length,
        "matlab_characteristic_length_m": result["case"].matlab_characteristic_length,
        "matlab_minimum": _minimum(result["frequency"], result["matlab_s11"]),
        "gprmax_corrected_minimum": _minimum(result["frequency"], result["s11"]),
        "gprmax_source_plane_minimum": _minimum(result["frequency"], result["s11_source"]),
        "corrected_s11_rms_complex": float(
            np.sqrt(np.mean(np.abs(result["s11"] - result["matlab_s11"]) ** 2))
        ),
        "source_plane_s11_rms_complex": float(
            np.sqrt(np.mean(np.abs(result["s11_source"] - result["matlab_s11"]) ** 2))
        ),
        "corrected_s11_rms_magnitude_db": float(
            np.sqrt(np.mean((gprmax_s11_db - matlab_s11_db) ** 2))
        ),
        "source_plane_s11_rms_magnitude_db": float(
            np.sqrt(np.mean((source_s11_db - matlab_s11_db) ** 2))
        ),
        "corrected_s11_rms_phase_deg": float(np.sqrt(np.mean((gprmax_phase - matlab_phase) ** 2))),
        "corrected_zin_rms_ohm": float(
            np.sqrt(np.mean(np.abs(result["zin"] - result["matlab_zin"]) ** 2))
        ),
    }


def write_csv(result):
    """Write the exact plotted native-bin values for independent inspection."""

    output = RESULTS_DIR / f"rx_port_{result['case'].name}_comparison.csv"
    headings = (
        "frequency_hz",
        "incident_relative_db",
        "gprmax_s11_real",
        "gprmax_s11_imag",
        "gprmax_source_s11_real",
        "gprmax_source_s11_imag",
        "matlab_s11_real",
        "matlab_s11_imag",
        "gprmax_zin_real_ohm",
        "gprmax_zin_imag_ohm",
        "gprmax_source_zin_real_ohm",
        "gprmax_source_zin_imag_ohm",
        "matlab_zin_real_ohm",
        "matlab_zin_imag_ohm",
    )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headings)
        for values in zip(
            result["frequency"],
            result["incident_relative_db"],
            result["s11"],
            result["s11_source"],
            result["matlab_s11"],
            result["zin"],
            result["zin_source"],
            result["matlab_zin"],
        ):
            (
                frequency,
                incident_db,
                s11,
                source_s11,
                matlab_s11,
                zin,
                source_zin,
                matlab_zin,
            ) = values
            writer.writerow(
                (
                    frequency,
                    incident_db,
                    s11.real,
                    s11.imag,
                    source_s11.real,
                    source_s11.imag,
                    matlab_s11.real,
                    matlab_s11.imag,
                    zin.real,
                    zin.imag,
                    source_zin.real,
                    source_zin.imag,
                    matlab_zin.real,
                    matlab_zin.imag,
                )
            )
    return output


def create_plots(results):
    """Create S11 magnitude/phase and input-impedance comparison figures."""

    colours = {"matlab": "black", "gprmax": "#0072B2", "source": "#D55E00"}
    figure_height = max(3.4 * len(results), 4.0)
    fig_s11, axes_s11 = plt.subplots(len(results), 2, figsize=(12, figure_height), sharex=False)
    fig_zin, axes_zin = plt.subplots(len(results), 2, figsize=(12, figure_height), sharex=False)
    axes_s11 = np.atleast_2d(axes_s11)
    axes_zin = np.atleast_2d(axes_zin)

    for row, result in enumerate(results):
        frequency_ghz = result["frequency"] / 1e9
        matlab_phase, gprmax_phase = _aligned_unwrapped_phase(result["matlab_s11"], result["s11"])
        _, source_phase = _aligned_unwrapped_phase(result["matlab_s11"], result["s11_source"])

        axes_s11[row, 0].plot(
            frequency_ghz,
            _magnitude_db(result["matlab_s11"]),
            color=colours["matlab"],
            linewidth=2.1,
            label="MATLAB MoM",
        )
        axes_s11[row, 0].plot(
            frequency_ghz,
            _magnitude_db(result["s11"]),
            color=colours["gprmax"],
            marker="o",
            markersize=3.5,
            label="gprMax voltage port",
        )
        axes_s11[row, 0].plot(
            frequency_ghz,
            _magnitude_db(result["s11_source"]),
            color=colours["source"],
            linestyle="--",
            linewidth=1.2,
            label="gprMax source plane",
        )
        axes_s11[row, 0].axhline(-10, color="0.55", linestyle=":", linewidth=1)
        axes_s11[row, 0].set_ylabel(r"$|S_{11}|$ [dB]")
        axes_s11[row, 0].set_title(result["case"].label)

        axes_s11[row, 1].plot(
            frequency_ghz,
            matlab_phase,
            color=colours["matlab"],
            linewidth=2.1,
            label="MATLAB MoM",
        )
        axes_s11[row, 1].plot(
            frequency_ghz,
            gprmax_phase,
            color=colours["gprmax"],
            marker="o",
            markersize=3.5,
            label="gprMax voltage port",
        )
        axes_s11[row, 1].plot(
            frequency_ghz,
            source_phase,
            color=colours["source"],
            linestyle="--",
            linewidth=1.2,
            label="gprMax source plane",
        )
        axes_s11[row, 1].set_ylabel(r"Phase($S_{11}$) [deg]")
        axes_s11[row, 1].set_title(result["case"].label)

        axes_zin[row, 0].plot(
            frequency_ghz,
            result["matlab_zin"].real,
            color=colours["matlab"],
            linewidth=2.1,
            label="MATLAB MoM",
        )
        axes_zin[row, 0].plot(
            frequency_ghz,
            result["zin"].real,
            color=colours["gprmax"],
            marker="o",
            markersize=3.5,
            label="gprMax from corrected S11",
        )
        axes_zin[row, 0].set_ylabel(r"Re($Z_\mathrm{in}$) [$\Omega$]")
        axes_zin[row, 0].set_title(result["case"].label)

        axes_zin[row, 1].plot(
            frequency_ghz,
            result["matlab_zin"].imag,
            color=colours["matlab"],
            linewidth=2.1,
            label="MATLAB MoM",
        )
        axes_zin[row, 1].plot(
            frequency_ghz,
            result["zin"].imag,
            color=colours["gprmax"],
            marker="o",
            markersize=3.5,
            label="gprMax from corrected S11",
        )
        axes_zin[row, 1].set_ylabel(r"Im($Z_\mathrm{in}$) [$\Omega$]")
        axes_zin[row, 1].set_title(result["case"].label)

        for axes in (axes_s11[row], axes_zin[row]):
            for axis in axes:
                axis.set_xlabel("Frequency [GHz]")
                axis.grid(True, alpha=0.25)
                axis.legend(fontsize=8)

    fig_s11.suptitle("gprMax voltage ports versus MATLAB Antenna Toolbox", fontsize=14)
    fig_s11.tight_layout()
    s11_path = RESULTS_DIR / "rx_port_matlab_comparison.png"
    fig_s11.savefig(s11_path, dpi=220, bbox_inches="tight")
    plt.close(fig_s11)

    fig_zin.suptitle("Input impedance derived from corrected S11", fontsize=14)
    fig_zin.tight_layout()
    zin_path = RESULTS_DIR / "rx_port_matlab_impedance_comparison.png"
    fig_zin.savefig(zin_path, dpi=220, bbox_inches="tight")
    plt.close(fig_zin)
    return s11_path, zin_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("all", *CASES),
        default="all",
        help="model(s) to run and compare",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="read existing production HDF5 port outputs without rerunning gprMax",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="CUDA device index; omit to use the CPU solver",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selected_cases = list(CASES.values()) if args.model == "all" else [CASES[args.model]]
    if not args.postprocess_only:
        for case in selected_cases:
            print(f"Running {case.label}...")
            run_case(case, gpu=args.gpu)

    results = [read_comparison(case) for case in selected_cases]
    metrics = {}
    for result in results:
        write_csv(result)
        metrics[result["case"].name] = calculate_metrics(result)
    metrics_path = RESULTS_DIR / "rx_port_matlab_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    s11_path, zin_path = create_plots(results)

    print(f"Saved S11 comparison: {s11_path}")
    print(f"Saved impedance comparison: {zin_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
