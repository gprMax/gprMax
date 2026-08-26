"""Run strip-dipole benchmarks at two heights above infinite PEC ground."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

import gprMax


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

DL = 1.5e-3
DOMAIN = (0.288, 0.288, 0.210)
CENTRE = (0.144, 0.144)
GROUND = 0.030
PML_CELLS = 12
TIME_WINDOW = 100e-9

DIPOLE_LENGTH = 0.150
DIPOLE_WIDTH = 0.015
FEED_GAP = DL
LAMBDA_8_SPACING = 0.0375
LAMBDA_4_SPACING = 0.075

SOURCE_FREQUENCY = 1e9
PATTERN_FREQUENCY = 1e9
REFERENCE_IMPEDANCE = 50.0
S11_FREQUENCIES = np.arange(0.5e9, 1.5e9 + 0.025e9, 0.05e9)
THETA_CUT = np.arange(0.0, 89.0 + 1.0, 1.0)
FULL_ANGLE_STEP = 2.0

SURFACE_P1 = (0.030, 0.030, GROUND)
SURFACE_P2 = (0.258, 0.258, 0.180)


@dataclass(frozen=True)
class Case:
    name: str
    spacing: float

    @property
    def dipole_z(self) -> float:
        return GROUND + self.spacing


CASES = (
    Case("lambda_8", LAMBDA_8_SPACING),
    Case("lambda_4", LAMBDA_4_SPACING),
)


def _case(name: str) -> Case:
    return next(case for case in CASES if case.name == name)


def build_scene(case: Case) -> gprMax.Scene:
    """Build one MATLAB-reflector-equivalent gprMax model."""

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=f"MATLAB reflector comparison: {case.name}"))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.PMLThickness(thickness=PML_CELLS))

    # A volumetric PEC region below the terminal plane makes every electric
    # edge on z=GROUND rigid while the five-face NTFF omits that plane.
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(DOMAIN[0], DOMAIN[1], GROUND),
            material_id="pec",
        )
    )

    left = CENTRE[0] - 0.5 * DIPOLE_LENGTH
    feed = CENTRE[0]
    right = CENTRE[0] + 0.5 * DIPOLE_LENGTH
    y0 = CENTRE[1] - 0.5 * DIPOLE_WIDTH
    y1 = CENTRE[1] + 0.5 * DIPOLE_WIDTH
    scene.add(
        gprMax.Plate(
            p1=(left, y0, case.dipole_z),
            p2=(feed, y1, case.dipole_z),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Plate(
            p1=(feed + FEED_GAP, y0, case.dipole_z),
            p2=(right, y1, case.dipole_z),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Waveform(
            wave_type="gaussian",
            amp=1,
            freq=SOURCE_FREQUENCY,
            id="pulse",
        )
    )
    scene.add(
        gprMax.VoltageSource(
            p1=(feed, CENTRE[1], case.dipole_z),
            polarisation="x",
            resistance=REFERENCE_IMPEDANCE,
            waveform_id="pulse",
            id="feed",
            spectrum_limit="nyquist",
        )
    )

    scene.add(
        gprMax.NTFFSurface(
            p1=SURFACE_P1,
            p2=SURFACE_P2,
            id="surface",
            origin=(CENTRE[0], CENTRE[1], case.dipole_z),
            omit_faces=("z0",),
        )
    )
    scene.add(
        gprMax.NTFFLayeredBackground(
            id="grounded_air",
            axis="z",
            materials=("free_space", "pec"),
            interfaces=(GROUND,),
        )
    )
    scene.add(
        gprMax.NTFFLayeredFrequencyTransform(
            surface_id="surface",
            id="spectrum",
            background_id="grounded_air",
            frequencies=(PATTERN_FREQUENCY,),
            window="rectangular",
            save_surface_dft=False,
        )
    )
    scene.add(gprMax.NTFFAntennaPorts(transform_id="spectrum", port_ids=("feed",)))
    for output_id, phi in (("e_plane", 0.0), ("h_plane", 90.0)):
        scene.add(
            gprMax.NTFFFarField(
                theta=THETA_CUT,
                phi=np.full_like(THETA_CUT, phi),
                transform_id="spectrum",
                id=output_id,
                outputs=("Etheta", "Ephi", "directivity_dbi"),
            )
        )
    scene.add(
        gprMax.NTFFFarFieldArray(
            theta_start=0,
            theta_stop=88,
            theta_step=FULL_ANGLE_STEP,
            phi_start=0,
            phi_stop=360 - FULL_ANGLE_STEP,
            phi_step=FULL_ANGLE_STEP,
            transform_id="spectrum",
            id="upper_hemisphere",
            outputs=(
                "Etheta",
                "Ephi",
                "radiation_intensity",
                "directivity_dbi",
                "gain_dbi",
                "realized_gain_dbi",
                "radiation_efficiency",
                "total_efficiency",
            ),
        )
    )
    scene.add(
        gprMax.GeometryView(
            p1=(0, 0, 0),
            p2=DOMAIN,
            dl=(DL, DL, DL),
            filename=str(RESULTS / f"reflector_dipole_{case.name}_geometry"),
            output_type="n",
        )
    )
    return scene


def _interpolate_complex(x, values, targets):
    return np.interp(targets, x, values.real) + 1j * np.interp(targets, x, values.imag)


def export_result(case: Case, output_path: Path, runtime_seconds: float | None) -> dict:
    """Export compact, stable tables from one gprMax HDF5 result."""

    with h5py.File(output_path, "r") as output:
        far = output["ntff/surface/frequency/spectrum/far_field"]
        for plane in ("e_plane", "h_plane"):
            group = far[plane]
            etheta = np.asarray(group["fields/Etheta"])[0]
            ephi = np.asarray(group["fields/Ephi"])[0]
            directivity = np.asarray(group["fields/directivity_dbi"])[0]
            with (RESULTS / f"reflector_dipole_gprmax_{case.name}_{plane}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    (
                        "theta_deg",
                        "directivity_dbi",
                        "Etheta_real",
                        "Etheta_imag",
                        "Ephi_real",
                        "Ephi_imag",
                    )
                )
                for index, theta in enumerate(THETA_CUT):
                    writer.writerow(
                        (
                            theta,
                            directivity[index],
                            etheta[index].real,
                            etheta[index].imag,
                            ephi[index].real,
                            ephi[index].imag,
                        )
                    )

        upper = far["upper_hemisphere"]
        fields = upper["fields"]
        summary = {
            "runtime_seconds": runtime_seconds,
            "maximum_directivity_dbi": float(upper["maximum_directivity_dbi"][0]),
            "maximum_directivity_theta_deg": float(upper["maximum_directivity_theta"][0]),
            "maximum_directivity_phi_deg": float(upper["maximum_directivity_phi"][0]),
            "maximum_gain_dbi": float(np.nanmax(fields["gain_dbi"][0])),
            "maximum_realized_gain_dbi": float(np.nanmax(fields["realized_gain_dbi"][0])),
            "radiation_efficiency": float(fields["radiation_efficiency"][0]),
            "total_efficiency": float(fields["total_efficiency"][0]),
        }

        port = output["ports/feed"]
        frequency = np.asarray(port["frequency"], dtype=float)
        valid = (
            np.asarray(port["valid_S11"], dtype=bool)
            & np.asarray(port["valid_Zin"], dtype=bool)
            & np.isfinite(np.asarray(port["S11"]))
            & np.isfinite(np.asarray(port["Zin"]))
        )
        if np.count_nonzero(valid) < 2:
            raise RuntimeError(f"{case.name} has fewer than two valid port frequencies")
        source_frequency = frequency[valid]
        s11 = _interpolate_complex(source_frequency, np.asarray(port["S11"])[valid], S11_FREQUENCIES)
        zin = _interpolate_complex(source_frequency, np.asarray(port["Zin"])[valid], S11_FREQUENCIES)
        with (RESULTS / f"reflector_dipole_gprmax_{case.name}_port.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "frequency_hz",
                    "Zin_real_ohm",
                    "Zin_imag_ohm",
                    "S11_real",
                    "S11_imag",
                    "S11_magnitude_db",
                    "S11_phase_deg",
                )
            )
            for index, requested in enumerate(S11_FREQUENCIES):
                writer.writerow(
                    (
                        requested,
                        zin[index].real,
                        zin[index].imag,
                        s11[index].real,
                        s11[index].imag,
                        20 * np.log10(max(abs(s11[index]), np.finfo(float).tiny)),
                        np.angle(s11[index], deg=True),
                    )
                )
        pattern_index = int(np.argmin(np.abs(S11_FREQUENCIES - PATTERN_FREQUENCY)))
        summary.update(
            input_impedance_real_ohm=float(zin[pattern_index].real),
            input_impedance_imag_ohm=float(zin[pattern_index].imag),
            s11_magnitude_db=float(20 * np.log10(abs(s11[pattern_index]))),
            s11_phase_deg=float(np.angle(s11[pattern_index], deg=True)),
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case.name for case in CASES], action="append")
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for double-precision CPU")
    parser.add_argument("--no-run", action="store_true", help="export existing HDF5 files")
    parser.add_argument("--double", action="store_true", help="use double precision on CUDA")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    selected = tuple(case for case in CASES if args.case is None or case.name in args.case)
    summary_path = RESULTS / "reflector_dipole_gprmax_summary.json"
    previous_cases = {}
    if args.no_run and summary_path.exists():
        previous_cases = json.loads(summary_path.read_text()).get("cases", {})
    summaries = {}
    for case in selected:
        output_path = RESULTS / f"reflector_dipole_gprmax_{case.name}.h5"
        runtime = previous_cases.get(case.name, {}).get("runtime_seconds")
        if not args.no_run:
            options = {"cpu_precision": "double"}
            if args.gpu is not None:
                options = {
                    "gpu": [args.gpu],
                    "gpu_precision": "double" if args.double else "single",
                }
            started = time.perf_counter()
            gprMax.run(
                scenes=[build_scene(case)],
                n=1,
                outputfile=output_path.with_suffix(""),
                hide_progress_bars=True,
                **options,
            )
            runtime = time.perf_counter() - started
        summaries[case.name] = export_result(case, output_path, runtime)

    report = {
        "model": {
            "dl_metres": DL,
            "domain_metres": DOMAIN,
            "dipole_length_metres": DIPOLE_LENGTH,
            "dipole_width_metres": DIPOLE_WIDTH,
            "feed_gap_metres": FEED_GAP,
            "lambda_8_spacing_metres": LAMBDA_8_SPACING,
            "lambda_4_spacing_metres": LAMBDA_4_SPACING,
            "pattern_frequency_hz": PATTERN_FREQUENCY,
        },
        "cases": summaries,
    }
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
