"""Run the gprMax/MATLAB three-turn axial-mode helix comparison model."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import gprMax

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_STEM = "helix_antenna_gprmax"

DL = 1e-3
DOMAIN = (0.260, 0.260, 0.250)
TIME_WINDOW = 30e-9
PML_CELLS = 12

CENTRE_X = 0.130
CENTRE_Y = 0.130
GROUND_Z = 0.050
GROUND_RADIUS = 75e-3
HELIX_RADIUS = 22e-3
HELIX_TURNS = 3
HELIX_SPACING = 35e-3
HELIX_SEGMENT_ANGLE_DEG = 15.0
FEED_STUB_HEIGHT = 2 * DL
FDTD_EFFECTIVE_WIRE_RADIUS = 0.23 * DL
HELIX_WIRE_RADIUS = DL
MATLAB_EQUIVALENT_STRIP_WIDTH = 4 * HELIX_WIRE_RADIUS

PATTERN_FREQUENCY = 2.2e9
SOURCE_FREQUENCY = PATTERN_FREQUENCY
PORT_REFERENCE_IMPEDANCE = 150.0
ANGULAR_STEP = 2.0

KSIR_P1 = (0.025, 0.025, 0.025)
KSIR_P2 = (0.235, 0.235, 0.225)
ORIGIN = (CENTRE_X, CENTRE_Y, GROUND_Z)


def helix_segments():
    """Return a connected piecewise-linear approximation of the helix."""

    # Fifteen-degree chords have less than 0.2 mm geometric sag at this radius.
    # After endpoints are rounded to the one-millimetre grid, their total path
    # is within 0.5 percent of the analytical helical centreline length.
    samples = int(HELIX_TURNS * 360 / HELIX_SEGMENT_ANGLE_DEG) + 1
    parameter = np.linspace(0, 2 * np.pi * HELIX_TURNS, samples)
    coordinates = np.column_stack(
        (
            CENTRE_X + HELIX_RADIUS * np.cos(parameter),
            CENTRE_Y + HELIX_RADIUS * np.sin(parameter),
            GROUND_Z + FEED_STUB_HEIGHT + HELIX_SPACING * parameter / (2 * np.pi),
        )
    )
    nodes = np.rint(coordinates / DL) * DL
    unique_nodes = [nodes[0]]
    unique_nodes.extend(node for node in nodes[1:] if not np.array_equal(node, unique_nodes[-1]))
    return [(tuple(start), tuple(stop)) for start, stop in zip(unique_nodes[:-1], unique_nodes[1:])]


def build_scene(source_mode="resistive"):
    """Build the default MATLAB helix geometry and antenna outputs."""

    scene = gprMax.Scene()
    scene.add(
        gprMax.Title(name=f"Three-turn axial-mode helix: MATLAB comparison ({source_mode} feed)")
    )
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.PMLThickness(thickness=PML_CELLS))
    scene.add(
        gprMax.Waveform(
            wave_type="gaussian",
            amp=1.0,
            freq=SOURCE_FREQUENCY,
            id="pulse",
        )
    )

    # Approximate the 150 mm circular zero-thickness ground plane with planar
    # triangular PEC sectors. This preserves its radius and avoids replacing
    # it with a square reflector.
    sector_angles = np.deg2rad(np.arange(0.0, 360.0 + 5.0, 5.0))
    ground_centre = (CENTRE_X, CENTRE_Y, GROUND_Z)
    for angle1, angle2 in zip(sector_angles[:-1], sector_angles[1:]):
        point1 = (
            CENTRE_X + GROUND_RADIUS * np.cos(angle1),
            CENTRE_Y + GROUND_RADIUS * np.sin(angle1),
            GROUND_Z,
        )
        point2 = (
            CENTRE_X + GROUND_RADIUS * np.cos(angle2),
            CENTRE_Y + GROUND_RADIUS * np.sin(angle2),
            GROUND_Z,
        )
        scene.add(
            gprMax.Triangle(
                p1=ground_centre,
                p2=point1,
                p3=point2,
                thickness=0,
                material_id="pec",
            )
        )

    segments = helix_segments()
    # Keep the finite-radius first cylinder clear of the driven Yee edge. The
    # first millimetre is the source gap and the second is a PEC feed stub.
    scene.add(
        gprMax.Edge(
            p1=(CENTRE_X + HELIX_RADIUS, CENTRE_Y, GROUND_Z + DL),
            p2=(CENTRE_X + HELIX_RADIUS, CENTRE_Y, GROUND_Z + FEED_STUB_HEIGHT),
            material_id="pec",
        )
    )
    for start, stop in segments:
        scene.add(
            gprMax.Cylinder(
                p1=start,
                p2=stop,
                r=HELIX_WIRE_RADIUS,
                material_id="pec",
                averaging="n",
            )
        )

    feed_position = (
        CENTRE_X + HELIX_RADIUS,
        CENTRE_Y,
        GROUND_Z,
    )
    source_kwargs = {}
    if source_mode != "resistive":
        source_kwargs["reference_impedance"] = PORT_REFERENCE_IMPEDANCE
    scene.add(
        gprMax.VoltageSource(
            p1=feed_position,
            polarisation="z",
            resistance=(PORT_REFERENCE_IMPEDANCE if source_mode == "resistive" else 0),
            waveform_id="pulse",
            id="helix_feed",
            **source_kwargs,
        )
    )

    scene.add(
        gprMax.NTFFSurface(
            p1=KSIR_P1,
            p2=KSIR_P2,
            id="helix_surface",
            origin=ORIGIN,
        )
    )
    scene.add(
        gprMax.KSIRFrequencyTransform(
            surface_id="helix_surface",
            id="helix_spectrum",
            frequencies=(PATTERN_FREQUENCY,),
            save_surface_dft=False,
        )
    )
    scene.add(gprMax.KSIRAntennaPorts("helix_spectrum", ("helix_feed",)))
    scene.add(
        gprMax.KSIRFarFieldArray(
            theta_start=0,
            theta_stop=180,
            theta_step=ANGULAR_STEP,
            phi_start=0,
            phi_stop=360 - ANGULAR_STEP,
            phi_step=ANGULAR_STEP,
            transform_id="helix_spectrum",
            id="metrics_3d",
            outputs=(
                "Etheta",
                "Ephi",
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
    )
    scene.add(
        gprMax.GeometryView(
            p1=(0.050, 0.050, GROUND_Z),
            p2=(0.210, 0.210, GROUND_Z + FEED_STUB_HEIGHT + HELIX_TURNS * HELIX_SPACING),
            dl=(DL, DL, DL),
            filename="helix_antenna_geometry",
            output_type="f",
        )
    )
    return scene, len(segments)


def write_model_metadata(segment_count, source_mode, output_stem):
    """Persist the modelling assumptions beside the HDF5 result."""

    metadata = {
        "discretisation_m": [DL, DL, DL],
        "domain_m": DOMAIN,
        "time_window_s": TIME_WINDOW,
        "pattern_frequency_hz": PATTERN_FREQUENCY,
        "port_reference_impedance_ohm": PORT_REFERENCE_IMPEDANCE,
        "voltage_source_mode": source_mode,
        "helix_radius_m": HELIX_RADIUS,
        "helix_turns": HELIX_TURNS,
        "helix_spacing_m": HELIX_SPACING,
        "helix_segment_angle_deg": HELIX_SEGMENT_ANGLE_DEG,
        "helix_height_m": FEED_STUB_HEIGHT + HELIX_TURNS * HELIX_SPACING,
        "ground_plane_radius_m": GROUND_RADIUS,
        "ordinary_yee_edge_effective_radius_m": FDTD_EFFECTIVE_WIRE_RADIUS,
        "helix_cylinder_radius_m": HELIX_WIRE_RADIUS,
        "matlab_equivalent_strip_width_m": MATLAB_EQUIVALENT_STRIP_WIDTH,
        "piecewise_linear_cylinder_segments": segment_count,
        "piecewise_linear_centreline_length_m": sum(
            float(np.linalg.norm(np.asarray(stop) - np.asarray(start)))
            for start, stop in helix_segments()
        ),
        "ksir_surface_m": {"p1": KSIR_P1, "p2": KSIR_P2},
    }
    path = RESULTS_DIR / f"{output_stem}_model.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for CPU")
    parser.add_argument(
        "--source-mode",
        choices=("resistive", "hard"),
        default="resistive",
        help="use the original 150-ohm source or an ideal hard delta-gap source",
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="build only and write the ParaView geometry",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scene, segment_count = build_scene(args.source_mode)
    output_stem = OUTPUT_STEM + ("_hard" if args.source_mode == "hard" else "")
    options = {}
    if args.gpu is not None:
        options["gpu"] = [args.gpu]
        options["gpu_precision"] = "single"
    gprMax.run(
        scenes=[scene],
        outputfile=RESULTS_DIR / output_stem,
        geometry_only=args.geometry_only,
        hide_progress_bars=False,
        log_level=logging.INFO,
        **options,
    )
    write_model_metadata(segment_count, args.source_mode, output_stem)
    print(f"Rasterised the three-turn helix as {segment_count} connected PEC cylinders")


if __name__ == "__main__":
    main()
