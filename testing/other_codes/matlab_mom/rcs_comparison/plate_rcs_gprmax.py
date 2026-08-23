"""Run the gprMax square- and circular-PEC-plate RCS comparisons."""

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.constants import c
from scipy.special import j1

import gprMax

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

WAVELENGTH = 3.25e-2
FREQUENCY = c / WAVELENGTH
SQUARE_LENGTH = 10.16e-2
SQUARE_WIDTH = 10.16e-2
CIRCLE_RADIUS = 10.16e-2

MESHES = {
    "coarse": 2.54e-3,
    "fine": 1.27e-3,
}
PML_CELLS = 10
TARGET_TO_TFSF_CELLS = 8
TFSF_TO_KSIR_CELLS = 5
KSIR_TO_PML_CELLS = 8
TIME_WINDOW = 5e-9
DPW_DIRECTION_SCALE = 64
SELECTED_ELEVATIONS = (0.05, 10.05, 20.05, 30.05, 45.05, 60.05, 75.05, 89.05)
DEFAULT_THREADS = min(20, os.cpu_count() or 1)


def _direction_vector(elevation):
    """Return a reduced integer DPW vector and its exact physical elevation."""

    radians = np.deg2rad(elevation)
    mx = int(np.rint(DPW_DIRECTION_SCALE * np.cos(radians)))
    mz = int(np.rint(DPW_DIRECTION_SCALE * np.sin(radians)))
    if mx == 0 and mz == 0:
        raise ValueError("The plane-wave direction rounded to a zero vector")
    divisor = math.gcd(abs(mx), abs(mz))
    mx //= divisor
    mz //= divisor
    actual_elevation = float(np.rad2deg(np.arctan2(mz, mx)))
    return (mx, 0, mz), actual_elevation


def _layout(target, dl):
    """Return cell-aligned domain, source, monitor, and target coordinates."""

    if target == "square":
        half_target_cells = int(round(max(SQUARE_LENGTH, SQUARE_WIDTH) / (2 * dl)))
    else:
        half_target_cells = int(round(CIRCLE_RADIUS / dl))
    half_tfsf = half_target_cells + TARGET_TO_TFSF_CELLS
    half_ksir = half_tfsf + TFSF_TO_KSIR_CELLS
    half_domain = half_ksir + KSIR_TO_PML_CELLS + PML_CELLS
    cells = 2 * half_domain
    centre_index = half_domain

    def point(index):
        return tuple(float(value * dl) for value in index)

    centre = point((centre_index,) * 3)
    tfsf_p1 = point((centre_index - half_tfsf,) * 3)
    tfsf_p2 = point((centre_index + half_tfsf,) * 3)
    ksir_p1 = point((centre_index - half_ksir,) * 3)
    ksir_p2 = point((centre_index + half_ksir,) * 3)
    return {
        "cells": cells,
        "domain": point((cells,) * 3),
        "centre": centre,
        "centre_index": centre_index,
        "half_target_cells": half_target_cells,
        "tfsf_p1": tfsf_p1,
        "tfsf_p2": tfsf_p2,
        "ksir_p1": ksir_p1,
        "ksir_p2": ksir_p2,
    }


def _add_square(scene, layout, dl):
    """Add the exact 40- or 80-cell zero-thickness square PEC plate."""

    cx, cy, cz = layout["centre"]
    scene.add(
        gprMax.Plate(
            p1=(cx - SQUARE_LENGTH / 2, cy - SQUARE_WIDTH / 2, cz),
            p2=(cx + SQUARE_LENGTH / 2, cy + SQUARE_WIDTH / 2, cz),
            material_id="pec",
        )
    )


def _add_circle(scene, layout, dl):
    """Add a cell-centred row raster of a zero-thickness circular PEC plate."""

    centre_index = layout["centre_index"]
    radius_cells = int(round(CIRCLE_RADIUS / dl))
    z = centre_index * dl
    for offset_y in range(-radius_cells, radius_cells):
        y_mid = offset_y + 0.5
        half_width = math.sqrt(max(radius_cells**2 - y_mid**2, 0.0))
        lower_x = math.ceil(-half_width - 0.5)
        upper_x = math.floor(half_width - 0.5) + 1
        if upper_x <= lower_x:
            continue
        scene.add(
            gprMax.Plate(
                p1=(
                    (centre_index + lower_x) * dl,
                    (centre_index + offset_y) * dl,
                    z,
                ),
                p2=(
                    (centre_index + upper_x) * dl,
                    (centre_index + offset_y + 1) * dl,
                    z,
                ),
                material_id="pec",
            )
        )


def build_scene(target, mesh, requested_elevation, threads):
    """Build one monostatic HH plate-scattering scene."""

    dl = MESHES[mesh]
    layout = _layout(target, dl)
    m_vec, actual_elevation = _direction_vector(requested_elevation)
    backscatter_theta = 90.0 + actual_elevation

    scene = gprMax.Scene()
    scene.add(
        gprMax.Title(
            name=(
                f"{target.capitalize()} plate monostatic RCS, "
                f"elevation {actual_elevation:.6f} degrees"
            )
        )
    )
    scene.add(gprMax.Discretisation(p1=(dl,) * 3))
    scene.add(gprMax.Domain(p1=layout["domain"]))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(gprMax.PMLThickness(thickness=PML_CELLS))
    scene.add(
        gprMax.Waveform(
            wave_type="ricker",
            amp=1.0,
            freq=FREQUENCY,
            id="plane_pulse",
        )
    )
    scene.add(
        gprMax.DiscretePlaneWaveVector(
            p1=layout["tfsf_p1"],
            p2=layout["tfsf_p2"],
            m_vec=m_vec,
            psi=0.0,
            waveform_id="plane_pulse",
        )
    )
    if target == "square":
        _add_square(scene, layout, dl)
    else:
        _add_circle(scene, layout, dl)

    surface_id = f"{target}_surface"
    transform_id = f"{target}_spectrum"
    scene.add(
        gprMax.NTFFSurface(
            p1=layout["ksir_p1"],
            p2=layout["ksir_p2"],
            id=surface_id,
            origin=layout["centre"],
        )
    )
    scene.add(
        gprMax.KSIRFrequencyTransform(
            surface_id=surface_id,
            id=transform_id,
            frequencies=(FREQUENCY,),
            save_surface_dft=False,
            plane_wave_index=0,
        )
    )
    scene.add(
        gprMax.KSIRFarField(
            theta=(backscatter_theta,),
            phi=(180.0,),
            transform_id=transform_id,
            id="monostatic_hh",
            outputs=("Etheta", "Ephi", "rcs"),
        )
    )
    return scene, layout, m_vec, actual_elevation


def _output_tag(elevation):
    """Return a path-safe fixed-precision requested-angle tag."""

    return f"{elevation:07.3f}".replace("-", "m").replace(".", "p")


def _read_result(path, target, actual_elevation):
    """Read one persisted KSIR RCS result and validate its coordinates."""

    root = f"ntff/{target}_surface/frequency/{target}_spectrum"
    far_root = f"{root}/far_field/monostatic_hh"
    with h5py.File(path, "r") as h5:
        transform = h5[root]
        frequency = np.asarray(transform["frequencies"])
        theta = np.asarray(h5[f"{far_root}/theta"])
        phi = np.asarray(h5[f"{far_root}/phi"])
        rcs = np.asarray(h5[f"{far_root}/fields/rcs"])
        collection_backend = transform.attrs["collection_backend"]
        solver_backend = transform.attrs["solver"]
        precision = transform.attrs["precision"]
    if frequency.shape != (1,) or not np.allclose(frequency, (FREQUENCY,)):
        raise ValueError(f"Unexpected frequency data in {path}")
    if theta.shape != (1,) or not np.allclose(theta, (90 + actual_elevation,)):
        raise ValueError(f"Unexpected backscatter theta in {path}")
    if phi.shape != (1,) or not np.allclose(phi, (180.0,)):
        raise ValueError(f"Unexpected backscatter phi in {path}")
    if rcs.shape != (1, 1) or not np.isfinite(rcs[0, 0]) or rcs[0, 0] <= 0:
        raise ValueError(f"Invalid monostatic RCS in {path}")
    metadata = {
        "collection_backend": collection_backend,
        "solver_backend": solver_backend,
        "precision": precision,
    }
    metadata = {
        key: value.decode() if isinstance(value, bytes) else str(value)
        for key, value in metadata.items()
    }
    return float(rcs[0, 0]), metadata


def analytical_square_rcs(elevation):
    """Return the rectangular-plate physical-optics HH monostatic RCS."""

    elevation = np.deg2rad(np.asarray(elevation, dtype=np.float64))
    argument = 2 * np.pi * SQUARE_LENGTH * np.cos(elevation) / WAVELENGTH
    aperture = SQUARE_LENGTH * SQUARE_WIDTH
    return (
        4
        * np.pi
        * aperture**2
        / WAVELENGTH**2
        * np.sin(elevation) ** 2
        * np.sinc(argument / np.pi) ** 2
    )


def analytical_circle_rcs(elevation):
    """Return the circular-plate physical-optics HH monostatic RCS."""

    elevation = np.deg2rad(np.asarray(elevation, dtype=np.float64))
    argument = 4 * np.pi * CIRCLE_RADIUS * np.cos(elevation) / WAVELENGTH
    airy = np.ones_like(argument)
    nonzero = np.abs(argument) > np.finfo(float).eps
    airy[nonzero] = 2 * j1(argument[nonzero]) / argument[nonzero]
    aperture = np.pi * CIRCLE_RADIUS**2
    return 4 * np.pi * aperture**2 / WAVELENGTH**2 * np.sin(elevation) ** 2 * airy**2


def run_sweep(target, mesh, elevations, threads, precision, gpu, force):
    """Run or resume independent monostatic scenes and write one result CSV."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    collection_backends = set()
    solver_backends = set()
    result_precisions = set()
    for requested_elevation in elevations:
        scene, layout, m_vec, actual_elevation = build_scene(
            target, mesh, requested_elevation, threads
        )
        stem = f"{target}_{mesh}_el_{_output_tag(requested_elevation)}"
        output_base = RESULTS_DIR / stem
        h5_path = output_base.with_suffix(".h5")
        if force or not h5_path.exists():
            solver_options = (
                {"cpu_precision": precision}
                if gpu is None
                else {"gpu": [gpu], "gpu_precision": precision}
            )
            gprMax.run(
                scenes=[scene],
                outputfile=output_base,
                hide_progress_bars=False,
                log_level=logging.INFO,
                **solver_options,
            )
        rcs, result_metadata = _read_result(h5_path, target, actual_elevation)
        collection_backends.add(result_metadata["collection_backend"])
        solver_backends.add(result_metadata["solver_backend"])
        result_precisions.add(result_metadata["precision"])
        analytic = (
            analytical_square_rcs((actual_elevation,))[0]
            if target == "square"
            else analytical_circle_rcs((actual_elevation,))[0]
        )
        rows.append(
            {
                "requested_elevation_deg": requested_elevation,
                "actual_elevation_deg": actual_elevation,
                "m_x": m_vec[0],
                "m_y": m_vec[1],
                "m_z": m_vec[2],
                "gprmax_rcs_m2": rcs,
                "gprmax_rcs_dbsm": 10 * np.log10(rcs),
                "analytical_po_rcs_m2": analytic,
                "analytical_po_rcs_dbsm": 10 * np.log10(max(analytic, np.finfo(float).tiny)),
                **result_metadata,
                "hdf5_file": h5_path.name,
            }
        )
        print(
            f"{target} {mesh}: requested {requested_elevation:.3f} deg, "
            f"actual {actual_elevation:.6f} deg, "
            f"RCS {10 * np.log10(rcs):.3f} dBsm"
        )

    rows.sort(key=lambda item: item["actual_elevation_deg"])
    csv_path = RESULTS_DIR / f"{target}_{mesh}_gprmax.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "target": target,
        "mesh": mesh,
        "discretisation_m": MESHES[mesh],
        "frequency_hz": FREQUENCY,
        "wavelength_m": WAVELENGTH,
        "time_window_s": TIME_WINDOW,
        "solver_backends": sorted(solver_backends),
        "collection_backends": sorted(collection_backends),
        "precisions": sorted(result_precisions),
        "polarisation": "HH; incident E parallel to y",
        "direction_mapping_scale": DPW_DIRECTION_SCALE,
        "layout": layout,
        "square_length_m": SQUARE_LENGTH,
        "square_width_m": SQUARE_WIDTH,
        "circle_radius_m": CIRCLE_RADIUS,
    }
    (RESULTS_DIR / f"{target}_{mesh}_model.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return csv_path


def main():
    """Parse command-line options and run a resumable plate RCS sweep."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("square", "circle"), default="square")
    parser.add_argument("--mesh", choices=tuple(MESHES), default="coarse")
    parser.add_argument(
        "--elevations",
        nargs="+",
        type=float,
        default=SELECTED_ELEVATIONS,
        help="requested MathWorks-style elevation angles in degrees",
    )
    parser.add_argument(
        "--mathworks-sweep",
        action="store_true",
        help="use the original 0.05:1:90 requested-angle sweep",
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--precision", choices=("single", "double"), default="single")
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for CPU")
    parser.add_argument("--force", action="store_true", help="rerun existing angles")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    elevations = np.arange(0.05, 90.0, 1.0) if args.mathworks_sweep else args.elevations
    if np.any(np.asarray(elevations) < 0) or np.any(np.asarray(elevations) > 90):
        parser.error("elevations must lie between 0 and 90 degrees")
    output = run_sweep(
        args.target,
        args.mesh,
        tuple(float(value) for value in elevations),
        args.threads,
        args.precision,
        args.gpu,
        args.force,
    )
    print(f"Saved resumable gprMax RCS sweep to {output}")


if __name__ == "__main__":
    main()
