"""Reproduce MATLAB's default thin-sheet Vivaldi using public gprMax objects."""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolygonPath

import gprMax

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ORIGIN = np.array((0.2205, 0.11225, 0.072))
DOMAIN = (0.440, 0.226, 0.144)
PATTERN_FREQUENCIES = (1e9, 1.5e9, 2e9)
ANGLE = np.arange(-180.0, 182.0, 2.0)
MESHES = {
    "coarse": (0.002, 0.0005, 0.002),
    "fine": (0.001, 0.0005, 0.001),
    "finer": (0.0005, 0.0005, 0.001),
}


def read_geometry(filename=RESULTS / "vivaldi_geometry.json"):
    """Read the portable MATLAB outline; no MATLAB installation is required."""
    geometry = json.loads(Path(filename).read_text())
    if (geometry["coordinate_units"], geometry["plane"], geometry["conductor"]) != (
        "m",
        "xy",
        "pec",
    ):
        raise ValueError("This example requires a PEC polygon in the xy plane, in metres")
    return geometry


def conductor_contains(points, geometry):
    """Even/odd polygon membership includes the sheet but excludes its cavity.

    Boundary points are included. The 1e-12 m polygon dilation only resolves
    floating-point ties; it is not a geometric smoothing or a mesh correction.
    """
    inside = np.zeros(len(points), dtype=bool)
    for vertices in geometry["boundary_loops_xy_m"]:
        vertices = np.asarray(vertices, dtype=float)
        # contains_points' radius follows polygon orientation.
        area = np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) - vertices[:, 1] * np.roll(vertices[:, 0], -1))
        polygon = PolygonPath(np.vstack((vertices, vertices[0])))
        inside ^= polygon.contains_points(points, radius=np.copysign(1e-12, area))
    return inside


def plate_mask(geometry, dl):
    """Rasterise the CAD polygon at xy face centres for public PEC Plates.

    Each retained face sets its four tangential electric edges, exactly as a
    normal gprMax Plate does. Sampling Ex/Ey independently at their midpoints
    would instead erode the Ex banks of the narrow slot and change its loading.
    There are no volumetric PEC voxels and no artificial conductor thickness.
    """
    nx, ny, _ = np.rint(np.asarray(DOMAIN) / dl).astype(int)
    i, j = np.indices((nx, ny))
    points = np.column_stack(((i.ravel() + 0.5) * dl[0] - ORIGIN[0], (j.ravel() + 0.5) * dl[1] - ORIGIN[1]))
    occupied = conductor_contains(points, geometry)
    # Remove the MoM metal feed bridge from the geometrical sheet so the
    # excitation is represented by an electric gap. Away from this 0.5 mm
    # square, the exported outline is used without modification.
    half = geometry["slot_width_m"] / 2
    bridge = (np.abs(points[:, 0] - geometry["feed_x_m"]) <= half + 1e-12) & (np.abs(points[:, 1]) < half - 1e-12)
    occupied[bridge] = False
    return occupied.reshape(nx, ny)


def edge_masks(geometry, dl):
    """Expected tangential PEC edges after the Plate objects are built."""
    cells = plate_mask(geometry, dl)
    nx, ny = cells.shape
    ex = np.zeros((nx, ny + 1), dtype=bool)
    ey = np.zeros((nx + 1, ny), dtype=bool)
    ex[:, :-1] |= cells
    ex[:, 1:] |= cells
    ey[:-1] |= cells
    ey[1:] |= cells
    return ex, ey


def runs(mask):
    """Yield half-open intervals of adjacent True entries."""
    boundaries = np.diff(np.r_[False, mask, False].astype(np.int8))
    yield from zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1))


def build_scene(mesh="fine", time_window=20e-9, geometry_path=RESULTS / "vivaldi_geometry.json"):
    geometry = read_geometry(geometry_path)
    # Keep the 0.5 mm feed slot exactly one Ey edge wide on all meshes.
    # Refinement changes x and/or z only; this is not isotropic convergence.
    dl = np.array(MESHES[mesh])
    step = dl[2]
    masks = edge_masks(geometry, dl)
    feed = ORIGIN + (geometry["feed_x_m"], -geometry["slot_width_m"] / 2, 0)
    feed_index = np.rint(feed / dl).astype(int)
    if not np.allclose(feed / dl, feed_index, rtol=0, atol=1e-10):
        raise ValueError("The feed must lie exactly on a Yee node")
    i, j, _ = feed_index
    # Do not silently clear a PEC edge if the mesh or CAD geometry changes:
    # that would hide a shorted feed or a misplaced source.
    if masks[1][i, j] or not (masks[1][i, j - 1] and masks[1][i, j + 1]):
        raise ValueError("The driven Ey edge must bridge air between two PEC banks")

    scene = gprMax.Scene()
    for item in (
        gprMax.Title(name="Default PEC Vivaldi: MATLAB MoM comparison"),
        gprMax.Discretisation(p1=tuple(dl)),
        gprMax.Domain(p1=DOMAIN),
        gprMax.TimeWindow(time=time_window),
        gprMax.OMPThreads(n=4),
        gprMax.PMLThickness(thickness=12),
        gprMax.Waveform(wave_type="gaussian", amp=1, freq=1.5e9, id="pulse"),
        gprMax.VoltageSource(p1=tuple(feed), polarisation="y", resistance=50, waveform_id="pulse", id="feed"),
    ):
        scene.add(item)
    # Rectangular strips tile occupied xy faces without triangulation seams.
    plate_count = 0
    for i, line in enumerate(plate_mask(geometry, dl)):
        for start, stop in runs(line):
            scene.add(
                gprMax.Plate(
                    p1=(i * dl[0], start * dl[1], ORIGIN[2]),
                    p2=((i + 1) * dl[0], stop * dl[1], ORIGIN[2]),
                    material_id="pec",
                )
            )
            plate_count += 1
    scene.add(
        gprMax.NTFFSurface(p1=(0.034, 0.018, 0.030), p2=(0.408, 0.210, 0.114), id="surface", origin=tuple(ORIGIN))
    )
    scene.add(
        gprMax.NTFFFrequencyTransform(
            surface_id="surface",
            id="spectrum",
            frequencies=PATTERN_FREQUENCIES,
            save_surface_dft=False,
        )
    )
    scene.add(gprMax.NTFFAntennaPorts("spectrum", ("feed",)))
    outputs = (
        "Etheta",
        "Ephi",
        "directivity_dbi",
        "gain_dbi",
        "realized_gain_dbi",
        "radiation_efficiency",
        "total_efficiency",
    )
    scene.add(
        gprMax.NTFFFarFieldArray(
            theta_start=0,
            theta_stop=180,
            theta_step=5,
            phi_start=0,
            phi_stop=355,
            phi_step=5,
            transform_id="spectrum",
            id="full",
            outputs=outputs,
        )
    )
    # Both cuts start at +x; positive angle moves towards +y or +z.
    scene.add(
        gprMax.NTFFFarField(
            theta=np.full(ANGLE.shape, 90),
            phi=ANGLE,
            transform_id="spectrum",
            id="xy",
            outputs=outputs,
        )
    )
    scene.add(
        gprMax.NTFFFarField(
            theta=np.degrees(np.arccos(np.sin(np.radians(ANGLE)))),
            phi=np.where(np.cos(np.radians(ANGLE)) >= -1e-15, 0, 180),
            transform_id="spectrum",
            id="xz",
            outputs=outputs,
        )
    )
    scene.add(
        gprMax.GeometryView(
            p1=(0.066, 0.046, ORIGIN[2] - step),
            p2=(0.376, 0.180, ORIGIN[2] + step),
            dl=tuple(dl),
            filename=f"vivaldi_{mesh}_geometry",
            output_type="f",
        )
    )
    metadata = dict(
        geometry_translation_m=ORIGIN.tolist(),
        feed_node_m=feed.tolist(),
        feed_ey_centre_m=(feed + (0, dl[1] / 2, 0)).tolist(),
        discretisation_m=dl.tolist(),
        domain_m=DOMAIN,
        time_window_s=time_window,
        cell_count=int(np.prod(np.rint(np.asarray(DOMAIN) / dl))),
        pec_edge_counts=[int(mask.sum()) for mask in masks],
        geometry_commands=plate_count,
        pattern_frequency_hz=PATTERN_FREQUENCIES,
    )
    return scene, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, help="CUDA device; omit for CPU")
    parser.add_argument("--mesh", choices=tuple(MESHES), default="fine")
    parser.add_argument("--precision", choices=("single", "double"), default="single")
    parser.add_argument("--time-ns", type=float, default=20)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    if not np.isfinite(args.time_ns) or args.time_ns <= 0:
        parser.error("--time-ns must be finite and positive")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    scene, metadata = build_scene(args.mesh, args.time_ns * 1e-9)
    stem = args.results_dir / f"vivaldi_{args.mesh}"
    if args.geometry_only and stem.with_suffix(".h5").exists():
        parser.error("Use a separate --results-dir for geometry-only when solver output exists")
    options = {"cpu_precision": args.precision}
    if args.gpu is not None:
        options = {"gpu": [args.gpu], "gpu_precision": args.precision}
    started = time.perf_counter()
    gprMax.run(
        scenes=[scene],
        outputfile=stem,
        geometry_only=args.geometry_only,
        hide_progress_bars=True,
        log_level=logging.WARNING,
        **options,
    )
    metadata.update(
        elapsed_seconds=time.perf_counter() - started,
        backend="CPU" if args.gpu is None else "CUDA",
        precision=args.precision,
        geometry_only=args.geometry_only,
    )
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
