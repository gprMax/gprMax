"""Render reproducible input-versus-voxel GeometryImport documentation figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from toolboxes.GeometryImport.examples.labelled_volume_example import create_example
from toolboxes.GeometryImport.examples.mesh_example import create_examples
from toolboxes.GeometryImport.volume import load_label_volume

_COLOURS = ("#5969a6", "#d43f8d", "#34a6a6", "#ef9f36")


def _image_from_cells(data, spacing, origin, name):
    image = pv.ImageData(
        dimensions=tuple(int(value) + 1 for value in data.shape),
        spacing=spacing,
        origin=origin,
    )
    image.cell_data[name] = np.asarray(data).ravel(order="F")
    return image


def _add_regions(plotter, dataset, scalar, region_ids, *, opacity=(0.35, 1.0)):
    for index, region_id in enumerate(region_ids):
        selected = dataset.threshold(
            (region_id - 0.25, region_id + 0.25),
            scalars=scalar,
            preference="cell",
        )
        plotter.add_mesh(
            selected,
            color=_COLOURS[index % len(_COLOURS)],
            opacity=opacity[min(index, len(opacity) - 1)],
            show_edges=True,
            edge_color="#303030",
            line_width=0.25,
        )


def _finish_panel(plotter, title):
    plotter.add_text(title, position="upper_edge", font_size=12, color="black")
    plotter.show_axes()
    plotter.view_isometric()
    plotter.reset_camera()
    plotter.camera.zoom(1.15)


def render_labelled_volume(work_dir: Path, output: Path) -> Path:
    result = create_example(work_dir)
    if result.preview_file is None:
        raise RuntimeError("PyVista is required to produce geometry previews")
    source = load_label_volume(work_dir / "synthetic_anatomy.nrrd")
    source_origin = tuple(
        centre - 0.5 * spacing for centre, spacing in zip(source.first_cell_centre_m, source.spacing_m)
    )
    input_grid = _image_from_cells(
        source.labels,
        source.spacing_m,
        source_origin,
        "SourceLabel",
    )
    output_grid = pv.read(result.preview_file)

    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 720))
    plotter.set_background("white")
    plotter.subplot(0, 0)
    _add_regions(plotter, input_grid, "SourceLabel", (1, 2))
    _finish_panel(plotter, "Input NRRD labels")
    plotter.subplot(0, 1)
    _add_regions(plotter, output_grid, "TagID", (1, 2))
    _finish_panel(plotter, "Voxel output: semantic TagID")
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(output)
    plotter.close()
    return output


def render_meshes(work_dir: Path, output: Path) -> Path:
    surface_result, volume_result = create_examples(work_dir)
    if surface_result.preview_file is None or volume_result.preview_file is None:
        raise RuntimeError("PyVista is required to produce geometry previews")

    surface_source = pv.read(work_dir / "closed_surface.vtp")
    surface_output = pv.read(surface_result.preview_file)
    volume_source = pv.read(work_dir / "two_region_volume.vtu")
    volume_output = pv.read(volume_result.preview_file)

    plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1600, 1300))
    plotter.set_background("white")
    plotter.subplot(0, 0)
    plotter.add_mesh(surface_source, color=_COLOURS[0], smooth_shading=True)
    _finish_panel(plotter, "Closed VTP surface input")
    plotter.subplot(0, 1)
    occupied = surface_output.threshold((-0.5, 0.5), scalars="MaterialIndex", preference="cell")
    plotter.add_mesh(
        occupied,
        color=_COLOURS[0],
        show_edges=True,
        edge_color="#303030",
        line_width=0.25,
    )
    _finish_panel(plotter, "Plane-sweep/scanline voxel output")
    plotter.subplot(1, 0)
    _add_regions(plotter, volume_source, "region_id", (1, 2), opacity=(0.65, 0.65))
    _finish_panel(plotter, "Two-region VTU volume input")
    plotter.subplot(1, 1)
    _add_regions(plotter, volume_output, "TagID", (1, 2), opacity=(0.65, 0.65))
    _finish_panel(plotter, "Cell-centre sampled voxel output")
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(output)
    plotter.close()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("images_shared"))
    parser.add_argument("--work", type=Path, default=Path("geometry_import_figure_work"))
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    print(render_labelled_volume(args.work / "labelled_volume", args.output / "geometry_import_labelled_volume.png"))
    print(render_meshes(args.work / "meshes", args.output / "geometry_import_mesh_voxelisation.png"))


if __name__ == "__main__":
    main()
