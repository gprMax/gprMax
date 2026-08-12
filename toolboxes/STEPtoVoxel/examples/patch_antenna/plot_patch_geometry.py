"""Render the documented patch-antenna geometry and CAD port overlay."""

from pathlib import Path

import numpy as np
import pyvista as pv

here = Path(__file__).parent
output_dir = here / "output"
geometry_file = output_dir / "geometry.vti"
reference_file = output_dir / "reference_geometry_cad.vtp"

if not geometry_file.is_file() or not reference_file.is_file():
    raise FileNotFoundError(
        "Run patch_antenna_geometry.py first to create geometry.vti and "
        "reference_geometry_cad.vtp."
    )

components = {
    0: ("PATCH", "#D5A54B"),
    1: ("SUB", "#55A89B"),
    2: ("GROUND", "#5D3300"),
    3: ("INNER", "#D43F3A"),
    4: ("DIE", "#70C9C3"),
    5: ("OUTER", "#2468A2"),
}


def add_components(plotter, geometry, identifiers=None, *, legend=False):
    identifiers = components if identifiers is None else identifiers
    for identifier in identifiers:
        name, colour = components[identifier]
        cells = geometry.threshold(
            (identifier - 0.25, identifier + 0.25),
            scalars="component_id",
        )
        plotter.add_mesh(cells, color=colour, label=name if legend else None)


def set_camera(plotter, geometry, *, below=False):
    xmin, xmax, ymin, ymax, zmin, zmax = geometry.bounds
    centre = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    scale = max(xmax - xmin, ymax - ymin)
    zoffset = -0.75 * scale if below else 0.75 * scale
    plotter.camera_position = [
        (centre[0] + 0.9 * scale, centre[1] - 0.9 * scale, centre[2] + zoffset),
        centre,
        (0, 0, 1),
    ]
    plotter.camera.parallel_projection = True
    plotter.reset_camera()


geometry = pv.read(geometry_file)

views = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1800, 850))
views.subplot(0, 0)
add_components(views, geometry)
set_camera(views, geometry)
views.add_text("Radiating side", position="upper_edge", font_size=16)
views.add_axes()

views.subplot(0, 1)
add_components(views, geometry, legend=True)
set_camera(views, geometry, below=True)
views.add_text("Coax-feed side", position="upper_edge", font_size=16)
views.add_legend(bcolor="white", face=None, size=(0.17, 0.34))
views.add_axes()
views.set_background("white")
views.screenshot(here / "probe_fed_70um_views.png")
views.close()

reference = pv.read(reference_file)
overlay = pv.Plotter(off_screen=True, window_size=(1400, 900))
add_components(overlay, geometry, identifiers=(2, 3, 4, 5))
overlay.add_mesh(reference, color="#D0008F", opacity=0.65)
set_camera(overlay, geometry, below=True)
overlay.add_text(
    "port1 reference geometry (magenta; visualisation only)",
    position="upper_edge",
    font_size=16,
)
overlay.add_axes()
overlay.set_background("white")
overlay.screenshot(here / "probe_fed_port1_overlay.png")
overlay.close()

gprmax_view_file = output_dir / "probe_fed_70um_gprmax_geometry.vtkhdf"
model_reference_file = output_dir / "reference_geometry_gprmax.vtp"
if gprmax_view_file.is_file() and model_reference_file.is_file():
    gprmax_geometry = pv.read(gprmax_view_file)
    model_reference = pv.read(model_reference_file)
    occupied = gprmax_geometry.extract_cells(
        np.flatnonzero(np.asarray(gprmax_geometry.cell_data["Material"]) != 2)
    )

    model_overlay = pv.Plotter(off_screen=True, window_size=(1400, 900))
    for material_id, colour in ((0, "#5D3300"), (3, "#55A89B"), (4, "#70C9C3")):
        cells = gprmax_geometry.threshold(
            (material_id - 0.25, material_id + 0.25),
            scalars="Material",
        )
        model_overlay.add_mesh(cells, color=colour)
    model_overlay.add_mesh(model_reference, color="#D0008F", opacity=0.65)
    set_camera(model_overlay, occupied, below=True)
    model_overlay.add_text(
        "gprMax GeometryView with translated port1 reference",
        position="upper_edge",
        font_size=16,
    )
    model_overlay.add_axes()
    model_overlay.set_background("white")
    model_overlay.screenshot(here / "probe_fed_gprmax_reference_overlay.png")
    model_overlay.close()
