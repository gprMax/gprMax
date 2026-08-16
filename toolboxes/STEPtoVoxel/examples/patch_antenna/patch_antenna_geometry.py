"""Convert the probe-fed patch STEP assembly and inspect it with gprMax."""

import argparse
import json
from pathlib import Path

import gprMax
from toolboxes.STEPtoVoxel import (
    ConversionConfig,
    convert_step,
    load_markers,
    translate_reference_geometry,
)

here = Path(__file__).parent
output_dir = here / "output"
dl = 70e-6

parser = argparse.ArgumentParser()
parser.add_argument(
    "--reuse-conversion",
    action="store_true",
    help="reuse files in the example output directory instead of reading STEP",
)
args = parser.parse_args()

if not args.reuse_conversion:
    result = convert_step(
        here / "PROBE_FED.stp",
        here / "materials.csv",
        output_dir,
        ConversionConfig(voxel_size=(dl, dl, dl)),
    )
    geometry_file = result.geometry_file
    materials_file = result.materials_file
    markers_file = result.markers_file
    reference_geometry_cad_file = result.reference_geometry_cad_file
    shape = result.shape
    voxel_grid_origin = result.origin
else:
    manifest_file = output_dir / "conversion.json"
    if not manifest_file.is_file():
        raise FileNotFoundError("Run the STEP conversion before using --reuse-conversion")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    geometry_file = output_dir / "geometry.h5"
    materials_file = output_dir / "materials.txt"
    markers_file = output_dir / "markers.json"
    reference_geometry_cad_file = output_dir / "reference_geometry_cad.vtp"
    shape = tuple(manifest["shape_cells"])
    voxel_grid_origin = tuple(manifest["origin_m"])

padding = 10
import_origin = tuple(padding * dl for _ in range(3))
domain = tuple((cells + 2 * padding) * dl for cells in shape)

# The converter VTI and VTP use original CAD coordinates. A gprMax
# GeometryView uses model coordinates after GeometryObjectsRead applies p1.
# Write a translated reference copy that overlays the latter directly.
gprmax_reference_file = output_dir / "reference_geometry_gprmax.vtp"
if reference_geometry_cad_file is not None and Path(reference_geometry_cad_file).is_file() and (
    not args.reuse_conversion or not gprmax_reference_file.is_file()
):
    translation = tuple(
        import_origin[axis] - voxel_grid_origin[axis] for axis in range(3)
    )
    translate_reference_geometry(
        Path(reference_geometry_cad_file),
        gprmax_reference_file,
        translation,
    )

# CAD markers are stored relative to geometry.h5. Translate port1 to the same
# model coordinates as the imported voxel geometry, then collapse its very
# small CAD thickness to the nearest Yee-grid plane.
port = load_markers(markers_file)["port1"]
port_centre = port.model_position(import_origin)
port_bounds = port.model_bounds(import_origin)


def snap_to_grid(value):
    return round(value / dl) * dl


xmin, ymin, _, xmax, ymax, _ = port_bounds
port_z = snap_to_grid(port_centre[2])
port_p1 = (snap_to_grid(xmin), snap_to_grid(ymin), port_z)
port_p2 = (snap_to_grid(xmax), snap_to_grid(ymax), port_z)
print(f"port1 CAD centre in model coordinates: {port_centre}")
print(f"port1 grid-aligned plane: p1={port_p1}, p2={port_p2}, axis={port.axis}")

scene = gprMax.Scene()
scene.add(gprMax.Title(name="STEP-to-voxel probe-fed patch geometry"))
scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
scene.add(gprMax.Domain(p1=domain))
scene.add(gprMax.TimeWindow(time=1e-12))
scene.add(gprMax.PMLThickness(thickness=0))
scene.add(
    gprMax.GeometryObjectsRead(
        p1=import_origin,
        geofile=geometry_file,
        matfile=materials_file,
    )
)
scene.add(
    gprMax.GeometryView(
        p1=(0, 0, 0),
        p2=domain,
        dl=(dl, dl, dl),
        output_type="n",
        filename=str(output_dir / "probe_fed_70um_gprmax_geometry"),
    )
)

gprMax.run(
    scenes=[scene],
    n=1,
    geometry_only=True,
    outputfile=output_dir / "probe_fed_geometry",
)
