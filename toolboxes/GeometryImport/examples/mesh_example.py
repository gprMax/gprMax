"""Create tagged VTP surface and VTU volume conversion examples."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyvista as pv

from toolboxes.GeometryImport.mesh import convert_mesh, load_mesh_source, write_mesh_template


def _set_assignments(path: Path, names: dict[str, tuple[str, str]]) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    for row in rows:
        material, tag = names[row["region"]]
        row.update(material_name=material, geometry_tag=tag)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def create_examples(output_dir: str | Path = "geometry_import_meshes"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    surface_file = output / "closed_surface.vtp"
    pv.Sphere(radius=10.0, theta_resolution=30, phi_resolution=30).save(surface_file)
    surface_source = load_mesh_source(surface_file, unit="mm")
    surface_csv = write_mesh_template(surface_source, output / "surface_regions.csv", overwrite=True)
    _set_assignments(surface_csv, {"0": ("surface_material", "closed_surface")})
    surface_result = convert_mesh(
        surface_file,
        surface_csv,
        output / "surface_converted",
        voxel_size=(1e-3, 1e-3, 1e-3),
        unit="mm",
    )

    image = pv.ImageData(dimensions=(7, 5, 4), spacing=(2.0, 2.0, 2.0))
    volume = image.cast_to_unstructured_grid()
    centres = volume.cell_centers().points
    volume.cell_data["region_id"] = np.where(centres[:, 0] < 6.0, 1, 2)
    volume_file = output / "two_region_volume.vtu"
    volume.save(volume_file)
    volume_source = load_mesh_source(volume_file, unit="mm", region_array="region_id")
    volume_csv = write_mesh_template(volume_source, output / "volume_regions.csv", overwrite=True)
    _set_assignments(
        volume_csv,
        {
            "1": ("shared_material", "left_region"),
            "2": ("shared_material", "right_region"),
        },
    )
    volume_result = convert_mesh(
        volume_file,
        volume_csv,
        output / "volume_converted",
        voxel_size=(1e-3, 1e-3, 1e-3),
        unit="mm",
        region_array="region_id",
    )
    return surface_result, volume_result


if __name__ == "__main__":
    for conversion in create_examples():
        print(conversion.geometry_file)
