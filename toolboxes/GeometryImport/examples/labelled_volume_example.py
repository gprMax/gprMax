"""Create and convert a small synthetic labelled-anatomy NRRD volume."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import nrrd
import numpy as np

from toolboxes.GeometryImport.volume import (
    convert_label_volume,
    load_label_volume,
    write_label_template,
)


def _complete_material_database(path: Path) -> None:
    """Supply illustrative properties to the generated editable template."""

    document = json.loads(path.read_text(encoding="utf-8"))
    properties = {
        "synthetic_body": (30.0, 0.40, 1000.0),
        "synthetic_organ": (50.0, 0.70, 1050.0),
    }
    for entry in document["materials"].values():
        er, conductivity, density = properties[entry["name"]]
        entry["base"] = {
            "relative_permittivity": er,
            "electric_conductivity_s_per_m": conductivity,
            "relative_permeability": 1.0,
            "magnetic_conductivity_s_per_m": 0.0,
        }
        entry["mass_density_kg_per_m3"] = density
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def create_example(output_dir: str | Path = "geometry_import_labelled_volume"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    shape = (48, 40, 32)
    spacing_mm = np.asarray((1.0, 1.0, 1.0))
    indices = np.indices(shape, dtype=float)
    xyz = np.moveaxis(indices, 0, -1) + 0.5
    body_centre = np.asarray(shape) / 2
    body_radius = np.asarray((20.0, 16.0, 12.0))
    organ_centre = body_centre + np.asarray((3.0, 0.0, 0.0))
    organ_radius = np.asarray((7.0, 6.0, 5.0))

    labels = np.zeros(shape, dtype=np.uint8)
    labels[np.sum(((xyz - body_centre) / body_radius) ** 2, axis=-1) <= 1] = 1
    labels[np.sum(((xyz - organ_centre) / organ_radius) ** 2, axis=-1) <= 1] = 2

    source = output / "synthetic_anatomy.nrrd"
    nrrd.write(
        str(source),
        labels,
        header={
            "space": "left-posterior-superior",
            "space directions": np.diag(spacing_mm),
            # Image formats locate sample centres. This choice places the
            # lower boundary of the converted gprMax volume at (0, 0, 0).
            "space origin": 0.5 * spacing_mm,
            "space units": ["mm", "mm", "mm"],
            "Segment0_LabelValue": "1",
            "Segment0_Name": "synthetic body",
            "Segment1_LabelValue": "2",
            "Segment1_Name": "synthetic organ",
        },
        index_order="F",
    )

    assignments = output / "labels.csv"
    write_label_template(load_label_volume(source), assignments, overwrite=True)
    with assignments.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    for row in rows:
        if row["label"] == "1":
            row.update(material_name="synthetic_body", geometry_tag="body", include="y")
        elif row["label"] == "2":
            row.update(material_name="synthetic_organ", geometry_tag="organ", include="y")
    with assignments.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    result = convert_label_volume(source, assignments, output / "converted")
    _complete_material_database(result.materials_file)
    return result


if __name__ == "__main__":
    conversion = create_example()
    print(conversion.geometry_file)
