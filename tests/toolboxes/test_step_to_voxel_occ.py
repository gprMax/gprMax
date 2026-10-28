# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

import csv
import importlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

try:
    pytest.importorskip("OCC")
except pytest.skip.Exception:
    pytest.importorskip(
        "OCP", reason="OpenCascade bindings (pythonocc-core or cadquery-ocp) are not installed"
    )

from gprMax.material_database import load_material_spec
from gprMax.toolboxes.STEPtoVoxel import (
    ConversionConfig,
    _occ,
    convert_step,
    inspect_step,
    step_parser,
    write_material_template,
)

EXAMPLE = Path(__file__).parents[2] / "gprMax" / "toolboxes" / "STEPtoVoxel" / "examples"


def _occ_class(module, name):
    """Return an OpenCascade class from whichever binding the toolbox uses."""
    root = "OCC.Core" if _occ.BACKEND == "pythonocc-core" else "OCP"
    return getattr(importlib.import_module(f"{root}.{module}"), name)


def _static(cls, name):
    """Call a static method, named with an ``_s`` suffix in OCP."""
    return getattr(cls, name + "_s", None) or getattr(cls, name)


def _write_step_assembly(path, parts):
    """Write a STEP assembly of boxes, in millimetres.

    Args:
        path: STEP file to write.
        parts: ``(name, boxes, translation, colour)`` for each part, where
            ``boxes`` lists ``(lower_corner, size)`` (several boxes make one
            part with several solids), ``translation`` places the part in the
            assembly and ``colour`` is an RGB triple or None.
    """
    gp_Pnt, gp_Trsf, gp_Vec = (_occ_class("gp", name) for name in ("gp_Pnt", "gp_Trsf", "gp_Vec"))
    make_box = _occ_class("BRepPrimAPI", "BRepPrimAPI_MakeBox")
    text = _occ_class("TCollection", "TCollection_ExtendedString")
    set_name = _static(_occ_class("TDataStd", "TDataStd_Name"), "Set")
    colour_type = _occ_class("XCAFDoc", "XCAFDoc_ColorType").XCAFDoc_ColorGen
    rgb = _occ_class("Quantity", "Quantity_TypeOfColor").Quantity_TOC_RGB
    document_tool = _occ_class("XCAFDoc", "XCAFDoc_DocumentTool")

    document = _occ_class("TDocStd", "TDocStd_Document")(text("MDTV-XCAF"))
    _static(_occ_class("XCAFApp", "XCAFApp_Application"), "GetApplication")().InitDocument(document)
    shapes = _static(document_tool, "ShapeTool")(document.Main())
    colours = _static(document_tool, "ColorTool")(document.Main())
    assembly = shapes.NewShape()
    set_name(assembly, text("ASSEMBLY"))
    for name, boxes, translation, colour in parts:
        solids = [make_box(gp_Pnt(*lower), *size).Shape() for lower, size in boxes]
        shape = solids[0]
        if len(solids) > 1:
            builder = _occ_class("BRep", "BRep_Builder")()
            shape = _occ_class("TopoDS", "TopoDS_Compound")()
            builder.MakeCompound(shape)
            for solid in solids:
                builder.Add(shape, solid)
        label = shapes.AddShape(shape, False)
        set_name(label, text(name))
        if colour is not None:
            colour = _occ_class("Quantity", "Quantity_Color")(*colour, rgb)
            colours.SetColor(label, colour, colour_type)
        placement = gp_Trsf()
        placement.SetTranslation(gp_Vec(*translation))
        location = _occ_class("TopLoc", "TopLoc_Location")(placement)
        set_name(shapes.AddComponent(assembly, label, location), text(name))
    shapes.UpdateAssemblies()

    # The STEP reader sets the session's length unit to metres; these shapes are
    # in millimetres.
    set_value = _static(_occ_class("Interface", "Interface_Static"), "SetCVal")
    set_value("xstep.cascade.unit", "MM")
    set_value("write.step.unit", "MM")
    writer = _occ_class("STEPCAFControl", "STEPCAFControl_Writer")()
    writer.SetColorMode(True)
    writer.SetNameMode(True)
    as_is = _occ_class("STEPControl", "STEPControl_StepModelType").STEPControl_AsIs
    writer.Transfer(document, as_is)
    assert int(writer.Write(str(path))) == 1
    return path


def _fill_materials(csv_file):
    """Give every row of a prepared material CSV a simple dielectric."""
    with csv_file.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.update(
            include="y",
            material_name="dielectric",
            relative_permittivity="4",
            conductivity="0",
        )
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.integration
def test_probe_fed_patch_step_conversion(tmp_path):
    example = EXAMPLE / "patch_antenna"
    step_file = example / "PROBE_FED.stp"
    parts = inspect_step(
        step_file,
        ConversionConfig(voxel_size=(0.2e-3, 0.2e-3, 0.2e-3)),
    )
    assert [part.name for part in parts] == [
        "PATCH",
        "SUB",
        "GROUND",
        "INNER",
        "DIE",
        "OUTER",
        "port1",
    ]
    assert all(part.name_confidence == "exact" for part in parts)
    assert parts[0].name_source == "MANIFOLD_SOLID_BREP"
    assert parts[-1].name_source == "SHELL_BASED_SURFACE_MODEL"
    assert parts[0].cad["topology_counts"]["faces"] > 0
    assert len(parts[0].cad["principal_moments_m5"]) == 3

    generated_materials = tmp_path / "generated_materials.csv"
    write_material_template(step_file, generated_materials, config=ConversionConfig())
    header = generated_materials.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("group_id,group_confidence,similar_group,part_count,part_names")

    result = convert_step(
        step_file,
        example / "materials.csv",
        tmp_path,
        ConversionConfig(voxel_size=(0.2e-3, 0.2e-3, 0.2e-3)),
        write_vtk=False,
    )

    assert result.component_cell_counts["PATCH"] > 0
    assert "port1" not in result.component_cell_counts
    assert result.vtk_file is None
    assert result.reference_geometry_cad_file is None
    with h5py.File(result.geometry_file) as f:
        assert f["data"].dtype == "int16"
        assert tuple(f.attrs["dx_dy_dz"]) == (0.2e-3, 0.2e-3, 0.2e-3)
        assert f.attrs["MaterialDatabase"] == "materials"
        material_keys = [value.decode() for value in f["material_keys"][:]]
        tag_names = [value.decode() for value in f["tag_names"][:]]
        assert tag_names == ["untagged", "PATCH", "SUB", "GROUND", "INNER", "DIE", "OUTER"]
        assert set(np.unique(f["tag_data"][:])) == set(range(7))
    database = json.loads(result.materials_file.read_text(encoding="utf-8"))
    assert material_keys == list(database["materials"])
    assert database["schema"] == "gprMax-material-database"
    assert "Infinity" not in result.materials_file.read_text(encoding="utf-8")

    specs = {
        key: load_material_spec("materials", key, search_directory=tmp_path)
        for key in material_keys
    }
    pec_specs = [spec for spec in specs.values() if spec.electric_conductivity == float("inf")]
    assert len(pec_specs) == 1
    assert pec_specs[0].metadata["original_id"] == "pec"
    assert (
        specs[next(key for key in material_keys if key.endswith("substrate"))].relative_permittivity
        == 2.2
    )

    # The generated JSON is the user's runtime database. A repeated
    # conversion must not discard properties or metadata edited after the
    # initial STEP conversion.
    first_key = material_keys[0]
    database["materials"][first_key]["metadata"]["user_note"] = "preserve me"
    result.materials_file.write_text(json.dumps(database), encoding="utf-8")
    repeated = convert_step(
        step_file,
        example / "materials.csv",
        tmp_path,
        ConversionConfig(voxel_size=(0.2e-3, 0.2e-3, 0.2e-3)),
        write_vtk=False,
    )
    preserved = json.loads(repeated.materials_file.read_text(encoding="utf-8"))
    assert preserved["materials"][first_key]["metadata"]["user_note"] == "preserve me"

    markers = json.loads(result.markers_file.read_text(encoding="utf-8"))["markers"]
    assert len(markers) == 1
    assert markers[0]["name"] == "port1"
    assert markers[0]["kind"] == "port"
    assert markers[0]["axis"] == "z"

    manifest = json.loads(result.manifest_file.read_text(encoding="utf-8"))
    assert manifest["sweep_axis"] == "auto"
    assert manifest["resolved_sweep_axis"] == "z"
    assert manifest["supersample"] == 1
    assert manifest["material_grouping"] == "exact"
    assert manifest["grouping_relative_tolerance"] == pytest.approx(0.01)


@pytest.fixture
def placed_assembly(tmp_path):
    """A placed single-solid box and a two-solid part that is split."""
    return _write_step_assembly(
        tmp_path / "assembly.stp",
        [
            ("box", [((0, 0, 0), (4, 3, 2))], (10, 20, 30), (0.1, 0.6, 0.2)),
            ("pair", [((0, 0, 0), (1, 1, 1)), ((3, 0, 0), (1, 1, 1))], (-5, 0, 0), None),
        ],
    )


def _bboxes_mm(parts):
    return {part.name: np.asarray(part.cad["bbox_xyzxyz"]) * 1e3 for part in parts}


@pytest.mark.integration
def test_probe_fed_patch_fills_its_area_on_70_um_cells(tmp_path):
    # The 35 um patch's top face lies on the centre plane of 70 um cells: it
    # came out as one triangle of its top face, half of it.
    example = EXAMPLE / "patch_antenna"
    patch = next(part for part in inspect_step(example / "PROBE_FED.stp") if part.name == "PATCH")
    lower, upper = np.array(patch.cad["bbox_xyzxyz"][:3]), np.array(patch.cad["bbox_xyzxyz"][3:])
    area_cells = patch.cad["vol_m3"] / (upper[2] - lower[2]) / 7e-5**2
    for pad in (0, 2):
        result = convert_step(
            example / "PROBE_FED.stp",
            example / "materials.csv",
            tmp_path / f"pad{pad}",
            ConversionConfig(voxel_size=(7e-5, 7e-5, 7e-5), pad_cells=pad),
            write_vtk=False,
        )
        assert result.component_cell_counts["PATCH"] == pytest.approx(area_cells, rel=0.02)


def test_unsplit_part_keeps_its_assembly_placement(placed_assembly):
    parts = step_parser.main(str(placed_assembly), step_parser.ParserConfig())

    bboxes = _bboxes_mm(parts)
    assert sorted(bboxes) == ["box", "pair_S1", "pair_S2"]
    # The split part forces a new operations-space map; the box must keep its
    # placement in it rather than revert to its local coordinates.
    np.testing.assert_allclose(bboxes["box"], [10, 20, 30, 14, 23, 32], atol=1e-3)
    np.testing.assert_allclose(bboxes["pair_S1"], [-5, 0, 0, -4, 1, 1], atol=1e-3)
    np.testing.assert_allclose(bboxes["pair_S2"], [-2, 0, 0, -1, 1, 1], atol=1e-3)


@pytest.mark.integration
def test_rotation_keeps_assembly_placements(placed_assembly):
    config = step_parser.ParserConfig(
        enable_global_rotation=True, rotate_deg=(0.0, 0.0, 90.0), rotate_about="origin"
    )
    parts = step_parser.main(str(placed_assembly), config)

    # A quarter turn about z through the origin maps (x, y) to (-y, x).
    bboxes = _bboxes_mm(parts)
    np.testing.assert_allclose(bboxes["box"], [-23, 10, 30, -20, 14, 32], atol=1e-3)
    np.testing.assert_allclose(bboxes["pair_S1"], [-1, -5, 0, 0, -4, 1], atol=1e-3)
    np.testing.assert_allclose(bboxes["pair_S2"], [-1, -2, 0, 0, -1, 1], atol=1e-3)


@pytest.mark.integration
def test_step_colours_are_read(placed_assembly):
    parts = step_parser.main(str(placed_assembly), step_parser.ParserConfig())

    colours = {part.name: part.color for part in parts}
    assert colours["box"] == pytest.approx((0.1, 0.6, 0.2), abs=1e-6)
    assert colours["pair_S1"] == pytest.approx((0.8, 0.8, 0.8))


@pytest.mark.integration
def test_rotated_box_voxelises_to_exact_cell_centres(tmp_path):
    centre = np.array([50.15, 49.85, 50.37])
    size = np.array([30.0, 10.0, 10.0])
    step_file = _write_step_assembly(
        tmp_path / "box.stp",
        [("box", [(tuple(centre - size / 2), tuple(size))], (0, 0, 0), None)],
    )
    config = ConversionConfig(voxel_size=(2e-3, 2e-3, 2e-3), rotate_deg=(0.0, 0.0, 30.0))
    materials_csv = write_material_template(step_file, tmp_path / "materials.csv", config)
    _fill_materials(materials_csv)

    result = convert_step(step_file, materials_csv, tmp_path / "out", config, write_vtk=False)

    with h5py.File(result.geometry_file) as f:
        occupied = f["data"][:] >= 0
    # Cell centres in millimetres, then in the frame of the unrotated box.
    axes = [
        1e3 * result.origin[axis] + 2.0 * (np.arange(occupied.shape[axis]) + 0.5)
        for axis in range(3)
    ]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1) - centre
    angle = np.deg2rad(30.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    inside = np.all(np.abs(points @ rotation) <= size / 2, axis=-1)

    assert np.count_nonzero(inside) == 375
    np.testing.assert_array_equal(occupied, inside)
    manifest = json.loads(result.manifest_file.read_text(encoding="utf-8"))
    assert manifest["rotation"] == {
        "rotate_deg": [0.0, 0.0, 30.0],
        "rotate_about": "bbox",
        "rotate_order": "xyz",
    }


@pytest.mark.integration
def test_step_entity_labels_ignore_label_like_text(tmp_path):
    original = EXAMPLE / "patch_antenna" / "PROBE_FED.stp"
    text = original.read_text(encoding="utf-8")
    # A comment and a quoted string that look like entity instances.
    text = text.replace("DATA;\n", "DATA;\n/* #9999=COMMENT('not an entity'); */\n", 1)
    text = text.replace("'PART--DESC'", "'see #7=NOTE'", 1)
    trapped = tmp_path / "trapped.stp"
    trapped.write_text(text, encoding="utf-8")

    def labels(path):
        parts = step_parser.load_step_with_hierarchy(str(path), step_parser.ParserConfig())
        return [(part.name, part.step_entity_id, part.name_source) for part in parts]

    expected = labels(original)
    assert [entity_id for _, entity_id, _ in expected] == [42, 44, 46, 48, 50, 52, 108]
    assert labels(trapped) == expected
