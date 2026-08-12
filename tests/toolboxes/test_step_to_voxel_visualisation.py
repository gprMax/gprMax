import numpy as np
import pytest

pyvista = pytest.importorskip("pyvista", reason="optional PyVista dependency is not installed")

from toolboxes.STEPtoVoxel.visualisation import (
    translate_reference_geometry,
    write_reference_geometry,
)


def test_reference_geometry_vtp_contains_surface_edge_and_point(tmp_path):
    items = (
        (
            0,
            "port1",
            np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=float),
            np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
        ),
        (
            1,
            "gprmax_source_feed",
            np.array(((0.5, 0.5, 0), (0.5, 0.5, 1)), dtype=float),
            np.empty((0, 3), dtype=np.int32),
        ),
        (
            2,
            "rx1",
            np.array(((0.5, 0.5, 2),), dtype=float),
            np.empty((0, 3), dtype=np.int32),
        ),
    )
    path = tmp_path / "reference_geometry_cad.vtp"

    write_reference_geometry(path, items)

    geometry = pyvista.read(path)
    assert geometry.n_cells == 4  # two triangles, one line and one point
    assert set(geometry.cell_data["reference_geometry_id"]) == {0, 1, 2}
    assert geometry.n_lines == 1
    assert geometry.n_verts == 1

    translated_path = tmp_path / "reference_geometry_gprmax.vtp"
    translate_reference_geometry(path, translated_path, (10.0, 20.0, 30.0))
    translated = pyvista.read(translated_path)
    np.testing.assert_allclose(
        np.asarray(translated.points) - np.asarray(geometry.points),
        np.tile((10.0, 20.0, 30.0), (geometry.n_points, 1)),
    )
