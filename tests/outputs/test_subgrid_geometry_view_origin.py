"""Global-coordinate placement of voxel and edge views owned by a subgrid."""

import h5py
import numpy as np
import pytest

import gprMax


@pytest.mark.integration
@pytest.mark.parametrize("output_type", ("n", "f"))
def test_subgrid_geometry_view_uses_requested_global_origin(tmp_path, output_type):
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.09, 0.09, 0.09)))
    scene.add(gprMax.Discretisation(p1=(0.003, 0.003, 0.003)))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.PMLThickness(thickness=0))
    subgrid = gprMax.SubGridHSG(
        p1=(0.03, 0.03, 0.03),
        p2=(0.06, 0.06, 0.06),
        ratio=3,
        id="fine_grid",
    )
    scene.add(subgrid)
    subgrid.add(
        gprMax.GeometryView(
            p1=(0.041, 0.042, 0.043),
            p2=(0.048, 0.049, 0.050),
            dl=(0.001, 0.001, 0.001),
            filename=f"fine_geometry_{output_type}",
            output_type=output_type,
        )
    )

    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / f"geometry_{output_type}",
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
    )

    filename = tmp_path / f"fine_geometry_{output_type}.vtkhdf"
    assert filename.exists()
    with h5py.File(filename, "r") as view:
        vtk = view["VTKHDF"]
        if output_type == "n":
            np.testing.assert_allclose(vtk.attrs["Origin"], (0.041, 0.042, 0.043))
        else:
            points = vtk["Points"][...]
            np.testing.assert_allclose(np.min(points, axis=0), (0.041, 0.042, 0.043))
