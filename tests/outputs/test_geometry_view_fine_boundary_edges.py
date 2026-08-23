import h5py

import gprMax


def test_fine_geometry_view_includes_domain_boundary_edges(tmp_path):
    dl = 1e-3
    nx = ny = nz = 2

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="fine_view_boundary_edges"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(nx * dl, ny * dl, nz * dl)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id="uniform"))
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(nx * dl, ny * dl, nz * dl),
            material_id="uniform",
        )
    )

    outfile = tmp_path / "fine_view"
    scene.add(
        gprMax.GeometryView(
            p1=(0, 0, 0),
            p2=(nx * dl, ny * dl, nz * dl),
            dl=(dl, dl, dl),
            output_type="f",
            filename=str(outfile),
        )
    )
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "run",
        hide_progress_bars=True,
    )

    expected_n_lines = (
        nx * (ny + 1) * (nz + 1)
        + ny * (nx + 1) * (nz + 1)
        + nz * (nx + 1) * (ny + 1)
    )
    assert expected_n_lines == 54

    with h5py.File(str(outfile) + ".vtkhdf") as h:
        material = h["VTKHDF/CellData/Material"][:]
        connectivity = h["VTKHDF/Connectivity"][:]

        assert material.shape[0] == expected_n_lines
        assert connectivity.shape[0] == 2 * expected_n_lines
        assert (material == material[0]).all()
