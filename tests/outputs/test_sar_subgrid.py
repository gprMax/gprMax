"""Frequency-domain SAR output from a fine HSG subgrid."""

import h5py
import numpy as np
import pytest

import gprMax


def _subgrid_sar_scene(
    *, source_on_main_grid=False, main_spacing=0.003, ratio=3, filtering=True, threads=1
):
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.09, 0.09, 0.09)))
    scene.add(gprMax.Discretisation(p1=(main_spacing,) * 3))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(threads))

    subgrid = gprMax.SubGridHSG(
        p1=(0.03, 0.03, 0.03),
        p2=(0.06, 0.06, 0.06),
        ratio=ratio,
        id="fine_grid",
        filter=filtering,
    )
    scene.add(subgrid)
    subgrid.add(gprMax.Material(er=4, se=0.5, mr=1, sm=0, id="tissue"))
    subgrid.add(gprMax.MaterialDensity(density=1000, material_ids="tissue"))
    subgrid.add(
        gprMax.Box(
            p1=(0.040, 0.040, 0.040),
            p2=(0.050, 0.050, 0.050),
            material_id="tissue",
            tag="target",
        )
    )

    waveform = gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse")
    source = gprMax.HertzianDipole(
        p1=(0.045, 0.045, 0.045) if not source_on_main_grid else (0.018, 0.045, 0.045),
        polarisation="z",
        waveform_id="pulse",
    )
    if source_on_main_grid:
        scene.add(waveform)
        scene.add(source)
    else:
        subgrid.add(waveform)
        subgrid.add(source)

    output = gprMax.SAR(
        frequencies=(5e9,),
        waveform_id="pulse",
        tags="target",
        id="fine_sar",
        spectrum_limit="nyquist",
        averaging_masses=(0.001,),
    )
    subgrid.add(output)
    return scene, output


def _uniform_fine_sar_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.09, 0.09, 0.09)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(8))
    scene.add(gprMax.Material(er=4, se=0.5, mr=1, sm=0, id="tissue"))
    scene.add(gprMax.MaterialDensity(density=1000, material_ids="tissue"))
    scene.add(
        gprMax.Box(
            p1=(0.040, 0.040, 0.040),
            p2=(0.050, 0.050, 0.050),
            material_id="tissue",
            tag="target",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(p1=(0.045, 0.045, 0.045), polarisation="z", waveform_id="pulse")
    )
    scene.add(
        gprMax.SAR(
            frequencies=(5e9,),
            waveform_id="pulse",
            tags="target",
            id="fine_sar",
            spectrum_limit="nyquist",
            averaging_masses=(0.001,),
        )
    )
    return scene


def _subgrid_power_normalised_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.09, 0.09, 0.09)))
    scene.add(gprMax.Discretisation(p1=(0.003, 0.003, 0.003)))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(1))
    fine = gprMax.SubGridHSG(p1=(0.03, 0.03, 0.03), p2=(0.06, 0.06, 0.06), ratio=3, id="fine_grid")
    scene.add(fine)
    fine.add(gprMax.Material(er=4, se=0.5, mr=1, sm=0, id="tissue"))
    fine.add(gprMax.MaterialDensity(density=1000, material_ids="tissue"))
    fine.add(
        gprMax.Box(
            p1=(0.040, 0.040, 0.040),
            p2=(0.050, 0.050, 0.050),
            material_id="tissue",
            tag="target",
        )
    )
    fine.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    fine.add(
        gprMax.VoltageSource(
            p1=(0.045, 0.045, 0.045),
            polarisation="z",
            resistance=50,
            waveform_id="pulse",
        )
    )
    fine.add(gprMax.RxPort(p1=(0.045, 0.045, 0.045), id="feed", spectrum_limit="nyquist"))
    output = gprMax.SAR(
        frequencies=(5e9,),
        waveform_id="pulse",
        tags="target",
        id="power_sar",
        spectrum_limit="nyquist",
        normalisation="incident_power",
        port_id="fine_grid/feed",
        target_power=1.0,
    )
    fine.add(output)
    return scene, output


@pytest.mark.integration
@pytest.mark.parametrize("source_on_main_grid", (False, True))
def test_sar_runs_on_fine_timestep_and_writes_subgrid_group(tmp_path, source_on_main_grid):
    scene, api_output = _subgrid_sar_scene(source_on_main_grid=source_on_main_grid)
    output = tmp_path / ("main_source" if source_on_main_grid else "fine_source")

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
        cpu_precision="single",
    )

    assert api_output.result.sar.shape[0] == 1
    assert api_output.result.cell_indices.shape[0] > 0
    assert np.all(api_output.result.valid)
    assert np.all(np.isfinite(api_output.result.sar))

    with h5py.File(output.with_suffix(".h5"), "r") as result:
        assert "sar" not in result
        fine = result["subgrids/fine_grid"]
        assert fine.attrs["Iterations"] == 3 * result.attrs["Iterations"]
        np.testing.assert_allclose(fine.attrs["dx_dy_dz"], (0.001, 0.001, 0.001))
        sar = fine["sar/fine_sar"]
        assert sar.attrs["CellIndexFrame"] == "subgrid-local"
        expected_origin = (
            np.asarray((0.03, 0.03, 0.03))
            - np.asarray(
                (
                    fine.attrs["is_os_sep"] * fine.attrs["ratio"]
                    + fine.attrs["pml_separation"]
                    + fine.attrs["subgrid_pml_thickness"],
                )
                * 3
            )
            * 0.001
        )
        np.testing.assert_allclose(sar.attrs["CellIndexOrigin"], expected_origin)
        np.testing.assert_allclose(sar.attrs["CellCentreOffset"], (0.0005,) * 3)
        np.testing.assert_array_equal(sar["cell_indices"], api_output.result.cell_indices)
        np.testing.assert_allclose(sar["sar"], api_output.result.sar)
        assert "spatial_average/1g" in sar
        assert np.isfinite(sar["spatial_average/1g/peak_sar"][0])


@pytest.mark.integration
def test_subgrid_sar_state_is_reset_for_geometry_fixed_runs(tmp_path):
    scene, _ = _subgrid_sar_scene()
    output = tmp_path / "reused"

    gprMax.run(
        scenes=[scene],
        n=2,
        geometry_fixed=True,
        outputfile=output,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
        cpu_precision="single",
    )

    with h5py.File(f"{output}1.h5", "r") as first, h5py.File(f"{output}2.h5", "r") as second:
        first_sar = first["subgrids/fine_grid/sar/fine_sar/sar"][...]
        second_sar = second["subgrids/fine_grid/sar/fine_sar/sar"][...]
        np.testing.assert_allclose(second_sar, first_sar, rtol=2e-6, atol=0)


@pytest.mark.integration
def test_subgrid_sar_agrees_with_uniform_fine_grid(tmp_path):
    uniform_path = tmp_path / "uniform"
    subgrid_path = tmp_path / "subgrid"
    subgrid_scene, _ = _subgrid_sar_scene()

    gprMax.run(
        scenes=[_uniform_fine_sar_scene()],
        n=1,
        outputfile=uniform_path,
        hide_progress_bars=True,
        cpu_precision="double",
    )
    gprMax.run(
        scenes=[subgrid_scene],
        n=1,
        outputfile=subgrid_path,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    paths = (
        "tags/target/absorbed_power",
        "tags/target/mass_average_sar",
        "tags/target/peak_voxel_sar",
        "spatial_average/1g/peak_sar",
    )
    with h5py.File(uniform_path.with_suffix(".h5"), "r") as uniform, h5py.File(
        subgrid_path.with_suffix(".h5"), "r"
    ) as nested:
        uniform_sar = uniform["sar/fine_sar"]
        nested_sar = nested["subgrids/fine_grid/sar/fine_sar"]
        for path in paths:
            np.testing.assert_allclose(
                nested_sar[path][...], uniform_sar[path][...], rtol=0.05, atol=0
            )


@pytest.mark.integration
def test_ratio_one_subgrid_sar_matches_identical_uniform_grid(tmp_path):
    uniform_path = tmp_path / "uniform_ratio_one"
    subgrid_path = tmp_path / "subgrid_ratio_one"
    subgrid_scene, _ = _subgrid_sar_scene(main_spacing=0.001, ratio=1, threads=8)

    gprMax.run(
        scenes=[_uniform_fine_sar_scene()],
        n=1,
        outputfile=uniform_path,
        hide_progress_bars=True,
        cpu_precision="double",
    )
    gprMax.run(
        scenes=[subgrid_scene],
        n=1,
        outputfile=subgrid_path,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    with h5py.File(uniform_path.with_suffix(".h5"), "r") as uniform, h5py.File(
        subgrid_path.with_suffix(".h5"), "r"
    ) as nested:
        uniform_sar = uniform["sar/fine_sar"]
        nested_sar = nested["subgrids/fine_grid/sar/fine_sar"]
        uniform_centres = (
            uniform_sar["cell_indices"][...] * 0.001
            + uniform_sar.attrs["CellIndexOrigin"]
            + uniform_sar.attrs["CellCentreOffset"]
        )
        nested_centres = (
            nested_sar["cell_indices"][...] * 0.001
            + nested_sar.attrs["CellIndexOrigin"]
            + nested_sar.attrs["CellCentreOffset"]
        )
        uniform_order = np.lexsort(uniform_centres.T[::-1])
        nested_order = np.lexsort(nested_centres.T[::-1])

        np.testing.assert_allclose(
            nested_centres[nested_order], uniform_centres[uniform_order], rtol=0, atol=1e-15
        )
        uniform_cells = uniform_sar["sar"][0][uniform_order]
        nested_cells = nested_sar["sar"][0][nested_order]
        cell_delta = nested_cells - uniform_cells
        assert np.linalg.norm(cell_delta) / np.linalg.norm(uniform_cells) < 1e-7
        assert np.max(np.abs(cell_delta)) / np.max(uniform_cells) < 1e-7
        for path in (
            "tags/target/absorbed_power",
            "tags/target/mass_average_sar",
            "tags/target/peak_voxel_sar",
            "spatial_average/1g/peak_sar",
        ):
            np.testing.assert_allclose(
                nested_sar[path][...], uniform_sar[path][...], rtol=1e-7, atol=0
            )


@pytest.mark.integration
def test_subgrid_sar_uses_qualified_model_port_for_power_normalisation(tmp_path):
    scene, api_output = _subgrid_power_normalised_scene()
    output = tmp_path / "power_normalised"

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
        cpu_precision="single",
    )

    assert np.all(api_output.result.valid)
    assert np.all(np.isfinite(api_output.result.normalising_power))
    with h5py.File(output.with_suffix(".h5"), "r") as result:
        sar = result["subgrids/fine_grid/sar/power_sar"]
        assert sar.attrs["Normalisation"] == "incident_power"
        assert sar.attrs["PortID"] == "fine_grid/feed"
        assert np.all(np.isfinite(sar["sar"]))
