"""Fine-grid eigenmode-port validation for HSG subgrids."""

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod
from gprMax.subgrids.subgrid_hsg import SubGridHSG


FREQUENCY = 12e9


def _add_modal_commands(
    owner,
    *,
    offset,
    normal_axis=0,
    passive_port=False,
    virtual_waveguide=False,
):
    """Add the same non-degenerate 12 by 10 mm aperture after translation."""

    transverse_axes = tuple(axis for axis in range(3) if axis != normal_axis)
    core_p1 = np.full(3, 0.012)
    core_p2 = np.full(3, 0.016)
    core_p1[normal_axis] = 0.006
    core_p2[normal_axis] = 0.027
    core_p1[transverse_axes[0]] = 0.013
    core_p2[transverse_axes[0]] = 0.017
    owner.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id="core"))
    owner.add(
        gprMax.Box(
            p1=tuple(core_p1 + offset),
            p2=tuple(core_p2 + offset),
            material_id="core",
        )
    )
    source_p1 = np.full(3, 0.009)
    source_p2 = source_p1.copy()
    source_p1[normal_axis] = 0.015
    source_p2[normal_axis] = 0.015
    source_p2[transverse_axes[0]] = 0.021
    source_p2[transverse_axes[1]] = 0.019

    owner.add(gprMax.Waveform(wave_type="contsine", amp=1, freq=FREQUENCY, id="wave"))
    owner.add(
        gprMax.EigenmodeBand(
            id="band",
            fmin=FREQUENCY,
            fmax=FREQUENCY,
            points=1,
        )
    )
    owner.add(
        gprMax.EigenmodePort(
            port=1,
            p1=tuple(source_p1 + offset),
            p2=tuple(source_p2 + offset),
            direction="+",
            modes=(1,),
            anchors=(FREQUENCY,),
            plot_fields=False,
        )
    )
    if passive_port:
        passive_p1 = source_p1.copy()
        passive_p2 = source_p2.copy()
        passive_p1[normal_axis] = 0.024
        passive_p2[normal_axis] = 0.024
        owner.add(
            gprMax.EigenmodePort(
                port=2,
                p1=tuple(passive_p1 + offset),
                p2=tuple(passive_p2 + offset),
                direction="-",
                modes=(1,),
                anchors=(FREQUENCY,),
                plot_fields=False,
            )
        )
    owner.add(
        gprMax.EigenmodeExcitation(
            port=1,
            mode=1,
            waveform="wave",
            plot_waveform=False,
        )
    )
    if virtual_waveguide:
        owner.add(
            gprMax.VirtualWaveguide(
                port=1,
                length_cells=12,
                pml_cells=4,
                source_clearance_cells=3,
            )
        )


def _uniform_fine_scene(
    normal_axis=0,
    *,
    iterations=None,
    passive_port=False,
    virtual_waveguide=False,
):
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.03, 0.03, 0.03)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    if iterations is None:
        scene.add(gprMax.TimeWindow(time=5e-10))
    else:
        scene.add(gprMax.TimeWindow(iterations=iterations))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(1))
    _add_modal_commands(
        scene,
        offset=np.zeros(3),
        normal_axis=normal_axis,
        passive_port=passive_port,
        virtual_waveguide=virtual_waveguide,
    )
    return scene


def _subgrid_scene(
    *,
    timewindow=5e-10,
    iterations=None,
    normal_axis=0,
    port_p1=None,
    port_p2=None,
    virtual_waveguide=False,
):
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.09, 0.09, 0.09)))
    scene.add(gprMax.Discretisation(p1=(0.003, 0.003, 0.003)))
    if iterations is None:
        scene.add(gprMax.TimeWindow(time=timewindow))
    else:
        scene.add(gprMax.TimeWindow(iterations=iterations))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(1))
    subgrid = gprMax.SubGridHSG(
        p1=(0.03, 0.03, 0.03),
        p2=(0.06, 0.06, 0.06),
        ratio=3,
        id="fine_grid",
    )
    scene.add(subgrid)
    if port_p1 is None:
        _add_modal_commands(
            subgrid,
            offset=np.full(3, 0.03),
            normal_axis=normal_axis,
            passive_port=True,
            virtual_waveguide=virtual_waveguide,
        )
    else:
        subgrid.add(
            gprMax.Waveform(wave_type="contsine", amp=1, freq=FREQUENCY, id="wave")
        )
        subgrid.add(
            gprMax.EigenmodeBand(
                id="band",
                fmin=FREQUENCY,
                fmax=FREQUENCY,
                points=1,
            )
        )
        subgrid.add(
            gprMax.EigenmodePort(
                port=1,
                p1=port_p1,
                p2=port_p2,
                direction="+",
                modes=(1,),
                anchors=(FREQUENCY,),
                plot_fields=False,
            )
        )
        subgrid.add(
            gprMax.EigenmodeExcitation(
                port=1,
                mode=1,
                waveform="wave",
                plot_waveform=False,
            )
        )
    return scene, subgrid


def _capture_built_grids(monkeypatch, *, uniform_source_stop_iteration=None):
    captured = []
    original_build = model_mod.Model.build

    def patched_build(self):
        original_build(self)
        if uniform_source_stop_iteration is not None and not self.subgrids:
            for source in self.G.eigenmodesources:
                source.stop = uniform_source_stop_iteration * self.G.dt
            for guide in self.G.virtual_waveguides:
                if guide.aux_source is not None:
                    guide.aux_source.stop = uniform_source_stop_iteration * self.G.dt
        captured.append([self.G, *self.subgrids])

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


@pytest.mark.integration
@pytest.mark.parametrize("normal_axis", range(3), ids=("x", "y", "z"))
def test_subgrid_fdfd_slice_matches_translated_uniform_fine_grid(
    monkeypatch, tmp_path, normal_axis
):
    captured = _capture_built_grids(monkeypatch)

    gprMax.run(
        scenes=[_uniform_fine_scene(normal_axis)],
        geometry_only=True,
        outputfile=tmp_path / "uniform",
        cpu_precision="double",
        hide_progress_bars=True,
    )
    scene, _ = _subgrid_scene(normal_axis=normal_axis)
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "subgrid",
        subgrid=True,
        autotranslate=True,
        cpu_precision="double",
        hide_progress_bars=True,
    )

    uniform_grid = captured[0][0]
    fine_grid = captured[1][1]
    uniform_source = uniform_grid.eigenmodesources[0]
    fine_source = fine_grid.eigenmodesources[0]

    np.testing.assert_allclose(fine_grid.dl, uniform_grid.dl)
    source_start = np.zeros(3, dtype=np.int32)
    source_start[fine_source.normal_axis] = fine_source.plane_index
    source_start[list(fine_source.transverse_axes)] = fine_source.transverse_start
    expected_start = np.full(3, 0.039)
    expected_start[normal_axis] = 0.045
    np.testing.assert_allclose(fine_grid.local_to_global(source_start), expected_start)
    assert fine_source._transverse_cell_shape() == uniform_source._transverse_cell_shape()
    assert fine_source.complex_neff == pytest.approx(
        uniform_source.complex_neff,
        rel=1e-12,
        abs=1e-12,
    )
    for fine_field, uniform_field in zip(
        (
            *fine_source.modal_e,
            *fine_source.modal_h,
            *fine_source.modal_e_real,
            *fine_source.modal_h_real,
        ),
        (
            *uniform_source.modal_e,
            *uniform_source.modal_h,
            *uniform_source.modal_e_real,
            *uniform_source.modal_h_real,
        ),
    ):
        np.testing.assert_allclose(fine_field, uniform_field, rtol=1e-11, atol=1e-9)


@pytest.mark.parametrize(
    ("p1", "p2"),
    (
        ((0.031, 0.039, 0.039), (0.031, 0.051, 0.051)),
        ((0.045, 0.030, 0.039), (0.045, 0.051, 0.051)),
        ((0.045, 0.039, 0.039), (0.045, 0.060, 0.051)),
    ),
    ids=("normal-stencil", "lower-transverse", "upper-transverse"),
)
def test_subgrid_eigenmode_plane_cannot_touch_coupling_surface(tmp_path, p1, p2):
    scene, _ = _subgrid_scene(timewindow=1e-10, port_p1=p1, port_p2=p2)

    with pytest.raises(ValueError, match="must lie strictly inside the subgrid working region"):
        gprMax.run(
            scenes=[scene],
            geometry_only=True,
            outputfile=tmp_path / "invalid",
            subgrid=True,
            autotranslate=True,
            hide_progress_bars=True,
        )


@pytest.mark.integration
@pytest.mark.parametrize("normal_axis", range(3), ids=("x", "y", "z"))
def test_subgrid_virtual_waveguide_inherits_fine_grid(
    monkeypatch, tmp_path, normal_axis
):
    captured = _capture_built_grids(monkeypatch)

    gprMax.run(
        scenes=[
            _uniform_fine_scene(
                normal_axis,
                virtual_waveguide=True,
            )
        ],
        geometry_only=True,
        outputfile=tmp_path / "uniform_virtual",
        cpu_precision="double",
        hide_progress_bars=True,
    )
    scene, _ = _subgrid_scene(
        normal_axis=normal_axis,
        virtual_waveguide=True,
    )
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "subgrid_virtual",
        subgrid=True,
        autotranslate=True,
        cpu_precision="double",
        hide_progress_bars=True,
    )

    uniform_grid = captured[0][0]
    fine_grid = captured[1][1]
    uniform_guide = uniform_grid.virtual_waveguides[0]
    fine_guide = fine_grid.virtual_waveguides[0]

    assert fine_guide.main_grid is fine_grid
    assert type(fine_guide.aux_grid) is type(uniform_guide.aux_grid)
    np.testing.assert_allclose(fine_guide.aux_grid.dl, fine_grid.dl)
    assert fine_guide.aux_grid.dt == pytest.approx(fine_grid.dt)
    assert fine_guide.aux_grid.iterations == fine_grid.iterations
    np.testing.assert_array_equal(fine_guide.aux_grid.size, uniform_guide.aux_grid.size)
    np.testing.assert_array_equal(fine_guide.aux_grid.solid, uniform_guide.aux_grid.solid)
    np.testing.assert_array_equal(fine_guide.aux_grid.ID, uniform_guide.aux_grid.ID)
    np.testing.assert_allclose(
        fine_guide.aux_grid.updatecoeffsE,
        uniform_guide.aux_grid.updatecoeffsE,
    )
    np.testing.assert_allclose(
        fine_guide.aux_grid.updatecoeffsH,
        uniform_guide.aux_grid.updatecoeffsH,
    )
    assert fine_guide.aux_source.plane_index == uniform_guide.aux_source.plane_index
    for fine_field, uniform_field in zip(
        (*fine_guide.aux_source.modal_e, *fine_guide.aux_source.modal_h),
        (*uniform_guide.aux_source.modal_e, *uniform_guide.aux_source.modal_h),
    ):
        np.testing.assert_allclose(fine_field, uniform_field, rtol=1e-11, atol=1e-9)


@pytest.mark.integration
def test_subgrid_virtual_waveguide_transient_matches_uniform_fine_grid(
    monkeypatch, tmp_path
):
    captured = _capture_built_grids(monkeypatch, uniform_source_stop_iteration=3)
    fine_iterations = 6
    gprMax.run(
        scenes=[
            _uniform_fine_scene(
                iterations=fine_iterations,
                passive_port=True,
                virtual_waveguide=True,
            )
        ],
        outputfile=tmp_path / "uniform_virtual_transient",
        cpu_precision="double",
        hide_progress_bars=True,
    )
    scene, _ = _subgrid_scene(
        iterations=fine_iterations // 3,
        virtual_waveguide=True,
    )
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "subgrid_virtual_transient",
        subgrid=True,
        autotranslate=True,
        cpu_precision="double",
        hide_progress_bars=True,
    )

    uniform_guide = captured[0][0].virtual_waveguides[0]
    fine_guide = captured[1][1].virtual_waveguides[0]
    for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        np.testing.assert_allclose(
            getattr(fine_guide.aux_grid, field_name),
            getattr(uniform_guide.aux_grid, field_name),
            rtol=1e-11,
            atol=1e-12,
            err_msg=field_name,
        )


@pytest.mark.integration
def test_subgrid_transient_modal_response_matches_uniform_fine_grid(
    monkeypatch, tmp_path
):
    """Compare identical fine updates before a boundary return can arrive."""

    # An HSG run performs ``ratio`` fine updates per main iteration but all
    # sources inherit the main grid's last output time. Give the uniform-fine
    # reference the same source stop time before comparing those updates.
    captured = _capture_built_grids(monkeypatch, uniform_source_stop_iteration=3)
    fine_iterations = 6
    uniform_output = tmp_path / "uniform_transient"
    subgrid_output = tmp_path / "subgrid_transient"
    gprMax.run(
        scenes=[
            _uniform_fine_scene(
                iterations=fine_iterations,
                passive_port=True,
            )
        ],
        outputfile=uniform_output,
        cpu_precision="double",
        hide_progress_bars=True,
    )
    scene, subgrid_object = _subgrid_scene(iterations=fine_iterations // 3)
    gprMax.run(
        scenes=[scene],
        outputfile=subgrid_output,
        subgrid=True,
        autotranslate=True,
        cpu_precision="double",
        hide_progress_bars=True,
    )

    assert subgrid_object.subgrid.iterations == fine_iterations
    uniform_grid = captured[0][0]
    fine_grid = captured[1][1]
    inner = np.asarray(
        (
            fine_grid.n_boundary_cells_x,
            fine_grid.n_boundary_cells_y,
            fine_grid.n_boundary_cells_z,
        )
    )
    outer = np.asarray(fine_grid.size) - inner
    cell_slices = tuple(slice(int(start), int(stop)) for start, stop in zip(inner, outer))
    component_slices = tuple(
        slice(int(start), int(stop + 1)) for start, stop in zip(inner, outer)
    )
    np.testing.assert_array_equal(fine_grid.solid[cell_slices], uniform_grid.solid)
    np.testing.assert_array_equal(
        fine_grid.ID[(slice(None), *component_slices)],
        uniform_grid.ID,
    )
    fine_slices = tuple(slice(int(start + 2), int(stop - 1)) for start, stop in zip(inner, outer))
    uniform_slices = (slice(2, -2),) * 3
    for field_name, fine_field, uniform_field in zip(
        ("Ey", "Ez", "Hx", "Hy", "Hz"),
        (fine_grid.Ey, fine_grid.Ez, fine_grid.Hx, fine_grid.Hy, fine_grid.Hz),
        (
            uniform_grid.Ey,
            uniform_grid.Ez,
            uniform_grid.Hx,
            uniform_grid.Hy,
            uniform_grid.Hz,
        ),
    ):
        np.testing.assert_allclose(
            fine_field[fine_slices],
            uniform_field[uniform_slices],
            rtol=1e-11,
            atol=1e-12,
            err_msg=field_name,
        )
    with (
        h5py.File(uniform_output.with_suffix(".h5"), "r") as uniform,
        h5py.File(subgrid_output.with_suffix(".h5"), "r") as refined,
    ):
        for port_name in ("port1", "port2"):
            uniform_port = uniform[f"eigenmode_ports/{port_name}"]
            refined_port = refined[f"subgrids/fine_grid/eigenmode_ports/{port_name}"]
            for dataset in ("incident", "outgoing"):
                np.testing.assert_allclose(
                    refined_port[dataset][...],
                    uniform_port[dataset][...],
                    rtol=1e-10,
                    atol=1e-20,
                )


@pytest.mark.integration
@pytest.mark.parametrize("virtual_waveguide", (False, True), ids=("direct", "virtual"))
def test_subgrid_eigenmode_source_runs_and_writes_fine_grid_port(
    tmp_path, virtual_waveguide
):
    scene, subgrid_object = _subgrid_scene(
        timewindow=5e-10,
        virtual_waveguide=virtual_waveguide,
    )
    variant = "virtual" if virtual_waveguide else "direct"
    outputfile = tmp_path / f"subgrid_eigenmode_{variant}"
    gprMax.run(
        scenes=[scene],
        outputfile=outputfile,
        subgrid=True,
        autotranslate=True,
        cpu_precision="double",
        hide_progress_bars=True,
    )
    assert (tmp_path / f"subgrid_eigenmode_{variant}_fine_grid_sparameters.csv").is_file()

    with h5py.File(outputfile.with_suffix(".h5"), "r") as output:
        group = output["subgrids/fine_grid/eigenmode_ports"]
        assert output["subgrids/fine_grid"].attrs["neigenmodeports"] == 2
        assert set(group) == {"port1", "port2"}
        for port in group.values():
            assert np.all(np.isfinite(port["incident"][...]))
            assert np.all(np.isfinite(port["outgoing"][...]))
        assert np.max(np.abs(group["port1/incident"][...])) > 0

    fine_grid = subgrid_object.subgrid
    assert all(
        port._next_iteration == fine_grid.iterations for port in fine_grid.eigenmodeports
    )
    assert max(
        float(np.max(np.abs(field)))
        for field in (
            fine_grid.Ex,
            fine_grid.Ey,
            fine_grid.Ez,
            fine_grid.Hx,
            fine_grid.Hy,
            fine_grid.Hz,
        )
    ) > 0
    if virtual_waveguide:
        assert len(fine_grid.virtual_waveguides) == 1
        guide = fine_grid.virtual_waveguides[0]
        assert guide.aux_grid.iterations == fine_grid.iterations
        assert max(
            float(np.max(np.abs(field)))
            for field in (
                guide.aux_grid.Ex,
                guide.aux_grid.Ey,
                guide.aux_grid.Ez,
                guide.aux_grid.Hx,
                guide.aux_grid.Hy,
                guide.aux_grid.Hz,
            )
        ) > 0
