import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.config as config
import gprMax.sources as sources_module
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.sources import EigenmodeSource as RuntimeEigenmodeSource

INF = float("inf")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _scene(mode):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.06, 0.05, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=0.12e-9))
    scene.add(gprMax.Waveform(wave_type="contsine", amp=1, freq=5e9, id="eig_pulse"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.06, 0.005, INF), material_id="pec"))
    scene.add(gprMax.Box(p1=(0, 0.045, 0), p2=(0.06, 0.05, INF), material_id="pec"))
    scene.add(gprMax.EigenmodeBand(id="band", fmin=5e9, fmax=5e9, points=1))
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.015, 0.005, 0),
            p2=(0.015, 0.045, INF),
            direction="+",
            modes=(1,),
            anchors=(5e9,),
        )
    )
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="eig_pulse"))
    scene.add(gprMax.Rx(p1=(0.035, 0.025, INF)))
    return scene


def _user_waveform_broadband_scene(direction):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.06, 0.05, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=4e-9))
    scene.add(
        gprMax.Waveform(
            wave_type="user",
            user_func=lambda time: np.sin(2 * np.pi * 5.5e9 * time)
            * np.exp(-(((time - 2e-9) / 0.5e-9) ** 2)),
            id="eig_pulse",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(0.06, 0.005, INF),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0, 0.045, 0),
            p2=(0.06, 0.05, INF),
            material_id="pec",
        )
    )
    scene.add(gprMax.EigenmodeBand(id="band", fmin=4e9, fmax=7e9, points=31))
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.015, 0.005, 0),
            p2=(0.015, 0.045, INF),
            direction=direction,
            modes=(1,),
            anchors=(3.8e9, 5.5e9, 7.2e9),
        )
    )
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="eig_pulse"))
    return scene


def _dielectric_scene(conductivity=0):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.08, 0.08, INF)))
    scene.add(gprMax.PMLThickness(thickness=(5, 5, 0, 5, 5, 0)))
    scene.add(gprMax.TimeWindow(time=0.14e-9))
    scene.add(
        gprMax.Material(
            er=9,
            se=conductivity,
            mr=1,
            sm=0,
            id="slab_core",
        )
    )
    scene.add(gprMax.Waveform(wave_type="contsine", amp=1, freq=5e9, id="eig_pulse"))
    scene.add(
        gprMax.Box(
            p1=(0, 0.03, 0),
            p2=(0.08, 0.05, INF),
            material_id="slab_core",
        )
    )
    scene.add(gprMax.EigenmodeBand(id="band", fmin=5e9, fmax=5e9, points=1))
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.015, 0.005, 0),
            p2=(0.015, 0.075, INF),
            direction="+",
            modes=(1,),
            anchors=(5e9,),
        )
    )
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="eig_pulse"))
    scene.add(gprMax.Rx(p1=(0.05, 0.04, INF)))
    return scene


def _pmc_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.06, 0.05, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=0.12e-9))
    scene.add(gprMax.Waveform(wave_type="contsine", amp=1, freq=5e9, id="eig_pulse"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.06, 0.005, INF), material_id="pmc"))
    scene.add(gprMax.Box(p1=(0, 0.045, 0), p2=(0.06, 0.05, INF), material_id="pmc"))
    scene.add(gprMax.EigenmodeBand(id="band", fmin=5e9, fmax=5e9, points=1))
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.015, 0.005, 0),
            p2=(0.015, 0.045, INF),
            direction="+",
            modes=(1,),
            anchors=(5e9,),
        )
    )
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="eig_pulse"))
    scene.add(gprMax.Rx(p1=(0.035, 0.025, INF)))
    return scene


@pytest.mark.parametrize(
    ("mode", "live_components", "dead_components"),
    [
        ("TM", ("Ez", "Hx", "Hy"), ("Ex", "Ey", "Hz")),
        ("TE", ("Ey", "Hz"), ("Ez", "Hx", "Hy")),
    ],
)
@pytest.mark.parametrize("cpu_precision", ["single", "double"])
def test_2d_eigenmode_injection_updates_only_live_system(
    tmp_path,
    mode,
    live_components,
    dead_components,
    cpu_precision,
):
    output = tmp_path / f"eigenmode_{mode.lower()}_{cpu_precision}"
    gprMax.run(
        scenes=[_scene(mode)],
        n=1,
        outputfile=output,
        cpu_precision=cpu_precision,
        hide_progress_bars=True,
    )
    assert not list(tmp_path.glob("*_Port*_Mode*.png"))
    assert not list(tmp_path.glob("*_EigenmodeExcitation.png"))

    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        receiver = handle["rxs/rx1"]
        for component in live_components:
            assert np.max(np.abs(receiver[component][...])) > 0
        for component in dead_components:
            assert np.max(np.abs(receiver[component][...])) == 0


def test_2d_dielectric_mode_decays_before_source_boundary(tmp_path):
    output = tmp_path / "eigenmode_dielectric"
    scene = _dielectric_scene()
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
    )

    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        assert np.max(np.abs(handle["rxs/rx1/Ez"][...])) > 0


@pytest.mark.parametrize(
    ("conductivity", "expected_quadrature"),
    [(0, False), (2, True)],
)
def test_single_frequency_real_solver_selects_complex_profile_path(
    tmp_path,
    monkeypatch,
    conductivity,
    expected_quadrature,
):
    captured = {}
    original_prepare = RuntimeEigenmodeSource._prepare_single_frequency_injection

    def capture_prepared_source(source, grid):
        original_prepare(source, grid)
        neff = complex(source.complex_neff)
        beta = 2 * np.pi * source.frequency * neff / config.sim_config.em_consts["c"]
        forward_factor = np.exp(-1j * beta * grid.dl[source.normal_axis] / 2)
        captured.update(
            residual=source.complex_profile_residual,
            uses_quadrature=source.uses_quadrature,
            neff=neff,
            forward_factor=forward_factor,
            modal_power=np.real(source._modal_cross_power(source.modal_e, source.modal_h, grid)),
        )

    monkeypatch.setattr(
        RuntimeEigenmodeSource,
        "_prepare_single_frequency_injection",
        capture_prepared_source,
    )
    output = tmp_path / f"single_frequency_phase_{conductivity:g}"
    gprMax.run(
        scenes=[_dielectric_scene(conductivity)],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
    )

    assert captured["uses_quadrature"] is expected_quadrature
    assert np.real(captured["neff"]) > 0
    assert captured["modal_power"] > 0
    if conductivity:
        assert np.imag(captured["neff"]) < 0
        assert abs(captured["forward_factor"]) < 1
    else:
        assert np.imag(captured["neff"]) == pytest.approx(0, abs=1e-12)
        assert abs(captured["forward_factor"]) == pytest.approx(1)

    if expected_quadrature:
        assert captured["residual"] > RuntimeEigenmodeSource.COMPLEX_PROFILE_TOLERANCE
    else:
        assert captured["residual"] <= RuntimeEigenmodeSource.COMPLEX_PROFILE_TOLERANCE
    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        assert np.max(np.abs(handle["rxs/rx1/Ez"][...])) > 0


def test_2d_pmc_mode_enforces_magnetic_wall_and_injects(tmp_path):
    output = tmp_path / "eigenmode_pmc"
    scene = _pmc_scene()
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
    )

    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        assert np.max(np.abs(handle["rxs/rx1/Hz"][...])) > 0


def test_eigenmode_port_rejected_with_mpi(monkeypatch):
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.general = {"solver": "cpu"}
    config.sim_config.mpi = True
    grid = FDTDGrid()
    grid.eigenmodeband = object()
    port = gprMax.EigenmodePort(
        port=1,
        p1=(0.015, 0.005, 0),
        p2=(0.015, 0.045, INF),
        direction="+",
        modes=(1,),
    )
    with pytest.raises(ValueError, match="MPI"):
        port.build(grid)


def test_2d_eigenmode_normal_cannot_be_invariant_axis(tmp_path):
    scene = _scene("TM")
    scene.grid_objects = [obj for obj in scene.grid_objects if not isinstance(obj, gprMax.EigenmodePort)]
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.005, 0.005, INF),
            p2=(0.045, 0.045, INF),
            direction="+",
            modes=(1,),
            anchors=(5e9,),
        )
    )
    with pytest.raises(ValueError):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "bad_normal",
            hide_progress_bars=True,
        )


def test_positive_direction_eigenmode_rejects_lower_boundary(tmp_path):
    scene = _scene("TM")
    scene.grid_objects = [obj for obj in scene.grid_objects if not isinstance(obj, gprMax.EigenmodePort)]
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0, 0.005, 0),
            p2=(0, 0.045, INF),
            direction="+",
            modes=(1,),
            anchors=(5e9,),
        )
    )

    with pytest.raises(
        ValueError,
        match="at least one cell inside the lower domain boundary",
    ):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "lower_boundary_source",
            hide_progress_bars=True,
        )


def test_real_solver_negative_broadband_power_and_user_waveform(tmp_path, monkeypatch):
    captured = {}
    warnings = []
    active_direction = {"value": None}
    original_prepare = RuntimeEigenmodeSource._prepare_broadband_time_traces

    def capture_prepared_source(source, grid, frequencies):
        original_prepare(source, grid, frequencies)
        direction_sign = 1 if source.direction == "+" else -1
        requested_power = np.asarray(
            [
                np.real(
                    source._modal_cross_power(
                        electric,
                        [direction_sign * field for field in magnetic],
                        grid,
                    )
                )
                for electric, magnetic in zip(source.anchor_modal_e, source.anchor_modal_h)
            ]
        )
        captured[source.direction] = {
            "requested_power": requested_power,
            "representative_frequency": source.representative_frequency,
        }

    def capture_warning(message, *args, **kwargs):
        del kwargs
        rendered = message % args if args else str(message)
        warnings.append((active_direction["value"], rendered))

    monkeypatch.setattr(
        RuntimeEigenmodeSource,
        "_prepare_broadband_time_traces",
        capture_prepared_source,
    )
    monkeypatch.setattr(sources_module.logger, "warning", capture_warning)

    for direction in ("+", "-"):
        active_direction["value"] = direction
        gprMax.run(
            scenes=[_user_waveform_broadband_scene(direction)],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / f"broadband_{direction}",
            hide_progress_bars=True,
        )

    negative_warnings = [message for direction, message in warnings if direction == "-"]
    assert not any("fallback normalization" in message for message in negative_warnings)
    assert np.all(captured["+"]["requested_power"] > 0)
    assert np.all(captured["-"]["requested_power"] < 0)
    assert np.abs(captured["-"]["requested_power"]) == pytest.approx(
        np.abs(captured["+"]["requested_power"]),
        rel=1e-6,
    )
    assert captured["+"]["representative_frequency"] == pytest.approx(5.5e9, rel=0.05)
    assert captured["-"]["representative_frequency"] == pytest.approx(5.5e9, rel=0.05)


@pytest.mark.parametrize("mode", ["TM", "TE"])
@pytest.mark.parametrize("invariant_axis", [0, 1, 2])
def test_2d_eigenmode_builds_for_every_invariant_axis(tmp_path, mode, invariant_axis):
    letters = "xyz"
    normal_axis = (invariant_axis + 1) % 3
    transverse_axis = (invariant_axis + 2) % 3
    domain = [0.04, 0.04, 0.04]
    domain[invariant_axis] = INF

    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=tuple(domain)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-12))
    scene.add(gprMax.Waveform(wave_type="contsine", amp=1, freq=10e9, id="w"))

    lower_wall_end = [0.04, 0.04, 0.04]
    lower_wall_end[invariant_axis] = INF
    lower_wall_end[transverse_axis] = 0.005
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=tuple(lower_wall_end), material_id="pec"))
    upper_wall_start = [0, 0, 0]
    upper_wall_end = [0.04, 0.04, 0.04]
    upper_wall_start[transverse_axis] = 0.035
    upper_wall_end[invariant_axis] = INF
    scene.add(
        gprMax.Box(
            p1=tuple(upper_wall_start),
            p2=tuple(upper_wall_end),
            material_id="pec",
        )
    )

    full_lower = [0, 0, 0]
    full_upper = [0, 0, 0]
    full_lower[normal_axis] = full_upper[normal_axis] = 0.01
    full_lower[transverse_axis] = 0.005
    full_upper[transverse_axis] = 0.035
    full_lower[invariant_axis] = 0
    full_upper[invariant_axis] = INF
    scene.add(gprMax.EigenmodeBand(id="band", fmin=10e9, fmax=10e9, points=1))
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=tuple(full_lower),
            p2=tuple(full_upper),
            direction="+",
            modes=(1,),
            anchors=(10e9,),
        )
    )
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="w"))

    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / f"{mode}_{letters[invariant_axis]}",
        hide_progress_bars=True,
    )


@pytest.mark.parametrize(
    ("relative_path", "snapshot_count"),
    [
        (
            Path("straight_waveguide/2d_tm/dielectric_waveguide/dielectric_waveguide.in"),
            4,
        ),
        (
            Path("straight_waveguide/2d_te/dielectric_waveguide/dielectric_waveguide.in"),
            4,
        ),
        (Path("bending_waveguide/2d_tm/small_bend/small_bend.in"), 4),
        (Path("bending_waveguide/2d_tm/medium_bend/medium_bend.in"), 4),
        (Path("bending_waveguide/2d_tm/large_bend/large_bend.in"), 4),
        (Path("bending_waveguide/2d_te/small_bend/small_bend.in"), 4),
        (Path("bending_waveguide/2d_te/medium_bend/medium_bend.in"), 4),
        (Path("bending_waveguide/2d_te/large_bend/large_bend.in"), 4),
        (Path("loss_comparison/nonlossy/nonlossy.in"), 2),
        (Path("loss_comparison/lossy/lossy.in"), 2),
        (Path("broadband_vs_single_frequency/broadband/broadband.in"), 2),
        (
            Path("broadband_vs_single_frequency/single_frequency/single_frequency.in"),
            2,
        ),
    ],
)
def test_2d_regression_example_builds(tmp_path, relative_path, snapshot_count):
    source = REPOSITORY_ROOT / "testing" / "regression" / "eigenmode_sources" / "cases" / relative_path
    copied_input = tmp_path / source.name
    shutil.copyfile(source, copied_input)

    assert copied_input.read_text().count("#snapshot:") == snapshot_count
    gprMax.run(
        inputfile=copied_input,
        n=1,
        geometry_only=True,
        outputfile=tmp_path / source.stem,
        hide_progress_bars=True,
    )
    assert list(tmp_path.glob(f"{source.stem}_Port*_Mode*.png"))
    assert (tmp_path / f"{source.stem}_EigenmodeExcitation.png").is_file()


@pytest.mark.parametrize(
    ("plot_control", "expected_plot_count"),
    [("n", 0), ("y", 4)],
)
def test_hash_modal_plot_control_overrides_geometry_only_default(
    tmp_path,
    plot_control,
    expected_plot_count,
):
    source = REPOSITORY_ROOT / "examples" / "features" / "eigenmode_sources" / "dielectric_slab_2d_tm.in"
    copied_input = tmp_path / f"dielectric_slab_{plot_control}.in"
    lines = source.read_text().splitlines()
    copied_input.write_text(
        "\n".join(
            f"{line} {plot_control}"
            if line.startswith("#eigenmode_port:")
            else line
            for line in lines
        )
        + "\n"
    )

    gprMax.run(
        inputfile=copied_input,
        n=1,
        geometry_only=True,
        outputfile=tmp_path / f"dielectric_slab_{plot_control}",
        hide_progress_bars=True,
    )

    plot_count = len(list(tmp_path.glob(f"{copied_input.stem}_Port*_Mode*.png")))
    assert plot_count == expected_plot_count
    waveform_plot_count = len(list(tmp_path.glob(f"{copied_input.stem}_EigenmodeExcitation.png")))
    assert waveform_plot_count == 1


@pytest.mark.parametrize(
    ("plot_control", "expected_plot_count"),
    [("n", 0), ("y", 1)],
)
def test_hash_excitation_plot_control_is_independent_of_modal_plots(
    tmp_path,
    plot_control,
    expected_plot_count,
):
    source = (
        REPOSITORY_ROOT
        / "examples"
        / "features"
        / "eigenmode_sources"
        / "dielectric_slab_2d_tm.in"
    )
    copied_input = tmp_path / f"dielectric_slab_excitation_{plot_control}.in"
    lines = source.read_text().splitlines()
    copied_input.write_text(
        "\n".join(
            f"{line} n"
            if line.startswith("#eigenmode_port:")
            else f"{line} {plot_control}"
            if line.startswith("#eigenmode_excitation:")
            else line
            for line in lines
        )
        + "\n"
    )

    gprMax.run(
        inputfile=copied_input,
        n=1,
        geometry_only=True,
        outputfile=tmp_path / copied_input.stem,
        hide_progress_bars=True,
    )

    assert not list(tmp_path.glob(f"{copied_input.stem}_Port*_Mode*.png"))
    waveform_plot_count = len(
        list(tmp_path.glob(f"{copied_input.stem}_EigenmodeExcitation.png"))
    )
    assert waveform_plot_count == expected_plot_count
