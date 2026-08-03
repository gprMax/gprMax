import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.config as config
import gprMax.sources as sources_module
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
    scene.add(
        gprMax.EigenmodeSource(
            normal="x",
            direction="+",
            p1=(0.005, 0),
            p2=(0.045, INF),
            w=0.015,
            mode_index=0,
            frequency=5e9,
            waveform_id="eig_pulse",
        )
    )
    scene.add(gprMax.Rx(p1=(0.035, 0.025, INF)))
    return scene


def _user_waveform_broadband_scene(direction):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.06, 0.05, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-9))
    scene.add(
        gprMax.Waveform(
            wave_type="user",
            user_func=lambda time: np.sin(2 * np.pi * 5e9 * time)
            * np.exp(-(((time - 0.5e-9) / 0.15e-9) ** 2)),
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
    scene.add(
        gprMax.EigenmodeSource(
            normal="x",
            direction=direction,
            p1=(0.005, 0),
            p2=(0.045, INF),
            w=0.015,
            mode_index=0,
            frequencies=(4e9, 5e9, 7e9),
            waveform_id="eig_pulse",
        )
    )
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
    scene.add(
        gprMax.EigenmodeSource(
            normal="x",
            direction="+",
            p1=(0.005, 0),
            p2=(0.075, INF),
            w=0.015,
            mode_index=0,
            frequency=5e9,
            waveform_id="eig_pulse",
        )
    )
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
    scene.add(
        gprMax.EigenmodeSource(
            normal="x",
            direction="+",
            p1=(0.005, 0),
            p2=(0.045, INF),
            w=0.015,
            mode_index=0,
            frequency=5e9,
            waveform_id="eig_pulse",
        )
    )
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
    assert not list(tmp_path.glob("*_eigenmode_*_fields.png"))

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
        captured.update(
            residual=source.complex_profile_residual,
            uses_quadrature=source.uses_quadrature,
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


def test_eigenmode_source_rejected_with_mpi(monkeypatch):
    # build() checks MPI before touching grid at all, so a bare object
    # stands in for the grid here - mirrors the fast, direct-build style
    # used for #magnetic_frill_source's own MPI-rejection test.
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.general = {"solver": "cpu"}
    config.sim_config.mpi = True
    source = gprMax.EigenmodeSource(
        normal="x",
        direction="+",
        p1=(0.005, 0),
        p2=(0.045, INF),
        w=0.015,
        mode_index=0,
        frequency=5e9,
        waveform_id="eig_pulse",
    )
    with pytest.raises(ValueError, match="MPI"):
        source.build(grid=None)


def test_2d_eigenmode_normal_cannot_be_invariant_axis(tmp_path):
    scene = _scene("TM")
    scene.grid_objects = [
        obj for obj in scene.grid_objects if not isinstance(obj, gprMax.EigenmodeSource)
    ]
    scene.add(
        gprMax.EigenmodeSource(
            normal="z",
            direction="+",
            p1=(0.005, 0.005),
            p2=(0.045, 0.045),
            w=INF,
            mode_index=0,
            frequency=5e9,
            waveform_id="eig_pulse",
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
    scene.grid_objects = [
        obj for obj in scene.grid_objects if not isinstance(obj, gprMax.EigenmodeSource)
    ]
    scene.add(
        gprMax.EigenmodeSource(
            normal="x",
            direction="+",
            p1=(0.005, 0),
            p2=(0.045, INF),
            w=0,
            mode_index=0,
            frequency=5e9,
            waveform_id="eig_pulse",
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
    assert captured["+"]["representative_frequency"] == pytest.approx(5e9, rel=0.05)
    assert captured["-"]["representative_frequency"] == pytest.approx(5e9, rel=0.05)


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
    source_transverse_axes = [axis for axis in range(3) if axis != normal_axis]
    scene.add(
        gprMax.EigenmodeSource(
            normal=letters[normal_axis],
            direction="+",
            p1=tuple(full_lower[axis] for axis in source_transverse_axes),
            p2=tuple(full_upper[axis] for axis in source_transverse_axes),
            w=0.01,
            mode_index=0,
            frequency=10e9,
            waveform_id="w",
        )
    )

    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / f"{mode}_{letters[invariant_axis]}",
        hide_progress_bars=True,
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("tm/pec_waveguide/pec_waveguide.in"),
        Path("te/pec_waveguide/pec_waveguide.in"),
        Path("tm/dielectric_slab/dielectric_slab.in"),
        Path("te/pmc_waveguide/pmc_waveguide.in"),
        Path("tm/dielectric_bend/dielectric_bend.in"),
        Path("te/dielectric_bend/dielectric_bend.in"),
    ],
)
def test_2d_example_builds_with_eight_timestamps(tmp_path, relative_path):
    source = (
        REPOSITORY_ROOT
        / "testing"
        / "regression"
        / "eigenmode_sources"
        / "cases"
        / "two_dimensional"
        / relative_path
    )
    copied_input = tmp_path / source.name
    shutil.copyfile(source, copied_input)

    assert copied_input.read_text().count("#snapshot:") == 8
    gprMax.run(
        inputfile=copied_input,
        n=1,
        geometry_only=True,
        outputfile=tmp_path / source.stem,
        hide_progress_bars=True,
    )
    assert len(list(tmp_path.glob(f"{source.stem}_eigenmode_*_fields.png"))) == 1


@pytest.mark.parametrize(
    ("plot_control", "expected_plot_count"),
    [("n", 0), ("y", 1)],
)
def test_hash_modal_plot_control_overrides_geometry_only_default(
    tmp_path,
    plot_control,
    expected_plot_count,
):
    source = (
        REPOSITORY_ROOT
        / "testing"
        / "regression"
        / "eigenmode_sources"
        / "cases"
        / "two_dimensional"
        / "tm"
        / "pec_waveguide"
        / "pec_waveguide.in"
    )
    copied_input = tmp_path / f"pec_waveguide_{plot_control}.in"
    lines = source.read_text().splitlines()
    copied_input.write_text(
        "\n".join(
            f"{line.rsplit(maxsplit=1)[0]} {plot_control}"
            if line.startswith("#eigenmode_source:")
            else line
            for line in lines
        )
        + "\n"
    )

    gprMax.run(
        inputfile=copied_input,
        n=1,
        geometry_only=True,
        outputfile=tmp_path / f"pec_waveguide_{plot_control}",
        hide_progress_bars=True,
    )

    assert (
        len(list(tmp_path.glob(f"{copied_input.stem}_eigenmode_*_fields.png")))
        == expected_plot_count
    )
