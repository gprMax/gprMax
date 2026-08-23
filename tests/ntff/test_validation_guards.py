"""Regression tests for KSIR context, enclosure, and sampling guards."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gprMax
from gprMax.ntff.closures import ResolvedKSIRClosure
from gprMax.ntff.frequency_domain import validate_nyquist_frequencies
from gprMax.ntff.interface import (
    _associate_plane_wave,
    _validate_external_points,
    compile_ntff_outputs,
    validate_ntff_source_enclosure,
)
from gprMax.ntff.surfaces import COMPONENTS, build_component_surface

DL = 0.002
FREQUENCY = 5e9
SURFACE_LOWER = (0.012, 0.012, 0.012)
SURFACE_UPPER = (0.028, 0.028, 0.028)


def _base_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(DL,) * 3))
    scene.add(gprMax.Domain(p1=(0.04, 0.04, 0.04)))
    scene.add(gprMax.TimeWindow(time=1e-10))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=FREQUENCY, id="pulse"))
    return scene


def test_reusable_interface_rejects_geometry_fixed_up_front(tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.NTFFSurface(
            p1=SURFACE_LOWER,
            p2=SURFACE_UPPER,
            id="surface",
        )
    )
    scene.add(
        gprMax.KSIRTimeRx(
            position=(0.032, 0.02, 0.021),
            surface_id="surface",
            outputs=("Ez",),
        )
    )

    with pytest.raises(ValueError, match="does not support geometry-fixed runs"):
        gprMax.run(
            scenes=[scene],
            n=2,
            geometry_fixed=True,
            geometry_only=True,
            outputfile=tmp_path / "geometry_fixed_ksir",
            hide_progress_bars=True,
        )


@pytest.mark.parametrize("source_type", ["hertzian", "voltage"])
def test_reusable_surface_rejects_localised_source_outside(tmp_path, source_type):
    scene = _base_scene()
    if source_type == "hertzian":
        source = gprMax.HertzianDipole(
            polarisation="z", p1=(0.032, 0.02, 0.02), waveform_id="pulse"
        )
        expected = "HertzianDipole"
    else:
        source = gprMax.VoltageSource(
            polarisation="z",
            p1=(0.032, 0.02, 0.02),
            resistance=50,
            waveform_id="pulse",
        )
        expected = "VoltageSource"
    scene.add(source)
    scene.add(gprMax.NTFFSurface(p1=SURFACE_LOWER, p2=SURFACE_UPPER, id="surface"))
    scene.add(
        gprMax.KSIRTimeRx(
            position=(0.032, 0.02, 0.021),
            surface_id="surface",
            outputs=("Ez",),
        )
    )

    with pytest.raises(ValueError, match=expected):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / f"outside_{source_type}",
            hide_progress_bars=True,
        )


def test_enclosure_check_uses_source_position_after_src_steps(tmp_path):
    def stepped_scene():
        scene = _base_scene()
        scene.add(
            gprMax.HertzianDipole(
                polarisation="z",
                p1=(0.02, 0.02, 0.02),
                waveform_id="pulse",
            )
        )
        scene.add(gprMax.SrcSteps(p1=(0.012, 0, 0)))
        scene.add(gprMax.NTFFSurface(p1=SURFACE_LOWER, p2=SURFACE_UPPER, id="surface"))
        scene.add(
            gprMax.KSIRTimeRx(
                position=(0.032, 0.02, 0.021),
                surface_id="surface",
                outputs=("Ez",),
            )
        )
        return scene

    with pytest.raises(ValueError, match="HertzianDipole"):
        gprMax.run(
            scenes=[stepped_scene(), stepped_scene()],
            n=2,
            geometry_only=True,
            outputfile=tmp_path / "stepped_source",
            hide_progress_bars=True,
        )


def test_reusable_surface_rejects_eigenmode_injection_plane_outside():
    closure = ResolvedKSIRClosure("closed", (), (), True, True)
    surface = closure.apply_quadrature(
        build_component_surface("Ez", (5, 5, 5), (15, 15, 15), (0.01, 0.01, 0.01), (25, 25, 25))
    )
    source = SimpleNamespace(
        normal_axis=0,
        transverse_axes=(1, 2),
        transverse_start=np.asarray((8, 8)),
        transverse_stop=np.asarray((12, 12)),
        plane_index=18,
    )
    main_grid = SimpleNamespace(
        dl=np.asarray((0.01, 0.01, 0.01)),
        eigenmodesources=[source],
        discreteplanewaves=[],
    )
    monitor = SimpleNamespace(
        name="field-transform",
        allow_external_sources=False,
        surfaces={"Ez": surface},
        closure=closure,
    )
    model = SimpleNamespace(G=main_grid, subgrids=[])
    output_grid = SimpleNamespace(ntff_monitors=[monitor])

    with pytest.raises(ValueError, match="EigenmodePort"):
        validate_ntff_source_enclosure(model, output_grid)


def test_reusable_frequency_rejects_above_nyquist(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.NTFFSurface(p1=SURFACE_LOWER, p2=SURFACE_UPPER, id="surface"))
    scene.add(
        gprMax.KSIRFrequencyTransform(surface_id="surface", id="spectrum", frequencies=(1e15,))
    )

    with pytest.raises(ValueError, match="Nyquist limit"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "above_nyquist",
            hide_progress_bars=True,
        )


def test_nyquist_frequency_itself_is_valid_but_higher_is_not():
    dt = 2e-11
    nyquist = 0.5 / dt

    assert validate_nyquist_frequencies((0, nyquist), dt) == nyquist
    with pytest.raises(ValueError, match="Nyquist limit"):
        validate_nyquist_frequencies((nyquist * (1 + 1e-12),), dt)


def test_tfsf_correction_stencil_requires_one_cell_clearance():
    spacing = (0.01, 0.01, 0.01)
    shape = (25, 25, 25)
    plane_wave = SimpleNamespace(corners=(8, 8, 8, 12, 12, 12))
    grid = SimpleNamespace(discreteplanewaves=[plane_wave], dl=np.asarray(spacing))
    closure = ResolvedKSIRClosure("closed", (), (), True, True)

    class Monitor:
        def associate_plane_wave(self, source, dl, index):
            self.association = source, dl, index

    monitor = Monitor()
    clear_surfaces = {
        component: closure.apply_quadrature(
            build_component_surface(component, (5, 5, 5), (15, 15, 15), spacing, shape)
        )
        for component in COMPONENTS
    }
    _associate_plane_wave(
        monitor,
        clear_surfaces,
        np.asarray((5, 5, 5)),
        np.asarray((15, 15, 15)),
        grid,
        None,
    )
    assert monitor.association[0] is plane_wave
    assert monitor.association[2] == 0

    touching_surface = {
        "Ez": closure.apply_quadrature(
            build_component_surface("Ez", (7, 5, 5), (15, 15, 15), spacing, shape)
        )
    }
    with pytest.raises(ValueError, match="TFSF correction stencil"):
        _associate_plane_wave(
            Monitor(),
            touching_surface,
            np.asarray((7, 5, 5)),
            np.asarray((15, 15, 15)),
            grid,
            None,
        )


def test_exact_receiver_validation_uses_full_patch_support():
    closure = ResolvedKSIRClosure("closed", (), (), True, True)
    surface = build_component_surface(
        "Ez", (5, 5, 5), (15, 15, 15), (0.01, 0.01, 0.01), (25, 25, 25)
    )

    with pytest.raises(ValueError, match="strictly outside"):
        _validate_external_points(np.asarray(((0.046, 0.1, 0.1),)), {"Ez": surface}, closure)


def _antenna_gain_input(*, window="rectangular", association=None, extra_source=""):
    association_command = (
        "" if association is None else f"#ksir_antenna_ports: band {association}\n"
    )
    return (
        "#domain: 0.04 0.04 0.04\n"
        "#dx_dy_dz: 0.002 0.002 0.002\n"
        "#time_window: 1e-10\n"
        "#pml_cells: 2\n"
        "#waveform: ricker 1 5e9 pulse\n"
        "#voltage_source: z 0.018 0.02 0.02 50 pulse 0 1e-10 feed1 10\n"
        "#voltage_source: z 0.022 0.02 0.02 50 pulse 0 1e-10 feed2 10\n"
        f"{extra_source}"
        "#ntff_surface: 0.012 0.012 0.012 0.028 0.028 0.028 surface\n"
        f"#ksir_frequency: surface band 5e9 {window}\n"
        f"{association_command}"
        "#ksir_far_field: 90 0 band broadside gain\n"
    )


@pytest.mark.parametrize(
    ("window", "association", "extra_source", "message"),
    [
        ("rectangular", None, "", "without an antenna-port association"),
        ("hann", "feed1 feed2", "", "requires rectangular"),
        ("rectangular", "feed1", "", "include every physical port"),
        (
            "rectangular",
            "feed1 feed2",
            "#hertzian_dipole: z 0.02 0.02 0.02 pulse\n",
            "active non-port sources",
        ),
    ],
)
def test_antenna_gain_rejects_ambiguous_normalisation(
    tmp_path,
    window,
    association,
    extra_source,
    message,
):
    inputfile = tmp_path / "invalid_antenna_gain.in"
    inputfile.write_text(
        _antenna_gain_input(
            window=window,
            association=association,
            extra_source=extra_source,
        )
    )

    with pytest.raises(ValueError, match=message):
        gprMax.run(
            inputfile=str(inputfile),
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "invalid_antenna_gain",
            hide_progress_bars=True,
        )


def test_exterior_metrics_require_planar_layered_transform(tmp_path):
    inputfile = tmp_path / "invalid_exterior_metrics.in"
    inputfile.write_text(
        _antenna_gain_input().replace(
            "#ksir_far_field: 90 0 band broadside gain\n",
            "#ksir_far_field: 90 0 band broadside exterior_power\n",
        )
    )

    with pytest.raises(ValueError, match="exterior metrics from a non-layered transform"):
        gprMax.run(
            inputfile=str(inputfile),
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "invalid_exterior_metrics",
            hide_progress_bars=True,
        )


def test_eigenmode_antenna_transform_must_use_port_dft_subset(tmp_path):
    example = (
        Path(__file__).parents[2]
        / "examples"
        / "features"
        / "eigenmode_ports"
        / "example_3_antenna_and_farfield"
        / "horn_antenna.in"
    )
    inputfile = tmp_path / "mismatched_eigenmode_antenna.in"
    inputfile.write_text(
        example.read_text(encoding="utf-8").replace(
            "#ntff_frequency: horn_surface antenna_band 8e9",
            "#ntff_frequency: horn_surface antenna_band 8.1e9",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a subset of eigenmode port"):
        gprMax.run(
            inputfile=str(inputfile),
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "mismatched_eigenmode_antenna",
            hide_progress_bars=True,
        )
