"""Regression test for Model._check_stateful_sources_with_geometry_fixed()
(gprMax/model.py): geometry_fixed=True with more than one model requested
(n > 1) is rejected outright when the scene contains a #transmission_line
or a discrete plane wave command.

Neither source can meaningfully vary between geometry_fixed's reused
runs - TransmissionLine is explicitly excluded from #src_steps
repositioning, and a discrete plane wave's angle/direction is fixed once
at scene-parse time - and neither source's internal state (TL:
voltage/current/ABC history; DPW: its own internal 1D E/H-field and
PML-integral arrays) is reset between reused-geometry runs (only grid
fields/PMLs are, via Model.reuse_geometry()). So every run after the
first would silently repeat the identical source, contaminated by the
previous run's leftover state - not a real "reuse geometry, vary
something else" use case, just a broken repeat. This is rejected loudly
at build time instead.

A stepped receiver/dipole elsewhere in the same geometry_fixed scene is
NOT affected by this guard - only the presence of a stateful
TransmissionLine/DiscretePlaneWave triggers it.
"""
import pytest

import gprMax

INF = float("inf")


def _base_scene(domain=(0.02, 0.02, 0.02), dl=1e-3, time=1e-11):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=time))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="w"))
    return scene


def test_transmission_line_rejected_with_geometry_fixed_and_multiple_models(tmp_path):
    scene = _base_scene(time=1e-9)  # TL needs enough iterations for its internal line length
    scene.add(
        gprMax.TransmissionLine(
            polarisation="z", p1=(0.01, 0.01, 0.01), resistance=50, waveform_id="w"
        )
    )

    with pytest.raises(ValueError, match="#transmission_line"):
        gprMax.run(
            scenes=[scene], n=2, geometry_fixed=True, geometry_only=True,
            outputfile=tmp_path / "tl_geom_fixed", hide_progress_bars=True,
        )


def test_transmission_line_allowed_with_geometry_fixed_and_single_model(tmp_path):
    """n=1 is fine - there's no "run 2" to be contaminated by."""
    scene = _base_scene(time=1e-9)
    scene.add(
        gprMax.TransmissionLine(
            polarisation="z", p1=(0.01, 0.01, 0.01), resistance=50, waveform_id="w"
        )
    )

    gprMax.run(
        scenes=[scene], n=1, geometry_fixed=True, geometry_only=True,
        outputfile=tmp_path / "tl_single", hide_progress_bars=True,
    )


def test_transmission_line_allowed_without_geometry_fixed_multiple_models(tmp_path):
    """n>1 without geometry_fixed rebuilds fresh sources every run - fine.
    Without geometry_fixed, every model fetches its own scene by index, so
    (unlike the geometry_fixed case, which only ever needs the first) two
    separate scene instances must be provided for n=2."""

    def _tl_scene():
        scene = _base_scene(time=1e-9)
        scene.add(
            gprMax.TransmissionLine(
                polarisation="z", p1=(0.01, 0.01, 0.01), resistance=50, waveform_id="w"
            )
        )
        return scene

    gprMax.run(
        scenes=[_tl_scene(), _tl_scene()], n=2, geometry_only=True,
        outputfile=tmp_path / "tl_no_geom_fixed", hide_progress_bars=True,
    )


def test_discrete_plane_wave_rejected_with_geometry_fixed_and_multiple_models(tmp_path):
    scene = _base_scene(domain=(0.06, 0.06, 0.06))
    scene.add(
        gprMax.DiscretePlaneWaveAngles(
            p1=(0.015, 0.015, 0.015), p2=(0.045, 0.045, 0.045),
            theta=0, phi=0, psi=0, waveform_id="w",
        )
    )

    with pytest.raises(ValueError, match="plane wave"):
        gprMax.run(
            scenes=[scene], n=2, geometry_fixed=True, geometry_only=True,
            outputfile=tmp_path / "dpw_geom_fixed", hide_progress_bars=True,
        )


def test_discrete_plane_wave_allowed_with_geometry_fixed_and_single_model(tmp_path):
    scene = _base_scene(domain=(0.06, 0.06, 0.06))
    scene.add(
        gprMax.DiscretePlaneWaveAngles(
            p1=(0.015, 0.015, 0.015), p2=(0.045, 0.045, 0.045),
            theta=0, phi=0, psi=0, waveform_id="w",
        )
    )

    gprMax.run(
        scenes=[scene], n=1, geometry_fixed=True, geometry_only=True,
        outputfile=tmp_path / "dpw_single", hide_progress_bars=True,
    )


def test_stepped_dipole_with_geometry_fixed_and_multiple_models_still_works(tmp_path):
    """The guard must not over-reach: an ordinary geometry_fixed sweep
    with no TransmissionLine/DiscretePlaneWave present (just a stepped
    Hertzian dipole) is exactly the legitimate use case and must still
    be allowed."""
    scene = _base_scene()
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.01, 0.01, 0.01), waveform_id="w"))
    scene.add(gprMax.SrcSteps(p1=(0.001, 0.0, 0.0)))

    gprMax.run(
        scenes=[scene], n=2, geometry_fixed=True, geometry_only=True,
        outputfile=tmp_path / "dipole_geom_fixed", hide_progress_bars=True,
    )
