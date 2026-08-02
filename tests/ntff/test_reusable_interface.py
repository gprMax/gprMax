"""Python API coverage for reusable KSIR surfaces and grouped monitors."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax
from gprMax import config


@pytest.mark.parametrize("solver", ["cuda", "opencl", "metal"])
def test_time_receiver_registration_accepts_accelerator_backends(monkeypatch, solver):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            general={"solver": solver, "subgrid": False},
            mpi=False,
            args=SimpleNamespace(geometry_fixed=False),
            dtypes={"float_or_double": np.float32},
        ),
    )
    monkeypatch.setattr(config, "get_model_config", lambda: SimpleNamespace(mode="3D"))
    grid = SimpleNamespace(
        ksir_surface_specs={"surface": object()},
        ksir_time_requests=[],
        ksir_request_owners={},
    )
    receiver = gprMax.KSIRTimeRx(
        position=(0.1, 0.2, 0.3),
        surface_id="surface",
        id="point",
        outputs=("Ez",),
    )

    receiver.build(None, grid)

    assert grid.ksir_time_requests[0].output_id == "point"
    assert grid.ksir_request_owners["time:surface:point"] is receiver


def test_python_api_groups_requests_and_exposes_results(tmp_path):
    dl = 0.004
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.08, 0.08, 0.08)))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.PMLThickness(thickness=3))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.04, 0.04, 0.04), waveform_id="pulse"))
    scene.add(
        gprMax.KSIRSurface(
            p1=(0.028, 0.028, 0.028),
            p2=(0.052, 0.052, 0.052),
            id="surface",
        )
    )
    transform = gprMax.KSIRFrequencyTransform(
        surface_id="surface", id="spectrum", frequencies=(5e9,)
    )
    scene.add(transform)
    time_cartesian = gprMax.KSIRTimeRx(
        position=(0.064, 0.04, 0.042),
        surface_id="surface",
        id="time_cartesian",
        outputs=("Ez",),
    )
    time_spherical = gprMax.KSIRTimeRxSpherical(
        0.03,
        90,
        0,
        "surface",
        id="time_spherical",
        outputs=("Etheta",),
    )
    frequency_receiver = gprMax.KSIRFrequencyRx(
        position=(0.064, 0.04, 0.042),
        transform_id="spectrum",
        id="frequency_receiver",
        outputs=("Ez",),
    )
    far_field = gprMax.KSIRFarField(
        theta=(0, 90, 180),
        phi=(0, 0, 0),
        transform_id="spectrum",
        id="far_field",
        outputs=("Etheta", "Ephi", "directivity", "directivity_dbi"),
    )
    for item in (time_cartesian, time_spherical, frequency_receiver, far_field):
        scene.add(item)

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=tmp_path / "reusable_api",
        hide_progress_bars=True,
        cpu_precision="double",
    )

    assert time_cartesian.result.fields["Ez"].shape[0] == 1
    assert time_spherical.result.fields["Etheta"].shape[0] == 1
    assert np.all(
        time_cartesian.result.fully_supported_lengths <= time_cartesian.result.valid_lengths
    )
    assert time_cartesian.result.point_field("Ez", 0).size == int(
        time_cartesian.result.fully_supported_lengths[0]
    )
    assert time_cartesian.result.point_raw_field("Ez", 0).size == int(
        time_cartesian.result.valid_lengths[0]
    )
    assert time_cartesian.result.terminal_field_ratios.shape == (1,)
    assert time_cartesian.result.terminal_decay_ok.shape == (1,)
    assert frequency_receiver.result.fields["Ez"].shape == (1, 1)
    assert far_field.result.fields["Etheta"].shape == (1, 3)
    assert not frequency_receiver.result.range_normalized
    assert far_field.result.range_normalized
    assert np.isfinite(frequency_receiver.result.fields["Ez"]).all()
    assert np.isfinite(far_field.result.fields["Etheta"]).all()
    assert np.isfinite(far_field.result.fields["directivity"]).all()
    assert far_field.result.radiation_metrics is not None
    assert transform.surface_data["Ez"].field.dtype == np.complex128

    writer = time_cartesian._compiled_outputs
    monitor_a, _ = writer.time_bindings[time_cartesian._request_key]
    monitor_b, _ = writer.time_bindings[time_spherical._request_key]
    assert monitor_a is monitor_b
