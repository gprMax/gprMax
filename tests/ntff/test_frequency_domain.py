from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import c

from gprMax.ntff.closures import ExperimentalMask, resolve_closure
from gprMax.ntff.conventions import engineering_dft
from gprMax.ntff.frequency_domain import (
    KSIRFrequencyDomainMonitor,
    evaluate_saved_surface_dft,
    validate_nyquist_frequencies,
)
from gprMax.ntff.surfaces import FACES, build_component_surface

REAL_DTYPE = np.dtype(float)
COMPLEX_DTYPE = np.dtype(complex)


def _material(num_id=1):
    return SimpleNamespace(
        numID=num_id,
        ID="free_space",
        er=1.0,
        mr=1.0,
        se=0.0,
        sm=0.0,
        poles=0,
    )


def _configured_monitor(
    component,
    surface,
    frequencies,
    dt,
    iterations,
    *,
    name="monitor",
    incident_surface_file=None,
    incident_monitor_name=None,
    real_dtype=REAL_DTYPE,
    complex_dtype=COMPLEX_DTYPE,
):
    monitor = KSIRFrequencyDomainMonitor(
        name,
        {component: surface},
        frequencies,
        [30.0, 90.0],
        [10.0, 70.0],
        dt,
        iterations,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        incident_surface_file=incident_surface_file,
        incident_monitor_name=incident_monitor_name,
    )
    ids = np.ones((6,) + surface.field_shape, dtype=np.uint32)
    monitor.validate_materials(ids, {component: 0})
    monitor.configure_background([_material()])
    return monitor


@pytest.mark.parametrize(
    "component,time_offset",
    [("Ex", 0.0), ("Hx", 0.5)],
)
def test_streaming_surface_dft_matches_reference_engineering_transform(
    component, time_offset
):
    iterations = 48
    dt = 2e-11
    frequencies = np.array([3 / (iterations * dt), 7 / (iterations * dt)])
    surface = build_component_surface(
        component, (2, 2, 2), (5, 6, 4), (0.1, 0.12, 0.08), (9, 10, 8)
    )
    monitor = _configured_monitor(
        component, surface, frequencies, dt, iterations
    )
    fields = []
    omega = 2 * np.pi * frequencies[0]
    spatial = np.indices(surface.field_shape).sum(axis=0) + 1.0
    zeros = np.zeros(surface.field_shape)
    for iteration in range(iterations):
        time = (iteration + time_offset) * dt
        field = spatial * np.cos(omega * time + 0.31)
        fields.append(field)
        if component == "Ex":
            monitor.observe_electric(iteration, field, zeros, zeros)
        else:
            monitor.observe_magnetic(iteration, field, zeros, zeros)
    monitor.finalise()

    inside_series = []
    outside_series = []
    for field in fields:
        inside = []
        outside = []
        for face in surface.faces:
            face_inside, face_outside = face.sample(field)
            inside.append(face_inside)
            outside.append(face_outside)
        inside_series.append(np.concatenate(inside))
        outside_series.append(np.concatenate(outside))
    inside_dft = engineering_dft(
        np.asarray(inside_series),
        frequencies,
        dt,
        time_offset=time_offset * dt,
    )
    outside_dft = engineering_dft(
        np.asarray(outside_series),
        frequencies,
        dt,
        time_offset=time_offset * dt,
    )
    expected_field = 0.5 * (outside_dft + inside_dft)
    derivative_parts = []
    start = 0
    for face in surface.faces:
        stop = start + face.npatches
        derivative_parts.append(
            (outside_dft[:, start:stop] - inside_dft[:, start:stop])
            / face.normal_spacing
        )
        start = stop
    expected_derivative = np.concatenate(derivative_parts, axis=1)

    actual = monitor.surface_data[component]
    assert_allclose(actual.field, expected_field, rtol=3e-14, atol=1e-20)
    assert_allclose(
        actual.normal_derivative, expected_derivative, rtol=3e-14, atol=1e-20
    )


def _observe_ex_monitor(monitor, surface, iterations, frequency):
    indices = np.indices(surface.field_shape)
    spatial = 0.7 + indices[0] - 0.4 * indices[1] + 0.2 * indices[2]
    zeros = np.zeros(surface.field_shape)
    for iteration in range(iterations):
        time = iteration * monitor.dt
        field = spatial * np.cos(2 * np.pi * frequency * time + 0.2)
        monitor.observe_electric(iteration, field, zeros, zeros)
    monitor.finalise()


def test_saved_surface_dft_round_trip_and_incident_subtraction(tmp_path):
    iterations = 40
    dt = 1e-11
    frequency = 5 / (iterations * dt)
    surface = build_component_surface(
        "Ex", (2, 2, 2), (5, 5, 5), (0.02, 0.02, 0.02), (9, 9, 9)
    )
    reference = _configured_monitor(
        "Ex", surface, [frequency], dt, iterations, name="reference"
    )
    _observe_ex_monitor(reference, surface, iterations, frequency)
    filename = tmp_path / "reference.h5"
    with h5py.File(filename, "w") as output:
        reference.write_hdf5(output)

    with h5py.File(filename, "r") as output:
        group = output["ntff/reference"]
        assert group.attrs["forward_transform_sign"] == "exp(-j*omega*t)"
        assert "surface/Ex/psi_dft" in group
        assert group["surface/Ex/psi_dft"].shape == (1, surface.npatches)

    reevaluated = evaluate_saved_surface_dft(
        filename, "reference", [30.0, 90.0], [10.0, 70.0]
    )
    assert_allclose(
        reevaluated.range_normalized_fields["Ex"],
        reference.result.range_normalized_fields["Ex"],
    )
    new_directions = evaluate_saved_surface_dft(
        filename, "reference", [45.0], [0.0, 90.0, 180.0]
    )
    assert new_directions.range_normalized_fields["Ex"].shape == (1, 3)

    target = _configured_monitor(
        "Ex",
        surface,
        [frequency],
        dt,
        iterations,
        name="target",
        incident_surface_file=filename,
        incident_monitor_name="reference",
    )
    _observe_ex_monitor(target, surface, iterations, frequency)

    assert_allclose(target.surface_data["Ex"].field, 0, atol=1e-25)
    assert_allclose(target.surface_data["Ex"].normal_derivative, 0, atol=1e-24)
    assert_allclose(target.result.range_normalized_fields["Ex"], 0, atol=1e-24)


def test_incident_surface_subtraction_rejects_precision_mismatch(tmp_path):
    iterations = 8
    dt = 1e-11
    frequency = 2 / (iterations * dt)
    surface = build_component_surface(
        "Ex", (2, 2, 2), (5, 5, 5), (0.02,) * 3, (9, 9, 9)
    )
    reference = _configured_monitor(
        "Ex",
        surface,
        [frequency],
        dt,
        iterations,
        name="single",
        real_dtype=np.dtype("f4"),
        complex_dtype=np.dtype("c8"),
    )
    _observe_ex_monitor(reference, surface, iterations, frequency)
    filename = tmp_path / "single.h5"
    with h5py.File(filename, "w") as output:
        reference.write_hdf5(output)

    target = _configured_monitor(
        "Ex",
        surface,
        [frequency],
        dt,
        iterations,
        name="double",
        incident_surface_file=filename,
        incident_monitor_name="single",
        real_dtype=np.dtype("f8"),
        complex_dtype=np.dtype("c16"),
    )
    indices = np.indices(surface.field_shape)
    zeros = np.zeros(surface.field_shape)
    for iteration in range(iterations):
        target.observe_electric(iteration, indices[0], zeros, zeros)
    with pytest.raises(ValueError, match="surface DFT is incompatible"):
        target.finalise()


def test_incident_subtraction_rejects_incompatible_surface(tmp_path):
    iterations = 12
    dt = 1e-11
    surface = build_component_surface(
        "Ex", (2, 2, 2), (5, 5, 5), (0.02, 0.02, 0.02), (9, 9, 9)
    )
    reference = _configured_monitor(
        "Ex", surface, [1e9], dt, iterations, name="reference"
    )
    _observe_ex_monitor(reference, surface, iterations, 1e9)
    filename = tmp_path / "reference.h5"
    with h5py.File(filename, "w") as output:
        reference.write_hdf5(output)

    target = _configured_monitor(
        "Ex",
        surface,
        [1.1e9],
        dt,
        iterations,
        name="target",
        incident_surface_file=filename,
        incident_monitor_name="reference",
    )
    indices = np.indices(surface.field_shape)
    zeros = np.zeros(surface.field_shape)
    for iteration in range(iterations):
        field = np.cos(2 * np.pi * 1.1e9 * iteration * dt) * (indices[0] + 1)
        target.observe_electric(iteration, field, zeros, zeros)
    with pytest.raises(ValueError, match="incompatible"):
        target.finalise()


def test_monitor_calculates_vector_spherical_outputs_and_full_sphere_metrics():
    iterations = 24
    dt = 2e-11
    frequency = 3 / (iterations * dt)
    shape = (9, 9, 9)
    surfaces = {
        component: build_component_surface(
            component, (2, 2, 2), (5, 5, 5), (0.02,) * 3, shape
        )
        for component in ("Ex", "Ey", "Ez")
    }
    monitor = KSIRFrequencyDomainMonitor(
        "vector",
        surfaces,
        [frequency],
        [0, 90, 180],
        [0, 180, 360],
        dt,
        iterations,
        real_dtype=REAL_DTYPE,
        complex_dtype=COMPLEX_DTYPE,
    )
    ids = np.ones((6,) + shape, dtype=np.uint32)
    monitor.validate_materials(ids, {"Ex": 0, "Ey": 1, "Ez": 2})
    monitor.configure_background([_material()])
    indices = np.indices(shape)
    component_fields = (
        indices[0] + 1.0,
        indices[1] + 2.0,
        indices[2] + 3.0,
    )
    for iteration in range(iterations):
        scale = np.cos(2 * np.pi * frequency * iteration * dt)
        monitor.observe_electric(
            iteration, *(scale * field for field in component_fields)
        )
    monitor.finalise()

    assert monitor.result.electric_cartesian.shape == (1, 9, 3)
    assert monitor.result.electric_spherical.shape == (1, 9, 3)
    assert monitor.result.radiation_intensity.shape == (1, 9)
    assert monitor.result.transversality_error.shape == (1, 9)
    assert monitor.result.radiated_power.shape == (1,)
    assert monitor.result.directivity.shape == (1, 9)
    assert monitor.result.maximum_directivity.shape == (1,)
    assert monitor.wave_speed == pytest.approx(c)


def test_production_monitor_rejects_heterogeneous_exterior_region():
    shape = (9, 9, 9)
    surface = build_component_surface(
        "Ex", (2, 2, 2), (5, 5, 5), (0.1, 0.1, 0.1), shape
    )
    monitor = KSIRFrequencyDomainMonitor(
        "heterogeneous",
        {"Ex": surface},
        [1e9],
        [90],
        [0],
        1e-11,
        2,
        real_dtype=REAL_DTYPE,
        complex_dtype=COMPLEX_DTYPE,
        exterior_index_bounds=((1, 7), (1, 7), (1, 7)),
    )
    ids = np.ones((6,) + shape, dtype=np.uint32)
    ids[0, 6, 3, 3] = 2

    with pytest.raises(ValueError, match="exterior region"):
        monitor.validate_materials(ids, {"Ex": 0})


def test_nyquist_frequency_itself_is_valid_but_higher_is_not():
    dt = 2e-11
    nyquist = 0.5 / dt

    assert validate_nyquist_frequencies((0, nyquist), dt) == nyquist
    with pytest.raises(ValueError, match="Nyquist limit"):
        validate_nyquist_frequencies((nyquist * (1 + 1e-12),), dt)


def test_experimental_mask_disables_integrated_metrics_and_reports_diagnostics(
    tmp_path,
):
    shape = (9, 9, 9)
    closure = resolve_closure(
        ExperimentalMask(("zmax",)),
        {},
        (2, 2, 2),
        (5, 5, 5),
        (8, 8, 8),
        (0.02, 0.02, 0.02),
    )
    surfaces = {
        component: build_component_surface(
            component,
            (2, 2, 2),
            (5, 5, 5),
            (0.02, 0.02, 0.02),
            shape,
            excluded_faces=closure.omitted_faces,
        )
        for component in ("Ex", "Ey", "Ez")
    }
    monitor = KSIRFrequencyDomainMonitor(
        "mask",
        surfaces,
        [1e9],
        [0, 90, 180],
        [0, 180, 360],
        1e-11,
        3,
        real_dtype=REAL_DTYPE,
        complex_dtype=COMPLEX_DTYPE,
        closure=closure,
    )
    ids = np.ones((6,) + shape, dtype=np.uint32)
    monitor.validate_materials(ids, {"Ex": 0, "Ey": 1, "Ez": 2})
    monitor.configure_background([_material()])
    indices = np.indices(shape)
    for iteration in range(3):
        scale = iteration + 1
        monitor.observe_electric(
            iteration,
            scale * (indices[0] + 1),
            scale * (indices[1] + 1),
            scale * (indices[2] + 1),
        )
    monitor.finalise()

    result = monitor.result
    assert result.closure == "experimental_mask"
    assert not result.mathematically_closed
    assert result.radiated_power is None
    assert result.directivity is None
    assert result.maximum_directivity is None
    assert result.missing_area_fraction["Ex"] > 0
    assert_allclose(
        result.face_contributions["Ex"][:, :, FACES.index("zmax")], 0
    )
    assert result.cancellation_indicator["Ex"].shape == (1, 9)

    filename = tmp_path / "masked.h5"
    with h5py.File(filename, "w") as output:
        monitor.write_hdf5(output)
    with h5py.File(filename, "r") as output:
        group = output["ntff/mask"]
        assert group.attrs["closure"] == "experimental_mask"
        assert not group.attrs["mathematically_closed"]
        assert "Prad" not in group
        assert "directivity" not in group
        assert "face_contributions/Ex" in group
        assert (
            group["closure_diagnostics/Ex"].attrs["missing_area_fraction"] > 0
        )
    reevaluated = evaluate_saved_surface_dft(
        filename, "mask", [0, 90, 180], [0, 180, 360]
    )
    assert reevaluated.closure == "experimental_mask"
    assert not reevaluated.mathematically_closed
    assert_allclose(
        reevaluated.range_normalized_fields["Ex"],
        result.range_normalized_fields["Ex"],
    )
