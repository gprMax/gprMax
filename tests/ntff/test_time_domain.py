import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from gprMax.ntff.closures import SymmetryCompletion, resolve_closure
from gprMax.ntff.surfaces import COMPONENT_OFFSETS, build_component_surface
from gprMax.ntff.time_domain import KSIRTimeDomainMonitor


def _component_field(component, shape, spacing, gradient, intercept):
    indices = np.indices(shape).reshape(3, -1).T
    positions = (indices + np.asarray(COMPONENT_OFFSETS[component])) * spacing
    return (positions @ gradient + intercept).reshape(shape)


def _reference_polynomial_deposition(
    surface,
    points,
    dt,
    iterations,
    wave_speed,
    sample_offset_steps,
    gradient,
    intercept,
):
    positions = surface.patch_positions
    normals = surface.normals
    areas = surface.area_weights
    spatial_field = positions @ gradient + intercept
    spatial_derivative = normals @ gradient
    displacement = points[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    direction = displacement / distance[:, :, np.newaxis]
    normal_projection = np.sum(normals[np.newaxis, :, :] * direction, axis=2)
    weight_a = -areas[np.newaxis, :] / (4 * np.pi * distance)
    weight_b = areas[np.newaxis, :] * normal_projection / (4 * np.pi * distance**2)
    weight_c = (
        areas[np.newaxis, :]
        * normal_projection
        / (4 * np.pi * wave_speed * distance)
    )
    delay = sample_offset_steps + distance / (wave_speed * dt)
    integer_delay = np.floor(delay).astype(np.int64)
    fractional_delay = delay - integer_delay
    output_length = iterations + int(np.max(integer_delay)) + 1
    expected = np.zeros((points.shape[0], output_length))

    for iteration in range(iterations):
        time = (iteration + sample_offset_steps) * dt
        time_factor = 1 + 0.7 * time + 0.3 * time**2
        derivative_factor = 0.7 + 0.6 * time
        field = time_factor * spatial_field
        derivative = time_factor * spatial_derivative
        time_derivative = derivative_factor * spatial_field
        integrand = (
            weight_a * derivative[np.newaxis, :]
            + weight_b * field[np.newaxis, :]
            + weight_c * time_derivative[np.newaxis, :]
        )
        destination = iteration + integer_delay
        for point_index in range(points.shape[0]):
            np.add.at(
                expected[point_index],
                destination[point_index],
                (1 - fractional_delay[point_index]) * integrand[point_index],
            )
            np.add.at(
                expected[point_index],
                destination[point_index] + 1,
                fractional_delay[point_index] * integrand[point_index],
            )
    return expected


@pytest.mark.parametrize("component,sample_offset_steps", [("Ex", 0.0), ("Hx", 0.5)])
def test_advanced_time_monitor_matches_independent_polynomial_deposition(
    component, sample_offset_steps
):
    spacing = np.array((0.1, 0.12, 0.08))
    shape = (10, 11, 12)
    surface = build_component_surface(component, (2, 2, 2), (6, 7, 8), spacing, shape)
    points = np.array(((0.9, 0.5, 0.4), (0.4, 1.1, 0.5)))
    dt = 0.05
    iterations = 7
    wave_speed = 2.3
    gradient = np.array((0.8, -0.35, 0.6))
    intercept = 0.2
    spatial_grid = _component_field(component, shape, spacing, gradient, intercept)
    monitor = KSIRTimeDomainMonitor(
        "polynomial",
        {component: surface},
        points,
        dt,
        iterations,
        real_dtype=np.dtype(float),
        wave_speed=wave_speed,
    )

    for iteration in range(iterations):
        time = (iteration + sample_offset_steps) * dt
        time_factor = 1 + 0.7 * time + 0.3 * time**2
        field = time_factor * spatial_grid
        zeros = np.zeros_like(field)
        if component == "Ex":
            monitor.observe_electric(iteration, field, zeros, zeros)
        else:
            monitor.observe_magnetic(iteration, field, zeros, zeros)
    monitor.finalise()

    expected = _reference_polynomial_deposition(
        surface,
        points,
        dt,
        iterations,
        wave_speed,
        sample_offset_steps,
        gradient,
        intercept,
    )
    result = monitor.result

    assert_allclose(result.fields[component], expected, rtol=2e-13, atol=2e-13)
    assert_allclose(result.times, dt * np.arange(monitor.output_length))
    assert result.sample_time_offsets[component] == sample_offset_steps * dt
    assert not result.fields[component].flags.writeable


def test_advanced_time_monitor_reconstructs_direct_outgoing_scalar_pulse():
    spacing = 0.04
    shape = (25, 25, 25)
    surface = build_component_surface(
        "Ex", (5, 5, 5), (20, 20, 20), (spacing,) * 3, shape
    )
    source_position = np.array((0.5, 0.5, 0.5))
    point = np.array((1.05, 0.5, 0.5))
    dt = 0.01
    iterations = 160
    wave_speed = 1.0
    monitor = KSIRTimeDomainMonitor(
        "pulse",
        {"Ex": surface},
        point,
        dt,
        iterations,
        real_dtype=np.dtype(float),
        wave_speed=wave_speed,
    )

    indices = np.indices(shape).reshape(3, -1).T
    positions = (indices + np.asarray(COMPONENT_OFFSETS["Ex"])) * spacing
    radius = np.linalg.norm(positions - source_position, axis=1)
    pulse_centre = 0.28
    pulse_width = 0.07

    for iteration in range(iterations):
        retarded_time = iteration * dt - radius / wave_speed
        pulse = np.exp(-((retarded_time - pulse_centre) / pulse_width) ** 2)
        field = np.divide(
            pulse,
            4 * np.pi * radius,
            out=np.zeros_like(radius),
            where=radius != 0,
        ).reshape(shape)
        zeros = np.zeros_like(field)
        monitor.observe_electric(iteration, field, zeros, zeros)
    monitor.finalise()

    result = monitor.result.fields["Ex"][0]
    direct_radius = np.linalg.norm(point - source_position)
    direct = np.exp(
        -(
            (monitor.result.times - direct_radius / wave_speed - pulse_centre)
            / pulse_width
        )
        ** 2
    ) / (4 * np.pi * direct_radius)
    significant = direct > 0.02 * np.max(direct)
    relative_error = np.linalg.norm(
        result[significant] - direct[significant]
    ) / np.linalg.norm(direct[significant])

    # Fifteen cells per side is intentionally a modest reference mesh; the
    # combined midpoint-surface and fractional-delay error should remain well
    # below ten percent without tuning the test to a very large surface.
    assert relative_error < 0.06


def test_monitor_rejects_points_on_or_inside_any_component_surface():
    surface = build_component_surface(
        "Ez", (2, 2, 2), (6, 6, 6), (0.1, 0.1, 0.1), (10, 10, 10)
    )

    with pytest.raises(ValueError, match="strictly outside"):
        KSIRTimeDomainMonitor(
            "inside",
            {"Ez": surface},
            surface.centre,
            0.01,
            5,
            real_dtype=np.dtype(float),
            wave_speed=1.0,
        )


def test_monitor_uses_patch_support_not_only_patch_centres_for_point_validation():
    surface = build_component_surface(
        "Ez", (2, 2, 2), (6, 6, 6), (0.1, 0.1, 0.1), (10, 10, 10)
    )
    # Ez patches on a y-normal face are centred from y=0.2, but their
    # quadrature support starts at y=0.15.  This point is therefore inside the
    # closed component surface even though it lies below every patch centre.
    with pytest.raises(ValueError, match="strictly outside"):
        KSIRTimeDomainMonitor(
            "inside_patch_support",
            {"Ez": surface},
            (0.4, 0.175, 0.4),
            0.01,
            5,
            real_dtype=np.dtype(float),
            wave_speed=1.0,
        )


def test_monitor_requires_every_expected_sample_before_finalising():
    surface = build_component_surface(
        "Ex", (2, 2, 2), (5, 5, 5), (0.1, 0.1, 0.1), (9, 9, 9)
    )
    monitor = KSIRTimeDomainMonitor(
        "incomplete",
        {"Ex": surface},
        (0.8, 0.4, 0.4),
        0.01,
        2,
        real_dtype=np.dtype(float),
        wave_speed=1.0,
    )
    field = np.zeros((9, 9, 9))
    monitor.observe_electric(0, field, field, field)

    with pytest.raises(RuntimeError, match="1 of 2 expected samples"):
        monitor.finalise()


def test_monitor_requires_one_material_id_across_all_straddling_samples():
    shape = (9, 9, 9)
    surface = build_component_surface(
        "Ex", (2, 2, 2), (5, 5, 5), (0.1, 0.1, 0.1), shape
    )
    monitor = KSIRTimeDomainMonitor(
        "materials",
        {"Ex": surface},
        (0.8, 0.4, 0.4),
        0.01,
        1,
        real_dtype=np.dtype(float),
        wave_speed=1.0,
    )
    material_ids = np.full((6,) + shape, 2, dtype=np.uint32)

    monitor.validate_materials(material_ids, {"Ex": 0})
    assert monitor.surface_material_id == 2

    first_outside = tuple(surface.faces[0].outside_indices[0])
    material_ids[(0,) + first_outside] = 3
    with pytest.raises(ValueError, match="multiple material IDs"):
        monitor.validate_materials(material_ids, {"Ex": 0})


def test_first_arrival_time_origin_removes_range_dependent_zero_prefix():
    shape = (9, 9, 9)
    surface = build_component_surface(
        "Ex", (2, 2, 2), (6, 6, 6), (0.1, 0.1, 0.1), shape
    )
    points = ((20.0, 0.4, 0.4), (35.0, 0.5, 0.4))
    kwargs = dict(
        surfaces={"Ex": surface},
        points=points,
        dt=0.01,
        iterations=6,
        real_dtype=np.dtype("f8"),
        wave_speed=1.0,
        nthreads=2,
    )
    absolute = KSIRTimeDomainMonitor("absolute", **kwargs)
    shifted = KSIRTimeDomainMonitor(
        "shifted", **kwargs, time_origin="first_arrival"
    )
    indices = np.indices(shape)
    zeros = np.zeros(shape)
    for iteration in range(6):
        field = (iteration + 1) * (indices[0] + 0.2 * indices[1])
        absolute.observe_electric(iteration, field, zeros, zeros)
        shifted.observe_electric(iteration, field, zeros, zeros)
    absolute.finalise()
    shifted.finalise()

    assert shifted.output_length < absolute.output_length / 50
    assert shifted.output_length < 2 * absolute.iterations + 50
    assert shifted.result.time_origin == "first_arrival"
    assert np.all(shifted.result.time_origins > 0)
    assert shifted.result.times[0] == 0
    for point_index, origin in enumerate(shifted.time_origin_steps):
        length = int(shifted.result.valid_lengths[point_index])
        assert_allclose(
            shifted.result.point_field("Ex", point_index),
            absolute.result.fields["Ex"][
                point_index, origin : origin + length
            ],
        )
        assert_allclose(
            shifted.result.point_times(point_index),
            absolute.result.times[origin : origin + length],
        )


@pytest.mark.parametrize("time_origin", [None, "arrival", 3])
def test_monitor_rejects_unknown_time_origin(time_origin):
    surface = build_component_surface(
        "Ex", (2, 2, 2), (5, 5, 5), (0.1, 0.1, 0.1), (9, 9, 9)
    )
    with pytest.raises(ValueError, match="time_origin"):
        KSIRTimeDomainMonitor(
            "invalid",
            {"Ex": surface},
            (0.8, 0.4, 0.4),
            0.01,
            5,
            real_dtype=np.dtype(float),
            time_origin=time_origin,
        )


def test_monitor_rejects_surface_faces_inconsistent_with_closure():
    closure = resolve_closure(
        SymmetryCompletion(),
        {"x0": "pec"},
        (0, 2, 2),
        (5, 5, 5),
        (9, 9, 9),
        (0.1, 0.1, 0.1),
    )
    surface = build_component_surface(
        "Ex",
        (0, 2, 2),
        (5, 5, 5),
        (0.1, 0.1, 0.1),
        (10, 10, 10),
        excluded_faces=("x0", "zmax"),
    )

    with pytest.raises(ValueError, match="do not match closure faces"):
        KSIRTimeDomainMonitor(
            "inconsistent",
            {"Ex": surface},
            (0.8, 0.4, 0.4),
            0.01,
            5,
            real_dtype=np.dtype(float),
            wave_speed=1.0,
            closure=closure,
        )

def test_compact_time_histories_and_metadata_are_written_to_hdf5(tmp_path):
    shape = (9, 9, 9)
    surface = build_component_surface(
        "Ex", (2, 2, 2), (6, 6, 6), (0.1, 0.1, 0.1), shape
    )
    monitor = KSIRTimeDomainMonitor(
        "compact",
        {"Ex": surface},
        ((20.0, 0.4, 0.4), (25.0, 0.5, 0.4)),
        0.01,
        3,
        real_dtype=np.dtype("f8"),
        wave_speed=1.0,
        time_origin="first_arrival",
        nthreads=2,
    )
    material_ids = np.ones((6,) + shape, dtype=np.uint32)
    monitor.validate_materials(material_ids, {"Ex": 0})
    indices = np.indices(shape)
    zeros = np.zeros(shape)
    for iteration in range(3):
        field = (iteration + 1) * (indices[0] + 0.2 * indices[1])
        monitor.observe_electric(iteration, field, zeros, zeros)
    monitor.finalise()

    filename = tmp_path / "compact.h5"
    with h5py.File(filename, "w") as output:
        monitor.write_hdf5(output)
    with h5py.File(filename, "r") as output:
        group = output["ntff/compact"]
        assert group.attrs["formulation"] == "KSIR"
        assert group.attrs["output_type"] == "time_domain_field_extension"
        assert group.attrs["solver"] == "cpu"
        assert group.attrs["time_origin"] == "first_arrival"
        assert group.attrs["background_material_id"] == 1
        assert group.attrs["openmp_threads"] == 2
        assert_allclose(group["points"][:], monitor.result.points)
        assert_allclose(group["times"][:], monitor.result.times)
        assert_allclose(group["time_origins"][:], monitor.result.time_origins)
        assert_allclose(group["valid_lengths"][:], monitor.result.valid_lengths)
        assert_allclose(group["fields/Ex"][:], monitor.result.fields["Ex"])


def test_symmetry_images_define_completed_surface_and_effective_patches():
    lower = (0, 0, 2)
    upper = (4, 4, 6)
    spacing = (0.1, 0.1, 0.1)
    closure = resolve_closure(
        SymmetryCompletion(),
        {"x0": "pmc", "y0": "pec"},
        lower,
        upper,
        (10, 10, 10),
        spacing,
    )
    surface = closure.apply_quadrature(
        build_component_surface(
            "Ez",
            lower,
            upper,
            spacing,
            (11, 11, 11),
            excluded_faces=closure.omitted_faces,
        )
    )
    monitor = KSIRTimeDomainMonitor(
        "quarter",
        {"Ez": surface},
        (0.8, 0.8, 0.4),
        0.01,
        5,
        real_dtype=np.dtype(float),
        wave_speed=1.0,
        closure=closure,
    )
    accumulator = monitor._accumulators["Ez"]

    assert closure.image_count == 4
    assert accumulator._source_patch_index.size == 4 * surface.npatches
    assert_allclose(accumulator.completed_physical_lower, (-0.45, -0.45, 0.2))
    assert_allclose(accumulator.completed_physical_upper, (0.45, 0.45, 0.6))

    with pytest.raises(ValueError, match="completed Ez surface"):
        KSIRTimeDomainMonitor(
            "inside_virtual_box",
            {"Ez": surface},
            (-0.2, 0.2, 0.4),
            0.01,
            5,
            real_dtype=np.dtype(float),
            wave_speed=1.0,
            closure=closure,
        )
