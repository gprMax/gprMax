import numpy as np
import pytest

from gprMax.cython.virtual_waveguide import (
    couple_virtual_waveguide_electric,
    couple_virtual_waveguide_electric_aperture,
    couple_virtual_waveguide_magnetic,
)


@pytest.mark.parametrize("normal_axis", (0, 1, 2))
@pytest.mark.parametrize("direction_sign", (-1, 1))
def test_virtual_waveguide_coupling_supports_every_orientation(normal_axis, direction_sign):
    shape = (9, 9, 9)
    fields = [np.ones(shape, dtype=np.float64) for _ in range(12)]
    main_e = fields[:3]
    main_h = fields[3:6]
    nu, nv = 4, 5
    aux_shape = [1, 1, 1]
    aux_shape[normal_axis] = 9
    transverse_axes = [axis for axis in range(3) if axis != normal_axis]
    aux_shape[transverse_axes[0]] = nu + 1
    aux_shape[transverse_axes[1]] = nv + 1
    aux_shape = tuple(aux_shape)
    aux_e = [np.ones(aux_shape, dtype=np.float64) for _ in range(3)]
    aux_h = [np.full(aux_shape, 2.0, dtype=np.float64) for _ in range(3)]
    u0, v0, u1, v1, plane = 1, 1, 1 + nu, 1 + nv, 4

    couple_virtual_waveguide_magnetic(
        1,
        normal_axis,
        direction_sign,
        u0,
        v0,
        u1,
        v1,
        plane,
        *main_h,
        *aux_h,
    )

    aperture = 0 if direction_sign < 0 else aux_shape[normal_axis] - 1
    aperture_index = [0, 0, 0]
    aperture_index[normal_axis] = aperture
    assert aux_h[normal_axis][tuple(aperture_index)] == pytest.approx(1.0)

    updatecoeffs = np.asarray([[1.0, 0.1, 0.1, 0.1, 0.0]], dtype=np.float64)
    component_ids = np.zeros((6, *aux_shape), dtype=np.uint32)
    couple_virtual_waveguide_electric(
        1,
        normal_axis,
        direction_sign,
        u0,
        v0,
        u1,
        v1,
        plane,
        updatecoeffs,
        component_ids,
        *main_e,
        *main_h,
        *aux_e,
        *aux_h,
    )

    detached_index = [0, 0, 0]
    detached_index[normal_axis] = plane + 2 if direction_sign < 0 else plane - 2
    detached_index[transverse_axes[0]] = u0 + 1
    detached_index[transverse_axes[1]] = v0 + 1
    assert all(field[tuple(detached_index)] == 0 for field in main_e)
    assert all(np.all(np.isfinite(field)) for field in (*main_e, *main_h, *aux_e, *aux_h))


@pytest.mark.parametrize("normal_axis", (0, 1, 2))
@pytest.mark.parametrize("direction_sign", (-1, 1))
def test_compact_mpi_aperture_update_matches_full_grid_coupling(normal_axis, direction_sign):
    rng = np.random.default_rng(20260814)
    shape = (10, 11, 12)
    main_e = [rng.standard_normal(shape) for _ in range(3)]
    main_h = [rng.standard_normal(shape) for _ in range(3)]
    nu, nv = 4, 5
    transverse_axes = [axis for axis in range(3) if axis != normal_axis]
    aux_shape = [1, 1, 1]
    aux_shape[normal_axis] = 9
    aux_shape[transverse_axes[0]] = nu + 1
    aux_shape[transverse_axes[1]] = nv + 1
    aux_shape = tuple(aux_shape)
    aux_e_full = [rng.standard_normal(aux_shape) for _ in range(3)]
    aux_e_compact = [field.copy() for field in aux_e_full]
    aux_h = [rng.standard_normal(aux_shape) for _ in range(3)]
    u0, v0, u1, v1, plane = 2, 3, 2 + nu, 3 + nv, 5

    aperture = 0 if direction_sign < 0 else aux_shape[normal_axis] - 1
    main_sheet = [slice(None)] * 3
    main_sheet[normal_axis] = plane
    main_sheet[transverse_axes[0]] = slice(u0, u1)
    main_sheet[transverse_axes[1]] = slice(v0, v1)
    aux_sheet = [slice(None)] * 3
    aux_sheet[normal_axis] = aperture
    aux_sheet[transverse_axes[0]] = slice(0, nu)
    aux_sheet[transverse_axes[1]] = slice(0, nv)
    aux_h[normal_axis][tuple(aux_sheet)] = main_h[normal_axis][tuple(main_sheet)]

    cross_plane = plane - 1 if direction_sign < 0 else plane

    def magnetic_sheet(component, u_points, v_points):
        sheet = [slice(None)] * 3
        sheet[normal_axis] = cross_plane
        sheet[transverse_axes[0]] = slice(u0, u0 + u_points)
        sheet[transverse_axes[1]] = slice(v0, v0 + v_points)
        return np.ascontiguousarray(main_h[component][tuple(sheet)])

    main_hu = magnetic_sheet(transverse_axes[0], nu + 1, nv)
    main_hv = magnetic_sheet(transverse_axes[1], nu, nv + 1)
    updatecoeffs = np.asarray([[0.97, 0.11, 0.12, 0.13, 0.0]], dtype=np.float64)
    component_ids = np.zeros((6, *aux_shape), dtype=np.uint32)

    couple_virtual_waveguide_electric(
        1,
        normal_axis,
        direction_sign,
        u0,
        v0,
        u1,
        v1,
        plane,
        updatecoeffs,
        component_ids,
        *(field.copy() for field in main_e),
        *main_h,
        *aux_e_full,
        *aux_h,
    )
    couple_virtual_waveguide_electric_aperture(
        1,
        normal_axis,
        direction_sign,
        updatecoeffs,
        component_ids,
        main_hu,
        main_hv,
        *aux_e_compact,
        *aux_h,
    )

    for full, compact in zip(aux_e_full, aux_e_compact):
        np.testing.assert_allclose(compact, full, rtol=0, atol=0)
