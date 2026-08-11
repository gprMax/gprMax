import numpy as np
import pytest

from gprMax.cython.virtual_waveguide import (
    couple_virtual_waveguide_electric,
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
