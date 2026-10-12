"""Independent checks of the analytical references used by the modal gallery."""
import numpy as np
import pytest

from testing.validation.eigenmode_tracking.profiles.waveguide_modal_profiles import analytical_fields, matching, vector


@pytest.mark.parametrize('height', [.02, .01])
@pytest.mark.parametrize('family,m,n', [('TE', 1, 0), ('TE', 6, 1), ('TM', 2, 3)])
def test_analytical_reference_satisfies_pec_walls(height, family, m, n):
    e, h = analytical_fields(family, m, n, 96, 100e9, 299792458., 376.730313, height)
    nv = round(96*height/.02)
    assert [a.shape for a in e] == [(96, nv+1), (97, nv), (97, nv+1)]
    assert [a.shape for a in h] == [(97, nv), (96, nv+1), (96, nv)]
    scale = max(np.max(abs(a)) for a in e)
    assert np.max(abs(e[0][:, [0, -1]])) < scale*1e-12
    assert np.max(abs(e[1][[0, -1], :])) < scale*1e-12
    assert np.max(abs(e[2][[0, -1], :])) < scale*1e-12
    assert np.max(abs(e[2][:, [0, -1]])) < scale*1e-12


def test_persistently_degenerate_rectangle_profiles_are_distinct():
    columns = [vector(*analytical_fields('TE', m, n, 96, 100e9,
                                        299792458., 376.730313, .01), 376.730313)
               for m, n in [(6, 1), (2, 3)]]
    fields = np.column_stack(columns)
    assert 6**2 + 4*1**2 == 2**2 + 4*3**2
    np.testing.assert_allclose(matching(fields, fields), np.eye(2), atol=1e-12)
    # Arbitrary phase and amplitude must not change the profile metric.
    np.testing.assert_allclose(matching(fields, fields * [2j, -3]), np.eye(2), atol=1e-12)
