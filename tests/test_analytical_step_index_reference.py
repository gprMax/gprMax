"""Independent Maxwell and interface checks for the dielectric audit oracle."""
from functools import lru_cache

import numpy as np
import pytest

from testing.eigenmode_support.dielectric.analytical_step_index import C, fields, modes, scalar


@lru_cache(maxsize=1)
def reference():
    return {r['name']: r for r in modes(40e9)}


@pytest.mark.parametrize('name', ['HE1,1', 'EH1,1', 'HE4,2', 'TE0,3', 'TM0,2'])
def test_reference_satisfies_all_dielectric_interface_conditions(name):
    m = reference()[name]; theta = .371; k0 = 2*np.pi*m['frequency']/C; beta = k0*m['neff']
    trace = []
    for sign, eps in ((-1, m['e1']), (1, m['e2'])):
        r = m['radius']*(1+sign*1e-10)
        x, y = np.array([r*np.cos(theta)]), np.array([r*np.sin(theta)])
        f, fx, fy = (m['amplitudes'][0]*a for a in scalar(m, x, y, 0.))
        g, gx, gy = (m['amplitudes'][1]*a for a in scalar(m, x, y, -np.pi/2 if m['m'] else 0.))
        q = k0*k0*eps-beta*beta
        et = 1j/q*(k0*(gx*np.cos(theta)+gy*np.sin(theta))-
                    beta*(-fx*np.sin(theta)+fy*np.cos(theta)))
        ht = -1j/q*(k0*eps*(fx*np.cos(theta)+fy*np.sin(theta))+
                     beta*(-gx*np.sin(theta)+gy*np.cos(theta)))
        er = -1j/q*(k0*(-gx*np.sin(theta)+gy*np.cos(theta))+
                     beta*(fx*np.cos(theta)+fy*np.sin(theta)))
        hr = 1j/q*(k0*eps*(-fx*np.sin(theta)+fy*np.cos(theta))-
                    beta*(gx*np.cos(theta)+gy*np.sin(theta)))
        trace.append(np.r_[f, g, et, ht, eps*er, hr])
    np.testing.assert_allclose(trace[0], trace[1], atol=1e-8*np.linalg.norm(trace[0]), rtol=1e-7)


def curl_error(m, cells):
    eu, ev, ew, gu, gv, gw = fields(m, cells)
    d = .04/cells; k0 = 2*np.pi*m['frequency']/C; beta = k0*m['neff']
    curl = ((ew[:, 1:]-ew[:, :-1])/d+1j*beta*ev,
            -1j*beta*eu-(ew[1:, :]-ew[:-1, :])/d,
            (ev[1:, :]-ev[:-1, :])/d-(eu[:, 1:]-eu[:, :-1])/d)
    errors = []; norms = []
    for value, exact, (nx, ny, ox, oy) in zip(curl, (gu, gv, gw),
            ((cells+1, cells, 0, .5), (cells, cells+1, .5, 0), (cells, cells, .5, .5))):
        x = (np.arange(nx)+ox-cells/2)[:, None]*d
        y = (np.arange(ny)+oy-cells/2)[None, :]*d
        radius = np.hypot(x, y)
        mask = ((radius < .008) | ((radius > .012) & (radius < .018)))
        errors.append(np.linalg.norm((value+1j*k0*exact)[mask])**2)
        norms.append(np.linalg.norm((k0*exact)[mask])**2)
    return np.sqrt(sum(errors)/sum(norms))


@pytest.mark.parametrize('name', ['HE1,1', 'EH1,1', 'HE4,2', 'TE0,3', 'TM0,2'])
def test_sampled_full_vector_reference_converges_to_maxwell_curl(name):
    coarse = curl_error(reference()[name], 64)
    fine = curl_error(reference()[name], 128)
    assert fine < .01
    assert fine < .35*coarse


def test_root_families_are_ordered_and_interface_certified():
    values = list(reference().values())
    assert values[0]['name'] == 'HE1,1'
    assert all(1 < r['neff'] < 2 and r['boundary_residual'] < 1e-8 for r in values)
    for family in ('HE', 'EH', 'TE', 'TM'):
        for m in range(8):
            rows = [r for r in values if r['family'] == family and r['m'] == m]
            assert [r['n'] for r in rows] == list(range(1, len(rows)+1))
