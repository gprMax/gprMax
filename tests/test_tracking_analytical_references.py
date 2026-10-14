"""Independent Maxwell and interface checks for analytical validation fields."""

import numpy as np
import pytest
from scipy.constants import c

from testing.validation.eigenmode_tracking.degenerate_subspace.dielectric_cylinder import (
    fields, modes,
)
from testing.validation.eigenmode_tracking.branch_crossing.square_anisotropic_bore import beta


ROOT_MODES = [mode for mode, orientation in modes(40e9) if orientation == 0]


@pytest.mark.parametrize('mode', ROOT_MODES)
@pytest.mark.parametrize('point', [(0.0037, 0.0029), (0.0137, 0.0029)])
def test_bessel_reference_satisfies_vector_maxwell(mode, point):
    x, y = point
    step = 1e-7
    value = np.array(fields(mode, 0, 40e9, x, y))
    dx = (np.array(fields(mode, 0, 40e9, x+step, y))-
          np.array(fields(mode, 0, 40e9, x-step, y)))/(2*step)
    dy = (np.array(fields(mode, 0, 40e9, x, y+step))-
          np.array(fields(mode, 0, 40e9, x, y-step)))/(2*step)
    dz = -1j*mode.beta*value
    curl_e = np.array([dy[2]-dz[1], dz[0]-dx[2], dx[1]-dy[0]])
    curl_h = np.array([dy[5]-dz[4], dz[3]-dx[5], dx[4]-dy[3]])
    k0 = 2*np.pi*40e9/c
    epsilon = 4 if np.hypot(x, y) < .01 else 1
    assert np.linalg.norm(curl_e+1j*k0*value[3:])/np.linalg.norm(k0*value[3:]) < 1e-6
    assert np.linalg.norm(curl_h-1j*k0*epsilon*value[:3])/np.linalg.norm(k0*epsilon*value[:3]) < 1e-6


@pytest.mark.parametrize('mode', ROOT_MODES)
def test_bessel_reference_continuity_at_dielectric_interface(mode):
    angle, radius, step = .37, .01, 1e-10
    inside = fields(mode, 0, 40e9, (radius-step)*np.cos(angle), (radius-step)*np.sin(angle))
    outside = fields(mode, 0, 40e9, (radius+step)*np.cos(angle), (radius+step)*np.sin(angle))
    def tangential(v):
        return np.array([-np.sin(angle)*v[0]+np.cos(angle)*v[1], v[2],
                         -np.sin(angle)*v[3]+np.cos(angle)*v[4], v[5]])
    a, b = tangential(inside), tangential(outside)
    assert np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-30) < 1e-6


def test_continuum_crossing_keeps_two_pairs_and_exchanges_order():
    for epsilon in (2.3, 2.5, 2.7):
        assert beta('TE', 2, 0, epsilon) == beta('TE', 0, 2, epsilon)
        assert beta('TM', 1, 2, epsilon) == beta('TM', 2, 1, epsilon)
    assert beta('TE', 2, 0, 2.3) > beta('TM', 1, 2, 2.3)
    assert beta('TE', 2, 0, 2.7) < beta('TM', 1, 2, 2.7)
    assert beta('TE', 2, 0, 2.5) == pytest.approx(beta('TM', 1, 2, 2.5), rel=1e-14)
