"""Full-vector step-index cylinder: independent J/K boundary-matching theory.

No weak-guidance/LP approximation is used. The time convention is exp(+jwt),
with propagation exp(-j beta z). H components returned below are scaled by Z0.
"""

from dataclasses import dataclass

import numpy as np
from scipy.constants import c
from scipy.optimize import brentq
from scipy.special import jv, jvp, kv, kvp


@dataclass(frozen=True)
class Mode:
    m: int
    radial: int
    family: str
    u: float
    beta: float


def characteristic(u, m, v, epsilon_core=4., epsilon_cladding=1., family='hybrid'):
    """Pole-free characteristic equation after multiplying by J_m(U)^2."""
    u = np.asarray(u)
    w = np.sqrt(v*v-u*u)
    j, jp = jv(m, u), jvp(m, u)
    logk = kvp(m, w)/kv(m, w)
    a = jp/u+j*logk/w
    b = epsilon_core*jp/u+epsilon_cladding*j*logk/w
    if family == 'TE':
        return a
    if family == 'TM':
        return b
    neff2 = epsilon_core-u*u*(epsilon_core-epsilon_cladding)/(v*v)
    return a*b-m*m*neff2*(1/u**2+1/w**2)**2*j*j


def modes(frequency, radius=.01, count=16, epsilon_core=4., epsilon_cladding=1.):
    k0 = 2*np.pi*frequency/c
    v = radius*k0*np.sqrt(epsilon_core-epsilon_cladding)
    candidates = []
    axis = np.linspace(.0001, v-.0001, 12001)
    for m in range(int(v)+2):
        for family in (('TE', 'TM') if m == 0 else ('hybrid',)):
            values = characteristic(axis, m, v, epsilon_core, epsilon_cladding, family)
            roots = []
            for index in np.flatnonzero(values[:-1]*values[1:] < 0):
                root = brentq(characteristic, axis[index], axis[index+1],
                              args=(m, v, epsilon_core, epsilon_cladding, family), xtol=1e-12)
                if not roots or abs(root-roots[-1]) > 1e-6:
                    roots.append(root)
            for radial, u in enumerate(roots, 1):
                beta = np.sqrt(epsilon_core*k0*k0-(u/radius)**2)
                mode = Mode(m, radial, family, float(u), float(beta))
                for orientation in range(2 if m else 1):
                    candidates.append((mode, orientation))
    candidates.sort(key=lambda item: -item[0].beta)
    # Retain complete analytical pairs at the requested edge.
    if len(candidates) > count and candidates[count-1][0] == candidates[count][0]:
        count += 1
    return candidates[:count]


def boundary_matrix(mode, frequency, radius=.01, epsilon_core=4., epsilon_cladding=1.):
    """Four tangential interface conditions on [Ez_in, Z0Hz_in, Ez_out, Z0Hz_out]."""
    k0, beta, m = 2*np.pi*frequency/c, mode.beta, mode.m
    q2 = epsilon_core*k0*k0-beta*beta
    w2 = beta*beta-epsilon_cladding*k0*k0
    din = np.sqrt(q2)*jvp(m, radius*np.sqrt(q2))/jv(m, radius*np.sqrt(q2))
    dout = np.sqrt(w2)*kvp(m, radius*np.sqrt(w2))/kv(m, radius*np.sqrt(w2))
    ti, to = m*beta/(radius*q2), -m*beta/(radius*w2)
    hi, ho = 1j*k0*din/q2, -1j*k0*dout/w2
    return np.array([[1, 0, -1, 0], [0, 1, 0, -1],
                     [ti, hi, -to, -ho],
                     [-epsilon_core*hi, ti, epsilon_cladding*ho, -to]], complex)


def amplitudes(mode, frequency, **kwargs):
    matrix = boundary_matrix(mode, frequency, **kwargs)
    _, _, vh = np.linalg.svd(matrix)
    return vh[-1].conj()


def fields(mode, orientation, frequency, x, y, radius=.01,
           epsilon_core=4., epsilon_cladding=1.):
    """Evaluate six Cartesian components of a standing angular orientation."""
    x, y = np.broadcast_arrays(x, y)
    # Evaluate the origin by its limiting value; avoid polar 0/0.
    x = np.where((x == 0) & (y == 0), radius*1e-9, x)
    r, theta = np.hypot(x, y), np.arctan2(y, x)
    m, beta, k0 = mode.m, mode.beta, 2*np.pi*frequency/c
    inside = r <= radius
    qi = np.sqrt(epsilon_core*k0*k0-beta*beta)
    qo = np.sqrt(beta*beta-epsilon_cladding*k0*k0)
    amplitude = amplitudes(mode, frequency, radius=radius,
                           epsilon_core=epsilon_core, epsilon_cladding=epsilon_cladding)
    # Use bounded arguments on the unused side of np.where.
    ri, ro = np.minimum(r, radius), np.maximum(r, radius)
    radial = np.where(inside, jv(m, qi*ri)/jv(m, qi*radius),
                      kv(m, qo*ro)/kv(m, qo*radius))
    derivative = np.where(inside, qi*jvp(m, qi*ri)/jv(m, qi*radius),
                          qo*kvp(m, qo*ro)/kv(m, qo*radius))
    q2 = np.where(inside, qi*qi, -qo*qo)
    epsilon = np.where(inside, epsilon_core, epsilon_cladding)
    ae, ah = np.where(inside, amplitude[0], amplitude[2]), np.where(inside, amplitude[1], amplitude[3])
    # Combine +m and -m travelling angular waves to get standing orientations.
    total = [np.zeros_like(r, complex) for _ in range(6)]
    for sign in ((1, -1) if m else (1,)):
        angular = np.exp(1j*sign*m*theta)
        weight = (1 if sign == 1 else (1 if orientation == 0 else -1)) / (2 if m else 1)
        if m and orientation == 1:
            weight /= 1j
        ez, hz = ae*radial*angular, sign*ah*radial*angular if m else ah*radial
        dez, dhz = ae*derivative*angular, sign*ah*derivative*angular if m else ah*derivative
        er = -1j*(beta*dez+k0*(1j*sign*m)*hz/r)/q2
        et = -1j*beta*(1j*sign*m)*ez/(q2*r)+1j*k0*dhz/q2
        hr = -1j*beta*dhz/q2+1j*k0*epsilon*(1j*sign*m)*ez/(q2*r)
        ht = -1j*(beta*(1j*sign*m)*hz/r+k0*epsilon*dez)/q2
        values = (er*np.cos(theta)-et*np.sin(theta), er*np.sin(theta)+et*np.cos(theta), ez,
                  hr*np.cos(theta)-ht*np.sin(theta), hr*np.sin(theta)+ht*np.cos(theta), hz)
        for result, value in zip(total, values):
            result += weight*value
    return total


def sampled_fields(mode, orientation, frequency, cells, window=.06):
    result = []
    for component, (nu, nv, ox, oy) in enumerate(((cells, cells+1, .5, 0),
            (cells+1, cells, 0, .5), (cells+1, cells+1, 0, 0),
            (cells+1, cells, 0, .5), (cells, cells+1, .5, 0), (cells, cells, .5, .5))):
        x = (np.arange(nu)+ox-cells/2)[:, None]*window/cells
        y = (np.arange(nv)+oy-cells/2)[None, :]*window/cells
        result.append(fields(mode, orientation, frequency, x, y)[component])
    return result
