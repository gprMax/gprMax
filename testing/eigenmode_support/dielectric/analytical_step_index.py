"""Full-vector, infinite-cladding step-index cylinder reference.

Time convention exp(i*omega*t-i*beta*z), nonmagnetic media.
Reference equations: Virginia Tech dissertation chapter 5, equations 5.1--5.6,
https://vtechworks.lib.vt.edu/bitstream/handle/10919/30522/jb-chap5.pdf
Only the complete-cylinder equations are used, not wedge boundary conditions.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, jvp, kv, kvp

C = 299792458.


def boundary_terms(u, m, v, e1=4., e2=1.):
    w = np.sqrt(v*v-u*u)
    neff = np.sqrt(e1-(e1-e2)*(u/v)**2)
    j = jv(m, u)
    a = jvp(m, u)/u
    b = kvp(m, w)/kv(m, w)/w*j
    c = m*neff*(1/u**2+1/w**2)*j
    return a+b, e1*a+e2*b, c, neff


def modes(frequency, radius=.01, e1=4., e2=1., max_m=16):
    """Find all guided roots, avoiding rational Bessel-pole false roots."""
    v = 2*np.pi*frequency/C*radius*np.sqrt(e1-e2)
    sample = np.linspace(1e-4, v*(1-1e-10), 20001)
    result = []
    for m in range(max_m+1):
        families = ('TE', 'TM') if m == 0 else ('hybrid',)
        for family in families:
            def equation(u):
                a, b, c, _ = boundary_terms(u, m, v, e1, e2)
                return a if family == 'TE' else b if family == 'TM' else a*b-c*c
            values = equation(sample)
            roots = []
            for i in np.flatnonzero(values[:-1]*values[1:] < 0):
                u = brentq(equation, sample[i], sample[i+1], xtol=5e-14)
                if roots and abs(roots[-1]-u) < 1e-7:
                    continue
                roots.append(u)
                a, b, c, neff = boundary_terms(u, m, v, e1, e2)
                if m == 0:
                    amplitudes = [0., 1.] if family == 'TE' else [1., 0.]
                    label = family
                    relative_residual = abs(a if family == 'TE' else b)/max(abs(a), abs(b), 1e-30)
                else:
                    matrix = np.array([[c, a], [b, c]])
                    _, singular, vh = np.linalg.svd(matrix)
                    amplitudes = vh[-1].tolist()
                    relative_residual = singular[-1]/singular[0]
                    # s=m(1/u^2+1/w^2)/(eta1+eta2); HE has s<0.
                    label = 'HE' if jv(m, u)/a < 0 else 'EH'
                if relative_residual > 1e-8:
                    raise ValueError(f'Uncertified analytical root m={m}, u={u}, residual={relative_residual}')
                result.append(dict(family=label, m=m, u=float(u), w=float(np.sqrt(v*v-u*u)),
                    neff=float(neff), amplitudes=amplitudes, boundary_residual=float(relative_residual),
                    frequency=frequency, radius=radius, e1=e1, e2=e2))
    result.sort(key=lambda r: -r['neff'])
    counts = {}
    for row in result:
        key = (row['family'], row['m']); counts[key] = counts.get(key, 0)+1
        row['n'] = counts[key]; row['name'] = f"{row['family']}{row['m']},{row['n']}"
    return result


def scalar(mode, x, y, phase):
    """Longitudinal scalar and its exact Cartesian derivatives."""
    m = mode['m']; radius = mode['radius']; u = mode['u']; w = mode['w']
    r = np.hypot(x, y); theta = np.arctan2(y, x); inside = r < radius
    argument = np.where(inside, u*r/radius, w*np.maximum(r, radius)/radius)
    external_scale = jv(m, u)/kv(m, w)
    radial = np.where(inside, jv(m, argument), external_scale*kv(m, argument))
    dr = np.where(inside, u/radius*jvp(m, argument), external_scale*w/radius*kvp(m, argument))
    angle = m*theta+phase; angular = np.cos(angle)
    dt = np.divide(-m*radial*np.sin(angle), r, out=np.zeros_like(r), where=r > 0)
    dx = np.cos(theta)*dr*angular-np.sin(theta)*dt
    dy = np.sin(theta)*dr*angular+np.cos(theta)*dt
    dx = np.where(r == 0, u/(2*radius)*np.cos(phase) if m == 1 else 0., dx)
    dy = np.where(r == 0, -u/(2*radius)*np.sin(phase) if m == 1 else 0., dy)
    return radial*angular, dx, dy


def fields(mode, cells, width=.04, partner=0):
    """Return [E, Z0*H] sampled at the six native Yee component locations."""
    k0 = 2*np.pi*mode['frequency']/C; beta = k0*mode['neff']
    ae, ah = mode['amplitudes']; phase = partner*np.pi/2
    output = []
    for component, (nx, ny, ox, oy) in enumerate(((cells, cells+1, .5, 0),
            (cells+1, cells, 0, .5), (cells+1, cells+1, 0, 0),
            (cells+1, cells, 0, .5), (cells, cells+1, .5, 0), (cells, cells, .5, .5))):
        x = (np.arange(nx)+ox-cells/2)[:, None]*width/cells
        y = (np.arange(ny)+oy-cells/2)[None, :]*width/cells
        eps = np.where(x*x+y*y < mode['radius']**2, mode['e1'], mode['e2'])
        q = k0*k0*eps-beta*beta
        f, fx, fy = (ae*a for a in scalar(mode, x, y, phase))
        # Ez ~ cos(m theta), Z0 Hz ~ sin(m theta), except m=0 TE.
        gphase = phase-np.pi/2 if mode['m'] else 0.
        g, gx, gy = (ah*a for a in scalar(mode, x, y, gphase))
        value = ((-1j/q)*(k0*gy+beta*fx), (1j/q)*(k0*gx-beta*fy), f,
                 (1j/q)*(k0*eps*fy-beta*gx), (-1j/q)*(k0*eps*fx+beta*gy), g)[component]
        output.append(value)
    return output


def frame(mode, cells, width=.04, partner=0):
    return np.concatenate([a.ravel() for a in fields(mode, cells, width, partner)])
