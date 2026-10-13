"""Analytical field-pattern diagnostics for validation; never selects a basis."""
import numpy as np
from scipy.fft import dctn, dstn


def longitudinal_coefficients(electric, magnetic, normal_axis, impedance):
    """Project native rectangular Yee samples onto TE/TM longitudinal patterns."""
    hz = np.stack([h[normal_axis] for h in magnetic], axis=-1) * impedance
    ez = np.stack([e[normal_axis][1:-1, 1:-1] for e in electric], axis=-1)
    te = dctn(hz, type=2, axes=(0, 1), norm='ortho')
    tm = dstn(ez, type=1, axes=(0, 1), norm='ortho')
    names = [('TE', m, n) for m in range(te.shape[0]) for n in range(te.shape[1])]
    names += [('TM', m+1, n+1) for m in range(tm.shape[0]) for n in range(tm.shape[1])]
    return np.vstack((te.reshape(-1, len(electric)), tm.reshape(-1, len(electric)))), names
