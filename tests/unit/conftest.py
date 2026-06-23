from types import SimpleNamespace

import pytest

from gprMax.materials import DispersiveMaterial, Material
from gprMax.waveforms import Waveform


@pytest.fixture
def make_waveform():
    """Factory for Waveform instances with sensible defaults.

    Usage:
        w = make_waveform("gaussian", freq=1e9, amp=2.0)
    """

    def _make(wave_type, freq=1e9, amp=1.0):
        w = Waveform()
        w.type = wave_type
        w.freq = freq
        w.amp = amp
        return w

    return _make


@pytest.fixture
def make_material():
    """Factory for non-dispersive Material instances.

    The constitutive parameters default to free space; override what each
    test cares about.
    """

    def _make(ID="test", numID=0, er=1.0, se=0.0, mr=1.0, sm=0.0):
        m = Material(numID, ID)
        m.er = er
        m.se = se
        m.mr = mr
        m.sm = sm
        return m

    return _make


@pytest.fixture
def make_dispersive():
    """Factory for DispersiveMaterial instances.

    ``poles`` is a sequence of ``(deltaer, tau, alpha)`` triples. Their
    meaning depends on ``model``:
        debye   — tau is the relaxation time; alpha is unused
        lorentz — tau is the pole frequency; alpha is the damping coeff
        drude   — tau is the pole frequency; alpha is the inverse relax time
    """

    def _make(ID="disp", numID=0, model="debye", er=1.0, se=0.0, poles=()):
        m = DispersiveMaterial(numID, ID)
        m.type = model
        m.er = er
        m.se = se
        m.poles = len(poles)
        for deltaer, tau, alpha in poles:
            m.deltaer.append(deltaer)
            m.tau.append(tau)
            m.alpha.append(alpha)
        return m

    return _make


@pytest.fixture
def fake_grid():
    """Factory for a minimal FDTDGrid stand-in.

    The real FDTDGrid carries hundreds of attributes — Material methods only
    read ``dt``, ``dx``, ``dy``, ``dz``, and a ``materials`` list. A
    ``SimpleNamespace`` is enough at the unit level.
    """

    def _make(dt=1e-12, dx=1e-3, dy=1e-3, dz=1e-3, materials=None):
        return SimpleNamespace(
            dt=dt,
            dx=dx,
            dy=dy,
            dz=dz,
            materials=materials if materials is not None else [],
        )

    return _make
