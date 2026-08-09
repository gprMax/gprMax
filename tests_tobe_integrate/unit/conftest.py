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

    Materials tests use the minimal attribute set (``dt``, ``dx``, ``dy``,
    ``dz``, ``materials``). Sources / receivers tests pass extra kwargs for
    the richer attributes those methods read (``iterations``, ``timewindow``,
    ``waveforms``, per-type source lists, ``IDlookup``, ``ID``, ``rxs``).
    A ``SimpleNamespace`` is enough at the unit level.
    """

    def _make(
        dt=1e-12,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        materials=None,
        iterations=10,
        timewindow=None,
        waveforms=None,
        voltagesources=None,
        hertziandipoles=None,
        magneticdipoles=None,
        transmissionlines=None,
        IDlookup=None,
        ID=None,
        rxs=None,
        **extra,
    ):
        if timewindow is None:
            timewindow = iterations * dt
        if IDlookup is None:
            IDlookup = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}
        ns = SimpleNamespace(
            dt=dt,
            dx=dx,
            dy=dy,
            dz=dz,
            materials=materials if materials is not None else [],
            iterations=iterations,
            timewindow=timewindow,
            waveforms=waveforms if waveforms is not None else [],
            voltagesources=voltagesources if voltagesources is not None else [],
            hertziandipoles=hertziandipoles if hertziandipoles is not None else [],
            magneticdipoles=magneticdipoles if magneticdipoles is not None else [],
            transmissionlines=transmissionlines if transmissionlines is not None else [],
            IDlookup=IDlookup,
            ID=ID,
            rxs=rxs if rxs is not None else [],
        )
        for key, val in extra.items():
            setattr(ns, key, val)
        return ns

    return _make


class _ConstantWaveform:
    """Test double for ``gprMax.waveforms.Waveform``.

    ``calculate_value(t, dt)`` returns ``value`` whenever ``t >= 0`` and
    ``0`` otherwise, so tests that need a non-trivial source amplitude can
    pin it without re-deriving the gaussian / ricker formula. Tests that
    care about the time-window logic should rely on ``Source.start``/
    ``Source.stop``, not on the waveform itself.
    """

    def __init__(self, ID="wf", freq=1e9, value=1.0, wave_type="gaussian"):
        self.ID = ID
        self.freq = freq
        self.type = wave_type
        self._value = value

    def calculate_value(self, time, dt):
        return self._value if time >= 0 else 0.0


@pytest.fixture
def make_constant_waveform():
    """Returns the ``_ConstantWaveform`` test double."""

    def _make(ID="wf", freq=1e9, value=1.0, wave_type="gaussian"):
        return _ConstantWaveform(ID=ID, freq=freq, value=value, wave_type=wave_type)

    return _make
