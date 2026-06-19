import pytest

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
