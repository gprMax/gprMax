"""Argument binding used by the Metal far-field and SAR collectors.

These run without Metal: they check the per-type cache that replaced a
per-argument ``hasattr`` on PyObjC buffer proxies, which dominated the
solve time of Metal models with far-field surfaces.
"""

import pytest

from gprMax import sar_device
from gprMax.grid.metal_grid import MetalBufferView
from gprMax.ntff import device as ntff_device


class _RawBuffer:
    """Stands in for a PyObjC MTLBuffer proxy (no buffer/offset attributes)."""


@pytest.mark.parametrize("module", [ntff_device, sar_device])
def test_raw_buffer_binds_at_offset_zero(module):
    raw = _RawBuffer()
    assert module._metal_buffer_binding(raw) == (raw, 0)


@pytest.mark.parametrize("module", [ntff_device, sar_device])
def test_buffer_view_binds_its_buffer_and_offset(module):
    raw = _RawBuffer()
    view = MetalBufferView(raw, offset=64)
    assert module._metal_buffer_binding(view) == (raw, 64)


@pytest.mark.parametrize("module", [ntff_device, sar_device])
def test_view_check_is_cached_per_type(module, monkeypatch):
    monkeypatch.setattr(module, "_METAL_VIEW_TYPES", {})
    calls = []
    real_hasattr = hasattr

    def counting_hasattr(obj, name):
        calls.append(name)
        return real_hasattr(obj, name)

    monkeypatch.setattr("builtins.hasattr", counting_hasattr)
    for _ in range(5):
        module._metal_buffer_binding(_RawBuffer())
    assert calls == ["buffer"]
    assert module._METAL_VIEW_TYPES == {_RawBuffer: False}
