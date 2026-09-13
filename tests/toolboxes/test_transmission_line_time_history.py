"""Native TL allocation endpoints are not physical plotted/exported samples."""

import h5py
import numpy as np
import pytest

from toolboxes.Utilities.trace_time import read_time_history

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("component", ["Vinc", "Iinc", "Vtotal", "Itotal"])
@pytest.mark.parametrize("native_axis", [False, True])
@pytest.mark.parametrize("extra", [0, 1])
def test_native_tl_histories_use_owning_grid_iterations(tmp_path, component, native_axis, extra):
    with h5py.File(tmp_path / "tl.h5", "w") as f:
        f.attrs.update(Iterations=12, dt=0.3)
        fine = f.create_group("subgrids/fine")
        fine.attrs.update(Iterations=36, dt=0.1)
        group = fine.create_group("tls/tl1")
        offset = -0.05 if component.startswith("I") else 0.0
        if native_axis:
            name = "time_current" if component.startswith("I") else "time_voltage"
            group[name] = offset + np.arange(36) * 0.1
        group[component] = np.arange(36 + extra)
        history = read_time_history(group[component])
        np.testing.assert_array_equal(history.samples, np.arange(36))
        np.testing.assert_allclose(history.time, offset + np.arange(36) * 0.1)


def test_bad_tl_length_is_not_silently_truncated(tmp_path):
    with h5py.File(tmp_path / "bad.h5", "w") as f:
        f.attrs.update(Iterations=5, dt=0.1)
        group = f.create_group("tls/tl1")
        group["Vinc"] = np.zeros(7)
        with pytest.raises(ValueError, match="Invalid physical transmission-line history length"):
            read_time_history(group["Vinc"])


def test_merged_tl_matrices_are_never_trimmed(tmp_path):
    with h5py.File(tmp_path / "merged.h5", "w") as f:
        f.attrs.update(Iterations=5, dt=0.1)
        group = f.create_group("tls/tl1")
        group["Vinc"] = np.zeros((6, 3))
        assert read_time_history(group["Vinc"], allow_matrix=True).samples.shape == (6, 3)
