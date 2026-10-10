# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Command-line interface tests for the waveform preview tool."""

import matplotlib.pyplot as plt
import pytest

from gprMax.toolboxes.Plotting.plot_source_wave import check_timewindow, main

pytestmark = pytest.mark.unit


class TestMainValidation:
    def test_impulse_requires_a_positive_placeholder_frequency(self):
        """Impulse ignores frequency but still needs the placeholder."""
        with pytest.raises(ValueError, match="finite excitation frequency greater than zero"):
            main(["impulse", "1", "0", "6e-9", "1.926e-12", "-save"])

    def test_impulse_rejects_a_non_finite_placeholder(self):
        with pytest.raises(ValueError, match="finite excitation frequency greater than zero"):
            main(["impulse", "1", "inf", "6e-9", "1.926e-12", "-save"])

    def test_builtin_rejects_a_non_positive_frequency(self):
        # "-100" parses as a positional; "-1e9" would be swallowed by
        # argparse as an option flag before validation is reached.
        with pytest.raises(ValueError, match="finite excitation frequency greater than zero"):
            main(["gaussian", "1", "-100", "6e-9", "1.926e-12", "-save"])

    def test_rejects_a_non_finite_amplitude(self):
        with pytest.raises(ValueError, match="finite amplitude"):
            main(["gaussian", "inf", "1e9", "6e-9", "1.926e-12", "-save"])

    def test_rejects_a_non_positive_time_step(self):
        with pytest.raises(ValueError, match="Time step"):
            main(["gaussian", "1", "1e9", "6e-9", "0", "-save"])

    def test_rejects_a_non_positive_time_window(self):
        with pytest.raises(ValueError, match="Time window"):
            main(["gaussian", "1", "1e9", "0", "1.926e-12", "-save"])

    def test_unknown_waveform_exits_with_usage_error(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["nosuchwaveform", "1", "1e9", "6e-9", "1.926e-12", "-save"])
        assert excinfo.value.code == 2
        assert "invalid choice" in capsys.readouterr().err


class TestMainSuccess:
    def test_gaussian_preview_is_saved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        try:
            main(["gaussian", "1", "1e9", "6e-9", "1.926e-12", "-save"])
        finally:
            plt.close("all")
        saved = tmp_path / "gaussian.png"
        assert saved.exists()
        assert saved.stat().st_size > 0

    def test_impulse_preview_with_placeholder_is_saved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        try:
            main(["impulse", "1", "1", "6e-9", "1.926e-12", "-save"])
        finally:
            plt.close("all")
        saved = tmp_path / "impulse.png"
        assert saved.exists()
        assert saved.stat().st_size > 0


class TestCheckTimewindow:
    @pytest.mark.parametrize("dt", [0, -1e-12])
    def test_rejects_a_non_positive_time_step(self, dt):
        with pytest.raises(ValueError, match="Time step must be finite and greater than zero"):
            check_timewindow("6e-9", dt)

    def test_valid_window_still_accepted(self):
        timewindow, iterations = check_timewindow("6e-9", 1.926e-12)
        assert iterations > 1
        assert timewindow > 0
