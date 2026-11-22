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

from gprMax.toolboxes.Plotting.plot_source_wave import check_timewindow, main, mpl_plot
from gprMax.waveforms import Waveform

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

    def test_rejects_an_infinite_time_window(self):
        with pytest.raises(
            ValueError, match="Time window must have a finite value greater than zero"
        ):
            main(["gaussian", "1", "1e9", "inf", "1.926e-12", "-save"])

    def test_unknown_waveform_exits_with_usage_error(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["nosuchwaveform", "1", "1e9", "6e-9", "1.926e-12", "-save"])
        assert excinfo.value.code == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_rejects_an_empty_output_file(self):
        with pytest.raises(ValueError, match="Output file path cannot be empty"):
            main(["gaussian", "1", "1e9", "6e-9", "1.926e-12", "-o", ""])

    def test_rejects_a_whitespace_output_file(self):
        with pytest.raises(ValueError, match="Output file path cannot be empty"):
            main(["gaussian", "1", "1e9", "6e-9", "1.926e-12", "--output-file", "   "])

    def test_mpl_plot_rejects_empty_output_file(self):
        w = Waveform()
        w.type = "gaussian"
        w.amp = 1
        w.freq = 1e9
        with pytest.raises(ValueError, match="Output file path cannot be empty"):
            mpl_plot(w, 6e-9, 1.926e-12, 100, output_file="")


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

    def test_custom_output_filename_is_saved(self, tmp_path):
        target = tmp_path / "custom_wave.png"
        try:
            main(["gaussian", "1", "1e9", "6e-9", "1.926e-12", "-o", str(target)])
        finally:
            plt.close("all")
        assert target.exists()
        assert target.stat().st_size > 0

    def test_custom_output_file_long_flags_are_accepted(self, tmp_path):
        target_file = tmp_path / "from_output_file.png"
        target_alias = tmp_path / "from_output.png"
        try:
            main(["gaussian", "1", "1e9", "6e-9", "1.926e-12", "--output-file", str(target_file)])
            plt.close("all")
            main(["gaussian", "1", "1e9", "6e-9", "1.926e-12", "--output", str(target_alias)])
        finally:
            plt.close("all")
        assert target_file.exists()
        assert target_file.stat().st_size > 0
        assert target_alias.exists()
        assert target_alias.stat().st_size > 0

    def test_custom_output_directory_is_saved(self, tmp_path):
        try:
            main(["gaussian", "1", "1e9", "6e-9", "1.926e-12", "-o", str(tmp_path)])
        finally:
            plt.close("all")
        saved = tmp_path / "gaussian.png"
        assert saved.exists()
        assert saved.stat().st_size > 0

    def test_custom_output_nested_directory_created(self, tmp_path):
        target = tmp_path / "nested" / "sub" / "wave.png"
        try:
            main(["gaussian", "1", "1e9", "6e-9", "1.926e-12", "-o", str(target)])
        finally:
            plt.close("all")
        assert target.exists()
        assert target.stat().st_size > 0

    def test_custom_output_trailing_slash_directory_created(self, tmp_path):
        target_dir = tmp_path / "new_folder"
        try:
            main(["gaussian", "1", "1e9", "6e-9", "1.926e-12", "-o", f"{target_dir}/"])
        finally:
            plt.close("all")
        saved = target_dir / "gaussian.png"
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

    @pytest.mark.parametrize("tw", ["inf", "nan", "-inf", float("inf"), float("nan")])
    def test_rejects_a_non_finite_time_window(self, tw):
        with pytest.raises(
            ValueError, match="Time window must have a finite value greater than zero"
        ):
            check_timewindow(tw, 1.926e-12)
