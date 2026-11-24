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

"""Frequency-window weights must be independent of rank partitioning."""

import importlib.util
import sys

import h5py
import numpy as np
import pytest

from .test_mpi_layered_time_integration import _mpi_launcher, _run


@pytest.mark.integration
@pytest.mark.skipif(
    _mpi_launcher() is None or importlib.util.find_spec("mpi4py") is None,
    reason="requires mpi4py and mpiexec",
)
def test_frequency_windows_match_serial_across_rank_boundary(tmp_path):
    model = tmp_path / "windows.in"
    commands = [
        "#domain: .08 .08 .08",
        "#dx_dy_dz: .004 .004 .004",
        "#time_window: 2e-10",
        "#pml_cells: 3",
        "#num_threads: 1",
        "#waveform: ricker 1 5e9 pulse",
        "#hertzian_dipole: z .04 .04 .04 pulse",
        "#ntff_surface: .028 .028 .028 .052 .052 .052 s",
        "#ntff_layered_background: bg z free_space",
    ]
    transforms = []
    for window in ("rectangular", "hann", "hann_end"):
        for kind in ("ksir", "ntff", "ntff_layered"):
            name = f"{kind}_{window}"
            transforms.append((name, window))
            background = " bg" if kind == "ntff_layered" else ""
            commands.append(f"#{kind}_frequency: s {name}{background} 5e9 {window}")
            far_command = "ksir_far_field" if kind == "ksir" else "ntff_far_field"
            commands.append(f"#{far_command}: 45 20 {name} {name}_far Etheta Ephi")
    model.write_text("\n".join(commands) + "\n", encoding="utf-8")
    common = (sys.executable, "-m", "gprMax", str(model), "--hide-progress-bars", "-cpu_precision", "double")
    serial_path, mpi_path = tmp_path / "serial", tmp_path / "mpi"
    _run((*common, "-o", str(serial_path)), tmp_path)
    _run((_mpi_launcher(), "-n", "2", *common, "--mpi", "2", "1", "1", "-o", str(mpi_path)), tmp_path)
    with h5py.File(serial_path.with_suffix(".h5")) as serial, h5py.File(mpi_path.with_suffix(".h5")) as mpi:
        for name, window in transforms:
            group_path = f"ntff/s/frequency/{name}"
            assert mpi[group_path].attrs["window"] == window
            if window == "hann_end":
                assert mpi[group_path].attrs["window_taper_fraction"] == 0.2
            for component in ("Etheta", "Ephi"):
                path = f"{group_path}/far_field/{name}_far/fields/{component}"
                expected = serial[path][...]
                np.testing.assert_allclose(mpi[path][...], expected, rtol=2e-12, atol=2e-12 * np.max(np.abs(expected)))
