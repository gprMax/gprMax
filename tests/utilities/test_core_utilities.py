# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

from types import SimpleNamespace

import numpy as np

from gprMax.utilities import host_info
from gprMax.utilities.utilities import fft_power


def test_fft_power_preserves_spectral_nulls():
    _, power = fft_power(np.ones(8), 1e-10)

    assert power[0] == 0
    assert np.all(np.isneginf(power[1:]))


def test_fft_power_zero_waveform_has_no_false_zero_db_peak():
    _, power = fft_power(np.zeros(8), 1e-10)

    assert np.all(np.isneginf(power))


def test_opencl_without_snapshots_skips_device_snapshot_check(monkeypatch):
    model_config = SimpleNamespace(mem_use=0, materials={"maxpoles": 0})
    sim_config = SimpleNamespace(general={"solver": "opencl"})
    monkeypatch.setattr(host_info.config, "sim_config", sim_config)
    monkeypatch.setattr(host_info.config, "get_model_config", lambda: model_config)
    monkeypatch.setattr(host_info, "mem_check_host", lambda memory: None)
    calls = []
    monkeypatch.setattr(
        host_info,
        "mem_check_device_snaps",
        lambda total, snapshots: calls.append((total, snapshots)),
    )

    grid = SimpleNamespace(
        mem_use=0,
        snapshots=[],
        mem_est_basic=lambda: 10,
        mem_est_dispersive=lambda: 0,
        name="main_grid",
    )
    host_info.mem_check_run_all([grid])

    assert calls == []


def test_linux_host_parser_supports_multi_digit_socket_counts(monkeypatch):
    responses = {
        ("cat", "/sys/class/dmi/id/sys_vendor"): b"Example Vendor\n",
        ("cat", "/sys/class/dmi/id/product_name"): b"Example Host\n",
        ("cat", "/proc/cpuinfo"): b"model name : Example CPU\n",
        ("lscpu",): b"Socket(s):             12\nThread(s) per core:    2\n",
    }

    def check_output(command, **kwargs):
        return responses[tuple(command)]

    monkeypatch.setattr(host_info.sys, "platform", "linux")
    monkeypatch.setattr(host_info.subprocess, "check_output", check_output)
    monkeypatch.setattr(host_info.psutil, "cpu_count", lambda logical=True: 24 if logical else 12)
    monkeypatch.setattr(
        host_info.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=1024),
    )

    result = host_info.get_host_info()

    assert result["sockets"] == 12
    assert result["hyperthreading"] is True
    assert result["machineID"] == "Example Vendor Example Host"
