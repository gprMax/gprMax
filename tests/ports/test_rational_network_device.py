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

"""End-to-end CPU/GPU parity for sparse rational-network terminals."""

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

import gprMax
from gprMax.cuda_opencl import knl_rational_network


def _scene():
    dl = 2e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.024, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))

    resistance = 25.0
    inductance = 2e-9
    capacitance = 0.5e-12
    alpha = -resistance / (2 * inductance)
    beta = np.sqrt(1 / (inductance * capacitance) - alpha**2)
    pole = alpha + 1j * beta
    residue = pole / (inductance * (pole - np.conj(pole)))
    scene.add(
        gprMax.RationalNetwork(
            id="series_rlc",
            poles=(pole, np.conj(pole)),
            residues=(residue, np.conj(residue)),
        )
    )
    scene.add(
        gprMax.NetworkTerminal(
            p1=(0.012, 0.01, 0.01),
            polarisation="z",
            network_id="series_rlc",
            id="feed",
        )
    )
    scene.add(gprMax.NetworkExcitation("feed", "pulse"))
    scene.add(gprMax.NetworkPort("feed", reference_impedance=50, spectrum_limit="nyquist"))

    real_pole = -2 * np.pi * 8e9
    scene.add(
        gprMax.RationalNetwork(
            id="real_pole",
            conductance=1 / 100,
            capacitance=0.2e-12,
            poles=(real_pole,),
            residues=(-real_pole / 100,),
        )
    )
    scene.add(
        gprMax.NetworkTerminal(
            p1=(0.01, 0.008, 0.008),
            polarisation="x",
            network_id="real_pole",
            id="feed_x",
        )
    )
    scene.add(gprMax.NetworkExcitation("feed_x", "pulse"))
    scene.add(gprMax.NetworkPort("feed_x", reference_impedance=50, spectrum_limit="nyquist"))

    scene.add(gprMax.RationalNetwork(id="rc", conductance=1 / 50, capacitance=0.3e-12))
    scene.add(
        gprMax.NetworkTerminal(
            p1=(0.014, 0.012, 0.01),
            polarisation="y",
            network_id="rc",
            id="feed_y",
        )
    )
    scene.add(gprMax.NetworkExcitation("feed_y", "pulse"))
    scene.add(gprMax.NetworkPort("feed_y", reference_impedance=50, spectrum_limit="nyquist"))
    scene.add(gprMax.Rx(p1=(0.014, 0.01, 0.01), id="field"))
    return scene


@pytest.mark.parametrize("backend", ["cuda", "opencl", "metal"])
def test_rational_network_template_substitutions_are_complete(backend):
    specification = knl_rational_network.update_rational_network
    arguments = specification[f"args_{backend}"].substitute({"REAL": "double"})
    body = specification["func"].substitute(
        {
            "CUDA_IDX": "int i = 0;" if backend == "cuda" else "",
            "REAL": "double",
            "NY_RNINFO": 6,
            "NY_RNPARAMS": 7,
            "NY_RNWAVEWHOLE": 8,
            "NY_RNWAVEHALF": 7,
            "NY_RNVOLTAGE": 8,
            "NY_RNCURRENT": 7,
        }
    )
    assert "$" not in arguments
    assert "$" not in body


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["cuda", "opencl"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_device_rational_network_matches_cpu(tmp_path, request, backend, precision):
    if backend == "cuda":
        device_options = {"gpu": [request.getfixturevalue("gpu_device")]}
    else:
        device_options = {"opencl": [request.getfixturevalue("opencl_device")]}

    cpu_path = tmp_path / f"cpu_network_{precision}"
    device_path = tmp_path / f"{backend}_network_{precision}"
    gprMax.run(
        scenes=[_scene()],
        n=1,
        outputfile=cpu_path,
        hide_progress_bars=True,
        cpu_precision=precision,
    )
    gprMax.run(
        scenes=[_scene()],
        n=1,
        outputfile=device_path,
        hide_progress_bars=True,
        gpu_precision=precision,
        **device_options,
    )

    paths = (
        "ports/feed/Vtotal",
        "ports/feed/Inetwork",
        "ports/feed/S11",
        "ports/feed/Zin",
        "ports/feed_x/Vtotal",
        "ports/feed_x/Inetwork",
        "ports/feed_x/S11",
        "ports/feed_x/Zin",
        "ports/feed_y/Vtotal",
        "ports/feed_y/Inetwork",
        "ports/feed_y/S11",
        "ports/feed_y/Zin",
        "rxs/rx1/Ez",
    )
    with h5py.File(str(cpu_path) + ".h5", "r") as output:
        cpu = {path: output[path][:] for path in paths}
        valid = {
            name: output[f"ports/{name}/valid_S11"][:].astype(bool)
            for name in ("feed", "feed_x", "feed_y")
        }
    with h5py.File(str(device_path) + ".h5", "r") as output:
        device = {path: output[path][:] for path in paths}
        for name in valid:
            valid[name] &= output[f"ports/{name}/valid_S11"][:].astype(bool)

    assert all(selector.any() for selector in valid.values())
    tolerance = 4e-5 if precision == "single" else 5e-8
    for path in paths:
        port_name = path.split("/")[1] if path.startswith("ports/") else None
        selector = valid[port_name] if path.endswith(("S11", "Zin")) else slice(None)
        scale = max(float(np.max(np.abs(cpu[path][selector]))), 1e-12)
        assert np.isfinite(device[path][selector]).all()
        assert_allclose(
            device[path][selector],
            cpu[path][selector],
            rtol=tolerance,
            atol=tolerance * scale,
        )
