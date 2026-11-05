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

"""Hardware-optional CUDA and OpenCL regressions for axial discrete plane waves.

The cases are those singled out in the review of the 1D-grid termination:
the corrected ``Ex`` total-field/scattered-field update of axial-y plane
waves, in 3D and in 2D TE (the two-cell Yee thickness); a dispersive axial
source whose hard source sits at a nonzero offset in the 1D grid; and a
material interface directly on the upper TF/SF y face. Each case checks that
the GPU traces match the CPU and that the scattered-field receivers stay at
the round-off level of the precision. Tests skip when no device is available.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
D, N, PML = 1e-3, 30, 8
A, B = PML + 3, N - PML - 3
INF = float("inf")
# GPU-vs-CPU difference and scattered-field leakage, relative to the peak of
# the total field (dB). Measured values are about -300 dB in double and
# -125 dB in single precision on CUDA and OpenCL.
LIMITS = {"double": -250.0, "single": -100.0}


def _common(scene, nt):
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.Discretisation(p1=(D, D, D)))
    scene.add(gprMax.TimeWindow(iterations=nt))
    scene.add(gprMax.Waveform(wave_type="gaussiandot", amp=1, freq=6e9, id="w"))


def _receivers_3d(scene, axis):
    """One total-field receiver, then scattered-field receivers below and above
    the box along the propagation axis and beside it."""
    c = N // 2 * D
    lo, hi = (A - 2) * D, (B + 2) * D
    along = {"y": ((c, lo, c), (c, hi, c)), "z": ((c, c, lo), (c, c, hi))}[axis]
    for p in ((c, c, c), *along, (lo, c, c)):
        scene.add(gprMax.Rx(p1=p))


def _axial_y_3d():
    scene = gprMax.Scene()
    _common(scene, 600)
    scene.add(gprMax.Domain(p1=(N * D,) * 3))
    scene.add(gprMax.PMLThickness(thickness=PML))
    scene.add(gprMax.DiscretePlaneWaveAxial(p1=(A * D,) * 3, p2=(B * D,) * 3, axis="y", psi=0, waveform_id="w"))
    _receivers_3d(scene, "y")
    return scene


def _axial_y_two_cell_te():
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    _common(scene, 600)
    scene.add(gprMax.Domain(p1=(N * D, N * D, INF)))
    scene.add(gprMax.PMLThickness(thickness=(PML, PML, 0, PML, PML, 0)))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(p1=(A * D, A * D, INF), p2=(B * D, B * D, INF), axis="y", psi=0, waveform_id="w")
    )
    c = N // 2 * D
    for p in ((c, c, INF), (c, (A - 2) * D, INF), (c, (B + 2) * D, INF), ((A - 2) * D, c, INF)):
        scene.add(gprMax.Rx(p1=p))
    return scene


def _interface_on_upper_y_face():
    """A half-space of er = 4 starting exactly on the upper TF/SF y face."""
    scene = gprMax.Scene()
    _common(scene, 600)
    scene.add(gprMax.Domain(p1=(N * D,) * 3))
    scene.add(gprMax.PMLThickness(thickness=PML))
    scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id="half"))
    scene.add(gprMax.Box(p1=(0, B * D, 0), p2=(N * D,) * 3, material_id="half"))
    scene.add(gprMax.DiscretePlaneWaveAxial(p1=(A * D,) * 3, p2=(B * D,) * 3, axis="y", psi=0, waveform_id="w"))
    _receivers_3d(scene, "y")
    return scene


def _dispersive_axial_with_offset():
    """Debye background; pml_cells=40 puts the hard source 20 1D cells before
    its legacy position in both precisions (nonzero source offset)."""
    scene = gprMax.Scene()
    _common(scene, 700)
    scene.add(gprMax.Domain(p1=(N * D,) * 3))
    scene.add(gprMax.PMLThickness(thickness=PML))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="debye"))
    scene.add(gprMax.AddDebyeDispersion(poles=1, er_delta=(1.5,), tau=(2e-11,), material_ids=("debye",)))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(N * D,) * 3, material_id="debye"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(p1=(A * D,) * 3, p2=(B * D,) * 3, axis="z", psi=0, waveform_id="w", pml_cells=40)
    )
    _receivers_3d(scene, "z")
    return scene


CASES = {
    "axial_y_3d": _axial_y_3d,
    "axial_y_two_cell_te": _axial_y_two_cell_te,
    "interface_upper_y_face": _interface_on_upper_y_face,
    "dispersive_axial_offset": _dispersive_axial_with_offset,
}


def _run(scene, output: Path, precision, backend=None, device=None):
    if backend == "cuda":
        options = {"gpu": [device], "gpu_precision": precision}
    elif backend == "opencl":
        options = {"opencl": [device], "gpu_precision": precision}
    else:
        options = {"cpu_precision": precision}
    gprMax.run(scenes=[scene], outputfile=output, hide_progress_bars=True, log_level=40, **options)
    with h5py.File(output.with_suffix(".h5"), "r") as h:
        return [np.stack([np.asarray(h[f"rxs/rx{i}/{c}"], dtype=np.float64) for c in COMPONENTS]) for i in (1, 2, 3, 4)]


def _db(x):
    return 20 * np.log10(max(x, 1e-300))


_FIELDS = (slice(0, 3), slice(3, 6))  # E and H, each relative to its own total-field peak


def _parity(gpu, cpu):
    return max(np.abs(g[s] - c[s]).max() / np.abs(cpu[0][s]).max() for g, c in zip(gpu, cpu) for s in _FIELDS)


def _leak(rx):
    return max(np.abs(r[s]).max() / np.abs(rx[0][s]).max() for r in rx[1:] for s in _FIELDS)


_CPU = {}  # (case, precision) -> CPU traces, shared between the backends


@pytest.mark.parametrize("backend", ("cuda", "opencl"))
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("case", sorted(CASES))
def test_axial_plane_wave_matches_cpu_and_does_not_leak(request, tmp_path, tmp_path_factory, backend, precision, case):
    device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
    key = (case, precision)
    if key not in _CPU:
        _CPU[key] = _run(CASES[case](), tmp_path_factory.mktemp("cpu") / f"{case}_{precision}", precision)
    cpu = _CPU[key]
    gpu = _run(CASES[case](), tmp_path / f"{case}_{backend}_{precision}", precision, backend, device)

    assert _db(_parity(gpu, cpu)) < LIMITS[precision], "GPU trace differs from CPU"
    assert _db(_leak(gpu)) < LIMITS[precision], "scattered-field leakage"
