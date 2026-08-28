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

"""End-to-end test that the PMC symmetry-boundary ghost-node update is
actually wired into the CPU solve loop (gprMax/solvers.py calling
CPUUpdates.update_symmetry_boundaries_electric, right after
update_electric_a()) - not just implemented and unit-tested in isolation.

A real time-stepped run (not geometry_only) with an x0 PMC boundary must
complete without exception and produce finite, genuinely propagating
fields at a nearby receiver - not NaN, and not a degenerate all-zero result
(which would mask a wave that never actually reached the receiver in time,
a trap this project has hit before with too-short time windows).
"""
import numpy as np
import h5py
import pytest
from numpy.testing import assert_allclose

import gprMax


def _dispersive_pmc_scene(kind):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pmc"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e10, id="w"))
    scene.add(gprMax.Material(er=3, se=0.001, mr=1, sm=0, id="medium"))
    if kind == "debye":
        scene.add(
            gprMax.AddDebyeDispersion(
                poles=1,
                er_delta=[2.0],
                tau=[1e-11],
                material_ids=["medium"],
            )
        )
    else:
        scene.add(
            gprMax.AddLorentzDispersion(
                poles=1,
                er_delta=[2.0],
                omega=[2e10],
                delta=[5e9],
                material_ids=["medium"],
            )
        )
    scene.add(
        gprMax.Box(
            p1=(0.0, 0.0, 0.0),
            p2=(0.02, 0.02, 0.02),
            material_id="medium",
        )
    )
    scene.add(
        gprMax.HertzianDipole(
            polarisation="z", p1=(0.0, 0.01, 0.01), waveform_id="w"
        )
    )
    scene.add(gprMax.Rx(p1=(0.01, 0.01, 0.01)))
    return scene


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["debye", "lorentz"])
def test_cuda_dispersive_pmc_matches_cpu(tmp_path, gpu_device, kind):
    outputs = {}
    for backend in ("cpu", "cuda"):
        path = tmp_path / f"pmc_{kind}_{backend}"
        options = (
            {"gpu": [gpu_device], "gpu_precision": "single"}
            if backend == "cuda"
            else {"cpu_precision": "single"}
        )
        gprMax.run(
            scenes=[_dispersive_pmc_scene(kind)],
            outputfile=path,
            hide_progress_bars=True,
            **options,
        )
        with h5py.File(path.with_suffix(".h5"), "r") as result:
            outputs[backend] = {
                component: result[f"rxs/rx1/{component}"][:]
                for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            }

    scale = max(np.max(np.abs(values)) for values in outputs["cpu"].values())
    assert scale > 0
    for component in outputs["cpu"]:
        assert_allclose(
            outputs["cuda"][component],
            outputs["cpu"][component],
            rtol=3e-4,
            atol=3e-4 * scale,
        )


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["debye", "lorentz"])
def test_opencl_dispersive_pmc_matches_cpu(tmp_path, opencl_device, kind):
    outputs = {}
    for backend in ("cpu", "opencl"):
        path = tmp_path / f"pmc_{kind}_{backend}"
        options = (
            {"opencl": [opencl_device], "gpu_precision": "single"}
            if backend == "opencl"
            else {"cpu_precision": "single"}
        )
        gprMax.run(
            scenes=[_dispersive_pmc_scene(kind)],
            outputfile=path,
            hide_progress_bars=True,
            **options,
        )
        with h5py.File(path.with_suffix(".h5"), "r") as result:
            outputs[backend] = {
                component: result[f"rxs/rx1/{component}"][:]
                for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            }

    scale = max(np.max(np.abs(values)) for values in outputs["cpu"].values())
    assert scale > 0
    for component in outputs["cpu"]:
        assert_allclose(
            outputs["opencl"][component],
            outputs["cpu"][component],
            rtol=3e-4,
            atol=3e-4 * scale,
        )


def test_pmc_symmetry_boundary_solve_produces_finite_propagating_field(tmp_path):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pmc"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e10, id="w"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.001, 0.01, 0.01), waveform_id="w"))
    scene.add(gprMax.Rx(p1=(0.01, 0.01, 0.01)))

    outputfile = tmp_path / "pmc_solve"
    gprMax.run(scenes=[scene], n=1, outputfile=outputfile, hide_progress_bars=True)

    import h5py

    with h5py.File(str(outputfile) + ".h5", "r") as f:
        ez = f["rxs/rx1/Ez"][:]
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            data = f["rxs/rx1/" + comp][:]
            assert not np.any(np.isnan(data)), f"{comp} contains NaN"

    assert np.max(np.abs(ez)) > 1e-3, "Ez is degenerate (all-zero) - wave never reached the receiver"


def test_pmc_symmetry_boundary_solve_with_lorentz_material_produces_finite_propagating_field(tmp_path):
    """Complex-pole (Lorentz) counterpart of the Debye test below - exercises
    CPUUpdates.update_symmetry_boundaries_electric()/_b()'s complex-pole
    branch (gprMax.cython.symmetry_boundaries_dispersive_complex), selected
    via materials["drudelorentz"].
    """
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pmc"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e10, id="w"))
    scene.add(gprMax.Material(er=3, se=0.001, mr=1, sm=0, id="metal"))
    scene.add(
        gprMax.AddLorentzDispersion(
            poles=1, er_delta=[2.0], omega=[2e10], delta=[5e9], material_ids=["metal"]
        )
    )
    scene.add(gprMax.Box(p1=(0.0, 0.0, 0.0), p2=(0.02, 0.02, 0.02), material_id="metal"))
    # Fed directly on the PMC wall - as with the Debye test, the scenario
    # most likely to catch a missing/broken Phase B.
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.0, 0.01, 0.01), waveform_id="w"))
    scene.add(gprMax.Rx(p1=(0.01, 0.01, 0.01)))

    outputfile = tmp_path / "pmc_solve_lorentz"
    gprMax.run(scenes=[scene], n=1, outputfile=outputfile, hide_progress_bars=True)

    import h5py

    with h5py.File(str(outputfile) + ".h5", "r") as f:
        ez = f["rxs/rx1/Ez"][:]
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            data = f["rxs/rx1/" + comp][:]
            assert not np.any(np.isnan(data)), f"{comp} contains NaN"

    assert np.max(np.abs(ez)) > 1e-3, "Ez is degenerate (all-zero) - wave never reached the receiver"


def test_pmc_symmetry_boundary_solve_with_debye_material_produces_finite_propagating_field(tmp_path):
    """Same wiring check as above, but with a Debye-dispersive material
    filling the domain (including the PMC wall itself) - exercises
    CPUUpdates.update_symmetry_boundaries_electric()'s dispersive branch and
    the new update_symmetry_boundaries_electric_b() call in gprMax/solvers.py
    (right before update_electric_b()), not just the non-dispersive path the
    test above covers.
    """
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pmc"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e10, id="w"))
    scene.add(gprMax.Material(er=3, se=0.001, mr=1, sm=0, id="soil"))
    scene.add(gprMax.AddDebyeDispersion(poles=1, er_delta=[2.0], tau=[1e-11], material_ids=["soil"]))
    scene.add(gprMax.Box(p1=(0.0, 0.0, 0.0), p2=(0.02, 0.02, 0.02), material_id="soil"))
    # Fed directly on the PMC wall - the scenario most likely to catch a
    # missing/broken Phase B (a source here is exactly the case where PML/
    # source corrections modify E after the ghost-node Phase A has run).
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.0, 0.01, 0.01), waveform_id="w"))
    scene.add(gprMax.Rx(p1=(0.01, 0.01, 0.01)))

    outputfile = tmp_path / "pmc_solve_debye"
    gprMax.run(scenes=[scene], n=1, outputfile=outputfile, hide_progress_bars=True)

    import h5py

    with h5py.File(str(outputfile) + ".h5", "r") as f:
        ez = f["rxs/rx1/Ez"][:]
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            data = f["rxs/rx1/" + comp][:]
            assert not np.any(np.isnan(data)), f"{comp} contains NaN"

    assert np.max(np.abs(ez)) > 1e-3, "Ez is degenerate (all-zero) - wave never reached the receiver"
