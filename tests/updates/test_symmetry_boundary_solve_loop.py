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

import gprMax


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
