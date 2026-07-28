"""End-to-end EM-correctness regression test for the subgrid src/rx fixes.

Runs a simple z-directed Hertzian dipole + receiver in free space two ways:
  1. plain fine grid everywhere (1mm), source/rx placed directly
  2. coarse main grid (3mm) + a subgrid (ratio 3 -> 1mm) with the source and
     receiver placed *inside the subgrid* via subgrid.add(...), with no
     explicit start/stop on the source

Both are compared against gprMax's own closed-form analytical solution for a
Hertzian dipole in free space. This exercises, on real field data rather than
just metadata, the two bugs fixed in SubGridBase.setup()/SubGridBaseGrid:

  - Position metadata (SubGridBaseGrid.local_to_global): if wrong, rxrel
    (rx position - src position, read back from the output file's Position
    attributes) would be wrong, and the analytical comparison below would
    fail even though the underlying field data was fine.
  - Timewindow inheritance (sg.timewindow = model.G.timewindow): before this
    fix, a subgrid source with no explicit stop defaulted to stop=0.0 and
    only fired at iteration 0, silently going to near-zero/noise for the
    rest of the run - the analytical comparison below would fail by many
    orders of magnitude (~0dB "agreement", per the original investigation).

This test takes ~15-20 seconds (it runs two real FDTD simulations) - it is
intentionally an integration test, not a fast unit test.
"""

from pathlib import Path

import gprMax
import h5py
import numpy as np
import pytest

from testing.analytical_solutions import hertzian_dipole_fs

OUTPUTS = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

# Agreement threshold (dB) for a healthy numerical model vs. the analytical
# solution. The previously-broken (timewindow=0) subgrid case measured ~0dB
# (i.e. no agreement); a correctly-functioning coarse/fine FDTD model of this
# size typically achieves -40dB or better. -20dB gives headroom against
# machine/precision variation while still failing loudly on a regression.
MAX_ACCEPTABLE_DIFF_DB = -20.0


def _max_diff_db(datatest: np.ndarray, dataref: np.ndarray) -> float:
    worst = -np.inf
    for i in range(dataref.shape[1]):
        maxi = np.amax(np.abs(dataref[:, i]))
        if maxi == 0:
            continue
        diff = np.abs(dataref[:, i] - datatest[:, i]) / maxi
        with np.errstate(divide="ignore"):
            diffdb = 20 * np.log10(diff)
        diffdb = diffdb[np.isfinite(diffdb)]
        if diffdb.size:
            worst = max(worst, np.amax(diffdb))
    return worst


def _compare_to_analytical(grp: h5py.Group) -> float:
    rxpos = grp["rxs/rx1"].attrs["Position"]
    txpos = grp["srcs/src1"].attrs["Position"]
    rxrel = tuple(rxpos - txpos)

    iterations = int(grp.attrs["Iterations"])
    dt = float(grp.attrs["dt"])
    dl = grp.attrs["dx_dy_dz"]

    dataref = hertzian_dipole_fs(iterations, dt, dl, rxrel)
    datatest = np.stack([grp[f"rxs/rx1/{o}"][:] for o in OUTPUTS], axis=1)

    return _max_diff_db(datatest, dataref)


def run_plain_model(tmp_path: Path) -> Path:
    """Fine grid (1mm) everywhere, source/rx placed directly."""
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="plain"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.06, 0.06, 0.06)))
    scene.add(gprMax.TimeWindow(time=3e-9))

    wf = gprMax.Waveform(wave_type="gaussianprime", amp=1, freq=1e9, id="mypulse")
    hd = gprMax.HertzianDipole(polarisation="z", p1=(0.025, 0.025, 0.025), waveform_id="mypulse")
    rx = gprMax.Rx(p1=(0.035, 0.035, 0.038))
    scene.add(wf)
    scene.add(hd)
    scene.add(rx)

    outfile = tmp_path / "plain"
    gprMax.run(scenes=[scene], n=1, outputfile=outfile, hide_progress_bars=True)
    return outfile.with_suffix(".h5")


def run_subgrid_model(tmp_path: Path) -> Path:
    """Coarse main grid (3mm) + a ratio-3 subgrid (1mm) enclosing the source
    and receiver, both added via subgrid.add(...) with no explicit
    start/stop - the exact scenario the timewindow-inheritance bug affected.
    Same relative source->rx offset as run_plain_model (0.01, 0.01, 0.013).
    """
    ratio = 3
    dl_sg = 1e-3
    dl_main = dl_sg * ratio

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="subgrid"))
    scene.add(gprMax.Discretisation(p1=(dl_main, dl_main, dl_main)))
    scene.add(gprMax.Domain(p1=(0.18, 0.18, 0.18)))
    scene.add(gprMax.TimeWindow(time=3e-9))

    subgrid = gprMax.SubGridHSG(p1=(0.075, 0.075, 0.075), p2=(0.105, 0.105, 0.105), ratio=ratio, id="sg")
    scene.add(subgrid)

    wf = gprMax.Waveform(wave_type="gaussianprime", amp=1, freq=1e9, id="mypulse")
    hd = gprMax.HertzianDipole(polarisation="z", p1=(0.085, 0.085, 0.085), waveform_id="mypulse")
    rx = gprMax.Rx(p1=(0.095, 0.095, 0.098))
    subgrid.add(wf)
    subgrid.add(hd)
    subgrid.add(rx)

    outfile = tmp_path / "subgrid"
    gprMax.run(
        scenes=[scene], n=1, outputfile=outfile, subgrid=True, autotranslate=True, hide_progress_bars=True
    )
    return outfile.with_suffix(".h5")


@pytest.fixture(scope="module")
def plain_h5(tmp_path_factory) -> Path:
    return run_plain_model(tmp_path_factory.mktemp("plain_model"))


@pytest.fixture(scope="module")
def subgrid_h5(tmp_path_factory) -> Path:
    return run_subgrid_model(tmp_path_factory.mktemp("subgrid_model"))


def test_subgrid_rx_position_matches_global_placement(subgrid_h5):
    """Regression guard for the Position-metadata bug: the subgrid source
    and receiver were placed at global (main-grid-frame) coordinates
    (0.085, 0.085, 0.085) and (0.095, 0.095, 0.098) - the output file must
    report those same global positions, not raw local subgrid indices.
    """
    with h5py.File(subgrid_h5, "r") as f:
        grp = f["subgrids/sg"]
        np.testing.assert_allclose(grp["srcs/src1"].attrs["Position"], (0.085, 0.085, 0.085))
        np.testing.assert_allclose(grp["rxs/rx1"].attrs["Position"], (0.095, 0.095, 0.098))


def test_plain_model_matches_analytical(plain_h5):
    """Sanity check on the reference model itself, and a baseline for how
    good agreement should be at this resolution/time window.
    """
    with h5py.File(plain_h5, "r") as f:
        maxdiff = _compare_to_analytical(f)
    assert maxdiff < MAX_ACCEPTABLE_DIFF_DB


def test_subgrid_model_matches_analytical(subgrid_h5):
    """The key EM-correctness check: a source/rx placed inside a subgrid
    with no explicit start/stop must radiate correctly for the full time
    window (timewindow-inheritance fix) at the correct physical position
    (local_to_global fix). Before both fixes this failed by many orders of
    magnitude (~0dB agreement, i.e. no agreement at all).
    """
    with h5py.File(subgrid_h5, "r") as f:
        maxdiff = _compare_to_analytical(f["subgrids/sg"])
    assert maxdiff < MAX_ACCEPTABLE_DIFF_DB
