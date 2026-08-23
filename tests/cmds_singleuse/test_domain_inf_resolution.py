"""Tests for `inf`-axis resolution in Domain.build() (gprMax/user_objects/
cmds_singleuse.py), driven by #domain_mode / DomainMode.

Validation matrix under test:
  - inf count 0, mode TM/TE            -> error (mode declared, no axis given)
  - inf count 1, mode None             -> defaults to TM (see below), does NOT error
  - inf count 1, mode 3D (explicit)    -> error (inf given, but 3D explicitly
                                           requested - a genuine contradiction,
                                           not silently papered over)
  - inf count >1, any mode             -> error (only one axis may be inf)
  - inf count 1, mode TM               -> resolves that axis to 1 cell
  - inf count 1, mode TE               -> resolves that axis to 2 cells
  - inf count 0, mode None/3D          -> unchanged legacy behaviour,
                                           including old-style auto-detected
                                           TM via an explicit 1-cell axis

Design note: `inf` with no `#domain_mode` command at all used to be a hard
error ("requires '#domain_mode' to be set to 'TM' or 'TE' first"). It now
defaults to TM instead (the more common case for GPR models) - this is
purely additive (a combination that always used to raise now succeeds with
a sensible default), so no previously-working input file's behaviour
changes. An explicit `#domain_mode: 3D` combined with `inf` still errors,
since that's a genuine contradiction the user should be told about, not a
"forgot to declare" case to default around.
"""
import math
import tempfile
from pathlib import Path

import pytest

import gprMax
import gprMax.config as config
import gprMax.model as model_mod

INF = float("inf")


def _pml_thickness_for_domain(domain, dl):
    """Pick a PML thickness (6-tuple x0,y0,z0,xmax,ymax,zmax) compatible
    with FDTDGrid._validate_pml_thickness() for these small test domains.

    These domains are 10 cells transverse (well under the default 10-cell
    PML on every side, which would now correctly be rejected as
    overlapping - see FDTDGrid._validate_pml_thickness()). PML itself is
    irrelevant to what this file tests (inf-axis resolution), except that
    several assertions below check that the invariant axis's PML is zeroed
    (pre-existing Domain.build() behaviour) while a transverse axis's PML
    is left non-zero. So: use a small non-zero thickness (3) on any axis
    that resolves to more than 2 cells, and 0 on any axis that is `inf` or
    otherwise resolves to only 1-2 cells (the invariant axis, whichever
    one that turns out to be) - this keeps every axis PML-valid without
    changing any domain size or cell-count assertion.
    """
    thickness = [3, 3, 3, 3, 3, 3]
    for i, v in enumerate(domain):
        if math.isinf(v):
            small = True
        else:
            small = round(v / dl[i]) <= 2
        if small:
            thickness[i] = 0
            thickness[i + 3] = 0
    return tuple(thickness)


def _capture_model_config(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["mode"] = config.get_model_config().mode
        captured["nx"] = int(self.G.nx)
        captured["ny"] = int(self.G.ny)
        captured["nz"] = int(self.G.nz)
        captured["pmls"] = dict(self.G.pmls["thickness"])
        captured["dt"] = self.G.dt

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _run(monkeypatch, tmp_path, label, domain, mode=None, dl=(1e-3, 1e-3, 1e-3)):
    scene = gprMax.Scene()
    if mode is not None:
        scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=dl))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.PMLThickness(thickness=_pml_thickness_for_domain(domain, dl)))
    scene.add(gprMax.TimeWindow(time=1e-12))

    captured = _capture_model_config(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / label,
        hide_progress_bars=True,
    )
    return captured


def test_tm_inf_axis_resolves_to_one_cell(monkeypatch, tmp_path):
    captured = _run(monkeypatch, tmp_path, "tm_z", (0.01, 0.01, INF), mode="TM")
    assert captured["mode"] == "2D TMz"
    assert (captured["nx"], captured["ny"], captured["nz"]) == (10, 10, 1)
    assert captured["pmls"]["z0"] == 0
    assert captured["pmls"]["zmax"] == 0
    assert captured["pmls"]["x0"] != 0
    assert captured["pmls"]["y0"] != 0


def test_te_inf_axis_resolves_to_two_cells(monkeypatch, tmp_path):
    captured = _run(monkeypatch, tmp_path, "te_z", (0.01, 0.01, INF), mode="TE")
    assert captured["mode"] == "2D TEz"
    assert (captured["nx"], captured["ny"], captured["nz"]) == (10, 10, 2)
    assert captured["pmls"]["z0"] == 0
    assert captured["pmls"]["zmax"] == 0


@pytest.mark.parametrize("axis_index,axis_letter", [(0, "x"), (1, "y"), (2, "z")])
def test_inf_resolves_on_any_axis(monkeypatch, tmp_path, axis_index, axis_letter):
    domain = [0.01, 0.01, 0.01]
    domain[axis_index] = INF
    captured = _run(monkeypatch, tmp_path, f"te_{axis_letter}", tuple(domain), mode="TE")
    assert captured["mode"] == f"2D TE{axis_letter}"
    cells = (captured["nx"], captured["ny"], captured["nz"])
    assert cells[axis_index] == 2
    assert captured["pmls"][f"{axis_letter}0"] == 0
    assert captured["pmls"][f"{axis_letter}max"] == 0


def test_tm_and_te_give_identical_transverse_cfl(monkeypatch, tmp_path):
    """CFL depends only on the two transverse axes, so TM and TE for the
    same invariant axis and discretisation must give the same dt."""
    tm = _run(monkeypatch, tmp_path, "cfl_tm", (0.01, 0.01, INF), mode="TM")
    te = _run(monkeypatch, tmp_path, "cfl_te", (0.01, 0.01, INF), mode="TE")
    assert tm["dt"] == te["dt"]


def test_inf_without_domain_mode_defaults_to_tm(monkeypatch, tmp_path, capsys):
    captured = _run(monkeypatch, tmp_path, "no_mode", (0.01, 0.01, INF), mode=None)
    assert captured["mode"] == "2D TMz"
    assert (captured["nx"], captured["ny"], captured["nz"]) == (10, 10, 1)
    assert captured["pmls"]["z0"] == 0
    assert captured["pmls"]["zmax"] == 0
    assert "defaulting to TM" in capsys.readouterr().out


def test_inf_with_3d_mode_raises(monkeypatch, tmp_path):
    """Explicit '#domain_mode: 3D' + inf is a genuine contradiction, not a
    "forgot to declare" case - must still error, not default to TM."""
    with pytest.raises(ValueError, match="domain_mode"):
        _run(monkeypatch, tmp_path, "mode_3d", (0.01, 0.01, INF), mode="3D")


def test_explicit_te_mode_with_inf_unaffected_by_tm_default(monkeypatch, tmp_path, capsys):
    """An explicit '#domain_mode: TE' + inf must resolve to TE as always,
    with no "defaulting to TM" message (that only fires when
    '#domain_mode' was never declared at all)."""
    captured = _run(monkeypatch, tmp_path, "te_explicit", (0.01, 0.01, INF), mode="TE")
    assert captured["mode"] == "2D TEz"
    assert "defaulting to TM" not in capsys.readouterr().out


def test_legacy_one_cell_axis_logs_nudge_but_behaves_unchanged(monkeypatch, tmp_path, capsys):
    """Old-style implicit 1-cell-thick-axis detection (no '#domain_mode',
    no 'inf') must keep working exactly as before - only a new,
    informational nudge is added, no behaviour change."""
    captured = _run(monkeypatch, tmp_path, "legacy", (0.01, 0.01, 0.001), mode=None)
    assert captured["mode"] == "2D TMz"
    assert (captured["nx"], captured["ny"], captured["nz"]) == (10, 10, 1)
    assert captured["pmls"]["z0"] == 0
    assert "detected a 2D model from a 1-cell-thick axis" in capsys.readouterr().out


def test_legacy_3d_model_no_nudge(monkeypatch, tmp_path, capsys):
    """A genuine 3D model (no small axis at all) must not trigger the
    legacy nudge message."""
    captured = _run(monkeypatch, tmp_path, "plain_3d", (0.01, 0.01, 0.01), mode=None)
    assert captured["mode"] == "3D"
    assert "detected a 2D model" not in capsys.readouterr().out


def test_tm_mode_without_inf_raises(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="inf"):
        _run(monkeypatch, tmp_path, "tm_no_inf", (0.01, 0.01, 0.001), mode="TM")


def test_te_mode_without_inf_raises(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="inf"):
        _run(monkeypatch, tmp_path, "te_no_inf", (0.01, 0.01, 0.002), mode="TE")


def test_two_inf_axes_raises(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="at most one axis"):
        _run(monkeypatch, tmp_path, "two_inf", (0.01, INF, INF), mode="TE")


def test_legacy_1cell_auto_detect_tm_unaffected(monkeypatch, tmp_path):
    """A model that never uses #domain_mode at all, using the pre-existing
    convention of an explicit 1-cell-thick axis, must keep working exactly
    as before - zero behaviour change for old input files."""
    captured = _run(monkeypatch, tmp_path, "legacy_tm", (0.001, 0.01, 0.01), mode=None)
    assert captured["mode"] == "2D TMx"
    assert (captured["nx"], captured["ny"], captured["nz"]) == (1, 10, 10)


def test_legacy_3d_domain_unaffected(monkeypatch, tmp_path):
    captured = _run(monkeypatch, tmp_path, "legacy_3d", (0.01, 0.01, 0.01), mode=None)
    assert captured["mode"] == "3D"
    assert (captured["nx"], captured["ny"], captured["nz"]) == (10, 10, 10)


def test_explicit_3d_mode_with_no_inf_behaves_like_default(monkeypatch, tmp_path):
    captured = _run(monkeypatch, tmp_path, "explicit_3d", (0.01, 0.01, 0.01), mode="3D")
    assert captured["mode"] == "3D"
    assert (captured["nx"], captured["ny"], captured["nz"]) == (10, 10, 10)
