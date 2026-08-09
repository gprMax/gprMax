"""Tests for the #domain_mode / DomainMode command.

DomainMode is the (optional) command a user declares to request 2D TM or
TE mode, ahead of the `inf`-coordinate resolution work it enables. It must
build before Domain (order 2 < Domain's order 3) so that
config.get_model_config().requested_2d_mode is already populated by the
time Domain.build() runs, and it must have zero effect on ordinary 3D
models that never use it.
"""
import tempfile
from pathlib import Path

import pytest

import gprMax
import gprMax.config as config
import gprMax.model as model_mod
from gprMax.hash_cmds_file import check_cmd_names
from gprMax.hash_cmds_singleuse import process_singlecmds
from gprMax.user_objects.cmds_singleuse import Domain, DomainMode


def test_requested_mode_is_uppercased():
    assert DomainMode(mode="te").requested_mode == "TE"
    assert DomainMode(mode="TM").requested_mode == "TM"
    assert DomainMode(mode="3d").requested_mode == "3D"


def test_order_is_before_domain():
    assert DomainMode(mode="TE").order < Domain(p1=(1, 1, 1)).order


@pytest.mark.parametrize("bad_mode", ["bogus", "2D", ""])
def test_invalid_mode_raises_on_build(bad_mode):
    with pytest.raises(ValueError):
        DomainMode(mode=bad_mode).build(None)


@pytest.mark.parametrize("mode", ["TM", "TE", "3D", "3d"])
def test_valid_modes_build_without_error(mode, monkeypatch):
    class _FakeModelConfig:
        requested_2d_mode = None

    fake = _FakeModelConfig()
    monkeypatch.setattr(config, "get_model_config", lambda: fake)

    DomainMode(mode=mode).build(None)

    assert fake.requested_2d_mode == mode.upper()


def test_hash_command_parses_single_token():
    lines = [
        "#domain_mode: TE\n",
        "#domain: 0.01 0.01 0.01\n",
        "#dx_dy_dz: 1e-3 1e-3 1e-3\n",
        "#time_window: 1e-12\n",
    ]
    singlecmds, _, _ = check_cmd_names(lines)
    assert singlecmds["#domain_mode"] == "TE"

    objs = process_singlecmds(singlecmds)
    domain_mode = next(o for o in objs if o.hash == "#domain_mode")
    assert domain_mode.requested_mode == "TE"


def test_hash_command_rejects_extra_tokens():
    lines = [
        "#domain_mode: TE extra\n",
        "#domain: 0.01 0.01 0.01\n",
        "#dx_dy_dz: 1e-3 1e-3 1e-3\n",
        "#time_window: 1e-12\n",
    ]
    singlecmds, _, _ = check_cmd_names(lines)
    with pytest.raises(ValueError):
        process_singlecmds(singlecmds)


def _capture_model_config(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["requested_2d_mode"] = config.get_model_config().requested_2d_mode
        captured["mode"] = config.get_model_config().mode

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def test_domain_mode_sets_requested_2d_mode_via_api(monkeypatch):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="te"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    # TE mode requires 'inf' on exactly one axis (see
    # tests/cmds_singleuse/test_domain_inf_resolution.py) - this test only
    # cares that requested_2d_mode is set, so any valid TE domain will do.
    scene.add(gprMax.Domain(p1=(0.01, 0.01, float("inf"))))
    # Domain is only 10 cells transverse; the default 10-cell PML on every
    # side would overlap itself (now correctly rejected - see
    # FDTDGrid._validate_pml_thickness()). PML is irrelevant to what this
    # test checks (requested_2d_mode wiring), so just disable it.
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))

    captured = _capture_model_config(monkeypatch)

    with tempfile.TemporaryDirectory() as td:
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=Path(td) / "domain_mode_test",
            hide_progress_bars=True,
        )

    assert captured["requested_2d_mode"] == "TE"


def test_no_domain_mode_leaves_requested_2d_mode_none(monkeypatch):
    """A model that never uses #domain_mode must be unaffected."""
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    # Domain is only 10 cells per axis; the default 10-cell PML on every
    # side would overlap itself. PML is irrelevant to what this test checks
    # (requested_2d_mode/mode wiring), so just disable it.
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))

    captured = _capture_model_config(monkeypatch)

    with tempfile.TemporaryDirectory() as td:
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=Path(td) / "no_domain_mode_test",
            hide_progress_bars=True,
        )

    assert captured["requested_2d_mode"] is None
    assert captured["mode"] == "3D"


def test_explicit_3d_mode_behaves_like_the_default(monkeypatch):
    """Explicitly requesting 3D must be accepted and resolve like the
    unset default, so a model can be toggled 2D<->3D by editing this
    command's argument alone, without deleting the command.
    """
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="3D"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    # Domain is only 10 cells per axis; the default 10-cell PML on every
    # side would overlap itself. PML is irrelevant to what this test checks
    # (requested_2d_mode/mode wiring), so just disable it.
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))

    captured = _capture_model_config(monkeypatch)

    with tempfile.TemporaryDirectory() as td:
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=Path(td) / "domain_mode_3d_test",
            hide_progress_bars=True,
        )

    assert captured["requested_2d_mode"] == "3D"
    assert captured["mode"] == "3D"
