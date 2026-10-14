"""Validate Scene scheduling before discovery/building, without changing reuse."""

import logging

import numpy as np
import pytest

import gprMax
from gprMax import config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("start", [1, 7])
def test_single_scene_expands_to_references_without_mutating_input(make_sim_config, start):
    scene = gprMax.Scene()
    supplied = [scene]
    cfg = make_sim_config(scenes=supplied, n=60, i=start)
    assert cfg.args.scenes is supplied
    assert supplied == [scene]
    assert len(cfg.scenes) == 60
    assert all(cfg.get_scene(index) is scene for index in range(start - 1, start + 59))
    assert cfg.geometry_fixed is False


def test_explicit_repetition_keeps_existing_list(make_sim_config):
    supplied = [gprMax.Scene()] * 3
    cfg = make_sim_config(scenes=supplied, n=3)
    assert cfg.scenes is supplied


def test_distinct_scenes_keep_order_on_restart(make_sim_config):
    supplied = [gprMax.Scene(), gprMax.Scene()]
    cfg = make_sim_config(scenes=supplied, n=2, i=7)
    assert cfg.get_scene(6) is supplied[0]
    assert cfg.get_scene(7) is supplied[1]


@pytest.mark.parametrize("supplied", [[], [gprMax.Scene(), gprMax.Scene()]])
def test_bad_scene_count_fails_before_hardware(make_args, monkeypatch, supplied):
    def unexpected_probe():
        pytest.fail("Hardware must not be probed for an invalid Scene count")

    monkeypatch.setattr(config, "get_host_info", unexpected_probe)
    with pytest.raises(ValueError, match="at least one Scene|only 2 Scene entries"):
        config.SimulationConfig(make_args(scenes=supplied, n=3, gpu=[0]))


@pytest.mark.parametrize("name", ["n", "i"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2", np.nan])
def test_run_indices_require_positive_integers(make_sim_config, name, value):
    with pytest.raises(ValueError, match="positive integer"):
        make_sim_config(**{name: value})


def test_numpy_integer_counts_remain_supported(make_sim_config):
    cfg = make_sim_config(n=np.int64(3), i=np.int64(7))
    assert (cfg.model_start, cfg.model_end) == (6, 9)


def test_hash_slots_are_independent_on_restart(make_sim_config):
    cfg = make_sim_config(n=3, i=7)
    first = gprMax.Scene()
    cfg.set_scene(first, 6)
    assert cfg.get_scene(6) is first
    assert cfg.get_scene(7) is None
    assert cfg.get_scene(8) is None


def test_cli_without_scenes_attribute(make_args):
    args = make_args(n=3)
    del args.scenes
    assert config.SimulationConfig(args).scenes == [None] * 3


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("stepping", [None, "zero", "source", "receiver"])
def test_repetition_warnings_match_for_both_forms(
    make_sim_config, caplog, explicit, fixed, stepping
):
    scene = gprMax.Scene()
    if stepping == "zero":
        scene.add(gprMax.SrcSteps(p1=(0, 0, 0)))
        scene.add(gprMax.RxSteps(p1=(0, 0, 0)))
    elif stepping == "source":
        scene.add(gprMax.SrcSteps(p1=(-0.001, 0, 0)))
    elif stepping == "receiver":
        scene.add(gprMax.RxSteps(p1=(0.001, 0, 0)))
    with caplog.at_level(logging.WARNING):
        make_sim_config(scenes=[scene] * (3 if explicit else 1), n=3, geometry_fixed=fixed)
    assert caplog.text.count("will be rebuilt for each") == int(not fixed)
    assert caplog.text.count("appear to repeat the same simulation") == int(
        stepping in (None, "zero")
    )


def test_prefix_selection_is_preserved_with_warning(make_sim_config, caplog):
    supplied = [gprMax.Scene(), gprMax.Scene(), gprMax.Scene()]
    cfg = make_sim_config(scenes=supplied, n=2)
    assert cfg.scenes is supplied
    assert "only the first 2 will run" in caplog.text


def test_fixed_geometry_distinct_scenes_warn_but_remain_compatible(make_sim_config, caplog):
    supplied = [gprMax.Scene(), gprMax.Scene()]
    cfg = make_sim_config(scenes=supplied, n=3, geometry_fixed=True)
    assert cfg.scenes is supplied
    assert "additional distinct Scenes will not be used" in caplog.text


def test_study_does_not_warn_about_unconfigured_repetition(make_sim_config, caplog):
    make_sim_config(scenes=[gprMax.Scene()], n=3, geometry_fixed=True, study=object())
    assert "repeat the same simulation" not in caplog.text
    assert "will be rebuilt" not in caplog.text


def test_single_run_does_not_warn(make_sim_config, caplog):
    make_sim_config(scenes=[gprMax.Scene()])
    assert not caplog.records
