"""Reject contradictory or ambiguous subgrid declarations before any build/device work."""
import pytest

import gprMax
from gprMax import config

pytestmark = pytest.mark.unit


def scene_with(*identifiers, ratio=1):
    scene = gprMax.Scene()
    for identifier in identifiers:
        scene.add(gprMax.SubGridHSG(p1=(0.02,) * 3, p2=(0.04,) * 3, ratio=ratio, id=identifier))
    return scene


@pytest.mark.parametrize("backend", [{}, {"gpu": [0]}, {"opencl": [0]}, {"metal": [0]}, {"mpi": [1, 1, 1]}])
@pytest.mark.parametrize("ratio", [1, 3])
def test_subgrid_flag_required_before_hardware_discovery(make_args, monkeypatch, backend, ratio):
    def unexpected():
        pytest.fail("Preflight must run before hardware discovery")

    monkeypatch.setattr(config, "get_host_info", unexpected)
    args = make_args(scenes=[scene_with("fine", ratio=ratio)], **backend)
    with pytest.raises(ValueError, match="subgrid=True"):
        config.SimulationConfig(args)


@pytest.mark.parametrize("identifier", ["", " ", ".", "..", "a/b", "a\\b", "a\x00b", None, 1])
def test_invalid_subgrid_identifier_rejected(make_args, identifier):
    with pytest.raises(ValueError, match="Subgrid ID"):
        config.SimulationConfig(make_args(scenes=[scene_with(identifier)], subgrid=True))


def test_duplicate_ids_rejected_before_build(make_args):
    with pytest.raises(ValueError, match="Duplicate subgrid ID 'fine'"):
        config.SimulationConfig(make_args(scenes=[scene_with("fine", "fine")], subgrid=True))


def test_ids_may_repeat_in_independent_scenes(make_sim_config):
    cfg = make_sim_config(scenes=[scene_with("fine"), scene_with("fine")], n=2, subgrid=True)
    assert cfg.general["subgrid"] is True


@pytest.mark.parametrize("backend", [{"opencl": [0]}, {"metal": [0]}, {"mpi": [1, 1, 1]}])
def test_enabled_subgrid_still_rejects_incompatible_backend(make_sim_config, backend):
    with pytest.raises(ValueError):
        make_sim_config(scenes=[scene_with("fine")], subgrid=True, **backend)


def test_enabled_subgrid_accepts_cuda(make_sim_config):
    """CUDA has subgrid kernels; ratio 3 selects the refining HSG path."""
    cfg = make_sim_config(scenes=[scene_with("fine", ratio=3)], subgrid=True, gpu=[0])
    assert cfg.general["subgrid"] is True
    assert cfg.general["solver"] == "cuda"
    # The subgrid block overrides gpu_precision, as it does for the CPU.
    assert cfg.general["precision"] == "double"
