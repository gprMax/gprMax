"""Regression tests for multi-use hash-command parser edge cases."""

from types import SimpleNamespace

import pytest

from gprMax.hash_cmds_file import get_user_objects
from gprMax.user_objects.cmds_multiuse import ExcitationFile, PMLCFS


@pytest.mark.parametrize(
    "command, message",
    (
        (
            "#plane_wave_angles: 0 0 0 1 1 1 90 0 90 w free_space 0",
            "requires ten parameters",
        ),
        (
            "#plane_wave_vector: 0 0 0 1 1 1 1 0 0 90",
            "requires 11 parameters",
        ),
        (
            "#plane_wave_vector: 0 0 0 1 1 1 1 0 0 90 w free_space 0",
            "requires 11 parameters",
        ),
        (
            "#plane_wave_axial: 0 0 0 1 1 1 90 x w 0",
            "requires nine parameters",
        ),
    ),
)
def test_plane_wave_commands_reject_incomplete_optional_groups(command, message):
    with pytest.raises(ValueError, match=message):
        get_user_objects([f"{command}\n"], checkessential=False)


def test_excitation_file_numeric_interpolation_values_use_numeric_types():
    objects = get_user_objects(["#excitation_file: waveform.txt 3 0.0\n"], checkessential=False)

    assert len(objects) == 1
    excitation = objects[0]
    assert isinstance(excitation, ExcitationFile)
    assert excitation.kind == 3
    assert excitation.fill_value == 0.0


def _pml_cfs(**overrides):
    kwargs = {
        "alphascalingprofile": "constant",
        "alphascalingdirection": "forward",
        "alphamin": 0,
        "alphamax": 0,
        "kappascalingprofile": "constant",
        "kappascalingdirection": "forward",
        "kappamin": 1,
        "kappamax": 1,
        "sigmascalingprofile": "quartic",
        "sigmascalingdirection": "forward",
        "sigmamin": 0,
        "sigmamax": None,
    }
    kwargs.update(overrides)
    return PMLCFS(**kwargs)


def test_pml_cfs_rejects_negative_sigma_maximum():
    grid = SimpleNamespace(pmls={"cfs": [], "profiles": {}})

    with pytest.raises(ValueError, match="zero or greater"):
        _pml_cfs(sigmamax=-1).build(grid)


def test_pml_cfs_none_is_case_insensitive():
    grid = SimpleNamespace(pmls={"cfs": [], "profiles": {}})

    _pml_cfs(sigmamax="none").build(grid)

    assert grid.pmls["cfs"][0].sigma.max is None
