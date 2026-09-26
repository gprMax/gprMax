"""Routine SIBC coverage and rendering-free physics fixtures.

All parameter combinations remain collected. ``-m 'not slow'`` selects the
routine matrix; an unfiltered run exercises the full scientific validation.
"""

import pytest


@pytest.fixture
def suppress_sibc_fit_plots(monkeypatch):
    """Keep real fitting and field solves, but omit unused fit PNG rendering."""
    import gprMax.user_objects.cmds_multiuse as commands

    monkeypatch.setattr(commands, "plot_good_conductor_surface_impedance_fit", lambda **kwargs: None)


def _routine_2d_case(name, parameters):
    axes = (parameters["invariant"], parameters["normal"])
    polarization = ("TE", "TM").index(parameters["polarization"])
    if name == "test_reduced_boundary_and_modal_build":
        # All six mappings and both polarizations for Foster; the canonical
        # mapping additionally checks the analytic PMC and resistive limits.
        return parameters["resistance"] == "foster" or axes == (2, 0)
    if name == "test_rotated_virtual_fields":
        # 24 of 96 cases: retain every mapping/polarization/direction. Rotate
        # precision and active/passive choices across those combinations.
        orientation = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)).index(axes)
        direction = ("+", "-").index(parameters["direction"])
        active = bool((orientation + polarization) % 2)
        precision = ("single", "double")[(orientation + direction) % 2]
        return parameters["active"] == active and parameters["precision"] == precision
    if name == "test_2d_matches_independent_3d_extrusion":
        # 12 of 36 cases: every orientation/polarization/precision, with the
        # three surface laws distributed across the selected combinations.
        orientation = ((0, 1), (1, 2), (2, 0)).index(axes)
        precision = ("single", "double").index(parameters["precision"])
        resistance = (float("inf"), 5.0, "foster")[(orientation + polarization + precision) % 3]
        return parameters["resistance"] == resistance
    return True


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    # Mark before pytest applies its -m selection. Scope this policy to the
    # three expensive matrices; other physics and negative controls stay put.
    matrices = {
        "test_reduced_boundary_and_modal_build",
        "test_rotated_virtual_fields",
        "test_2d_matches_independent_3d_extrusion",
    }
    for item in items:
        if item.path.name == "test_2d_sibc.py" and item.originalname in matrices:
            if not _routine_2d_case(item.originalname, item.callspec.params):
                item.add_marker(pytest.mark.slow)
