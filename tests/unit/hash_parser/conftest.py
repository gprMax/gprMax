"""Shared fixtures for the hash parser test suite.

The three dispatcher functions under test all consume *plain* dicts/lists
produced by ``check_cmd_names`` — no globals, no I/O. Fixtures here just
provide fresh template dicts per test so that mutating one command's value
in place can't leak into the next test.

``process_include_files`` is the only function in this suite that touches
``gprMax.config`` (it reads ``sim_config.input_file_path.parent`` as the
fallback include-search root). Tests that exercise that branch set up the
patch locally; we do not autouse-patch it here, to keep the dispatcher
tests pure.
"""

import pytest


SINGLE_KEYS = (
    "#domain",
    "#dx_dy_dz",
    "#time_window",
    "#title",
    "#omp_threads",
    "#time_step_stability_factor",
    "#pml_formulation",
    "#pml_cells",
    "#src_steps",
    "#rx_steps",
    "#output_dir",
)

MULTI_KEYS = (
    "#geometry_view",
    "#geometry_objects_write",
    "#material",
    "#material_range",
    "#material_list",
    "#soil_peplinski",
    "#add_dispersion_debye",
    "#add_dispersion_lorentz",
    "#add_dispersion_drude",
    "#waveform",
    "#voltage_source",
    "#hertzian_dipole",
    "#magnetic_dipole",
    "#transmission_line",
    "#plane_wave_angles",
    "#plane_wave_axial",
    "#plane_wave_vector",
    "#excitation_file",
    "#rx",
    "#rx_array",
    "#snapshot",
    "#pml_cfs",
    "#include_file",
)


@pytest.fixture
def singlecmds_template():
    """Fresh dict mirroring what ``check_cmd_names`` builds for single cmds.

    Every legal key maps to ``None`` so a test only has to set the one
    command it cares about — every other branch in ``process_singlecmds``
    short-circuits on the ``is not None`` check.
    """
    return dict.fromkeys(SINGLE_KEYS, None)


@pytest.fixture
def multicmds_template():
    """Fresh dict mirroring what ``check_cmd_names`` builds for multi cmds.

    Each legal key maps to an empty list. ``process_multicmds`` iterates
    each list, so empty lists produce no scene objects and only the keys
    we populate exercise their dispatch branch.
    """
    return {key: [] for key in MULTI_KEYS}
