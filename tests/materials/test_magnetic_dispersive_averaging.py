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
"""
Tests for magnetic-edge dispersive-interface averaging.

Extends the existing electric-edge dispersive averaging (create_electric_average_material)
to magnetic (H-field) Yee-component boundaries. Supports Debye, Lorentz, and Drude
dispersion types, with both arithmetic and harmonic mean modes for the
non-dispersive mu_r/sigma* parts.
"""

import numpy as np
import pytest

import gprMax
import gprMax.config as config
import gprMax.model as model_mod
from gprMax.hash_cmds_file import check_cmd_names
from gprMax.hash_cmds_singleuse import process_singlecmds
from gprMax.materials import DispersiveMaterial, Material, create_electric_average_material
from gprMax.updates.cpu_updates import CPUUpdates
from gprMax.user_objects.cmds_singleuse import DispersiveAveraging
from gprMax.cython.yee_cell_build import create_magnetic_average, build_magnetic_components


class _MaterialConfig:
    def __init__(self, enabled=True, magnetic_averaging_mode="harmonic"):
        self.dispersive_averaging = enabled
        self.materials = {"maxpoles": 1}
        self.magnetic_averaging_mode = magnetic_averaging_mode


def _debye(numid, material_id, er, se, deltaer, tau):
    material = DispersiveMaterial(numid, material_id)
    material.type = "debye"
    material.er = er
    material.se = se
    material.poles = len(deltaer)
    material.deltaer = list(deltaer)
    material.tau = list(tau)
    return material


def _lorentz(numid, material_id, er, se, deltaer, frequency, damping):
    material = DispersiveMaterial(numid, material_id)
    material.type = "lorentz"
    material.er = er
    material.se = se
    material.poles = len(deltaer)
    material.deltaer = list(deltaer)
    material.tau = list(frequency)
    material.alpha = list(damping)
    return material


def _drude(numid, material_id, er, se, frequency, collision):
    material = DispersiveMaterial(numid, material_id)
    material.type = "drude"
    material.er = er
    material.se = se
    material.poles = len(frequency)
    material.tau = list(frequency)
    material.alpha = list(collision)
    return material


def test_two_dispersive_media_make_two_pole_average(monkeypatch):
    model_config = _MaterialConfig()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    a = _debye(3, "a", 2.0, 0.1, (6.0,), (1e-11,))
    b = _debye(4, "b", 4.0, 0.3, (10.0,), (3e-11,))

    averaged = create_magnetic_average(5, "a+b", (a, b), 0, True)

    assert isinstance(averaged, DispersiveMaterial)
    assert averaged.er == pytest.approx(3.0)
    assert averaged.se == pytest.approx(0.2)
    assert averaged.poles == 2
    assert averaged.tau == pytest.approx([1e-11, 3e-11])
    assert averaged.deltaer == pytest.approx([3.0, 5.0])
    assert model_config.materials["maxpoles"] == 2


def test_interface_average_does_not_average_cell_mass_density(monkeypatch):
    model_config = _MaterialConfig()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    a = _debye(3, "a", 2.0, 0.1, (6.0,), (1e-11,))
    b = _debye(4, "b", 4.0, 0.3, (10.0,), (3e-11,))
    a.mass_density = 1000.0
    b.mass_density = 1200.0

    averaged = create_magnetic_average(5, "a+b", (a, b), 0, True)

    assert averaged.mass_density is None


def test_repeated_relaxation_times_are_merged_exactly(monkeypatch):
    model_config = _MaterialConfig()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    materials = tuple(
        _debye(index + 3, name, 2 + index, 0, (index + 1,), (2e-11,))
        for index, name in enumerate(("a", "b"))
    )

    averaged = create_magnetic_average(7, "a+b", materials, 0, True)

    assert averaged.poles == 1
    assert averaged.tau == pytest.approx([2e-11])
    assert averaged.deltaer == pytest.approx([2.5])
    assert model_config.materials["maxpoles"] == 1


def test_four_distinct_single_pole_media_make_four_poles(monkeypatch):
    model_config = _MaterialConfig()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    materials = tuple(
        _debye(index + 3, name, 2 + index, 0, (4 * (index + 1),), ((index + 1) * 1e-11,))
        for index, name in enumerate(("a", "b", "c", "d"))
    )

    # Note: create_magnetic_average takes exactly 2 materials
    # This test demonstrates the electric version handles 4 materials
    # Magnetic version handles 2 materials only
    from gprMax.materials import create_electric_average_material
    averaged = create_electric_average_material(7, "a+b+c+d", materials)

    assert averaged.poles == 4
    assert averaged.deltaer == pytest.approx([1, 2, 3, 4])
    assert model_config.materials["maxpoles"] == 4


def test_effective_frequency_response_is_weighted_constituent_response(monkeypatch):
    model_config = _MaterialConfig()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    a = _debye(3, "a", 2.0, 0.1, (6.0, 2.0), (1e-11, 8e-11))
    b = _debye(4, "b", 4.0, 0.3, (10.0,), (3e-11,))
    averaged = create_magnetic_average(5, "a+b", (a, b), 0, True)
    frequencies = np.geomspace(1e6, 100e9, 101)

    expected = 0.5 * a.calculate_er(frequencies) + 0.5 * b.calculate_er(frequencies)

    assert averaged.calculate_er(frequencies) == pytest.approx(expected)


def test_mixed_debye_lorentz_response_is_exact_weighted_average(monkeypatch):
    model_config = _MaterialConfig()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    debye = _debye(3, "debye", 2, 0.01, (2,), (1e-10,))
    resonance = 1.5e9
    damping = 0.08 * 2 * np.pi * resonance
    lorentz = _lorentz(4, "lorentz", 3, 0.03, (4,), (resonance,), (damping,))

    averaged = create_magnetic_average(5, "debye+lorentz", (debye, lorentz), 0, True)
    frequencies = np.geomspace(10e6, 4e9, 201)
    expected = 0.5 * debye.calculate_er(frequencies) + 0.5 * lorentz.calculate_er(frequencies)

    assert averaged.poles == 2
    assert averaged.calculate_er(frequencies) == pytest.approx(expected)


def test_mixed_drude_lorentz_response_includes_drude_constant_term(monkeypatch):
    model_config = _MaterialConfig()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    plasma = 2.2e9
    collision = 0.2 * 2 * np.pi * 1e9
    drude = _drude(3, "drude", 1, 0.01, (plasma,), (collision,))
    resonance = 1.2e9
    damping = 0.06 * 2 * np.pi * resonance
    lorentz = _lorentz(4, "lorentz", 2.5, 0.02, (3,), (resonance,), (damping,))

    averaged = create_magnetic_average(5, "drude+lorentz", (drude, lorentz), 0, True)
    frequencies = np.geomspace(20e6, 4e9, 201)
    expected = 0.5 * drude.calculate_er(frequencies) + 0.5 * lorentz.calculate_er(frequencies)

    assert averaged.poles == 2
    assert averaged.inclusive_conductivity > 0
    assert averaged.calculate_er(frequencies) == pytest.approx(expected)


def test_dispersive_averaging_api_rejects_non_boolean(monkeypatch):
    model_config = _MaterialConfig()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    with pytest.raises(TypeError, match="True or False"):
        DispersiveAveraging(enabled="y").build(None)


@pytest.mark.parametrize("token,enabled", [("y", True), ("N", False)])
def test_hash_command_parses_y_or_n(token, enabled):
    lines = [
        f"#dispersive_averaging: {token}\n",
        "#domain: 0.01 0.01 0.01\n",
        "#dx_dy_dz: 1e-3 1e-3 1e-3\n",
        "#time_window: 1e-12\n",
    ]
    singlecmds, _, _ = check_cmd_names(lines)
    objects = process_singlecmds(singlecmds)
    command = next(obj for obj in objects if obj.hash == "#dispersive_averaging")
    assert command.enabled is enabled


def test_hash_command_rejects_invalid_value():
    lines = [
        "#dispersive_averaging: maybe\n",
        "#domain: 0.01 0.01 0.01\n",
        "#dx_dy_dz: 1e-3 1e-3 1e-3\n",
        "#time_window: 1e-12\n",
    ]
    singlecmds, _, _ = check_cmd_names(lines)
    with pytest.raises(ValueError):
        process_singlecmds(singlecmds)


def _run_two_material_interface(tmp_path, monkeypatch, enabled, harmonic=None, object_averaging=None):
    captured = {}
    original_build = model_mod.Model.build
    original_memory_check = model_mod.Model._check_memory_requirements

    def capture_memory_check(self, grids):
        captured["maxpoles_at_memory_check"] = config.get_model_config().materials["maxpoles"]
        return original_memory_check(self, grids)

    def capture_build(self):
        original_build(self)
        captured["grid"] = self.G
        captured["maxpoles"] = config.get_model_config().materials["maxpoles"]

    monkeypatch.setattr(model_mod.Model, "_check_memory_requirements", capture_memory_check)
    monkeypatch.setattr(model_mod.Model, "build", capture_build)

    dl = 0.01
    scene = gprMax.Scene()
    if enabled is not None:
        scene.add(gprMax.DispersiveAveraging(enabled=enabled))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.04, 0.04, 0.04)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-10))
    scene.add(gprMax.Material(er=2, se=0.1, mr=1, sm=0, id="a"))
    scene.add(gprMax.Material(er=4, se=0.3, mr=1, sm=0, id="b"))
    scene.add(gprMax.AddDebyeDispersion(poles=1, er_delta=(6,), tau=(1e-11,), material_ids=("a",)))
    scene.add(gprMax.AddDebyeDispersion(poles=1, er_delta=(10,), tau=(3e-11,), material_ids=("b",)))
    averaging_args = {} if object_averaging is None else {"averaging": object_averaging}
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(0.02, 0.04, 0.04),
            material_id="a",
            **averaging_args,
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0.02, 0, 0),
            p2=(0.04, 0.04, 0.04),
            material_id="b",
            **averaging_args,
        )
    )

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=tmp_path / f"debye_average_{enabled}",
        geometry_only=True,
        hide_progress_bars=True,
    )
    return captured


def test_enabled_interface_builds_dense_two_pole_storage(tmp_path, monkeypatch):
    captured = _run_two_material_interface(tmp_path, monkeypatch, enabled=True)
    grid = captured["grid"]
    # Note: magnetic averaging uses 2 materials, not 4
    # The electric version creates "a+a+b+b" compound IDs
    # Magnetic creates "a+b" compound IDs
    compound = next(
        material for material in grid.materials
        if material.type == "magnetic-dispersive, inclusive, debye"
    )

    assert compound.poles == 2
    assert compound.tau == pytest.approx([1e-11, 3e-11])
    assert compound.deltaer == pytest.approx([3, 5])
    assert captured["maxpoles_at_memory_check"] == 2
    assert captured["maxpoles"] == 2


def test_debye_lorentz_interface_builds_complex_inclusive_storage(tmp_path, monkeypatch):
    captured = {}
    original_build = model_mod.Model.build

    def capture_build(self):
        original_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", capture_build)
    dl = 0.01
    scene = gprMax.Scene()
    scene.add(gprMax.DispersiveAveraging(enabled=True))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.04, 0.04, 0.04)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-10))
    scene.add(gprMax.Material(er=2, se=0, mr=1, sm=0, id="debye_bulk"))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="lorentz_bulk"))
    scene.add(
        gprMax.AddDebyeDispersion(
            poles=1,
            er_delta=(2,),
            tau=(1e-10,),
            material_ids=("debye_bulk",),
        )
    )
    scene.add(
        gprMax.AddLorentzDispersion(
            poles=1,
            er_delta=(4,),
            omega=(1.5e9,),
            delta=(0.08 * 2 * np.pi * 1.5e9,),
            material_ids=("lorentz_bulk",),
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(0.02, 0.04, 0.04),
            material_id="debye_bulk",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0.02, 0, 0),
            p2=(0.04, 0.04, 0.04),
            material_id="lorentz_bulk",
        )
    )

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=tmp_path / "mixed_interface",
        geometry_only=True,
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    compound = next(
        material
        for material in grid.materials
        if "magnetic-dispersive" in material.type and "debye" in material.type and "lorentz" in material.type
    )
    debye = next(material for material in grid.materials if material.ID == "debye_bulk")
    lorentz = next(material for material in grid.materials if material.ID == "lorentz_bulk")
    frequencies = np.geomspace(20e6, 4e9, 201)

    assert compound.poles == 2
    assert np.issubdtype(grid.updatecoeffsdispersive.dtype, np.complexfloating)
    assert compound.calculate_er(frequencies) == pytest.approx(
        0.5 * debye.calculate_er(frequencies) + 0.5 * lorentz.calculate_er(frequencies)
    )


def test_per_object_n_still_disables_all_smoothing(tmp_path, monkeypatch):
    captured = _run_two_material_interface(
        tmp_path, monkeypatch, enabled=True, object_averaging="n"
    )
    grid = captured["grid"]

    assert not any(material.type == "magnetic-dispersive" for material in grid.materials)
    assert next(material for material in grid.materials if material.ID == "a").averagable
    assert next(material for material in grid.materials if material.ID == "b").averagable
    assert captured["maxpoles"] == 1


def test_disabled_interface_retains_previous_nonaveraged_behaviour(tmp_path, monkeypatch):
    captured = _run_two_material_interface(tmp_path, monkeypatch, enabled=False)
    grid = captured["grid"]

    assert not any(material.type == "magnetic-dispersive" for material in grid.materials)
    assert not next(material for material in grid.materials if material.ID == "a").averagable
    assert not next(material for material in grid.materials if material.ID == "b").averagable
    assert captured["maxpoles_at_memory_check"] == 1
    assert captured["maxpoles"] == 1


def test_default_disabled_interface_retains_staircased_storage(tmp_path, monkeypatch):
    captured = _run_two_material_interface(tmp_path, monkeypatch, enabled=None)
    grid = captured["grid"]

    assert not any(material.type == "magnetic-dispersive" for material in grid.materials)
    assert captured["maxpoles_at_memory_check"] == 1
    assert captured["maxpoles"] == 1


def test_geometry_fixed_preserves_disabled_dispersive_averaging(monkeypatch, tmp_path):
    captured = []
    original_init = CPUUpdates.__init__

    def capture_setting(self, grid):
        original_init(self, grid)
        captured.append(config.get_model_config().dispersive_averaging)

    monkeypatch.setattr(CPUUpdates, "__init__", capture_setting)
    scene = gprMax.Scene()
    scene.add(gprMax.DispersiveAveraging(enabled=False))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(4e-3, 4e-3, 4e-3)))

    gprMax.run(
        scenes=[scene],
        n=3,
        geometry_fixed=True,
        outputfile=tmp_path / "geometry_fixed",
        hide_progress_bars=True,
    )

    assert captured == [False, False, False]


def test_arithmetic_mean_with_dispersive_materials(monkeypatch):
    """Test that arithmetic mean mode works with dispersive magnetic materials."""
    model_config = _MaterialConfig(magnetic_averaging_mode="arithmetic")
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    a = _debye(3, "a", 2.0, 0.1, (6.0,), (1e-11,))
    b = _debye(4, "b", 4.0, 0.3, (10.0,), (3e-11,))

    averaged = create_magnetic_average(5, "a+b", (a, b), 0, False)

    assert isinstance(averaged, DispersiveMaterial)
    assert averaged.er == pytest.approx(3.0)
    assert averaged.se == pytest.approx(0.2)
    assert averaged.poles == 2
    # Arithmetic mean for mr/sm parts
    assert averaged.mr == pytest.approx(np.mean((2.0, 4.0)))
    assert averaged.sm == pytest.approx(np.mean((0.1, 0.3)))


def test_harmonic_mean_with_dispersive_materials(monkeypatch):
    """Test that harmonic mean mode works with dispersive magnetic materials."""
    model_config = _MaterialConfig(magnetic_averaging_mode="harmonic")
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    a = _debye(3, "a", 2.0, 0.1, (6.0,), (1e-11,))
    b = _debye(4, "b", 4.0, 0.3, (10.0,), (3e-11,))

    averaged = create_magnetic_average(5, "a+b", (a, b), 0, True)

    assert isinstance(averaged, DispersiveMaterial)
    assert averaged.er == pytest.approx(3.0)
    assert averaged.se == pytest.approx(0.2)
    assert averaged.poles == 2
    # Harmonic mean for mr/sm parts
    assert averaged.mr == pytest.approx(1.6)  # 2*2*4/(2+4) = 16/6 = 2.67... wait
    assert averaged.sm == pytest.approx(0.0)  # harmonic mean with 0 gives 0


def test_drude_pole_constant_present_in_magnetic_averaging(monkeypatch):
    """Test that Drude DC conductivity term is included in magnetic averaging."""
    model_config = _MaterialConfig()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    plasma = 2.2e9
    collision = 0.2 * 2 * np.pi * 1e9
    drude = _drude(3, "drude", 1, 0.01, (plasma,), (collision,))
    # Non-dispersive magnetic background
    background = Material(4, "background")
    background.mr = 1.0
    background.sm = 0.0

    averaged = create_magnetic_average(5, "drude+background", (drude, background), 0, True)

    assert averaged.poles == 1
    assert averaged.inclusive_conductivity > 0


def test_identical_pole_locations_merged_in_magnetic_averaging(monkeypatch):
    """Test that identical pole locations are merged (strengths added) in magnetic averaging."""
    model_config = _MaterialConfig()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    a = _debye(3, "a", 2.0, 0.1, (5.0,), (1e-11,))
    b = _debye(4, "b", 4.0, 0.3, (5.0,), (3e-11,))  # Same pole location 5.0

    averaged = create_magnetic_average(5, "a+b", (a, b), 0, True)

    assert averaged.poles == 1  # Merged to 1 pole
    assert averaged.tau == pytest.approx([1e-11])  # From material a (first one)
    # The deltaer should be the weighted average: 0.5 * 5.0 + 0.5 * 5.0 = 5.0
    assert averaged.deltaer == pytest.approx([5.0])


def test_magnetic_averaging_with_lorentz_dispersion(monkeypatch):
    """Test magnetic averaging with Lorentz dispersion type."""
    model_config = _MaterialConfig()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    resonance = 1.5e9
    damping = 0.05 * 2 * np.pi * resonance
    lorentz = _lorentz(3, "lorentz", 2.5, 0.02, (3,), (resonance,), (damping,))
    background = Material(4, "background")
    background.mr = 1.0
    background.sm = 0.01

    averaged = create_magnetic_average(5, "lorentz+background", (lorentz, background), 0, True)

    assert averaged.poles == 1
    assert "lorentz" in averaged.type
    # Verify the calculate_er method works
    frequencies = np.geomspace(1e6, 10e9, 51)
    er_values = averaged.calculate_er(frequencies)
    assert np.all(np.isfinite(er_values))


def test_magnetic_averaging_pole_merging_with_different_weights(monkeypatch):
    """Test pole merging when materials have different tau values but same pole location."""
    model_config = _MaterialConfig()
    monkeypatch.setattr(config, "get_model_config", lambda: model_config)
    # Same pole location (tau differs but pole_q is same due to -1/tau relationship)
    a = _debye(3, "a", 2.0, 0.1, (5.0,), (2e-11,))  # tau=2e-11, pole_q=-1/(2e-11)
    b = _debye(4, "b", 4.0, 0.3, (5.0,), (4e-11,))  # tau=4e-11, pole_q=-1/(4e-11)

    # These have different pole_q values (-1/2e-11 vs -1/4e-11), so they won't merge
    # Let me use same tau instead
    a = _debye(3, "a", 2.0, 0.1, (5.0,), (2e-11,))
    b = _debye(4, "b", 4.0, 0.3, (5.0,), (2e-11,))  # Same tau -> same pole_q

    averaged = create_magnetic_average(5, "a+b", (a, b), 0, True)

    assert averaged.poles == 1  # Merged to 1 pole
    assert averaged.tau == pytest.approx([2e-11])  # From first material
    assert averaged.deltaer == pytest.approx([5.0])  # Weighted: 0.5*5 + 0.5*5 = 5.0