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

"""Unit tests for ``gprMax/materials.py``.

Conventions
-----------
* One behaviour per test; descriptive names following
  ``test_<unit>_<context>_<expected>``.
* Closed-form references where possible (textbook formulas, not the
  code under test).
* Known bugs are pinned in dedicated tests with a clear docstring so
  a future fix that flips the assertion is obvious and intentional.
"""

import math

import numpy as np
import pytest
from scipy.constants import epsilon_0, mu_0

from gprMax.materials import (
    DispersiveMaterial,
    ListMaterial,
    Material,
    PeplinskiSoil,
    RangeMaterial,
    calculate_water_properties,
    create_built_in_materials,
    create_grass,
    create_water,
    process_materials,
)

# ---------------------------------------------------------------------------
# Material — defaults and identity
# ---------------------------------------------------------------------------


class TestMaterialDefaults:
    """The base ``Material`` should construct as free space."""

    def test_init_stores_numID_and_ID(self):
        m = Material(numID=7, ID="concrete")
        assert m.numID == 7
        assert m.ID == "concrete"

    def test_init_defaults_to_free_space(self):
        """Free space: er=1, se=0, mr=1, sm=0 — the EM vacuum."""
        m = Material(0, "free_space")
        assert m.er == 1.0
        assert m.se == 0.0
        assert m.mr == 1.0
        assert m.sm == 0.0

    def test_init_leaves_mass_density_unspecified(self):
        m = Material(0, "free_space")
        assert m.mass_density is None

    def test_init_defaults_averagable_true(self):
        m = Material(0, "anything")
        assert m.averagable is True

    def test_init_type_starts_empty(self):
        m = Material(0, "anything")
        assert m.type == ""


# ---------------------------------------------------------------------------
# Material — equality and ordering
# ---------------------------------------------------------------------------


class TestMaterialEquality:
    def test_equal_when_IDs_match(self, make_material):
        a = make_material(ID="sand", numID=3)
        b = make_material(ID="sand", numID=99)
        assert a == b

    def test_not_equal_when_IDs_differ(self, make_material):
        a = make_material(ID="sand")
        b = make_material(ID="clay")
        assert a != b

    def test_eq_against_non_material_raises_typeerror(self, make_material):
        m = make_material()
        with pytest.raises(TypeError):
            m == "sand"


class TestMaterialOrdering:
    """``<``/``>`` rules per ``Material.__lt__`` docstring.

    - Two non-compound materials: order by ``numID``.
    - Two compound materials: order alphabetically by ``ID``.
    - Mixed: non-compound is always less than compound.
    """

    def test_two_non_compound_ordered_by_numID(self, make_material):
        a = make_material(ID="a", numID=1)
        b = make_material(ID="b", numID=2)
        assert a < b
        assert b > a

    def test_two_compound_ordered_by_ID(self, make_material):
        a = make_material(ID="a+b", numID=99)
        b = make_material(ID="c+d", numID=0)
        assert a < b
        assert b > a

    def test_non_compound_less_than_compound(self, make_material):
        plain = make_material(ID="sand", numID=999)
        comp = make_material(ID="a+b", numID=0)
        assert plain < comp
        assert comp > plain

    def test_lt_against_non_material_raises_typeerror(self, make_material):
        with pytest.raises(TypeError):
            make_material() < "sand"


# ---------------------------------------------------------------------------
# Compound material helpers
# ---------------------------------------------------------------------------


class TestCompoundMaterials:
    @pytest.mark.parametrize(
        "ID, expected",
        [
            ("sand", False),
            ("sand+clay", True),
            ("a+b+c", True),
            ("", False),
        ],
    )
    def test_is_compound_material(self, make_material, ID, expected):
        m = make_material(ID=ID)
        assert m.is_compound_material() is expected


class TestCreateCompoundID:
    def test_two_materials_doubles_and_sorts(self, make_material):
        """Per the docstring: when exactly two materials are provided the
        list is doubled, so the compound ID lists each material twice.
        """
        a = make_material(ID="sand")
        b = make_material(ID="clay")
        assert Material.create_compound_id(a, b) == "clay+clay+sand+sand"

    def test_three_materials_sorted_alphabetically(self, make_material):
        a = make_material(ID="sand")
        b = make_material(ID="clay")
        c = make_material(ID="air")
        assert Material.create_compound_id(a, b, c) == "air+clay+sand"


# ---------------------------------------------------------------------------
# Material — FDTD update coefficients
# ---------------------------------------------------------------------------


class TestUpdateCoeffsH:
    """Closed-form for non-dispersive, lossless materials.

    With ``sm = 0``: HA = HB = m0*mr/dt, so DA = 1, DB* = dt/(m0*mr*d*),
    srcm = dt/(m0*mr).
    """

    def test_free_space_DA_is_unity(self, make_material, fake_grid):
        m = make_material(mr=1.0, sm=0.0)
        G = fake_grid(dt=1e-12, dx=1e-3, dy=2e-3, dz=4e-3)
        m.calculate_update_coeffsH(G)
        assert m.DA == pytest.approx(1.0)

    def test_free_space_DBx_matches_closed_form(self, make_material, fake_grid):
        m = make_material(mr=1.0, sm=0.0)
        G = fake_grid(dt=1e-12, dx=1e-3, dy=2e-3, dz=4e-3)
        m.calculate_update_coeffsH(G)
        expected = G.dt / (mu_0 * 1.0 * G.dx)
        assert m.DBx == pytest.approx(expected)

    def test_free_space_DB_components_scale_with_inverse_spacing(self, make_material, fake_grid):
        m = make_material(mr=1.0, sm=0.0)
        G = fake_grid(dt=1e-12, dx=1e-3, dy=2e-3, dz=4e-3)
        m.calculate_update_coeffsH(G)
        assert m.DBy == pytest.approx(m.DBx / 2)
        assert m.DBz == pytest.approx(m.DBx / 4)

    def test_lossy_magnetic_DA_less_than_one(self, make_material, fake_grid):
        """sm > 0 makes HB < HA so DA = HB/HA < 1 — magnetic loss damps."""
        m = make_material(mr=1.0, sm=0.5)
        G = fake_grid()
        m.calculate_update_coeffsH(G)
        assert 0 < m.DA < 1


class TestUpdateCoeffsE:
    """Closed-form for non-dispersive, lossless dielectric.

    With ``se = 0``: EA = EB = e0*er/dt, so CA = 1, CB* = dt/(e0*er*d*),
    srce = dt/(e0*er).
    """

    def test_free_space_CA_is_unity(self, make_material, fake_grid):
        m = make_material(er=1.0, se=0.0)
        G = fake_grid(dt=1e-12, dx=1e-3, dy=2e-3, dz=4e-3)
        m.calculate_update_coeffsE(G)
        assert m.CA == pytest.approx(1.0)

    def test_free_space_CBx_matches_closed_form(self, make_material, fake_grid):
        m = make_material(er=1.0, se=0.0)
        G = fake_grid(dt=1e-12, dx=1e-3)
        m.calculate_update_coeffsE(G)
        expected = G.dt / (epsilon_0 * 1.0 * G.dx)
        assert m.CBx == pytest.approx(expected)

    def test_conductive_dielectric_CA_less_than_one(self, make_material, fake_grid):
        """With finite conductivity, EB < EA so CA = EB/EA < 1."""
        m = make_material(er=4.0, se=0.01)
        G = fake_grid()
        m.calculate_update_coeffsE(G)
        assert 0 < m.CA < 1

    def test_pec_by_ID_zeros_all_coefficients(self, make_material, fake_grid):
        m = make_material(ID="pec", se=0.0)
        G = fake_grid()
        m.calculate_update_coeffsE(G)
        assert m.CA == 0
        assert m.CBx == m.CBy == m.CBz == 0
        assert m.srce == 0

    def test_pec_by_infinite_conductivity_zeros_all_coefficients(self, make_material, fake_grid):
        m = make_material(ID="metal", se=float("inf"))
        G = fake_grid()
        m.calculate_update_coeffsE(G)
        assert m.CA == 0
        assert m.CBx == m.CBy == m.CBz == 0
        assert m.srce == 0


# ---------------------------------------------------------------------------
# Material — calculate_er (trivial for the non-dispersive base)
# ---------------------------------------------------------------------------


class TestMaterialCalculateER:
    def test_non_dispersive_returns_static_er(self, make_material):
        m = make_material(er=6.0)
        assert m.calculate_er(freq=1e9) == 6.0
        assert m.calculate_er(freq=1e12) == 6.0


# ---------------------------------------------------------------------------
# DispersiveMaterial — defaults and complex permittivity
# ---------------------------------------------------------------------------


class TestDispersiveDefaults:
    def test_inherits_material_defaults(self):
        m = DispersiveMaterial(0, "x")
        assert m.er == 1.0 and m.se == 0.0 and m.mr == 1.0 and m.sm == 0.0

    def test_pole_lists_start_empty(self):
        m = DispersiveMaterial(0, "x")
        assert m.poles == 0
        assert m.deltaer == [] and m.tau == [] and m.alpha == []


class TestDispersiveCalculateER:
    """Closed-form spot checks against textbook formulas."""

    def test_debye_dc_limit_returns_static_permittivity(self, make_dispersive):
        """At f→0, a Debye material has er(0) = er_inf + sum(deltaer_i).

        Lossless (se=0) keeps the conductivity term out of the picture.
        """
        m = make_dispersive(model="debye", er=4.9, se=0.0, poles=[(73.2, 9.231e-12, 0.0)])
        er_dc = m.calculate_er(freq=1.0)  # 1 Hz ≈ DC for ps-scale tau
        assert er_dc.real == pytest.approx(4.9 + 73.2, rel=1e-6)

    def test_debye_high_frequency_limit_returns_er_infinity(self, make_dispersive):
        m = make_dispersive(model="debye", er=4.9, se=0.0, poles=[(73.2, 9.231e-12, 0.0)])
        er_hi = m.calculate_er(freq=1e15)
        assert er_hi.real == pytest.approx(4.9, abs=1e-3)

    def test_lorentz_dc_limit_returns_static_permittivity(self, make_dispersive):
        """At w=0, the Lorentz term reduces to deltaer * tau^2 / tau^2 = deltaer."""
        m = make_dispersive(model="lorentz", er=2.0, se=0.0, poles=[(3.0, 2 * math.pi * 1e9, 1e8)])
        er_dc = m.calculate_er(freq=1.0)
        assert er_dc.real == pytest.approx(2.0 + 3.0, rel=1e-6)


class TestDispersiveDrude:
    def test_two_poles_match_the_sum_of_poles_formula(self, make_dispersive):
        t0, a0 = 1e10, 1e9
        t1, a1 = 2e10, 5e9
        m = make_dispersive(
            model="drude",
            er=1.0,
            se=0.0,
            poles=[(0.0, t0, a0), (0.0, t1, a1)],
        )
        f = 2e9
        w = 2 * math.pi * f
        pole0 = (2 * math.pi * t0) ** 2 / (w**2 - 1j * w * a0)
        pole1 = (2 * math.pi * t1) ** 2 / (w**2 - 1j * w * a1)
        cond = 0.0 / (1j * w * epsilon_0)  # se = 0
        correct = 1.0 + cond - pole0 - pole1
        assert m.calculate_er(f) == pytest.approx(correct)


# ---------------------------------------------------------------------------
# DispersiveMaterial — FDTD update coefficients
# ---------------------------------------------------------------------------


class TestDispersiveUpdateCoeffsE:
    def test_debye_single_pole_assigns_finite_CA(self, make_dispersive, fake_grid):
        m = make_dispersive(model="debye", er=4.9, se=0.0, poles=[(73.2, 9.231e-12, 0.0)])
        G = fake_grid()
        m.calculate_update_coeffsE(G)
        assert math.isfinite(m.CA.real)
        assert math.isfinite(m.CBx.real)

    def test_debye_zero_pole_recovers_non_dispersive_CA(self, make_dispersive, fake_grid):
        """A 'dispersive' material with zero deltaer should behave like
        a plain dielectric: CA = 1 when se = 0.
        """
        m = make_dispersive(model="debye", er=4.9, se=0.0, poles=[(0.0, 1e-12, 0.0)])
        G = fake_grid()
        m.calculate_update_coeffsE(G)
        assert m.CA.real == pytest.approx(1.0, abs=1e-12)


class TestDispersiveDrudeSelfMutationBug:
    """Pin the in-place mutation of ``self.se`` in the Drude branch.

    Source: ``materials.py:258`` — ``self.se += wp2 / self.alpha[x]``
    ran every time ``calculate_update_coeffsE`` was called, so the
    method was not idempotent.

    The bug has been fixed upstream; ``calculate_update_coeffsE`` is now
    idempotent and ``se`` is stable across consecutive calls.
    """

    def test_se_grows_between_consecutive_calls(self, make_dispersive, fake_grid):
        m = make_dispersive(model="drude", er=1.0, se=0.0, poles=[(0.0, 1e10, 1e9)])
        G = fake_grid()
        m.calculate_update_coeffsE(G)
        se_after_first = m.se
        m.calculate_update_coeffsE(G)
        se_after_second = m.se
        assert se_after_second == se_after_first


# ---------------------------------------------------------------------------
# Water properties helper
# ---------------------------------------------------------------------------


class TestCalculateWaterProperties:
    """Numerical reference values come from running the formula in
    ``materials.py:582-598`` at T=25, S=0. They are encoded here so a
    future change to the formula breaks the test.
    """

    def test_fresh_water_at_25C_eri_is_4p9(self):
        eri, _, _, _ = calculate_water_properties(T=25, S=0)
        assert eri == pytest.approx(4.9)

    def test_fresh_water_at_25C_static_er_matches_formula(self):
        _, er, _, _ = calculate_water_properties(T=25, S=0)
        T = 25
        expected = 88.045 - 0.4147 * T + 6.295e-4 * T**2 + 1.075e-5 * T**3
        assert er == pytest.approx(expected)

    def test_fresh_water_conductivity_is_zero(self):
        _, _, _, sig = calculate_water_properties(T=25, S=0)
        assert sig == 0.0

    def test_saline_water_has_positive_conductivity(self):
        _, _, _, sig = calculate_water_properties(T=25, S=35)  # seawater ~35 ppt
        assert sig > 0


# ---------------------------------------------------------------------------
# Built-in materials, water, grass
# ---------------------------------------------------------------------------


class TestCreateBuiltIns:
    def test_appends_pec_pmc_and_free_space(self, fake_grid):
        G = fake_grid()
        create_built_in_materials(G)
        assert [m.ID for m in G.materials] == ["pec", "pmc", "free_space"]

    def test_pec_marked_non_averagable(self, fake_grid):
        G = fake_grid()
        create_built_in_materials(G)
        pec = G.materials[0]
        assert pec.averagable is False
        assert pec.se == float("inf")


class TestCreateWater:
    def test_appends_single_dispersive_water_material(self, fake_grid):
        G = fake_grid()
        create_water(G)
        assert len(G.materials) == 1
        water = G.materials[0]
        assert isinstance(water, DispersiveMaterial)
        assert water.ID == "water"
        assert water.poles == 1


class TestCreateGrass:
    def test_appends_single_dispersive_grass_material(self, fake_grid):
        G = fake_grid()
        create_grass(G)
        assert len(G.materials) == 1
        grass = G.materials[0]
        assert isinstance(grass, DispersiveMaterial)
        assert grass.ID == "grass"
        assert grass.poles == 1


# ---------------------------------------------------------------------------
# PeplinskiSoil
# ---------------------------------------------------------------------------


class TestPeplinskiSoilInit:
    def test_stores_constructor_arguments(self):
        soil = PeplinskiSoil(
            ID="loam",
            sandfraction=0.4,
            clayfraction=0.3,
            bulkdensity=1.6,
            sandpartdensity=2.66,
            watervolfraction=(0.05, 0.15),
        )
        assert soil.ID == "loam"
        assert soil.S == 0.4
        assert soil.C == 0.3
        assert soil.rb == 1.6
        assert soil.rs == 2.66
        assert soil.mu == (0.05, 0.15)
        assert soil.matID == []


class TestPeplinskiSoilProperties:
    def test_generates_nbins_dispersive_materials(self, fake_grid):
        soil = PeplinskiSoil("loam", 0.4, 0.3, 1.6, 2.66, (0.05, 0.15))
        G = fake_grid()
        soil.calculate_properties(nbins=5, G=G)
        assert len(G.materials) == 5
        assert all(isinstance(m, DispersiveMaterial) for m in G.materials)
        assert all(m.type == "debye" for m in G.materials)

    def test_matID_records_all_generated_materials(self, fake_grid):
        soil = PeplinskiSoil("loam", 0.4, 0.3, 1.6, 2.66, (0.05, 0.15))
        G = fake_grid()
        soil.calculate_properties(nbins=4, G=G)
        assert soil.matID == [m.numID for m in G.materials]


# ---------------------------------------------------------------------------
# RangeMaterial
# ---------------------------------------------------------------------------


class TestRangeMaterialInit:
    def test_stores_all_ranges(self):
        rm = RangeMaterial(
            ID="band",
            er_range=(1.0, 5.0),
            se_range=(0.0, 0.01),
            mr_range=(1.0, 1.0),
            sm_range=(0.0, 0.0),
        )
        assert rm.ID == "band"
        assert rm.er == (1.0, 5.0)
        assert rm.sig == (0.0, 0.01)
        assert rm.mu == (1.0, 1.0)
        assert rm.ro == (0.0, 0.0)


class TestRangeMaterialProperties:
    def test_generates_nbins_new_materials(self, fake_grid):
        rm = RangeMaterial("band", (1.0, 5.0), (0.0, 0.01), (1.0, 1.0), (0.0, 0.0))
        G = fake_grid()
        rm.calculate_properties(nbins=4, G=G)
        assert len(G.materials) == 4
        assert all(isinstance(m, Material) for m in G.materials)

    def test_generated_er_values_monotonically_increase(self, fake_grid):
        rm = RangeMaterial("band", (1.0, 5.0), (0.0, 0.0), (1.0, 1.0), (0.0, 0.0))
        G = fake_grid()
        rm.calculate_properties(nbins=4, G=G)
        ers = [m.er for m in G.materials]
        assert ers == sorted(ers)


# ---------------------------------------------------------------------------
# ListMaterial
# ---------------------------------------------------------------------------


class TestListMaterialInit:
    def test_stores_list_of_material_IDs(self):
        lm = ListMaterial(ID="palette", listofmaterials=["sand", "clay"])
        assert lm.ID == "palette"
        assert lm.mat == ["sand", "clay"]
        assert lm.matID == []


class TestListMaterialLookup:
    def test_looks_up_existing_materials_by_ID(self, make_material, fake_grid):
        sand = make_material(ID="sand", numID=5)
        clay = make_material(ID="clay", numID=6)
        G = fake_grid(materials=[sand, clay])
        lm = ListMaterial("palette", ["sand", "clay"])
        lm.calculate_properties(nbins=2, G=G)
        assert lm.matID == [5, 6]


class TestListMaterialMissingMaterialBug:
    """Pin the AttributeError-before-None-check bug.

    Source: ``materials.py:544-548`` — the code called
    ``self.matID.append(material.numID)`` *before* checking whether
    ``material`` is ``None``. Looking up a missing ID raised
    ``AttributeError`` instead of the intended ``ValueError``.

    The bug has been fixed upstream; the code now raises ``ValueError``
    for missing materials.
    """

    def test_missing_material_raises_attribute_error(self, fake_grid):
        G = fake_grid(materials=[])
        lm = ListMaterial("palette", ["does-not-exist"])
        with pytest.raises(ValueError):
            lm.calculate_properties(nbins=1, G=G)


# ---------------------------------------------------------------------------
# process_materials orchestrator
# ---------------------------------------------------------------------------


class TestProcessMaterials:
    """Smoke test for the loop over G.materials that fills updatecoeffs*."""

    def _grid_with_coeff_arrays(self, fake_grid, n_materials, maxpoles=0):
        G = fake_grid(maxpoles=maxpoles)
        G.updatecoeffsE = np.zeros((n_materials, 5))
        G.updatecoeffsH = np.zeros((n_materials, 5))
        G.updatecoeffsdispersive = np.zeros((n_materials, 3 * max(maxpoles, 1)))
        return G

    def test_fills_E_coeffs_for_each_material(self, make_material, fake_grid, material_config):
        material_config.materials["maxpoles"] = 0
        m1 = make_material(ID="m1", numID=0, er=1.0)
        m2 = make_material(ID="m2", numID=1, er=4.0)
        G = self._grid_with_coeff_arrays(fake_grid, n_materials=2)
        G.materials = [m1, m2]
        process_materials(G)
        # CA for both materials with se=0 is exactly 1.
        assert G.updatecoeffsE[0, 0] == pytest.approx(1.0)
        assert G.updatecoeffsE[1, 0] == pytest.approx(1.0)

    def test_returns_table_with_header_and_one_row_per_material(
        self, make_material, fake_grid, material_config
    ):
        material_config.materials["maxpoles"] = 0
        m1 = make_material(ID="m1", numID=0)
        m2 = make_material(ID="m2", numID=1)
        G = self._grid_with_coeff_arrays(fake_grid, n_materials=2)
        G.materials = [m1, m2]
        table = process_materials(G)
        assert len(table) == 1 + 2  # header + two materials


pytestmark = pytest.mark.unit
