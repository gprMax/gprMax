"""Unit tests for ``gprMax.materials.CrimMixture``.

Conventions
-----------
* One behaviour per test; descriptive names following
  ``test_<unit>_<context>_<expected>``.
* Closed-form references where possible.
"""

import numpy as np
import pytest

from gprMax.materials import CrimMixture, calculate_water_properties


@pytest.mark.unit
class TestCrimMixture:
    def _water(self, make_dispersive, T=25, S=0):
        eri, er, tau, sig = calculate_water_properties(T=T, S=S)
        return make_dispersive(
            ID="water", model="debye", er=eri, se=sig, poles=[(er - eri, tau, 0.0)]
        )

    def test_calculate_properties_creates_one_material_per_bin(self, make_material, fake_grid):
        matrix = make_material(ID="sand", er=5.0, se=0.0)
        eri, er, tau, sig = calculate_water_properties(T=25, S=0)
        from gprMax.materials import DispersiveMaterial

        water = DispersiveMaterial(numID=1, ID="water")
        water.type = "debye"
        water.poles = 1
        water.er = eri
        water.se = sig
        water.deltaer.append(er - eri)
        water.tau.append(tau)

        G = fake_grid(materials=[matrix, water])
        crim = CrimMixture(
            ID="wetsand",
            matrix_id="sand",
            matrix_fraction=0.6,
            dispersive_id="water",
            fraction_lower=0.02,
            fraction_upper=0.35,
            f_min=1e6,
            f_max=3e9,
            a=0.5,
        )
        crim.calculate_properties(nbins=8, G=G)

        assert len(crim.matID) == 8
        assert len(set(crim.matID)) == 8  # every bin got a distinct material

    def test_calculate_properties_reuses_dispersive_relaxation_time_unchanged(
        self, make_material, fake_grid
    ):
        """Every bin's fitted pole should keep exactly the dispersive
        phase's own known relaxation time - only weight/e_inf/sigma vary."""
        matrix = make_material(ID="sand", er=5.0, se=0.0)
        from gprMax.materials import DispersiveMaterial

        eri, er, tau, sig = calculate_water_properties(T=25, S=0)
        water = DispersiveMaterial(numID=1, ID="water")
        water.type = "debye"
        water.poles = 1
        water.er = eri
        water.se = sig
        water.deltaer.append(er - eri)
        water.tau.append(tau)

        G = fake_grid(materials=[matrix, water])
        crim = CrimMixture(
            ID="wetsand",
            matrix_id="sand",
            matrix_fraction=0.6,
            dispersive_id="water",
            fraction_lower=0.02,
            fraction_upper=0.35,
            f_min=1e6,
            f_max=3e9,
        )
        crim.calculate_properties(nbins=5, G=G)

        for numid in crim.matID:
            m = next(x for x in G.materials if x.numID == numid)
            assert m.poles == 1
            assert m.tau[0] == pytest.approx(tau)

    def test_calculate_properties_matches_exact_crim_curve_closely(self, make_material, fake_grid):
        """The fitted single-pole material should reproduce the true CRIM
        mixing curve (matrix + water + air) to within a small tolerance
        across the fitted band."""
        matrix = make_material(ID="sand", er=5.0, se=0.0)
        from gprMax.materials import DispersiveMaterial

        eri, er, tau, sig = calculate_water_properties(T=25, S=0)
        water = DispersiveMaterial(numID=1, ID="water")
        water.type = "debye"
        water.poles = 1
        water.er = eri
        water.se = sig
        water.deltaer.append(er - eri)
        water.tau.append(tau)

        G = fake_grid(materials=[matrix, water])
        f_min, f_max, a = 1e6, 3e9, 0.5
        matrix_fraction = 0.6
        water_fraction = 0.185

        crim = CrimMixture(
            ID="wetsand",
            matrix_id="sand",
            matrix_fraction=matrix_fraction,
            dispersive_id="water",
            fraction_lower=water_fraction,
            fraction_upper=water_fraction,
            f_min=f_min,
            f_max=f_max,
            a=a,
        )
        crim.calculate_properties(nbins=1, G=G)

        m = next(x for x in G.materials if x.numID == crim.matID[0])

        freq = np.logspace(np.log10(f_min), np.log10(f_max), 60)
        w = 2 * np.pi * freq
        air_fraction = 1 - matrix_fraction - water_fraction
        eps_water = eri + (er - eri) / (1 + 1j * w * tau)
        exact = (
            matrix_fraction * 5.0**a + water_fraction * eps_water**a + air_fraction * 1.0**a
        ) ** (1 / a)

        fitted = m.er + m.deltaer[0] / (1 + 1j * w * m.tau[0])

        rms_real = np.sqrt(np.mean(((fitted.real - exact.real) / exact.real) ** 2)) * 100
        rms_imag = np.sqrt(np.mean(((fitted.imag - exact.imag) / exact.imag) ** 2)) * 100
        assert rms_real < 1.0
        assert rms_imag < 1.0

    def test_calculate_properties_mixes_conductivity_linearly_by_fraction(
        self, make_material, fake_grid
    ):
        matrix = make_material(ID="brick", er=4.0, se=0.01)
        from gprMax.materials import DispersiveMaterial

        water = DispersiveMaterial(numID=1, ID="brine")
        water.type = "debye"
        water.poles = 1
        water.er = 4.9
        water.se = 2.0
        water.deltaer.append(70.0)
        water.tau.append(9e-12)

        G = fake_grid(materials=[matrix, water])
        crim = CrimMixture(
            ID="wetbrick",
            matrix_id="brick",
            matrix_fraction=0.5,
            dispersive_id="brine",
            fraction_lower=0.1,
            fraction_upper=0.1,
            f_min=1e6,
            f_max=3e9,
        )
        crim.calculate_properties(nbins=1, G=G)

        m = next(x for x in G.materials if x.numID == crim.matID[0])
        expected_sigma = 0.5 * 0.01 + 0.1 * 2.0
        assert m.se == pytest.approx(expected_sigma)

    def test_calculate_properties_rejects_missing_matrix_material(self, fake_grid):
        G = fake_grid(materials=[])
        crim = CrimMixture("c", "nope", 0.6, "alsonope", 0.02, 0.35, 1e6, 3e9)
        with pytest.raises(ValueError):
            crim.calculate_properties(1, G)

    def test_calculate_properties_rejects_dispersive_material_used_as_matrix(
        self, fake_grid, make_dispersive
    ):
        water = self._water(make_dispersive)
        G = fake_grid(materials=[water])
        crim = CrimMixture("c", "water", 0.6, "water", 0.02, 0.35, 1e6, 3e9)
        with pytest.raises(ValueError):
            crim.calculate_properties(1, G)

    def test_calculate_properties_rejects_non_dispersive_material_used_as_dispersive(
        self, make_material, fake_grid
    ):
        matrix = make_material(ID="sand", er=5.0)
        G = fake_grid(materials=[matrix])
        crim = CrimMixture("c", "sand", 0.6, "sand", 0.02, 0.35, 1e6, 3e9)
        with pytest.raises(ValueError):
            crim.calculate_properties(1, G)

    def test_calculate_properties_rejects_multi_pole_dispersive_material(
        self, make_material, fake_grid, make_dispersive
    ):
        matrix = make_material(ID="sand", er=5.0)
        two_pole = make_dispersive(
            ID="water2", model="debye", er=4.9, poles=[(70.0, 9e-12, 0.0), (5.0, 1e-10, 0.0)]
        )
        G = fake_grid(materials=[matrix, two_pole])
        crim = CrimMixture("c", "sand", 0.6, "water2", 0.02, 0.35, 1e6, 3e9)
        with pytest.raises(ValueError):
            crim.calculate_properties(1, G)

    def test_calculate_properties_rejects_fractions_exceeding_one(
        self, make_material, fake_grid, make_dispersive
    ):
        matrix = make_material(ID="sand", er=5.0)
        water = self._water(make_dispersive)
        G = fake_grid(materials=[matrix, water])
        # matrix (0.8) + dispersive (0.35 upper bound) > 1 -> negative air fraction
        crim = CrimMixture("c", "sand", 0.8, "water", 0.1, 0.35, 1e6, 3e9)
        with pytest.raises(ValueError):
            crim.calculate_properties(2, G)

    @pytest.mark.parametrize(
        ("matrix_kwargs", "dispersive_kwargs"),
        [
            ({"se": np.inf}, {}),
            ({"mr": 2.0}, {}),
            ({}, {"mr": 2.0}),
        ],
    )
    def test_calculate_properties_rejects_unsupported_constituents(
        self, make_material, make_dispersive, fake_grid, matrix_kwargs, dispersive_kwargs
    ):
        matrix = make_material(ID="matrix", er=4.0, **matrix_kwargs)
        water = make_dispersive(
            ID="water",
            model="debye",
            er=4.9,
            poles=[(70.0, 9e-12, 0.0)],
        )
        for name, value in dispersive_kwargs.items():
            setattr(water, name, value)
        G = fake_grid(materials=[matrix, water])
        crim = CrimMixture("mix", "matrix", 0.5, "water", 0.1, 0.2, 1e6, 3e9)

        with pytest.raises(ValueError):
            crim.calculate_properties(2, G)

    @pytest.mark.parametrize(
        ("er", "se", "deltaer", "tau"),
        [
            (0.9, 0.0, 70.0, 9e-12),
            (4.9, -0.1, 70.0, 9e-12),
            (4.9, 0.0, -1.0, 9e-12),
            (4.9, 0.0, 70.0, 0.0),
        ],
    )
    def test_calculate_properties_rejects_non_passive_debye_phase(
        self, make_material, make_dispersive, fake_grid, er, se, deltaer, tau
    ):
        matrix = make_material(ID="matrix", er=4.0)
        dispersive = make_dispersive(
            ID="phase", model="debye", er=er, se=se, poles=[(deltaer, tau, 0.0)]
        )
        G = fake_grid(materials=[matrix, dispersive])
        crim = CrimMixture("mix", "matrix", 0.5, "phase", 0.1, 0.2, 1e6, 3e9)

        with pytest.raises(ValueError):
            crim.calculate_properties(2, G)


@pytest.mark.integration
def test_crim_api_builds_a_fractal_volume(monkeypatch, tmp_path):
    """Exercise the public API through material creation and voxelisation."""
    import gprMax
    import gprMax.model as model_module

    captured = {}
    original_build = model_module.Model.build

    def capture_build(self):
        original_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_module.Model, "build", capture_build)

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="CRIM API integration"))
    scene.add(gprMax.Discretisation(p1=(2e-3, 2e-3, 2e-3)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Material(er=5, se=0, mr=1, sm=0, id="sand"))
    scene.add(gprMax.Material(er=4.9, se=0, mr=1, sm=0, id="water"))
    scene.add(
        gprMax.AddDebyeDispersion(
            poles=1,
            er_delta=(73.3389,),
            tau=(8.0994e-12,),
            material_ids=("water",),
        )
    )
    scene.add(
        gprMax.MaterialCrim(
            matrix_id="sand",
            matrix_fraction=0.6,
            dispersive_id="water",
            fraction_lower=0.02,
            fraction_upper=0.35,
            f_min=1e6,
            f_max=3e9,
            a=0.5,
            id="wetsand",
        )
    )
    scene.add(
        gprMax.FractalBox(
            p1=(2e-3, 2e-3, 2e-3),
            p2=(18e-3, 18e-3, 18e-3),
            frac_dim=1.5,
            weighting=(1, 1, 1),
            n_materials=4,
            mixing_model_id="wetsand",
            id="wet_volume",
            seed=1,
        )
    )

    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "crim_api",
        hide_progress_bars=True,
        log_level=30,
    )

    grid = captured["grid"]
    mixture = next(model for model in grid.mixingmodels if model.ID == "wetsand")
    assert len(mixture.matID) == 4
    assert set(np.unique(grid.solid)).issuperset(mixture.matID)


@pytest.mark.integration
def test_crim_hash_command_builds_a_fractal_volume(tmp_path):
    """Exercise the documented positional hash-command path end to end."""
    import gprMax

    inputfile = tmp_path / "crim.in"
    inputfile.write_text(
        "#title: CRIM hash integration\n"
        "#dx_dy_dz: 0.002 0.002 0.002\n"
        "#domain: 0.02 0.02 0.02\n"
        "#pml_cells: 0\n"
        "#time_window: 1e-11\n"
        "#material: 5 0 1 0 sand\n"
        "#material: 4.9 0 1 0 water\n"
        "#add_dispersion_debye: 1 73.3389 8.0994e-12 water\n"
        "#material_crim: sand 0.6 water 0.02 0.35 1e6 3e9 0.5 wetsand\n"
        "#fractal_box: 0.002 0.002 0.002 0.018 0.018 0.018 "
        "1.5 1 1 1 4 wetsand wet_volume 1\n"
    )

    gprMax.run(
        inputfile=inputfile,
        geometry_only=True,
        outputfile=tmp_path / "crim_hash",
        hide_progress_bars=True,
        log_level=30,
    )
