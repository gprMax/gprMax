"""DebyeFit input contracts, optimizer controls, and export boundary regressions."""

import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import epsilon_0

import gprMax
import gprMax.model as model_module
from gprMax.hash_cmds_file import get_user_objects
from gprMax.toolboxes.DebyeFit.Debye_Fit import Crim, HavriliakNegami, Jonscher, Rawdata
from gprMax.toolboxes.DebyeFit.optimization import DA_DLS, DE_DLS, DLS, PSO_DLS


def hn(**kwargs):
    options = dict(
        f_min=1e6,
        f_max=1e9,
        alpha=1,
        beta=1,
        e_inf=3,
        de=5,
        tau_0=1e-9,
        sigma=0,
        mu=1,
        mu_sigma=0,
        material_name="audit",
        number_of_debye_poles=1,
        f_n=24,
        plot=False,
        save=False,
        optimizer_options={"swarmsize": 4, "maxiter": 2, "seed": 1},
    )
    return HavriliakNegami(**(options | kwargs))


@pytest.mark.parametrize(
    "bad",
    [
        {"mu": 0},
        {"mu": np.nan},
        {"sigma": np.nan},
        {"sigma": np.inf},
        {"sigma": -1},
        {"mu_sigma": -1},
        {"tau_0": np.nan},
        {"tau_0": 0},
        {"tau_0": (1e-9,)},
        {"number_of_debye_poles": 0},
        {"number_of_debye_poles": -2},
        {"number_of_debye_poles": 1.5},
        {"number_of_debye_poles": True},
        {"f_n": 0},
        {"f_n": 1},
        {"f_n": 1.5},
        {"f_n": True},
        {"f_min": 0},
        {"f_min": -1},
        {"f_min": np.nan},
        {"f_max": np.inf},
        {"f_max": 1e6},
        {"e_inf": 0.5},
        {"de": -1},
        {"alpha": 0},
        {"alpha": 1.1},
        {"beta": -1},
        {"beta": np.nan},
        {"material_name": "bad name"},
        {"material_name": "bad\nname"},
        {"material_name": ""},
        {"material_name": "a+b"},
        {"material_name": "pec"},
        {"material_name": "pmc"},
        {"material_name": "free_space"},
    ],
)
@pytest.mark.unit
def test_invalid_inputs_raise_value_error(bad):
    with pytest.raises(ValueError):
        hn(**bad).check_inputs()


@pytest.mark.unit
def test_run_revalidates_current_parameters_and_frequency_grid():
    model = hn()
    model.alpha = -1
    with pytest.raises(ValueError, match="alpha"):
        model.run()
    model.alpha, model.de, model.f_n, model.f_max = 1, 7, 12, 1e10
    model.check_inputs()
    assert model.freq.shape == (12,)
    assert model.freq[-1] == 1e10
    assert model.params["Delta_eps"] == 7


@pytest.mark.parametrize(
    "bad",
    [
        {"a": 0},
        {"materials": [[3, 5], [1, 0]]},
        {"materials": [[3, 5, 0], [1, 0, 1]]},
        {"materials": [[3, 5, 1], [1, np.nan, 1]]},
        {"volumetric_fractions": [[0.5], [0.5]]},
        {"volumetric_fractions": [np.nan, 0.5]},
    ],
)
@pytest.mark.unit
def test_crim_validates_shapes_and_physical_parameters(bad):
    options = dict(
        f_min=1e6,
        f_max=1e9,
        a=0.5,
        volumetric_fractions=[0.5, 0.5],
        materials=[[3, 5, 1e-9], [1, 0, 1]],
        sigma=0,
        mu=1,
        mu_sigma=0,
        material_name="mix",
    )
    with pytest.raises(ValueError):
        Crim(**(options | bad)).check_inputs()


@pytest.mark.parametrize("bad", [{"n_p": 0}, {"n_p": 1}, {"omega_p": 0}, {"a_p": -1}])
@pytest.mark.unit
def test_jonscher_rejects_singular_parameters(bad):
    options = dict(
        f_min=1e6,
        f_max=1e9,
        e_inf=3,
        a_p=1,
        omega_p=1e9,
        n_p=0.7,
        sigma=0,
        mu=1,
        mu_sigma=0,
        material_name="jonscher",
    )
    with pytest.raises(ValueError):
        Jonscher(**(options | bad)).check_inputs()


@pytest.mark.parametrize(
    "data",
    [
        [[1e6, 3, 1]],
        [[1e6, 3], [1e9, 2]],
        [[1e6, 3, 1, 0], [1e9, 2, 1, 0]],
        [[1e6, 3, 1], [1e6, 2, 1]],
        [[0, 3, 1], [1e9, 2, 1]],
        [[1e6, 3, -1], [1e9, 2, 1]],
        [[1e6, np.nan, 1], [1e9, 2, 1]],
        [[1e6, 0.5, 1], [1e9, 2, 1]],
    ],
)
@pytest.mark.unit
def test_rawdata_rejects_invalid_columns_values_and_duplicates(tmp_path, data):
    path = tmp_path / "data.csv"
    np.savetxt(path, data, delimiter=",")
    with pytest.raises(ValueError):
        Rawdata(path, 0, 1, 0, "raw").run()


@pytest.mark.unit
def test_rawdata_sorts_data_and_preserves_loss_sign(tmp_path):
    path = tmp_path / "data.csv"
    np.savetxt(path, [[1e9, 2, 0.1], [1e6, 3, 1]], delimiter=",")
    model = Rawdata(path, 0, 1, 0, "raw", f_n=10)
    q = model.calculation()
    np.testing.assert_allclose(q[[0, -1]], [3 - 1j, 2 - 0.1j])
    assert np.all(np.diff(model.freq) > 0)
    model.sigma = 0.1
    with pytest.warns(UserWarning, match="sigma adds extra loss"):
        np.testing.assert_array_equal(model.calculation(), q)


@pytest.mark.parametrize("delimiter", [", ", ";", None])
@pytest.mark.unit
def test_rawdata_preserves_separator_options(tmp_path, delimiter):
    path = tmp_path / "data.txt"
    np.savetxt(path, [[1e6, 3, 1], [1e9, 2, 0.1]], delimiter=delimiter or " ")
    model = Rawdata(path, 0, 1, 0, "raw", delimiter=delimiter, f_n=2)
    np.testing.assert_allclose(model.calculation(), [3 - 1j, 2 - 0.1j])


@pytest.mark.unit
def test_missing_file_and_explicit_missing_save_directory(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        Rawdata(tmp_path / "missing.csv", 0, 1, 0, "raw").run()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "materials").mkdir()
    with pytest.raises(FileNotFoundError):
        hn().save_result(["#material: 3 0 1 0 m\n"], fdir=tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        hn().save_result(["#material: 3 0 1 0 m\n"], fdir="../materials")
    assert not (tmp_path / "materials/my_materials.txt").exists()
    hn().save_result(["#material: 3 0 1 0 m\n"])
    assert (tmp_path / "materials/my_materials.txt").is_file()


@pytest.mark.unit
def test_save_material_only_and_dispersive_results(tmp_path):
    model = hn()
    model.save_result(model.print_output([], [], 3), fdir=tmp_path)
    model.material_name = "dispersive"
    model.save_result(model.print_output([-9], [5], 3), fdir=tmp_path)
    lines = (tmp_path / "my_materials.txt").read_text().splitlines()
    # get_user_objects consumes preprocessed commands; the file loader strips comments.
    objects = get_user_objects(
        [line for line in lines if line and not line.startswith("##")], checkessential=False
    )
    assert len(objects) == 3  # two materials and one dispersion command


def _named_objective(x, center, scale):
    return scale * np.sum((x - center) ** 2)


@pytest.mark.parametrize(
    "cls, options",
    [
        (PSO_DLS, {"swarmsize": 8, "maxiter": 2}),
        (DA_DLS, {"maxiter": 2, "no_local_search": True}),
        (DE_DLS, {"maxiter": 2, "popsize": 5, "polish": False}),
    ],
)
@pytest.mark.unit
def test_optimizer_keyword_order_seed_and_global_rng(cls, options):
    np.random.seed(123)
    before = np.random.get_state()
    results = []
    for params in ({"center": 0.25, "scale": 3}, {"scale": 3, "center": 0.25}):
        results.append(
            cls(seed=11, **options).calc_relaxation_times(_named_objective, [-1], [1], params)
        )
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]
    assert results[0][1] == pytest.approx(_named_objective(results[0][0], 0.25, 3))
    after = np.random.get_state()
    for a, b in zip(before, after):
        np.testing.assert_equal(a, b)


@pytest.mark.unit
def test_de_honours_iteration_limit():
    calls = []
    opt = DE_DLS(
        maxiter=1,
        popsize=5,
        polish=False,
        tol=0,
        seed=1,
        callback=lambda x, convergence: calls.append(x.copy()),
    )
    opt.calc_relaxation_times(_named_objective, [-1], [1], {"center": 0.123, "scale": 2})
    assert len(calls) == 1


@pytest.mark.unit
def test_de_parallel_matches_serial_deferred_evaluation():
    model = hn()
    q = model.calculation()
    args = {"freq": model.freq, "im": q.imag, "rl": q.real}
    results = []
    for workers in (1, 2):
        opt = DE_DLS(
            maxiter=2, popsize=5, polish=False, workers=workers, updating="deferred", seed=11
        )
        results.append(opt.fit(opt.cost_function, [-12.0], [-6.0], args))
    for a, b in zip(*results):
        np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_de_supports_spawn_workers():
    model = hn()
    q = model.calculation()
    args = {"freq": model.freq, "im": q.imag, "rl": q.real}
    options = dict(maxiter=1, popsize=5, polish=False, updating="deferred", seed=11)
    serial = DE_DLS(**options)
    expected = serial.fit(serial.cost_function, [-12.0], [-6.0], args)
    # Spawn exercises serialization instead of relying on inherited fork state,
    # including the bound cost function and named array arguments.
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        parallel = DE_DLS(workers=pool.map, **options)
        actual = parallel.fit(parallel.cost_function, [-12.0], [-6.0], args)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("cls", [PSO_DLS, DA_DLS, DE_DLS])
@pytest.mark.parametrize(
    "bounds", [([], []), ([0], [0]), ([0], [np.inf]), ([0, 1], [2]), ([0j], [1])]
)
@pytest.mark.unit
def test_optimizers_reject_invalid_bounds(cls, bounds):
    with pytest.raises(ValueError, match="Bounds"):
        cls().calc_relaxation_times(_named_objective, *bounds, {"center": 0, "scale": 1})


@pytest.mark.parametrize("cls", [PSO_DLS, DA_DLS, DE_DLS])
@pytest.mark.parametrize("value", [-1, 1.5, True])
@pytest.mark.unit
def test_optimizers_reject_invalid_iteration_counts(cls, value):
    with pytest.raises(ValueError, match="maxiter"):
        cls(maxiter=value)


@pytest.mark.unit
def test_loss_error_plot_uses_finite_consistent_normalisation(monkeypatch):
    model = hn(f_n=5)
    model.rl = np.full(5, 4.0)
    model.im = -np.array([0.0, 0.5, 1.0, 2.0, 1e6])
    monkeypatch.setattr(plt, "show", lambda: None)
    model.plot_result(model.rl, model.im + 0.01)
    fig = plt.gcf()
    expected = -0.01 / (1 + abs(model.im))
    np.testing.assert_allclose(fig.axes[1].lines[1].get_ydata(), expected)
    assert model.error(model.rl, model.im + 0.01)[1] == pytest.approx(np.mean(abs(expected)) * 100)
    plt.close(fig)


@pytest.mark.unit
def test_plot_does_not_hide_fit_overshoot(monkeypatch):
    model = hn()
    q = model.calculation()
    model.rl, model.im = q.real, q.imag
    monkeypatch.setattr(plt, "show", lambda: None)
    model.plot_result(np.full(model.f_n, 100), model.im)
    assert plt.gcf().axes[0].get_ylim()[1] > 100
    plt.close(plt.gcf())


@pytest.mark.parametrize("poles", [1, 3])
@pytest.mark.unit
def test_dls_exact_analytic_debye_spectra(poles):
    f = np.logspace(6, 11, 101)
    logt = np.array([-9.0]) if poles == 1 else np.array([-11.0, -9.0, -7.0])
    weights = np.arange(1, poles + 1, dtype=float)
    target = 3 + np.sum(weights / (1 + 2j * np.pi * f[:, None] * 10**logt), axis=1)
    ci, cr, actual, ee, real, imag = DLS(logt, target.real, target.imag, f)
    np.testing.assert_allclose(actual, weights, rtol=1e-12)
    np.testing.assert_allclose(ee + real + 1j * imag, target, rtol=1e-12)
    assert ci + cr < 1e-12


@pytest.mark.parametrize(
    "tau, weights, ee",
    [
        ([-9], [-1], 3),
        ([np.nan], [1], 3),
        ([-9], [np.inf], 3),
        ([-9], [1], 0.5),
        ([-9], [1, 2], 3),
        ([400], [1], 3),
        ([-400], [1], 3),
        ([-9 + 1j], [1], 3),
        ([-9], [1j], 3),
    ],
)
@pytest.mark.unit
def test_export_rejects_invalid_fit(tau, weights, ee):
    with pytest.raises(ValueError):
        hn().print_output(tau, weights, ee)


@pytest.mark.unit
def test_export_retains_every_positive_strength():
    model = hn()
    properties = model.print_output([-9, -8, -7], [0, 1e-20, 5], 3)
    assert model.fitted_pole_count == 2
    dispersion = get_user_objects(properties, checkessential=False)[1]
    assert dispersion.kwargs["er_delta"] == [1e-20, 5]


@pytest.mark.parametrize("lossless, sigma", [(False, 0), (False, 0.01), (True, 0), (True, 0.01)])
def test_export_roundtrip_into_solver(tmp_path, monkeypatch, lossless, sigma):
    captured = {}
    original = model_module.Model.build

    def capture(model):
        original(model)
        captured["grid"] = model.G

    monkeypatch.setattr(model_module.Model, "build", capture)
    model = hn(de=0 if lossless else 5, sigma=sigma)
    if lossless:
        error, properties = model.run()
        assert error == 0 and model.fitted_pole_count == 0
        # A second run must remain valid; fitted count must not corrupt the request.
        assert model.run()[0] == 0
    else:
        properties = model.print_output([-9.0], [5.0], 3.0)
    scene = gprMax.Scene()
    for obj in [
        gprMax.Domain(p1=(0.012,) * 3),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.TimeWindow(iterations=4),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(n=1),
        *get_user_objects(properties, checkessential=False),
        gprMax.Box(p1=(0.003,) * 3, p2=(0.009,) * 3, material_id="audit"),
    ]:
        scene.add(obj)
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "roundtrip",
        log_level=50,
        hide_progress_bars=True,
        cpu_precision="double",
    )
    material = next(m for m in captured["grid"].materials if m.ID == "audit")
    for freq in (1e6, 1e8, 1e9):
        expected = 3 + (0 if lossless else 5 / (1 + 2j * np.pi * freq * 1e-9))
        # Non-dispersive Material.calculate_er reports er only; account for its
        # separately stored physical conductivity in this comparison.
        actual = material.calculate_er(freq)
        if lossless:
            actual += material.se / (2j * np.pi * freq * epsilon_0)
        expected += sigma / (2j * np.pi * freq * epsilon_0)
        assert actual == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_auto_search_warns_when_accuracy_target_is_not_met(monkeypatch):
    model = hn(number_of_debye_poles=-1)

    def poor_fit():
        n = model.number_of_debye_poles
        return np.full(n, -9.0), np.ones(n), 1.0, np.zeros(model.f_n), np.zeros(model.f_n)

    monkeypatch.setattr(model, "optimize", poor_fit)
    with pytest.warns(UserWarning, match="reached 20 poles"):
        error, _ = model.run()
    assert error > 5 and model.number_of_debye_poles == 20


def test_exported_fit_fields_match_analytic_debye_material(tmp_path):
    import h5py

    model = hn(sigma=0.01)
    q = model.calculation()
    _, _, weights, ee, _, _ = DLS(np.array([-9.0]), q.real, q.imag, model.freq)
    fitted = get_user_objects(model.print_output([-9.0], weights, ee), checkessential=False)
    reference = [
        gprMax.Material(er=3, se=0.01, mr=1, sm=0, id="audit"),
        gprMax.AddDebyeDispersion(poles=1, er_delta=[5], tau=[1e-9], material_ids=["audit"]),
    ]
    traces = []
    for label, material in (("fitted", fitted), ("reference", reference)):
        scene = gprMax.Scene()
        for obj in [
            gprMax.Domain(p1=(0.020,) * 3),
            gprMax.Discretisation(p1=(0.001,) * 3),
            gprMax.TimeWindow(iterations=300),
            gprMax.PMLThickness(thickness=3),
            gprMax.OMPThreads(n=1),
            *material,
            gprMax.Box(p1=(0.008, 0.003, 0.003), p2=(0.012, 0.017, 0.017), material_id="audit"),
            gprMax.Waveform(wave_type="gaussian", amp=1, freq=5e9, id="pulse"),
            gprMax.HertzianDipole(p1=(0.006, 0.010, 0.010), polarisation="z", waveform_id="pulse"),
            gprMax.Rx(p1=(0.014, 0.010, 0.010), id="probe", outputs=["Ez"]),
        ]:
            scene.add(obj)
        gprMax.run(
            scenes=[scene],
            outputfile=tmp_path / label,
            log_level=50,
            hide_progress_bars=True,
            cpu_precision="double",
        )
        with h5py.File(tmp_path / f"{label}.h5") as output:
            traces.append(output["rxs/rx1/Ez"][:])
    assert np.linalg.norm(traces[1]) > 0
    assert np.isfinite(traces[0]).all()
    np.testing.assert_allclose(traces[0], traces[1], rtol=1e-11, atol=1e-12 * max(abs(traces[1])))
