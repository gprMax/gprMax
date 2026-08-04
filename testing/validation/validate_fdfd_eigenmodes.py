"""Validate FDFD eigenmode effective indices against analytical dispersion.

The validation covers a 1D PEC parallel-plate guide, a symmetric dielectric
slab, and 2D rectangular and circular PEC waveguides. It directly exercises
the production FDFD mode solvers at multiple frequencies and retains a CSV,
comparison plot, JSON summary, and Markdown report.

Run from the repository root::

    python -m testing.validation.validate_fdfd_eigenmodes
"""

import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import matplotlib
import numpy as np
from scipy.constants import c, epsilon_0, mu_0
from scipy.optimize import brentq
from scipy.special import jnp_zeros

import gprMax.config as config
from gprMax.fdfd_eigenmode_solver.fdfd_1d_mode_solver import (
    FDFD_1D_mode_solver,
)
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import (
    FDFD_2D_mode_solver,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

CASE_TITLES = {
    "pec_parallel_plate_1d": "1D PEC parallel-plate guide: TM1",
    "dielectric_slab_1d": "1D dielectric slab: even TE0",
    "pec_rectangular_waveguide_2d": "2D PEC rectangular guide: TE10",
    "pec_cylindrical_waveguide_2d": "2D PEC cylindrical guide: TE11 pair",
}
FREQUENCIES = {
    "pec_parallel_plate_1d": np.asarray((5, 6, 7, 8, 10, 12)) * 1e9,
    "dielectric_slab_1d": np.asarray((2, 3, 4, 5, 6, 8)) * 1e9,
    "pec_rectangular_waveguide_2d": np.asarray((5, 6, 7, 8, 10, 12)) * 1e9,
    "pec_cylindrical_waveguide_2d": np.asarray((6, 7, 8, 10, 12, 14)) * 1e9,
}
RELATIVE_ERROR_LIMITS_PERCENT = {
    "pec_parallel_plate_1d": 0.1,
    "dielectric_slab_1d": 0.1,
    "pec_rectangular_waveguide_2d": 0.1,
    "pec_cylindrical_waveguide_2d": 1.5,
}

PARALLEL_PLATE_WIDTH = 40e-3
PARALLEL_PLATE_SPACING = 0.25e-3
DIELECTRIC_SLAB_WIDTH = 20e-3
DIELECTRIC_DOMAIN_WIDTH = 120e-3
DIELECTRIC_SPACING = 0.25e-3
DIELECTRIC_CORE_INDEX = 3.0
DIELECTRIC_CLADDING_INDEX = 1.0
RECTANGULAR_WIDTH = 40e-3
RECTANGULAR_HEIGHT = 20e-3
RECTANGULAR_SPACING = 1e-3
CIRCULAR_RADIUS = 20e-3
CIRCULAR_SPACING = 1e-3
CIRCULAR_TE11_ROOT = float(jnp_zeros(1, 1)[0])


@contextmanager
def _configured_solver_constants():
    previous = getattr(config, "sim_config", None)
    config.sim_config = SimpleNamespace(
        em_consts={
            "e0": epsilon_0,
            "m0": mu_0,
            "c": c,
            "z0": np.sqrt(mu_0 / epsilon_0),
        }
    )
    try:
        yield
    finally:
        config.sim_config = previous


def _cutoff_neff(frequency, cutoff_frequency):
    frequency = np.asarray(frequency, dtype=np.float64)
    return np.sqrt(1.0 - np.square(cutoff_frequency / frequency))


def _parallel_plate_theory(frequency):
    cutoff = c / (2 * PARALLEL_PLATE_WIDTH)
    return _cutoff_neff(frequency, cutoff)


def _rectangular_theory(frequency):
    cutoff = c / (2 * RECTANGULAR_WIDTH)
    return _cutoff_neff(frequency, cutoff)


def _circular_theory(frequency):
    cutoff = CIRCULAR_TE11_ROOT * c / (2 * np.pi * CIRCULAR_RADIUS)
    return _cutoff_neff(frequency, cutoff)


def _dielectric_slab_theory_scalar(frequency):
    k0 = 2 * np.pi * frequency / c
    half_width = DIELECTRIC_SLAB_WIDTH / 2
    v_number = k0 * half_width * np.sqrt(DIELECTRIC_CORE_INDEX**2 - DIELECTRIC_CLADDING_INDEX**2)
    upper = min(v_number, np.pi / 2) - 1e-12

    def dispersion(u_number):
        w_number = np.sqrt(max(v_number**2 - u_number**2, 0.0))
        return u_number * np.tan(u_number) - w_number

    u_number = brentq(dispersion, 1e-12, upper)
    transverse_wavenumber = u_number / half_width
    return np.sqrt(DIELECTRIC_CORE_INDEX**2 - (transverse_wavenumber / k0) ** 2)


def _dielectric_slab_theory(frequency):
    values = np.atleast_1d(frequency)
    result = np.asarray([_dielectric_slab_theory_scalar(float(value)) for value in values])
    return result if np.ndim(frequency) else float(result[0])


THEORY_FUNCTIONS = {
    "pec_parallel_plate_1d": _parallel_plate_theory,
    "dielectric_slab_1d": _dielectric_slab_theory,
    "pec_rectangular_waveguide_2d": _rectangular_theory,
    "pec_cylindrical_waveguide_2d": _circular_theory,
}


def _solve_parallel_plate(frequency):
    cells = round(PARALLEL_PLATE_WIDTH / PARALLEL_PLATE_SPACING)
    pec_a = np.zeros(cells + 1, dtype=bool)
    pec_a[[0, -1]] = True
    expected = float(_parallel_plate_theory(frequency))
    solver = FDFD_1D_mode_solver(
        frequency=frequency,
        dt=PARALLEL_PLATE_SPACING,
        mode_index=0,
        polarization="TM",
        eps_r_t=np.ones(cells),
        eps_r_a=np.ones(cells + 1),
        eps_r_w=np.ones(cells + 1),
        mu_r_t=np.ones(cells + 1),
        mu_r_a=np.ones(cells),
        mu_r_w=np.ones(cells),
        pec_a_mask=pec_a,
        guess=-(expected**2),
    )
    solver.solve()
    return (float(solver.modal_real_neff),)


def _solve_dielectric_slab(frequency):
    cells = round(DIELECTRIC_DOMAIN_WIDTH / DIELECTRIC_SPACING)
    node_coordinate = np.arange(cells + 1) * DIELECTRIC_SPACING
    cell_coordinate = (np.arange(cells) + 0.5) * DIELECTRIC_SPACING
    lower = (DIELECTRIC_DOMAIN_WIDTH - DIELECTRIC_SLAB_WIDTH) / 2
    upper = lower + DIELECTRIC_SLAB_WIDTH
    eps_nodes = np.where(
        (node_coordinate >= lower) & (node_coordinate <= upper),
        DIELECTRIC_CORE_INDEX**2,
        DIELECTRIC_CLADDING_INDEX**2,
    )
    eps_cells = np.where(
        (cell_coordinate >= lower) & (cell_coordinate <= upper),
        DIELECTRIC_CORE_INDEX**2,
        DIELECTRIC_CLADDING_INDEX**2,
    )
    expected = float(_dielectric_slab_theory(frequency))
    solver = FDFD_1D_mode_solver(
        frequency=frequency,
        dt=DIELECTRIC_SPACING,
        mode_index=0,
        polarization="TM",
        eps_r_t=eps_cells,
        eps_r_a=eps_nodes,
        eps_r_w=eps_nodes,
        mu_r_t=np.ones(cells + 1),
        mu_r_a=np.ones(cells),
        mu_r_w=np.ones(cells),
        guess=-(expected**2),
    )
    solver.solve()
    return (float(solver.modal_real_neff),)


def _homogeneous_2d_arrays(nu, nv):
    return {
        "eps_r_uu": np.ones((nu, nv + 1)),
        "eps_r_vv": np.ones((nu + 1, nv)),
        "eps_r_ww": np.ones((nu + 1, nv + 1)),
        "mu_r_uu": np.ones((nu + 1, nv)),
        "mu_r_vv": np.ones((nu, nv + 1)),
        "mu_r_ww": np.ones((nu, nv)),
    }


def _solve_rectangular_waveguide(frequency):
    nu = round(RECTANGULAR_WIDTH / RECTANGULAR_SPACING)
    nv = round(RECTANGULAR_HEIGHT / RECTANGULAR_SPACING)
    arrays = _homogeneous_2d_arrays(nu, nv)
    pec_u = np.zeros((nu, nv + 1), dtype=bool)
    pec_v = np.zeros((nu + 1, nv), dtype=bool)
    pec_w = np.zeros((nu + 1, nv + 1), dtype=bool)
    pec_u[:, [0, -1]] = True
    pec_v[[0, -1], :] = True
    pec_w[[0, -1], :] = True
    pec_w[:, [0, -1]] = True
    expected = float(_rectangular_theory(frequency))
    solver = FDFD_2D_mode_solver(
        frequency=frequency,
        du=RECTANGULAR_SPACING,
        dv=RECTANGULAR_SPACING,
        mode_index=0,
        pec_u_mask=pec_u,
        pec_v_mask=pec_v,
        pec_w_mask=pec_w,
        guess=-(expected**2),
        **arrays,
    )
    solver.solve()
    return (float(solver.modal_real_neff),)


def _circular_pec_masks(cells):
    domain_width = cells * CIRCULAR_SPACING
    origin = -domain_width / 2
    cell_coordinate = origin + (np.arange(cells) + 0.5) * CIRCULAR_SPACING
    cell_pec = (
        np.square(cell_coordinate[:, np.newaxis]) + np.square(cell_coordinate[np.newaxis, :]) >= CIRCULAR_RADIUS**2
    )
    pec_u = np.zeros((cells, cells + 1), dtype=bool)
    pec_v = np.zeros((cells + 1, cells), dtype=bool)
    pec_w = np.zeros((cells + 1, cells + 1), dtype=bool)
    pec_u[:, :-1] |= cell_pec
    pec_u[:, 1:] |= cell_pec
    pec_v[:-1, :] |= cell_pec
    pec_v[1:, :] |= cell_pec
    pec_w[:-1, :-1] |= cell_pec
    pec_w[1:, :-1] |= cell_pec
    pec_w[:-1, 1:] |= cell_pec
    pec_w[1:, 1:] |= cell_pec
    return pec_u, pec_v, pec_w


def _solve_circular_waveguide(frequency):
    cells = round(2 * CIRCULAR_RADIUS / CIRCULAR_SPACING) + 4
    arrays = _homogeneous_2d_arrays(cells, cells)
    pec_u, pec_v, pec_w = _circular_pec_masks(cells)
    expected = float(_circular_theory(frequency))
    solver = FDFD_2D_mode_solver(
        frequency=frequency,
        du=CIRCULAR_SPACING,
        dv=CIRCULAR_SPACING,
        mode_index=1,
        pec_u_mask=pec_u,
        pec_v_mask=pec_v,
        pec_w_mask=pec_w,
        guess=-(expected**2),
        **arrays,
    )
    solver.solve()
    return tuple(float(value) for value in solver.real_neff)


SOLVER_FUNCTIONS = {
    "pec_parallel_plate_1d": _solve_parallel_plate,
    "dielectric_slab_1d": _solve_dielectric_slab,
    "pec_rectangular_waveguide_2d": _solve_rectangular_waveguide,
    "pec_cylindrical_waveguide_2d": _solve_circular_waveguide,
}


def calculate_results():
    rows = []
    with _configured_solver_constants():
        for case, frequencies in FREQUENCIES.items():
            for frequency in frequencies:
                theoretical = float(THEORY_FUNCTIONS[case](frequency))
                for mode_number, calculated in enumerate(
                    SOLVER_FUNCTIONS[case](float(frequency)),
                    start=1,
                ):
                    absolute_error = abs(calculated - theoretical)
                    rows.append(
                        {
                            "case": case,
                            "mode_number": mode_number,
                            "frequency_hz": float(frequency),
                            "calculated_neff": calculated,
                            "theoretical_neff": theoretical,
                            "absolute_error": absolute_error,
                            "relative_error_percent": 100 * absolute_error / theoretical,
                        }
                    )
    return rows


def _save_csv(path, rows):
    fieldnames = (
        "case",
        "mode_number",
        "frequency_hz",
        "calculated_neff",
        "theoretical_neff",
        "absolute_error",
        "relative_error_percent",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, rows):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True,
    )
    for axis, case in zip(axes.flat, CASE_TITLES):
        frequencies = FREQUENCIES[case]
        dense_frequency = np.linspace(frequencies[0], frequencies[-1], 400)
        axis.plot(
            dense_frequency * 1e-9,
            THEORY_FUNCTIONS[case](dense_frequency),
            color="black",
            linewidth=1.8,
            label="theoretical",
        )
        case_rows = [row for row in rows if row["case"] == case]
        mode_numbers = sorted({row["mode_number"] for row in case_rows})
        for mode_number in mode_numbers:
            mode_rows = [row for row in case_rows if row["mode_number"] == mode_number]
            label = "FDFD" if len(mode_numbers) == 1 else f"FDFD TE11 partner {mode_number}"
            axis.plot(
                [row["frequency_hz"] * 1e-9 for row in mode_rows],
                [row["calculated_neff"] for row in mode_rows],
                marker=("o", "x")[mode_number - 1],
                linestyle="none",
                markersize=5,
                label=label,
            )
        axis.set(
            xlabel="Frequency (GHz)",
            ylabel=r"Effective index, $n_{\mathrm{eff}}$",
            title=CASE_TITLES[case],
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.suptitle("gprMax FDFD eigenmode effective-index validation")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _summarize(rows, runtime_seconds):
    cases = {}
    for case in CASE_TITLES:
        errors = np.asarray([row["relative_error_percent"] for row in rows if row["case"] == case])
        maximum = float(np.max(errors))
        rms = float(np.sqrt(np.mean(np.square(errors))))
        limit = RELATIVE_ERROR_LIMITS_PERCENT[case]
        cases[case] = {
            "title": CASE_TITLES[case],
            "frequency_samples": int(FREQUENCIES[case].size),
            "reported_modes": int(len({row["mode_number"] for row in rows if row["case"] == case})),
            "maximum_relative_error_percent": maximum,
            "rms_relative_error_percent": rms,
            "maximum_allowed_relative_error_percent": limit,
            "passed": maximum <= limit,
        }
    return {
        "runtime_seconds": runtime_seconds,
        "row_count": len(rows),
        "cases": cases,
        "acceptance": {
            "passed": all(case["passed"] for case in cases.values()),
        },
    }


def _write_report(path, summary):
    lines = [
        "# FDFD eigenmode effective-index validation",
        "",
        "Production 1D and 2D FDFD eigenmode results are compared with ",
        "independent analytical dispersion relations at multiple frequencies.",
        "The cylindrical-guide comparison reports both numerically solved members ",
        "of the degenerate TE11 pair.",
        "",
        f"Overall validation status: **{'PASS' if summary['acceptance']['passed'] else 'FAIL'}**.",
        "",
        "| Case | Frequencies | Modes | RMS error | Maximum error | Limit | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for case in CASE_TITLES:
        result = summary["cases"][case]
        lines.append(
            f"| {result['title']} | {result['frequency_samples']} | "
            f"{result['reported_modes']} | "
            f"{result['rms_relative_error_percent']:.4f}% | "
            f"{result['maximum_relative_error_percent']:.4f}% | "
            f"{result['maximum_allowed_relative_error_percent']:.3g}% | "
            f"{'PASS' if result['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "The 1D dielectric reference solves the even symmetric-slab ",
            "transcendental equation. PEC parallel-plate and rectangular-guide ",
            "references use their closed-form cutoff dispersion. The cylindrical ",
            "TE11 reference uses the first zero of the derivative of J1.",
            "",
            "## Outputs",
            "",
            "- [Effective-index comparison](neff_comparison.png)",
            "- `neff_comparison.csv`",
            "- `summary.json`",
            "",
        ]
    )
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )


def run_validation(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    rows = calculate_results()
    runtime_seconds = perf_counter() - start
    _save_csv(output_dir / "neff_comparison.csv", rows)
    _plot(output_dir / "neff_comparison.png", rows)
    summary = _summarize(rows, runtime_seconds)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "report.md", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("fdfd_eigenmode_results"),
    )
    args = parser.parse_args()
    summary = run_validation(args.output_dir)
    print(json.dumps(summary, indent=2))
    if not summary["acceptance"]["passed"]:
        raise SystemExit("FDFD eigenmode analytical validation failed")


if __name__ == "__main__":
    main()
