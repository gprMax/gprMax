import csv

from testing.validation.validate_fdfd_eigenmodes import run_validation


def test_fdfd_eigenmode_validation_matches_analytical_dispersion(tmp_path):
    summary = run_validation(tmp_path)

    assert summary["acceptance"]["passed"]
    assert summary["row_count"] == 30
    assert set(summary["cases"]) == {
        "pec_parallel_plate_1d",
        "dielectric_slab_1d",
        "pec_rectangular_waveguide_2d",
        "pec_cylindrical_waveguide_2d",
    }
    assert all(result["passed"] for result in summary["cases"].values())

    with (tmp_path / "neff_comparison.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    circular_rows = [row for row in rows if row["case"] == "pec_cylindrical_waveguide_2d"]
    assert len(circular_rows) == 12
    assert {int(row["mode_number"]) for row in circular_rows} == {1, 2}
    for filename in (
        "neff_comparison.png",
        "summary.json",
        "report.md",
    ):
        assert (tmp_path / filename).stat().st_size > 0
