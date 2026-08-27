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
