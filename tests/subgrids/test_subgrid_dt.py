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

"""Legacy CFL regression tests for HSG subgrids.

``SubGridBaseGrid`` inherits ``FDTDGrid.calculate_dt()``, so the narrow
compatibility safeguard in that method covers both main and fine grids. These
tests pin the historical ``Decimal(float)``/``ROUND_FLOOR`` values through the
real ``calculate_dt()`` method — including the production
``SubGridHSG`` user-object build path — rather than reproducing the CFL
formula in test code.

Expected literals were derived from the pre-PR flooring and verified by
``repr()``/``hex()`` equality. They are deliberately not computed through
``round_float()`` or ``round_value()``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.subgrids.user_objects import SubGridHSG


class TestSubgridDtLegacyExact:
    def test_one_to_one_matches_main_exact(self, make_main_grid, make_subgrid):
        """A 1:1 subgrid refines nothing, so its step equals the main step."""
        main = make_main_grid()
        assert main.dt == 1.92583320154647e-12

        sub = make_subgrid(ratio=1)
        sub.calculate_dt()

        assert sub.dt == 1.92583320154647e-12
        assert sub.dt == main.dt

    def test_refined_ratio3_exact(self, make_main_grid, make_subgrid):
        """A 3:1 refinement of a 1 mm main grid.

        Asserts the exact historical fine-grid step; a bare
        ``fine < main`` check would not catch a one-ulp shift.
        """
        main = make_main_grid()
        sub = make_subgrid()
        sub.calculate_dt()

        assert sub.dt == 6.4194440051549e-13
        assert sub.dt < main.dt

    def test_7p5mm_three_to_one_via_production_build(self, make_material):
        """Full ``#subgrid_hsg`` build path for a 7.5 mm main grid, 3:1.

        The user object sets the fine discretisation (``dl / ratio``) and
        then calls the inherited ``calculate_dt()``. The raw fine-grid CFL
        float is ``4.814583003866176e-12``; legacy flooring gives
        ``4.814583003866175e-12``, which the ``str()``-based helper would
        otherwise shift by one ulp.
        """
        main = FDTDGrid()
        main.size = np.array([30, 30, 30], dtype=np.int64)
        main.dl = np.array([0.0075, 0.0075, 0.0075], dtype=np.float64)
        main.materials = [
            make_material(ID="pec", numID=0),
            make_material(ID="free_space", numID=1),
        ]
        main.calculate_dt()
        assert main.dt == 1.444374901159853e-11
        main.iterations = 100

        model = SimpleNamespace(G=main, subgrids=[], dt_mod=1.0, iterations=100)
        cmd = SubGridHSG(
            p1=(0.075, 0.075, 0.075),
            p2=(0.12, 0.12, 0.12),
            ratio=3,
            id="fine_7p5mm",
            is_os_sep=1,
            subgrid_pml_thickness=2,
            interpolation=1,
            filter=True,
        )
        fine = cmd.build(model)

        assert fine.dt == 4.814583003866175e-12


pytestmark = pytest.mark.unit
