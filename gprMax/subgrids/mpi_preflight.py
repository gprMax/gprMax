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

import logging

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.model import Model
from gprMax.subgrids.grid import SubGridBaseGrid

from ..updates.cpu_updates import CPUUpdates
from .precursor_nodes import PrecursorNodes, PrecursorNodesEqualResolution, PrecursorNodesFiltered
from .subgrid_hsg import SubGridHSG

logger = logging.getLogger(__name__)

from typing import List, Sequence, Tuple

SUBGRIDS_ON_MPI_SOLVER_SUPPORTED = False
AXES = "xyz"


class MPISubgridConfigurationError(ValueError):
    """MPI domain decomposition is incompatible with a subgrid."""


def partition_boundaries(n_cells: int, n_ranks: int) -> List[int]:
    """Mirrors MPIGrid.calculate_local_extents: first cell owned by each rank after the first."""
    size, overflow = divmod(int(n_cells), int(n_ranks))
    return [size * r + min(r, overflow) for r in range(1, int(n_ranks))]


def coupling_footprint(i0, i1, is_os_sep, interpolation, dispersive_averaging):
    pad = int(is_os_sep) + 1 + int(interpolation) + (1 if dispersive_averaging else 0)
    return tuple(int(v) - pad for v in i0), tuple(int(v) + pad for v in i1)


def find_straddled_boundaries(lo, hi, global_size, mpi_tasks) -> List[Tuple[int, int]]:
    hits = []
    for axis in range(3):
        for b in partition_boundaries(global_size[axis], mpi_tasks[axis]):
            if lo[axis] < b <= hi[axis]:
                hits.append((axis, b))
    return hits


def validate_subgrid_against_mpi(
    name, i0, i1, is_os_sep, interpolation, dispersive_averaging, global_size, mpi_tasks
) -> None:
    lo, hi = coupling_footprint(
        i0, i1, is_os_sep, interpolation, dispersive_averaging
    )
    hits = find_straddled_boundaries(lo, hi, global_size, mpi_tasks)
    decomposition = (
        f"{tuple(int(t) for t in mpi_tasks)} MPI ranks over "
        f"{tuple(int(n) for n in global_size)} cells"
    )

    if hits:
        crossings = "\n".join(
            f"  - {AXES[a]}: coupling region cells {lo[a]}..{hi[a]} "
            f"contain the rank boundary at {AXES[a]} = {b}"
            for a, b in hits
        )

        cause = ""
        if dispersive_averaging:
            lo_without_dispersive, hi_without_dispersive = coupling_footprint(
                i0, i1, is_os_sep, interpolation, False
            )
            hits_without_dispersive = find_straddled_boundaries(
                lo_without_dispersive, hi_without_dispersive,
                global_size, mpi_tasks
            )
            if not hits_without_dispersive:
                cause = (
                    " dispersive averaging expands the coupling region "
                    "across the MPI boundary."
                )

        raise MPISubgridConfigurationError(
            f"Subgrid '{name}' straddles an MPI subdomain boundary "
            f"({decomposition}).\n"
            f"{crossings}\n"
            f"{cause}"
            + (
                " Dispersive averaging is enabled."
                if dispersive_averaging and not cause
                else ""
            )
            + " Subgrids are not currently supported with MPI domain "
            "decomposition. Omit --mpi to run this subgrid."
)

    if not SUBGRIDS_ON_MPI_SOLVER_SUPPORTED:
        raise MPISubgridConfigurationError(
            f"Subgrid '{name}': the complete coupling region lies "
            "within a single rank, but subgrids are not currently "
            "supported with MPI domain decomposition. "
            "Omit --mpi to run this subgrid."
        )
