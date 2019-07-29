# Copyright (C) 2015-2019: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

from .subgrid_solver import SubGridSolver
from .cpu_solvers import CPUSolver
from .subgrid_updaters import SubgridUpdater
from ..config import get_iterations
from ..config import filter
from precursor_nodes import PrecursorNodes
from precursor_nodes import PrecursorNodesFiltered


def create_solver(G, config):
    """Factory method to return a solver instance for a CPU or subgrid simulation"""

    iterations = get_iterations(config, G)

    if filter:
        from precursor_nodes import PrecursorNodesFiltered as PrecursorNodes
    else:
        from precursor_nodes import PrecursorNodes

    updaters = []
    for sg in G.subgrids:
        precursors = PrecursorNodes(G, sg)
        sgu = SubgridUpdater(sg, precursors, G)
        updaters.append(sgu)

    solver = SubGridSolver(G, updaters, iterations)

    return solver
