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
