from types import SimpleNamespace

import numpy as np

from gprMax.materials import Material
from gprMax.sources import EigenmodeSource


def test_unused_builtin_pmc_does_not_block_eigenmode_material_slice():
    """The built-in PMC may exist in the grid without being on the source plane."""
    pec = Material(0, "pec")
    pec.se = float("inf")
    pmc = Material(1, "pmc")
    pmc.sm = float("inf")
    free_space = Material(2, "free_space")

    grid = SimpleNamespace(
        materials=[pec, pmc, free_space],
        ID=np.full((6, 3, 3, 3), free_space.numID, dtype=np.uint32),
    )
    source = EigenmodeSource(grid)
    source.normal_axis = 0
    source.transverse_axes = (1, 2)
    source.transverse_start = np.array((0, 0), dtype=np.int32)
    source.transverse_stop = np.array((2, 2), dtype=np.int32)
    source.plane_index = 1
    source.frequency = 1e9

    tensors = source._extract_local_complex_property_tensors(grid, electric=False)

    assert [tensor.shape for tensor in tensors] == [(3, 2), (2, 3), (2, 2)]
    assert all(np.all(tensor == 1) for tensor in tensors)
