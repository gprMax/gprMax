import numpy as np

from gprMax.materials import PeplinskiSoil, Material
from gprMax.grid import FDTDGrid


def make_dummy_grid():
    # create a minimal grid object with one existing material
    G = FDTDGrid()
    m = Material(len(G.materials), 'air')
    G.materials.append(m)
    return G


def test_peplinski_materials_added():
    """Ensure calculate_debye_properties creates the expected number of bins."""
    G = make_dummy_grid()
    soil = PeplinskiSoil('my_soil', 0.5, 0.5, 2.0, 2.66, (0.001, 0.25))
    soil.calculate_debye_properties(3, G, 'fract')
    assert soil.startmaterialnum == 1  # one material existed before
    assert len(G.materials) == 1 + 3
    ids = [mat.ID for mat in G.materials[1:]]
    assert ids == ['|fract_1|', '|fract_2|', '|fract_3|']
