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

"""Unit tests for the plane-wave TFSF background check."""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.materials import DispersiveMaterial, Material
from gprMax.utilities.tfsf_background import validate_plane_wave_tfsf_background

FREE_SPACE = 0
VACUUM = 1
BLOCK = 2
ELECTRIC_ONLY = 3
MAGNETIC_ONLY = 4
PEC = 5
PMC = 6

CORNERS_2D = [1, 1, 0, 7, 7, 1]
CORNERS_3D = [1, 1, 1, 7, 7, 7]
OWNED_LOWER_2D = [0, 0, 0]
OWNED_UPPER_2D = [10, 10, 2]
OWNED_LOWER_3D = [0, 0, 0]
OWNED_UPPER_3D = [10, 10, 10]
PROJECTIONS = np.ones(6)

# The z-face tests use Hx as the active incident component and switch Hz off,
# so the x faces cannot mask a missing or extra z-face node.
PROJECTIONS_WITHOUT_HZ = np.array([1, 1, 1, 1, 1, 0], dtype=float)


def _materials():
    free_space = Material(FREE_SPACE, "free_space")
    vacuum = Material(VACUUM, "vacuum")
    block = Material(BLOCK, "block")
    block.er = 4.0
    electric_only = Material(ELECTRIC_ONLY, "electric_only")
    electric_only.mr = 4.0
    magnetic_only = Material(MAGNETIC_ONLY, "magnetic_only")
    magnetic_only.er = 4.0
    pec = Material(PEC, "pec")
    pec.se = float("inf")
    pmc = Material(PMC, "pmc")
    pmc.sm = float("inf")
    return [free_space, vacuum, block, electric_only, magnetic_only, pec, pmc]


def _grid():
    return SimpleNamespace(
        ID=np.zeros((6, 10, 10, 2), dtype=np.uint32),
        materials=_materials(),
        rigidE=np.zeros((12, 9, 9, 2), dtype=np.int8),
        rigidH=np.zeros((6, 9, 9, 2), dtype=np.int8),
    )


def _grid_3d():
    return SimpleNamespace(
        ID=np.zeros((6, 9, 9, 9), dtype=np.uint32),
        materials=_materials(),
        rigidE=np.zeros((12, 8, 8, 8), dtype=np.int8),
        rigidH=np.zeros((6, 8, 8, 8), dtype=np.int8),
    )


def _validate(
    grid,
    background=FREE_SPACE,
    skip_axis=2,
    projections=PROJECTIONS,
    owned_lower=OWNED_LOWER_2D,
    owned_upper=OWNED_UPPER_2D,
    comm=None,
):
    validate_plane_wave_tfsf_background(
        grid,
        CORNERS_2D,
        owned_lower,
        owned_upper,
        grid.materials[background],
        projections,
        skip_axis,
        comm=comm,
    )


def _validate_3d(grid, background=FREE_SPACE, projections=PROJECTIONS, comm=None):
    validate_plane_wave_tfsf_background(
        grid,
        CORNERS_3D,
        OWNED_LOWER_3D,
        OWNED_UPPER_3D,
        grid.materials[background],
        projections,
        -1,
        comm=comm,
    )


def _debye(grid, name, deltaer, tau):
    material = DispersiveMaterial(len(grid.materials), name)
    material.type = "debye"
    material.poles = len(deltaer)
    material.deltaer = list(deltaer)
    material.tau = list(tau)
    grid.materials.append(material)
    return material


def test_a_homogeneous_boundary_is_accepted():
    _validate(_grid())


def test_a_differently_named_material_with_the_same_properties_is_accepted():
    grid = _grid()
    grid.ID[:, 1:8, 1:8, 0] = VACUUM

    _validate(grid)


def test_a_different_material_on_the_upper_x_face_is_rejected():
    grid = _grid()
    grid.ID[:, 7, 1:8, 0] = BLOCK

    with pytest.raises(ValueError, match="block"):
        _validate(grid)


def test_a_material_fully_inside_the_box_is_accepted():
    grid = _grid()
    grid.ID[:, 3:5, 3:5, 0] = BLOCK

    _validate(grid)


def test_non_averaging_magnetic_components_on_a_homogeneous_boundary_are_accepted():
    grid = _grid()
    grid.materials[FREE_SPACE].averagable = False
    grid.rigidH[2, 7, 3, 0] = 1  # Hy at the upper x face

    _validate(grid)


def test_non_averaging_electric_components_on_a_homogeneous_boundary_are_accepted():
    grid = _grid()
    grid.materials[FREE_SPACE].averagable = False
    grid.rigidE[8, 7, 3, 0] = 1  # Ez at the upper x face

    _validate(grid)


def test_y_face_hz_correction_uses_the_ex_projection():
    grid = _grid()
    grid.ID[5, 3, 0, 0] = ELECTRIC_ONLY  # Hz one cell outside the lower y face
    projections = np.zeros(6)
    projections[5] = 1.0  # Hz is live, but its y-face correction uses Ex.

    _validate(grid, projections=projections)

    projections[0] = 1.0
    with pytest.raises(ValueError, match="electric_only"):
        _validate(grid, projections=projections)


def test_a_different_material_at_the_lower_face_h_node_is_rejected():
    grid = _grid()
    grid.ID[4, 0, 3, 0] = ELECTRIC_ONLY  # Hy one cell outside the lower x face

    with pytest.raises(ValueError, match="electric_only"):
        _validate(grid)


def test_a_node_outside_the_correction_range_is_accepted():
    grid = _grid()
    grid.ID[2, 0, 3, 0] = BLOCK  # Ez one cell outside the lower x face

    _validate(grid)


def test_a_material_on_a_box_edge_is_rejected():
    grid = _grid()
    grid.ID[2, 1, 7, 0] = BLOCK  # Ez on the shared x/y box edge

    with pytest.raises(ValueError, match="block"):
        _validate(grid)


@pytest.mark.parametrize("k", [1, 7])
def test_z_face_ey_correction_includes_the_x1_edge(k):
    grid = _grid_3d()
    grid.ID[1, 7, 3, k] = BLOCK  # Ey at the x1 edge of a z face

    with pytest.raises(ValueError, match="block"):
        _validate_3d(grid, projections=PROJECTIONS_WITHOUT_HZ)


@pytest.mark.parametrize("k", [1, 7])
def test_z_face_ey_correction_excludes_the_y1_edge(k):
    grid = _grid_3d()
    grid.ID[1, 3, 7, k] = BLOCK  # Ey one cell beyond the y1 edge

    _validate_3d(grid, projections=PROJECTIONS_WITHOUT_HZ)


@pytest.mark.parametrize("k", [1, 7])
def test_z_face_ex_correction_includes_the_y1_edge(k):
    grid = _grid_3d()
    grid.ID[0, 3, 7, k] = BLOCK  # Ex at the y1 edge of a z face

    with pytest.raises(ValueError, match="block"):
        _validate_3d(grid, projections=PROJECTIONS_WITHOUT_HZ)


@pytest.mark.parametrize("k", [1, 7])
def test_z_face_ex_correction_excludes_the_x1_edge(k):
    grid = _grid_3d()
    grid.ID[0, 7, 3, k] = BLOCK  # Ex one cell beyond the x1 edge

    _validate_3d(grid, projections=PROJECTIONS_WITHOUT_HZ)


def test_permeability_at_an_electric_correction_node_is_ignored():
    grid = _grid()
    grid.ID[2, 7, 3, 0] = ELECTRIC_ONLY  # Ez on the upper x face

    _validate(grid)


def test_permeability_at_a_magnetic_correction_node_is_rejected():
    grid = _grid()
    grid.ID[4, 7, 3, 0] = ELECTRIC_ONLY  # Hy on the upper x face

    with pytest.raises(ValueError, match="electric_only"):
        _validate(grid)


def test_permittivity_at_a_magnetic_correction_node_is_ignored():
    grid = _grid()
    grid.ID[4, 7, 3, 0] = MAGNETIC_ONLY  # Hy on the upper x face

    _validate(grid)


def test_permittivity_at_an_electric_correction_node_is_rejected():
    grid = _grid()
    grid.ID[2, 7, 3, 0] = MAGNETIC_ONLY  # Ez on the upper x face

    with pytest.raises(ValueError, match="magnetic_only"):
        _validate(grid)


def test_pec_electric_properties_on_the_boundary_are_rejected():
    grid = _grid()
    grid.ID[2, 7, 3, 0] = PEC  # Ez on the upper x face

    with pytest.raises(ValueError, match="pec"):
        _validate(grid)


def test_pmc_magnetic_properties_on_the_boundary_are_rejected():
    grid = _grid()
    grid.ID[4, 7, 3, 0] = PMC  # Hy on the upper x face

    with pytest.raises(ValueError, match="pmc"):
        _validate(grid)


def test_dispersive_pole_order_does_not_change_the_electric_signature():
    grid = _grid()
    first = _debye(grid, "debye_a", [1.0, 3.0], [2e-11, 4e-11])
    second = _debye(grid, "debye_b", [3.0, 1.0], [4e-11, 2e-11])
    grid.ID[:] = first.numID
    grid.ID[2, 7, 3, 0] = second.numID

    _validate(grid, background=first.numID)


def test_different_dispersive_poles_are_rejected_at_an_electric_node():
    grid = _grid()
    first = _debye(grid, "debye_a", [1.0, 3.0], [2e-11, 4e-11])
    second = _debye(grid, "debye_c", [2.0, 3.0], [2e-11, 4e-11])
    grid.ID[:] = first.numID
    grid.ID[2, 7, 3, 0] = second.numID

    with pytest.raises(ValueError, match="debye_c"):
        _validate(grid, background=first.numID)


def test_zero_weight_dispersive_poles_are_ignored():
    grid = _grid()
    first = _debye(grid, "debye_a", [1.0, 3.0], [2e-11, 4e-11])
    second = _debye(grid, "debye_zero", [0.0, 1.0, 3.0], [9e-11, 2e-11, 4e-11])
    grid.ID[:] = first.numID
    grid.ID[2, 7, 3, 0] = second.numID

    _validate(grid, background=first.numID)


def test_electric_dispersion_is_not_read_at_a_magnetic_node():
    grid = _grid()
    first = _debye(grid, "debye_a", [1.0, 3.0], [2e-11, 4e-11])
    second = _debye(grid, "debye_c", [2.0, 3.0], [2e-11, 4e-11])
    grid.ID[:] = first.numID
    grid.ID[4, 7, 3, 0] = second.numID

    _validate(grid, background=first.numID)


def test_directional_materials_warn_instead_of_being_rejected():
    grid = _grid()
    directional = Material(len(grid.materials), "directional")
    directional.directional_materials = (grid.materials[FREE_SPACE],) * 3
    grid.materials.append(directional)
    grid.ID[2, 7, 3, 0] = directional.numID

    with pytest.warns(UserWarning, match="cannot be compared reliably"):
        _validate(grid)


def test_unknown_dispersive_models_warn_instead_of_being_rejected():
    grid = _grid()
    unknown = DispersiveMaterial(len(grid.materials), "unknown")
    unknown.type = "weird"
    unknown.poles = 1
    unknown.deltaer = [1.0]
    unknown.tau = [1.0]
    unknown.alpha = [1.0]
    grid.materials.append(unknown)
    grid.ID[2, 7, 3, 0] = unknown.numID

    with pytest.warns(UserWarning, match="cannot be compared reliably"):
        _validate(grid)


def test_mixed_dispersive_models_warn_instead_of_being_rejected():
    grid = _grid()
    mixed = DispersiveMaterial(len(grid.materials), "mixed")
    mixed.type = "debye, lorentz"
    mixed.poles = 2
    mixed.deltaer = [1.0, 2.0]
    mixed.tau = [2e-11, 1e9]
    mixed.alpha = [0.0, 1e7]
    grid.materials.append(mixed)
    grid.ID[2, 7, 3, 0] = mixed.numID

    with pytest.warns(UserWarning, match="cannot be compared reliably"):
        _validate(grid)


def test_a_node_outside_the_owned_transverse_range_is_ignored():
    grid = _grid()
    grid.ID[4, 7, 3, 0] = ELECTRIC_ONLY  # Hy on the upper x face
    _validate(grid, owned_lower=[0, 4, 0], owned_upper=[10, 10, 2])


def test_a_node_outside_the_owned_normal_range_is_ignored():
    grid = _grid()
    grid.ID[4, 7, 3, 0] = ELECTRIC_ONLY  # Hy on the upper x face
    _validate(grid, owned_upper=[7, 10, 2])


class _Collective:
    def __init__(self, gathered):
        self.gathered = gathered

    def allgather(self, value):
        return self.gathered


def test_mpi_collective_combines_rank_errors():
    grid = _grid()
    comm = _Collective([(None, None), ("Hy at (7, 3, 0) uses material 'block'", None)])

    with pytest.raises(ValueError, match="homogeneous"):
        _validate(grid, comm=comm)


def test_mpi_collective_combines_rank_warnings():
    grid = _grid()
    comm = _Collective(
        [
            (None, None),
            (
                None,
                "Hy at (7, 3, 0) uses material 'directional', whose properties "
                "cannot be compared reliably",
            ),
        ]
    )

    with pytest.warns(UserWarning, match="cannot be compared reliably"):
        _validate(grid, comm=comm)
