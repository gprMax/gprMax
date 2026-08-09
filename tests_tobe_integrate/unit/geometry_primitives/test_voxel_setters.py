"""Unit tests for the atomic voxel/edge/face setters in
``gprMax/cython/geometry_primitives.pyx``.

``build_edge_*``, ``build_face_*`` and ``build_voxel`` are the writes
every shape rasteriser bottoms out in. Each test calls a setter for a
single cell of a tiny grid and pins the exact set of array slots that
change — the Yee-cell bookkeeping (which neighbouring cells share an
edge, which ``ID`` components get stamped) is the whole contract.

The ``set_rigid_*`` / ``unset_rigid_*`` helpers in
``yee_cell_setget_rigid.pyx`` are ``cdef`` (C-only) and unreachable
from Python; they are covered transitively here, since each setter is
a thin wrapper over them.
"""

import numpy as np
import pytest

from gprMax.cython.geometry_primitives import (
    build_edge_x,
    build_edge_y,
    build_edge_z,
    build_face_xy,
    build_face_xz,
    build_face_yz,
    build_voxel,
)

from .conftest import nonzero_set

NUM_ID = 1
NUM_IDX = 2
NUM_IDY = 3
NUM_IDZ = 4


class TestBuildEdges:
    """Each edge setter flips the rigid-E slots of the edge shared between
    the cell and its (up to three) neighbours, and stamps one ID entry."""

    def test_edge_x_interior_cell(self, grid_arrays):
        g = grid_arrays(4, 4, 4)
        build_edge_x(2, 2, 2, NUM_IDX, g.rigidE, g.rigidH, g.ID)

        # An x-edge at (i, j, k) is shared with the -y / -z / -yz neighbours.
        assert nonzero_set(g.rigidE) == {
            (0, 2, 2, 2),
            (1, 2, 1, 2),
            (3, 2, 2, 1),
            (2, 2, 1, 1),
        }
        assert nonzero_set(g.ID) == {(0, 2, 2, 2)}
        assert g.ID[0, 2, 2, 2] == NUM_IDX
        assert not g.rigidH.any()
        assert not g.solid.any()

    def test_edge_y_interior_cell(self, grid_arrays):
        g = grid_arrays(4, 4, 4)
        build_edge_y(2, 2, 2, NUM_IDY, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.rigidE) == {
            (4, 2, 2, 2),
            (7, 1, 2, 2),
            (5, 2, 2, 1),
            (6, 1, 2, 1),
        }
        assert nonzero_set(g.ID) == {(1, 2, 2, 2)}
        assert g.ID[1, 2, 2, 2] == NUM_IDY
        assert not g.rigidH.any()

    def test_edge_z_interior_cell(self, grid_arrays):
        g = grid_arrays(4, 4, 4)
        build_edge_z(2, 2, 2, NUM_IDZ, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.rigidE) == {
            (8, 2, 2, 2),
            (9, 1, 2, 2),
            (11, 2, 1, 2),
            (10, 1, 1, 2),
        }
        assert nonzero_set(g.ID) == {(2, 2, 2, 2)}
        assert g.ID[2, 2, 2, 2] == NUM_IDZ
        assert not g.rigidH.any()

    @pytest.mark.parametrize(
        "builder, base_slot, id_component",
        [
            (build_edge_x, 0, 0),
            (build_edge_y, 4, 1),
            (build_edge_z, 8, 2),
        ],
    )
    def test_edge_at_origin_skips_neighbour_writes(self, grid_arrays, builder, base_slot, id_component):
        # At (0, 0, 0) the neighbour cells do not exist; only the base
        # rigid slot of the origin cell may be flipped.
        g = grid_arrays(4, 4, 4)
        builder(0, 0, 0, NUM_IDX, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.rigidE) == {(base_slot, 0, 0, 0)}
        assert nonzero_set(g.ID) == {(id_component, 0, 0, 0)}


class TestBuildFaces:
    """Each face setter rigidifies the four E-edges bounding the face —
    two in the home cell, two in the +1 neighbours along the face plane —
    and stamps the four matching ID entries."""

    def test_face_yz_interior_cell(self, grid_arrays):
        g = grid_arrays(4, 4, 4)
        build_face_yz(2, 2, 2, NUM_IDY, NUM_IDZ, g.rigidE, g.rigidH, g.ID)

        # set_rigid_Ey(2,2,2) + set_rigid_Ez(2,2,2)
        # + set_rigid_Ey(2,2,3) + set_rigid_Ez(2,3,2), each fanning out
        # to its edge-sharing neighbours.
        assert nonzero_set(g.rigidE) == {
            (4, 2, 2, 2), (7, 1, 2, 2), (5, 2, 2, 1), (6, 1, 2, 1),
            (8, 2, 2, 2), (9, 1, 2, 2), (11, 2, 1, 2), (10, 1, 1, 2),
            (4, 2, 2, 3), (7, 1, 2, 3), (5, 2, 2, 2), (6, 1, 2, 2),
            (8, 2, 3, 2), (9, 1, 3, 2), (11, 2, 2, 2), (10, 1, 2, 2),
        }
        assert nonzero_set(g.ID) == {
            (1, 2, 2, 2),
            (1, 2, 2, 3),
            (2, 2, 2, 2),
            (2, 2, 3, 2),
        }
        assert g.ID[1, 2, 2, 2] == NUM_IDY
        assert g.ID[1, 2, 2, 3] == NUM_IDY
        assert g.ID[2, 2, 2, 2] == NUM_IDZ
        assert g.ID[2, 2, 3, 2] == NUM_IDZ
        assert not g.rigidH.any()
        assert not g.solid.any()

    def test_face_xz_interior_cell(self, grid_arrays):
        g = grid_arrays(4, 4, 4)
        build_face_xz(2, 2, 2, NUM_IDX, NUM_IDZ, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.rigidE) == {
            (0, 2, 2, 2), (1, 2, 1, 2), (3, 2, 2, 1), (2, 2, 1, 1),
            (8, 2, 2, 2), (9, 1, 2, 2), (11, 2, 1, 2), (10, 1, 1, 2),
            (0, 2, 2, 3), (1, 2, 1, 3), (3, 2, 2, 2), (2, 2, 1, 2),
            (8, 3, 2, 2), (9, 2, 2, 2), (11, 3, 1, 2), (10, 2, 1, 2),
        }
        assert nonzero_set(g.ID) == {
            (0, 2, 2, 2),
            (0, 2, 2, 3),
            (2, 2, 2, 2),
            (2, 3, 2, 2),
        }
        assert g.ID[0, 2, 2, 2] == NUM_IDX
        assert g.ID[0, 2, 2, 3] == NUM_IDX
        assert g.ID[2, 2, 2, 2] == NUM_IDZ
        assert g.ID[2, 3, 2, 2] == NUM_IDZ
        assert not g.rigidH.any()

    def test_face_xy_interior_cell(self, grid_arrays):
        g = grid_arrays(4, 4, 4)
        build_face_xy(2, 2, 2, NUM_IDX, NUM_IDY, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.rigidE) == {
            (0, 2, 2, 2), (1, 2, 1, 2), (3, 2, 2, 1), (2, 2, 1, 1),
            (4, 2, 2, 2), (7, 1, 2, 2), (5, 2, 2, 1), (6, 1, 2, 1),
            (0, 2, 3, 2), (1, 2, 2, 2), (3, 2, 3, 1), (2, 2, 2, 1),
            (4, 3, 2, 2), (7, 2, 2, 2), (5, 3, 2, 1), (6, 2, 2, 1),
        }
        assert nonzero_set(g.ID) == {
            (0, 2, 2, 2),
            (0, 2, 3, 2),
            (1, 2, 2, 2),
            (1, 3, 2, 2),
        }
        assert g.ID[0, 2, 2, 2] == NUM_IDX
        assert g.ID[0, 2, 3, 2] == NUM_IDX
        assert g.ID[1, 2, 2, 2] == NUM_IDY
        assert g.ID[1, 3, 2, 2] == NUM_IDY
        assert not g.rigidH.any()

    def test_face_yz_at_origin_skips_neighbour_writes(self, grid_arrays):
        g = grid_arrays(4, 4, 4)
        build_face_yz(0, 0, 0, NUM_IDY, NUM_IDZ, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.rigidE) == {
            (4, 0, 0, 0),
            (8, 0, 0, 0),
            (4, 0, 0, 1),
            (5, 0, 0, 0),
            (8, 0, 1, 0),
            (11, 0, 0, 0),
        }


class TestBuildVoxelAveraging:
    """``averaging=True`` — the smoothed-material path: write ``solid``
    and *clear* the cell's rigid flags so it may average with
    neighbours. ``ID`` is left for the later averaging pass."""

    def test_writes_solid_and_clears_rigid_column(self, grid_arrays):
        g = grid_arrays(4, 4, 4)
        g.rigidE[:] = 1
        g.rigidH[:] = 1

        build_voxel(1, 2, 3, NUM_ID, NUM_IDX, NUM_IDY, NUM_IDZ, True,
                    g.solid, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.solid) == {(1, 2, 3)}
        assert g.solid[1, 2, 3] == NUM_ID
        # Exactly the 12 + 6 rigid slots of this one cell are cleared.
        assert not g.rigidE[:, 1, 2, 3].any()
        assert not g.rigidH[:, 1, 2, 3].any()
        assert np.count_nonzero(g.rigidE == 0) == 12
        assert np.count_nonzero(g.rigidH == 0) == 6
        assert not g.ID.any()


class TestBuildVoxelHard:
    """``averaging=False`` — the hard-boundary path: write ``solid``, set
    every rigid flag of the cell, and stamp all 24 ID entries (six field
    components at the four corners each component touches)."""

    # The 24 ID writes of a hard voxel at (i, j, k). Components 0-2 are
    # Ex/Ey/Ez, components 3-5 are Hx/Hy/Hz and reuse the same corners.
    @staticmethod
    def expected_id_writes(i, j, k):
        corners = {
            0: [(i, j, k), (i, j + 1, k + 1), (i, j + 1, k), (i, j, k + 1)],
            1: [(i, j, k), (i + 1, j, k + 1), (i + 1, j, k), (i, j, k + 1)],
            2: [(i, j, k), (i + 1, j + 1, k), (i + 1, j, k), (i, j + 1, k)],
        }
        values = {0: NUM_IDX, 1: NUM_IDY, 2: NUM_IDZ}
        expected = {}
        for comp in range(6):
            for corner in corners[comp % 3]:
                expected[(comp, *corner)] = values[comp % 3]
        return expected

    def test_stamps_solid_rigid_and_all_24_id_entries(self, grid_arrays):
        g = grid_arrays(4, 4, 4)
        build_voxel(1, 1, 1, NUM_ID, NUM_IDX, NUM_IDY, NUM_IDZ, False,
                    g.solid, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.solid) == {(1, 1, 1)}
        assert g.solid[1, 1, 1] == NUM_ID
        assert np.all(g.rigidE[:, 1, 1, 1] == 1)
        assert np.all(g.rigidH[:, 1, 1, 1] == 1)
        assert np.count_nonzero(g.rigidE) == 12
        assert np.count_nonzero(g.rigidH) == 6

        expected = self.expected_id_writes(1, 1, 1)
        assert nonzero_set(g.ID) == set(expected)
        for slot, value in expected.items():
            assert g.ID[slot] == value

    def test_far_corner_writes_into_id_padding(self, grid_arrays):
        # The +1 padding on every ID dimension exists exactly so the last
        # voxel of the domain can stamp its far corners without going out
        # of bounds.
        g = grid_arrays(4, 4, 4)
        build_voxel(3, 3, 3, NUM_ID, NUM_IDX, NUM_IDY, NUM_IDZ, False,
                    g.solid, g.rigidE, g.rigidH, g.ID)

        assert g.solid[3, 3, 3] == NUM_ID
        assert nonzero_set(g.ID) == set(self.expected_id_writes(3, 3, 3))
        assert g.ID[0, 3, 4, 4] == NUM_IDX
        assert g.ID[2, 4, 4, 3] == NUM_IDZ

    def test_overwrite_flips_averaged_cell_to_hard(self, grid_arrays):
        # Later objects overwrite earlier ones in the same cell: an
        # averaged write followed by a hard write leaves the cell rigid
        # with the new material.
        g = grid_arrays(4, 4, 4)
        build_voxel(2, 2, 2, 7, 7, 7, 7, True,
                    g.solid, g.rigidE, g.rigidH, g.ID)
        build_voxel(2, 2, 2, NUM_ID, NUM_IDX, NUM_IDY, NUM_IDZ, False,
                    g.solid, g.rigidE, g.rigidH, g.ID)

        assert g.solid[2, 2, 2] == NUM_ID
        assert np.all(g.rigidE[:, 2, 2, 2] == 1)
        assert np.all(g.rigidH[:, 2, 2, 2] == 1)
