"""Unit coverage for rank-local NTFF surface ownership."""

import numpy as np

from gprMax.ntff.mpi import global_patch_indices, localise_component_surface
from gprMax.ntff.surfaces import build_component_surface


class _TwoRankGrid:
    """Minimal 20-cell x-slab decomposition including one negative halo."""

    def __init__(self, rank):
        self.rank = rank
        if rank == 0:
            self.lower_extent = np.asarray((0, 0, 0), dtype=np.int32)
            self.negative_halo_offset = np.asarray((0, 0, 0), dtype=np.int32)
            self.size = np.asarray((10, 20, 20), dtype=np.int32)
        else:
            self.lower_extent = np.asarray((9, 0, 0), dtype=np.int32)
            self.negative_halo_offset = np.asarray((1, 0, 0), dtype=np.int32)
            self.size = np.asarray((11, 20, 20), dtype=np.int32)

    @staticmethod
    def get_rank_from_coordinate(coordinate):
        return 0 if int(coordinate[0]) < 10 else 1


def test_component_surface_has_one_owner_and_local_halo_indices():
    global_surface = build_component_surface(
        "Ez",
        (7, 7, 7),
        (13, 13, 13),
        (0.004, 0.004, 0.004),
        (21, 21, 21),
        real_dtype=np.float64,
    )
    local = [localise_component_surface(global_surface, _TwoRankGrid(rank)) for rank in (0, 1)]

    indices = np.concatenate([global_patch_indices(surface) for surface in local])
    np.testing.assert_array_equal(np.sort(indices), np.arange(global_surface.npatches))
    assert np.unique(indices).size == global_surface.npatches

    global_faces = {face.face_id: face for face in global_surface.faces}
    face_offsets = {}
    offset = 0
    for face in global_surface.faces:
        face_offsets[face.face_id] = offset
        offset += face.npatches

    for rank, surface in enumerate(local):
        grid = _TwoRankGrid(rank)
        shape = np.asarray(surface.field_shape)
        for face in surface.faces:
            assert np.all(face.inside_indices >= 0)
            assert np.all(face.outside_indices >= 0)
            assert np.all(face.inside_indices < shape)
            assert np.all(face.outside_indices < shape)
            within_face = face.global_patch_indices - face_offsets[face.face_id]
            expected = global_faces[face.face_id]
            np.testing.assert_array_equal(
                face.inside_indices + grid.lower_extent,
                expected.inside_indices[within_face],
            )
            np.testing.assert_array_equal(
                face.outside_indices + grid.lower_extent,
                expected.outside_indices[within_face],
            )


def test_rank_with_no_surface_patches_retains_empty_canonical_faces():
    surface = build_component_surface(
        "Hy",
        (7, 7, 7),
        (13, 13, 13),
        (0.004, 0.004, 0.004),
        (21, 21, 21),
        real_dtype=np.float64,
    )
    grid = _TwoRankGrid(1)
    grid.get_rank_from_coordinate = lambda coordinate: 0
    local = localise_component_surface(surface, grid)

    assert tuple(face.face_id for face in local.faces) == tuple(
        face.face_id for face in surface.faces
    )
    assert local.npatches == 0
    assert global_patch_indices(local).size == 0
