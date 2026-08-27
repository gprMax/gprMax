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

"""Unit coverage for MPI discrete-plane-wave partition metadata.

End-to-end serial/MPI trace parity is exercised separately in the MPI
integration model; these tests keep the coordinate and coefficient-remapping
contracts fast enough for the normal pytest suite.
"""

from types import SimpleNamespace

import numpy as np

from gprMax.sources import DiscretePlaneWave


def test_tfsf_partition_translates_global_geometry_and_excludes_negative_halo():
    dpw = SimpleNamespace(
        origin=np.asarray((31, 0, 0), dtype=np.int32),
        corners=np.asarray((8, 6, 4, 24, 18, 14), dtype=np.int32),
    )
    grid = SimpleNamespace(
        global_size=np.asarray((30, 24, 20), dtype=np.int32),
        lower_extent=np.asarray((14, 0, 0), dtype=np.int32),
        negative_halo_offset=np.asarray((1, 0, 0), dtype=np.int32),
        size=np.asarray((16, 24, 20), dtype=np.int32),
    )

    DiscretePlaneWave._configure_tfsf_partition(dpw, grid)

    np.testing.assert_array_equal(dpw.tfsf_origin, (17, 0, 0))
    np.testing.assert_array_equal(dpw.tfsf_corners, (-6, 6, 4, 10, 18, 14))
    np.testing.assert_array_equal(dpw.tfsf_owned_lower, (1, 0, 0))
    np.testing.assert_array_equal(dpw.tfsf_owned_upper, (16, 24, 20))


def test_axial_profile_remaps_gathered_coefficients_to_dpw_local_ids():
    n_prop = 4
    coeffs_e = np.arange(5, dtype=np.float32)
    coeffs_h = np.arange(5, dtype=np.float32) + 10
    records = {}
    for component in range(6):
        for prop_idx in range(1, n_prop):
            records[("profile", component, prop_idx)] = (
                coeffs_e + component + prop_idx,
                coeffs_h + component + prop_idx,
                None,
                0,
            )
    source_material = SimpleNamespace(ID="source", poles=0)
    far_material = SimpleNamespace(ID="far", poles=0)
    records[("material", "source")] = source_material
    records[("material", "far_pml")] = far_material

    class _Comm:
        @staticmethod
        def allgather(_local):
            return [records]

    grid = SimpleNamespace(
        updatecoeffsE=np.zeros((1, 5), dtype=np.float32),
        updatecoeffsH=np.zeros((1, 5), dtype=np.float32),
        comm=_Comm(),
    )
    grid.global_to_local_coordinate = lambda point: point
    grid.within_bounds = lambda point: False

    dpw = SimpleNamespace(
        transverse_pos=[1, 1, 1],
        origin_axial=2,
        length=10,
        ID=np.zeros((6, 10), dtype=np.uint32),
    )

    DiscretePlaneWave._build_mpi_axial_profile(dpw, grid, prop=0, n_prop=n_prop)

    assert dpw.material is source_material
    assert dpw.materialPML is far_material
    assert not dpw.dispersive
    assert dpw.max_poles == 0
    for component in range(6):
        first_id = component * n_prop + 1
        last_id = component * n_prop + n_prop - 1
        np.testing.assert_array_equal(dpw.ID[component, :3], first_id)
        np.testing.assert_array_equal(dpw.ID[component, 6:], last_id)
        np.testing.assert_array_equal(dpw.axial_updatecoeffsE[first_id], coeffs_e + component + 1)


def test_axial_profile_zero_pads_mixed_rank_dispersive_orders(monkeypatch):
    """Use the global pole width when rank-local averaged materials differ."""

    n_prop = 4
    coeffs_e = np.arange(5, dtype=np.float64)
    coeffs_h = coeffs_e + 10
    records = {}
    for component in range(6):
        for prop_idx in range(1, n_prop):
            poles = 2 if prop_idx == 1 else 3
            coeffs_d = np.arange(3 * poles, dtype=np.complex128) + 1j * component
            records[("profile", component, prop_idx)] = (
                coeffs_e,
                coeffs_h,
                coeffs_d,
                poles,
            )
    source_material = SimpleNamespace(ID="source", poles=0)
    far_material = SimpleNamespace(ID="far", poles=0)
    records[("material", "source")] = source_material
    records[("material", "far_pml")] = far_material

    class _Comm:
        @staticmethod
        def allgather(_local):
            return [records]

    grid = SimpleNamespace(
        updatecoeffsE=np.zeros((1, 5), dtype=np.float64),
        updatecoeffsH=np.zeros((1, 5), dtype=np.float64),
        updatecoeffsdispersive=np.zeros((1, 6), dtype=np.complex128),
        comm=_Comm(),
    )
    grid.global_to_local_coordinate = lambda point: point
    grid.within_bounds = lambda point: False

    dpw = SimpleNamespace(
        transverse_pos=[1, 1, 1],
        origin_axial=2,
        length=10,
        ID=np.zeros((6, 10), dtype=np.uint32),
    )
    model_config = SimpleNamespace(materials={"dispersivedtype": np.complex128})
    monkeypatch.setattr("gprMax.sources.config.get_model_config", lambda: model_config)

    DiscretePlaneWave._build_mpi_axial_profile(dpw, grid, prop=0, n_prop=n_prop)

    assert dpw.max_poles == 3
    assert dpw.axial_updatecoeffsdispersive.shape == (6 * n_prop, 9)
    for component in range(6):
        two_pole_id = component * n_prop + 1
        np.testing.assert_array_equal(
            dpw.axial_updatecoeffsdispersive[two_pole_id, :6],
            records[("profile", component, 1)][2],
        )
        np.testing.assert_array_equal(dpw.axial_updatecoeffsdispersive[two_pole_id, 6:], 0)
