"""Array allocation, memory estimation and 2D-mode tests for ``FDTDGrid``.

The Yee lattice has one more *node* than *cell* along each axis, which is why
field and ``ID`` arrays are ``(nx+1, ny+1, nz+1)`` while the cell-centred
``solid`` and ``rigid`` arrays are ``(nx, ny, nz)``. Getting one of those
``+1``s wrong does not raise — it corrupts one face of the domain — so the
shapes are asserted explicitly.

Initial values matter too: ``solid`` and ``ID`` start at **1** (free space),
the rigid arrays at **0** (dielectric smoothing permitted).
"""

import numpy as np
import pytest

from .conftest import DL, nonzero_set


class TestInitialiseGeometryArrays:
    def test_solid_is_cell_centred(self, make_grid):
        g = make_grid(nx=4, ny=5, nz=6)
        assert g.solid.shape == (4, 5, 6)

    def test_rigid_e_has_twelve_components(self, make_grid):
        """Twelve edges per Yee cell."""
        g = make_grid(nx=4, ny=5, nz=6)
        assert g.rigidE.shape == (12, 4, 5, 6)

    def test_rigid_h_has_six_components(self, make_grid):
        """Six faces per Yee cell."""
        g = make_grid(nx=4, ny=5, nz=6)
        assert g.rigidH.shape == (6, 4, 5, 6)

    def test_id_is_node_centred_with_six_components(self, make_grid):
        g = make_grid(nx=4, ny=5, nz=6)
        assert g.ID.shape == (6, 5, 6, 7)

    def test_solid_starts_as_free_space(self, make_grid):
        """Material 1 is free space; 0 would be PEC."""
        g = make_grid(nx=4, ny=4, nz=4)
        assert np.all(g.solid == 1)

    def test_id_starts_as_free_space(self, make_grid):
        g = make_grid(nx=4, ny=4, nz=4)
        assert np.all(g.ID == 1)

    def test_rigid_arrays_start_permissive(self, make_grid):
        """Zero means dielectric smoothing is allowed."""
        g = make_grid(nx=4, ny=4, nz=4)
        assert np.all(g.rigidE == 0)
        assert np.all(g.rigidH == 0)

    @pytest.mark.parametrize(
        "name,dtype",
        [
            ("solid", np.uint32),
            ("rigidE", np.int8),
            ("rigidH", np.int8),
            ("ID", np.uint32),
        ],
    )
    def test_dtypes(self, make_grid, name, dtype):
        g = make_grid(nx=4, ny=4, nz=4)
        assert getattr(g, name).dtype == dtype

    def test_reallocates_on_a_second_call(self, make_grid):
        g = make_grid(nx=4, ny=4, nz=4)
        g.solid[0, 0, 0] = 7
        g.initialise_geometry_arrays()
        assert g.solid[0, 0, 0] == 1


class TestInitialiseFieldArrays:
    @pytest.mark.parametrize("name", ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
    def test_all_six_components_are_node_centred(self, make_grid, name):
        g = make_grid(nx=4, ny=5, nz=6)
        assert getattr(g, name).shape == (5, 6, 7)

    @pytest.mark.parametrize("name", ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
    def test_all_six_components_start_at_zero(self, make_grid, name):
        g = make_grid(nx=4, ny=4, nz=4)
        assert np.all(getattr(g, name) == 0)

    @pytest.mark.parametrize("name", ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
    def test_dtype_comes_from_config(self, make_grid, grid_config, name):
        g = make_grid(nx=4, ny=4, nz=4)
        assert (
            getattr(g, name).dtype
            == grid_config.sim_config.dtypes["float_or_double"]
        )

    def test_components_are_distinct_arrays(self, make_grid):
        """Aliasing any two would couple unrelated field components."""
        g = make_grid(nx=4, ny=4, nz=4)
        g.Ex[0, 0, 0] = 1.0
        assert g.Ey[0, 0, 0] == 0.0
        assert g.Ez[0, 0, 0] == 0.0
        assert g.Hx[0, 0, 0] == 0.0


class TestInitialiseUpdateCoeffArrays:
    def test_shape_follows_material_count(self, make_grid, make_material):
        g = make_grid(nx=4, ny=4, nz=4)
        g.materials = [make_material(ID=f"m{i}", numID=i) for i in range(3)]
        g.initialise_std_update_coeff_arrays()
        assert g.updatecoeffsE.shape == (3, 5)
        assert g.updatecoeffsH.shape == (3, 5)

    def test_starts_at_zero(self, make_grid, make_material):
        g = make_grid(nx=4, ny=4, nz=4)
        g.materials = [make_material(ID="m0", numID=0)]
        g.initialise_std_update_coeff_arrays()
        assert np.all(g.updatecoeffsE == 0)
        assert np.all(g.updatecoeffsH == 0)

    def test_empty_material_list_gives_empty_arrays(self, make_grid):
        g = make_grid(nx=4, ny=4, nz=4)
        g.initialise_std_update_coeff_arrays()
        assert g.updatecoeffsE.shape == (0, 5)


class TestInitialiseDispersiveArrays:
    def test_shape_includes_the_pole_count(self, make_grid, grid_config):
        grid_config.model_config.materials["maxpoles"] = 2
        g = make_grid(nx=4, ny=5, nz=6)
        g.initialise_dispersive_arrays()
        assert g.Tx.shape == (2, 5, 6, 7)
        assert g.Ty.shape == (2, 5, 6, 7)
        assert g.Tz.shape == (2, 5, 6, 7)

    def test_dtype_comes_from_model_config(self, make_grid, grid_config):
        grid_config.model_config.materials["maxpoles"] = 1
        g = make_grid(nx=4, ny=4, nz=4)
        g.initialise_dispersive_arrays()
        assert g.Tx.dtype == np.complex128

    def test_update_coeff_array_has_three_entries_per_pole(
        self, make_grid, grid_config, make_material
    ):
        grid_config.model_config.materials["maxpoles"] = 2
        g = make_grid(nx=4, ny=4, nz=4)
        g.materials = [make_material(ID="m0", numID=0), make_material(ID="m1", numID=1)]
        g.initialise_dispersive_update_coeff_array()
        assert g.updatecoeffsdispersive.shape == (2, 6)


class TestResetFields:
    def test_zeroes_the_field_arrays(self, make_grid):
        g = make_grid(nx=4, ny=4, nz=4)
        g.Ex[1, 1, 1] = 5.0
        g.Hz[2, 2, 2] = -3.0
        g.pmls["slabs"] = []
        g.reset_fields()
        assert np.all(g.Ex == 0)
        assert np.all(g.Hz == 0)

    def test_preserves_the_shapes(self, make_grid):
        g = make_grid(nx=4, ny=5, nz=6)
        g.pmls["slabs"] = []
        g.reset_fields()
        assert g.Ex.shape == (5, 6, 7)

    def test_does_not_touch_geometry_arrays(self, make_grid):
        """Only fields are cleared between runs; the built geometry stays."""
        g = make_grid(nx=4, ny=4, nz=4)
        g.solid[1, 1, 1] = 7
        g.pmls["slabs"] = []
        g.reset_fields()
        assert g.solid[1, 1, 1] == 7

    def test_allocates_dispersive_arrays_when_poles_present(
        self, make_grid, grid_config
    ):
        grid_config.model_config.materials["maxpoles"] = 1
        g = make_grid(nx=4, ny=4, nz=4)
        g.pmls["slabs"] = []
        g.reset_fields()
        assert g.Tx.shape == (1, 5, 5, 5)


class TestMemoryEstimates:
    def test_basic_matches_hand_arithmetic(self, make_grid):
        nx = ny = nz = 4
        g = make_grid(nx=nx, ny=ny, nz=nz, pml_thickness=0)

        solid = nx * ny * nz * np.dtype(np.uint32).itemsize
        rigid = (12 + 6) * nx * ny * nz * np.dtype(np.int8).itemsize
        fields = (6 + 6) * (nx + 1) * (ny + 1) * (nz + 1) * np.dtype(np.float64).itemsize

        assert g.mem_est_basic() == solid + rigid + fields

    def test_basic_grows_with_pml_thickness(self, make_grid):
        thin = make_grid(nx=20, ny=20, nz=20, pml_thickness=0)
        thick = make_grid(nx=20, ny=20, nz=20, pml_thickness=10)
        assert thick.mem_est_basic() > thin.mem_est_basic()

    def test_basic_grows_with_domain_size(self, make_grid):
        small = make_grid(nx=4, ny=4, nz=4, pml_thickness=0)
        large = make_grid(nx=8, ny=8, nz=8, pml_thickness=0)
        assert large.mem_est_basic() > small.mem_est_basic()

    def test_dispersive_matches_hand_arithmetic(self, make_grid, grid_config):
        grid_config.model_config.materials["maxpoles"] = 2
        nx = ny = nz = 4
        g = make_grid(nx=nx, ny=ny, nz=nz)
        expected = (
            3 * 2 * (nx + 1) * (ny + 1) * (nz + 1) * np.dtype(np.complex128).itemsize
        )
        assert g.mem_est_dispersive() == expected

    def test_dispersive_is_zero_without_poles(self, make_grid, grid_config):
        grid_config.model_config.materials["maxpoles"] = 0
        g = make_grid(nx=4, ny=4, nz=4)
        assert g.mem_est_dispersive() == 0

    def test_fractals_is_zero_with_no_volumes(self, make_grid):
        g = make_grid(nx=4, ny=4, nz=4)
        assert g.mem_est_fractals() == 0


class TestTwoDimensionalModes:
    """Each 2D TM mode makes one axis invariant by forcing the two in-plane
    electric components to PEC (material 0) on the first two node layers.
    """

    def test_tmx_zeroes_ey_and_ez_on_the_first_two_x_layers(self, make_grid):
        g = make_grid(nx=4, ny=4, nz=4)
        g.tmx()
        assert np.all(g.ID[1, 0:2, :, :] == 0)
        assert np.all(g.ID[2, 0:2, :, :] == 0)

    def test_tmx_leaves_ex_untouched(self, make_grid):
        g = make_grid(nx=4, ny=4, nz=4)
        g.tmx()
        assert np.all(g.ID[0] == 1)

    def test_tmy_zeroes_ex_and_ez_on_the_first_two_y_layers(self, make_grid):
        g = make_grid(nx=4, ny=4, nz=4)
        g.tmy()
        assert np.all(g.ID[0, :, 0:2, :] == 0)
        assert np.all(g.ID[2, :, 0:2, :] == 0)

    def test_tmy_leaves_ey_untouched(self, make_grid):
        g = make_grid(nx=4, ny=4, nz=4)
        g.tmy()
        assert np.all(g.ID[1] == 1)

    def test_tmz_zeroes_ex_and_ey_on_the_first_two_z_layers(self, make_grid):
        g = make_grid(nx=4, ny=4, nz=4)
        g.tmz()
        assert np.all(g.ID[0, :, :, 0:2] == 0)
        assert np.all(g.ID[1, :, :, 0:2] == 0)

    def test_tmz_leaves_ez_untouched(self, make_grid):
        g = make_grid(nx=4, ny=4, nz=4)
        g.tmz()
        assert np.all(g.ID[2] == 1)

    @pytest.mark.parametrize("mode", ["tmx", "tmy", "tmz"])
    def test_magnetic_components_are_never_touched(self, make_grid, mode):
        """Only the electric components (0-2) are forced to PEC."""
        g = make_grid(nx=4, ny=4, nz=4)
        getattr(g, mode)()
        assert np.all(g.ID[3:] == 1)

    def test_tmx_changes_exactly_the_documented_cells(self, make_grid):
        """Pin the full footprint, not just a sample of it."""
        g = make_grid(nx=3, ny=3, nz=3)
        before = g.ID.copy()
        g.tmx()
        changed = nonzero_set(before != g.ID)
        expected = {
            (comp, i, j, k)
            for comp in (1, 2)
            for i in (0, 1)
            for j in range(g.ny + 1)
            for k in range(g.nz + 1)
        }
        assert changed == expected

    def test_modes_are_idempotent(self, make_grid):
        g = make_grid(nx=4, ny=4, nz=4)
        g.tmz()
        once = g.ID.copy()
        g.tmz()
        assert np.array_equal(g.ID, once)
