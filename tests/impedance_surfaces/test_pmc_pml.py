"""PMC/PML geometry guards and independent reflected-domain comparisons."""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.impedance_pml import _ImpedancePMLHoldMaterial, prepare_impedance_pml
from gprMax.materials import Material


def sparse_fixture(stretch=2, sign="plus"):
    normal = (stretch + 1) % 3
    component = (stretch + 2) % 3
    host, marker, hold = (
        Material(index, name) for index, name in enumerate(("host", "marker", "hold"))
    )
    host.er = 2.0
    solid = np.zeros((7, 7, 7), dtype=np.uint32)
    selected = [slice(None)] * 3
    selected[normal] = slice(0, 3)
    solid[tuple(selected)] = 1
    lower, upper = [1] * 3, [6] * 3
    lower[stretch] = 2
    pml = SimpleNamespace(
        direction="xyz"[stretch] + sign,
        xs=lower[0],
        xf=upper[0],
        ys=lower[1],
        yf=upper[1],
        zs=lower[2],
        zf=upper[2],
    )
    grid = SimpleNamespace(
        solid=solid,
        ID=np.full((6, 8, 8, 8), 2, dtype=np.uint32),
        materials=[host, marker, hold],
        impedance_marker_models={1: "wall"},
        dt=1e-12,
        dx=0.001,
        dy=0.0013,
        dz=0.0017,
        pmls={"slabs": [pml]},
    )
    coordinate = [3] * 3
    coordinate[stretch] = 4
    system = SimpleNamespace(
        edge_count=1,
        edge_info=np.asarray([[component, *coordinate, 0, 0, 0, 1]]),
        port_info=np.asarray([[0, 0]]),
        model_Z0=np.asarray([np.inf]),
        model_info=np.asarray([[0, 0]]),
        port_normal=np.asarray([[normal, 1]]),
        model_ids=("wall",),
        edge_fraction=np.asarray([0.5]),
        edge_runtime=np.zeros((1, 2)),
    )
    return grid, system, stretch, normal, component, coordinate


@pytest.mark.parametrize("stretch", range(3))
@pytest.mark.parametrize("sign", ("minus", "plus"))
def test_extruded_pmc_preserves_pml_increment_without_bulk_update(stretch, sign):
    grid, system, _, _, component, coordinate = sparse_fixture(stretch, sign)
    prepare_impedance_pml(grid, system)
    material = grid.materials[grid.ID[(component, *coordinate)]]
    assert isinstance(material, _ImpedancePMLHoldMaterial)
    material.calculate_update_coeffsE(grid)
    material.calculate_update_coeffsH(grid)
    assert material.CA == 1
    assert material.CBx == material.CBy == material.CBz == 0
    assert material.srce == 1
    assert material.DA == material.srcm == 0
    assert system.pml_edge_count == 1
    assert system.pml_edge_indices.tolist() == [0]
    spacing = (grid.dx, grid.dy, grid.dz)
    assert (
        system.pml_edge_area[0] == 0.5 * spacing[(component + 1) % 3] * spacing[(component + 2) % 3]
    )
    assert np.array_equal(system.pml_edge_scale, system.pml_edge_area)
    # Reusing geometry must reuse the private row, not grow the catalogue.
    prepare_impedance_pml(grid, system)
    assert len(grid.materials) == 4


@pytest.mark.parametrize("order", (0, 2))
def test_finite_constant_and_rational_surface_impedance_can_intersect_pml(order):
    grid, system, *_ = sparse_fixture()
    system.model_Z0[0] = 100
    system.model_info[0, 0] = order
    prepare_impedance_pml(grid, system)
    assert system.pml_edge_count == 1


def test_active_surface_impedance_is_rejected_in_pml():
    grid, system, *_ = sparse_fixture()
    grid.surface_impedance_models = {"wall": SimpleNamespace(allow_active=True)}
    with pytest.raises(ValueError, match="passive models"):
        prepare_impedance_pml(grid, system)


def test_pml_cannot_stretch_normal_to_pmc():
    grid, system, stretch, *_ = sparse_fixture()
    system.port_normal[0, 0] = stretch
    with pytest.raises(ValueError, match="tangent"):
        prepare_impedance_pml(grid, system)


@pytest.mark.parametrize("attribute,value", [("se", 0.01), ("sm", 0.01), ("poles", 1)])
def test_pmc_pml_keeps_loss_and_poles_out_of_curl_capture_material(attribute, value):
    grid, system, *_ = sparse_fixture()
    setattr(grid.materials[0], attribute, value)
    prepare_impedance_pml(grid, system)
    hold = grid.materials[-1]
    assert isinstance(hold, _ImpedancePMLHoldMaterial)
    assert hold.se == hold.sm == 0
    assert not getattr(hold, "poles", 0)


@pytest.mark.parametrize("attribute,value", [("se", np.inf), ("sm", np.inf), ("er", 0), ("se", -0.1)])
def test_pmc_pml_rejects_invalid_retained_host(attribute, value):
    grid, system, *_ = sparse_fixture()
    setattr(grid.materials[0], attribute, value)
    with pytest.raises(ValueError, match="isotropic retained host"):
        prepare_impedance_pml(grid, system)


def test_pmc_pml_requires_uniform_wall_extrusion():
    grid, system, stretch, normal, component, coordinate = sparse_fixture()
    cell = list(coordinate)
    cell[normal] -= 1
    cell[stretch] = 1  # The entrance stencil neighbour must be extruded too.
    grid.solid[tuple(cell)] = 0
    with pytest.raises(ValueError, match="uniformly extruded"):
        prepare_impedance_pml(grid, system)


def test_internal_pml_must_cover_upper_pmc_wall_samples():
    grid, system, _, normal, _, coordinate = sparse_fixture()
    setattr(grid.pmls["slabs"][0], "xyz"[normal] + "f", coordinate[normal])
    with pytest.raises(ValueError, match="upper transverse bound"):
        prepare_impedance_pml(grid, system)


@pytest.mark.integration
@pytest.mark.parametrize("stretch", range(3))
@pytest.mark.parametrize("precision", ("double", "single"))
@pytest.mark.parametrize(
    "formulation,order", [("HORIPML", 1), ("HORIPML", 2), ("MRIPML", 1), ("MRIPML", 2)]
)
def test_pmc_through_pml_matches_all_fields_and_histories_of_mirror(
    tmp_path, stretch, precision, formulation, order
):
    from testing.validation.sibc_based_pmc.pml_mirror import pml_image_case

    result = pml_image_case(tmp_path, stretch, precision, formulation, order, steps=80)
    assert result["passed"], result


@pytest.mark.integration
def test_pmc_pml_image_comparison_detects_missing_pml_forcing(tmp_path):
    from testing.validation.sibc_based_pmc.pml_mirror import pml_image_case

    result = pml_image_case(tmp_path, steps=80, disable_coupling=True)
    assert not result["passed"]
    assert result["maximum_field_relative_error"] > 0.01


@pytest.mark.parametrize("kind", ("lossy", "debye", "lorentz", "drude"))
@pytest.mark.parametrize("layered", (False, True))
@pytest.mark.parametrize("formulation,order", (("HORIPML", 1), ("MRIPML", 2)))
@pytest.mark.parametrize("stretch,precision", ((0, "double"), (1, "single"), (2, "double")))
def test_pml_material_contacts_match_independent_mirrored_bulk_grid(
    tmp_path, kind, layered, formulation, order, stretch, precision,
):
    from testing.validation.sibc_based_pmc.pml_mirror import pml_image_case

    result = pml_image_case(tmp_path, steps=80, host_kind=kind, layered=layered,
                            formulation=formulation, order=order, stretch=stretch, precision=precision)
    assert result["passed"], result
