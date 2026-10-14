"""PMC/PML intersections must update the complete, component-specific Yee plane."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gprMax.pml import PML, MetalPML

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("family", ("electric", "magnetic"))
@pytest.mark.parametrize("formulation", PML.formulations)
@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("backend", ("cuda", "opencl", "metal"))
def test_every_device_signature_declares_the_pmc_mask(family, formulation, axis, backend):
    module = import_module(f"gprMax.cuda_opencl.knl_pml_updates_{family}_{formulation}")
    signature = getattr(module, axis + "_args")[backend].template
    compact = "".join(signature.split())
    expected = (
        "deviceconst$REAL&d,deviceconstint&PMC_MASK,"
        if backend == "metal"
        else "$REALd,intPMC_MASK"
    )
    assert expected in compact
    assert compact.count("PMC_MASK") == 1


@pytest.mark.parametrize("direction", PML.directions)
@pytest.mark.parametrize("ownership", ("global", "internal-rank", "short-slab"))
def test_only_owned_upper_pmc_faces_extend_the_slab(make_pml_grid, direction, ownership):
    grid = make_pml_grid(nx=8, ny=9, nz=10)
    grid.symmetry_boundaries = {a + side: "pmc" for a in "xyz" for side in ("0", "max")}
    if ownership == "internal-rank":
        # MPI keeps global declarations, but this rank owns none of them.
        grid.get_local_symmetry_boundaries = lambda: {}
    normal = "xyz".index(direction[0])
    upper = list(grid.size)
    upper[normal] = 3
    if ownership == "short-slab":
        upper = [v - 1 for v in upper]
    slab = PML(grid, "test", direction, 0, upper[0], 0, upper[1], 0, upper[2])
    expected = (7 ^ (1 << normal)) if ownership == "global" else 0
    assert slab.pmc_mask == expected


@pytest.mark.parametrize("boundary", ("pec", "pmc"))
def test_lower_symmetry_faces_do_not_extend_upper_bounds(make_pml_grid, boundary):
    grid = make_pml_grid()
    grid.symmetry_boundaries = {a + "0": boundary for a in "xyz"}
    slab = PML(grid, "test", "zplus", 0, grid.nx, 0, grid.ny, 7, 10)
    assert slab.pmc_mask == 0


@pytest.mark.parametrize("family", ("electric", "magnetic"))
@pytest.mark.parametrize("formulation", PML.formulations)
@pytest.mark.parametrize("order", (1, 2))
@pytest.mark.parametrize("direction", PML.directions)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("selected", (0, 1, 2), ids=("first-face", "second-face", "both-faces"))
def test_wall_correction_matches_interior_fields_and_histories(
    family, formulation, order, direction, dtype, selected
):
    """Use an ordinary larger slab as an independent spatial-coverage oracle.

    Nonzero initial fields AND histories catch missing recurrence updates.
    Compare all storage, including untouched padding and the other family.
    No PMC correction formula is duplicated in this test.
    """
    normal = "xyz".index(direction[0])
    components = [a for a in range(3) if a != normal]
    chosen = components if selected == 2 else [components[selected]]
    mask = sum(1 << a for a in chosen)
    lower, size = np.array((2, 2, 2)), np.array((3, 4, 5))
    upper = lower + size
    large_upper = upper.copy()
    large_upper[components] += 1
    shape = (11, 12, 13)
    rng = np.random.default_rng(617)
    initial = [rng.standard_normal(shape).astype(dtype) for _ in range(6)]
    actual, reference = ([a.copy() for a in initial] for _ in range(2))
    ids = rng.integers(0, 3, size=(6, *shape), dtype=np.uint32)
    material = rng.uniform(0.05, 0.15, size=(3, 5)).astype(dtype)
    material[0, :] = 0  # inactive/PEC entries must remain inactive
    coefficients = [
        rng.uniform(0.1, 0.8, size=(order, size[normal])).astype(dtype) for _ in range(4)
    ]
    histories, large_histories, before = [], [], []
    for component in components:
        padded = (
            next(a for a in components if a != component) if family == "electric" else component
        )
        extent = size.copy()
        extent[padded] += 1
        if family == "electric":
            extent[normal] += 1
        big = rng.standard_normal((order, 8, 8, 8)).astype(dtype)
        h = big[(slice(None), *(slice(0, int(n)) for n in extent))].copy()
        histories.append(h)
        before.append(h.copy())
        large_histories.append(big)
    kernel = getattr(
        import_module(f"gprMax.cython.pml_updates_{family}_{formulation}"),
        f"order{order}_{direction}",
    )

    def call(fields, phis, stop, pmc):
        bounds = [int(v) for pair in zip(lower, stop) for v in pair]
        kernel(*bounds, 2, material, ids, *fields, *phis, *coefficients, dtype(0.0013), pmc)

    for _ in range(3):
        call(actual, histories, upper, mask)
        call(reference, large_histories, large_upper, 0)
    expected = [a.copy() for a in initial]
    for index, component in enumerate(components):
        padded = (
            next(a for a in components if a != component) if family == "electric" else component
        )
        extent = size.copy()
        extent[padded] += int(padded in chosen)
        coords = [np.arange(n) + lower[a] for a, n in enumerate(extent)]
        if direction.endswith("minus"):
            coords[normal] = upper[normal] - np.arange(size[normal]) - (family == "magnetic")
        indices = np.ix_(*coords)
        field_index = component + (3 if family == "magnetic" else 0)
        expected[field_index][indices] = reference[field_index][indices]
        region = (slice(None), *(slice(0, int(n)) for n in extent))
        before[index][region] = large_histories[index][region]
        assert_array_equal(histories[index], before[index])
    for result, want in zip(actual, expected):
        assert_array_equal(result, want)


@pytest.mark.parametrize("family", ("electric", "magnetic"))
def test_metal_uploads_the_slab_mask_at_the_kernel_argument_index(make_pml_grid, family):
    """Exercise actual MetalPML dispatch without requiring an Apple host."""
    grid = make_pml_grid()
    grid.symmetry_boundaries = {"ymax": "pmc"}
    slab = MetalPML(grid, "test", "zplus", 0, 10, 0, 10, 7, 10)
    slab.calculate_update_coeffs(1, 1)
    assert slab.pmc_mask == 2
    recorded = {}

    class Encoder:
        def setBytes_length_atIndex_(self, data, length, index):
            recorded[index] = data

        def __getattr__(self, name):
            return lambda *args: None

    encoder = Encoder()
    command = SimpleNamespace(
        computeCommandEncoder=lambda: encoder, commit=lambda: None, waitUntilCompleted=lambda: None
    )
    slab.queue = SimpleNamespace(commandBuffer=lambda: command)
    slab.psoE = slab.psoH = object()
    grid.tptg = grid.tgs = (1, 1, 1)
    for name in ("ID", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        setattr(grid, name + "_dev", object())
    for prefix in ("E", "H"):
        for name in ("Phi1", "Phi2", "RA", "RB", "RE", "RF"):
            setattr(slab, prefix + name + "_dev", object())
        for name in ("Phi1", "Phi2"):
            setattr(slab, prefix + name + "_shape", getattr(slab, prefix + name).shape)
    getattr(slab, "update_" + family)()
    assert int(np.frombuffer(recorded[27], dtype=np.int32)[0]) == 2
