"""Auxiliary backend selection and exception-safe CUDA HSG construction."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gprMax import config
from gprMax.subgrids import cuda_precursor_nodes, cuda_subgrid_updates as hsg
from gprMax.subgrids.subgrid_hsg import SubGridHSG
from gprMax.updates.cuda_updates import CUDAUpdates
from gprMax.virtual_waveguide import VirtualWaveguide

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("backend,class_name", [("cuda", "CUDAGrid"), ("opencl", "OpenCLGrid"), ("metal", "MetalGrid")])
def test_auxiliary_grid_of_a_subgrid_uses_device_storage(monkeypatch, backend, class_name):
    class SelectedGrid(Exception):
        pass

    guide = VirtualWaveguide.__new__(VirtualWaveguide)
    guide.main_grid = SubGridHSG.__new__(SubGridHSG)
    guide.mpi = False
    monkeypatch.setattr(config, "sim_config", SimpleNamespace(general={"solver": backend}))
    module = import_module(f"gprMax.grid.{backend}_grid")
    factory = Mock(side_effect=SelectedGrid)
    monkeypatch.setattr(module, class_name, factory)
    # Stop at construction: no actual GPU or numerical field allocation needed.
    with pytest.raises(SelectedGrid):
        guide._build_auxiliary_grid()
    factory.assert_called_once_with()


@pytest.mark.parametrize(
    "backend,class_name", [("cuda", "CUDAUpdates"), ("opencl", "OpenCLUpdates"), ("metal", "MetalUpdates")]
)
def test_auxiliary_grid_uses_plain_backend_not_parent_subclass(monkeypatch, backend, class_name):
    class SpecialisedParent:
        def __init__(self, grid, required_hsg_argument):
            raise AssertionError("An auxiliary grid is not an HSG orchestrator")

    parent = SpecialisedParent.__new__(SpecialisedParent)
    parent.queue, parent.dev = object(), object()
    guide = VirtualWaveguide.__new__(VirtualWaveguide)
    guide.aux_updates = None
    guide.aux_grid = SimpleNamespace(htod_mat_coeff_arrays=Mock(), htod_material_arrays=Mock(), dt=0.25)
    module = import_module(f"gprMax.updates.{backend}_updates")
    factory = Mock()
    monkeypatch.setattr(module, class_name, factory)
    monkeypatch.setattr(config, "sim_config", SimpleNamespace(general={"solver": backend}))

    guide.initialise_device(parent)
    factory.assert_called_once_with(guide.aux_grid, shared=parent)
    assert guide.aux_updates is factory.return_value
    assert guide.aux_grid.dt == 0.25
    if backend == "cuda":
        guide.aux_grid.htod_mat_coeff_arrays.assert_called_once_with()
    elif backend == "opencl":
        guide.aux_grid.htod_mat_coeff_arrays.assert_called_once_with(parent.queue)
    else:
        guide.aux_grid.htod_material_arrays.assert_called_once_with(parent.dev)
    guide.initialise_device(parent)
    assert factory.call_count == 1
    # A new run must replace both the updater and context-bound material tables.
    guide.aux_updates = None
    guide.initialise_device(parent)
    assert factory.call_count == 2


def test_partial_main_initialisation_can_clean_up(monkeypatch):
    calls = []

    def initialise(updater, grid):
        updater.ctx = SimpleNamespace(pop=lambda: calls.append("pop"), detach=lambda: calls.append("detach"))
        updater._owns_context = True
        try:
            raise RuntimeError("main setup failed")
        finally:
            updater.cleanup()

    monkeypatch.setattr(CUDAUpdates, "__init__", initialise)
    with pytest.raises(RuntimeError, match="main setup failed"):
        hsg.CUDASubgridUpdates(object(), [])
    assert calls == ["pop", "detach"]


def test_parent_context_is_released_even_if_child_cleanup_fails():
    calls = []
    updater = hsg.CUDASubgridUpdates.__new__(hsg.CUDASubgridUpdates)
    updater.ctx = SimpleNamespace(pop=lambda: calls.append("pop"), detach=lambda: calls.append("detach"))
    updater._owns_context = True
    updater.updaters = [SimpleNamespace(cleanup=Mock(side_effect=RuntimeError("child cleanup")))]
    with pytest.raises(RuntimeError, match="child cleanup"):
        updater.cleanup()
    assert calls == ["pop", "detach"]


def test_unsupported_subgrid_is_rejected_before_allocating_context(monkeypatch):
    class Subgrid:
        equal_resolution = True

    factory = Mock()
    monkeypatch.setattr(hsg, "CUDASubgridUpdates", factory)
    model = SimpleNamespace(G=object(), subgrids=[object()])
    with pytest.raises(ValueError):
        hsg.create_cuda_updates(model, Subgrid)
    factory.assert_not_called()


@pytest.mark.parametrize("stage", ["coefficients", "precursors", "child", "interface", "device", "seed"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_factory_cleans_parent_and_registered_children_on_setup_error(monkeypatch, stage, cleanup_fails):
    class Subgrid:
        filter = True
        equal_resolution = False

    sg = Subgrid()
    sg.gpuarray, sg.tpb = object(), (128, 1, 1)
    sg.setup_cuda_interface = Mock()
    parent = SimpleNamespace(updaters=[], cleanup=Mock())
    if cleanup_fails:
        parent.cleanup.side_effect = RuntimeError("secondary cleanup failure")
    child = SimpleNamespace(knls_interface={}, knls_precursor={})
    precursors = SimpleNamespace(setup_device=Mock(), update_electric=Mock())
    grid = SimpleNamespace(htod_mat_coeff_arrays=Mock())
    monkeypatch.setattr(hsg, "CUDASubgridUpdates", Mock(return_value=parent))
    child_factory = Mock(return_value=child)
    monkeypatch.setattr(hsg, "CUDASubgridUpdater", child_factory)
    precursor_factory = Mock(return_value=precursors)
    monkeypatch.setattr(cuda_precursor_nodes, "CUDAPrecursorNodesFiltered", precursor_factory)
    failing_call = {
        "coefficients": grid.htod_mat_coeff_arrays,
        "precursors": precursor_factory,
        "child": child_factory,
        "interface": sg.setup_cuda_interface,
        "device": precursors.setup_device,
        "seed": precursors.update_electric,
    }[stage]
    failing_call.side_effect = RuntimeError("controlled setup failure")
    with pytest.raises(RuntimeError, match="controlled setup failure"):
        hsg.create_cuda_updates(SimpleNamespace(G=grid, subgrids=[sg]), Subgrid)
    parent.cleanup.assert_called_once_with()
    if stage in ("interface", "device", "seed"):
        assert parent.updaters == [child]


def test_material_coefficients_are_uploaded_for_each_new_solver():
    grid = SimpleNamespace(updatecoeffsE_dev=object(), htod_mat_coeff_arrays=Mock())
    hsg._upload_mat_coeffs(grid)
    hsg._upload_mat_coeffs(grid)
    assert grid.htod_mat_coeff_arrays.call_count == 2
