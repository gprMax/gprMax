# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.

import numpy as np
import pytest

from gprMax.user_objects.cmds_geometry.cmds_geometry import (
    validate_distributed_geometry_rasterisation,
)
from gprMax.user_objects.cmds_geometry.edge import Edge
from gprMax.user_objects.cmds_geometry.magnetic_edge import MagneticEdge
from gprMax.user_objects.cmds_geometry.thin_wire import ThinWire


class _Comm:
    def __init__(self, counts):
        self._counts = counts

    def allgather(self, value):
        return [value for _ in self._counts]

    def Allreduce(self, send, receive):
        receive[:] = np.array([sum(self._counts)], dtype=np.int64)


class _Grid:
    def __init__(self, size=(20, 20, 20), dl=(0.001, 0.001, 0.001), distributed=False, counts=None):
        self.size = np.array(size, dtype=np.int32)
        self.nx, self.ny, self.nz = size
        self.dx, self.dy, self.dz = dl
        self.dl = np.array(dl, dtype=float)
        self.averagevolumeobjects = True
        self.materials = []
        self.is_distributed = distributed
        self.geometry_rasterisation_records = []
        self.thinwires = []
        self.rigidE = np.zeros((12, self.nx, self.ny, self.nz), dtype=np.int8)
        self.rigidH = np.zeros((6, self.nx, self.ny, self.nz), dtype=np.int8)
        self.ID = np.zeros((6, self.nx + 1, self.ny + 1, self.nz + 1), dtype=np.uint32)
        self.solid = np.zeros(size, dtype=np.uint32)
        self.fractalvolumes = []
        if distributed:
            self.comm = _Comm(counts or [0, 0])
            self.lower_extent = np.array([0, 0, 0], dtype=np.int32)
            self.upper_extent = np.array(size, dtype=np.int32)
            self.global_size = np.array(size, dtype=np.int32)

    def global_to_local_coordinate(self, point):
        return np.asarray(point, dtype=np.int32) - self.lower_extent

    def is_coordinator(self):
        return True


class _Material:
    def __init__(self, numID=1):
        self.numID = numID
        self.ID = "pec"
        self.averagable = False
        self.is_pec = True


class _UIP:
    def __init__(self, within=True, p1=(1, 1, 1), p2=(5, 1, 1)):
        self._within = within
        self._p1 = np.array(p1, dtype=np.int32)
        self._p2 = np.array(p2, dtype=np.int32)

    def resolve_inf_point(self, p, role=None):
        return p

    def check_box_points(self, p1, p2, cmd_str):
        return self._within, self._p1, self._p2

    def round_to_grid_static_point(self, p):
        return tuple(float(v) for v in p)

    def discretise_static_point(self, p):
        return tuple(int(round(v / 0.001)) for v in p)


@pytest.mark.parametrize("cls,kwargs", [
    (Edge, dict(p1=(0.001, 0.001, 0.001), p2=(0.005, 0.001, 0.001), material_id="pec")),
    (MagneticEdge, dict(p1=(0.001, 0.001, 0.001), p2=(0.005, 0.001, 0.001), material_id="pec")),
])
def test_edge_like_commands_record_zero_on_distributed_early_return(monkeypatch, cls, kwargs):
    obj = cls(**kwargs)
    grid = _Grid(distributed=True, counts=[0, 0])
    grid.materials = [_Material()]

    monkeypatch.setattr(obj, "_create_uip", lambda g: _UIP(within=False))
    monkeypatch.setattr("gprMax.user_objects.cmds_geometry.edge.resolve_geometry_materials", lambda *a, **k: [_Material()])
    monkeypatch.setattr("gprMax.user_objects.cmds_geometry.magnetic_edge.resolve_geometry_materials", lambda *a, **k: [_Material()])
    monkeypatch.setattr("gprMax.config.get_model_config", lambda: type("C", (), {"mode": "3D"})())

    obj.build(grid)

    assert grid.geometry_rasterisation_records == [(0, obj.params_str())]
    with pytest.raises(ValueError, match="does not occupy any Yee cells or faces"):
        validate_distributed_geometry_rasterisation(grid)


@pytest.mark.parametrize("cls,kwargs,patch_path", [
    (Edge, dict(p1=(0.001, 0.001, 0.001), p2=(0.005, 0.001, 0.001), material_id="pec"), "gprMax.user_objects.cmds_geometry.edge.resolve_geometry_materials"),
    (MagneticEdge, dict(p1=(0.001, 0.001, 0.001), p2=(0.005, 0.001, 0.001), material_id="pec"), "gprMax.user_objects.cmds_geometry.magnetic_edge.resolve_geometry_materials"),
])
def test_edge_like_commands_record_occupied_count(monkeypatch, cls, kwargs, patch_path):
    obj = cls(**kwargs)
    grid = _Grid(distributed=True, counts=[4, 0])
    monkeypatch.setattr(obj, "_create_uip", lambda g: _UIP(within=True, p1=(1, 1, 1), p2=(5, 1, 1)))
    monkeypatch.setattr(patch_path, lambda *a, **k: [_Material()])
    monkeypatch.setattr("gprMax.user_objects.cmds_geometry.edge.build_edge_x", lambda *a, **k: None)
    monkeypatch.setattr("gprMax.user_objects.cmds_geometry.magnetic_edge.build_magnetic_edge_x", lambda *a, **k: None)
    monkeypatch.setattr("gprMax.config.get_model_config", lambda: type("C", (), {"mode": "3D"})())

    obj.build(grid)

    assert grid.geometry_rasterisation_records == [(4, obj.params_str())]
    validate_distributed_geometry_rasterisation(grid)


def test_thin_wire_records_zero_on_distributed_early_return(monkeypatch):
    obj = ThinWire(p1=(0.001, 0.001, 0.001), p2=(0.005, 0.001, 0.001), radius=1e-4)
    grid = _Grid(distributed=True, counts=[0, 0])
    grid.lower_extent = np.array([10, 0, 0], dtype=np.int32)
    grid.upper_extent = np.array([20, 20, 20], dtype=np.int32)
    monkeypatch.setattr(obj, "_create_uip", lambda g: _UIP(within=False))
    monkeypatch.setattr("gprMax.config.get_model_config", lambda: type("C", (), {"mode": "3D"})())
    obj.build(grid)
    assert grid.geometry_rasterisation_records == [(0, obj.params_str())]
    with pytest.raises(ValueError, match="does not occupy any Yee cells or faces"):
        validate_distributed_geometry_rasterisation(grid)


def test_thin_wire_records_occupied_count(monkeypatch):
    obj = ThinWire(p1=(0.001, 0.001, 0.001), p2=(0.005, 0.001, 0.001), radius=1e-4)
    grid = _Grid(distributed=True, counts=[4, 0])
    monkeypatch.setattr(obj, "_create_uip", lambda g: _UIP(within=True, p1=(1, 1, 1), p2=(5, 1, 1)))
    monkeypatch.setattr("gprMax.config.get_model_config", lambda: type("C", (), {"mode": "3D"})())
    obj.build(grid)
    assert grid.geometry_rasterisation_records == [(4, obj.params_str())]
    validate_distributed_geometry_rasterisation(grid)


pytestmark = pytest.mark.unit
