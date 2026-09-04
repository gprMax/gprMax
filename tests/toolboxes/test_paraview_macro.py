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

"""Regression tests for the standalone gprMax ParaView macro."""

import ast
from pathlib import Path


MACRO = Path(__file__).parents[2] / "toolboxes" / "Utilities" / "Paraview" / "gprMax.py"


class _MaterialNames:
    def __init__(self, names):
        self.names = names

    def GetNumberOfValues(self):
        return len(self.names)

    def GetValue(self, index):
        return self.names[index]


class _Version:
    major = 5
    minor = 12


def _load_function(name, namespace):
    """Load one function without importing the ParaView-only dependencies."""

    tree = ast.parse(MACRO.read_text(encoding="utf-8"), filename=str(MACRO))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(MACRO), "exec"), namespace)
    return namespace[name]


def test_free_space_is_hidden_by_name_not_numeric_position():
    """PMC at index one remains visible while remapped free space is hidden."""

    thresholds = []
    renamed = []
    shown = []

    def make_threshold(**kwargs):
        threshold = type("ThresholdProxy", (), {})()
        threshold.kwargs = kwargs
        threshold.UpdatePipeline = lambda: None
        thresholds.append(threshold)
        return threshold

    namespace = {
        "SourceProxy": object,
        "Proxy": object,
        "vtkStringArray": object,
        "MATERIAL_SCALARS": ("CELLS", "Material"),
        "GetParaViewVersion": lambda: _Version(),
        "Threshold": make_threshold,
        "RenameSource": lambda name, threshold: renamed.append((name, threshold)),
        "Show": lambda threshold, view: shown.append((threshold, view)),
    }
    threshold_materials = _load_function("threshold_materials", namespace)
    names = _MaterialNames(["pec", "pmc", "soil", "free_space"])
    view = object()

    threshold_materials(object(), view, names)

    assert len(thresholds) == 4
    assert [name for name, _threshold in renamed] == names.names
    shown_names = [renamed[thresholds.index(threshold)][0] for threshold, _view in shown]
    assert shown_names == ["pec", "pmc", "soil"]


def test_geometry_tags_create_hidden_thresholds_using_catalogue_ids():
    """Tag filters use semantic IDs and do not obscure the material view."""

    thresholds = []
    renamed = []

    def make_threshold(**kwargs):
        threshold = type("ThresholdProxy", (), {})()
        threshold.kwargs = kwargs
        threshold.UpdatePipeline = lambda: None
        thresholds.append(threshold)
        return threshold

    namespace = {
        "SourceProxy": object,
        "Proxy": object,
        "dsa": type("DSA", (), {"VTKArray": object}),
        "vtkStringArray": object,
        "TAG_SCALARS": ("CELLS", "TagID"),
        "HaltException": RuntimeError,
        "GetParaViewVersion": lambda: _Version(),
        "Threshold": make_threshold,
        "RenameSource": lambda name, threshold: renamed.append((name, threshold)),
    }
    threshold_tags = _load_function("threshold_geometry_tags", namespace)

    threshold_tags(object(), [3, 7], _MaterialNames(["head", "eyes"]))

    assert [threshold.LowerThreshold for threshold in thresholds] == [3, 7]
    assert [threshold.UpperThreshold for threshold in thresholds] == [3, 7]
    assert all(threshold.kwargs["Scalars"] == ("CELLS", "TagID") for threshold in thresholds)
    assert [name for name, _threshold in renamed] == ["Tag - head", "Tag - eyes"]


def test_typed_source_geometry_creates_points_boxes_and_planes():
    """The macro renders each geometry kind and exposes its type in the name."""

    boxes = []
    planes = []
    renamed = []
    displays = []

    def make_proxy(kind, kwargs):
        proxy = type("SourceProxy", (), {})()
        proxy.kind = kind
        proxy.kwargs = kwargs
        return proxy

    def make_box(**kwargs):
        proxy = make_proxy("box", kwargs)
        boxes.append(proxy)
        return proxy

    def make_plane(**kwargs):
        proxy = make_proxy("plane", kwargs)
        planes.append(proxy)
        return proxy

    def show(source, view):
        display = type("DisplayProxy", (), {"Representation": "Surface"})()
        displays.append((source, view, display))
        return display

    namespace = {
        "Proxy": object,
        "dsa": type("DSA", (), {"VTKArray": object}),
        "vtkStringArray": object,
        "HaltException": RuntimeError,
        "Box": make_box,
        "Plane": make_plane,
        "Show": show,
        "RenameSource": lambda name, source: renamed.append((name, source)),
    }
    create_sources = _load_function("create_source_geometries", namespace)
    view = object()

    create_sources(
        _MaterialNames(["feed", "plane_wave_1", "plane_wave_2", "port2"]),
        _MaterialNames(
            ["VoltageSource", "DiscretePlaneWave", "DiscretePlaneWave", "EigenmodeSource"]
        ),
        _MaterialNames(["point", "box", "rectangle", "plane"]),
        [
            [1, 2, 3, 4, 5, 6],
            [0, 10, 0, 20, 0, 30],
            [0, 10, 0, 20, 0, 0],
            [4, 4, 2, 7, 3, 9],
        ],
        view,
    )

    assert len(boxes) == 2
    assert boxes[0].kwargs == {
        "Center": [1.5, 3.5, 5.5],
        "XLength": 1.0,
        "YLength": 1.0,
        "ZLength": 1.0,
    }
    assert displays[1][2].Representation == "Wireframe"
    assert displays[2][2].Representation == "Wireframe"
    assert len(planes) == 2
    assert planes[1].kwargs == {
        "Origin": [4.0, 2.0, 3.0],
        "Point1": [4.0, 7.0, 3.0],
        "Point2": [4.0, 2.0, 9.0],
    }
    assert [name for name, _source in renamed] == [
        "Source - VoltageSource - feed",
        "Source - DiscretePlaneWave - plane_wave_1",
        "Source - DiscretePlaneWave - plane_wave_2",
        "Source - EigenmodeSource - port2",
    ]

    create_sources(
        _MaterialNames(["receive_port"]),
        _MaterialNames(["VoltageSourcePort"]),
        _MaterialNames(["point"]),
        [[1, 2, 3, 4, 5, 6]],
        view,
        role="Receiver",
    )
    assert renamed[-1][0] == "Receiver - VoltageSourcePort - receive_port"
