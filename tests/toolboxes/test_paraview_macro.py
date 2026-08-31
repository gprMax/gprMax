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
