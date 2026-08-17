"""Regression tests for the standalone gprMax ParaView macro."""

import ast
from pathlib import Path


MACRO = (
    Path(__file__).parents[2] / "toolboxes" / "Utilities" / "Paraview" / "gprMax.py"
)


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


def _load_threshold_materials(namespace):
    """Load one function without importing the ParaView-only dependencies."""

    tree = ast.parse(MACRO.read_text(encoding="utf-8"), filename=str(MACRO))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "threshold_materials"
    )
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(MACRO), "exec"), namespace)
    return namespace["threshold_materials"]


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
        "COLOUR_SCALARS": ("CELLS", "Material"),
        "GetParaViewVersion": lambda: _Version(),
        "Threshold": make_threshold,
        "RenameSource": lambda name, threshold: renamed.append((name, threshold)),
        "Show": lambda threshold, view: shown.append((threshold, view)),
    }
    threshold_materials = _load_threshold_materials(namespace)
    names = _MaterialNames(["pec", "pmc", "soil", "free_space"])
    view = object()

    threshold_materials(object(), view, names)

    assert len(thresholds) == 4
    assert [name for name, _threshold in renamed] == names.names
    shown_names = [renamed[thresholds.index(threshold)][0] for threshold, _view in shown]
    assert shown_names == ["pec", "pmc", "soil"]
