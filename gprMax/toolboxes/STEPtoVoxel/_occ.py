# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""OpenCascade bindings used by the STEP reader.

Two Python bindings of OpenCascade work:

* ``pythonocc-core`` (module ``OCC``), installed from conda-forge;
* ``cadquery-ocp`` (module ``OCP``), installed with ``pip``.

``pythonocc-core`` is used when both are installed. This module exposes the
same names for either binding, so ``step_parser`` is written once. With OCP,
static methods carry an ``_s`` suffix, strings are ``TCollection`` objects,
and arguments written by reference are passed in as objects; the small
adapters below hide those differences.

``OCCError`` is the exception the binding raises for an OpenCascade failure.
"""

from typing import Any, Callable, Tuple

try:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.Quantity import Quantity_Color
    from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
    from OCC.Core.TDataStd import TDataStd_Name
    from OCC.Core.TDF import TDF_Label, TDF_LabelSequence
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.XCAFApp import XCAFApp_Application
    from OCC.Core.XCAFDoc import XCAFDoc_ColorTool, XCAFDoc_ColorType
    from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool as _XCAFDoc_DocumentTool
    from OCC.Extend.TopologyUtils import TopologyExplorer

    BACKEND = "pythonocc-core"
    OCCError = RuntimeError

    class _ColorTool:
        """``GetColor(label, color_type)`` returning ``(found, colour)``.

        pythonocc accepts only the static form that fills a ``Quantity_Color``;
        the instance call ``GetColor(label, color_type)`` raises ``TypeError``.
        """

        def GetColor(self, label: Any, color_type: Any) -> Tuple[bool, Any]:
            color = Quantity_Color()
            return bool(XCAFDoc_ColorTool.GetColor(label, color_type, color)), color

    class XCAFDoc_DocumentTool:
        ShapeTool = staticmethod(_XCAFDoc_DocumentTool.ShapeTool)

        @staticmethod
        def ColorTool(label: Any) -> Any:
            _XCAFDoc_DocumentTool.ColorTool(label)  # creates the tool if absent
            return _ColorTool()

    def label_name(label: Any) -> str:
        """The label's TDataStd_Name, else ''."""
        attr = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID(), attr):
            return str(attr.Get()).strip()
        return ""

    def label_fallback_name(label: Any) -> str:
        """pythonocc's ``TDF_Label.GetLabelName``."""
        return label.GetLabelName()

    def step_entity_labeller(reader: Any, metadata: Any) -> Callable[[Any], str]:
        """Return a function giving a STEP entity's label in the file (``#42``).

        Args:
            reader: The ``STEPCAFControl_Reader`` that read the file.
            metadata: The file's ``StepMetadata`` (not needed with pythonocc).
        """
        interface_model = reader.Reader().WS().Model()

        def entity_label(entity: Any) -> str:
            label = interface_model.StringLabel(entity)
            return "" if label is None else label.ToCString()

        return entity_label

except ImportError as _occ_error:
    try:
        import OCP  # noqa: F401
    except ImportError:
        raise _occ_error from None

    from OCP.Bnd import Bnd_Box as _Bnd_Box
    from OCP.BRep import BRep_Tool as _BRep_Tool
    from OCP.BRepBndLib import BRepBndLib as _BRepBndLib
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.BRepGProp import BRepGProp as _BRepGProp
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static as _Interface_Static
    from OCP.Quantity import Quantity_Color
    from OCP.Standard import Standard_Failure as OCCError
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDF import TDF_Label, TDF_Tool
    from OCP.TDocStd import TDocStd_Document as _TDocStd_Document
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX
    from OCP.TopExp import TopExp
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.XCAFApp import XCAFApp_Application as _XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_ColorType
    from OCP.XCAFDoc import XCAFDoc_DocumentTool as _XCAFDoc_DocumentTool

    try:  # OCP 7.x
        from OCP.TDF import TDF_LabelSequence
        from OCP.TopTools import TopTools_IndexedMapOfShape
    except ImportError:  # OCP 8.x
        from OCP.collections import (
            IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as TopTools_IndexedMapOfShape,
        )
        from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence

    BACKEND = "cadquery-ocp"

    def _downcast(kind: str) -> Any:
        """TopoDS.Face etc.: a static ``Face_s`` in OCP 7.x, a function in 8.x."""
        return getattr(TopoDS, kind + "_s", None) or getattr(TopoDS, kind)

    _as_face = _downcast("Face")
    _as_vertex = _downcast("Vertex")

    class _Static:
        """Call ``name`` as ``name`` or, failing that, OCP's static ``name_s``."""

        def __init__(self, target: Any):
            self._target = target

        def __getattr__(self, name: str) -> Any:
            try:
                return getattr(self._target, name)
            except AttributeError:
                return getattr(self._target, name + "_s")

    brepbndlib = _Static(_BRepBndLib)
    brepgprop = _Static(_BRepGProp)
    Interface_Static = _Static(_Interface_Static)
    XCAFApp_Application = _Static(_XCAFApp_Application)

    class BRep_Tool:
        @staticmethod
        def Triangulation(face: Any, location: Any) -> Any:
            return _BRep_Tool.Triangulation_s(_as_face(face), location)

        @staticmethod
        def Pnt(vertex: Any) -> Any:
            return _BRep_Tool.Pnt_s(_as_vertex(vertex))

    class Bnd_Box(_Bnd_Box):
        """OCP 8's ``Bnd_Box.Get`` returns an unregistered C++ struct."""

        def Get(self) -> Tuple[float, float, float, float, float, float]:
            lo, hi = self.CornerMin(), self.CornerMax()
            return lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z()

    def TDocStd_Document(storage_format: str) -> Any:
        return _TDocStd_Document(TCollection_ExtendedString(storage_format))

    class _ColorTool(_Static):
        def GetColor(self, label: Any, color_type: Any) -> Tuple[bool, Any]:
            color = Quantity_Color()
            return bool(self._target.GetColor_s(label, color_type, color)), color

    class XCAFDoc_DocumentTool:
        @staticmethod
        def ShapeTool(label: Any) -> Any:
            return _Static(_XCAFDoc_DocumentTool.ShapeTool_s(label))

        @staticmethod
        def ColorTool(label: Any) -> Any:
            return _ColorTool(_XCAFDoc_DocumentTool.ColorTool_s(label))

    class TopologyExplorer:
        """The parts of ``OCC.Extend.TopologyUtils.TopologyExplorer`` used here.

        Sub-shapes are unique (orientation ignored) and in the order they are
        first met, as with pythonocc's explorer.
        """

        _CAST = {
            TopAbs_SOLID: _downcast("Solid"),
            TopAbs_SHELL: _downcast("Shell"),
            TopAbs_FACE: _downcast("Face"),
            TopAbs_EDGE: _downcast("Edge"),
            TopAbs_VERTEX: _downcast("Vertex"),
        }

        def __init__(self, shape: Any):
            self._shape = shape

        def _unique(self, kind: Any):
            shapes = TopTools_IndexedMapOfShape()
            TopExp.MapShapes_s(self._shape, kind, shapes)
            cast = self._CAST[kind]
            return (cast(shapes.FindKey(i)) for i in range(1, shapes.Extent() + 1))

        def solids(self):
            return self._unique(TopAbs_SOLID)

        def shells(self):
            return self._unique(TopAbs_SHELL)

        def faces(self):
            return self._unique(TopAbs_FACE)

        def edges(self):
            return self._unique(TopAbs_EDGE)

        def vertices(self):
            return self._unique(TopAbs_VERTEX)

    def label_name(label: Any) -> str:
        """The label's TDataStd_Name, else ''."""
        attr = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
            return attr.Get().ToExtString().strip()
        return ""

    # pythonocc's TDF_Label.GetLabelName returns the TDataStd_Name, else ''.
    label_fallback_name = label_name

    def step_entity_labeller(reader: Any, metadata: Any) -> Callable[[Any], str]:
        """Return a function giving a STEP entity's label in the file (``#42``).

        This works around an OCP bug: a ``Handle(Standard_Transient)`` argument
        arrives null when a subclass is passed, so ``Interface_InterfaceModel``
        lookups such as ``StringLabel`` or ``Number`` find nothing. The model
        keeps the entities in file order, as does gprMax's own STEP parser, so
        an entity's number is found by identity among the model's entities and
        its label is the parsed entity at that position. The work session's own
        label search confirms each label. Once OCP passes such handles
        correctly, the pythonocc version above can be used instead.

        Args:
            reader: The ``STEPCAFControl_Reader`` that read the file.
            metadata: The file's ``StepMetadata``; its ``entities`` are in file
                order.
        """
        work_session = reader.Reader().WS()
        model = work_session.Model()
        entities = [model.Value(i) for i in range(1, model.NbEntities() + 1)]
        numbers = {id(entity): i for i, entity in enumerate(entities, 1)}
        labels = [f"#{identifier}" for identifier in metadata.entities]

        def entity_label(entity: Any) -> str:
            number = numbers.get(id(entity))
            if number is None or number > len(labels):
                return ""
            label = labels[number - 1]
            if work_session.NumberFromLabel(label, number - 1) != number:
                return ""
            return label

        entity_label.entities = entities  # keep the Python wrappers alive
        return entity_label


__all__ = [
    "BACKEND",
    "Bnd_Box",
    "BRep_Tool",
    "brepbndlib",
    "BRepBuilderAPI_Transform",
    "brepgprop",
    "BRepMesh_IncrementalMesh",
    "gp_Ax1",
    "gp_Dir",
    "gp_Pnt",
    "gp_Trsf",
    "GProp_GProps",
    "IFSelect_RetDone",
    "Interface_Static",
    "STEPCAFControl_Reader",
    "TDataStd_Name",
    "TDF_Label",
    "TDF_LabelSequence",
    "TDocStd_Document",
    "TopLoc_Location",
    "TopologyExplorer",
    "XCAFApp_Application",
    "XCAFDoc_ColorType",
    "XCAFDoc_DocumentTool",
    "OCCError",
    "label_fallback_name",
    "label_name",
    "step_entity_labeller",
]
