# Copyright (c) 2026 Mahdee Abir
# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# Originally developed for STEP-to-gprMax and distributed under the MIT
# License. This adapted version is part of gprMax and is distributed under the
# GNU General Public License, version 3 or (at your option) any later version.
# See LICENSE in this directory.

"""Read STEP assemblies and tessellate their components with OpenCascade."""

import itertools
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
from OCC.Core.GProp import GProp_GProps

# ------------------ OpenCascade imports ------------------
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.TDataStd import TDataStd_Name
from OCC.Core.TDF import TDF_Label, TDF_LabelSequence
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.XCAFApp import XCAFApp_Application
from OCC.Core.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool
from OCC.Extend.TopologyUtils import TopologyExplorer

from .step_metadata import (
    StepMetadata,
    is_generated_name,
    is_generic_cad_name,
    semantic_leaf_name,
)

_REPRESENTATION_NAME_RE = re.compile(
    r"(?:ADVANCED_BREP_SHAPE_REPRESENTATION|MANIFOLD_SURFACE_SHAPE_REPRESENTATION)" r"\s*\(\s*'((?:[^']|'')*)'",
    flags=re.IGNORECASE,
)


# ===============================================================
# 1) Parser Configuration
# ===============================================================
@dataclass
class ParserConfig:
    # Reporting / debug
    verbose_per_part: bool = False

    # Placement / explode policy
    apply_occurrence_locations: bool = True
    auto_explode_solids: bool = True

    # Unit import
    force_units_to_metres: bool = True

    # Global rotation
    enable_global_rotation: bool = False
    rotate_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotate_about: str = "bbox_global"  # "bbox_global" or "origin"
    rotate_order: str = "xyz"

    # Tessellation controls
    linear_deflection: float = 0.001
    angular_deflection: float = 0.5
    is_relative_deflection: bool = True

    # Degenerate triangle filter
    eps_area2: float = 1e-30

    # Post-processing
    compact_tessellation: bool = True

    export_stl: bool = False
    stl_outname: str = "assembly.stl"


# ===============================================================
# 2) Data Structures
# ===============================================================
_uid_counter = itertools.count(1)


def next_uid() -> int:
    return next(_uid_counter)


def reset_uid_counter(start: int = 1) -> None:
    global _uid_counter
    _uid_counter = itertools.count(start)


@dataclass
class Part:
    uid: int
    shape: Any
    loc: TopLoc_Location
    name: str
    orig_name: str
    color: Tuple[float, float, float]
    label: Optional[Any] = None  # debug / provenance
    step_entity_id: Optional[int] = None
    name_source: str = "xcaf"
    name_confidence: str = "exact"
    raw_step_name: Optional[str] = None


# ===============================================================
# 3) Small utilities
# ===============================================================
def sanitise_identifier(name: str) -> str:
    s = (name or "").strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "Unnamed"


def _step_representation_names(step_path: str) -> List[str]:
    """Return shape-representation names as a fallback for weak STEP metadata.

    Some exporters, including the CST AP242 file used by the toolbox example,
    attach useful names to shape representations but not to the XCAF labels
    produced by OpenCascade. In that case the labels look like ``0:1:1:2``.
    The representation order is stable for these flat part files and recovers
    names such as ``PATCH`` and ``Ports|port1``. Proper XCAF names always take
    precedence, particularly for hierarchical assemblies.
    """
    try:
        with open(step_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return []

    names = []
    for match in _REPRESENTATION_NAME_RE.finditer(text):
        name = match.group(1).replace("''", "'").strip()
        if "|" in name:
            name = name.rsplit("|", 1)[-1].strip()
        names.append(name)
    return names


def _is_generated_xcaf_name(name: str) -> bool:
    """Whether *name* is an OpenCascade label path rather than a CAD name."""
    return is_generated_name(name)


def _is_null_shape(shp: Any) -> bool:
    try:
        return shp is None or (hasattr(shp, "IsNull") and shp.IsNull())
    except Exception:
        return True


def attach_cad_metrics(
    p: Part,
    ok_bbox: bool,
    ok_props: bool,
    bb: Optional[Bnd_Box],
    vol: Optional[float],
    area: Optional[float],
    *,
    topology_counts: Optional[Dict[str, int]] = None,
    principal_moments: Optional[Tuple[float, float, float]] = None,
) -> None:
    """
    Attach CAD-derived metrics to the Part without touching p.label (which is a TDF_Label).
    Stores a JSON-safe dict in p.cad.
    """
    bbox_xyzxyz = None
    bbox_dims_xyz = None

    if ok_bbox and bb is not None:
        try:
            xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
            bbox_xyzxyz = (float(xmin), float(ymin), float(zmin), float(xmax), float(ymax), float(zmax))
            bbox_dims_xyz = (float(xmax - xmin), float(ymax - ymin), float(zmax - zmin))
        except Exception:
            bbox_xyzxyz = None
            bbox_dims_xyz = None
            ok_bbox = False

    p.cad = {
        "ok_bbox": bool(ok_bbox),
        "ok_props": bool(ok_props),
        "bbox_xyzxyz": bbox_xyzxyz,  # (xmin,ymin,zmin,xmax,ymax,zmax) or None
        "bbox_dims_xyz": bbox_dims_xyz,  # (dx,dy,dz) or None
        "vol_m3": (float(vol) if vol is not None else None),
        "area_m2": (float(area) if area is not None else None),
        "topology_counts": topology_counts,
        "principal_moments_m5": principal_moments,
    }


def _shape_topology_counts(shape: Any) -> Dict[str, int]:
    """Return rotation-independent topological counts for grouping hints."""
    topology = TopologyExplorer(shape)
    return {
        "solids": sum(1 for _ in topology.solids()),
        "shells": sum(1 for _ in topology.shells()),
        "faces": sum(1 for _ in topology.faces()),
        "edges": sum(1 for _ in topology.edges()),
        "vertices": sum(1 for _ in topology.vertices()),
    }


def _principal_moments(properties: GProp_GProps, volume: Optional[float]) -> Optional[Tuple[float, float, float]]:
    """Return sorted principal volume moments about the centre of mass."""
    if volume is None or volume <= 0:
        return None
    matrix = properties.MatrixOfInertia()
    values = np.asarray(
        [[float(matrix.Value(row, column)) for column in range(1, 4)] for row in range(1, 4)],
        dtype=np.float64,
    )
    # OpenCascade's matrix is symmetric; symmetrising suppresses insignificant
    # round-off before the Hermitian eigensolver is used.
    moments = np.linalg.eigvalsh(0.5 * (values + values.T))
    moments = np.maximum(moments, 0.0)
    return tuple(float(value) for value in moments)


def get_label_name(label: Any) -> str:
    """Human-readable label name via TDataStd_Name, else ''."""
    try:
        attr = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID(), attr):
            return str(attr.Get()).strip()
    except Exception:
        pass
    return ""


def create_xcaf_document() -> Tuple[Any, TDocStd_Document]:
    """Create and initialise an XCAF document."""
    app = XCAFApp_Application.GetApplication()
    doc = TDocStd_Document("MDTV-XCAF")
    app.InitDocument(doc)
    return app, doc


def _tri_area2(a: Tuple[float, float, float], b: Tuple[float, float, float], c: Tuple[float, float, float]) -> float:
    """Squared magnitude of (AB x AC). Proportional to (2*area)^2."""
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c

    abx, aby, abz = (bx - ax), (by - ay), (bz - az)
    acx, acy, acz = (cx - ax), (cy - ay), (cz - az)

    cxp = aby * acz - abz * acy
    cyp = abz * acx - abx * acz
    czp = abx * acy - aby * acx
    return cxp * cxp + cyp * cyp + czp * czp


def compact_mesh(
    verts: List[Tuple[float, float, float]], tris: List[Tuple[int, int, int]]
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    """Remove unused vertices and remap indices to dense 0..N-1."""
    if not tris:
        return [], []

    used = set()
    for i1, i2, i3 in tris:
        used.add(i1)
        used.add(i2)
        used.add(i3)

    used_sorted = sorted(used)
    remap = {old_i: new_i for new_i, old_i in enumerate(used_sorted)}

    new_verts = [verts[i] for i in used_sorted]
    new_tris = [(remap[i1], remap[i2], remap[i3]) for (i1, i2, i3) in tris]
    return new_verts, new_tris


def tessellate_shape(
    shape: Any,
    cfg: ParserConfig,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    """Tessellate TopoDS_Shape -> (verts, tris) using cfg tessellation settings."""
    if _is_null_shape(shape):
        return [], []

    mesh = BRepMesh_IncrementalMesh(
        shape,
        cfg.linear_deflection,
        cfg.is_relative_deflection,
        cfg.angular_deflection,
    )
    mesh.Perform()
    if not mesh.IsDone():
        return [], []

    verts: List[Tuple[float, float, float]] = []
    tris: List[Tuple[int, int, int]] = []

    topo = TopologyExplorer(shape)
    for face in topo.faces():
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None:
            continue

        trsf = loc.Transformation()
        start = len(verts)

        # vertices
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i).Transformed(trsf)
            verts.append((p.X(), p.Y(), p.Z()))

        # triangles
        for i in range(1, tri.NbTriangles() + 1):
            t = tri.Triangle(i).Get()
            i1 = start + t[0] - 1
            i2 = start + t[1] - 1
            i3 = start + t[2] - 1

            a, b, c = verts[i1], verts[i2], verts[i3]
            if _tri_area2(a, b, c) <= cfg.eps_area2:
                continue
            tris.append((i1, i2, i3))

    if not tris:
        return [], []

    if cfg.compact_tessellation:
        verts, tris = compact_mesh(verts, tris)

    return verts, tris


def topology_vertices(shape: Any) -> List[Tuple[float, float, float]]:
    """Return unique topological vertices for point and wire CAD markers."""
    if _is_null_shape(shape):
        return []
    points = []
    seen = set()
    for vertex in TopologyExplorer(shape).vertices():
        point = BRep_Tool.Pnt(vertex)
        value = (float(point.X()), float(point.Y()), float(point.Z()))
        key = tuple(round(coordinate, 15) for coordinate in value)
        if key not in seen:
            seen.add(key)
            points.append(value)
    return points


def get_best_color(color_tool: Any, labels_to_try: List[Any], default=(0.8, 0.8, 0.8)) -> Tuple[float, float, float]:
    """Colour inheritance: try Surf then Gen across labels in order."""
    for lbl in labels_to_try:
        if lbl is None:
            continue
        try:
            for ct in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
                ok, c = color_tool.GetColor(lbl, ct)
                if ok:
                    return (c.Red(), c.Green(), c.Blue())
        except Exception:
            continue
    return default


# ===============================================================
# 4) Placement handling
# ===============================================================
def shape_for_ops(p: Part, cfg: ParserConfig) -> Any:
    """
    Return an operations-space (world-space) shape.

    In an XCAF assembly, part geometry is typically defined in local coordinates
    and the instance placement is stored as a TopLoc_Location on the occurrence.
    This function applies that placement exactly once.
    """
    shp = p.shape
    if _is_null_shape(shp):
        return shp

    if cfg.apply_occurrence_locations and (p.loc is not None) and (not p.loc.IsIdentity()):
        shp = BRepBuilderAPI_Transform(shp, p.loc.Transformation(), True).Shape()

    return shp


# ===============================================================
# 5) Traversal (XCAF assemblies + occurrences) with colour inheritance
# ===============================================================
def collect_shapes_recursive(
    shape_tool: Any,
    label: Any,
    out_list: List[Part],
    color_tool: Any,
    loc_accum: TopLoc_Location = TopLoc_Location(),
    name_accum: str = "",
    occ_label: Optional[Any] = None,
    parent_labels: Optional[List[Any]] = None,
) -> None:
    """
    Occurrence-aware traversal.

    Naming: occurrence label name wins (if present).
    Placement: accumulate reference locations.
    Colour: occurrence -> definition -> nearest parents.
    """
    if parent_labels is None:
        parent_labels = []

    # Reference / occurrence
    if shape_tool.IsReference(label):
        ref = TDF_Label()
        shape_tool.GetReferredShape(label, ref)

        occ_name = (get_label_name(label) or label.GetLabelName() or "").strip()
        new_name = occ_name or name_accum
        new_loc = loc_accum.Multiplied(shape_tool.GetLocation(label))

        collect_shapes_recursive(
            shape_tool=shape_tool,
            label=ref,
            out_list=out_list,
            color_tool=color_tool,
            loc_accum=new_loc,
            name_accum=new_name,
            occ_label=label,
            parent_labels=parent_labels + [label],
        )
        return

    # Assembly
    if shape_tool.IsAssembly(label):
        children = TDF_LabelSequence()
        shape_tool.GetComponents(label, children)

        asm_name = (get_label_name(label) or label.GetLabelName() or "").strip()
        if asm_name:
            name_accum = asm_name

        for i in range(children.Length()):
            child = children.Value(i + 1)
            collect_shapes_recursive(
                shape_tool=shape_tool,
                label=child,
                out_list=out_list,
                color_tool=color_tool,
                loc_accum=loc_accum,
                name_accum=name_accum,
                occ_label=occ_label,
                parent_labels=parent_labels + [label],
            )
        return

    # Leaf
    if shape_tool.IsSimpleShape(label):
        shape = shape_tool.GetShape(label)

        leaf_name = (get_label_name(label) or label.GetLabelName() or "").strip()
        name = name_accum or leaf_name or f"Part_{len(out_list) + 1}"

        labels_to_try = []
        if occ_label is not None:
            labels_to_try.append(occ_label)
        labels_to_try.append(label)
        labels_to_try.extend(reversed(parent_labels[-6:]))

        color = get_best_color(color_tool, labels_to_try, default=(0.8, 0.8, 0.8))

        out_list.append(
            Part(uid=next_uid(), shape=shape, loc=loc_accum, name=name, orig_name=name, color=color, label=label)
        )
        return

    # Rare fallback: still a shape
    if shape_tool.IsShape(label):
        shape = shape_tool.GetShape(label)
        nm = (name_accum or get_label_name(label) or label.GetLabelName() or "").strip() or f"Part_{len(out_list)+1}"

        labels_to_try = []
        if occ_label is not None:
            labels_to_try.append(occ_label)
        labels_to_try.append(label)
        labels_to_try.extend(reversed(parent_labels[-6:]))

        color = get_best_color(color_tool, labels_to_try, default=(0.8, 0.8, 0.8))

        out_list.append(
            Part(uid=next_uid(), shape=shape, loc=loc_accum, name=nm, orig_name=nm, color=color, label=label)
        )
        return


# ===============================================================
# 6) Load STEP
# ===============================================================
def load_step_with_hierarchy(step_path: str, cfg: ParserConfig) -> List[Part]:
    """Load STEP into XCAF, return leaf occurrences with accumulated placement & names."""
    _, doc = create_xcaf_document()
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())

    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    reader.SetLayerMode(True)

    if cfg.force_units_to_metres:
        Interface_Static.SetCVal("xstep.cascade.unit", "M")
        Interface_Static.SetCVal("read.scale.unit", "M")

    print(f"\n📂 Reading STEP → {os.path.basename(step_path)}")

    if reader.ReadFile(step_path) != IFSelect_RetDone:
        raise RuntimeError("❌ STEP read error.")
    if not reader.Transfer(doc):
        raise RuntimeError("❌ Transfer to XCAF failed.")

    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)

    parts: List[Part] = []
    for i in range(roots.Length()):
        collect_shapes_recursive(
            shape_tool=shape_tool,
            label=roots.Value(i + 1),
            out_list=parts,
            color_tool=color_tool,
            loc_accum=TopLoc_Location(),
            name_accum="",
        )

    metadata = StepMetadata.from_file(step_path)
    transfer_reader = reader.Reader().WS().TransferReader()
    interface_model = reader.Reader().WS().Model()

    for part in parts:
        resolved = None
        entity = None
        for mode in (1, 0):
            entity = transfer_reader.EntityFromShapeResult(part.shape, mode)
            if entity is not None:
                break
        if entity is not None:
            label = interface_model.StringLabel(entity)
            label_text = "" if label is None else label.ToCString()
            match = re.fullmatch(r"#(\d+)", label_text)
            if match:
                part.step_entity_id = int(match.group(1))
                resolved = metadata.resolve_name(part.step_entity_id)

        weak_xcaf_name = _is_generated_xcaf_name(part.name) or is_generic_cad_name(part.name)
        if not weak_xcaf_name:
            part.name_source = "xcaf"
            part.name_confidence = "exact"
            part.raw_step_name = part.name
        elif resolved is not None:
            part.name = resolved.name
            part.orig_name = resolved.name
            part.name_source = resolved.source_entity_type
            part.name_confidence = resolved.confidence
            part.raw_step_name = resolved.raw_name
        elif is_generic_cad_name(part.name):
            part.name_source = "xcaf_generic"
            part.name_confidence = "low"
            part.raw_step_name = part.name

    # Last-resort compatibility for flat files whose transferred shapes cannot
    # be linked back to STEP entities. This fallback is explicitly lower
    # confidence because it uses representation order.
    fallback_names = _step_representation_names(step_path)
    if len(fallback_names) == len(parts):
        for part, fallback_name in zip(parts, fallback_names):
            if fallback_name and _is_generated_xcaf_name(part.name):
                part.name = semantic_leaf_name(fallback_name)
                part.orig_name = part.name
                part.name_source = "representation_order_fallback"
                part.name_confidence = "low"
                part.raw_step_name = fallback_name
    return parts


# ===============================================================
# 7) Explode multi-solid leaves (in ops space)
# ===============================================================
def explode_to_solids(parts: List[Part], ops_shape: Dict[int, Any]) -> List[Part]:
    """Explode any leaf whose ops-shape contains >1 solid. Returns new Parts in ops space."""
    new_parts: List[Part] = []
    for p in parts:
        shp = ops_shape.get(p.uid)
        if _is_null_shape(shp):
            new_parts.append(p)
            continue

        solids = list(TopologyExplorer(shp).solids())
        if len(solids) <= 1:
            new_parts.append(p)
            continue

        base = (p.name or "Unnamed").strip()
        for i, s in enumerate(solids, 1):
            new_parts.append(
                Part(
                    uid=next_uid(),
                    shape=s,  # already ops space
                    loc=TopLoc_Location(),  # identity
                    name=f"{base} #S{i}",
                    orig_name=p.orig_name,
                    color=p.color,
                    label=p.label,
                )
            )
    return new_parts


def disambiguate_duplicate_names(parts: List[Part]) -> None:
    """Make names unique (stable, deterministic, identifier-safe)."""
    counts = defaultdict(int)
    for p in parts:
        base = (p.name or "").strip() or "Unnamed"
        counts[base] += 1
        if counts[base] > 1:
            p.name = f"{base}_{counts[base]}"


# ===============================================================
# 8) Optional STL export (simple concat)
# ===============================================================
def export_combined_stl(parts: List[Part], ops_shape: Dict[int, Any], filename: str, cfg: ParserConfig) -> None:
    """ASCII STL export for quick debugging (not production)."""
    all_vertices: List[Tuple[float, float, float]] = []
    all_triangles: List[Tuple[int, int, int]] = []
    offset = 0

    for p in parts:
        shp = ops_shape.get(p.uid)
        verts, tris = tessellate_shape(shp, cfg)
        if not tris:
            continue
        all_vertices.extend(verts)
        all_triangles.extend([(i1 + offset, i2 + offset, i3 + offset) for (i1, i2, i3) in tris])
        offset += len(verts)

    if not all_triangles:
        print("⚠️ No triangles → STL not written.")
        return

    with open(filename, "w", encoding="utf-8") as f:
        f.write("solid assembly\n")
        for i1, i2, i3 in all_triangles:
            a, b, c = all_vertices[i1], all_vertices[i2], all_vertices[i3]
            f.write("  facet normal 0 0 0\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {a[0]} {a[1]} {a[2]}\n")
            f.write(f"      vertex {b[0]} {b[1]} {b[2]}\n")
            f.write(f"      vertex {c[0]} {c[1]} {c[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid assembly\n")

    print(f"💾 Exported STL → {filename}")


# ===============================================================
# 9) Rotation utilities
# ===============================================================
def _bbox_center_pnt(bb: Bnd_Box) -> gp_Pnt:
    xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
    return gp_Pnt(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))


def _build_rotation_trsf(pivot: gp_Pnt, rx_deg: float, ry_deg: float, rz_deg: float, order: str = "xyz") -> gp_Trsf:
    def _rot(axis_dir: gp_Dir, ang_deg: float) -> gp_Trsf:
        t = gp_Trsf()
        t.SetRotation(gp_Ax1(pivot, axis_dir), float(np.deg2rad(ang_deg)))
        return t

    angles = {"x": rx_deg, "y": ry_deg, "z": rz_deg}
    dirs = {"x": gp_Dir(1, 0, 0), "y": gp_Dir(0, 1, 0), "z": gp_Dir(0, 0, 1)}

    R = gp_Trsf()  # <-- default is identity in pythonOCC

    for ax in order.lower():
        ang = float(angles[ax])
        if ang != 0.0:
            R = _rot(dirs[ax], ang).Multiplied(R)  # compose

    return R


# ===============================================================
# 10) Main pipeline
# ===============================================================
def main(step_path: str, cfg: ParserConfig) -> List[Part]:
    """
    STEP-to-geometry parsing pipeline (XCAF-aware).

    This routine:
      1) Loads the STEP file into an XCAF document and traverses the assembly tree to
         collect *leaf occurrences* (instance-aware parts) with their accumulated placements.
      2) Constructs a single, canonical "operations-space" shape per part, where placements
         are applied exactly once (when required), ensuring consistent downstream geometry.
      3) Optionally decomposes any leaf shape that contains multiple solids into separate
         per-solid parts, preserving naming provenance and ensuring one solid per Part.
      4) Computes CAD-derived metrics (bounding boxes, volumes, surface areas) in a single pass
         and caches them for deterministic reporting and to avoid repeated OCCT evaluations.
      5) Tessellates parts on demand and caches triangle soups for performance, enabling
         voxelisation without recomputation.

    Returns:
      A list of Part objects with stable names, colours (best-effort), and shapes suitable for
      tessellation and voxelisation.
    """
    try:
        reset_uid_counter(1)
        # ---------------------------------------------------------------------
        # 1) Read STEP and collect leaf occurrences (instance-aware traversal)
        # ---------------------------------------------------------------------
        parts = load_step_with_hierarchy(step_path, cfg)
        print(f"📦 Found {len(parts)} leaf occurrences")

        print(
            "📌 Placement mode:",
            "APPLY OCCURRENCE LOCATIONS" if cfg.apply_occurrence_locations else "DO NOT APPLY LOCATIONS",
        )

        # Canonical operations-space shape map (single source of truth for geometry)
        ops_shape: Dict[int, Any] = {p.uid: shape_for_ops(p, cfg) for p in parts}

        # ---------------------------------------------------------------------
        # 2) Optional: split multi-solid leaf shapes into separate solids
        # ---------------------------------------------------------------------
        if cfg.auto_explode_solids:
            before = len(parts)
            parts = explode_to_solids(parts, ops_shape=ops_shape)
            after = len(parts)
            if after > before:
                print(f"🧩 Exploded {after - before} additional solids → {after} solids total")
                ops_shape = {p.uid: p.shape for p in parts}  # exploded shapes are already ops-space
            else:
                print(f"🧩 No multi-solid leaf shapes detected → {after} solids total (no explode)")
        else:
            print(f"🧩 Explode disabled → {len(parts)} solids total")

        # 1) sanitise ONLY the working identifier
        for p in parts:
            p.name = sanitise_identifier(p.name)

        # 2) ensure uniqueness AFTER sanitising
        disambiguate_duplicate_names(parts)

        print("\n🔎 Part names:")
        for p in parts:
            print(p.name)

        # ---------------------------------------------------------------------
        # 2.5) Optional: global rotation (apply to ops-space shapes BEFORE metrics/tessellation)
        # ---------------------------------------------------------------------
        if cfg.enable_global_rotation:
            rx, ry, rz = cfg.rotate_deg

            # Choose pivot
            if cfg.rotate_about == "bbox_global":
                bb_global = Bnd_Box()
                any_ok = False
                for shp in ops_shape.values():
                    if _is_null_shape(shp):
                        continue
                    bb_tmp = Bnd_Box()
                    try:
                        brepbndlib.Add(shp, bb_tmp)
                    except Exception:
                        continue
                    if not bb_tmp.IsVoid():
                        bb_global.Add(bb_tmp)
                        any_ok = True

                if (not any_ok) or bb_global.IsVoid():
                    raise RuntimeError("Cannot rotate: global bbox is void (no valid shapes).")

                pivot = _bbox_center_pnt(bb_global)

            elif cfg.rotate_about == "origin":
                pivot = gp_Pnt(0.0, 0.0, 0.0)

            else:
                raise ValueError("cfg.rotate_about must be 'bbox_global' or 'origin'")

            # Build rotation transform (Rx then Ry then Rz by default)
            R = _build_rotation_trsf(pivot, rx, ry, rz, order=cfg.rotate_order)

            # Apply to every ops-space shape
            for uid, shp in list(ops_shape.items()):
                if _is_null_shape(shp):
                    continue
                ops_shape[uid] = BRepBuilderAPI_Transform(shp, R, True).Shape()

            # bake rotated ops-space shapes back into Parts so later voxelisation uses them
            for p in parts:
                shp = ops_shape.get(p.uid)
                if _is_null_shape(shp):
                    continue
                p.shape = shp
                p.loc = TopLoc_Location()  # identity (prevents re-applying placement/transform)

            # keep a single source of truth after rotation
            ops_shape = {p.uid: p.shape for p in parts}

            print(
                f"🔄 Applied global rotation rotate_deg={cfg.rotate_deg} about {cfg.rotate_about} (order={cfg.rotate_order})"
            )

        # ---------------------------------------------------------------------
        # 3) CAD-derived metrics pass (bbox, volume, surface) with caching
        # ---------------------------------------------------------------------
        cad_cache: Dict[int, Dict[str, Any]] = {}

        total_box = Bnd_Box()
        total_vol = GProp_GProps()
        total_area = GProp_GProps()
        empty_count = 0
        added_any = False

        for p in parts:
            shp = ops_shape.get(p.uid)

            if _is_null_shape(shp):
                cad_cache[p.uid] = {"ok_bbox": False, "ok_props": False, "bb": None, "vol": None, "area": None}
                attach_cad_metrics(p, ok_bbox=False, ok_props=False, bb=None, vol=None, area=None)
                empty_count += 1
                continue

            bb = Bnd_Box()
            try:
                brepbndlib.Add(shp, bb)
            except Exception:
                cad_cache[p.uid] = {"ok_bbox": False, "ok_props": False, "bb": None, "vol": None, "area": None}
                attach_cad_metrics(p, ok_bbox=False, ok_props=False, bb=None, vol=None, area=None)
                empty_count += 1
                continue

            if bb.IsVoid():
                cad_cache[p.uid] = {"ok_bbox": False, "ok_props": False, "bb": None, "vol": None, "area": None}
                attach_cad_metrics(p, ok_bbox=False, ok_props=False, bb=None, vol=None, area=None)
                empty_count += 1
                continue

            # Union using per-part bounding boxes (robust global bbox accumulation)
            total_box.Add(bb)
            added_any = True

            vol = area = None
            volume_properties = None
            ok_props = False
            try:
                pv, pa = GProp_GProps(), GProp_GProps()
                brepgprop.VolumeProperties(shp, pv)
                brepgprop.SurfaceProperties(shp, pa)
                vol = pv.Mass()
                area = pa.Mass()
                volume_properties = pv
                total_vol.Add(pv)
                total_area.Add(pa)
                ok_props = True
            except Exception:
                ok_props = False

            cad_cache[p.uid] = {"ok_bbox": True, "ok_props": ok_props, "bb": bb, "vol": vol, "area": area}

            attach_cad_metrics(
                p,
                ok_bbox=True,
                ok_props=ok_props,
                bb=bb,
                vol=vol,
                area=area,
                topology_counts=_shape_topology_counts(shp),
                principal_moments=(
                    _principal_moments(volume_properties, vol)
                    if volume_properties is not None
                    else None
                ),
            )

        # ---------------------------------------------------------------------
        # 4) Tessellation cache (computed on demand; reused for totals and exports)
        # ---------------------------------------------------------------------
        tri_cache: Dict[int, Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]] = {}

        def get_tris(p: Part):
            uid = p.uid
            if uid not in tri_cache:
                shp = ops_shape.get(uid)
                tri_cache[uid] = ([], []) if _is_null_shape(shp) else tessellate_shape(shp, cfg)
            return tri_cache[uid]

        # ---------------------------------------------------------------------
        # 5) Per-part reporting (CAD metrics + tessellation statistics)
        # ---------------------------------------------------------------------
        if cfg.verbose_per_part:
            for p in parts:
                info = cad_cache.get(p.uid, {})
                if not info.get("ok_bbox", False):
                    print(f"\n🧩 {p.name}\n⚠️ Invalid/empty shape or bbox failed, skipping.")
                    continue

                bb: Bnd_Box = info["bb"]
                xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()

                print(f"\n🧩 {p.name}")
                print(f"📦 BBox: X[{xmin:.5f},{xmax:.5f}] Y[{ymin:.5f},{ymax:.5f}] Z[{zmin:.5f},{zmax:.5f}]")

                if info.get("ok_props", False):
                    print(f"📐 Volume {info['vol']:.6e} m³, Surface {info['area']:.6e} m², Color={p.color}")
                else:
                    print(f"📐 Volume/Surface: ⚠️ failed, Color={p.color}")

                verts, tris = get_tris(p)
                if not tris:
                    print("⚠️ Tessellation produced 0 triangles (voxelisation may fail).")
                print(f"🔺 Tessellated → {len(tris)} triangles, {len(verts)} vertices")

        # ---------------------------------------------------------------------
        # 6) Overall assembly metrics (CAD-derived)
        # ---------------------------------------------------------------------
        print("\n📊 ===== Overall Assembly Summary (CAD) =====")
        if not added_any or total_box.IsVoid():
            print("⚠️ Global bbox is void (no valid shapes were accumulated).")
        else:
            xmin, ymin, zmin, xmax, ymax, zmax = total_box.Get()
            print(f"📦 Global BBox: X[{xmin:.5f},{xmax:.5f}] Y[{ymin:.5f},{ymax:.5f}] Z[{zmin:.5f},{zmax:.5f}]")

        vol_m3 = total_vol.Mass()
        print(f"📐 Total Volume ≈ {vol_m3:.6e} m³")
        print(f"📐 Total Volume ≈ {vol_m3 * 1e9:.3f} mm³ (sanity)")
        print(f"🧱 Total Surface ≈ {total_area.Mass():.6e} m²")
        if empty_count:
            print(f"⚠️ Empty/invalid shapes skipped: {empty_count}")
        print("============================================")

        # ---------------------------------------------------------------------
        # 7) Overall tessellation totals (from cached per-part triangle soups)
        # ---------------------------------------------------------------------
        total_tris = 0
        total_verts = 0
        zero_tri_count = 0

        for p in parts:
            if not cad_cache.get(p.uid, {}).get("ok_bbox", False):
                continue
            verts, tris = get_tris(p)
            if not tris:
                zero_tri_count += 1
            total_tris += len(tris)
            total_verts += len(verts)

        print("\n📊 ===== Tessellation Summary =====")
        print(f"🔺 Total Tessellation → {total_tris} triangles, {total_verts} vertices")
        if zero_tri_count:
            print(f"⚠️ Parts with 0 triangles: {zero_tri_count}")
        print("==================================")

        # ---------------------------------------------------------------------
        # 8) Optional: combined STL export (debug convenience)
        # ---------------------------------------------------------------------
        if cfg.export_stl:
            out_path = os.path.join(os.path.dirname(step_path), cfg.stl_outname)
            export_combined_stl(parts, ops_shape, out_path, cfg)

        print("\n✅ Completed parsing.")
        return parts

    except Exception as e:
        print("❌ Error:", e)
        raise RuntimeError(f"STEP parsing failed for {step_path}") from e
