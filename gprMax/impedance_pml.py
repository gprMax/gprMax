"""Longitudinal CPU PML coupling for uniformly extruded impedance surfaces.

For a wall extruded along a PML stretch direction, the longitudinal H
circulation has the same geometric retained fraction as the electric area.
This identity is independent of the retained materials' constitutive laws.
The ordinary PML derivative and convolution histories therefore remain valid.
The sparse system captures the PML increment as an additional circulation,
preserving the original electric field for its implicit boundary/ADE solve.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from gprMax.materials import Material


class _ImpedancePMLHoldMaterial(Material):
    """Hold the bulk update and expose the PML derivative for sparse capture."""

    def __init__(self, numID: int, er: float):
        super().__init__(numID, f"__impedance_pml_hold_{numID}")
        self.type = "impedance-surface-hold"
        self.impedance_role = "surface-hold"
        self.averagable = False
        self.er = er

    def calculate_update_coeffsE(self, grid):
        self.CA = 1.0
        self.CBx = self.CBy = self.CBz = 0.0
        # E is temporarily zero during the electric PML phase. Unit srce
        # exposes the signed stretched-derivative correction without mixing
        # in the old E or the boundary load; the sparse owner restores E.
        self.srce = 1.0

    def calculate_update_coeffsH(self, grid):
        self.DA = self.DBx = self.DBy = self.DBz = self.srcm = 0.0


def _edge_cells(grid, component, coordinate):
    """Physical quarter-area cells incident on an electric Yee edge."""
    transverse = [axis for axis in range(3) if axis != component]
    for offsets in product((-1, 0), repeat=2):
        cell = list(coordinate)
        for axis, offset in zip(transverse, offsets):
            cell[axis] += offset
        if all(0 <= value < size for value, size in zip(cell, grid.solid.shape)):
            yield tuple(cell)


def _slab_bounds(pml):
    return np.asarray((pml.xs, pml.ys, pml.zs)), np.asarray((pml.xf, pml.yf, pml.zf))


def prepare_impedance_pml(grid, system):
    """Validate SIBC/PML overlap and install PML-aware electric hold rows.

    Call after sparse SIBC compilation and before material coefficient-table
    allocation. It also accepts an extruded virtual-guide sparse system when
    its grid retains the original solid marker IDs and marker-model mapping.
    Only affected edges are changed. Passive constant and Foster loads,
    including exact zero-admittance PMC, use the same captured forcing.
    Loss and bulk polarization remain owned by the compiled sparse edge;
    its denominator and history corrections include every retained quadrant.

    Internal slab upper transverse bounds must extend into the excluded
    volume: existing CPU kernels use half-open transverse bounds for both
    E and H. Ending the slab on the upper wall plane would omit that plane's
    PML corrections.
    """
    slabs = grid.pmls["slabs"]
    if not slabs or not system.edge_count:
        system.pml_edge_indices = np.empty(0, dtype=np.int32)
        system.pml_edge_area = np.empty(0, dtype=system.edge_runtime.dtype)
        system.pml_source_coeff = np.empty(0, dtype=system.edge_runtime.dtype)
        system.pml_edge_scale = np.empty(0, dtype=system.edge_runtime.dtype)
        system.pml_edge_count = 0
        return

    markers = set(grid.impedance_marker_models)
    bounds = [_slab_bounds(pml) for pml in slabs]
    extrusion_checked = set()
    holds = {
        material.er: material
        for material in grid.materials
        if isinstance(material, _ImpedancePMLHoldMaterial)
    }
    affected, areas = [], []
    spacing = (grid.dx, grid.dy, grid.dz)
    for edge_index, edge in enumerate(system.edge_info):
        component, i, j, k, _, _, port_start, port_count = map(int, edge)
        coordinate = np.asarray((i, j, k))
        overlaps = [
            (index, pml)
            for index, pml in enumerate(slabs)
            if np.all(coordinate >= bounds[index][0]) and np.all(coordinate <= bounds[index][1])
        ]
        if not overlaps:
            continue

        port_slice = slice(port_start, port_start + port_count)
        models = system.port_info[port_slice, 0]
        declared = getattr(grid, "surface_impedance_models", {})
        if any(
            declared[system.model_ids[int(index)]].allow_active
            for index in models
            if system.model_ids[int(index)] in declared
        ):
            raise ValueError("surface-impedance walls inside PML require passive models")

        cells = tuple(_edge_cells(grid, component, coordinate))
        retained = [
            grid.materials[int(grid.solid[cell])]
            for cell in cells
            if int(grid.solid[cell]) not in markers
        ]
        if not retained or any(
            host.is_pec
            or host.is_pmc
            or host.directional_materials is not None
            or not np.isfinite(host.se)
            or host.se < 0
            or not np.isfinite(host.sm)
            or host.sm < 0
            or not np.isfinite(host.er)
            or host.er <= 0
            or not np.isfinite(host.mr)
            or host.mr <= 0
            for host in retained
        ):
            raise ValueError(
                "SIBC edges inside PML require isotropic retained host materials "
                "with finite nonnegative conductivities and positive finite er and mr"
            )

        for index, pml in overlaps:
            stretch = "xyz".index(pml.direction[0])
            if np.any(system.port_normal[port_slice, 0] == stretch):
                raise ValueError(
                    "a surface-impedance wall inside PML must be tangent to the PML "
                    "absorption direction and continue through the slab"
                )
            lower, upper = bounds[index]
            if component != stretch and any(
                coordinate[axis] == upper[axis] for axis in range(3) if axis != stretch
            ):
                raise ValueError(
                    "extend the PML upper transverse bound at least one "
                    "cell into the excluded impedance volume so its electric "
                    "and magnetic wall samples receive PML corrections"
                )

            # Include the entrance neighbour and terminal neighbour when they
            # exist: both backward-E and forward-H stencils need extrusion.
            start = max(0, int(lower[stretch]) - 1)
            stop = min(grid.solid.shape[stretch], int(upper[stretch]) + 1)
            for cell in cells:
                key = (index, *(cell[axis] for axis in range(3) if axis != stretch))
                if key in extrusion_checked:
                    continue
                section = list(cell)
                section[stretch] = slice(start, stop)
                line = grid.solid[tuple(section)]
                if not np.all(line == line[0]):
                    raise ValueError(
                        "surface-impedance walls and their retained host must be "
                        "uniformly extruded through the PML and its "
                        "neighbouring stencil cells"
                    )
                extrusion_checked.add(key)

        # This private material only captures the curl correction (srce=1).
        # Never put an effective conductivity or bulk poles here: the sparse
        # row already owns their area-weighted masses and histories. Retain
        # the quarter-area mean epsilon-infinity as descriptive material data.
        effective_er = float(np.mean([host.er for host in retained]))
        material = holds.get(effective_er)
        if material is None:
            material = _ImpedancePMLHoldMaterial(len(grid.materials), effective_er)
            grid.materials.append(material)
            holds[effective_er] = material
        grid.ID[component, i, j, k] = material.numID
        affected.append(edge_index)
        areas.append(
            float(system.edge_fraction[edge_index])
            * spacing[(component + 1) % 3]
            * spacing[(component + 2) % 3]
        )

    system.pml_edge_indices = np.asarray(affected, dtype=np.int32)
    system.pml_edge_area = np.asarray(areas, dtype=system.edge_runtime.dtype)
    system.pml_source_coeff = np.ones(len(areas), dtype=system.edge_runtime.dtype)
    system.pml_edge_scale = system.pml_edge_area.copy()
    system.pml_edge_count = len(affected)
