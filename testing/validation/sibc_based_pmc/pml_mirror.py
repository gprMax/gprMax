"""Independent mirrored-grid validation of PMC walls crossing CPU PML.

Run ``python -m testing.validation.sibc_based_pmc.pml_mirror`` from the
repository root. The reference has no PMC material or sparse edge update.
Random initial fields obey PMC image parity, exciting all six components.
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np

import gprMax
from gprMax.cython.fields_updates_normal import update_electric, update_magnetic
from gprMax.impedance_surfaces import _component_valid_view
from gprMax.pml import CFS
from gprMax.updates.cpu_updates import CPUUpdates

from .runtime import assert_zero_admittance, build_grid
from .validate import ETA, fields


def advance_pml(grid):
    arrays = fields(grid)
    update_magnetic(grid.nx, grid.ny, grid.nz, -1, 1, grid.updatecoeffsH, grid.ID, *arrays)
    for slab in grid.pmls["slabs"]:
        slab.update_magnetic()
    update_electric(grid.nx, grid.ny, grid.nz, -1, 1, grid.updatecoeffsE, grid.ID, *arrays)
    capture = (
        grid.impedance_surfaces.capture_electric_pml(grid)
        if grid.impedance_surfaces is not None
        else nullcontext()
    )
    with capture:
        for slab in grid.pmls["slabs"]:
            slab.update_electric()
    if grid.impedance_surfaces is not None:
        grid.impedance_surfaces.update(grid)


def pml_image_case(
    cache,
    stretch=0,
    precision="double",
    formulation="HORIPML",
    order=1,
    steps=200,
    disable_coupling=False,
    host_kind=None,
    layered=False,
):
    """Compare the exact PMC half with its independent full-domain image."""
    normal = (stretch + 1) % 3
    cells, spacing = np.array([10, 10, 10]), np.array([0.001, 0.0013, 0.0017])
    cells[normal] = 12
    wall = cells[normal] // 2
    thickness = [0] * 6
    thickness[stretch] = thickness[stretch + 3] = 3
    scenes = []
    for is_pmc in (False, True):
        scene = gprMax.Scene()
        for obj in (
            gprMax.Domain(p1=tuple(cells * spacing)),
            gprMax.Discretisation(p1=tuple(spacing)),
            gprMax.TimeStepStabilityFactor(f=0.99),
            gprMax.TimeWindow(iterations=steps),
            gprMax.PMLThickness(thickness=tuple(thickness)),
            gprMax.OMPThreads(1),
        ):
            scene.add(obj)
        for axis in range(3):
            if axis != stretch:
                for side in ("0", "max"):
                    scene.add(gprMax.SymmetryBoundary(face="xyz"[axis] + side, type="pec"))
        if host_kind is not None:
            from testing.validation.impedance_surface.validate_sibc_pml import add_bulk_host

            add_bulk_host(scene, host_kind)
            upper = cells * spacing
            if layered:
                transverse = (stretch + 2) % 3
                upper[transverse] *= 0.5
            scene.add(gprMax.Box(p1=(0, 0, 0), p2=tuple(upper), material_id="host"))
        if is_pmc:
            scene.add(gprMax.SurfaceImpedance(id="wall", resistance=np.inf))
            lower = np.zeros(3)
            lower[normal] = wall * spacing[normal]
            scene.add(
                gprMax.Box(
                    p1=tuple(lower), p2=tuple(cells * spacing), material_id="wall", averaging="n"
                )
            )
        scenes.append(scene)
    label = f"{stretch}_{precision}_{formulation}_{order}_{host_kind}_{layered}"
    full = build_grid(scenes[0], Path(cache) / f"pml_full_{label}", precision)
    half = build_grid(scenes[1], Path(cache) / f"pml_pmc_{label}", precision)
    system = assert_zero_admittance(half)
    assert full.dt == half.dt and system.pml_edge_count > 0
    if disable_coupling:
        for material in half.materials:
            if material.ID.startswith("__impedance_pml_hold_"):
                half.updatecoeffsE[material.numID, 4] = 0

    # Use identical explicitly configured coefficients on both grids. This
    # exercises the existing one/two-pole kernels without material averaging
    # of the opaque half affecting the PML profile's automatic sigma choice.
    for grid in (full, half):
        for slab in grid.pmls["slabs"]:
            slab.formulation = formulation
            slab.CFS = [CFS() for _ in range(order)]
            for index, cfs in enumerate(slab.CFS):
                cfs.alpha.max = 0.02 * (index + 1)
                if formulation == "HORIPML" and order == 2 and index == 1:
                    # This test measures mirror/coupling equivalence, so use
                    # a shifted product rather than the now-rejected unstable
                    # near-unshifted double profile.
                    cfs.alpha.max = 20.0
                cfs.kappa.min = 1 / order if formulation == "MRIPML" else 1
                cfs.kappa.max = cfs.kappa.min * (1.2 + 0.1 * index)
            slab.initialise_field_arrays()
            slab.calculate_update_coeffs(1.0, 1.0)

    rng, masks = np.random.default_rng(927 + stretch), []
    for component, (f, h) in enumerate(zip(fields(full), fields(half))):
        view = _component_valid_view(f, component, full)
        indices = np.indices(view.shape)
        offset = int(component == normal) if component < 3 else int(component - 3 != normal)
        retained = indices[normal] + 0.5 * offset <= wall
        view[:] = rng.standard_normal(view.shape) / (ETA if component >= 3 else 1)
        for axis in range(3):
            constrained = axis != component if component < 3 else axis == component - 3
            if constrained:
                view[(indices[axis] == 0) | (indices[axis] == cells[axis])] = 0
        for position in range(view.shape[normal]):
            if position + 0.5 * offset <= wall:
                continue
            target, source = [slice(None)] * 3, [slice(None)] * 3
            target[normal] = position
            source[normal] = 2 * wall - position - offset
            view[tuple(target)] = (-1 if offset else 1) * view[tuple(source)]
        _component_valid_view(h, component, half)[retained] = view[retained]
        masks.append(retained)
    norm = np.sqrt(
        sum(
            np.sum((_component_valid_view(f, c, full)[mask] * (ETA if c >= 3 else 1)) ** 2)
            for c, (f, mask) in enumerate(zip(fields(full), masks))
        )
    )
    errors, history_errors = [], []
    updates = [CPUUpdates(grid) for grid in (full, half)]
    for update in updates:
        if update.grid.maxpoles:
            update.set_dispersive_updates()
    for _ in range(steps):
        for update in updates:
            update.update_magnetic()
            update.update_magnetic_pml()
            update.update_electric_a()
            update.update_electric_pml()
            update.update_electric_b()
            update.update_impedance_surfaces()
        error = sum(
            np.sum(
                (
                    (
                        _component_valid_view(f, c, full)[mask]
                        - _component_valid_view(h, c, half)[mask]
                    )
                    * (ETA if c >= 3 else 1)
                )
                ** 2
            )
            for c, (f, h, mask) in enumerate(zip(fields(full), fields(half), masks))
        )
        errors.append(float(np.sqrt(error) / norm))
        transverse = [axis for axis in range(3) if axis != stretch]
        for reference_slab, pmc_slab in zip(full.pmls["slabs"], half.pmls["slabs"]):
            for family in ("E", "H"):
                for number, axis in enumerate(transverse, start=1):
                    first = getattr(reference_slab, f"{family}Phi{number}")
                    second = getattr(pmc_slab, f"{family}Phi{number}")
                    offset = int(axis == normal) if family == "E" else int(axis != normal)
                    retained_indices = np.flatnonzero(
                        np.arange(first.shape[normal + 1]) + 0.5 * offset <= wall
                    )
                    first = np.take(first, retained_indices, axis=normal + 1)
                    second = np.take(second, retained_indices, axis=normal + 1)
                    denominator = max(np.linalg.norm(first), 1e-20)
                    history_errors.append(float(np.linalg.norm(first - second) / denominator))
    limit = 3e-5 if precision == "single" else 3e-12
    return dict(
        stretch_axis="xyz"[stretch],
        wall_normal="xyz"[normal],
        precision=precision,
        formulation=formulation,
        order=order,
        steps=steps,
        maximum_field_relative_error=max(errors),
        limit=limit,
        maximum_pml_history_relative_error=max(history_errors),
        pmc_pml_edge_count=system.pml_edge_count,
        coupling_disabled=disable_coupling,
        passed=max(errors) < limit and max(history_errors) < limit,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results" / "pml_mirror.json"
    )
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args(argv)
    cache = args.output.parent.parent / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    for axis in range(3):
        for precision in ("double", "single"):
            for formulation in ("HORIPML", "MRIPML"):
                for order in (1, 2):
                    case = pml_image_case(cache, axis, precision, formulation, order, args.steps)
                    cases.append(case)
                    print(json.dumps(case), flush=True)
    negative = pml_image_case(cache, steps=args.steps, disable_coupling=True)
    report = dict(
        cases=cases,
        disabled_coupling_control=negative,
        passed=all(case["passed"] for case in cases) and not negative["passed"],
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
