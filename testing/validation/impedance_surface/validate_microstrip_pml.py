"""Copper microstrip over a lossy/dispersive substrate continued into PML.

Compare a short line with a causally longer line, with identical transverse
geometry. Run with ``python -m testing.validation.impedance_surface.validate_microstrip_pml``.
The transverse domain is fixed in both runs; this measures longitudinal PML
reflection, not open-space radiation or convergence of a microstrip model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import c

import gprMax
from gprMax.updates.cpu_updates import CPUUpdates
from .validate_sibc_pml import DL, add_bulk_host, advance, build_grid, field_norm


def microstrip_scene(length, host_kind, axis=2):
    from .validate_sibc_pml import xyz

    scene = gprMax.Scene()
    thickness = [0] * 6
    thickness[axis] = thickness[axis + 3] = 16
    for obj in (
        gprMax.Domain(p1=xyz(np.asarray((16, 14, length)) * DL, axis)),
        gprMax.Discretisation(p1=(DL, DL, DL)),
        gprMax.TimeWindow(iterations=1),
        gprMax.OMPThreads(1),
        gprMax.PMLThickness(thickness=tuple(thickness)),
        gprMax.SurfaceImpedance(id="copper", preset="copper", fit_frequency_range=(1e9, 200e9)),
    ):
        scene.add(obj)
    add_bulk_host(scene, host_kind)
    # Substrate, finite-width ground, and strip all reach both PML terminations.
    for lower, upper, material in (
        ((2, 3, 0), (14, 7, length), "host"),
        ((2, 2, 0), (14, 3, length), "copper"),
        ((6, 7, 0), (10, 8, length), "copper"),
    ):
        scene.add(gprMax.Box(p1=xyz(np.asarray(lower) * DL, axis),
                            p2=xyz(np.asarray(upper) * DL, axis), material_id=material))
    return scene


def run_microstrip(cache, host_kind, *, length=120, steps=800, axis=2, precision="double"):
    from .validate_sibc_pml import permutation

    grid = build_grid(microstrip_scene(length, host_kind, axis),
                      cache / f"microstrip_{host_kind}_{length}_{axis}_{precision}", precision)
    axes = permutation(axis)
    field = (grid.Ex, grid.Ey, grid.Ez)[axes[1]]
    x, y, z = np.meshgrid(np.arange(6, 10), np.arange(3, 7), np.arange(length), indexing="ij")
    coords = [None] * 3
    for physical, index in zip(axes, (x, y, z)):
        coords[physical] = index
    field[tuple(coords)] = np.exp(-((z - 36) / 5.0) ** 2) * np.cos(0.6 * (z - 36))
    updates = CPUUpdates(grid)
    if grid.maxpoles:
        updates.set_dispersive_updates()
    probe = [slice(None)] * 3
    probe[axis] = 56
    trace = []
    initial_norm = field_norm(grid)
    energy = []
    bulk_peak = surface_peak = 0.0
    system = grid.impedance_surfaces
    # Confirm that air/substrate junctions actually occur on PML-owned edges.
    mixed_edges = 0
    from gprMax.impedance_pml import _edge_cells

    for index in system.pml_edge_indices:
        component, *coordinate = system.edge_info[index, :4]
        hosts = {int(grid.solid[cell]) for cell in _edge_cells(grid, component, coordinate)
                 if int(grid.solid[cell]) not in grid.impedance_marker_models}
        mixed_edges += len(hosts) > 1
    for step in range(steps):
        trace.append(field[tuple(probe)].copy())
        advance(updates)
        if step % 20 == 0 or step == steps - 1:
            energy.append(field_norm(grid) / initial_norm)
            surface_peak = max(surface_peak, float(np.max(np.abs(system.state_y))))
            if system.state_p.size:
                bulk_peak = max(bulk_peak, float(np.max(np.abs(system.state_p))))
    return grid, np.asarray(trace), np.asarray(energy), dict(
        surface_state_peak=surface_peak, bulk_state_peak=bulk_peak,
        mixed_host_pml_edges=int(mixed_edges), pml_edges=system.pml_edge_count,
    )


def compare_microstrip(cache, host_kind="lossy", **kwargs):
    short, trace, energy, metrics = run_microstrip(cache, host_kind, **kwargs)
    reference, reference_trace, _, _ = run_microstrip(cache, host_kind, length=320, **kwargs)
    assert short.dt == reference.dt
    peak = np.max(np.abs(reference_trace))
    difference = trace - reference_trace
    early = np.arange(len(trace)) * short.dt < 0.9 * ((104 - 36) + (104 - 56)) * DL / c
    reference_return = ((304 - 36) + (304 - 56)) * DL / c
    metrics.update(
        host_kind=host_kind,
        relative_reflection_peak=float(np.max(np.abs(difference)) / peak),
        relative_reflection_l2=float(np.linalg.norm(difference) / np.linalg.norm(reference_trace)),
        early_relative_error=float(np.max(np.abs(difference[early])) / peak),
        peak_field_norm=float(np.max(energy)), final_field_norm=float(energy[-1]),
        record_s=len(trace) * short.dt, reference_return_s=reference_return,
        finite=bool(np.isfinite(trace).all() and np.isfinite(energy).all()),
    )
    metrics["passed"] = bool(
        metrics["finite"] and metrics["mixed_host_pml_edges"] > 0
        and metrics["record_s"] < reference_return
        and metrics["early_relative_error"] < 1e-5
        and metrics["relative_reflection_peak"] < 0.01
        and metrics["final_field_norm"] < 0.1
        and metrics["surface_state_peak"] > 0
        and (host_kind == "lossy" or metrics["bulk_state_peak"] > 0)
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".codex_tmp/microstrip_pml"))
    parser.add_argument("--host", choices=("lossy", "debye", "lorentz", "drude", "all"), default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kinds = ("lossy", "debye", "lorentz", "drude") if args.host == "all" else (args.host,)
    results = [compare_microstrip(args.output_dir, kind) for kind in kinds]
    report = dict(cases=results, passed=all(case["passed"] for case in results))
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
