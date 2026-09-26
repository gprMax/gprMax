"""Reproduce full-solver surface-impedance virtual-waveguide comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

import gprMax
from gprMax.grid.fdtd_grid import FDTDGrid


def guide_scene(
    *,
    virtual,
    resistance=np.inf,
    normal_axis=2,
    direction="+",
    steps=600,
    formulation="HORIPML",
    active=False,
    host_kind=None,
    layered=False,
):
    """Mixed PEC/SIBC guide; auxiliary and monolithic PML positions coincide."""

    dl = 1e-3
    frequency = 12e9 if host_kind is not None else 22e9
    transverse = tuple(axis for axis in range(3) if axis != normal_axis)
    basis = (*transverse, normal_axis)

    def point(local):
        result = np.zeros(3)
        result[list(basis)] = np.asarray(local) * dl
        return tuple(result)

    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=point((16, 16, 72))),
        gprMax.Discretisation(p1=(dl, dl, dl)),
        gprMax.TimeStepStabilityFactor(f=0.99),
        gprMax.TimeWindow(iterations=steps),
        gprMax.OMPThreads(1),
    ):
        scene.add(obj)
    pml = [0] * 6
    pml[normal_axis] = pml[normal_axis + 3] = 8
    scene.add(gprMax.PMLThickness(thickness=tuple(pml)))
    scene.add(gprMax.PMLFormulation(formulation=formulation))
    if host_kind is not None:
        from .validate_sibc_pml import add_bulk_host

        # Cropping opaque padding changes the default cross-section average
        # used to choose sigma_max. Fix it in both guides so this comparison
        # measures aperture coupling rather than different PML profiles.
        scene.add(gprMax.PMLCFS(
            alphascalingprofile="constant", alphascalingdirection="forward", alphamin=0, alphamax=0,
            kappascalingprofile="constant", kappascalingdirection="forward", kappamin=1, kappamax=1,
            sigmascalingprofile="quartic", sigmascalingdirection="forward", sigmamin=0, sigmamax=8,
        ))
        add_bulk_host(scene, host_kind)
        scene.add(gprMax.Box(
            p1=point((0, 0, 0)), p2=point((8 if layered else 16, 16, 72)), material_id="host"
        ))
    for lower, upper in (((0, 0, 0), (3, 16, 72)), ((13, 0, 0), (16, 16, 72))):
        scene.add(gprMax.Box(p1=point(lower), p2=point(upper), material_id="pec"))
    surface = (
        gprMax.SurfaceImpedance(
            id="wall", preset="copper", fit_frequency_range=(1e8, 1e11), fit_order=8
        )
        if resistance == "copper"
        else gprMax.SurfaceImpedance(id="wall", resistance=resistance)
    )
    scene.add(surface)
    for lower, upper in (((3, 1, 0), (13, 4, 72)), ((3, 12, 0), (13, 15, 72))):
        scene.add(gprMax.Box(p1=point(lower), p2=point(upper), material_id="wall"))
    scene.add(
        gprMax.Waveform(
            wave_type="contsine" if active else "gaussian", amp=1, freq=frequency, id="pulse"
        )
    )
    plane = 24 if direction == "+" else 48
    if active and not virtual:
        plane += -12 if direction == "+" else 12
    source = 48 if direction == "+" else 24
    if not active:
        scene.add(
            gprMax.HertzianDipole(
                p1=point((8, 8, source)), polarisation="xyz"[transverse[0]], waveform_id="pulse"
            )
        )
    for y in (4, 8):
        scene.add(gprMax.Rx(p1=point((8, y, 36)), id=f"probe_{y}"))
    if virtual or active:
        scene.add(gprMax.EigenmodeBand(id="band", fmin=frequency, fmax=frequency, points=1))
        scene.add(
            gprMax.EigenmodePort(
                port=1,
                p1=point((2, 2, plane)),
                p2=point((14, 14, plane)),
                direction=direction,
                modes=(1,),
                anchors=(frequency,),
                plot_fields=False,
            )
        )
        if virtual:
            scene.add(
                gprMax.VirtualWaveguide(port=1, length_cells=24, pml_cells=8, source_clearance_cells=4)
            )
        if active:
            scene.add(
                gprMax.EigenmodeExcitation(port=1, mode=1, waveform="pulse", plot_waveform=False)
            )
    return scene


def run_guide(path, *, precision="double", **kwargs):
    grids = []
    original = FDTDGrid.build

    def capture(grid):
        original(grid)
        grids.append(grid)

    with patch.object(FDTDGrid, "build", capture):
        gprMax.run(
            scenes=[guide_scene(**kwargs)],
            outputfile=path,
            cpu_precision=precision,
            hide_progress_bars=True,
            log_level=40,
        )
    with h5py.File(path.with_suffix(".h5")) as output:
        traces = np.asarray(
            [
                [
                    output[f"rxs/rx{rx}/{component}"][...]
                    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
                ]
                for rx in (1, 2)
            ]
        )
    return traces, grids[0]


def compare_guides(path, **kwargs):
    reference, _ = run_guide(path / "reference", virtual=False, **kwargs)
    actual, grid = run_guide(path / "split", virtual=True, **kwargs)
    scale = np.max(np.abs(reference), axis=-1, keepdims=True)
    scale = np.maximum(scale, np.max(scale) * 1e-12)
    error = float(np.max(np.abs(actual - reference) / scale))
    aux = grid.virtual_waveguides[0].aux_grid
    return {
        "max_normalized_trace_error": error,
        "max_absolute_trace_error": float(np.max(np.abs(actual - reference))),
        "main_boundary_edges": grid.impedance_surfaces.edge_count,
        "aux_boundary_edges": aux.impedance_surfaces.edge_count,
        "aux_pml_boundary_edges": len(aux.impedance_surfaces.pml_edge_indices),
        "aux_ade_states": int(
            np.sum(aux.impedance_surfaces.model_info[aux.impedance_surfaces.port_info[:, 0], 0])
        ),
        "dt_s": float(grid.dt),
        "passed": error < 2e-8,
    }


def active_guide(path, **kwargs):
    traces, grid = run_guide(path, virtual=True, active=True, **kwargs)
    with h5py.File(path.with_suffix(".h5")) as output:
        reflection = float(np.abs(output["eigenmode_ports/port1/S"][0, 0]))
    guide = grid.virtual_waveguides[0]
    system = guide.aux_grid.impedance_surfaces
    fields = (guide.aux_grid.Ex, guide.aux_grid.Ey, guide.aux_grid.Ez)
    source_e = np.asarray(
        [
            fields[edge[0]][tuple(edge[1:4])]
            for edge in system.edge_info[guide._impedance_source_edge_indices]
        ]
    )
    finite = bool(np.all(np.isfinite(traces)) and np.all(np.isfinite(system.state_y)))
    return {
        "reflection_magnitude": reflection,
        "source_boundary_e_max": float(np.max(np.abs(source_e))),
        "max_electric_receiver_field": float(np.max(np.abs(traces[:, :3]))),
        "finite": finite,
        "passed": finite and reflection < 0.01 and bool(np.any(source_e)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--active-steps", type=int, default=1200)
    parser.add_argument(
        "--workdir", type=Path, default=Path(".codex_tmp/impedance_virtual_validation")
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results/virtual_waveguide.json"
    )
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    results = {
        "precision": "double",
        "timestep_factor": 0.99,
        "passive_steps": args.steps,
        "active_steps": args.active_steps,
        "passive": [],
        "active": [],
    }
    models = (("exact_pmc", np.inf), ("constant_1000_ohm", 1000.0), ("copper_8_poles", "copper"))
    for label, resistance in models:
        for formulation in ("HORIPML", "MRIPML"):
            for normal_axis in range(3):
                for direction in ("-", "+"):
                    name = f"{label}_{formulation}_{'xyz'[normal_axis]}{direction}"
                    workdir = args.workdir / name
                    workdir.mkdir(exist_ok=True)
                    row = {
                        "model": label,
                        "formulation": formulation,
                        "normal_axis": "xyz"[normal_axis],
                        "direction": direction,
                    }
                    row.update(
                        compare_guides(
                            workdir,
                            resistance=resistance,
                            normal_axis=normal_axis,
                            direction=direction,
                            steps=args.steps,
                            formulation=formulation,
                        )
                    )
                    results["passive"].append(row)
                    print(name, row["max_normalized_trace_error"], flush=True)
            row = {"model": label, "formulation": formulation, "normal_axis": "z", "direction": "+"}
            row.update(
                active_guide(
                    args.workdir / f"{label}_{formulation}_active",
                    resistance=resistance,
                    steps=args.active_steps,
                    formulation=formulation,
                )
            )
            results["active"].append(row)
            print(label, formulation, "active reflection", row["reflection_magnitude"], flush=True)
    results["passed"] = all(
        row["passed"] for group in ("passive", "active") for row in results[group]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    write_report(results, args.output)
    if not results["passed"]:
        raise SystemExit("One or more virtual-waveguide validation checks failed")


def write_report(results, output):
    """Keep the measured acceptance criteria and reproduction command beside results."""
    import matplotlib.pyplot as plt

    labels = ("exact_pmc", "constant_1000_ohm", "copper_8_poles")
    names = {
        "exact_pmc": "Exact PMC",
        "constant_1000_ohm": "1000 ohm",
        "copper_8_poles": "Copper (8 poles)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    lines = []
    for label in labels:
        passive = [row for row in results["passive"] if row["model"] == label]
        active = [row for row in results["active"] if row["model"] == label]
        error = max(row["max_normalized_trace_error"] for row in passive)
        reflection = max(row["reflection_magnitude"] for row in active)
        axes[0].scatter(names[label], error, s=45)
        axes[0].annotate(
            "0 (bitwise)" if error == 0 else f"{error:.2e}",
            (names[label], error),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
        axes[1].bar(names[label], reflection)
        lines.append(f"| {label} | {len(passive)} | {error:.3e} | {reflection:.3e} |")
    axes[0].axhline(2e-8, color="gray", linestyle="--", label="Acceptance threshold")
    axes[0].set_ylim(-0.15e-8, 2.5e-8)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Maximum normalized trace error")
    axes[0].set_title("Virtual versus monolithic guide")
    axes[1].set_ylabel("Maximum |S11| at 22 GHz")
    axes[1].set_title("Active virtual source, both PML formulations")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(fig)
    report = (
        """# Surface-impedance virtual-waveguide validation

The mixed PEC/SIBC rectangular guide is solved with its physical rear intact,
then with a virtual guide replacing that rear at the same Yee plane. Auxiliary
and physical PML locations coincide. Both center and wall receivers record all
six fields. The pulse test covers every propagation axis and direction with
HORIPML and MRIPML. All runs use double precision and timestep factor 0.99.

| Surface model | Passive comparisons | Maximum normalized trace error | Maximum active reflection |
|---|---:|---:|---:|
"""
        + "\n".join(lines)
        + """

Passive acceptance is a maximum normalized six-field trace difference below
2e-8. Each channel uses its own reference peak, floored at 1e-12 of the largest
channel peak. Active runs use a 22 GHz continuous sine and require |S11| < 0.01,
finite fields/ADE states, and nonzero surface E on the auxiliary source plane.
The active test exercises the sparse source correction and Foster history;
the passive comparisons exercise full-mass aperture closure and PML histories.

The copper case uses the public copper preset fitted with eight Foster poles
over 0.1–100 GHz. The zero-admittance case uses public resistance=inf. The finite
constant case uses 1000 ohms. These measurements cover the stated scenes rather
than establishing unconditional stability for arbitrary material realizations.

The supported extension is a CPU 3D guide with longitudinally invariant walls,
lossless nondispersive retained materials, and surfaces strictly inside the
modal window. The window includes an opaque voxel beyond each wall. Auxiliary
surface rows retain the full main-grid dual-cell masses; unused tangential-H
padding samples supply the missing main-side curl at the aperture. Surface
source increments use the local implicit denominator and update ADE histories.

Reproduce from the repository root:

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
python -m testing.validation.impedance_surface.virtual_waveguide
```

The JSON records individual cases, timestep, sparse boundary counts, PML row
counts, and active ADE-state counts. HDF5 run outputs remain in the workdir.
"""
    )
    output.with_suffix(".md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
