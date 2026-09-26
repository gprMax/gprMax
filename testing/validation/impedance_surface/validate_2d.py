"""Reproducible reduced-mode SIBC and virtual-waveguide validation."""

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

import gprMax
from gprMax.grid.fdtd_grid import FDTDGrid

FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
DL = 1e-3


def normalized_field_error(reference, actual):
    """Use a common E/eta-H amplitude scale, including zero-component modes."""
    weights = np.asarray((1.0, 1.0, 1.0, 376.730313666, 376.730313666, 376.730313666))[
        None, :, None
    ]
    return float(
        np.max(abs(actual - reference) * weights) / max(np.max(abs(reference) * weights), 1e-300)
    )


def scene_2d(
    *,
    polarization="TM",
    invariant=2,
    normal=0,
    direction="+",
    resistance=np.inf,
    virtual=False,
    active=False,
    port=None,
    steps=600,
    formulation="HORIPML",
    length=72,
    repaired=False,
    spacing=DL,
    host_kind=None,
):
    frequency = 12e9 if host_kind is not None else 22e9
    transverse = next(axis for axis in range(3) if axis not in (invariant, normal))
    if port is None:
        port = virtual or active

    def point(n, t, a=0):
        result = np.zeros(3)
        result[normal], result[transverse], result[invariant] = n * DL, t * DL, a
        return tuple(result)

    scene = gprMax.Scene()
    dl = [DL] * 3
    dl[invariant] = spacing
    thickness = [0] * 6
    thickness[normal] = thickness[normal + 3] = 8
    for obj in (
        gprMax.DomainMode(mode=polarization),
        gprMax.Domain(p1=point(length, 26, np.inf)),
        gprMax.Discretisation(p1=tuple(dl)),
        gprMax.PMLThickness(thickness=tuple(thickness)),
        gprMax.PMLFormulation(formulation=formulation),
        gprMax.TimeWindow(iterations=steps),
        gprMax.OMPThreads(1),
    ):
        scene.add(obj)
    if repaired:
        from testing.validation.impedance_surface.investigate_pml_profile import SIGMA_MAX

        for factor in range(2):
            scene.add(
                gprMax.PMLCFS(
                    alphascalingprofile="quartic",
                    alphascalingdirection="forward",
                    alphamin=0.0,
                    alphamax=0.0 if factor == 0 else 1.1 * SIGMA_MAX,
                    kappascalingprofile="constant",
                    kappascalingdirection="forward",
                    kappamin=1.0,
                    kappamax=1.0,
                    sigmascalingprofile="quartic",
                    sigmascalingdirection="forward",
                    sigmamin=0.0,
                    sigmamax=SIGMA_MAX,
                )
            )
    if host_kind is not None:
        from .validate_sibc_pml import add_bulk_host

        if not repaired:
            scene.add(gprMax.PMLCFS(
                alphascalingprofile="constant", alphascalingdirection="forward", alphamin=0, alphamax=0,
                kappascalingprofile="constant", kappascalingdirection="forward", kappamin=1, kappamax=1,
                sigmascalingprofile="quartic", sigmascalingdirection="forward", sigmamin=0, sigmamax=8,
            ))
        add_bulk_host(scene, host_kind)
        scene.add(gprMax.Box(p1=point(0, 0), p2=point(length, 13, np.inf), material_id="host"))
    surface = (
        gprMax.SurfaceImpedance(
            id="wall", conductivity=1e4, fit_frequency_range=(1e9, 200e9), fit_order=4
        )
        if resistance == "foster"
        else None
        if resistance == "pec"
        else gprMax.SurfaceImpedance(id="wall", resistance=resistance)
    )
    if surface is not None:
        scene.add(surface)
    else:
        scene.add(gprMax.TimeStepStabilityFactor(f=0.99))
    for lower, upper in ((2, 5), (21, 24)):
        scene.add(
            gprMax.Box(
                p1=point(0, lower),
                p2=point(length, upper, np.inf),
                material_id="pec" if resistance == "pec" else "wall",
                averaging="n",
            )
        )
    scene.add(
        gprMax.Waveform(
            wave_type="contsine" if active else "gaussian", amp=1.0, freq=frequency, id="pulse"
        )
    )
    plane = 24 if direction == "+" else 48
    if active and not virtual:
        plane += -12 if direction == "+" else 12
    if not active:
        source = 48 if direction == "+" else 24
        scene.add(
            gprMax.HertzianDipole(
                p1=point(source, 13, np.inf),
                polarisation="xyz"[invariant if polarization == "TM" else transverse],
                waveform_id="pulse",
            )
        )
    for t in (5, 13, 20):
        scene.add(gprMax.Rx(p1=point(36, t, np.inf), id=f"probe_{t}"))
    if port:
        window = (5, 21) if resistance == "pec" else (3, 23)
        scene.add(gprMax.EigenmodeBand(id="band", fmin=frequency, fmax=frequency, points=1))
        scene.add(
            gprMax.EigenmodePort(
                port=1,
                p1=point(plane, window[0]),
                p2=point(plane, window[1], np.inf),
                direction=direction,
                modes=(1,),
                anchors=(frequency,),
                plot_fields=False,
            )
        )
        if virtual:
            scene.add(
                gprMax.VirtualWaveguide(
                    port=1, length_cells=24, pml_cells=8, source_clearance_cells=4
                )
            )
        if active:
            scene.add(
                gprMax.EigenmodeExcitation(port=1, mode=1, waveform="pulse", plot_waveform=False)
            )
    return scene


def run_2d(path, *, precision="double", geometry_only=False, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grids = []
    original = FDTDGrid.build

    def capture(grid):
        original(grid)
        grids.append(grid)

    with patch.object(FDTDGrid, "build", capture):
        gprMax.run(
            scenes=[scene_2d(**kwargs)],
            outputfile=path,
            cpu_precision=precision,
            geometry_only=geometry_only,
            hide_progress_bars=True,
            log_level=40,
        )
    if geometry_only:
        return None, grids[0]
    with h5py.File(path.with_suffix(".h5")) as output:
        traces = np.asarray(
            [[output[f"rxs/rx{rx}/{component}"][...] for component in FIELDS] for rx in (1, 2, 3)]
        )
    return traces, grids[0]


def compare_invariant_3d(
    path,
    *,
    polarization="TM",
    invariant=2,
    normal=0,
    resistance=np.inf,
    precision="double",
    steps=200,
    corner=True,
):
    """Compare a reduced grid with a separately compiled 3D image extrusion."""
    from gprMax import config
    from gprMax.updates.cpu_updates import CPUUpdates
    from testing.validation.impedance_surface.validate_sibc_pml import advance, build_grid

    transverse = next(axis for axis in range(3) if axis not in (invariant, normal))
    grids, updates = [], []
    for reduced in (True, False):

        def point(n, t, a):
            value = np.zeros(3)
            value[normal], value[transverse], value[invariant] = n * DL, t * DL, a
            return tuple(value)

        end = np.inf if reduced else 4 * DL
        scene = gprMax.Scene()
        if reduced:
            scene.add(gprMax.DomainMode(mode=polarization))
        pml = [0] * 6
        pml[normal] = pml[normal + 3] = 4
        for obj in (
            gprMax.Domain(p1=point(32, 32, end)),
            gprMax.Discretisation(p1=(DL,) * 3),
            gprMax.TimeStepStabilityFactor(f=0.5 if reduced else 0.5 * np.sqrt(1.5)),
            gprMax.TimeWindow(iterations=steps),
            gprMax.OMPThreads(1),
            gprMax.PMLThickness(thickness=tuple(pml)),
        ):
            scene.add(obj)
        if not reduced:
            for side in ("0", "max"):
                scene.add(
                    gprMax.SymmetryBoundary(
                        face="xyz"[invariant] + side, type="pec" if polarization == "TM" else "pmc"
                    )
                )
        scene.add(
            gprMax.SurfaceImpedance(
                id="wall", conductivity=1e4, fit_frequency_range=(1e9, 200e9), fit_order=4
            )
            if resistance == "foster"
            else gprMax.SurfaceImpedance(id="wall", resistance=resistance)
        )
        boxes = [(0, 4, 32, 7), (0, 25, 32, 28)]
        if corner:
            boxes.append((14, 12, 18, 17))
        for n0, t0, n1, t1 in boxes:
            scene.add(
                gprMax.Box(
                    p1=point(n0, t0, 0), p2=point(n1, t1, end), material_id="wall", averaging="n"
                )
            )
        grid = build_grid(scene, Path(path) / ("reduced" if reduced else "extruded"), precision)
        grids.append(grid)
        updates.append(CPUUpdates(grid))
    reduced, full = grids
    assert np.isclose(reduced.dt, full.dt, rtol=1e-13, atol=0)
    eta = config.sim_config.em_consts["z0"]
    live = 0 if polarization == "TM" else 1
    active = (
        [invariant] + [3 + axis for axis in (normal, transverse)]
        if polarization == "TM"
        else [normal, transverse, 3 + invariant]
    )
    rng = np.random.default_rng(1048)
    for component in active:
        field = getattr(reduced, FIELDS[component])
        selection = [slice(None)] * 3
        selection[invariant] = live
        view = field[tuple(selection)]
        seeds = rng.standard_normal(view.shape) * (1 if component < 3 else 1 / eta)
        interior = np.zeros(view.shape, dtype=bool)
        interior[8:24, 8:24] = True
        ids = reduced.ID[(component, *selection)]
        coefficient = reduced.updatecoeffsE if component < 3 else reduced.updatecoeffsH
        free = coefficient[ids, 0] != 0
        view[:] = seeds * interior * free
        getattr(full, FIELDS[component])[:] = np.expand_dims(view, invariant)
    denominator = np.sqrt(
        sum(np.sum(getattr(reduced, FIELDS[c]) ** 2) * (eta**2 if c >= 3 else 1) for c in active)
    )
    peak = 0.0

    def close_periodic_images(names):
        # The independent 3D reference owns invariant layers 1 and 2.
        # Layers 0/3/4 are periodic images. This avoids involving the bulk
        # PML's half-open transverse outer face in an invariance test.
        for name in names:
            field = getattr(full, name)
            for destination, source in ((0, 2), (3, 1), (4, 2)):
                selected = [slice(None)] * 3
                selected[invariant] = destination
                field[tuple(selected)] = np.take(field, source, axis=invariant)

    for step in range(steps):
        for update in updates:
            update.update_magnetic()
            update.update_magnetic_pml()
            if update.grid is full:
                close_periodic_images(FIELDS[3:])
            update.update_electric_a()
            update.update_symmetry_boundaries_electric()
            update.update_electric_pml()
            update.update_electric_b()
            update.update_impedance_surfaces()
            if update.grid is full:
                close_periodic_images(FIELDS[:3])
        if step % 10 == 0 or step == steps - 1:
            squared = 0.0
            for component in active:
                first = np.take(getattr(reduced, FIELDS[component]), live, axis=invariant)
                second = np.take(getattr(full, FIELDS[component]), 2, axis=invariant)
                squared += np.sum((first - second) ** 2) * (eta**2 if component >= 3 else 1)
            peak = max(peak, float(np.sqrt(squared) / denominator))
    limit = 2e-4 if precision == "single" else 2e-11
    return dict(
        polarization=polarization,
        invariant=invariant,
        normal=normal,
        resistance=str(resistance),
        precision=precision,
        steps=steps,
        corner=corner,
        maximum_relative_field_error=peak,
        limit=limit,
        passed=peak < limit,
    )


def _advance_guide(updates, iteration):
    updates.update_magnetic()
    updates.update_magnetic_pml()
    updates.update_eigenmode_sources_magnetic(iteration)
    updates.update_electric_a()
    updates.update_symmetry_boundaries_electric()
    updates.update_electric_pml()
    updates.update_eigenmode_sources_electric(iteration)
    updates.update_electric_b()
    updates.update_impedance_surfaces()


def _seed_packet(grid, polarization, center=48, noise=False):
    """Canonical z-invariant, x-propagating packet entirely in the main guide."""
    live = 0 if polarization == "TM" else 1
    n, t = np.meshgrid(np.arange(34, 63) + center - 48, np.arange(6, 21), indexing="ij")
    envelope = np.exp(-(((n - center) / 3.0) ** 2)) * np.cos(0.7 * (n - center))
    profile = np.cos(np.pi * (t - 5) / 16) if polarization == "TM" else np.sin(np.pi * (t - 5) / 16)
    field = grid.Ez if polarization == "TM" else grid.Ey
    field[n, t, live] = envelope * profile
    if noise:
        rng = np.random.default_rng(773)
        names = ("Ez", "Hx", "Hy") if polarization == "TM" else ("Ex", "Ey", "Hz")
        for name in names:
            scale = 1e-6 if name[0] == "E" else 1e-6 / 376.730313666
            getattr(grid, name)[n, t, live] += scale * rng.standard_normal(n.shape)


def packet_validation(
    path,
    *,
    polarization="TM",
    resistance=np.inf,
    precision="double",
    repaired=False,
    steps=20000,
    long_reference=False,
):
    from gprMax.updates.cpu_updates import CPUUpdates
    from testing.validation.impedance_surface.validate_sibc_pml import field_norm

    _, grid = run_2d(
        Path(path) / "grid",
        polarization=polarization,
        resistance=resistance,
        precision=precision,
        repaired=repaired,
        geometry_only=True,
        virtual=not long_reference,
        length=144 if long_reference else 72,
        steps=steps,
    )
    _seed_packet(grid, polarization, center=120 if long_reference else 48, noise=steps >= 20000)
    updates = CPUUpdates(grid)
    grids = [grid] + [guide.aux_grid for guide in grid.virtual_waveguides]
    initial = sum(field_norm(item) for item in grids)
    records, trace = [], []
    live = 0 if polarization == "TM" else 1
    probe = 108 if long_reference else 36
    for step in range(steps):
        # t=13 is a symmetry node of the seeded TM profile. Sampling there
        # would normalize a reflection against numerical noise.
        trace.append(float((grid.Ez if polarization == "TM" else grid.Ey)[probe, 10, live]))
        _advance_guide(updates, step)
        if step % 50 == 0 or step == steps - 1:
            norm = sum(field_norm(item) for item in grids) / initial
            ade = max(float(np.max(abs(item.impedance_surfaces.state_y))) for item in grids)
            pml = max(
                float(np.max(abs(getattr(slab, name))))
                for item in grids
                for slab in item.pmls["slabs"]
                for name in ("EPhi1", "EPhi2", "HPhi1", "HPhi2")
            )
            records.append(((step + 1) * grid.dt, norm, ade, pml))
            if not np.all(np.isfinite(records[-1])) or norm > 1e6:
                break
    records = np.asarray(records)
    fit = records[-min(20, len(records)) :]
    slope = float(np.polyfit(fit[:, 0], np.log(np.maximum(fit[:, 1], 1e-300)), 1)[0] / 2)
    bounded = bool(
        step + 1 == steps and np.all(np.isfinite(records)) and np.max(records[:, 1]) < 2.0
    )
    return (
        dict(
            polarization=polarization,
            resistance=str(resistance),
            precision=precision,
            repaired=repaired,
            steps=step + 1,
            dt_s=grid.dt,
            duration_s=(step + 1) * grid.dt,
            peak_squared_norm=float(np.max(records[:, 1])),
            final_squared_norm=float(records[-1, 1]),
            max_ade_state=float(np.max(records[:, 2])),
            max_pml_history=float(np.max(records[:, 3])),
            fitted_late_amplitude_growth_per_s=slope,
            bounded=bounded,
        ),
        records,
        np.asarray(trace),
    )


def main():
    import argparse
    import json
    from itertools import product

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/2d_sibc")
    parser.add_argument("--section", choices=("all", "matrix", "late"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "_cache"
    cache.mkdir(exist_ok=True)
    (args.output / ".gitignore").write_text("_cache/\n")
    summary = args.output / "summary.json"
    results = json.loads(summary.read_text()) if summary.exists() else {}
    models = (np.inf, 5.0, "foster")
    if args.section in ("all", "matrix"):
        results["extrusion"] = []
        results["virtual"] = []
        for pol, inv, res, precision in product(
            ("TE", "TM"), range(3), models, ("double", "single")
        ):
            row = compare_invariant_3d(
                cache / "extrusion",
                polarization=pol,
                invariant=inv,
                normal=(inv + 1) % 3,
                resistance=res,
                precision=precision,
            )
            results["extrusion"].append(row)
            print(
                "extrusion",
                pol,
                inv,
                res,
                precision,
                row["maximum_relative_field_error"],
                flush=True,
            )
        orientations = [(inv, n) for inv in range(3) for n in range(3) if n != inv]
        for pol, (inv, normal), direction, precision, res, active in product(
            ("TE", "TM"), orientations, ("+", "-"), ("double", "single"), models, (False, True)
        ):
            kw = dict(
                polarization=pol,
                invariant=inv,
                normal=normal,
                direction=direction,
                precision=precision,
                resistance=res,
                active=active,
                steps=600,
            )
            first, _ = run_2d(cache / "physical", **kw)
            second, grid = run_2d(cache / "virtual", virtual=True, **kw)
            error = normalized_field_error(first, second)
            limit = 2e-4 if precision == "single" else 2e-8
            row = dict(
                kw,
                resistance=str(res),
                maximum_relative_field_error=error,
                limit=limit,
                passed=error < limit,
            )
            if active:
                with h5py.File((cache / "virtual").with_suffix(".h5")) as output:
                    row["reflection_magnitude"] = float(
                        abs(output["eigenmode_ports/port1/S"][0, 0])
                    )
            results["virtual"].append(row)
            print("virtual", pol, inv, normal, direction, precision, res, active, error, flush=True)
        summary.write_text(json.dumps(results, indent=2) + "\n")
    if args.section in ("all", "late"):
        results["late"] = []
        results["reflection"] = []
        for pol, res, repaired in product(("TE", "TM"), models, (False, True)):
            precision = "single" if repaired else "double"
            name = f"{pol}_{res}_{precision}_{'repaired' if repaired else 'default'}"
            row, records, _ = packet_validation(
                cache / name,
                polarization=pol,
                resistance=res,
                precision=precision,
                repaired=repaired,
            )
            results["late"].append(row)
            np.savetxt(
                args.output / (name + ".csv"),
                records,
                delimiter=",",
                header="time_s,squared_field_norm,maximum_ade_state,maximum_pml_history",
                comments="",
            )
            first, _, signal = packet_validation(
                cache / "short", polarization=pol, resistance=res, repaired=repaired, steps=240
            )
            _, _, reference = packet_validation(
                cache / "long",
                polarization=pol,
                resistance=res,
                repaired=repaired,
                steps=240,
                long_reference=True,
            )
            error = float(np.max(abs(signal - reference)) / max(np.max(abs(reference)), 1e-300))
            results["reflection"].append(
                dict(
                    polarization=pol,
                    resistance=str(res),
                    repaired=repaired,
                    maximum_relative_difference=error,
                )
            )
            np.savetxt(
                args.output / (name + "_reflection.csv"),
                np.column_stack((np.arange(len(signal)) * first["dt_s"], signal, reference)),
                delimiter=",",
                header="time_s,virtual_pml_field,long_reference_field",
                comments="",
            )
            summary.write_text(json.dumps(results, indent=2) + "\n")
            print("late", name, row, "reflection", error, flush=True)
    results["passed"] = all(
        row["passed"] for group in ("extrusion", "virtual") for row in results.get(group, [])
    ) and all(row["bounded"] for row in results.get("late", []))
    summary.write_text(json.dumps(results, indent=2) + "\n")
    write_report(results, args.output)
    if not results["passed"]:
        raise SystemExit("2D SIBC validation failed; see summary.json")


def validate_pmc_modes(path):
    """Discrete Neumann (TM) and Dirichlet (TE) parallel-wall spectra."""
    from itertools import product

    rows = []
    orientations = [(inv, normal) for inv in range(3) for normal in range(3) if inv != normal]
    for pol, (inv, normal), precision in product(("TE", "TM"), orientations, ("double", "single")):
        _, grid = run_2d(
            Path(path) / "mode",
            polarization=pol,
            invariant=inv,
            normal=normal,
            precision=precision,
            active=True,
            geometry_only=True,
            steps=100,
        )
        solver = grid.eigenmodesources[0].mode_solver
        transverse = 0.0 if pol == "TM" else 2 * np.sin(np.pi / 32) / DL
        expected = float(np.sqrt(1 - (transverse / solver.operator_k0) ** 2))
        error = float(abs(solver.operator_neff[0] - expected) / expected)
        limit = 2e-6 if precision == "single" else 1e-11
        rows.append(
            dict(
                polarization=pol,
                invariant=inv,
                normal=normal,
                precision=precision,
                expected_operator_neff=expected,
                measured_operator_neff=[
                    float(solver.operator_neff[0].real),
                    float(solver.operator_neff[0].imag),
                ],
                relative_error=error,
                passed=error < limit,
            )
        )
    return rows


def write_report(results, output):
    """Write publication-independent plots and an auditable validation summary."""
    import json

    import matplotlib.pyplot as plt

    output = Path(output)
    virtual, late = results.get("virtual", []), results.get("late", [])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    errors = {}
    for pol in ("TE", "TM"):
        for precision in ("double", "single"):
            group = [
                row["maximum_relative_field_error"]
                for row in virtual
                if row["polarization"] == pol and row["precision"] == precision
            ]
            if group:
                errors[f"{pol} {precision}"] = max(group)
    if errors:
        axes[0].bar(list(errors), np.maximum(list(errors.values()), 1e-17))
        for index, value in enumerate(errors.values()):
            if value == 0:
                axes[0].annotate(
                    "0 (bitwise)",
                    (index, 1e-17),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        axes[0].set_yscale("log")
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].axhline(2e-8, color="tab:green", linestyle="--", label="Double limit")
        axes[0].axhline(2e-4, color="tab:orange", linestyle="--", label="Single limit")
        axes[0].legend(fontsize=8)
    axes[0].set(title="Physical / virtual continuation", ylabel="Maximum E / eta-H scaled error")
    for row in late:
        name = f"{row['polarization']}_{row['resistance']}_{row['precision']}_{'repaired' if row['repaired'] else 'default'}"
        trace = np.loadtxt(output / (name + ".csv"), delimiter=",", skiprows=1)
        axes[1].semilogy(
            trace[:, 0] * 1e9,
            np.maximum(trace[:, 1], 1e-20),
            label=name.replace("_", " "),
            linewidth=1.0,
        )
    axes[1].set(
        title="Source-free virtual guides: 20,000 steps",
        xlabel="Time (ns)",
        ylabel="Squared field norm / initial",
    )
    if late:
        axes[1].legend(fontsize=6, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(output / "validation.png", dpi=170)
    plt.close(fig)
    rows = [
        "# Full 2D SIBC and virtual-guide validation",
        "",
        "CPU kernels; 1 mm physical cells; automatic timestep factor 0.99. "
        "The long-run timestep is 2.335067793382187 ps (46.7014 ns for 20,000 steps).",
        "",
        f"- {len(results.get('extrusion', []))} reduced/3D comparisons, including corners and fitted surface states.",
        f"- {len(virtual)} physical/virtual comparisons: TE/TM, every invariant/propagation axis, "
        "both directions and precisions, passive and active, PMC / 5 ohm / four-pole fitted surfaces.",
        f"- {len(late)} source-free 20,000-step guide runs with default and repaired HORIPML.",
        "",
        "## Measured field agreement",
        "",
        "| Case | Maximum relative error |",
        "|---|---:|",
    ]
    rows += [f"| {name} | {error:.6g} |" for name, error in errors.items()]
    for precision in ("double", "single"):
        cases = [
            row["maximum_relative_field_error"]
            for row in results.get("extrusion", [])
            if row["precision"] == precision
        ]
        if cases:
            rows.append(f"| Reduced / 3D, {precision} | {max(cases):.6g} |")
    rows += [
        "",
        "Field differences use a common maximum of |E| and eta0 |H|. "
        "This keeps analytically zero TEM components from being normalized by their own roundoff noise.",
        "The separate 3D reference uses the full 3D field kernels and compiler, with two owned "
        "invariant layers and periodic image closure on its outer storage planes. This isolates "
        "the reduced stencil from the native PML's half-open transverse outer face.",
        "",
        "## Long-time bounds and reflection",
        "",
        "Norm and history bounds are sampled every 50 steps and at the final step.",
        "",
        "| Mode / surface / profile | Peak sampled squared norm | Final squared norm |",
        "|---|---:|---:|",
    ]
    rows += [
        f"| {r['polarization']} / {r['resistance']} / {'repaired' if r['repaired'] else 'default'} "
        f"| {r['peak_squared_norm']:.6g} | {r['final_squared_norm']:.6g} |"
        for r in late
    ]
    reflection = results.get("reflection", [])
    if reflection:
        rows += [
            "",
            f"Peak differences against the longer causal reference range from "
            f"{min(r['maximum_relative_difference'] for r in reflection):.6g} to "
            f"{max(r['maximum_relative_difference'] for r in reflection):.6g} "
            "for these eight-cell PMLs. They measure finite-absorber reflection, separately from aperture coupling.",
        ]
    status = (
        "All reported long runs remain bounded with finite field, PML, and ADE histories. "
        if all(row["bounded"] for row in late)
        else "One or more long runs failed the bounds check. "
    )
    rows += [
        "",
        status
        + "The TE packet has nonzero initial electric divergence and retains stationary charge fields; "
        "its late norm need not decay to zero. The JSON and CSV files retain late fitted slopes and "
        "history maxima; slopes at a stationary or roundoff floor are not an exponential-growth diagnosis.",
        "",
        "![Validation](validation.png)",
        "",
        "## Reproduce",
        "",
        "```text",
        "python -m testing.validation.impedance_surface.validate_2d",
        "python -m pytest tests/impedance_surfaces/test_2d_sibc.py -q",
        "```",
        "",
        "The broader test selection and environment limitations are recorded in "
        "[regression checks](regression_checks.md).",
        "",
        "The repaired profile uses alpha1=0, unit kappa, and alpha2(z)=1.1*sigma1(z) "
        "with matching quartic grading. Default profiles use double precision and repaired long runs "
        "use single precision; the full coupling matrix checks both precisions for every surface.",
        "",
        "Scope: CPU main-grid TE/TM and 3D; passive walls, uniform longitudinal PML continuation, "
        "isotropic lossless nondispersive retained hosts in PML and virtual guides. "
        "These tests do not certify every mesh, material, source, or custom PML profile.",
    ]
    (output / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if virtual:
        pmc = {
            key: [row for row in results.get(key, []) if row["resistance"] == "inf"]
            for key in ("extrusion", "virtual", "late", "reflection")
        }
        pmc["analytic_modes"] = validate_pmc_modes(output / "_cache/pmc_modes")
        root = Path(__file__).parent.parent / "sibc_based_pmc"
        (root / "results").mkdir(parents=True, exist_ok=True)
        (root / "results/2d_results.json").write_text(json.dumps(pmc, indent=2) + "\n")
        errors = {
            precision: max(
                row["relative_error"]
                for row in pmc["analytic_modes"]
                if row["precision"] == precision
            )
            for precision in ("double", "single")
        }
        (root / "2d_validation.md").write_text(
            "# Exact SIBC PMC in 2D TE/TM and virtual guides\n\n"
            "The native `resistance=float('inf')` boundary was checked in all invariant-axis orientations "
            "and both propagation directions. It retains electric dual mass and circulation, with exactly "
            "zero admittance and surface current.\n\n"
            f"The data contain {len(pmc['extrusion'])} independent 3D-extrusion comparisons, "
            f"{len(pmc['virtual'])} physical/virtual comparisons, {len(pmc['late'])} 20,000-step runs, "
            "and 24 analytical discrete-mode checks.\n\n"
            "For the 16-cell parallel-wall guide, the TM fundamental is the constant Neumann mode "
            "with operator index 1. The TE fundamental has transverse symbol "
            "`2 sin(pi/32)/dl`, giving `sqrt(1-(kt/operator_k0)^2)`. "
            f"Maximum relative modal errors are {errors['double']:.6g} in double and "
            f"{errors['single']:.6g} in single precision.\n\n"
            "See [machine-readable results](results/2d_results.json) and the "
            "[general 2D validation report](../impedance_surface/results/2d_sibc/README.md) "
            "for field errors, finite-PML reflection, history bounds, and reproduction commands. "
            "The 3D reference uses periodic invariant images; no legacy PMC volume is used as an oracle.\n",
            encoding="utf-8",
        )
        if not all(row["passed"] for row in pmc["analytic_modes"]):
            raise RuntimeError("Analytical 2D PMC mode validation failed")


if __name__ == "__main__":
    main()
