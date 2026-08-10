"""Validate normal-incidence reflection from mixed dispersive multilayers.

The FDTD geometry is two-dimensional TM but invariant in the transverse
direction, so its normal-incidence physics is one-dimensional.  A free-space
run supplies the incident field.  Generalized interface averaging and
ordinary staircasing are compared at the same user-requested first interface;
no fitted or retrospectively inferred half-cell corrections are applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gprmax-matplotlib")

import h5py
import matplotlib
import numpy as np

import gprMax
from testing.validation import validate_plane_wave_dispersive_halfspace as halfspace

from .layered_media import PlanarMedium, normal_incidence_reflection
from .pole_models import debye_term, drude_term, lorentz_term, make_material

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "multilayer_fdtd"

FIRST_INTERFACE_X = halfspace.INTERFACE_X
FREQUENCY_MIN = 0.25e9
FREQUENCY_MAX = 4.0e9
ANALYTICAL_PLOT_FREQUENCIES = np.linspace(FREQUENCY_MIN, FREQUENCY_MAX, 2001)
PHASE_MAGNITUDE_THRESHOLD = 0.02


@dataclass(frozen=True)
class MaterialDefinition:
    """One bulk material in both gprMax and analytical forms."""

    id: str
    er_inf: float
    conductivity: float = 0.0
    family: str = "dielectric"
    poles: tuple[tuple[float, ...], ...] = ()

    def analytical(self):
        terms = []
        if self.family == "debye":
            terms = [debye_term(delta_er, tau) for delta_er, tau in self.poles]
        elif self.family == "lorentz":
            terms = [
                lorentz_term(delta_er, frequency, damping)
                for delta_er, frequency, damping in self.poles
            ]
        elif self.family == "drude":
            terms = [drude_term(frequency, damping) for frequency, damping in self.poles]
        return make_material(
            self.id,
            self.er_inf,
            tuple(terms),
            conductivity=self.conductivity,
        )

    def add_to_scene(self, scene):
        scene.add(
            gprMax.Material(
                er=self.er_inf,
                se=self.conductivity,
                mr=1,
                sm=0,
                id=self.id,
            )
        )
        if self.family == "debye":
            scene.add(
                gprMax.AddDebyeDispersion(
                    poles=len(self.poles),
                    er_delta=tuple(pole[0] for pole in self.poles),
                    tau=tuple(pole[1] for pole in self.poles),
                    material_ids=(self.id,),
                )
            )
        elif self.family == "lorentz":
            scene.add(
                gprMax.AddLorentzDispersion(
                    poles=len(self.poles),
                    er_delta=tuple(pole[0] for pole in self.poles),
                    omega=tuple(pole[1] for pole in self.poles),
                    delta=tuple(pole[2] for pole in self.poles),
                    material_ids=(self.id,),
                )
            )
        elif self.family == "drude":
            scene.add(
                gprMax.AddDrudeDispersion(
                    poles=len(self.poles),
                    omega=tuple(pole[0] for pole in self.poles),
                    alpha=tuple(pole[1] for pole in self.poles),
                    material_ids=(self.id,),
                )
            )


@dataclass(frozen=True)
class LayerDefinition:
    material: MaterialDefinition
    cells: int

    @property
    def thickness(self):
        return self.cells * halfspace.DL


@dataclass(frozen=True)
class StackDefinition:
    label: str
    layers: tuple[LayerDefinition, ...]
    substrate: MaterialDefinition | None


DIELECTRIC_3 = MaterialDefinition("dielectric_3", 3.0)
DIELECTRIC_4 = MaterialDefinition("dielectric_4", 4.0)
DEBYE_1 = MaterialDefinition(
    "debye_1",
    2.5,
    family="debye",
    poles=((4.5, 80e-12),),
)
DEBYE_3 = MaterialDefinition(
    "debye_3",
    2.2,
    family="debye",
    poles=((3.0, 20e-12), (2.0, 100e-12), (1.0, 500e-12)),
)
LORENTZ_2 = MaterialDefinition(
    "lorentz_2",
    3.0,
    family="lorentz",
    poles=((1.5, 1.2e9, 0.4e9), (0.8, 3.2e9, 0.8e9)),
)
DRUDE_2 = MaterialDefinition(
    "drude_2",
    9.0,
    family="drude",
    poles=((0.5e9, 0.6e9), (1.0e9, 1.0e9)),
)

STACKS = {
    "dielectric_slab": StackDefinition(
        "Dielectric slab in free space",
        (LayerDefinition(DIELECTRIC_4, 30),),
        None,
    ),
    "debye_on_dielectric": StackDefinition(
        "Debye slab on dielectric substrate",
        (LayerDefinition(DEBYE_1, 24),),
        DIELECTRIC_3,
    ),
    "debye_lorentz": StackDefinition(
        "Debye--Lorentz stack on dielectric substrate",
        (LayerDefinition(DEBYE_1, 20), LayerDefinition(LORENTZ_2, 24)),
        DIELECTRIC_3,
    ),
    "lorentz_drude": StackDefinition(
        "Lorentz--Drude stack on dielectric substrate",
        (LayerDefinition(LORENTZ_2, 20), LayerDefinition(DRUDE_2, 20)),
        DIELECTRIC_4,
    ),
    "multipole_debye_lorentz": StackDefinition(
        "Three-pole Debye--Lorentz stack",
        (LayerDefinition(DEBYE_3, 20), LayerDefinition(LORENTZ_2, 20)),
        DIELECTRIC_3,
    ),
}


def _base_scene(threads):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(halfspace.DL,) * 3))
    scene.add(gprMax.Domain(p1=halfspace.DOMAIN))
    scene.add(gprMax.TimeWindow(time=halfspace.TIME_WINDOW))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(
        gprMax.PMLThickness(
            thickness=(
                halfspace.PML_CELLS,
                halfspace.PML_CELLS,
                0,
                halfspace.PML_CELLS,
                halfspace.PML_CELLS,
                0,
            )
        )
    )
    return scene


def build_scene(stack_name, averaging, threads, construction_order="normal"):
    """Build one finite-layer stack with a common requested geometry."""

    if construction_order not in {"normal", "reverse"}:
        raise ValueError(f"Unknown construction order: {construction_order}")
    scene = _base_scene(threads)
    scene.add(gprMax.DispersiveAveraging(enabled=averaging))
    if stack_name != "free_space":
        stack = STACKS[stack_name]
        materials = {layer.material.id: layer.material for layer in stack.layers}
        if stack.substrate is not None:
            materials[stack.substrate.id] = stack.substrate
        for material in materials.values():
            material.add_to_scene(scene)

        x = FIRST_INTERFACE_X
        layer_extents = []
        for layer in stack.layers:
            layer_extents.append((x, x + layer.thickness, layer.material.id))
            x += layer.thickness
        averaging_flag = "y" if averaging else "n"
        if construction_order == "normal":
            if stack.substrate is not None:
                scene.add(
                    gprMax.Box(
                        p1=(x, 0, float("inf")),
                        p2=halfspace.DOMAIN,
                        material_id=stack.substrate.id,
                        averaging=averaging_flag,
                    )
                )
            for x_start, x_end, material_id in reversed(layer_extents):
                scene.add(
                    gprMax.Box(
                        p1=(x_start, 0, float("inf")),
                        p2=(x_end, halfspace.DOMAIN[1], float("inf")),
                        material_id=material_id,
                        averaging=averaging_flag,
                    )
                )
        else:
            # Construct exactly the same final voxels from the opposite side.
            # The first layer initially fills the domain; source-side free
            # space and every subsequent layer then overwrite it in order.
            # Averaged edges should depend only on the final four voxels,
            # whereas legacy non-averaged edges retain construction ownership.
            first_material = stack.layers[0].material.id
            scene.add(
                gprMax.Box(
                    p1=(0, 0, float("inf")),
                    p2=halfspace.DOMAIN,
                    material_id=first_material,
                    averaging=averaging_flag,
                )
            )
            scene.add(
                gprMax.Box(
                    p1=(0, 0, float("inf")),
                    p2=(FIRST_INTERFACE_X, halfspace.DOMAIN[1], float("inf")),
                    material_id="free_space",
                    averaging=averaging_flag,
                )
            )
            for x_start, x_end, material_id in layer_extents[1:]:
                scene.add(
                    gprMax.Box(
                        p1=(x_start, 0, float("inf")),
                        p2=(x_end, halfspace.DOMAIN[1], float("inf")),
                        material_id=material_id,
                        averaging=averaging_flag,
                    )
                )
            final_material = "free_space" if stack.substrate is None else stack.substrate.id
            scene.add(
                gprMax.Box(
                    p1=(x, 0, float("inf")),
                    p2=halfspace.DOMAIN,
                    material_id=final_material,
                    averaging=averaging_flag,
                )
            )

    scene.add(
        gprMax.Waveform(
            wave_type="ricker",
            amp=1,
            freq=halfspace.SOURCE_FREQUENCY,
            id="plane_pulse",
        )
    )
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=halfspace.TFSF_P1,
            p2=halfspace.TFSF_P2,
            axis="x",
            psi=90,
            waveform_id="plane_pulse",
        )
    )
    scene.add(gprMax.Rx(p1=halfspace.RECEIVER, id="reflection_probe", outputs=["Ez"]))
    return scene


def _run(
    stack_name,
    mode,
    cache_dir,
    *,
    gpu,
    precision,
    threads,
    reuse,
    construction_order="normal",
):
    cache_name = "free_space" if stack_name == "free_space" else f"{stack_name}_{mode}"
    if construction_order != "normal" and stack_name != "free_space":
        cache_name += f"_{construction_order}"
    path = cache_dir / f"{cache_name}.h5"
    runtime = 0.0
    if not (reuse and path.exists()):
        options = (
            {"cpu_precision": precision}
            if gpu is None
            else {"gpu": [gpu], "gpu_precision": precision}
        )
        start = perf_counter()
        gprMax.run(
            scenes=[
                build_scene(
                    stack_name,
                    mode == "averaged",
                    threads,
                    construction_order=construction_order,
                )
            ],
            n=1,
            outputfile=path.with_suffix(""),
            hide_progress_bars=True,
            log_level=logging.WARNING,
            **options,
        )
        runtime = perf_counter() - start
    with h5py.File(path, "r") as output:
        trace = np.asarray(output["rxs/rx1/Ez"])
        dt = float(output.attrs["dt"])
    return trace, dt, runtime


def _analytical_stack(stack, frequencies):
    air = PlanarMedium(make_material("free space", 1.0))
    finite_layers = tuple(
        PlanarMedium(layer.material.analytical(), thickness=layer.thickness)
        for layer in stack.layers
    )
    substrate = air if stack.substrate is None else PlanarMedium(stack.substrate.analytical())
    return normal_incidence_reflection(frequencies, air, finite_layers, substrate)


def _analyse(incident, total, dt, stack):
    reflected = total - incident
    frequencies = np.fft.rfftfreq(incident.size, dt)
    incident_fft = np.fft.rfft(incident)
    reflected_fft = np.fft.rfft(reflected)
    selected = (
        (frequencies >= FREQUENCY_MIN)
        & (frequencies <= FREQUENCY_MAX)
        & (
            np.abs(incident_fft)
            >= halfspace.INCIDENT_SPECTRUM_THRESHOLD * np.max(np.abs(incident_fft))
        )
    )
    frequencies = frequencies[selected]
    gamma_receiver = reflected_fft[selected] / incident_fft[selected]
    distance = FIRST_INTERFACE_X - halfspace.RECEIVER[0]
    numerical_wavenumber = halfspace.free_space_numerical_wavenumber(frequencies, dt)
    gamma_fdtd = gamma_receiver * np.exp(1j * 2 * numerical_wavenumber * distance)
    gamma_analytical = _analytical_stack(stack, frequencies)
    return {
        "frequencies": frequencies,
        "gamma_fdtd": gamma_fdtd,
        "gamma_analytical": gamma_analytical,
    }


def _metrics(result):
    fdtd = result["gamma_fdtd"]
    analytical = result["gamma_analytical"]
    magnitude_error = np.abs(fdtd) - np.abs(analytical)
    complex_error = np.abs(fdtd - analytical)
    phase_mask = np.abs(analytical) >= PHASE_MAGNITUDE_THRESHOLD
    phase_error = np.angle(fdtd[phase_mask] / analytical[phase_mask], deg=True)
    return {
        "frequency_samples": int(fdtd.size),
        "magnitude_rmse": float(np.sqrt(np.mean(magnitude_error**2))),
        "magnitude_max_error": float(np.max(np.abs(magnitude_error))),
        "complex_relative_l2_error": float(
            np.linalg.norm(fdtd - analytical) / np.linalg.norm(analytical)
        ),
        "complex_absolute_rmse": float(np.sqrt(np.mean(complex_error**2))),
        "phase_rmse_degrees": float(np.sqrt(np.mean(phase_error**2))),
        "phase_samples": int(np.count_nonzero(phase_mask)),
    }


def _phase_for_plot(values, reference):
    phase = np.unwrap(np.angle(values)) * 180 / np.pi
    reference_phase = np.unwrap(np.angle(reference)) * 180 / np.pi
    phase += 360 * np.round(np.median((reference_phase - phase) / 360))
    return phase


def _write_case(stack_name, stack, results, output_dir):
    sampled = results["averaged"]
    dense_frequency = ANALYTICAL_PLOT_FREQUENCIES
    dense_analytical = _analytical_stack(stack, dense_frequency)
    path = output_dir / f"{stack_name}.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "frequency_hz",
                "analytical_magnitude",
                "analytical_phase_degrees",
                "averaged_magnitude",
                "averaged_phase_degrees",
                "staircased_magnitude",
                "staircased_phase_degrees",
                "averaged_complex_absolute_error",
                "staircased_complex_absolute_error",
            )
        )
        for index, frequency in enumerate(sampled["frequencies"]):
            writer.writerow(
                (
                    frequency,
                    abs(sampled["gamma_analytical"][index]),
                    np.angle(sampled["gamma_analytical"][index], deg=True),
                    abs(results["averaged"]["gamma_fdtd"][index]),
                    np.angle(results["averaged"]["gamma_fdtd"][index], deg=True),
                    abs(results["staircased"]["gamma_fdtd"][index]),
                    np.angle(results["staircased"]["gamma_fdtd"][index], deg=True),
                    abs(
                        results["averaged"]["gamma_fdtd"][index]
                        - sampled["gamma_analytical"][index]
                    ),
                    abs(
                        results["staircased"]["gamma_fdtd"][index]
                        - sampled["gamma_analytical"][index]
                    ),
                )
            )
    with (output_dir / f"{stack_name}_analytical_dense.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frequency_hz", "magnitude", "phase_degrees"))
        writer.writerows(
            zip(dense_frequency, np.abs(dense_analytical), np.angle(dense_analytical, deg=True))
        )

    frequency_ghz = sampled["frequencies"] / 1e9
    dense_frequency_ghz = dense_frequency / 1e9
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.5, 1.5, 1)},
    )
    axes[0].plot(
        dense_frequency_ghz,
        np.abs(dense_analytical),
        color="black",
        linewidth=2,
        label="Analytical recursion (2001 samples)",
    )
    dense_phase = np.unwrap(np.angle(dense_analytical)) * 180 / np.pi
    axes[1].plot(dense_frequency_ghz, dense_phase, color="black", linewidth=2)
    colours = {"averaged": "black", "staircased": "black"}
    markers = {"averaged": "o", "staircased": "s"}
    labels = {"averaged": "General pole average", "staircased": "Staircased"}
    for mode, result in results.items():
        axes[0].plot(
            frequency_ghz,
            np.abs(result["gamma_fdtd"]),
            linestyle="none",
            marker=markers[mode],
            markerfacecolor="none",
            markersize=3.5,
            color=colours[mode],
            label=labels[mode],
        )
        analytical_samples = result["gamma_analytical"]
        axes[1].plot(
            frequency_ghz,
            _phase_for_plot(result["gamma_fdtd"], analytical_samples),
            linestyle="none",
            marker=markers[mode],
            markerfacecolor="none",
            markersize=3.5,
            color=colours[mode],
            label=labels[mode],
        )
    spacing_ghz = float(np.median(np.diff(frequency_ghz)))
    bar_width = 0.36 * spacing_ghz
    offsets = {"averaged": -bar_width / 2, "staircased": bar_width / 2}
    facecolours = {"averaged": "white", "staircased": "0.65"}
    hatches = {"averaged": "///", "staircased": "..."}
    for mode, result in results.items():
        axes[2].bar(
            frequency_ghz + offsets[mode],
            np.abs(result["gamma_fdtd"] - result["gamma_analytical"]),
            width=bar_width,
            facecolor=facecolours[mode],
            edgecolor="black",
            hatch=hatches[mode],
            linewidth=0.45,
            label=labels[mode],
        )
    axes[0].set(ylabel=r"$|\Gamma|$", title=stack.label)
    axes[1].set_ylabel(r"Unwrapped phase of $\Gamma$ [deg]")
    axes[2].set(
        xlabel="Frequency [GHz]",
        ylabel=r"Pointwise $|\Gamma_{\mathrm{FDTD}}-\Gamma_{\mathrm{exact}}|$",
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
    figure.savefig(output_dir / f"{stack_name}.png", dpi=180)
    plt.close(figure)


def _serialise_stack(stack):
    return {
        "label": stack.label,
        "first_interface_x_metres": FIRST_INTERFACE_X,
        "layers": [
            {
                "material": layer.material.id,
                "family": layer.material.family,
                "thickness_metres": layer.thickness,
            }
            for layer in stack.layers
        ],
        "substrate": "free_space" if stack.substrate is None else stack.substrate.id,
    }


def _load_case_results(stack_name, output_dir):
    """Reconstruct complex samples from a committed CSV for plot-only use."""

    table = np.genfromtxt(output_dir / f"{stack_name}.csv", delimiter=",", names=True)
    analytical = table["analytical_magnitude"] * np.exp(
        1j * np.deg2rad(table["analytical_phase_degrees"])
    )
    results = {}
    for mode in ("averaged", "staircased"):
        results[mode] = {
            "frequencies": table["frequency_hz"],
            "gamma_analytical": analytical,
            "gamma_fdtd": table[f"{mode}_magnitude"]
            * np.exp(1j * np.deg2rad(table[f"{mode}_phase_degrees"])),
        }
    return results


def _write_report(output_dir, summary):
    lines = [
        "# Mixed-material multilayer validation",
        "",
        "All primary results are de-embedded to the same first interface Xs",
        "requested by the geometry. No fitted half-cell correction is applied.",
        "The FDTD values are discrete FFT samples; dense analytical curves are",
        "used only for visualisation. Pointwise errors are plotted as bars.",
        "",
        "| Stack | Mode | Complex relative L2 | Magnitude RMSE | Phase RMSE |",
        "|---|---|---:|---:|---:|",
    ]
    for stack_name, stack_values in summary["stacks"].items():
        for mode in ("averaged", "staircased"):
            values = stack_values["modes"][mode]
            lines.append(
                f"| {stack_name} | {mode} | "
                f"{values['complex_relative_l2_error']:.6g} | "
                f"{values['magnitude_rmse']:.6g} | "
                f"{values['phase_rmse_degrees']:.6g} deg |"
            )
    lines.extend(("", "## Figures", ""))
    for stack_name in STACKS:
        lines.append(f"- [{STACKS[stack_name].label}]({stack_name}.png)")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def run_validation(output_dir, cache_dir, *, gpu, precision, threads, reuse, cases):
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    previous_runtimes = {}
    if reuse and summary_path.exists():
        previous_runtimes = json.loads(summary_path.read_text()).get("runtime_seconds", {})

    incident, dt, incident_runtime = _run(
        "free_space",
        "reference",
        cache_dir,
        gpu=gpu,
        precision=precision,
        threads=threads,
        reuse=reuse,
    )
    summary = {
        "model": {
            "dl_metres": halfspace.DL,
            "dt_seconds": dt,
            "time_window_seconds": halfspace.TIME_WINDOW,
            "frequency_band_hz": [FREQUENCY_MIN, FREQUENCY_MAX],
            "requested_first_interface_x_metres": FIRST_INTERFACE_X,
            "analytical_plot_samples": int(ANALYTICAL_PLOT_FREQUENCIES.size),
            "backend": "cpu" if gpu is None else f"cuda:{gpu}",
            "precision": precision,
        },
        "runtime_seconds": {
            "free_space": incident_runtime or previous_runtimes.get("free_space", 0.0)
        },
        "stacks": {},
    }
    for stack_name in cases:
        stack = STACKS[stack_name]
        stack_results = {}
        stack_summary = {"definition": _serialise_stack(stack), "modes": {}}
        for mode in ("averaged", "staircased"):
            total, case_dt, runtime = _run(
                stack_name,
                mode,
                cache_dir,
                gpu=gpu,
                precision=precision,
                threads=threads,
                reuse=reuse,
            )
            if not np.isclose(case_dt, dt, rtol=1e-12, atol=0):
                raise RuntimeError(f"Time-step mismatch for {stack_name}/{mode}")
            result = _analyse(incident, total, dt, stack)
            stack_results[mode] = result
            stack_summary["modes"][mode] = _metrics(result)
            runtime_key = f"{stack_name}_{mode}"
            summary["runtime_seconds"][runtime_key] = runtime or previous_runtimes.get(
                runtime_key, 0.0
            )
        _write_case(stack_name, stack, stack_results, output_dir)
        summary["stacks"][stack_name] = stack_summary
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(output_dir, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from existing sampled CSV files without running FDTD.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--cache-dir", type=Path, default=RESULTS / "cache")
    parser.add_argument("--cases", nargs="+", choices=tuple(STACKS), default=tuple(STACKS))
    args = parser.parse_args()
    if args.plot_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for stack_name in args.cases:
            _write_case(
                stack_name,
                STACKS[stack_name],
                _load_case_results(stack_name, args.output_dir),
                args.output_dir,
            )
        return
    summary = run_validation(
        args.output_dir,
        args.cache_dir,
        gpu=args.gpu,
        precision=args.precision,
        threads=args.threads,
        reuse=args.reuse,
        cases=args.cases,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
