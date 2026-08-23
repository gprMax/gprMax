"""Validate rational-network terminals against canonical loaded-guide solutions.

The principal benchmark reproduces the finite-width parallel-plate guide used
by Pereda et al. (IEEE T-MTT, 1999).  A transverse sheet of one-edge networks
is illuminated by a TEM pulse.  Its reflection coefficient has the exact
closed form

    Gamma = -Zc / (2 Zsheet + Zc),    Zsheet = 2 Zedge / 15.

The factor two is the number of series z-directed Yee edges through the guide
height, and 15 is the effective number of parallel branches across the PMC-
bounded width.  Separate R, C, and L sheets exercise every elementary term in
Y(s) = G + sC + sum(r/(s-p)); the published series-RC and series-RLC examples
exercise real and conjugate pole recurrences.

The capacitance printed for the series-RC example in the 1999 paper is
0.02 pF.  Its published magnitude and phase curves instead correspond exactly
to 0.2 pF, which is therefore used here and identified in the report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, physical_constants
from scipy.signal.windows import tukey

import gprMax

DL = 1e-3
WIDTH = 15e-3
HEIGHT = 2e-3
LENGTH = 0.25
SOURCE_X = 0.03
PROBE_X = 0.08
LOAD_X = 0.15
TIME_WINDOW = 1e-9
LONG_TIME_WINDOW = 10e-9
NY = round(WIDTH / DL)
NZ = round(HEIGHT / DL)
ETA0 = physical_constants["characteristic impedance of vacuum"][0]
GUIDE_IMPEDANCE = ETA0 * HEIGHT / WIDTH

FREQUENCIES = np.linspace(1e9, 30e9, 291)


@dataclass(frozen=True)
class NetworkCase:
    """One edge impedance and its equivalent rational admittance."""

    name: str
    label: str
    resistance: float = 0.0
    inductance: float = 0.0
    capacitance: float = 0.0
    conductance: float = 0.0
    direct_capacitance: float = 0.0
    poles: tuple[complex, ...] = ()
    residues: tuple[complex, ...] = ()

    def edge_impedance(self, frequency: np.ndarray) -> np.ndarray:
        """Return the continuous-time series impedance of one edge."""

        s = 2j * np.pi * frequency
        impedance = np.full(frequency.shape, self.resistance, dtype=np.complex128)
        if self.inductance:
            impedance += s * self.inductance
        if self.capacitance:
            impedance += 1 / (s * self.capacitance)
        return impedance


def _series_rlc_terms(
    resistance: float, inductance: float, capacitance: float
) -> tuple[tuple[complex, complex], tuple[complex, complex]]:
    """Return a stable conjugate pole/residue representation of series RLC."""

    alpha = -resistance / (2 * inductance)
    beta = np.sqrt(1 / (inductance * capacitance) - alpha**2)
    pole = alpha + 1j * beta
    conjugate = np.conj(pole)
    residue = pole / (inductance * (pole - conjugate))
    return (pole, conjugate), (residue, np.conj(residue))


RLC_POLES, RLC_RESIDUES = _series_rlc_terms(75.0, 3e-9, 0.08e-12)
CASES = (
    NetworkCase(
        "resistor",
        r"resistor: $R=75\,\Omega$",
        resistance=75.0,
        conductance=1 / 75.0,
    ),
    NetworkCase(
        "capacitor",
        r"capacitor: $C=0.2\,\mathrm{pF}$",
        capacitance=0.2e-12,
        direct_capacitance=0.2e-12,
    ),
    NetworkCase(
        "inductor",
        r"inductor: $L=3\,\mathrm{nH}$",
        inductance=3e-9,
        poles=(0.0,),
        residues=(1 / 3e-9,),
    ),
    NetworkCase(
        "series_rc",
        r"Pereda series RC: $R=25\,\Omega$, $C=0.2\,\mathrm{pF}$",
        resistance=25.0,
        capacitance=0.2e-12,
        conductance=1 / 25.0,
        poles=(-1 / (25.0 * 0.2e-12),),
        residues=(-1 / (25.0**2 * 0.2e-12),),
    ),
    NetworkCase(
        "series_rlc",
        r"Pereda series RLC: $R=75\,\Omega$, $L=3\,\mathrm{nH}$, " r"$C=0.08\,\mathrm{pF}$",
        resistance=75.0,
        inductance=3e-9,
        capacitance=0.08e-12,
        poles=RLC_POLES,
        residues=RLC_RESIDUES,
    ),
)


def _add_terminal_sheet(scene: gprMax.Scene, case: NetworkCase) -> None:
    """Place the 15-by-2 effective network sheet of the Pereda benchmark."""

    scene.add(
        gprMax.RationalNetwork(
            id=case.name,
            conductance=case.conductance,
            capacitance=case.direct_capacitance,
            poles=case.poles,
            residues=case.residues,
        )
    )
    for j in range(NY + 1):
        for k in range(NZ):
            scene.add(
                gprMax.NetworkTerminal(
                    p1=(LOAD_X, j * DL, k * DL),
                    polarisation="z",
                    network_id=case.name,
                    id=f"load_{j}_{k}",
                )
            )


def _build_scene(case: NetworkCase | None, time_window: float = TIME_WINDOW) -> gprMax.Scene:
    """Create the TEM guide, matched distributed source, and optional sheet."""

    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(LENGTH, WIDTH, HEIGHT)))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.TimeWindow(time=time_window))
    scene.add(gprMax.PMLThickness(thickness=(10, 0, 0, 10, 0, 0)))
    scene.add(gprMax.SymmetryBoundary(face="y0", type="pmc"))
    scene.add(gprMax.SymmetryBoundary(face="ymax", type="pmc"))
    scene.add(gprMax.SymmetryBoundary(face="z0", type="pec"))
    scene.add(gprMax.SymmetryBoundary(face="zmax", type="pec"))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Waveform(wave_type="gaussian", amp=0.5, freq=30e9, id="pulse"))

    # The endpoint terminals on PMC walls carry half quadrature weight.  The
    # effective number of parallel source branches is consequently NY, not
    # NY + 1.  NZ series edge sources span the guide height.
    source_resistance = NY * GUIDE_IMPEDANCE / NZ
    scene.add(gprMax.RationalNetwork(id="source", conductance=1 / source_resistance))
    for j in range(NY + 1):
        for k in range(NZ):
            terminal_id = f"source_{j}_{k}"
            scene.add(
                gprMax.NetworkTerminal(
                    p1=(SOURCE_X, j * DL, k * DL),
                    polarisation="z",
                    network_id="source",
                    id=terminal_id,
                )
            )
            scene.add(gprMax.NetworkExcitation(terminal_id, "pulse"))

    if case is not None:
        _add_terminal_sheet(scene, case)
    scene.add(
        gprMax.Rx(
            p1=(PROBE_X, 7 * DL, DL),
            id="probe",
            outputs=["Ez", "Hy"],
        )
    )
    return scene


def run_models(output_dir: Path) -> None:
    """Run the reference guide and all loaded-guide cases in double precision."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, case in (("reference", None), *((item.name, item) for item in CASES)):
        gprMax.run(
            scenes=[_build_scene(case)],
            n=1,
            outputfile=output_dir / name,
            hide_progress_bars=True,
            cpu_precision="double",
        )

    inductor = next(case for case in CASES if case.name == "inductor")
    gprMax.run(
        scenes=[_build_scene(inductor, LONG_TIME_WINDOW)],
        n=1,
        outputfile=output_dir / "inductor_long",
        hide_progress_bars=True,
        cpu_precision="double",
    )


def _read_trace(path: Path) -> tuple[float, np.ndarray]:
    """Read one Ez trace and its time step."""

    with h5py.File(path, "r") as output:
        return float(output.attrs["dt"]), output["rxs/rx1/Ez"][...]


def _gate(trace: np.ndarray, time: np.ndarray, start: float, stop: float) -> np.ndarray:
    """Isolate a pulse without changing its absolute time reference."""

    result = np.zeros_like(trace)
    selected = (time >= start) & (time <= stop)
    result[selected] = trace[selected] * tukey(np.count_nonzero(selected), alpha=0.15)
    return result


def _dft(trace: np.ndarray, time: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    """Evaluate the engineering-convention Fourier transform at chosen points."""

    return np.exp(-2j * np.pi * frequency[:, None] * time[None, :]) @ trace


def _yee_wavenumber(frequency: np.ndarray, dt: float) -> np.ndarray:
    """Return the axial numerical wavenumber of the 1-D Yee dispersion relation."""

    argument = DL / (c * dt) * np.sin(np.pi * frequency * dt)
    return 2 / DL * np.arcsin(np.clip(argument, -1, 1))


def _exact_reflection(case: NetworkCase, frequency: np.ndarray) -> np.ndarray:
    """Return the exact TEM reflection at the lumped sheet."""

    sheet_impedance = NZ * case.edge_impedance(frequency) / NY
    return -GUIDE_IMPEDANCE / (2 * sheet_impedance + GUIDE_IMPEDANCE)


def analyse(output_dir: Path) -> dict[str, dict[str, float]]:
    """Extract, de-embed, plot, and tabulate all reflection coefficients."""

    dt, incident_trace = _read_trace(output_dir / "reference.h5")
    time = np.arange(incident_trace.size) * dt
    incident = _gate(incident_trace, time, 0.10e-9, 0.44e-9)
    incident_spectrum = _dft(incident, time, FREQUENCIES)
    propagation = np.exp(-2j * _yee_wavenumber(FREQUENCIES, dt) * (LOAD_X - PROBE_X))

    results: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for case in CASES:
        loaded_dt, loaded_trace = _read_trace(output_dir / f"{case.name}.h5")
        if loaded_dt != dt:
            raise RuntimeError(f"time-step mismatch in {case.name}")
        # The first sheet reflection occupies approximately 0.64--0.88 ns.
        # The tighter gate excludes the late signal returning from the outer
        # guide termination while retaining the complete network response.
        reflected = _gate(loaded_trace - incident_trace, time, 0.60e-9, 0.90e-9)
        simulated = _dft(reflected, time, FREQUENCIES) / incident_spectrum / propagation
        exact = _exact_reflection(case, FREQUENCIES)
        magnitude_error = np.abs(simulated) - np.abs(exact)
        phase_error = np.angle(simulated / exact, deg=True)
        results[case.name] = {
            "simulated": simulated,
            "exact": exact,
            "magnitude_error": magnitude_error,
            "phase_error": phase_error,
        }
        metrics[case.name] = {
            "magnitude_rmse": float(np.sqrt(np.mean(magnitude_error**2))),
            "magnitude_max_abs_error": float(np.max(np.abs(magnitude_error))),
            "phase_rmse_deg": float(np.sqrt(np.mean(phase_error**2))),
            "phase_max_abs_error_deg": float(np.max(np.abs(phase_error))),
        }

        table = np.column_stack(
            (
                FREQUENCIES,
                simulated.real,
                simulated.imag,
                exact.real,
                exact.imag,
                magnitude_error,
                phase_error,
            )
        )
        np.savetxt(
            output_dir / f"{case.name}_reflection.csv",
            table,
            delimiter=",",
            header=(
                "frequency_hz,simulated_real,simulated_imag,exact_real,exact_imag,"
                "magnitude_error,phase_error_deg"
            ),
            comments="",
        )

    stability = _analyse_long_inductor(output_dir)
    _plot_elementary(results, output_dir)
    _plot_pereda(results, output_dir)
    _plot_time_traces(incident_trace, time, output_dir)
    all_metrics = {"frequency_domain": metrics, "inductor_late_time": stability}
    (output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    _write_report(metrics, stability, output_dir)
    return metrics


def _analyse_long_inductor(output_dir: Path) -> dict[str, float]:
    """Check that the lossless zero-pole recurrence remains bounded for 10 ns."""

    dt, trace = _read_trace(output_dir / "inductor_long.h5")
    if not np.isfinite(trace).all():
        raise RuntimeError("the long lossless-inductor trace contains non-finite values")
    time = np.arange(trace.size) * dt
    intervals = np.arange(10, dtype=np.float64)
    maxima = np.empty(intervals.size)
    rms = np.empty(intervals.size)
    for index, start_ns in enumerate(intervals):
        selected = (time >= start_ns * 1e-9) & (time < (start_ns + 1) * 1e-9)
        maxima[index] = np.max(np.abs(trace[selected]))
        rms[index] = np.sqrt(np.mean(trace[selected] ** 2))

    late_peak_ratio = float(maxima[-1] / maxima[0])
    if late_peak_ratio >= 0.02:
        raise RuntimeError(
            f"lossless-inductor late peak ratio {late_peak_ratio:.5f} does not demonstrate decay"
        )
    np.savetxt(
        output_dir / "inductor_late_time.csv",
        np.column_stack((intervals, intervals + 1, maxima, rms)),
        delimiter=",",
        header="start_ns,stop_ns,maximum_abs_ez_v_per_m,rms_ez_v_per_m",
        comments="",
    )

    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    centre = intervals + 0.5
    axis.semilogy(centre, maxima, "ko-", markerfacecolor="none", label="maximum")
    axis.semilogy(centre, rms, "ks--", markerfacecolor="none", label="RMS")
    axis.set_xlabel("Time-window centre (ns)")
    axis.set_ylabel(r"$|E_z|$ (V/m)")
    axis.set_title("Ten-nanosecond lossless-inductor stability check")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "inductor_late_time.png", dpi=200)
    figure.savefig(output_dir / "inductor_late_time.pdf")
    plt.close(figure)
    return {
        "initial_1ns_peak_v_per_m": float(maxima[0]),
        "final_1ns_peak_v_per_m": float(maxima[-1]),
        "final_to_initial_peak_ratio": late_peak_ratio,
    }


def _plot_elementary(results: dict, output_dir: Path) -> None:
    """Plot independent R, C, and L sheet checks."""

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    styles = {"resistor": "o", "capacitor": "s", "inductor": "^"}
    for case in CASES[:3]:
        result = results[case.name]
        axes[0].plot(
            FREQUENCIES / 1e9,
            np.abs(result["exact"]),
            color="black",
            linestyle={"resistor": "-", "capacitor": "--", "inductor": ":"}[case.name],
            label=f"exact, {case.label}",
        )
        selected = np.arange(0, FREQUENCIES.size, 10)
        axes[0].plot(
            FREQUENCIES[selected] / 1e9,
            np.abs(result["simulated"])[selected],
            linestyle="none",
            marker=styles[case.name],
            markerfacecolor="none",
            color="black",
            label=f"gprMax, {case.name}",
        )
        axes[1].plot(
            FREQUENCIES[selected] / 1e9,
            result["phase_error"][selected],
            linestyle="none",
            marker=styles[case.name],
            markerfacecolor="none",
            color="black",
            label=case.name,
        )
    axes[0].set_ylabel(r"$|\Gamma|$")
    axes[0].set_ylim(0, 1.04)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.25)
    axes[1].axhline(0, color="0.55", linewidth=0.8)
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel("Phase residual (degrees)")
    axes[1].grid(True, alpha=0.25)
    figure.suptitle("Canonical one-edge R, C, and L sheets in a TEM guide")
    figure.tight_layout()
    figure.savefig(output_dir / "elementary_network_reflection.png", dpi=200)
    figure.savefig(output_dir / "elementary_network_reflection.pdf")
    plt.close(figure)


def _plot_pereda(results: dict, output_dir: Path) -> None:
    """Plot the two networks used in Pereda et al. (1999)."""

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), sharex=True)
    for column, case in enumerate(CASES[3:]):
        result = results[case.name]
        selected = np.arange(0, FREQUENCIES.size, 10)
        axes[0, column].plot(
            FREQUENCIES / 1e9,
            np.abs(result["exact"]),
            color="black",
            label="exact continuous-time network",
        )
        axes[0, column].plot(
            FREQUENCIES[selected] / 1e9,
            np.abs(result["simulated"])[selected],
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            color="black",
            label="gprMax FDTD",
        )
        axes[1, column].plot(
            FREQUENCIES / 1e9,
            np.angle(result["exact"], deg=True),
            color="black",
        )
        axes[1, column].plot(
            FREQUENCIES[selected] / 1e9,
            np.angle(result["simulated"], deg=True)[selected],
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            color="black",
        )
        axes[0, column].set_title(case.label, fontsize=10)
        axes[0, column].set_ylabel(r"$|\Gamma|$")
        axes[1, column].set_ylabel(r"$\angle\Gamma$ (degrees)")
        axes[1, column].set_xlabel("Frequency (GHz)")
        axes[0, column].set_ylim(0, 1.0)
        for row in range(2):
            axes[row, column].grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Pereda et al. (1999) loaded parallel-plate guide benchmark")
    figure.tight_layout()
    figure.savefig(output_dir / "pereda_1999_reflection.png", dpi=200)
    figure.savefig(output_dir / "pereda_1999_reflection.pdf")
    plt.close(figure)


def _plot_time_traces(incident_trace: np.ndarray, time: np.ndarray, output_dir: Path) -> None:
    """Show pulse separation and bounded late-time behaviour."""

    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    axis.plot(time * 1e9, incident_trace, color="black", label="empty-guide incident field")
    for case, linestyle in zip(CASES, ("--", ":", "-.", (0, (5, 2)), (0, (1, 1)))):
        _, loaded = _read_trace(output_dir / f"{case.name}.h5")
        axis.plot(
            time * 1e9,
            loaded - incident_trace,
            color="0.35",
            linestyle=linestyle,
            linewidth=0.9,
            label=f"{case.name} reflected",
        )
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel(r"$E_z$ (V/m)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "time_domain_separation.png", dpi=200)
    figure.savefig(output_dir / "time_domain_separation.pdf")
    plt.close(figure)


def _write_report(
    metrics: dict[str, dict[str, float]], stability: dict[str, float], output_dir: Path
) -> None:
    """Write a concise, reproducible interpretation of the benchmark."""

    lines = [
        "# Rational-network literature validation",
        "",
        "The model is the 15 mm by 2 mm parallel-plate TEM guide of Pereda et al. "
        "(1999), discretised at 1 mm and run in double precision. The simulated "
        "reflection is de-embedded by the axial Yee numerical wavenumber.",
        "",
        "The paper prints 0.02 pF for its series-RC example, but its published "
        "magnitude and phase curves are those of 0.2 pF. The latter is used here.",
        "",
        "| case | magnitude RMS | maximum magnitude | phase RMS (deg) | maximum phase (deg) |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in CASES:
        item = metrics[case.name]
        lines.append(
            f"| {case.name} | {item['magnitude_rmse']:.5f} | "
            f"{item['magnitude_max_abs_error']:.5f} | {item['phase_rmse_deg']:.3f} | "
            f"{item['phase_max_abs_error_deg']:.3f} |"
        )
    lines.extend(
        (
            "",
            "A separate 10 ns run of the ideal lossless-inductor sheet remained finite. "
            f"Its final 1 ns peak was {stability['final_to_initial_peak_ratio']:.5f} of "
            "the initial pulse peak; the residual decays rather than grows.",
            "",
            "References:",
            "",
            "- J. A. Pereda et al., *A New Algorithm for the Incorporation of "
            "Arbitrary Linear Lumped Networks into FDTD Simulators*, IEEE T-MTT, 1999.",
            "- J. A. Pereda et al., *FDTD Modeling of an Inductor by a Discrete-Time "
            "Technique*, IEEE T-MTT, 2004.",
            "",
        )
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def _check_metrics(metrics: dict[str, dict[str, float]]) -> None:
    """Fail the validation if a recurrence or field-coupling regression occurs."""

    for case, result in metrics.items():
        if result["magnitude_rmse"] >= 0.02:
            raise RuntimeError(
                f"{case} magnitude RMS error {result['magnitude_rmse']:.5f} exceeds 0.02"
            )
        if result["phase_rmse_deg"] >= 2.0:
            raise RuntimeError(
                f"{case} phase RMS error {result['phase_rmse_deg']:.3f} degrees exceeds 2"
            )


def main() -> None:
    """Run or reuse the models, then generate all validation artefacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("rational_network_results"),
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="analyse existing HDF5 outputs without rerunning gprMax",
    )
    args = parser.parse_args()
    if not args.reuse:
        run_models(args.output_dir)
    metrics = analyse(args.output_dir)
    _check_metrics(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
