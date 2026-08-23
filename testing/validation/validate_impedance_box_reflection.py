"""Benchmark normal-incidence reflection from a closed ``ImpedanceBox``.

The source is the homogeneous-background, integer-vector DPW rather than the
axial DPW.  The vector source supports both signs of every Cartesian axis and
does not sample the opaque box while constructing its auxiliary 1-D grid.

Each loaded run is paired with a geometrically identical free-space run.  At
the centre-line receiver the reference is the incident field and
``loaded - reference`` is the scattered field.  A short gate retains the
specular return from the broad flat face and excludes its later edge
diffraction.  Propagation is removed with the axial Yee wavenumber.

The comparison is deliberately against the *algorithm implemented by the
time step*, not ``Z(j omega)`` inserted into a continuous Fresnel formula.
The state-space model is trapezoidally discretised, its transfer function is
evaluated on the unit circle, tangential E is centred from integer to magnetic
half time, and the exterior half Yee cell is retained in Ampere's law.

Example::

    python -m testing.validation.validate_impedance_box_reflection --threads 4
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from scipy.constants import c, epsilon_0, mu_0
from scipy.signal.windows import tukey

import gprMax
from gprMax.impedance_surfaces import SurfaceImpedanceModel


DL = 1.0e-3
NORMAL_EXTENT = 0.080
TRANSVERSE_EXTENT = 0.100
PML_CELLS = 6
TFSF_CLEARANCE = 0.008
BOX_TRANSVERSE_LOWER = 0.010
BOX_TRANSVERSE_UPPER = 0.090
BOX_THICKNESS = 0.005
FACE_LOW = 0.030
FACE_HIGH = 0.050
RECEIVER_DISTANCE = 0.015
SOURCE_FREQUENCY = 30.0e9
TIME_WINDOW = 0.6e-9
FREQUENCIES = np.linspace(18.0e9, 42.0e9, 61)
GATE_HALF_WIDTH = 2.0 / SOURCE_FREQUENCY
ETA0 = np.sqrt(mu_0 / epsilon_0)
ACCEPTANCE_LIMITS = {
    "magnitude_rmse": 0.005,
    "phase_rmse_degrees": 0.7,
    "complex_relative_l2_error": 0.01,
}

# A passive first-order impedance with appreciable magnitude and phase change
# across the benchmark band.  B=A_RATE makes C carry ohms directly.
A_RATE = 2 * np.pi * 25.0e9
MODEL_A = ((-A_RATE,),)
MODEL_B = (A_RATE,)
MODEL_C = (120.0,)
MODEL_D = 70.0
MODEL_ID = "reflection_wall"


@dataclass(frozen=True)
class Orientation:
    """One illuminated box face and a co-polar receiver component."""

    name: str
    normal_axis: int
    normal_sign: int
    electric_axis: int
    psi_degrees: float

    @property
    def propagation(self) -> tuple[int, int, int]:
        vector = [0, 0, 0]
        vector[self.normal_axis] = -self.normal_sign
        return tuple(vector)

    @property
    def electric_component(self) -> str:
        return f"E{'xyz'[self.electric_axis]}"


ORIENTATIONS = {
    item.name: item
    for item in (
        Orientation("-x", 0, -1, 2, 90.0),
        Orientation("+x", 0, +1, 2, 90.0),
        Orientation("-y", 1, -1, 0, 0.0),
        Orientation("+y", 1, +1, 0, 0.0),
        Orientation("-z", 2, -1, 1, 0.0),
        Orientation("+z", 2, +1, 1, 0.0),
    )
}


def benchmark_model() -> SurfaceImpedanceModel:
    """Return the exact continuous model supplied to the user API."""

    return SurfaceImpedanceModel(
        MODEL_ID,
        A=MODEL_A,
        B=MODEL_B,
        C=MODEL_C,
        D=MODEL_D,
        fit_fmin_hz=float(FREQUENCIES[0]),
        fit_fmax_hz=float(FREQUENCIES[-1]),
    )


def algorithmic_impedance(
    model: SurfaceImpedanceModel, frequencies_hz: np.ndarray, dt: float
) -> np.ndarray:
    """Return the trapezoidal ADE impedance at magnetic half time.

    If ``z = exp(j omega dt)``, the implemented recurrence gives

    ``Z_alg(z) = Z0 + L (z I - F)^-1 G``.

    It equals the continuous realization evaluated at the bilinear-warped
    frequency ``s = (2j/dt) tan(omega dt/2)``.
    """

    frequency = np.asarray(frequencies_hz, dtype=np.float64)
    discrete = model.discretise(dt)
    result = np.full(frequency.shape, discrete.Z0, dtype=np.complex128)
    if model.order:
        identity = np.eye(model.order)
        for index, value in np.ndenumerate(frequency):
            z = np.exp(2j * np.pi * value * dt)
            result[index] += discrete.L @ np.linalg.solve(
                z * identity - discrete.F, discrete.G
            )
    return result


def yee_wavenumber(frequencies_hz: np.ndarray, dt: float, dl: float = DL) -> np.ndarray:
    """Return the real axial Yee wavenumber in free space."""

    frequency = np.asarray(frequencies_hz, dtype=np.float64)
    courant = c * dt / dl
    argument = np.sin(np.pi * frequency * dt) / courant
    if np.any(np.abs(argument) > 1 + 64 * np.finfo(np.float64).eps):
        raise ValueError("benchmark frequency lies outside the axial Yee propagating band")
    return 2 / dl * np.arcsin(np.clip(argument, -1, 1))


def algorithmic_reflection(
    model: SurfaceImpedanceModel,
    frequencies_hz: np.ndarray,
    dt: float,
    dl: float = DL,
) -> np.ndarray:
    """Return the exact discrete reflection at the boundary-E plane.

    For a flat face, per unit tangential length, the compiled update is

    ``eps0*dl/2 * dE/dt + K = n x H_out``.

    Tangential E and surface current are related at half time by ``Z_alg``.
    Solving this equation with incident/reflected axial Yee waves gives the
    expression below.  It includes all three easily missed terms: bilinear
    ADE warping, ``cos(omega*dt/2)`` E centring, and the exterior half-cell
    displacement current.
    """

    frequency = np.asarray(frequencies_hz, dtype=np.float64)
    omega_dt = 2 * np.pi * frequency * dt
    impedance = algorithmic_impedance(model, frequency, dt)
    boundary_admittance = (
        1j * epsilon_0 * dl / dt * np.sin(omega_dt / 2)
        + np.cos(omega_dt / 2) / impedance
    )
    phase = np.exp(0.5j * yee_wavenumber(frequency, dt, dl) * dl)
    normalised = ETA0 * boundary_admittance
    return (phase - normalised) / (1 / phase + normalised)


def _geometry(orientation: Orientation):
    extent = np.full(3, TRANSVERSE_EXTENT, dtype=np.float64)
    extent[orientation.normal_axis] = NORMAL_EXTENT
    lower = np.full(3, BOX_TRANSVERSE_LOWER, dtype=np.float64)
    upper = np.full(3, BOX_TRANSVERSE_UPPER, dtype=np.float64)
    receiver = 0.5 * extent

    if orientation.normal_sign < 0:
        lower[orientation.normal_axis] = FACE_HIGH
        upper[orientation.normal_axis] = FACE_HIGH + BOX_THICKNESS
        receiver[orientation.normal_axis] = FACE_HIGH - RECEIVER_DISTANCE
        face_position = FACE_HIGH
    else:
        lower[orientation.normal_axis] = FACE_LOW - BOX_THICKNESS
        upper[orientation.normal_axis] = FACE_LOW
        receiver[orientation.normal_axis] = FACE_LOW + RECEIVER_DISTANCE
        face_position = FACE_LOW

    tfsf_lower = np.full(3, TFSF_CLEARANCE, dtype=np.float64)
    tfsf_upper = extent - TFSF_CLEARANCE
    return extent, lower, upper, receiver, tfsf_lower, tfsf_upper, face_position


def build_scene(orientation: Orientation, loaded: bool, threads: int = 1) -> gprMax.Scene:
    """Build a reference or impedance-box TFSF scene."""

    extent, lower, upper, receiver, tfsf_lower, tfsf_upper, _ = _geometry(orientation)
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=tuple(extent)))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.PMLThickness(thickness=PML_CELLS))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(
        gprMax.Waveform(
            wave_type="ricker", amp=1.0, freq=SOURCE_FREQUENCY, id="plane_pulse"
        )
    )
    scene.add(
        gprMax.DiscretePlaneWaveVector(
            p1=tuple(tfsf_lower),
            p2=tuple(tfsf_upper),
            m_vec=orientation.propagation,
            psi=orientation.psi_degrees,
            waveform_id="plane_pulse",
            material_id="free_space",
        )
    )
    if loaded:
        scene.add(
            gprMax.SurfaceImpedance(
                id=MODEL_ID,
                A=MODEL_A,
                B=MODEL_B,
                C=MODEL_C,
                D=MODEL_D,
                fit_fmin_hz=float(FREQUENCIES[0]),
                fit_fmax_hz=float(FREQUENCIES[-1]),
            )
        )
        scene.add(gprMax.ImpedanceBox(tuple(lower), tuple(upper), MODEL_ID))
    scene.add(
        gprMax.Rx(
            p1=tuple(receiver),
            id="reflection_probe",
            outputs=[orientation.electric_component],
        )
    )
    return scene


def _run_trace(
    orientation: Orientation,
    loaded: bool,
    cache_dir: Path,
    threads: int,
    reuse: bool,
) -> tuple[np.ndarray, float]:
    stem = f"{orientation.name.replace('+', 'p').replace('-', 'm')}_{'box' if loaded else 'reference'}"
    output = cache_dir / stem
    path = output.with_suffix(".h5")
    if not (reuse and path.exists()):
        gprMax.run(
            scenes=[build_scene(orientation, loaded, threads)],
            outputfile=output,
            hide_progress_bars=True,
            log_level=logging.WARNING,
            cpu_precision="double",
        )
    with h5py.File(path, "r") as data:
        trace = np.asarray(data[f"rxs/rx1/{orientation.electric_component}"])
        dt = float(data.attrs["dt"])
    return trace, dt


def _window_about(trace: np.ndarray, centre: int, dt: float) -> np.ndarray:
    half_samples = max(2, int(round(GATE_HALF_WIDTH / dt)))
    start = max(0, centre - half_samples)
    stop = min(trace.size, centre + half_samples + 1)
    result = np.zeros_like(trace)
    result[start:stop] = trace[start:stop] * tukey(stop - start, alpha=0.25)
    return result


def _dft(trace: np.ndarray, dt: float, frequencies_hz: np.ndarray) -> np.ndarray:
    time = np.arange(trace.size, dtype=np.float64) * dt
    return np.exp(-2j * np.pi * frequencies_hz[:, None] * time[None, :]) @ trace


def analyse_traces(
    orientation: Orientation,
    incident_trace: np.ndarray,
    total_trace: np.ndarray,
    dt: float,
) -> dict[str, np.ndarray | float | int]:
    """Gate, de-embed, and compare one orientation."""

    if incident_trace.shape != total_trace.shape:
        raise ValueError("reference and loaded traces have different lengths")
    reflected_trace = total_trace - incident_trace
    incident_peak = int(np.argmax(np.abs(incident_trace)))
    expected_delay = int(round(2 * RECEIVER_DISTANCE / (c * dt)))
    search_half_width = int(round(3 / (SOURCE_FREQUENCY * dt)))
    predicted = incident_peak + expected_delay
    search_start = max(0, predicted - search_half_width)
    search_stop = min(reflected_trace.size, predicted + search_half_width + 1)
    if search_stop <= search_start:
        raise RuntimeError("reflection search window is empty")
    reflection_peak = search_start + int(
        np.argmax(np.abs(reflected_trace[search_start:search_stop]))
    )

    incident = _window_about(incident_trace, incident_peak, dt)
    reflected = _window_about(reflected_trace, reflection_peak, dt)
    incident_spectrum = _dft(incident, dt, FREQUENCIES)
    reflected_spectrum = _dft(reflected, dt, FREQUENCIES)
    if np.min(np.abs(incident_spectrum)) < 1e-8 * np.max(np.abs(incident_spectrum)):
        raise RuntimeError("incident spectrum is too small in the benchmark band")

    gamma_receiver = reflected_spectrum / incident_spectrum
    propagation = np.exp(
        2j * yee_wavenumber(FREQUENCIES, dt) * RECEIVER_DISTANCE
    )
    simulated = gamma_receiver * propagation
    exact = algorithmic_reflection(benchmark_model(), FREQUENCIES, dt)
    magnitude_error = np.abs(simulated) - np.abs(exact)
    phase_error = np.angle(simulated / exact, deg=True)
    return {
        "frequency": FREQUENCIES.copy(),
        "simulated": simulated,
        "exact": exact,
        "magnitude_error": magnitude_error,
        "phase_error_degrees": phase_error,
        "incident_peak_iteration": incident_peak,
        "reflection_peak_iteration": reflection_peak,
        "magnitude_rmse": float(np.sqrt(np.mean(magnitude_error**2))),
        "magnitude_max_abs_error": float(np.max(np.abs(magnitude_error))),
        "phase_rmse_degrees": float(np.sqrt(np.mean(phase_error**2))),
        "phase_max_abs_error_degrees": float(np.max(np.abs(phase_error))),
        "complex_relative_l2_error": float(
            np.linalg.norm(simulated - exact) / np.linalg.norm(exact)
        ),
    }


def _write_csv(path: Path, result: dict) -> None:
    table = np.column_stack(
        (
            result["frequency"],
            result["simulated"].real,
            result["simulated"].imag,
            result["exact"].real,
            result["exact"].imag,
            result["magnitude_error"],
            result["phase_error_degrees"],
        )
    )
    np.savetxt(
        path,
        table,
        delimiter=",",
        header=(
            "frequency_hz,simulated_real,simulated_imag,exact_real,exact_imag,"
            "magnitude_error,phase_error_degrees"
        ),
        comments="",
    )


def run_benchmark(
    output_dir: Path,
    orientation_names: tuple[str, ...] = tuple(ORIENTATIONS),
    threads: int = 1,
    reuse: bool = False,
) -> dict:
    """Run requested faces and write CSV plus a machine-readable summary."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "_cache"
    cache_dir.mkdir(exist_ok=True)
    started = perf_counter()
    results = {}
    for name in orientation_names:
        orientation = ORIENTATIONS[name]
        incident, reference_dt = _run_trace(
            orientation, False, cache_dir, threads, reuse
        )
        total, loaded_dt = _run_trace(orientation, True, cache_dir, threads, reuse)
        if reference_dt != loaded_dt:
            raise RuntimeError(f"time-step mismatch for orientation {name}")
        result = analyse_traces(orientation, incident, total, reference_dt)
        _write_csv(output_dir / f"{name.replace('+', 'p').replace('-', 'm')}_reflection.csv", result)
        results[name] = result

    metrics = {
        name: {
            key: value
            for key, value in result.items()
            if key
            in {
                "incident_peak_iteration",
                "reflection_peak_iteration",
                "magnitude_rmse",
                "magnitude_max_abs_error",
                "phase_rmse_degrees",
                "phase_max_abs_error_degrees",
                "complex_relative_l2_error",
            }
        }
        for name, result in results.items()
    }
    checks = {
        f"{name}_{metric}": {
            "value": values[metric],
            "maximum": maximum,
            "passed": values[metric] <= maximum,
        }
        for name, values in metrics.items()
        for metric, maximum in ACCEPTANCE_LIMITS.items()
    }
    summary = {
        "orientations": list(orientation_names),
        "dl_metres": DL,
        "source_frequency_hz": SOURCE_FREQUENCY,
        "frequency_band_hz": [float(FREQUENCIES[0]), float(FREQUENCIES[-1])],
        "surface_model_hash": benchmark_model().model_hash,
        "algorithm": (
            "bilinear ADE + half-time E centring + exterior half-cell displacement current"
        ),
        "runtime_seconds": perf_counter() - started,
        "metrics": metrics,
        "acceptance": {
            "passed": all(value["passed"] for value in checks.values()),
            "checks": checks,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("impedance_box_reflection_results"),
    )
    parser.add_argument(
        "--orientations",
        default=",".join(ORIENTATIONS),
        help="comma-separated face normals chosen from -x,+x,-y,+y,-z,+z",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    orientation_names = tuple(value.strip() for value in args.orientations.split(","))
    invalid = [value for value in orientation_names if value not in ORIENTATIONS]
    if not orientation_names or invalid:
        parser.error(f"invalid orientation(s): {', '.join(invalid) if invalid else 'none'}")
    summary = run_benchmark(
        args.output_dir,
        orientation_names=orientation_names,
        threads=args.threads,
        reuse=args.reuse,
    )
    print(json.dumps(summary, indent=2))
    if not summary["acceptance"]["passed"]:
        raise SystemExit("Impedance-box reflection benchmark failed")


if __name__ == "__main__":
    main()
