"""Reproduce the grounded-slab time-domain benchmark in Capoglu Fig. 11.

The published horizontal and vertical Hertzian dipoles are simulated at the
reported 0.1 mm FDTD resolution.  The independent reference evaluates the
closed grounded-slab echo series in Eqs. (59) and (63)--(65) of Capoglu's
2007 thesis and differentiates the prescribed Gaussian current analytically.
It does not call either gprMax layered propagation implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, physical_constants

import gprMax


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "capoglu_grounded_time"
DL = 0.1e-3
DOMAIN = (8e-3, 8e-3, 6e-3)
GROUND = 1e-3
SLAB_THICKNESS = 2e-3
INTERFACE = GROUND + SLAB_THICKNESS
SOURCE = (4e-3, 4e-3, GROUND + 1e-3)
ORIGIN = (SOURCE[0], SOURCE[1], INTERFACE)
ER_SLAB = 2.5
TAU = 5e-12
CENTRE_MULTIPLIER = 6.0
# The direct transform deliberately saves only times supported by every
# retained echo path. A common pre-delay moves the published waveform into
# that complete interval; it is removed again from the reported time axis.
VALIDATION_PREDELAY = 230e-12
THETA = 45.0
PHI = 45.0
Z0 = physical_constants["characteristic impedance of vacuum"][0]


def differentiated_gaussian(time: float) -> float:
    """Equation (99), normalised so its positive and negative peaks are one."""

    coordinate = (time - VALIDATION_PREDELAY - CENTRE_MULTIPLIER * TAU) / TAU
    return float(-coordinate * np.exp(-0.5 * coordinate**2 + 0.5))


def build_scene(polarisation: str) -> gprMax.Scene:
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=f"Capoglu Fig. 11 {polarisation}-dipole validation"))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    # The complete-output rule must accommodate the retained grounded-slab
    # echo train in addition to the plotted 0--100 ps reduced-time interval.
    scene.add(gprMax.TimeWindow(time=500e-12))
    scene.add(gprMax.Material(er=ER_SLAB, se=0, mr=1, sm=0, id="substrate"))
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(DOMAIN[0], DOMAIN[1], INTERFACE),
            material_id="substrate",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(DOMAIN[0], DOMAIN[1], GROUND),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Waveform(
            wave_type="user",
            user_func=differentiated_gaussian,
            id="differentiated_gaussian",
        )
    )
    scene.add(
        gprMax.HertzianDipole(
            p1=SOURCE,
            polarisation=polarisation,
            waveform_id="differentiated_gaussian",
        )
    )
    scene.add(
        gprMax.NTFFSurface(
            # As in thesis Fig. 11, the lower transform boundary coincides
            # with the PEC ground. Its face is omitted because the grounded
            # Green function already enforces the image cancellation there.
            p1=(2e-3, 2e-3, GROUND),
            p2=(6e-3, 6e-3, 4.5e-3),
            id="surface",
            origin=ORIGIN,
            omit_faces=("z0",),
        )
    )
    scene.add(
        gprMax.NTFFLayeredBackground(
            id="grounded_slab",
            axis="z",
            materials=("free_space", "substrate", "pec"),
            interfaces=(INTERFACE, GROUND),
        )
    )
    scene.add(
        gprMax.NTFFLayeredTimeTransform(
            surface_id="surface",
            id="transient",
            background_id="grounded_slab",
            impulse_tolerance=1e-5,
            max_impulses=10_000,
        )
    )
    scene.add(
        gprMax.NTFFLayeredTimeFarField(
            theta=THETA,
            phi=PHI,
            transform_id="transient",
            id="figure11",
            outputs=("Etheta",),
        )
    )
    return scene


def analytical_normalised_field(
    times: np.ndarray,
    polarisation: str,
    source_samples: np.ndarray,
    source_dt: float,
    source_offset: float,
    source_anchor: np.ndarray,
) -> np.ndarray:
    """Evaluate the thesis grounded-slab series without production helpers."""

    theta = np.deg2rad(THETA)
    phi = np.deg2rad(PHI)
    transverse_squared = np.sin(theta) ** 2
    relative_permittivity = np.asarray((1.0, ER_SLAB))
    axial_index = np.sqrt(relative_permittivity - transverse_squared)
    tm_impedance = axial_index / relative_permittivity
    transmission_10 = 2 * tm_impedance[0] / (tm_impedance[1] + tm_impedance[0])
    reflection_10 = (tm_impedance[0] - tm_impedance[1]) / (tm_impedance[0] + tm_impedance[1])

    source_position = np.asarray(source_anchor, dtype=float).copy()
    source_position["xyz".index(polarisation)] += 0.5 * DL
    source_depth = INTERFACE - source_position[2]
    reflection_number = np.arange(20)
    amplitudes = transmission_10 * (-reflection_10) ** reflection_number
    slowness = axial_index[1] / c
    upward_delays = (source_depth + 2 * reflection_number * SLAB_THICKNESS) * slowness
    ground_delays = (2 * SLAB_THICKNESS - source_depth + 2 * reflection_number * SLAB_THICKNESS) * slowness

    if polarisation == "y":
        response_amplitudes = tm_impedance[1] * np.concatenate((amplitudes, -amplitudes))
        angular_factor = -np.sin(phi)
    elif polarisation == "z":
        response_amplitudes = np.concatenate((amplitudes, amplitudes))
        angular_factor = np.sin(theta) / ER_SLAB
    else:
        raise ValueError("the Figure 11 reference supports y or z polarisation")

    delays = np.concatenate((upward_delays, ground_delays))
    direction = np.asarray((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)))
    delays -= np.dot(direction, source_position[:2] - np.asarray(ORIGIN[:2])) / c
    source_derivative = np.diff(source_samples) / source_dt
    derivative_times = source_offset + (np.arange(source_derivative.size) + 0.5) * source_dt
    response = np.zeros_like(times, dtype=float)
    for amplitude, delay in zip(response_amplitudes, delays):
        response += amplitude * np.interp(
            times - delay,
            derivative_times,
            source_derivative,
            left=0.0,
            right=0.0,
        )

    # The production response convention is twice the physical V/I Green
    # response. After Eq. (19) and the thesis normalisation by
    # (J0 / Delta) Z0 / r, the resulting coefficient is Delta/(4*pi*c).
    return DL * angular_factor * response / (4 * np.pi * c)


def _read_result(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with h5py.File(path, "r") as output:
        group = output["ntff/surface/time_far_field/figure11"]
        times = np.asarray(group["times"])
        # The source waveform has unit peak current. The published
        # normalisation therefore reduces to r E_theta / Z0 for gprMax's
        # one-cell Hertzian current element (J0 = I Delta).
        field = np.asarray(group["fields/Etheta"])[0] / Z0
        source = output["srcs/src1"]
        excitation = source["excitation"]
        source_data = {
            "samples": np.asarray(excitation["samples"]),
            "dt": float(excitation.attrs["SampleInterval"]),
            "offset": float(excitation.attrs["TimeSampleOffset"]),
            "anchor": np.asarray(source.attrs["Position"]),
        }
    return times, field, source_data


def compare(output_directory: Path) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    results = {}
    retained = {}
    for polarisation, label in (("y", "HED"), ("z", "VED")):
        times, actual, source = _read_result(output_directory / f"{label.lower()}.h5")
        reduced_times = times - VALIDATION_PREDELAY
        expected = analytical_normalised_field(
            times,
            polarisation,
            source["samples"],
            source["dt"],
            source["offset"],
            source["anchor"],
        )
        plotted = (reduced_times >= 0) & (reduced_times <= 100e-12)
        peak = float(np.max(np.abs(expected[plotted])))
        difference = actual - expected
        results[label] = {
            "maximum_error_normalised_to_analytical_peak": float(np.max(np.abs(difference[plotted])) / peak),
            "rms_error_normalised_to_analytical_peak": float(np.sqrt(np.mean(difference[plotted] ** 2)) / peak),
            "analytical_peak": peak,
        }
        retained[label] = (reduced_times[plotted], actual[plotted], expected[plotted])
        np.savetxt(
            output_directory / f"{label.lower()}.csv",
            np.column_stack(retained[label]),
            delimiter=",",
            header="reduced_time_s,Etheta_fdtd_normalised,Etheta_analytical_normalised",
            comments="",
        )

    results["passed"] = bool(
        all(
            item["maximum_error_normalised_to_analytical_peak"] < 0.08
            and item["rms_error_normalised_to_analytical_peak"] < 0.03
            for item in results.values()
        )
    )
    (output_directory / "summary.json").write_text(json.dumps(results, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.3), sharex=True)
    for axis, label in zip(axes, ("HED", "VED")):
        times, actual, expected = retained[label]
        axis.plot(times * 1e12, expected, "k-", lw=1.5, label="analytical echo series")
        axis.plot(
            times[::4] * 1e12,
            actual[::4],
            "ko",
            ms=3,
            markerfacecolor="none",
            label="gprMax FDTD + direct-time NTFF",
        )
        axis.set_ylabel(r"normalised $rE_\theta$")
        axis.set_title(label)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("reduced time (ps)")
    fig.suptitle(r"Capoglu thesis Fig. 11: grounded $\epsilon_r=2.5$ slab, " r"$\theta=\phi=45^\circ$")
    fig.tight_layout()
    fig.savefig(output_directory / "capoglu_grounded_time.png", dpi=220)
    plt.close(fig)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-run", action="store_true", help="analyse existing HDF5 outputs")
    parser.add_argument(
        "--gpu",
        type=int,
        help="run on the selected CUDA device instead of the CPU",
    )
    parser.add_argument(
        "--precision",
        choices=("single", "double"),
        default="double",
        help="CPU or accelerator field precision (default: double)",
    )
    args = parser.parse_args()
    if not args.no_run:
        args.output.mkdir(parents=True, exist_ok=True)
        for polarisation, label in (("y", "HED"), ("z", "VED")):
            solver_options = (
                {"gpu": [args.gpu], "gpu_precision": args.precision}
                if args.gpu is not None
                else {"cpu_precision": args.precision}
            )
            gprMax.run(
                scenes=[build_scene(polarisation)],
                n=1,
                outputfile=args.output / label.lower(),
                hide_progress_bars=True,
                **solver_options,
            )
    metrics = compare(args.output)
    print(json.dumps(metrics, indent=2))
    if not metrics["passed"]:
        raise SystemExit("Capoglu grounded-slab time-domain validation failed")


if __name__ == "__main__":
    main()
