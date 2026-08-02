# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

"""Run reproducible end-to-end validations of the reusable KSIR interface.

Available cases are a Hertzian-dipole far-field pattern, CPU PEC-sphere Mie
scattering and broadband backscatter sweep, a finite-distance time-domain
comparison with direct receivers, and CPU/accelerator time-domain parity.
Results are written as JSON; an optional summary plot can be generated without
storing simulation output.

Examples::

    python -m testing.validation.validate_ntff --case all --backend cpu
    python -m testing.validation.validate_ntff --case dipole --backend cuda
    python -m testing.validation.validate_ntff --case parity --backend opencl
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from scipy.constants import c
from scipy.signal import find_peaks

import gprMax
from testing.validation.mie_pec import pec_sphere_bistatic_rcs

ACCELERATOR_ARGUMENTS = {"cuda": "gpu", "opencl": "opencl", "metal": "metal"}

MIE_FREQUENCY = 5e9
MIE_RADIUS = 0.0096
MIE_CENTRE = (0.048, 0.048, 0.048)
MIE_ANGLES = np.arange(0.0, 181.0, 5.0)
MIE_SWEEP_SIZE_PARAMETERS = np.arange(0.3, 4.0001, 0.1)
MIE_SWEEP_FREQUENCIES = MIE_SWEEP_SIZE_PARAMETERS * c / (2 * np.pi * MIE_RADIUS)

NEAR_FREQUENCY = 4e9
NEAR_DL = 0.002
NEAR_SOURCE = (0.05, 0.05, 0.05)
NEAR_OBSERVATION_X = np.asarray((0.062, 0.070, 0.078))


def _solver_options(backend, device, precision):
    if backend == "cpu":
        return {"cpu_precision": precision}
    return {
        ACCELERATOR_ARGUMENTS[backend]: [device],
        "gpu_precision": precision,
    }


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _collection_backend(outputfile, path):
    with h5py.File(outputfile.with_suffix(".h5"), "r") as output:
        backend = output[path].attrs["collection_backend"]
    return backend.decode() if isinstance(backend, bytes) else str(backend)


def _run(scene, outputfile, *, backend, device, precision):
    start = perf_counter()
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        log_level=logging.WARNING,
        **_solver_options(backend, device, precision),
    )
    return perf_counter() - start


def dipole_scene(threads=4):
    """Build the closed-form Hertzian-dipole validation scene."""

    dl = 0.002
    frequency = 5e9
    centre = (0.05, 0.05, 0.05)
    e_theta = np.arange(0.0, 181.0, 2.0)
    h_phi = np.arange(0.0, 361.0, 2.0)

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl,) * 3))
    scene.add(gprMax.Domain(p1=(0.1,) * 3))
    scene.add(gprMax.TimeWindow(time=1e-9))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(gprMax.PMLThickness(thickness=10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=frequency, id="pulse"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=centre, waveform_id="pulse"))
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.034,) * 3,
            p2=(0.066,) * 3,
            id="dipole_surface",
            origin=centre,
        )
    )
    transform = gprMax.KSIRFrequencyTransform(
        "dipole_surface",
        "dipole_spectrum",
        (frequency,),
        save_surface_dft=False,
    )
    e_plane = gprMax.KSIRFarField(
        theta=e_theta,
        phi=np.zeros(e_theta.shape),
        transform_id="dipole_spectrum",
        id="e_plane",
        outputs=("Etheta",),
    )
    h_plane = gprMax.KSIRFarField(
        theta=np.full(h_phi.shape, 90.0),
        phi=h_phi,
        transform_id="dipole_spectrum",
        id="h_plane",
        outputs=("Etheta",),
    )
    for item in (transform, e_plane, h_plane):
        scene.add(item)
    return scene, transform, e_plane, h_plane, e_theta, h_phi


def run_dipole_validation(workdir, *, backend, device, precision, threads):
    scene, _, e_plane, h_plane, e_theta, h_phi = dipole_scene(threads)
    outputfile = workdir / f"dipole_{backend}_{precision}"
    runtime = _run(
        scene,
        outputfile,
        backend=backend,
        device=device,
        precision=precision,
    )

    e_magnitude = np.abs(e_plane.result.fields["Etheta"][0])
    peak = np.max(e_magnitude)
    e_normalized = e_magnitude / peak
    h_normalized = np.abs(h_plane.result.fields["Etheta"][0]) / peak
    analytic = np.abs(np.sin(np.deg2rad(e_theta)))
    interior = slice(1, -1)
    nonzero_h = h_normalized[h_normalized > np.finfo(h_normalized.dtype).eps]

    return {
        "backend": backend,
        "collection_backend": _collection_backend(
            outputfile, "ntff/dipole_surface/frequency/dipole_spectrum"
        ),
        "runtime_seconds": runtime,
        "theta_degrees": e_theta,
        "e_plane_normalized_Etheta": e_normalized,
        "analytic_abs_sin_theta": analytic,
        "e_plane_rms_error": float(
            np.sqrt(np.mean((e_normalized[interior] - analytic[interior]) ** 2))
        ),
        "e_plane_maximum_error": float(np.max(np.abs(e_normalized[interior] - analytic[interior]))),
        "pole_maximum": float(max(e_normalized[0], e_normalized[-1])),
        "phi_degrees": h_phi,
        "h_plane_normalized_Etheta": h_normalized,
        "h_plane_rms_deviation_from_unity": float(np.sqrt(np.mean((h_normalized - 1) ** 2))),
        "h_plane_peak_to_peak_ripple_db": float(
            20 * np.log10(np.max(nonzero_h) / np.min(nonzero_h))
        ),
    }


def mie_scene(threads=4):
    """Build the CPU TFSF/PEC-sphere Mie validation scene."""

    dl = 0.0016
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl,) * 3))
    scene.add(gprMax.Domain(p1=(0.096,) * 3))
    scene.add(gprMax.TimeWindow(iterations=900))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(gprMax.PMLThickness(thickness=10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=MIE_FREQUENCY, id="plane_pulse"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=(0.032,) * 3,
            p2=(0.064,) * 3,
            axis="x",
            psi=90,
            waveform_id="plane_pulse",
        )
    )
    scene.add(gprMax.Sphere(p1=MIE_CENTRE, r=MIE_RADIUS, material_id="pec"))
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.024,) * 3,
            p2=(0.072,) * 3,
            id="mie_surface",
            origin=MIE_CENTRE,
        )
    )
    transform = gprMax.KSIRFrequencyTransform(
        "mie_surface",
        "mie_spectrum",
        (MIE_FREQUENCY,),
        save_surface_dft=False,
        plane_wave_index=0,
    )
    far_field = gprMax.KSIRFarField(
        theta=np.full(MIE_ANGLES.shape, 90.0),
        phi=MIE_ANGLES,
        transform_id="mie_spectrum",
        id="mie_pattern",
        outputs=("Etheta", "Ephi", "rcs"),
    )
    scene.add(transform)
    scene.add(far_field)
    return scene, transform, far_field


def run_mie_validation(workdir, *, precision, threads):
    scene, _, far_field = mie_scene(threads)
    outputfile = workdir / f"mie_cpu_{precision}"
    runtime = _run(
        scene,
        outputfile,
        backend="cpu",
        device=0,
        precision=precision,
    )

    simulated = np.asarray(far_field.result.fields["rcs"][0])
    analytic = pec_sphere_bistatic_rcs(
        MIE_FREQUENCY,
        MIE_RADIUS,
        np.deg2rad(MIE_ANGLES),
        polarisation="perpendicular",
    )
    simulated_pattern = simulated / np.max(simulated)
    analytic_pattern = analytic / np.max(analytic)
    copolar = np.abs(far_field.result.fields["Etheta"][0])
    crosspolar = np.abs(far_field.result.fields["Ephi"][0])

    return {
        "backend": "cpu",
        "collection_backend": _collection_backend(
            outputfile, "ntff/mie_surface/frequency/mie_spectrum"
        ),
        "runtime_seconds": runtime,
        "scattering_angle_degrees": MIE_ANGLES,
        "simulated_rcs_m2": simulated,
        "analytic_mie_rcs_m2": analytic,
        "simulated_normalized_pattern": simulated_pattern,
        "analytic_normalized_pattern": analytic_pattern,
        "pattern_rms_error": float(np.sqrt(np.mean((simulated_pattern - analytic_pattern) ** 2))),
        "absolute_rcs_relative_l2_error": float(
            np.linalg.norm(simulated - analytic) / np.linalg.norm(analytic)
        ),
        "forward_scatter_relative_error": float(abs(simulated[0] - analytic[0]) / analytic[0]),
        "backscatter_relative_error": float(abs(simulated[-1] - analytic[-1]) / analytic[-1]),
        "maximum_crosspolar_ratio": float(np.max(crosspolar) / np.max(copolar)),
    }


def mie_sweep_scene(threads=4):
    """Build a broadband CPU backscatter-resonance validation scene."""

    dl = 0.0016
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl,) * 3))
    scene.add(gprMax.Domain(p1=(0.096,) * 3))
    scene.add(gprMax.TimeWindow(iterations=900))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(gprMax.PMLThickness(thickness=10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="broadband_plane_pulse"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=(0.032,) * 3,
            p2=(0.064,) * 3,
            axis="x",
            psi=90,
            waveform_id="broadband_plane_pulse",
        )
    )
    scene.add(gprMax.Sphere(p1=MIE_CENTRE, r=MIE_RADIUS, material_id="pec"))
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.024,) * 3,
            p2=(0.072,) * 3,
            id="mie_sweep_surface",
            origin=MIE_CENTRE,
        )
    )
    transform = gprMax.KSIRFrequencyTransform(
        "mie_sweep_surface",
        "mie_sweep_spectrum",
        MIE_SWEEP_FREQUENCIES,
        save_surface_dft=False,
        plane_wave_index=0,
    )
    backscatter = gprMax.KSIRFarField(
        theta=(90.0,),
        phi=(180.0,),
        transform_id="mie_sweep_spectrum",
        id="backscatter",
        outputs=("rcs",),
    )
    scene.add(transform)
    scene.add(backscatter)
    return scene, transform, backscatter


def _mie_backscatter(frequencies):
    return np.asarray(
        [
            pec_sphere_bistatic_rcs(
                frequency,
                MIE_RADIUS,
                (np.pi,),
                polarisation="perpendicular",
            )[0]
            for frequency in frequencies
        ]
    )


def _resonance_locations(normalized_rcs, prominence_db=0.15):
    values_db = 10 * np.log10(normalized_rcs)
    peaks, _ = find_peaks(values_db, prominence=prominence_db)
    nulls, _ = find_peaks(-values_db, prominence=prominence_db)
    return {
        "peaks": MIE_SWEEP_SIZE_PARAMETERS[peaks],
        "nulls": MIE_SWEEP_SIZE_PARAMETERS[nulls],
    }


def _frequency_band_errors(error_db):
    bands = ((0.3, 1.5), (1.5, 2.8), (2.8, 4.01))
    results = []
    for lower, upper in bands:
        selected = (MIE_SWEEP_SIZE_PARAMETERS >= lower) & (MIE_SWEEP_SIZE_PARAMETERS < upper)
        values = error_db[selected]
        results.append(
            {
                "size_parameter_min": lower,
                "size_parameter_max": min(upper, 4.0),
                "rms_error_db": float(np.sqrt(np.mean(values**2))),
                "maximum_absolute_error_db": float(np.max(np.abs(values))),
            }
        )
    return results


def run_mie_sweep_validation(workdir, *, precision, threads):
    scene, _, backscatter = mie_sweep_scene(threads)
    outputfile = workdir / f"mie_sweep_cpu_{precision}"
    runtime = _run(
        scene,
        outputfile,
        backend="cpu",
        device=0,
        precision=precision,
    )

    simulated = np.asarray(backscatter.result.fields["rcs"][:, 0])
    analytic = _mie_backscatter(MIE_SWEEP_FREQUENCIES)
    projected_area = np.pi * MIE_RADIUS**2
    simulated_normalized = simulated / projected_area
    analytic_normalized = analytic / projected_area
    error_db = 10 * np.log10(simulated / analytic)

    return {
        "backend": "cpu",
        "collection_backend": _collection_backend(
            outputfile, "ntff/mie_sweep_surface/frequency/mie_sweep_spectrum"
        ),
        "runtime_seconds": runtime,
        "size_parameter": MIE_SWEEP_SIZE_PARAMETERS,
        "frequency_hz": MIE_SWEEP_FREQUENCIES,
        "wavelength_cells": c / (MIE_SWEEP_FREQUENCIES * 0.0016),
        "simulated_rcs_m2": simulated,
        "analytic_mie_rcs_m2": analytic,
        "simulated_normalized_rcs": simulated_normalized,
        "analytic_normalized_rcs": analytic_normalized,
        "error_db": error_db,
        "overall_rms_error_db": float(np.sqrt(np.mean(error_db**2))),
        "band_errors": _frequency_band_errors(error_db),
        "simulated_resonances": _resonance_locations(simulated_normalized),
        "analytic_resonances": _resonance_locations(analytic_normalized),
    }


def near_field_scene(threads=4, *, time_origin="simulation"):
    """Build collocated direct-receiver and advanced-time KSIR outputs."""

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(NEAR_DL,) * 3))
    scene.add(gprMax.Domain(p1=(0.1,) * 3))
    scene.add(gprMax.TimeWindow(time=1e-9))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(gprMax.PMLThickness(thickness=10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=NEAR_FREQUENCY, id="pulse"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=NEAR_SOURCE, waveform_id="pulse"))

    points = []
    for index, x_position in enumerate(NEAR_OBSERVATION_X, start=1):
        receiver_position = (float(x_position), NEAR_SOURCE[1], NEAR_SOURCE[2])
        scene.add(
            gprMax.Rx(
                p1=receiver_position,
                outputs=["Ez"],
                id=f"near_field_{index}",
            )
        )
        # Rx samples Ez at the z-directed Yee-edge centre.
        points.append(
            (
                receiver_position[0],
                receiver_position[1],
                receiver_position[2] + 0.5 * NEAR_DL,
            )
        )

    scene.add(
        gprMax.NTFFSurface(
            p1=(0.042,) * 3,
            p2=(0.058,) * 3,
            id="near_surface",
            origin=NEAR_SOURCE,
        )
    )
    receiver = gprMax.KSIRTimeRx(
        position=points,
        surface_id="near_surface",
        id="near_field",
        outputs=("Ez",),
        time_origin=time_origin,
    )
    scene.add(receiver)
    return scene, receiver, np.asarray(points)


def _waveform_metrics(direct, reconstructed, dt):
    peak = np.max(np.abs(direct))
    significant = np.abs(direct) > 0.02 * peak
    difference = reconstructed - direct
    correlation = np.corrcoef(reconstructed[significant], direct[significant])[0, 1]
    cross_correlation = np.correlate(reconstructed, direct, mode="full")
    lag = int(np.argmax(cross_correlation) - (direct.size - 1))
    return {
        "relative_l2_error_significant": float(
            np.linalg.norm(difference[significant]) / np.linalg.norm(direct[significant])
        ),
        "relative_l2_error_full": float(np.linalg.norm(difference) / np.linalg.norm(direct)),
        "correlation_significant": float(correlation),
        "peak_amplitude_relative_error": float(abs(np.max(np.abs(reconstructed)) - peak) / peak),
        "cross_correlation_lag_samples": lag,
        "cross_correlation_lag_seconds": float(lag * dt),
    }


def run_near_field_validation(workdir, *, backend, device, precision, threads):
    scene, receiver, points = near_field_scene(threads, time_origin="simulation")
    outputfile = workdir / f"near_field_{backend}_{precision}"
    runtime = _run(
        scene,
        outputfile,
        backend=backend,
        device=device,
        precision=precision,
    )

    direct = []
    with h5py.File(outputfile.with_suffix(".h5"), "r") as output:
        dt = float(output.attrs["dt"])
        for index in range(1, NEAR_OBSERVATION_X.size + 1):
            direct.append(output[f"rxs/rx{index}/Ez"][:])
    direct = np.asarray(direct)
    reconstructed = np.asarray(receiver.result.fields["Ez"][:, : direct.shape[1]])
    metrics = [
        _waveform_metrics(expected, actual, dt) for expected, actual in zip(direct, reconstructed)
    ]
    source_position = np.asarray((NEAR_SOURCE[0], NEAR_SOURCE[1], NEAR_SOURCE[2] + 0.5 * NEAR_DL))

    return {
        "backend": backend,
        "collection_backend": _collection_backend(outputfile, "ntff/near_surface/time/near_field"),
        "runtime_seconds": runtime,
        "dt_seconds": dt,
        "points_m": points,
        "distance_m": np.linalg.norm(points - source_position, axis=1),
        "distance_wavelengths": np.linalg.norm(points - source_position, axis=1)
        / (c / NEAR_FREQUENCY),
        "metrics": metrics,
    }


def _parity_metrics(expected, actual):
    difference = actual - expected
    scale = np.max(np.abs(expected))
    return {
        "relative_l2_error": float(np.linalg.norm(difference) / np.linalg.norm(expected)),
        "maximum_error_normalized_to_peak": float(np.max(np.abs(difference)) / scale),
    }


def run_accelerator_parity(workdir, *, backend, device, precision, threads):
    if backend == "cpu":
        raise ValueError("accelerator parity requires cuda, opencl, or metal")

    cpu_scene, cpu_receiver, _ = near_field_scene(threads, time_origin="first_arrival")
    device_scene, device_receiver, _ = near_field_scene(threads, time_origin="first_arrival")
    cpu_output = workdir / f"parity_cpu_{precision}"
    device_output = workdir / f"parity_{backend}_{precision}"
    cpu_runtime = _run(
        cpu_scene,
        cpu_output,
        backend="cpu",
        device=0,
        precision=precision,
    )
    device_runtime = _run(
        device_scene,
        device_output,
        backend=backend,
        device=device,
        precision=precision,
    )

    expected = cpu_receiver.result.fields["Ez"]
    actual = device_receiver.result.fields["Ez"]
    point_metrics = []
    for point, valid_length in enumerate(cpu_receiver.result.valid_lengths):
        stop = int(valid_length)
        point_metrics.append(_parity_metrics(expected[point, :stop], actual[point, :stop]))

    return {
        "backend": backend,
        "cpu_collection_backend": _collection_backend(
            cpu_output, "ntff/near_surface/time/near_field"
        ),
        "device_collection_backend": _collection_backend(
            device_output, "ntff/near_surface/time/near_field"
        ),
        "runtime_seconds": {"cpu": cpu_runtime, backend: device_runtime},
        "valid_lengths": cpu_receiver.result.valid_lengths,
        "time_origins_seconds": cpu_receiver.result.time_origins,
        "point_metrics": point_metrics,
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _plot(results, filename):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [
        name for name in ("dipole", "mie", "mie_sweep", "near_field", "parity") if name in results
    ]
    figure, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, available):
        result = results[name]
        if name == "dipole":
            axis.plot(
                result["theta_degrees"],
                result["e_plane_normalized_Etheta"],
                label="gprMax",
            )
            axis.plot(
                result["theta_degrees"],
                result["analytic_abs_sin_theta"],
                "--",
                label="analytic",
            )
            axis.set(xlabel="Theta [deg]", ylabel="Normalized magnitude", title="Dipole")
        elif name == "mie":
            axis.semilogy(
                result["scattering_angle_degrees"],
                result["simulated_rcs_m2"],
                label="gprMax",
            )
            axis.semilogy(
                result["scattering_angle_degrees"],
                result["analytic_mie_rcs_m2"],
                "--",
                label="Mie",
            )
            axis.set(xlabel="Scattering angle [deg]", ylabel="RCS [m2]", title="PEC sphere")
        elif name == "mie_sweep":
            axis.semilogy(
                result["size_parameter"],
                result["simulated_normalized_rcs"],
                label="gprMax",
            )
            axis.semilogy(
                result["size_parameter"],
                result["analytic_normalized_rcs"],
                "--",
                label="Mie",
            )
            axis.set(
                xlabel="Size parameter ka",
                ylabel="Normalized backscatter RCS",
                title="PEC-sphere sweep",
            )
        elif name == "near_field":
            errors = [item["relative_l2_error_significant"] for item in result["metrics"]]
            axis.plot(result["distance_wavelengths"], errors, "o-")
            axis.set(
                xlabel="Distance [wavelengths]",
                ylabel="Relative L2 error",
                title="Near field",
            )
        else:
            errors = [item["relative_l2_error"] for item in result["point_metrics"]]
            axis.bar(np.arange(len(errors)) + 1, errors)
            axis.set(xlabel="Point", ylabel="Relative L2 error", title="CPU/device parity")
        axis.grid(True, alpha=0.3)
        if name in ("dipole", "mie", "mie_sweep"):
            axis.legend()
    figure.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_selected(args):
    results = {
        "configuration": {
            "case": args.case,
            "backend": args.backend,
            "device": None if args.backend == "cpu" else args.device,
            "precision": args.precision,
            "threads": args.threads,
        }
    }
    with tempfile.TemporaryDirectory(prefix="gprmax_ntff_validation_") as temporary:
        workdir = Path(temporary)
        if args.case in ("dipole", "all"):
            results["dipole"] = run_dipole_validation(
                workdir,
                backend=args.backend,
                device=args.device,
                precision=args.precision,
                threads=args.threads,
            )
        if args.case in ("mie", "all"):
            if args.backend == "cpu":
                results["mie"] = run_mie_validation(
                    workdir, precision=args.precision, threads=args.threads
                )
            elif args.case == "mie":
                raise ValueError("the TFSF Mie case is currently CPU-only")
            else:
                results["mie_skipped"] = "the TFSF Mie case is currently CPU-only"
        if args.case in ("mie-sweep", "all"):
            if args.backend == "cpu":
                results["mie_sweep"] = run_mie_sweep_validation(
                    workdir, precision=args.precision, threads=args.threads
                )
            elif args.case == "mie-sweep":
                raise ValueError("the TFSF Mie sweep is currently CPU-only")
            else:
                results["mie_sweep_skipped"] = "the TFSF Mie sweep is currently CPU-only"
        if args.case in ("near-field", "all"):
            results["near_field"] = run_near_field_validation(
                workdir,
                backend=args.backend,
                device=args.device,
                precision=args.precision,
                threads=args.threads,
            )
        if args.case in ("parity", "all"):
            if args.backend != "cpu":
                results["parity"] = run_accelerator_parity(
                    workdir,
                    backend=args.backend,
                    device=args.device,
                    precision=args.precision,
                    threads=args.threads,
                )
            elif args.case == "parity":
                raise ValueError("parity requires cuda, opencl, or metal")
            else:
                results["parity_skipped"] = "parity requires cuda, opencl, or metal"
    return results


def _parser():
    parser = argparse.ArgumentParser(
        description="Validate reusable KSIR outputs against closed-form or direct references."
    )
    parser.add_argument(
        "--case",
        choices=("dipole", "mie", "mie-sweep", "near-field", "parity", "all"),
        default="all",
    )
    parser.add_argument("--backend", choices=("cpu", "cuda", "opencl", "metal"), default="cpu")
    parser.add_argument("--device", type=_nonnegative_int, default=0)
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--threads", type=_positive_int, default=4)
    parser.add_argument("--output", type=Path, default=Path("ntff_validation_results.json"))
    parser.add_argument("--plot", type=Path)
    return parser


def main():
    args = _parser().parse_args()
    results = run_selected(args)
    serialisable = _json_ready(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serialisable, indent=2) + "\n", encoding="utf-8")
    if args.plot is not None:
        _plot(results, args.plot)
    print(json.dumps(serialisable, indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
