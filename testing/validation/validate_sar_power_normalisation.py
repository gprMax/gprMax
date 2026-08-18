"""Validate SAR incident/accepted-port-power normalization invariants.

This is an end-to-end production-path consistency validation rather than an
independent electromagnetic reference problem. It checks the quadratic field
scaling and the relationship between incident- and accepted-power normalized
SAR using an actual voltage source and :class:`RxPort` output.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import gprMax

FREQUENCIES = (0.75e9, 1.0e9, 1.25e9)


def build_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.024, 0.024, 0.024)))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.TimeWindow(time=8e-9))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Material(er=4, se=0.5, mr=1, sm=0, id="tissue"))
    scene.add(gprMax.MaterialDensity(density=1000, material_ids="tissue"))
    scene.add(
        gprMax.Box(
            p1=(0.004, 0.004, 0.004),
            p2=(0.020, 0.020, 0.020),
            material_id="tissue",
            tag="target",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"))
    scene.add(
        gprMax.VoltageSource(
            p1=(0.012, 0.012, 0.012),
            polarisation="z",
            resistance=50,
            waveform_id="pulse",
        )
    )
    scene.add(gprMax.RxPort(p1=(0.012, 0.012, 0.012), id="feed"))
    outputs = {}
    for output_id, normalisation, power in (
        ("incident_1W", "incident_power", 1.0),
        ("incident_4W", "incident_power", 4.0),
        ("accepted_1W", "accepted_power", 1.0),
    ):
        outputs[output_id] = gprMax.SAR(
            frequencies=FREQUENCIES,
            waveform_id="pulse",
            tags="target",
            id=output_id,
            normalisation=normalisation,
            port_id="feed",
            target_power=power,
            averaging_masses=(),
        )
        scene.add(outputs[output_id])
    return scene, outputs


def _maximum_relative(actual, expected):
    retained = np.isfinite(actual) & np.isfinite(expected) & (expected != 0)
    return float(np.max(np.abs(actual[retained] / expected[retained] - 1)))


def run(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    scene, outputs = build_scene()
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output_dir / "sar_power_normalisation",
        hide_progress_bars=True,
        cpu_precision="double",
        log_level=logging.WARNING,
    )
    incident_1w = outputs["incident_1W"].result
    incident_4w = outputs["incident_4W"].result
    accepted_1w = outputs["accepted_1W"].result
    four_watt_error = _maximum_relative(incident_4w.sar, 4 * incident_1w.sar)
    expected_accepted_ratio = (incident_1w.normalising_power / accepted_1w.normalising_power)[
        :, np.newaxis
    ]
    accepted_error = _maximum_relative(accepted_1w.sar, incident_1w.sar * expected_accepted_ratio)
    metrics = {
        "frequencies_hz": list(FREQUENCIES),
        "incident_power_before_normalisation_W": (incident_1w.normalising_power.tolist()),
        "accepted_power_before_normalisation_W": (accepted_1w.normalising_power.tolist()),
        "maximum_relative_error_4W_over_1W": four_watt_error,
        "maximum_relative_error_accepted_vs_incident_identity": accepted_error,
    }
    (output_dir / "sar_power_normalisation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    tolerance = 5e-12
    if four_watt_error > tolerance or accepted_error > tolerance:
        raise AssertionError(f"SAR power-normalisation validation failed: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("sar_power_normalisation_results"))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
