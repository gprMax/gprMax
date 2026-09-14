"""Copy this file and run: python thin_wire_dipole.py

Follow sections 1–4: parameters, model, objective, then launch settings.
The supporting resonance calculation is at the end, below the main workflow.
The old CLI benchmark is in examples/advanced/thin_wire_dipole.py.
"""

import math
from pathlib import Path

import numpy as np

import gprMax
from gprMax.toolboxes.Optimisation import Integer, LocalExecutor, optimise

# 1. CHOOSE WHAT MAY CHANGE AND WHAT YOU WANT TO ACHIEVE.
PARAMETERS = {"arm_cells": Integer(54, 83)}
TARGET_FREQUENCY_HZ = 1e9
DZ_M = 0.001  # Fixed 1 mm axial spacing throughout this search.
DX_M = DY_M = 299792458.0 / TARGET_FREQUENCY_HZ / 100
WIRE_RADIUS_M = 0.2 * DX_M  # Fixed radius; only the arms change length.
FEED_RESISTANCE_OHM = 50.0
TIME_WINDOW_S = 100e-9  # 100 ns gives approximately 10 MHz independent bins.
MAX_FREQUENCY_BIN_HZ = 10e6  # Data-resolution check, not an optimiser stop rule.
RESULTS = Path(__file__).with_name("dipole_results")  # New directory for each run.

# The optimiser proposes an integer number of cells PER ARM. With a one-cell gap:
# arm length = arm_cells * DZ_M; total length = (2 * arm_cells + 1) * DZ_M.
# These bounds give 109–167 mm total length, in symmetric increments of 2 mm.
# If you change DZ_M or the target frequency, review the length bounds as well.


# 2. BUILD ONE gprMax MODEL FROM THE TRIAL VALUES.
def build_model(parameters):
    """Use the proposed arm length; return the complete, unexecuted Scene."""
    arm_cells = parameters["arm_cells"]  # This is the optimisation-to-geometry link.
    arm_length_m = arm_cells * DZ_M

    # Use the longest allowed arm to size a fixed domain for the entire search.
    # Padding includes PML; its axial cell counts preserve the transverse thickness.
    axial_padding_cells = math.ceil(16 * DX_M / DZ_M)
    axial_pml_cells = math.ceil(8 * DX_M / DZ_M)
    centre_z_cells = PARAMETERS["arm_cells"].upper + axial_padding_cells
    x, y, feed_z = 20 * DX_M, 20 * DY_M, centre_z_cells * DZ_M

    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(40 * DX_M, 40 * DY_M, (2 * centre_z_cells + 1) * DZ_M)))
    scene.add(gprMax.Discretisation(p1=(DX_M, DY_M, DZ_M)))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW_S))
    scene.add(gprMax.PMLThickness(thickness=(8, 8, axial_pml_cells, 8, 8, axial_pml_cells)))

    # Two symmetric arms, with a one-cell feed gap between them. Do not put
    # ThinWire in that gap: doing so would short the voltage-source electric edge.
    scene.add(
        gprMax.ThinWire(
            p1=(x, y, feed_z - arm_length_m),
            p2=(x, y, feed_z),
            radius=WIRE_RADIUS_M,
        )
    )
    scene.add(
        gprMax.ThinWire(
            p1=(x, y, feed_z + DZ_M),
            p2=(x, y, feed_z + DZ_M + arm_length_m),
            radius=WIRE_RADIUS_M,
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=TARGET_FREQUENCY_HZ, id="pulse"))
    # The saved port name "feed" is chosen here and read in evaluate() below.
    scene.add(
        gprMax.VoltageSource(
            p1=(x, y, feed_z),
            polarisation="z",
            resistance=FEED_RESISTANCE_OHM,
            waveform_id="pulse",
            id="feed",
        )
    )
    return scene  # The toolbox executes it; do not call gprMax.run here.


# 3. SAY HOW TO SCORE THE COMPLETED SIMULATION: SMALLER IS BETTER.
def evaluate(parameters, output):
    """Minimise the distance of the S11 minimum from the target, measured in Hz.

    Change this function to change the optimisation objective. The model and
    objective functions are the same for TPE, RF, GA, PSO and DE.
    """
    spectrum = output.port("feed")  # Matches the VoltageSource id above.

    # Supporting calculation at the END of this file: estimate_resonance().
    # It checks the saved spectrum and estimates its dip between frequency bins.
    resonance_hz = estimate_resonance(spectrum, TARGET_FREQUENCY_HZ, MAX_FREQUENCY_BIN_HZ)
    # THIS is the criterion: closeness of the S11 dip to the target in Hz.
    # The helper estimates the dip; this line decides how that estimate is scored.
    frequency_error_hz = abs(resonance_hz - TARGET_FREQUENCY_HZ)

    # Optional: keep the spectrum so you can inspect the result of each trial.
    # These array names are this example's choices, not required toolbox fields.
    output.save_npz(
        "s11.npz",
        frequency_hz=spectrum.frequency,
        s11=spectrum.s11,
        valid=spectrum.valid,
        resonance_hz=resonance_hz,
    )
    return frequency_error_hz  # Zero means the estimated S11 minimum is at the target.


# 4. RUN THE SEARCH. Edit these settings without changing the two functions above.
def run_search():
    """Launch the public interface once; the small main guard is at the file end."""
    result = optimise(
        parameters=PARAMETERS,
        model=build_model,
        objective=evaluate,
        directory=RESULTS,
        optimiser="tpe",  # Also: "rf", "ga", "pso", "de".
        evaluations=12,  # A budget; reaching it does not establish convergence.
        seed=7,
        execution=LocalExecutor(cpu_threads=2, timeout=600),
    )
    best_length_m = (2 * result.best_parameters["arm_cells"] + 1) * DZ_M
    print(f"Best evaluated length: {best_length_m * 1e3:.1f} mm")
    print(f"S11-dip frequency error: {result.best_value / 1e6:.3f} MHz")
    return result


# SUPPORTING CALCULATION — called only by evaluate() in section 3.
# You can normally leave this unchanged. Edit it if you want a different way
# to locate resonance. It contains numerical checks, not optimiser-specific code.
def estimate_resonance(spectrum, target_hz, max_frequency_bin_hz):
    """Locate the S11 dip in 0.6–1.4 times the target using a three-bin power fit.

    Reject spectra with insufficient decay/resolution or an unresolved dip.
    Interpolation estimates a frequency within this numerical model; it does
    not improve independent frequency resolution or establish mesh convergence.
    """
    # These checks decide whether the simulated spectrum supports the
    # measurement. They are data-quality checks, not optimiser stopping rules.
    if np.isnan(spectrum.tail_relative_db) or spectrum.tail_relative_db > -40:
        raise ValueError("Terminal trace has not decayed below -40 dB; increase TIME_WINDOW_S")
    resolution_hz = spectrum.independent_frequency_resolution_hz
    if (
        resolution_hz is None
        or not np.isfinite(resolution_hz)
        or resolution_hz <= 0
        or resolution_hz > max_frequency_bin_hz * (1 + 1e-6)
    ):
        raise ValueError("Frequency bins are too coarse; increase TIME_WINDOW_S")

    # Locate a valid, interior minimum in the chosen search band.
    frequency = spectrum.frequency
    band = np.flatnonzero((frequency >= 0.6 * target_hz) & (frequency <= 1.4 * target_hz))
    if len(band) < 3 or not spectrum.valid[band].all():
        raise ValueError("Need at least three valid S11 bins throughout the resonance search band")
    index = int(band[np.argmin(np.abs(spectrum.s11[band]))])
    if index in (band[0], band[-1]):
        raise ValueError("S11 minimum is at the search-band edge; review the length bounds or band")

    # Fit reflected power at the minimum and its two neighbours. This retains
    # the earlier dipole example's requirement for a resolved dip below -10 dB.
    frequencies = frequency[index - 1 : index + 2].astype(float)
    power = np.abs(spectrum.s11[index - 1 : index + 2].astype(complex)) ** 2
    if not (power[1] < power[0] and power[1] < power[2]) or power[1] > 0.1:
        raise ValueError("Need a resolved interior S11 dip below -10 dB")
    x = (frequencies - frequencies[1]) / resolution_hz  # Scale GHz values before fitting.
    a, b, _ = np.polyfit(x, power, 2)
    vertex = -b / (2 * a) if a > 0 else float("nan")
    if not np.isfinite(vertex) or not x[0] < vertex < x[2]:
        raise ValueError("Invalid local S11-minimum fit")
    return float(frequencies[1] + vertex * resolution_hz)


# Define the helper above before launching. Workers load this file without running this guard.
if __name__ == "__main__":
    run_search()
