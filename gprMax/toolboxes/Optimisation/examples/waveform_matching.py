"""Match a simulated waveform to reference data using the simple public interface.

Copy this file AND reference_waveform.npz into your folder, then run:
    python waveform_matching.py

The bundled reference is synthetic, generated at width=0.012 m and permittivity=4.
For measured data, replace REFERENCE and the loading block inside evaluate().
See start_here.py for the smaller example without a reference data file.
"""

from pathlib import Path

import numpy as np

import gprMax
from gprMax.toolboxes.Optimisation import Real, optimise

# 1. DEFINE THE SEARCH PARAMETERS AND FIXED EXPERIMENT SETTINGS.
# Both variables belong to one candidate. Each proposed combination builds
# one model, whose waveform produces one score.
PARAMETERS = {
    "width": Real(0.008, 0.016, "m"),
    "permittivity": Real(2.0, 8.0),
}
CELL_SIZE_M = 0.002
REFERENCE = Path(__file__).with_name("reference_waveform.npz")
RESULTS = Path(__file__).with_name("waveform_results")  # New directory for each run.


# 2. USE THE TRIAL PARAMETERS IN YOUR MODEL.
def build_model(parameters):
    """Build one dielectric block using the proposed width and permittivity."""
    # The mesh is fixed in this example. Real widths are rounded to its cell size.
    # This geometry choice belongs to this model, not to the optimiser/toolbox.
    width_m = round(parameters["width"] / CELL_SIZE_M) * CELL_SIZE_M
    relative_permittivity = parameters["permittivity"]

    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.064, 0.064, 0.064)))
    scene.add(gprMax.Discretisation(p1=(CELL_SIZE_M, CELL_SIZE_M, CELL_SIZE_M)))
    scene.add(gprMax.TimeWindow(time=3e-9))
    scene.add(gprMax.PMLThickness(thickness=6))
    scene.add(gprMax.Material(er=relative_permittivity, se=0, mr=1, sm=0, id="block"))
    scene.add(
        gprMax.Box(
            p1=(0.028, 0.024, 0.024), p2=(0.028 + width_m, 0.040, 0.040), material_id="block"
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(p1=(0.016, 0.032, 0.032), polarisation="z", waveform_id="pulse")
    )
    scene.add(gprMax.Rx(p1=(0.050, 0.032, 0.032), id="probe", outputs=["Ez"]))
    return scene


# 3. PROCESS THE OUTPUT AND RETURN THE ERROR TO MINIMISE.
def evaluate(parameters, output):
    """Align simulated/reference samples, then return normalised RMS error.

    This function owns the comparison method. Edit it to change the objective;
    the same function works with every optimiser supported by the toolbox.
    """
    simulated = output.receiver("probe", "Ez")  # Matches the Rx definition above.

    # These keys belong to THIS reference file, not to the optimisation framework.
    # Replace this block to load your own measured data and convert its units.
    with np.load(REFERENCE, allow_pickle=False) as reference:
        time_s = reference["time_s"]
        target = reference["Ez"]
        unit = str(reference["unit"].item())

    # Reject incompatible data instead of calculating a misleading objective.
    if unit != simulated.unit:
        raise ValueError("Reference and simulated waveforms must use the same units")
    if (
        time_s.ndim != 1
        or time_s.size < 2
        or target.shape != time_s.shape
        or not np.isfinite(time_s).all()
        or not np.isfinite(target).all()
        or np.any(np.diff(time_s) <= 0)
    ):
        raise ValueError(
            "Reference needs increasing finite time samples and matching waveform values"
        )
    if time_s[0] < simulated.time[0] or time_s[-1] > simulated.time[-1]:
        raise ValueError(
            "The simulation must cover the reference time window; extrapolation is disabled"
        )

    # Match sampling times before subtracting. Add your filtering/windowing here.
    predicted = np.interp(time_s, simulated.time, simulated.values)
    residual = predicted - target
    reference_rms = float(np.sqrt(np.mean(target**2)))
    if reference_rms == 0:
        raise ValueError("Cannot normalise by a zero reference waveform")
    # THIS is the criterion. Divide by the reference RMS to make the error
    # dimensionless; the optimiser minimises this one number.
    score = float(np.sqrt(np.mean(residual**2))) / reference_rms

    # Optional: save intermediate arrays so you can inspect why a trial scored well.
    output.save_npz(
        "comparison.npz", time_s=time_s, reference=target, simulated=predicted, residual=residual
    )
    return score  # A dimensionless number; zero means an exact waveform match.


# 4. RUN. The functions above need no optimiser-specific or worker-specific code.
if __name__ == "__main__":
    result = optimise(
        parameters=PARAMETERS,
        model=build_model,
        objective=evaluate,
        directory=RESULTS,
        optimiser="rf",
        evaluations=12,
        seed=7,
        files=(REFERENCE,),  # Record the data file and detect changes during the run.
    )
