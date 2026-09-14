"""Copy this file to your own folder and run: python start_here.py

Edit the four numbered sections. The toolbox chooses trial values, runs your
model and sends the number returned by evaluate() back to the optimiser.
For a first check, use simulate() as shown in the README before a longer search.
"""

from pathlib import Path

import numpy as np

import gprMax
from gprMax.toolboxes.Optimisation import Real, optimise

# 1. CHOOSE WHAT MAY CHANGE AND WHAT YOU WANT TO ACHIEVE.
# The key "permittivity" will also be used in build_model() below.
PARAMETERS = {"permittivity": Real(2.0, 8.0)}
TARGET_PEAK_V_PER_M = 50.0
RESULTS = Path(__file__).with_name("peak_field_results")  # Use a new name for each run.


# 2. BUILD ONE gprMax MODEL FROM THE TRIAL VALUES.
def build_model(parameters):
    """Receive e.g. {"permittivity": 4.2}; return a complete, unexecuted Scene.

    Replace this body with your model. The toolbox calls it for every candidate.
    All distances below are metres and the time window is in seconds.
    """
    relative_permittivity = parameters["permittivity"]  # This is the optimisation link.

    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.064, 0.064, 0.064)))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.TimeWindow(time=3e-9))
    scene.add(gprMax.PMLThickness(thickness=6))
    scene.add(gprMax.Material(er=relative_permittivity, se=0, mr=1, sm=0, id="block"))
    scene.add(gprMax.Box(p1=(0.028, 0.024, 0.024), p2=(0.040, 0.040, 0.040), material_id="block"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(p1=(0.016, 0.032, 0.032), polarisation="z", waveform_id="pulse")
    )
    # The name "probe" and component "Ez" are selected here, then read below.
    scene.add(gprMax.Rx(p1=(0.050, 0.032, 0.032), id="probe", outputs=["Ez"]))
    return scene  # The toolbox calls gprMax.run; you do not call it here.


# 3. SAY HOW TO SCORE THE COMPLETED SIMULATION: SMALLER IS BETTER.
def evaluate(parameters, output):
    """Read the simulated waveform and measure distance from the desired peak.

    Replace this calculation with your own objective. You can filter data,
    compute a spectrum or use an existing postprocessor on output.file.
    'parameters' contains the same trial values used to build this model.
    """
    trace = output.receiver("probe", "Ez")
    peak_v_per_m = float(np.max(np.abs(trace.values)))
    # THIS is the optimisation criterion. The optimiser receives this scalar,
    # not the entire waveform; change this formula to express your own target.
    error_v_per_m = abs(peak_v_per_m - TARGET_PEAK_V_PER_M)
    return error_v_per_m


# 4. RUN THE SEARCH. Change these settings without editing the functions above.
# This guard lets solver workers load the functions without starting another search.
if __name__ == "__main__":
    result = optimise(
        parameters=PARAMETERS,
        model=build_model,
        # Pass the function itself. The toolbox calls it after each model finishes.
        objective=evaluate,
        directory=RESULTS,
        optimiser="tpe",  # Also: "rf", "ga", "pso", "de".
        evaluations=12,  # Maximum number of candidate models to evaluate.
        seed=7,
        # Optional: target_value=1.0 stops when peak error is <= 1 V/m.
        # Without a target, this example uses the evaluation limit.
    )
    # The toolbox prints progress and the best result. These are also available as:
    # result.best_parameters, result.best_value, result.trials
