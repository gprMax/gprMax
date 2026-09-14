"""Run two prescribed dielectric-block designs through the advanced API.

This is a fixed table demonstration: no optimiser chooses parameters and
no objective is calculated. build_model shows how physical width and
permittivity enter gprMax; run_example defines and executes the table.
For a copy-and-edit optimisation, start with examples/start_here.py.
"""

from pathlib import Path

from gprMax.toolboxes.Optimisation import (
    Campaign,
    LocalExecutor,
    ParameterSpace,
    PreparedModel,
    Problem,
    Real,
)


# 1. MODEL: use the candidate values and record any mesh rounding.
def build_model(parameters, scenario, context):
    """Use width/permittivity to build a Scene and record the realised geometry.

    Width is snapped to the 2 mm mesh here, as a model-building choice.
    scenario/context are the advanced builder arguments; this example needs
    neither additional scenario settings nor context-generated files.
    """
    import gprMax

    dl = 0.002
    width = round(parameters["width"] / dl) * dl
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.064, 0.064, 0.064)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=1.5e-9))
    scene.add(gprMax.PMLThickness(thickness=6))
    scene.add(gprMax.Material(er=parameters["permittivity"], se=0, mr=1, sm=0, id="block"))
    scene.add(
        gprMax.Box(p1=(0.028, 0.024, 0.024), p2=(0.028 + width, 0.040, 0.040), material_id="block")
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(p1=(0.016, 0.032, 0.032), polarisation="z", waveform_id="pulse")
    )
    scene.add(gprMax.Rx(p1=(0.050, 0.032, 0.032), id="received", outputs=["Ez"]))
    return PreparedModel(
        scene,
        effective_parameters={"width": width, "permittivity": parameters["permittivity"]},
        metadata={"block_start": [0.028, 0.024, 0.024], "block_end": [0.028 + width, 0.040, 0.040]},
    )


# 2. PARAMETER TABLE AND EXECUTION: these combinations are supplied by us.
def run_example(directory):
    """Declare the parameter space and run two complete parameter combinations.

    Return one row of RunResults per combination, with one entry per
    scenario. This makes the simulation stage usable without an optimiser.
    """
    problem = Problem(
        parameters=ParameterSpace({"width": Real(0.008, 0.016, "m"), "permittivity": Real(2, 8)}),
        builder="gprMax.toolboxes.Optimisation.examples.dielectric_block:build_model",
        dependencies=(Path(__file__),),
    )
    return Campaign(problem, directory, LocalExecutor()).evaluate(
        [
            {"width": 0.008, "permittivity": 4},
            {"width": 0.016, "permittivity": 6},
        ]
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New directory for campaign results")
    args = parser.parse_args()
    results = run_example(args.directory)
    for candidate in results:
        for run in candidate:
            print(run.candidate_id, run.scenario_id, run.status, run.output_file)
    raise SystemExit(0 if all(r.status == "complete" for c in results for r in c) else 1)
