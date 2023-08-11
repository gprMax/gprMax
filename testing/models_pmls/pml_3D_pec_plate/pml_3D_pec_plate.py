from pathlib import Path

import numpy as np

import gprMax

# File path for output
fn = Path(__file__)
parts = fn.parts

# Discretisation
dl = 0.001

# Domain
x = 0.051
y = 0.126
z = 0.026

domain = gprMax.Domain(p1=(x, y, z))
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
time_window = gprMax.TimeWindow(iterations=2100)
tssf = gprMax.TimeStepStabilityFactor(f=0.99)

waveform = gprMax.Waveform(wave_type="gaussiandotnorm", amp=1, freq=9.42e9, id="mypulse")
hertzian_dipole = gprMax.HertzianDipole(polarisation="z", p1=(0.013, 0.013, 0.014), waveform_id="mypulse")
rx = gprMax.Rx(p1=(0.038, 0.114, 0.013))

plate = gprMax.Plate(p1=(0.013, 0.013, 0.013), p2=(0.038, 0.113, 0.013), material_id="pec")

gv1 = gprMax.GeometryView(
    p1=(0, 0, 0),
    p2=(x, y, z),
    dl=(dl, dl, dl),
    filename=Path(*parts[:-1], f"{parts[-1]}_n"),
    output_type="n",
)
gv2 = gprMax.GeometryView(
    p1=(0, 0, 0),
    p2=(x, y, z),
    dl=(dl, dl, dl),
    filename=Path(*parts[:-1], f"{parts[-1]}_f"),
    output_type="f",
)

pmls = {
    "CFS-PML": {
        "pml": gprMax.PMLProps(formulation="HORIPML", thickness=10),
        # Parameters from http://dx.doi.org/10.1109/TAP.2018.2823864
        "pml_cfs": [
            gprMax.PMLCFS(
                alphascalingprofile="constant",
                alphascalingdirection="forward",
                alphamin=0.05,
                alphamax=0.05,
                kappascalingprofile="quartic",
                kappascalingdirection="forward",
                kappamin=1,
                kappamax=8,
                sigmascalingprofile="quartic",
                sigmascalingdirection="forward",
                sigmamin=0,
                sigmamax=1.1 * ((4 + 1) / (150 * np.pi * dl)),
            )
        ],
    },
    "HORIPML-1": {
        "pml": gprMax.PMLProps(formulation="HORIPML", thickness=10),
        # Parameters from http://dx.doi.org/10.1109/TAP.2011.2180344
        "pml_cfs": [
            gprMax.PMLCFS(
                alphascalingprofile="constant",
                alphascalingdirection="forward",
                alphamin=0,
                alphamax=0,
                kappascalingprofile="quartic",
                kappascalingdirection="forward",
                kappamin=1,
                kappamax=12,
                sigmascalingprofile="quartic",
                sigmascalingdirection="forward",
                sigmamin=0,
                sigmamax=0.7 * ((4 + 1) / (150 * np.pi * dl)),
            )
        ],
    },
    "HORIPML-2": {
        "pml": gprMax.PMLProps(formulation="HORIPML", thickness=10),
        # Parameters from http://dx.doi.org/10.1109/TAP.2018.2823864
        "pml_cfs": [
            gprMax.PMLCFS(
                alphascalingprofile="constant",
                alphascalingdirection="forward",
                alphamin=0,
                alphamax=0,
                kappascalingprofile="constant",
                kappascalingdirection="forward",
                kappamin=1,
                kappamax=1,
                sigmascalingprofile="sextic",
                sigmascalingdirection="forward",
                sigmamin=0,
                sigmamax=0.275 / (150 * np.pi * dl),
            ),
            gprMax.PMLCFS(
                alphascalingprofile="sextic",
                alphascalingdirection="forward",
                alphamin=0.07,
                alphamax=0.07 + (0.275 / (150 * np.pi * dl)),
                kappascalingprofile="cubic",
                kappascalingdirection="forward",
                kappamin=1,
                kappamax=8,
                sigmascalingprofile="quadratic",
                sigmascalingdirection="forward",
                sigmamin=0,
                sigmamax=2.75 / (150 * np.pi * dl),
            ),
        ],
    },
    "MRIPML-1": {
        "pml": gprMax.PMLProps(formulation="MRIPML", thickness=10),
        # Parameters from Antonis' MATLAB script (M3Dparams.m)
        "pml_cfs": [
            gprMax.PMLCFS(
                alphascalingprofile="constant",
                alphascalingdirection="forward",
                alphamin=0.05,
                alphamax=0.05,
                kappascalingprofile="quartic",
                kappascalingdirection="forward",
                kappamin=1,
                kappamax=8,
                sigmascalingprofile="quartic",
                sigmascalingdirection="forward",
                sigmamin=0,
                sigmamax=1.1 * ((4 + 1) / (150 * np.pi * dl)),
            )
        ],
    },
    "MRIPML-2": {
        "pml": gprMax.PMLProps(formulation="MRIPML", thickness=10),
        # Parameters from http://dx.doi.org/10.1109/TAP.2018.2823864
        "pml_cfs": [
            gprMax.PMLCFS(
                alphascalingprofile="quadratic",
                alphascalingdirection="reverse",
                alphamin=0,
                alphamax=0.15,
                kappascalingprofile="quartic",
                kappascalingdirection="forward",
                kappamin=1,
                kappamax=12,
                sigmascalingprofile="quartic",
                sigmascalingdirection="forward",
                sigmamin=0,
                sigmamax=0.65 * ((4 + 1) / (150 * np.pi * dl)),
            ),
            gprMax.PMLCFS(
                alphascalingprofile="linear",
                alphascalingdirection="reverse",
                alphamin=0.07,
                alphamax=0.8,
                kappascalingprofile="constant",
                kappascalingdirection="forward",
                kappamin=0,
                kappamax=0,
                sigmascalingprofile="quadratic",
                sigmascalingdirection="forward",
                sigmamin=0,
                sigmamax=0.65 * ((2 + 1) / (150 * np.pi * dl)),
            ),
        ],
    },
}

scenes = []
for k, v in pmls.items():
    scene = gprMax.Scene()
    title = gprMax.Title(name=fn.with_suffix("").name + "_" + k)
    scene.add(title)
    scene.add(domain)
    scene.add(dxdydz)
    scene.add(time_window)
    scene.add(tssf)
    scene.add(waveform)
    scene.add(hertzian_dipole)
    scene.add(rx)
    # scene.add(gv1)
    # scene.add(gv2)

    scene.add(v["pml"])
    for pml_cfs in v["pml_cfs"]:
        scene.add(pml_cfs)

    scenes.append(scene)

# Run model
gprMax.run(scenes=scenes, n=len(pmls), geometry_only=False, outputfile=fn)
