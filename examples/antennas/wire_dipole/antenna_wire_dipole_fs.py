"""Half-wavelength wire dipole excited by a one-cell voltage-source port.

The source-owned port calculates and stores frequency, corrected complex
S11, input impedance, and input admittance under ``/ports/feed`` in the model
HDF5 output. No post-processing of source voltage and current is required.
"""

from pathlib import Path

import gprMax

fn = Path(__file__)

scene = gprMax.Scene()
scene.add(gprMax.Title(name=fn.stem))
scene.add(gprMax.Domain(p1=(0.050, 0.050, 0.200)))
scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
scene.add(gprMax.TimeWindow(time=60e-9))
scene.add(gprMax.Waveform(wave_type="gaussian", amp=1, freq=1e9, id="mypulse"))
scene.add(
    gprMax.Edge(
        p1=(0.025, 0.025, 0.025),
        p2=(0.025, 0.025, 0.175),
        material_id="pec",
    )
)
scene.add(
    gprMax.Edge(
        p1=(0.025, 0.025, 0.100),
        p2=(0.025, 0.025, 0.101),
        material_id="free_space",
    )
)
scene.add(
    gprMax.VoltageSource(
        p1=(0.025, 0.025, 0.100),
        polarisation="z",
        resistance=50,
        waveform_id="mypulse",
        id="feed",
    )
)
scene.add(
    gprMax.GeometryView(
        p1=(0.020, 0.020, 0.020),
        p2=(0.030, 0.030, 0.180),
        dl=(0.001, 0.001, 0.001),
        filename=fn.stem,
        output_type="f",
    )
)

gprMax.run(scenes=[scene], n=1, outputfile=fn)
