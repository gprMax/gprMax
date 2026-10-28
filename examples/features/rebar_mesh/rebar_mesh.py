"""Two-layer steel reinforcement mesh inside a concrete slab.

The model uses a 2 mm grid and a 1.5 GHz Ricker source. The upper bar layer
runs along x, the lower layer runs along y, with 20 mm cover below the top
concrete surface. Results are written beside this script.
"""

from pathlib import Path

import gprMax

output = Path(__file__).resolve().parent / "rebar_mesh_results"

scene = gprMax.Scene()
scene.add(gprMax.Title(name="Reinforced concrete slab"))
scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
scene.add(gprMax.Domain(p1=(0.24, 0.18, 0.16)))
scene.add(gprMax.PMLThickness(thickness=10))
scene.add(gprMax.TimeWindow(time=3e-9))
scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e9, id="w"))
scene.add(gprMax.Material(er=6, se=0.01, mr=1, sm=0, id="concrete"))
scene.add(
    gprMax.Box(
        p1=(0.01, 0.01, 0.01),
        p2=(0.23, 0.17, 0.11),
        material_id="concrete",
    )
)
scene.add(
    gprMax.RebarMesh(
        p1=(0.02, 0.02, 0.02),
        p2=(0.22, 0.16, 0.10),
        spacing=(0.04, 0.04),
        diameter=0.008,
        cover=0.02,
        layer_spacing=0.04,
        material_id="pec",
        tag="reinforcement",
    )
)
scene.add(
    gprMax.HertzianDipole(
        polarisation="z",
        p1=(0.12, 0.09, 0.12),
        waveform_id="w",
    )
)
scene.add(gprMax.Rx(p1=(0.16, 0.09, 0.12), id="rx1"))

gprMax.run(scenes=[scene], n=1, outputfile=output)
