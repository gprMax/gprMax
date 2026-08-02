"""2D TM eigenmode source in a dielectric slab waveguide.

This is the Python API equivalent of ``dielectric_slab_2d_tm.in``. Set
``geometry_only=True`` in the final call first to inspect the solved
modal-field plot before performing the time-domain simulation.
"""

from pathlib import Path

import gprMax

fn = Path(__file__)
inf = float("inf")

scene = gprMax.Scene()
scene.add(gprMax.Title(name=fn.stem))
scene.add(gprMax.DomainMode(mode="TM"))
scene.add(gprMax.Domain(p1=(0.24, 0.08, inf)))
scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
scene.add(gprMax.TimeWindow(time=1.8e-9))
scene.add(gprMax.PMLThickness(thickness=(5, 5, 0, 5, 5, 0)))
scene.add(gprMax.Material(er=9, se=0, mr=1, sm=0, id="slab_core"))
scene.add(
    gprMax.Waveform(
        wave_type="contsine",
        amp=1,
        freq=5e9,
        id="eig_pulse",
    )
)
scene.add(
    gprMax.Box(
        p1=(0, 0.03, 0),
        p2=(0.24, 0.05, inf),
        material_id="slab_core",
    )
)
scene.add(
    gprMax.EigenmodeSource(
        normal="x",
        direction="+",
        p1=(0.005, 0),
        p2=(0.075, inf),
        w=0.02,
        mode_index=0,
        frequency=5e9,
        waveform_id="eig_pulse",
    )
)

for x in (0.08, 0.14, 0.20):
    scene.add(gprMax.Rx(p1=(x, 0.04, inf)))

for time, label in (
    (4e-10, "400ps"),
    (1e-9, "1000ps"),
    (1.6e-9, "1600ps"),
):
    scene.add(
        gprMax.Snapshot(
            p1=(0, 0, 0),
            p2=(0.24, 0.08, inf),
            dl=(0.001, 0.001, 0.001),
            time=time,
            filename=f"dielectric_slab_2d_tm_{label}",
            fileext=".h5",
        )
    )

# For a first inspection, add ``geometry_only=True`` to this call. The solved
# modal-field plot is then written without running the FDTD time loop.
gprMax.run(scenes=[scene], n=1, outputfile=fn)
