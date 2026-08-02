"""TFSF plane-wave scattering from a dielectric sphere.

This is the Python API equivalent of ``dielectric_sphere_tfsf.in``. The
receiver inside the TFSF box samples total field; the receiver outside the box
samples scattered field.
"""

from pathlib import Path

import gprMax

fn = Path(__file__)

scene = gprMax.Scene()
scene.add(gprMax.Title(name="TFSF plane wave scattering from a dielectric sphere"))
scene.add(gprMax.Domain(p1=(0.160, 0.160, 0.160)))
scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
scene.add(gprMax.TimeWindow(time=3e-9))
scene.add(gprMax.PMLThickness(thickness=8))
scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id="dielectric"))
scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e9, id="incident"))
scene.add(gprMax.Sphere(p1=(0.080, 0.080, 0.080), r=0.016, material_id="dielectric"))
scene.add(
    gprMax.DiscretePlaneWaveVector(
        p1=(0.040, 0.040, 0.040),
        p2=(0.120, 0.120, 0.120),
        m_vec=(1, 0, 0),
        psi=90,
        waveform_id="incident",
    )
)
scene.add(gprMax.Rx(p1=(0.060, 0.080, 0.080), id="total_field"))
scene.add(gprMax.Rx(p1=(0.130, 0.080, 0.080), id="scattered_field"))
scene.add(
    gprMax.GeometryView(
        p1=(0, 0, 0),
        p2=(0.160, 0.160, 0.160),
        dl=(0.002, 0.002, 0.002),
        filename="dielectric_sphere_tfsf",
        output_type="n",
    )
)

gprMax.run(scenes=[scene], n=1, outputfile=fn)
