from pathlib import Path

import gprMax

# File path for output
fn = Path(__file__)

# Discretisation
dl = 0.001

# Domain
x = 0.200
y = 0.200
z = 0.100

scene = gprMax.Scene()

title = gprMax.Title(name=fn.with_suffix("").name)
domain = gprMax.Domain(p1=(x, y, z))
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
time_window = gprMax.TimeWindow(time=30e-9)

scene.add(title)
scene.add(domain)
scene.add(dxdydz)
scene.add(time_window)

bowtie_dims = (0.050, 0.100)  # Length, height
tx_pos = (x / 2, y / 2, z / 2)

# Source excitation and type
wave = gprMax.Waveform(wave_type="gaussian", amp=1, freq=1.5e9, id="mypulse")
tl = gprMax.TransmissionLine(
    p1=(tx_pos[0], tx_pos[1], tx_pos[2]), polarisation="x", resistance=50, waveform_id="mypulse"
)
scene.add(wave)
scene.add(tl)

# Bowtie - upper x half
t1 = gprMax.Triangle(
    p1=(tx_pos[0], tx_pos[1], tx_pos[2]),
    p2=(tx_pos[0] + bowtie_dims[0] + 2 * dl, tx_pos[1] - bowtie_dims[1] / 2, tx_pos[2]),
    p3=(tx_pos[0] + bowtie_dims[0] + 2 * dl, tx_pos[1] + bowtie_dims[1] / 2, tx_pos[2]),
    thickness=0,
    material_id="pec",
)

# Bowtie - lower x half
t2 = gprMax.Triangle(
    p1=(tx_pos[0] + dl, tx_pos[1], tx_pos[2]),
    p2=(tx_pos[0] - bowtie_dims[0], tx_pos[1] - bowtie_dims[1] / 2, tx_pos[2]),
    p3=(tx_pos[0] - bowtie_dims[0], tx_pos[1] + bowtie_dims[1] / 2, tx_pos[2]),
    thickness=0,
    material_id="pec",
)

scene.add(t1)
scene.add(t2)

# Detailed geometry view around bowtie and feed position
gv1 = gprMax.GeometryView(
    p1=(tx_pos[0] - bowtie_dims[0] - 2 * dl, tx_pos[1] - bowtie_dims[1] / 2 - 2 * dl, tx_pos[2] - 2 * dl),
    p2=(tx_pos[0] + bowtie_dims[0] + 2 * dl, tx_pos[1] + bowtie_dims[1] / 2 + 2 * dl, tx_pos[2] + 2 * dl),
    dl=(dl, dl, dl),
    filename="antenna_bowtie_fs_pcb",
    output_type="f",
)
scene.add(gv1)

# Run model
gprMax.run(scenes=[scene], geometry_only=False, outputfile=fn, gpu=None)
