"""A series of models with different domain sizes used for benchmarking.
    The domain is free space with a simple source (Hertzian Dipole) and
    receiver at the centre.
"""

from pathlib import Path
import gprMax

# File path for output
fn = Path(__file__)

# Cube side lengths (in cells) for different domains
domains = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# Number of OpenMP threads to benchmark each domain size
ompthreads = [1, 2, 4, 8, 16, 32, 64, 128]

scenes = []

for d in domains:
    for threads in ompthreads:
        # Discretisation
        dl = 0.001

        # Domain
        x = d
        y = x
        z = x

        scene = gprMax.Scene()

        title = gprMax.Title(name=fn.with_suffix("").name)
        domain = gprMax.Domain(p1=(x, y, z))
        dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
        time_window = gprMax.TimeWindow(time=3e-9)
        wv = gprMax.Waveform(wave_type="gaussiandotnorm", amp=1, freq=900e6, id="MySource")
        src = gprMax.HertzianDipole(p1=(x / 2, y / 2, z / 2), polarisation="x", waveform_id="MySource")

        omp = gprMax.OMPThreads(n=threads)

        scene.add(title)
        scene.add(domain)
        scene.add(dxdydz)
        scene.add(time_window)
        scene.add(wv)
        scene.add(src)
        scene.add(omp)
        scenes.append(scene)

# Run model
gprMax.run(scenes=scenes, n=len(scenes), geometry_only=False, outputfile=fn, gpu=None)
