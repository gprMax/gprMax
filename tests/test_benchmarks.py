"""A series of models with different domain sizes used for benchmarking.
    The domain is free space with a simple source (Hertzian Dipole) and
    receiver at the centre.
"""


import os
from pathlib import Path

import pytest

import gprMax

# Cube side lengths (in cells) for different domains
DOMAINS = [0.10] # [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# Number of OpenMP threads to benchmark each domain size
OMP_THREADS = [1, 2, 4, 8, 16, 32, 64, 128]

# Discretisation
dl = 0.001


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("omp_threads", OMP_THREADS)
def test_simple_benchmarks(request, benchmark, domain, omp_threads):

    output_dir = Path(os.path.dirname(request.fspath), "tmp", request.node.name)
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = output_dir / "model.h5"

    # Domain
    x = domain
    y = x
    z = x

    scene = gprMax.Scene()

    title = gprMax.Title(name=request.node.name)
    domain = gprMax.Domain(p1=(x, y, z))
    dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
    time_window = gprMax.TimeWindow(time=3e-9)
    wv = gprMax.Waveform(wave_type="gaussiandotnorm", amp=1, freq=900e6, id="MySource")
    src = gprMax.HertzianDipole(p1=(x / 2, y / 2, z / 2), polarisation="x", waveform_id="MySource")
    rx = gprMax.Rx(p1=(x / 4, y / 4, z / 4))

    omp = gprMax.OMPThreads(n=omp_threads)

    scenes = []
    scene.add(title)
    scene.add(domain)
    scene.add(dxdydz)
    scene.add(time_window)
    scene.add(wv)
    scene.add(src)
    scene.add(omp)
    scene.add(rx)
    scenes.append(scene)

    # Run benchmark once (i.e. 1 round)
    # benchmark.pedantic(gprMax.run, kwargs={'scenes': scenes, 'n': len(scenes), 'geometry_only': False, 'outputfile': output_filepath, 'gpu': None})
    
    # Automatically choose number of rounds.
    benchmark(gprMax.run, scenes=scenes, n=len(scenes), geometry_only=False, outputfile=output_filepath, gpu=None)
