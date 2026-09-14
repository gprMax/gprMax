"""Run a target model with a simple source and receiver.

From the gprMax checkout:
    python -m gprMax.toolboxes.LandmineModels.examples.free_space --model PMN --resolution-mm 2

The default is a short CPU check, not a calibrated antenna or a buried-target
experiment. Edit build_model() to supply your own background, antenna and scan.
The HDF5 geometry and its JSON material database must stay beside each other.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

import gprMax

# 1. CHOOSE A TARGET. Paths are relative to this toolbox, not your working folder.
MODEL_NAMES = ("PMA", "PMN", "TS50", "can")
DATA_DIRECTORY = Path(__file__).resolve().parents[1]
PADDING_CELLS = 16
TIME_WINDOW_S = 3e-9


# 2. BUILD THE MODEL. The file determines the actual mesh spacing and array size.
def build_model(model="PMN", resolution_mm=2, *, background_er=1.0, time_window=TIME_WINDOW_S):
    """Insert one target into a uniform background; return an unexecuted Scene.

    background_er=1 means free space. The optional value also permits checking
    that transparent (-1) voxels preserve an existing background material.
    Distances supplied to gprMax below are in metres.
    """
    if model not in MODEL_NAMES or resolution_mm not in (1, 2):
        raise ValueError("Choose PMA, PMN, TS50 or can, at 1 or 2 mm")
    geometry = DATA_DIRECTORY / f"{model}_{resolution_mm}x{resolution_mm}x{resolution_mm}.h5"
    with h5py.File(geometry, "r") as handle:
        cells = np.asarray(handle["data"].shape)
        spacing = np.asarray(handle.attrs["dx_dy_dz"])
        database_name = str(handle.attrs["MaterialDatabase"])
    domain = (cells + 2 * PADDING_CELLS) * spacing
    origin = PADDING_CELLS * spacing

    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=tuple(domain)))
    scene.add(gprMax.Discretisation(p1=tuple(spacing)))
    scene.add(gprMax.TimeWindow(time=time_window))
    scene.add(gprMax.PMLThickness(thickness=6))
    scene.add(gprMax.OMPThreads(n=2))
    # Build the background BEFORE importing the target. A -1 voxel leaves it
    # alone; a stored air/plastic/metal voxel replaces it with its own material.
    scene.add(gprMax.Material(er=background_er, se=0, mr=1, sm=0, id="background"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=tuple(domain), material_id="background", averaging=False))
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=tuple(origin),
            geofile=geometry,
            material_database=database_name,  # JSON filename without .json, e.g. PMN_materials.
            averaging=False,
        )
    )

    # Place a theoretical source and receiver above the imported array and
    # outside the PML. These are example choices, not part of the stored target.
    source_cells = np.array(
        [PADDING_CELLS + cells[0] // 2, PADDING_CELLS + cells[1] // 2, PADDING_CELLS + cells[2] + 6]
    )
    receiver_cells = source_cells + np.array([4, 0, 0])
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(
            p1=tuple(source_cells * spacing), polarisation="x", waveform_id="pulse"
        )
    )
    scene.add(gprMax.Rx(p1=tuple(receiver_cells * spacing), id="probe", outputs=["Ex", "Ey", "Ez"]))
    return scene


# 3. EXECUTE AND READ THE OUTPUT. A fresh directory prevents accidental overwrites.
def run_example(model="PMN", resolution_mm=2, directory=None):
    directory = Path(directory or f"landmine_results/{model}_{resolution_mm}mm")
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "target"
    gprMax.run(
        scenes=[build_model(model, resolution_mm)],
        outputfile=output,
        hide_progress_bars=True,
        log_level=30,
    )
    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        field = handle["rxs/rx1/Ex"][...]
    if not np.isfinite(field).all() or not np.any(field):
        raise RuntimeError("Expected a finite, nonzero receiver signal")
    print(f"Completed {model} at {resolution_mm} mm: {output.with_suffix('.h5')}")
    print(f"Receiver Ex: {field.size} samples, peak |Ex| = {np.max(np.abs(field)):.6g} V/m")
    return output.with_suffix(".h5")


# 4. LAUNCH ONCE. Solver workers can import the builder without launching a run.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="PMN")
    parser.add_argument("--resolution-mm", type=int, choices=(1, 2), default=2)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    run_example(args.model, args.resolution_mm, args.directory)
