"""Small, reusable model runners for the example notebooks.

Generated input and output files are kept in the operating system's temporary
directory, so executing a notebook does not modify the source tree.
"""

import shutil
import tempfile
from pathlib import Path

import gprMax

NOTEBOOK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = NOTEBOOK_DIR.parents[1]
OUTPUT_DIR = Path(tempfile.gettempdir()) / "gprMax-notebooks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _copy_input(filename):
    """Copy an organised example input file to the notebook workspace."""

    source = REPOSITORY_ROOT / "examples" / "gpr" / "basic" / filename
    destination = OUTPUT_DIR / filename
    shutil.copyfile(source, destination)
    return destination


def ascan_input():
    """Return a temporary copy of the 2-D cylinder A-scan input file."""

    return _copy_input("cylinder_Ascan_2D.in")


def bscan_input():
    """Return a temporary copy of the 2-D cylinder B-scan input file."""

    return _copy_input("cylinder_Bscan_2D.in")


def ensure_ascan(force=False):
    """Run the small A-scan example if its output is not already available."""

    output = OUTPUT_DIR / "cylinder_Ascan_2D.h5"
    if force or not output.exists():
        gprMax.run(
            inputfile=ascan_input(),
            outputfile=output,
            n=1,
            hide_progress_bars=True,
        )
    return output


def ensure_bscan(force=False, traces=60):
    """Run and merge the cylinder B-scan example when required."""

    from toolboxes.Utilities.outputfiles_merge import merge_files

    merged = OUTPUT_DIR / "cylinder_Bscan_2D_merged.h5"
    if force or not merged.exists():
        base = OUTPUT_DIR / "cylinder_Bscan_2D.h5"
        if force:
            for output in OUTPUT_DIR.glob("cylinder_Bscan_2D[0-9]*.h5"):
                output.unlink()
            merged.unlink(missing_ok=True)
        gprMax.run(
            inputfile=bscan_input(),
            outputfile=base,
            n=traces,
            hide_progress_bars=True,
        )
        outputs = [OUTPUT_DIR / f"cylinder_Bscan_2D{i}.h5" for i in range(1, traces + 1)]
        merge_files([str(output) for output in outputs], merged_outputfile=merged)
    return merged


def ensure_port_demo(force=False):
    """Run a compact voltage-source/RxPort model for plotting demonstrations."""

    output = OUTPUT_DIR / "wire_dipole_port_demo.h5"
    if force or not output.exists():
        scene = gprMax.Scene()
        scene.add(gprMax.Title(name="Notebook wire-dipole port demonstration"))
        scene.add(gprMax.Domain(p1=(0.08, 0.08, 0.08)))
        scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
        scene.add(gprMax.TimeWindow(time=4e-9))
        scene.add(gprMax.Waveform(wave_type="gaussian", amp=1, freq=1.5e9, id="pulse"))
        scene.add(
            gprMax.Edge(
                p1=(0.04, 0.04, 0.024),
                p2=(0.04, 0.04, 0.056),
                material_id="pec",
            )
        )
        scene.add(
            gprMax.Edge(
                p1=(0.04, 0.04, 0.04),
                p2=(0.04, 0.04, 0.042),
                material_id="free_space",
            )
        )
        scene.add(
            gprMax.VoltageSource(
                p1=(0.04, 0.04, 0.04),
                polarisation="z",
                resistance=50,
                waveform_id="pulse",
            )
        )
        scene.add(gprMax.RxPort(p1=(0.04, 0.04, 0.04), id="feed"))
        gprMax.run(scenes=[scene], outputfile=output, n=1, hide_progress_bars=True)
    return output
