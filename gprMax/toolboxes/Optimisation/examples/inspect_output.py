"""Inspect an existing gprMax HDF5 file without running another simulation.

Examples (replace model.h5 with your output file):
    python inspect_output.py model.h5
    python inspect_output.py model.h5 --group frills
    python inspect_output.py model.h5 --receiver probe --component Ez
    python inspect_output.py model.h5 --terminal frills/frill1 --frequency 3.1e9

The inventory shows what was actually saved. Optional reader calls demonstrate
how to use the same public readers outside an optimisation objective.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from gprMax.toolboxes.Optimisation import read_port, read_receiver


# 1. CHOOSE THE FILE AND, OPTIONALLY, A RECEIVER OR TERMINAL.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--group", default="/", help="HDF5 group to list")
    parser.add_argument("--grid", default="/", help="Main grid or explicit subgrid for the reader")
    parser.add_argument("--receiver", help="Receiver ID from your model")
    parser.add_argument("--component", default="Ez")
    parser.add_argument("--terminal", help="For example ports/feed or frills/frill1")
    parser.add_argument(
        "--frequency", type=float, help="Frequency in Hz at which to read complex S11"
    )
    args = parser.parse_args()
    if args.frequency is not None and args.terminal is None:
        parser.error("--frequency requires --terminal")

    # 2. LIST NAMES, SHAPES AND METADATA. The helper below does not load arrays.
    inspect_group(args.file, args.group)

    # 3. READ ONLY THE QUANTITY YOU SELECTED.
    if args.receiver:
        trace = read_receiver(args.file, args.receiver, args.component, grid=args.grid)
        print(f"Receiver: {trace.dataset}; {trace.values.size} samples; unit={trace.unit}")
        print(f"Time: {trace.time[0]:.6g} to {trace.time[-1]:.6g} s")
        print(f"Peak magnitude: {np.max(np.abs(trace.values)):.6g} {trace.unit}")
    if args.terminal:
        spectrum = read_port(args.file, args.terminal, grid=args.grid)
        print(f"Terminal: {spectrum.group}; Zref={spectrum.reference_impedance} ohm")
        print(f"Valid S11 bins: {np.count_nonzero(spectrum.valid)}/{spectrum.frequency.size}")
        print(
            f"Independent frequency resolution: {spectrum.independent_frequency_resolution_hz} Hz"
        )
        if args.frequency is not None:
            reflection = spectrum.at(args.frequency)
            print(f"S11 at {args.frequency:g} Hz: {reflection} (complex)")
            print(f"S11 magnitude: {20 * np.log10(max(abs(reflection), 1e-12)):.6f} dB")


# SUPPORTING HELPER: a raw inventory is useful before writing your objective.
def inspect_group(filename, group_name):
    """Print dataset metadata while leaving the HDF5 arrays on disk."""
    with h5py.File(filename, "r") as handle:
        group = handle[group_name]
        print(f"{group.name}: attributes={dict(group.attrs)}")

        def describe(name, item):
            if isinstance(item, h5py.Dataset):
                print(
                    f"  {item.name}: shape={item.shape}, dtype={item.dtype}, attributes={dict(item.attrs)}"
                )
            elif item.attrs:
                print(f"  {item.name}: attributes={dict(item.attrs)}")

        group.visititems(describe)


if __name__ == "__main__":
    main()
