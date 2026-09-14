"""Plot the small study examples from their authoritative HDF5 outputs."""

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(kind, prefix):
    prefix = Path(prefix)
    figure, axis = plt.subplots(layout="constrained")
    if kind == "port":
        with h5py.File(f"{prefix}_study.h5", "r") as output:
            frequency = output["frequency"][...]
            matrix = output["S"][...]
            valid = output["valid_S"][...].astype(bool)
            ids = output["port_ids"].asstr()[...]
        for row, output_id in enumerate(ids):
            for col, input_id in enumerate(ids):
                values = np.where(valid[:, row, col], np.abs(matrix[:, row, col]), np.nan)
                axis.plot(frequency / 1e9, values, label=f"{output_id} <- {input_id}")
        axis.set(xlabel="Frequency [GHz]", ylabel="|S| (linear)")
    elif kind == "plane_wave":
        # The fixed requests contain both observation directions. Each case
        # selects its own backscatter direction; the other request is bistatic.
        for index, name in ((1, "back_x"), (2, "back_y")):
            with h5py.File(f"{prefix}{index}.h5", "r") as output:
                transform = output["ntff/surface/frequency/band"]
                values = transform[f"far_field/{name}/fields/rcs"][...]
                axis.plot(transform["frequencies"][...] / 1e9, np.asarray(values).reshape(-1), "o", label=name)
        axis.set(xlabel="Frequency [GHz]", ylabel="Backscatter RCS [m²]")
    else:
        cases = (("", "passive receiver"),) if kind == "passive" else (("1", "feed only"), ("2", "opposed"))
        for suffix, label in cases:
            with h5py.File(f"{prefix}{suffix}.h5", "r") as output:
                port = output["ports/receive"]
                axis.plot(port["time"][...] * 1e9, port["Vtotal"][...], label=label)
        axis.set(xlabel="Time [ns]", ylabel="Receive gap voltage [V]")
    axis.legend()
    filename = prefix.with_name(prefix.name + "_comparison.png")
    figure.savefig(filename, dpi=150)
    plt.close(figure)
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("passive", "source", "port", "plane_wave"))
    parser.add_argument("prefix", type=Path)
    args = parser.parse_args()
    print(plot_results(args.kind, args.prefix))
