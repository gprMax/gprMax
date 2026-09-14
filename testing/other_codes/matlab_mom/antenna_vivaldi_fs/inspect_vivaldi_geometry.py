"""Check the actual gprMax fine-geometry export and plot its PEC edges."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from vivaldi_antenna_gprmax import MESHES, ORIGIN, RESULTS, edge_masks, read_geometry


def inspect(results, mesh):
    metadata = json.loads((results / f"vivaldi_{mesh}.json").read_text())
    dl = np.asarray(metadata["discretisation_m"])
    geometry = read_geometry()
    expected = edge_masks(geometry, dl)
    with h5py.File(results / f"vivaldi_{mesh}_geometry.vtkhdf", "r") as output:
        vtk = output["VTKHDF"]
        material = np.asarray(vtk["CellData/Material"])
        connections = np.asarray(vtk["Connectivity"]).reshape(-1, 2)
        points = np.asarray(vtk["Points"])
        segments = points[connections[material == 0]]
    delta = segments[:, 1] - segments[:, 0]
    axes = np.argmax(np.abs(delta), axis=1)
    if np.any(axes == 2) or not np.allclose(segments[:, :, 2], ORIGIN[2], atol=1e-8, rtol=0):
        raise AssertionError("PEC edges must lie in the xy sheet; no Ez or thickness")
    observed = [np.zeros_like(mask) for mask in expected]
    for axis in (0, 1):
        selected = segments[axes == axis]
        nodes = np.rint(np.min(selected, axis=1) / dl).astype(int)
        observed[axis][nodes[:, 0], nodes[:, 1]] = True
        np.testing.assert_array_equal(observed[axis], expected[axis])
    feed_node = np.rint(np.asarray(metadata["feed_node_m"]) / dl).astype(int)
    i, j, _ = feed_node
    assert not observed[1][i, j]
    assert observed[1][i, j - 1] and observed[1][i, j + 1]

    fig, axes_plot = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]})
    xy = (segments[:, :, :2] - ORIGIN[:2]) * 1e3
    feed_x = geometry["feed_x_m"] * 1e3
    for ax in axes_plot:
        ax.add_collection(LineCollection(xy, colors="0.25", linewidths=0.35))
        for loop in geometry["boundary_loops_xy_m"]:
            loop = np.asarray(loop) * 1e3
            loop = np.vstack((loop, loop[0]))
            ax.plot(loop[:, 0], loop[:, 1], color="tab:purple", linewidth=0.8)
        ax.plot([feed_x, feed_x], [-0.25, 0.25], color="tab:red", linewidth=2)
        ax.set_aspect("equal")
        ax.set_xlabel("MATLAB x (mm)")
        ax.set_ylabel("y (mm)")
    axes_plot[0].set(
        xlim=(-155, 155),
        ylim=(-67, 67),
        title="MATLAB outline (purple), actual gprMax PEC edges (grey), voltage gap (red)",
    )
    axes_plot[1].set(xlim=(feed_x - 6, feed_x + 6), ylim=(-2, 2), title="Feed detail")
    fig.tight_layout()
    fig.savefig(results / f"vivaldi_{mesh}_geometry_comparison.png", dpi=180)
    plt.close(fig)
    report = dict(
        mesh=mesh,
        inspected_file=f"vivaldi_{mesh}_geometry.vtkhdf",
        pec_ex_edges=int(observed[0].sum()),
        pec_ey_edges=int(observed[1].sum()),
        pec_ez_edges=0,
        all_pec_edges_match_expected=True,
        feed_gap_unshorted=True,
        feed_banks_connected=True,
    )
    (results / f"vivaldi_{mesh}_geometry_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", choices=tuple(MESHES), default="fine")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    inspect(args.results_dir, args.mesh)
