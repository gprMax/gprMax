"""Subprocess helper for actual MPI SAR/radiometry and snapshot regressions."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def solve(args):
    import gprMax
    from gprMax.model import Model
    from gprMax.mode2d import mode2d_geometry
    from unittest.mock import patch

    geometry = mode2d_geometry(args.mode)
    dimensions = np.asarray((0.020, 0.024, 0.028))
    lower, upper, source = (
        np.asarray((0.004,) * 3),
        np.asarray((0.016, 0.020, 0.024)),
        np.asarray((0.010, 0.012, 0.014)),
    )
    if geometry is not None:
        dimensions[geometry.invariant_axis] = np.inf
        lower[geometry.invariant_axis] = upper[geometry.invariant_axis] = source[geometry.invariant_axis] = np.inf
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.002,) * 3))
    if geometry is not None:
        scene.add(gprMax.DomainMode(mode=geometry.polarisation))
    scene.add(gprMax.Domain(p1=tuple(dimensions)))
    scene.add(gprMax.TimeWindow(time=8e-11))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Material(er=4, se=0.5, mr=1, sm=0, id="tissue"))
    scene.add(gprMax.MaterialDensity(density=1000, material_ids="tissue"))
    scene.add(gprMax.Box(p1=tuple(lower), p2=tuple(upper), material_id="tissue", tag="target"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"))
    component = "Ez" if geometry is None else geometry.active_electric[0]
    scene.add(gprMax.HertzianDipole(p1=tuple(source), polarisation=component[-1].lower(), waveform_id="pulse"))
    for output, name, normalisation in (
        (gprMax.SAR, "dose", "waveform"),
        (gprMax.Radiometry, "receive", "waveform"),
    ):
        scene.add(
            output(
                frequencies=(0.8e9, 1e9),
                waveform_id="pulse",
                tags="target",
                id=name,
                normalisation=normalisation,
                spectrum_limit="nyquist",
            )
        )
    snapshot_specs = [
        ("coarse", np.asarray((1, 1, 1)), np.asarray((9, 11, 13)), np.asarray((2, 3, 4))),
        ("native", np.asarray((1, 1, 1)), np.asarray((9, 11, 13)), np.ones(3, dtype=int)),
    ]
    boundary = (5, 6, 7)[args.split]
    start, stop, step = np.ones(3, dtype=int), np.full(3, 3), np.full(3, 2)
    start[args.split], stop[args.split], step[args.split] = boundary - 1, boundary, 4
    snapshot_specs.append(("remote_only", start, stop, step))
    for name, start, stop, step in snapshot_specs:
        p1, p2, spacing = start * 0.002, stop * 0.002, step * 0.002
        if geometry is not None:
            p1[geometry.invariant_axis] = p2[geometry.invariant_axis] = np.inf
            spacing[geometry.invariant_axis] = 0.002
        for suffix in (".h5", ".vtkhdf"):
            scene.add(
                gprMax.Snapshot(
                    p1=tuple(p1),
                    p2=tuple(p2),
                    dl=tuple(spacing),
                    iterations=4,
                    filename=name + suffix,
                    fileext=suffix,
                )
            )
    split = [1, 1, 1]
    split[args.split] = 2
    options = {"mpi": tuple(split)} if args.mpi else {}
    # Retain the actual serial grid catalogue for identity-based comparison.
    # MPI SAR/radiometry intentionally use their own output-local IDs.
    original_build = Model.build
    material_names = {}

    def capture_materials(model):
        result = original_build(model)
        material_names.update({int(m.numID): m.ID for m in model.G.materials})
        return result

    with patch.object(Model, "build", capture_materials):
        gprMax.run(
            scenes=[scene],
            n=1,
            outputfile=args.directory / "result",
            hide_progress_bars=True,
            cpu_precision=args.precision,
            **options
        )
    if not args.mpi:
        (args.directory / "material_names.json").write_text(json.dumps(material_names))


def affine(args):
    from mpi4py import MPI
    from gprMax import config
    from gprMax.grid.fdtd_grid import FDTDGrid
    from gprMax.grid.mpi_grid import MPIGrid
    from gprMax.mode2d import mode2d_geometry
    from gprMax.snapshots import MPISnapshot, Snapshot, YEE_OFFSETS

    dtype = np.float32 if args.precision == "single" else np.float64
    config.sim_config = SimpleNamespace(dtypes={"float_or_double": dtype}, general={"solver": "cpu"})
    config.get_model_config = lambda: SimpleNamespace(mode=args.mode, ompthreads=1)
    split = [1, 1, 1]
    split[args.split] = MPI.COMM_WORLD.size
    if args.corners:
        assert MPI.COMM_WORLD.size == 8
        split = [2, 2, 2]
    comm = MPI.COMM_WORLD.Create_cart(split)
    grid = MPIGrid(comm)
    geometry = mode2d_geometry(args.mode)
    grid.global_size = np.asarray((14, 16, 18), dtype=np.int32)
    if geometry is not None:
        grid.global_size[geometry.invariant_axis] = 1 if geometry.polarisation == "TM" else 2
    grid.calculate_local_extents()
    grid.dl, grid.dt = np.asarray((0.001, 0.002, 0.003)), 1e-12
    indices = np.indices(tuple(grid.size + 1)) + grid.lower_extent[:, None, None, None]
    for component, offsets in YEE_OFFSETS.items():
        values = sum((indices[a] + offsets[a] / 2) * grid.dl[a] * (a + 1) for a in range(3))
        if geometry is not None:
            values = np.where(indices[geometry.invariant_axis] == geometry.live_index, values, 0)
        setattr(grid, component, np.ascontiguousarray(values, dtype=dtype))
    active = tuple(YEE_OFFSETS) if geometry is None else geometry.active_electric + geometry.active_magnetic
    outputs = {name: name in active for name in YEE_OFFSETS}
    reference_grid = FDTDGrid()
    reference_grid.size, reference_grid.dl, reference_grid.dt = (
        grid.global_size.copy(),
        grid.dl,
        grid.dt,
    )
    reference_indices = np.indices(tuple(reference_grid.size + 1))
    for component, offsets in YEE_OFFSETS.items():
        values = sum((reference_indices[a] + offsets[a] / 2) * grid.dl[a] * (a + 1) for a in range(3))
        if geometry is not None:
            values = np.where(reference_indices[geometry.invariant_axis] == geometry.live_index, values, 0)
        setattr(reference_grid, component, np.ascontiguousarray(values, dtype=dtype))
    reports = []
    for step in ((1, 1, 1), (2, 2, 2), (3, 3, 3), (2, 3, 4), (3, 4, 2), (10, 2, 3)):
        start, stop, step = np.asarray((1, 1, 1)), np.asarray((12, 14, 16)), np.asarray(step)
        # A one-output-cell ROI whose native stencil belongs to the other rank.
        if step[0] == 10:
            boundary = grid.global_size[args.split] // 2
            start[args.split], stop[args.split], step[args.split] = boundary - 1, boundary, 4
            for axis in range(3):
                if axis != args.split:
                    stop[axis], step[axis] = 4, 2
        if geometry is not None:
            axis = geometry.invariant_axis
            start[axis], stop[axis], step[axis] = geometry.live_index, geometry.live_index + 1, 1
        local_start, local_stop = grid.global_to_local_coordinate(start), grid.global_to_local_coordinate(stop)
        snap = MPISnapshot(*local_start, *local_stop, *step, 4, "unused", ".h5", outputs, grid)
        snap._SAMPLE_BATCH_SIZE = 193  # Exercise uneven and empty final rank batches.
        snap.initialise_snapfields()
        snap.store()
        reference = Snapshot(*start, *stop, *step, 4, "unused", ".h5", outputs, reference_grid)
        reference.initialise_snapfields()
        reference.store()
        output_slice = snap.grid_view.get_3d_output_slice()
        for component in active:
            assert (
                snap.snapfields[component].tobytes()
                == np.ascontiguousarray(reference.snapfields[component][output_slice]).tobytes()
            ), component
        local_indices = np.indices(tuple(snap.grid_view.size)) + snap.grid_view.offset[:, None, None, None]
        origin = snap._physical_origin()
        expected = sum((origin[a] + (local_indices[a] + 0.5) * step[a] * grid.dl[a]) * (a + 1) for a in range(3))
        for component in active:
            np.testing.assert_allclose(snap.snapfields[component], expected, rtol=3e-7, atol=1e-9)
        total = comm.allreduce(int(np.prod(snap.grid_view.size)))
        assert total == int(np.prod(snap.grid_view.global_size))
        reports.append(
            {
                "step": step.tolist(),
                "shape": snap.grid_view.global_size.tolist(),
                "local_count": comm.allgather(int(np.prod(snap.grid_view.size))),
            }
        )
    # Invalid global support must fail on all ranks before any storage collective.
    start, stop, step = np.zeros(3, dtype=int), grid.global_size.copy(), np.full(3, 5)
    stop[args.split] = grid.global_size[args.split]
    step[args.split] = int(grid.global_size[args.split]) - 1
    if geometry is not None:
        start[geometry.invariant_axis] = geometry.live_index
        stop[geometry.invariant_axis] = geometry.live_index + 1
        step[geometry.invariant_axis] = 1
    failed = False
    try:
        MPISnapshot(
            *grid.global_to_local_coordinate(start),
            *grid.global_to_local_coordinate(stop),
            *step,
            0,
            "invalid",
            ".h5",
            outputs,
            grid
        )
    except ValueError as exc:
        failed = "outside native Yee support" in str(exc)
    assert comm.allreduce(int(failed)) == comm.size

    # A local monitor error must become a collective error, not strand peers.
    def prepare():
        if comm.rank == 0:
            raise ValueError("deliberate incomplete rank payload")
        return object()

    grid.sar_monitors = [SimpleNamespace(mpi_signature=lambda: ("same",), local_payload=prepare)]
    failed = False
    try:
        grid.gather_sar_payloads()
    except RuntimeError as exc:
        failed = "deliberate incomplete rank payload" in str(exc)
    assert comm.allreduce(int(failed)) == comm.size
    grid.sar_monitors = [] if comm.rank == 0 else grid.sar_monitors
    failed = False
    try:
        grid.gather_sar_payloads()
    except RuntimeError as exc:
        failed = "inconsistent SAR monitors" in str(exc)
    assert comm.allreduce(int(failed)) == comm.size
    if comm.rank == 0:
        (args.directory / "affine.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("solve", "affine"), default="solve")
    parser.add_argument("--mode", default="3D")
    parser.add_argument("--precision", default="double")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--corners", action="store_true")
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    (solve if args.case == "solve" else affine)(args)
