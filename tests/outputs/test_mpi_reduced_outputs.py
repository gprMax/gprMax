"""Actual two-rank output tests for all reduced modes and live-axis splits."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest


HELPER = Path(__file__).with_name("mpi_reduced_output_probe.py")
LAUNCHER = Path(sys.executable).parent / "mpiexec"
LAUNCHER = str(LAUNCHER) if LAUNCHER.is_file() else shutil.which("mpiexec")
MODES = [
    (f"2D {family}{'xyz'[axis]}", split)
    for family in ("TM", "TE")
    for axis in range(3)
    for split in range(3)
    if split != axis
]
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(LAUNCHER is None, reason="MPI launcher unavailable"),
]


def run_probe(directory, mode, split, precision, *, mpi, case="solve", corners=False):
    command = [
        sys.executable,
        str(HELPER),
        "--mode",
        mode,
        "--split",
        str(split),
        "--precision",
        precision,
        "--case",
        case,
        "--directory",
        str(directory),
    ]
    if mpi:
        command = [LAUNCHER, "-n", "8" if corners else "2", *command, "--mpi"]
    if corners:
        command.append("--corners")
    environment = os.environ.copy()
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.update(FI_PROVIDER="shm", OMP_NUM_THREADS="1", MPLCONFIGDIR=str(directory / "mpl"))
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, env=environment)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


def compare_material_ids(serial, distributed, material_names):
    """Compare cell identities without assuming serial and MPI ID numbering."""
    assert "materials" not in serial
    catalogue = distributed["materials"]
    assert set(catalogue) == {"id", "name"}
    assert distributed["material_id"].attrs["Catalogue"] == "materials"
    ids, names = catalogue["id"][...], catalogue["name"].asstr()[...]
    assert ids.dtype.kind in "iu"
    assert ids.ndim == names.ndim == 1 and ids.shape == names.shape
    assert len(set(ids)) == len(ids), "Duplicate catalogue ID"
    assert len(set(names)) == len(names), "Duplicate catalogue name"
    mapping = dict(zip(ids, names))
    left_ids, right_ids = serial["material_id"][...], distributed["material_id"][...]
    assert left_ids.shape == right_ids.shape
    assert set(np.unique(right_ids)) == set(mapping), "Unmapped or unused catalogue ID"
    assert set(np.unique(left_ids)) <= set(material_names), "Unknown serial material ID"
    left = np.array([material_names[value] for value in left_ids.flat]).reshape(left_ids.shape)
    right = np.array([mapping[value] for value in right_ids.flat]).reshape(right_ids.shape)
    np.testing.assert_array_equal(right, left, err_msg=serial["material_id"].name)


def compare_group(serial, distributed, precision, *, exact_float=False, material_names=None):
    extra = {"materials"} if material_names is not None else set()
    assert set(serial.keys()) | extra == set(distributed.keys())
    if material_names is not None:
        compare_material_ids(serial, distributed, material_names)
    for name in serial:
        if name == "material_id" and material_names is not None:
            continue  # Compared by identity, including catalogue validation, above.
        left, right = serial[name], distributed[name]
        if isinstance(left, h5py.Group):
            compare_group(left, right, precision, exact_float=exact_float)
        elif left.dtype.kind in "fc":
            if exact_float:
                assert right[...].tobytes() == left[...].tobytes(), left.name
            else:
                np.testing.assert_allclose(
                    right[...],
                    left[...],
                    rtol=2e-5 if precision == "single" else 2e-12,
                    atol=1e-25,
                    equal_nan=True,
                    err_msg=left.name,
                )
        else:
            np.testing.assert_array_equal(right[...], left[...], err_msg=left.name)
    for name in serial.attrs:
        if name in {"DFTCollectionBackend"}:
            continue
        assert name in distributed.attrs, name
        left, right = np.asarray(serial.attrs[name]), np.asarray(distributed.attrs[name])
        if left.dtype.kind in "fc":
            np.testing.assert_allclose(right, left, rtol=1e-6, atol=1e-25, equal_nan=True, err_msg=name)
        else:
            np.testing.assert_array_equal(right, left, err_msg=name)


@pytest.mark.parametrize("mode,split", MODES)
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.skipif(not h5py.get_config().mpi, reason="Parallel snapshot output requires MPI-enabled h5py")
def test_reduced_sar_radiometry_and_snapshots_match_serial(tmp_path, mode, split, precision):
    serial, distributed = tmp_path / "serial", tmp_path / "mpi"
    run_probe(serial, mode, split, precision, mpi=False)
    run_probe(distributed, mode, split, precision, mpi=True)
    material_names = {
        int(key): value for key, value in json.loads((serial / "material_names.json").read_text()).items()
    }
    with h5py.File(serial / "result.h5") as left, h5py.File(distributed / "result.h5") as right:
        for path in ("sar/dose", "radiometry/receive"):
            compare_group(left[path], right[path], precision, material_names=material_names)
        assert np.nanmax(right["sar/dose/sar"][...]) > 0
        assert right["sar/dose/tags/target"].attrs["MassPerLengthUnits"] == "kg/m"
    left_files = (
        sorted(serial.glob("**/coarse.h5"))
        + sorted(serial.glob("**/native.h5"))
        + sorted(serial.glob("**/remote_only.h5"))
    )
    assert len(left_files) == 3
    for path in left_files:
        for suffix in (".h5", ".vtkhdf"):
            path = path.with_suffix(suffix)
            peer = distributed / path.relative_to(serial)
            with h5py.File(path) as left, h5py.File(peer) as right:
                compare_group(left, right, precision, exact_float=True)


@pytest.mark.parametrize("mode,split", [("3D", axis) for axis in range(3)] + MODES)
@pytest.mark.parametrize("precision", ["single", "double"])
def test_mpi_native_affine_interpolation_and_zero_owned_rank(tmp_path, mode, split, precision):
    run_probe(tmp_path, mode, split, precision, mpi=True, case="affine")
    assert (tmp_path / "affine.json").is_file()


@pytest.mark.parametrize("precision", ["single", "double"])
def test_eight_rank_snapshot_corner_native_and_coarse_stencils(tmp_path, precision):
    run_probe(tmp_path, "3D", 0, precision, mpi=True, case="affine", corners=True)
