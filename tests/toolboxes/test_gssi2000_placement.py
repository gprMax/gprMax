"""Rigid placement and quarter-turn rotation of the complete 2 GHz antenna."""

import hashlib
import itertools
import json
import warnings

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.model import Model
from gprMax.toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_2000
from gprMax.utilities.utilities import round_int


DL = 0.001
GEOMETRY = (gprMax.Box, gprMax.Plate, gprMax.Edge)


def cells(point):
    return tuple(round_int(value / DL) for value in point)


def geometry_signature(objects, anchor=(0, 0, 0)):
    return [
        [
            type(obj).__name__,
            *[tuple(v - a for v, a in zip(cells(obj.kwargs[key]), anchor)) for key in ("p1", "p2")],
            obj.kwargs["material_id"],
        ]
        for obj in objects
        if isinstance(obj, GEOMETRY)
    ]


@pytest.mark.parametrize("rotation", (0, 90, 180, 270))
def test_placement_preserves_every_primitive_at_cell_and_subcell_offsets(rotation):
    reference = geometry_signature(antenna_like_GSSI_2000(0.125, 0.125, 0.04, rotation=rotation), (125, 125, 40))
    # Includes the formerly collapsing Rx edge at y=.2505 and simultaneous
    # half-cell offsets in x/y/z. Do not use numpy's different tie-rounding rule.
    for base in ((0.063, 0.250, 0.019), (0.125, 0.125, 0.04), (0.301, 0.201, 0.100)):
        for phase in itertools.product((0, 0.25, 0.5, 0.75), repeat=3):
            point = tuple(a + b * DL for a, b in zip(base, phase))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                objects = antenna_like_GSSI_2000(*point, rotation=rotation)
            assert geometry_signature(objects, cells(point)) == reference
            for obj in objects:
                if isinstance(obj, GEOMETRY):
                    extent = np.subtract(cells(obj.kwargs["p2"]), cells(obj.kwargs["p1"]))
                    assert np.all(extent >= 0)
                    assert np.count_nonzero(extent) == {gprMax.Box: 3, gprMax.Plate: 2, gprMax.Edge: 1}[type(obj)]


def test_aligned_default_preserves_legacy_geometry():
    # Fingerprint of all 96 effective primitive commands, in build order, from
    # the pre-change model at this aligned placement (integer cell coordinates).
    signature = geometry_signature(antenna_like_GSSI_2000(0.125, 0.125, 0.04))
    assert (
        hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        == "c79542f845b90c262a9f90744ae847978c8c941fffb26c95cd4e8508d4ea7ede"
    )


@pytest.mark.parametrize(
    "rotation,axis,sign,offsets",
    [
        (0, "y", 1, ((-20, -1, 3), (20, -1, 3))),
        (90, "x", -1, ((0, -20, 3), (0, 20, 3))),
        (180, "y", -1, ((20, 0, 3), (-20, 0, 3))),
        (270, "x", 1, ((-1, 20, 3), (-1, -20, 3))),
    ],
)
def test_rotation_transforms_all_geometry_and_directed_feeds(rotation, axis, sign, offsets):
    anchor = np.array((125, 125, 40))
    objects = antenna_like_GSSI_2000(*(anchor * DL), rotation=rotation)
    reference = antenna_like_GSSI_2000(*(anchor * DL))
    # Independent matrix transform; min/max reorders corners after reversal.
    matrices = (
        np.eye(3, dtype=int),
        np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1))),
        np.diag((-1, -1, 1)),
        np.array(((0, 1, 0), (-1, 0, 0), (0, 0, 1))),
    )
    for before, after in zip(geometry_signature(reference, anchor), geometry_signature(objects, anchor)):
        endpoints = np.array(before[1:3]) @ matrices[rotation // 90].T
        assert after == [before[0], tuple(endpoints.min(axis=0)), tuple(endpoints.max(axis=0)), before[3]]

    sources = [obj for obj in objects if isinstance(obj, gprMax.VoltageSource)]
    waveforms = {obj.kwargs["id"]: obj for obj in objects if isinstance(obj, gprMax.Waveform)}
    for source, offset in zip(sources, offsets):
        assert cells(source.point) == tuple(anchor + offset)
        assert source.polarisation == axis
    assert waveforms[sources[0].waveform_id].kwargs["amp"] == -sign
    assert waveforms[sources[1].waveform_id].kwargs["amp"] == 0
    edge = next(obj for obj in objects if isinstance(obj, gprMax.Edge))
    assert edge.kwargs["p1"] == sources[1].point
    assert np.subtract(cells(edge.kwargs["p2"]), cells(edge.kwargs["p1"])).tolist() == [int(a == axis) for a in "xyz"]
    rx = next(obj for obj in objects if isinstance(obj, gprMax.Rx))
    assert rx.point == sources[1].point
    assert rx.outputs == ["E" + axis]
    assert [obj.kwargs for obj in objects if isinstance(obj, gprMax.Material)] == [
        obj.kwargs for obj in reference if isinstance(obj, gprMax.Material)
    ]


def test_off_grid_warning_and_exact_grid_silence():
    with pytest.warns(UserWarning, match="rounded to.*feed gaps are preserved"):
        antenna_like_GSSI_2000(0.063, 0.2505, 0.019)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        antenna_like_GSSI_2000(0.1 + 0.2, 0.250, 0.019)
    assert not caught


@pytest.mark.parametrize("y", (np.nextafter(0.2505, 0), 0.2505, np.nextafter(0.2505, 1)))
def test_half_cell_neighbours_keep_the_gap(y):
    with pytest.warns(UserWarning):
        objects = antenna_like_GSSI_2000(0.063, y, 0.019)
    edge = next(obj for obj in objects if isinstance(obj, gprMax.Edge))
    assert cells(edge.kwargs["p1"]) == (83, round_int(y / DL) - 1, 22)
    assert cells(edge.kwargs["p2"]) == (83, round_int(y / DL), 22)


@pytest.mark.parametrize("rotation", (-90, 45, 360, 90.01, True, "90", None, np.nan, np.inf))
def test_rejects_unsupported_rotations(rotation):
    with pytest.raises(ValueError, match="rotation must be"):
        antenna_like_GSSI_2000(0.125, 0.125, 0.04, rotation=rotation)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_rejects_nonfinite_placement(axis, value):
    point = [0.125, 0.125, 0.04]
    point[axis] = value
    with pytest.raises(ValueError, match="coordinates must be finite"):
        antenna_like_GSSI_2000(*point)


@pytest.mark.parametrize("rotation", (0, 90, 180, 270))
def test_full_geometry_build_preserves_feed_gaps(tmp_path, monkeypatch, rotation):
    build = Model.build
    checked = []

    def inspect(model):
        build(model)
        grid = model.G
        for index, source in enumerate(grid.voltagesources):
            axis = "xyz".index(source.polarisation)
            point = np.array(source.coord)
            material = grid.materials[grid.ID[(axis, *point)]]
            assert not material.is_pec
            assert material.er == pytest.approx((1.261, 1.056)[index])
            assert source.resistance == pytest.approx((560, 200008.0107)[index])
            for step in (-1, 1):
                neighbour = point.copy()
                neighbour[axis] += step
                assert grid.materials[grid.ID[(axis, *neighbour)]].is_pec
            checked.append(source.ID)

    monkeypatch.setattr(Model, "build", inspect)
    scene = gprMax.Scene()
    with pytest.warns(UserWarning):
        objects = antenna_like_GSSI_2000(0.0655, 0.0655, 0.0195, rotation=rotation)
    for obj in (
        gprMax.Domain(p1=(0.130, 0.130, 0.112)),
        gprMax.Discretisation(p1=(DL,) * 3),
        gprMax.TimeWindow(iterations=16),
        gprMax.OMPThreads(1),
        *objects,
    ):
        scene.add(obj)
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "antenna",
        cpu_precision="double",
        hide_progress_bars=True,
        log_level=40,
    )
    assert len(checked) == 2


@pytest.mark.parametrize("rotation", (0, 90, 180, 270))
@pytest.mark.parametrize("ratio,autotranslate", ((1, True), (3, True), (3, False)))
def test_subgrid_placement_matches_snapped_geometry(tmp_path, monkeypatch, rotation, ratio, autotranslate):
    build = Model.build
    fingerprints = []

    def inspect(model):
        build(model)
        grid = model.subgrids[0]
        digest = hashlib.sha256()
        for name in ("solid", "ID", "rigidE", "rigidH", "updatecoeffsE", "updatecoeffsH"):
            digest.update(getattr(grid, name).tobytes())
        for source in grid.voltagesources:
            digest.update(np.asarray(source.coord).tobytes())
            axis = "xyz".index(source.polarisation)
            material = grid.materials[grid.ID[(axis, *source.coord)]]
            assert not material.is_pec
        fingerprints.append(digest.hexdigest())

    monkeypatch.setattr(Model, "build", inspect)
    position = np.array((0.120, 0.120, 0.070) if autotranslate else (0.060, 0.060, 0.010))
    for offset in (0, 0.00025):
        scene = gprMax.Scene()
        for obj in (
            gprMax.Domain(p1=(0.240, 0.240, 0.210)),
            gprMax.Discretisation(p1=(ratio * DL,) * 3),
            gprMax.TimeWindow(iterations=16),
            gprMax.OMPThreads(1),
        ):
            scene.add(obj)
        subgrid = gprMax.SubGridHSG(p1=(0.06, 0.06, 0.06), p2=(0.18, 0.18, 0.15), ratio=ratio, id="antenna")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for obj in antenna_like_GSSI_2000(*(position + offset), rotation=rotation):
                subgrid.add(obj)
        scene.add(subgrid)
        gprMax.run(
            scenes=[scene],
            subgrid=True,
            autotranslate=autotranslate,
            geometry_only=True,
            outputfile=tmp_path / f"antenna_{offset}",
            cpu_precision="double",
            hide_progress_bars=True,
            log_level=40,
        )
    assert len(fingerprints) == 2
    assert fingerprints[0] == fingerprints[1]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("metal", (False, True))
def test_full_antenna_rotation_preserves_response(tmp_path, gpu_device, metal):
    """Full legacy materials, 8 ns including ringing; compare local polarities.

    The unchanged highly conducting gasket requires explicit underresolution
    permission. Research spectra are requested only for covariance checks,
    not as a claim that this grid resolves propagation inside that material.
    """
    reference = None
    for rotation, axis, sign in ((0, "y", 1), (90, "x", -1), (180, "y", -1), (270, "x", 1)):
        scene = gprMax.Scene()
        for obj in (
            gprMax.Domain(p1=(0.130, 0.130, 0.112)),
            gprMax.Discretisation(p1=(DL,) * 3),
            gprMax.TimeWindow(time=8e-9),
            gprMax.OMPThreads(1),
            *antenna_like_GSSI_2000(0.065, 0.065, 0.019, rotation=rotation, spectrum_limit="nyquist"),
        ):
            scene.add(obj)
        if metal:
            scene.add(gprMax.Plate(p1=(0, 0, 0.019), p2=(0.130, 0.130, 0.019), material_id="pec"))
        output = tmp_path / f"rotation_{rotation}"
        gprMax.run(
            scenes=[scene],
            outputfile=output,
            gpu=[gpu_device],
            gpu_precision="double",
            allow_underresolved=True,
            hide_progress_bars=True,
            log_level=40,
        )
        with h5py.File(output.with_suffix(".h5")) as data:
            histories = {"field": sign * data[f"rxs/rx1/E{axis}"][:]}
            for port_id in ("gssi2000_tx", "gssi2000_rx"):
                port = data[f"ports/{port_id}"]
                for key in ("Vgenerator", "Vtotal"):
                    histories[f"{port_id}/{key}"] = sign * port[key][:]
            tx = data["ports/gssi2000_tx"]
            # Research output retains mesh-validity flags: the original gasket
            # leaves positive frequencies unresolved. Check numerical rotation
            # covariance without pretending those flags certify convergence.
            band = tx["source_valid"][:].astype(bool) & tx["gap_correction_valid"][:].astype(bool)
            frequency = tx["frequency"][:]
            band &= (frequency >= 0.5e9) & (frequency <= 4e9)
            assert np.any(band)
            histories["band"] = band
            histories["valid_S11"] = tx["valid_S11"][:]
            histories["valid_Zin"] = tx["valid_Zin"][:]
            for key in ("S11", "Zin"):
                histories[key] = tx[key][:]
        assert np.all(np.isfinite(histories["field"]))
        assert np.max(np.abs(histories["field"])) > 1
        if reference is None:
            reference = histories
        else:
            for key in histories:
                if key in ("band", "valid_S11", "valid_Zin"):
                    np.testing.assert_array_equal(histories[key], reference[key])
                    continue
                actual, expected = histories[key], reference[key]
                if key in ("S11", "Zin"):
                    actual, expected = actual[band], expected[band]
                assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
                scale = np.max(np.abs(expected))
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10 * scale)
