"""Rigid placement and quarter-turn rotation of the MALA 1.2 GHz, GSSI 1.5 GHz and
GSSI 400 MHz antenna models (the GSSI 2 GHz model has its own tests)."""

import hashlib
import itertools
import json
import warnings

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.model import Model
from gprMax.toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_400, antenna_like_GSSI_1500
from gprMax.toolboxes.GPRAntennaModels.MALA import antenna_like_MALA_1200
from gprMax.utilities.utilities import round_int

GEOMETRY = (gprMax.Box, gprMax.Plate, gprMax.Edge, gprMax.Cylinder, gprMax.Triangle)

# (model, resolution, documented case size in metres, total height in metres)
MODELS = [
    pytest.param(antenna_like_MALA_1200, 0.001, (0.184, 0.109, 0.040), 0.046, id="MALA_1200-1mm"),
    pytest.param(antenna_like_MALA_1200, 0.002, (0.184, 0.109, 0.040), 0.046, id="MALA_1200-2mm"),
    pytest.param(antenna_like_GSSI_1500, 0.001, (0.170, 0.108, 0.043), 0.047, id="GSSI_1500-1mm"),
    pytest.param(antenna_like_GSSI_1500, 0.002, (0.170, 0.108, 0.043), 0.047, id="GSSI_1500-2mm"),
    pytest.param(antenna_like_GSSI_400, 0.002, (0.300, 0.300, 0.178), 0.178, id="GSSI_400-2mm"),
]
# Large enough for every model with its default 10-cell PML at 2 mm.
DOMAIN = (0.400, 0.400, 0.260)
CENTRE = (0.200, 0.200, 0.030)


def cells(point, dl):
    return tuple(round_int(value / dl) for value in point)


def layout(objects, anchor, dl):
    """Every object's points in cells relative to ``anchor``, in build order."""
    rows = []
    for obj in objects:
        points = [obj.kwargs[key] for key in ("p1", "p2", "p3") if key in obj.kwargs]
        label = (
            obj.kwargs.get("material_id") or obj.kwargs.get("polarisation") or obj.kwargs.get("id")
        )
        rows.append(
            [
                type(obj).__name__,
                label,
                [tuple(c - a for c, a in zip(cells(p, dl), anchor)) for p in points],
            ]
        )
    return rows


def build(model, point, dl, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return model(*point, resolution=dl, **kwargs)


@pytest.mark.parametrize("rotation", (0, 90, 180, 270))
@pytest.mark.parametrize("model,dl,case,height", MODELS)
def test_placement_preserves_every_object_at_cell_and_subcell_offsets(
    model, dl, case, height, rotation
):
    base = (0.200, 0.115, 0.050)
    reference = layout(build(model, base, dl, rotation=rotation), cells(base, dl), dl)
    # y = 0.115 / 0.116 gave different MALA case widths before the fix; the
    # bases cover odd and even cell counts at 2 mm and a half-millimetre point.
    for start in ((0.200, 0.115, 0.050), (0.200, 0.116, 0.050), (0.301, 0.2505, 0.119)):
        for phase in itertools.product((0, 0.25, 0.5, 0.75), repeat=3):
            point = tuple(a + b * dl for a, b in zip(start, phase))
            objects = build(model, point, dl, rotation=rotation)
            assert layout(objects, cells(point, dl), dl) == reference


@pytest.mark.parametrize("model,dl,case,height", MODELS)
def test_same_layout_at_every_on_grid_placement(model, dl, case, height):
    rng = np.random.default_rng(0)
    layouts = set()
    for _ in range(50):
        anchor = tuple(int(v) for v in rng.integers((100, 100, 20), (300, 300, 150)))
        objects = build(model, np.array(anchor) * dl, dl)
        layouts.add(json.dumps(layout(objects, anchor, dl)))
    assert len(layouts) == 1


@pytest.mark.parametrize(
    "model,dl,digest",
    [
        # Fingerprints of the models before rigid placement was added, at an
        # on-grid placement where they were already rigid: unchanged geometry.
        (
            antenna_like_GSSI_1500,
            0.001,
            "08d9a33132f902d9754ab0a7726cfe94195276f9990147d7ea3b01d7343115f9",
        ),
        (
            antenna_like_GSSI_400,
            0.002,
            "315368c0b6c3a894ffedfebb698357a24df88ce3082e29042560a71228dd35e6",
        ),
    ],
    ids=("GSSI_1500-1mm", "GSSI_400-2mm"),
)
def test_aligned_default_preserves_legacy_geometry(model, dl, digest):
    anchor = (200, 150, 40)
    signature = layout(build(model, np.array(anchor) * dl, dl), anchor, dl)
    assert hashlib.sha256(json.dumps(signature).encode()).hexdigest() == digest


@pytest.mark.parametrize("model,dl,case,height", MODELS)
def test_case_has_the_documented_size(model, dl, case, height):
    anchor = (150, 115, 50)
    objects = build(model, np.array(anchor) * dl, dl)
    boxes = [obj for obj in objects if isinstance(obj, gprMax.Box)]
    sizes = [
        np.subtract(cells(obj.kwargs["p2"], dl), cells(obj.kwargs["p1"], dl)) * dl for obj in boxes
    ]
    largest = max(sizes, key=lambda size: size[0] * size[1])
    np.testing.assert_allclose(largest[:2], case[:2], atol=dl / 2)
    top = max(cells(obj.kwargs["p2"], dl)[2] for obj in boxes)
    # Heights that are not whole cells (47 mm at 2 mm) come out one cell lower.
    assert (top - anchor[2]) * dl == pytest.approx(height, abs=dl)


@pytest.mark.parametrize("rotation", (0, 90, 180, 270))
@pytest.mark.parametrize("model,dl,case,height", MODELS)
def test_rotation_transforms_all_geometry_and_directed_feeds(model, dl, case, height, rotation):
    anchor = np.array((200, 200, 50))
    objects = build(model, anchor * dl, dl, rotation=rotation)
    reference = build(model, anchor * dl, dl)
    axis, sign = {0: ("y", 1), 90: ("x", -1), 180: ("y", -1), 270: ("x", 1)}[rotation]
    # Independent matrix transform; min/max reorders corners after reversal.
    matrix = (
        np.eye(3, dtype=int),
        np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1))),
        np.diag((-1, -1, 1)),
        np.array(((0, 1, 0), (-1, 0, 0), (0, 0, 1))),
    )[rotation // 90]
    moved = [obj for obj in objects if isinstance(obj, GEOMETRY)]
    original = [obj for obj in reference if isinstance(obj, GEOMETRY)]
    assert len(moved) == len(original)
    for obj, before, after in zip(moved, layout(original, anchor, dl), layout(moved, anchor, dl)):
        points = np.array(before[2]) @ matrix.T
        if isinstance(obj, (gprMax.Box, gprMax.Plate, gprMax.Edge)):
            points = np.array((points.min(axis=0), points.max(axis=0)))
        assert after[:2] == before[:2]
        assert after[2] == [tuple(int(v) for v in p) for p in points]

    feeds = [obj for obj in objects if isinstance(obj, (gprMax.VoltageSource, gprMax.Rx))]
    feeds_before = [obj for obj in reference if isinstance(obj, (gprMax.VoltageSource, gprMax.Rx))]
    for after, before in zip(feeds, feeds_before):
        if isinstance(after, gprMax.VoltageSource):
            assert after.polarisation == axis
        else:
            assert after.outputs == ["E" + axis]
        # Each feed is the positive-axis Yee edge starting at its stored point.
        start = np.subtract(cells(before.point, dl), anchor)
        ends = np.array((start, start + (0, 1, 0))) @ matrix.T
        assert tuple(np.subtract(cells(after.point, dl), anchor)) == tuple(ends.min(axis=0))
        assert tuple(ends.max(axis=0) - ends.min(axis=0)) == tuple(int(a == axis) for a in "xyz")
    assert [obj.kwargs for obj in objects if isinstance(obj, gprMax.Material)] == [
        obj.kwargs for obj in reference if isinstance(obj, gprMax.Material)
    ]


@pytest.mark.parametrize("rotation", (0, 90, 180, 270))
@pytest.mark.parametrize("model,dl,case,height", MODELS[:4])
def test_rotation_drives_the_feed_with_its_local_polarity(model, dl, case, height, rotation):
    sign = {0: 1, 90: -1, 180: -1, 270: 1}[rotation]
    objects = build(model, CENTRE, dl, rotation=rotation)
    source = next(obj for obj in objects if isinstance(obj, gprMax.VoltageSource))
    waveform = next(
        obj
        for obj in objects
        if isinstance(obj, gprMax.Waveform) and obj.kwargs["id"] == source.waveform_id
    )
    assert waveform.kwargs["amp"] == sign


@pytest.mark.parametrize("rotation", (0, 90, 180, 270))
def test_rotated_gssi_400_excitation_file_keeps_its_local_polarity(rotation):
    sign = {0: 1, 90: -1, 180: -1, 270: 1}[rotation]
    objects = build(antenna_like_GSSI_400, CENTRE, 0.002, rotation=rotation)
    reference = build(antenna_like_GSSI_400, CENTRE, 0.002)
    source = next(obj for obj in objects if isinstance(obj, gprMax.VoltageSource))
    excitation = next(obj for obj in reference if isinstance(obj, gprMax.ExcitationFile))
    if sign > 0:
        assert [obj.kwargs for obj in objects if isinstance(obj, gprMax.ExcitationFile)] == [
            excitation.kwargs
        ]
        return
    assert not any(isinstance(obj, gprMax.ExcitationFile) for obj in objects)
    waveform = next(
        obj
        for obj in objects
        if isinstance(obj, gprMax.Waveform) and obj.kwargs["id"] == source.waveform_id
    )
    data = np.loadtxt(excitation.filepath, skiprows=1)
    np.testing.assert_array_equal(waveform.kwargs["user_time"], data[:, 0])
    np.testing.assert_array_equal(waveform.kwargs["user_values"], -data[:, 1])
    assert waveform.kwargs["kind"] == excitation.kind
    assert waveform.kwargs["fill_value"] == excitation.fill_value


@pytest.mark.parametrize("model,dl,case,height", MODELS)
def test_off_grid_warning_and_exact_grid_silence(model, dl, case, height):
    with pytest.warns(UserWarning, match="rounded to.*dimensions are preserved"):
        model(0.200, 0.200 + dl / 4, 0.050, resolution=dl)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(0.1 + 0.2 - 0.1, 0.116 if dl == 0.002 else 0.115, 0.050, resolution=dl)
    assert not caught


@pytest.mark.parametrize(
    "model", (antenna_like_MALA_1200, antenna_like_GSSI_1500, antenna_like_GSSI_400)
)
@pytest.mark.parametrize("rotation", (-90, 45, 360, 90.01, True, "90", None, np.nan, np.inf))
def test_rejects_unsupported_rotations(model, rotation):
    with pytest.raises(ValueError, match="rotation must be"):
        model(0.200, 0.200, 0.050, rotation=rotation)


@pytest.mark.parametrize(
    "model", (antenna_like_MALA_1200, antenna_like_GSSI_1500, antenna_like_GSSI_400)
)
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_rejects_nonfinite_placement(model, axis, value):
    point = [0.200, 0.200, 0.050]
    point[axis] = value
    with pytest.raises(ValueError, match="coordinates must be finite"):
        model(*point)


@pytest.mark.parametrize(
    "model,resolution,match",
    [
        (antenna_like_MALA_1200, 0.003, "1mm or 2mm"),
        (antenna_like_GSSI_1500, 0.003, "1mm or 2mm"),
        (antenna_like_GSSI_400, 0.001, "2 mm resolution"),
    ],
)
def test_rejects_unsupported_resolution(model, resolution, match):
    with pytest.raises(ValueError, match=match):
        model(0.200, 0.200, 0.050, resolution=resolution)


@pytest.mark.parametrize("rotation", (0, 90, 180, 270))
@pytest.mark.parametrize("model,dl,case,height", MODELS)
def test_full_geometry_build_keeps_the_feed_gaps(
    tmp_path, monkeypatch, model, dl, case, height, rotation
):
    """Every feed edge is not metal and both neighbouring edges along it are."""
    run_build = Model.build
    checked = []

    def inspect(model_):
        run_build(model_)
        grid = model_.G
        for source in grid.voltagesources:
            axis = "xyz".index(source.polarisation)
            point = np.array(source.coord)
            assert not grid.materials[grid.ID[(axis, *point)]].is_pec
            for step in (-1, 1):
                neighbour = point.copy()
                neighbour[axis] += step
                assert grid.materials[grid.ID[(axis, *neighbour)]].is_pec
            checked.append(source.polarisation)

    monkeypatch.setattr(Model, "build", inspect)
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=DOMAIN),
        gprMax.Discretisation(p1=(dl,) * 3),
        gprMax.TimeWindow(iterations=4),
        gprMax.OMPThreads(1),
        *build(model, np.add(CENTRE, dl / 2), dl, rotation=rotation),
    ):
        scene.add(obj)
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "antenna",
        hide_progress_bars=True,
        log_level=40,
    )
    assert checked == ["y" if rotation in (0, 180) else "x"]


def _trace(
    tmp_path, name, model, point, rotation=0, iterations=500, dl=0.002, domain=DOMAIN, gpu=None
):
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=domain),
        gprMax.Discretisation(p1=(dl,) * 3),
        gprMax.TimeWindow(iterations=iterations),
        gprMax.OMPThreads(1),
        *build(model, point, dl, rotation=rotation),
    ):
        scene.add(obj)
    output = tmp_path / name
    if gpu is None:
        precision = {"cpu_precision": "double"}
    else:
        precision = {"gpu": [gpu], "gpu_precision": "double"}
    gprMax.run(
        scenes=[scene],
        outputfile=output,
        hide_progress_bars=True,
        log_level=40,
        **precision,
    )
    with h5py.File(output.with_suffix(".h5")) as data:
        group = data["rxs/rx1"]
        (component,) = [key for key in group if key.startswith("E")]
        return group[component][:]


@pytest.mark.slow
@pytest.mark.parametrize(
    "model", (antenna_like_MALA_1200, antenna_like_GSSI_1500, antenna_like_GSSI_400)
)
def test_response_is_unchanged_by_moving_and_rotating(tmp_path, model):
    """2 mm free-space traces: moved within a cell, and all quarter turns.

    The antenna rotates about the centre of a square domain, so every rotation
    sees the same boundaries and traces agree to round-off. Rotations are
    compared in the antenna-local polarity.
    """
    reference = _trace(tmp_path, "reference", model, CENTRE)
    scale = np.max(np.abs(reference))
    assert scale > 0
    moved = _trace(tmp_path, "subcell", model, np.add(CENTRE, (0.0005, 0.0003, 0.0001)))
    np.testing.assert_array_equal(moved, reference)
    for rotation, sign in ((90, -1), (180, -1), (270, 1)):
        rotated = _trace(tmp_path, f"rotation_{rotation}", model, CENTRE, rotation)
        np.testing.assert_allclose(sign * rotated, reference, rtol=0, atol=1e-6 * scale)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "model,dl,domain,centre,iterations",
    [
        (antenna_like_MALA_1200, 0.001, (0.240, 0.240, 0.080), (0.120, 0.120, 0.015), 1000),
        (antenna_like_GSSI_1500, 0.001, (0.240, 0.240, 0.080), (0.120, 0.120, 0.015), 1000),
        (antenna_like_GSSI_400, 0.002, DOMAIN, CENTRE, 500),
    ],
    ids=("MALA_1200-1mm", "GSSI_1500-1mm", "GSSI_400-2mm"),
)
def test_gpu_response_is_unchanged_by_moving_and_rotating(
    tmp_path, gpu_device, model, dl, domain, centre, iterations
):
    """CUDA, double precision, each model at its native resolution."""
    run = dict(dl=dl, domain=domain, iterations=iterations, gpu=gpu_device)
    reference = _trace(tmp_path, "reference", model, centre, **run)
    scale = np.max(np.abs(reference))
    assert np.all(np.isfinite(reference)) and scale > 0
    moved = _trace(
        tmp_path, "subcell", model, np.add(centre, np.multiply((0.4, 0.3, 0.2), 1e-3)), **run
    )
    np.testing.assert_array_equal(moved, reference)
    for rotation, sign in ((90, -1), (180, -1), (270, 1)):
        rotated = _trace(tmp_path, f"rotation_{rotation}", model, centre, rotation, **run)
        np.testing.assert_allclose(sign * rotated, reference, rtol=0, atol=1e-10 * scale)


# (model, fine resolution, ratio, subgrid p1, p2, domain, antenna placement (global))
SUBGRID_CASES = [
    pytest.param(
        antenna_like_MALA_1200,
        0.001,
        3,
        (0.12, 0.12, 0.12),
        (0.36, 0.36, 0.24),
        (0.48, 0.48, 0.36),
        (0.24, 0.24, 0.15),
        id="MALA_1200-1mm",
    ),
    pytest.param(
        antenna_like_MALA_1200,
        0.002,
        3,
        (0.12, 0.12, 0.12),
        (0.36, 0.36, 0.24),
        (0.48, 0.48, 0.36),
        (0.24, 0.24, 0.15),
        id="MALA_1200-2mm",
    ),
    pytest.param(
        antenna_like_GSSI_1500,
        0.001,
        3,
        (0.12, 0.12, 0.12),
        (0.36, 0.36, 0.24),
        (0.48, 0.48, 0.36),
        (0.24, 0.24, 0.15),
        id="GSSI_1500-1mm",
    ),
    pytest.param(
        antenna_like_GSSI_1500,
        0.002,
        3,
        (0.12, 0.12, 0.12),
        (0.36, 0.36, 0.24),
        (0.48, 0.48, 0.36),
        (0.24, 0.24, 0.15),
        id="GSSI_1500-2mm",
    ),
    pytest.param(
        antenna_like_GSSI_400,
        0.002,
        3,
        (0.12, 0.12, 0.12),
        (0.48, 0.48, 0.36),
        (0.60, 0.60, 0.48),
        (0.30, 0.30, 0.15),
        id="GSSI_400-2mm",
    ),
    # Ratio 1: the subgrid has the outer grid's cell size, so the domain only
    # needs room for the 10-cell PML around the subgrid.
    pytest.param(
        antenna_like_MALA_1200,
        0.001,
        1,
        (0.03, 0.03, 0.03),
        (0.27, 0.27, 0.15),
        (0.30, 0.30, 0.18),
        (0.15, 0.15, 0.06),
        id="MALA_1200-1mm-ratio1",
    ),
    pytest.param(
        antenna_like_GSSI_1500,
        0.001,
        1,
        (0.03, 0.03, 0.03),
        (0.27, 0.27, 0.15),
        (0.30, 0.30, 0.18),
        (0.15, 0.15, 0.06),
        id="GSSI_1500-1mm-ratio1",
    ),
    pytest.param(
        antenna_like_GSSI_400,
        0.002,
        1,
        (0.04, 0.04, 0.04),
        (0.40, 0.40, 0.28),
        (0.44, 0.44, 0.32),
        (0.22, 0.22, 0.07),
        id="GSSI_400-2mm-ratio1",
    ),
]


@pytest.mark.parametrize("rotation", (0, 90))
@pytest.mark.parametrize("autotranslate", (True, False))
@pytest.mark.parametrize("model,dl,ratio,sub_p1,sub_p2,domain,placement", SUBGRID_CASES)
def test_subgrid_placement_matches_snapped_geometry(
    tmp_path,
    monkeypatch,
    model,
    dl,
    ratio,
    sub_p1,
    sub_p2,
    domain,
    placement,
    autotranslate,
    rotation,
):
    """Subgrid at the antenna's resolution: a quarter-cell offset builds the same grid."""
    run_build = Model.build
    fingerprints = []

    def inspect(model_):
        run_build(model_)
        grid = model_.subgrids[0]
        digest = hashlib.sha256()
        for name in ("solid", "ID", "rigidE", "rigidH", "updatecoeffsE", "updatecoeffsH"):
            digest.update(getattr(grid, name).tobytes())
        assert grid.voltagesources
        for source in grid.voltagesources:
            digest.update(np.asarray(source.coord).tobytes())
            axis = "xyz".index(source.polarisation)
            assert not grid.materials[grid.ID[(axis, *source.coord)]].is_pec
        fingerprints.append(digest.hexdigest())

    monkeypatch.setattr(Model, "build", inspect)
    # With autotranslate, positions are global; without it they are relative to
    # the subgrid's lower corner.
    position = np.array(placement) if autotranslate else np.subtract(placement, sub_p1)
    for offset in (0, dl / 4):
        scene = gprMax.Scene()
        for obj in (
            gprMax.Domain(p1=domain),
            gprMax.Discretisation(p1=(ratio * dl,) * 3),
            gprMax.TimeWindow(iterations=4),
            gprMax.OMPThreads(1),
        ):
            scene.add(obj)
        subgrid = gprMax.SubGridHSG(p1=sub_p1, p2=sub_p2, ratio=ratio, id="antenna")
        for obj in build(model, position + offset, dl, rotation=rotation):
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
