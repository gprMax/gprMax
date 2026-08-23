"""Axis-rotation coverage for surface-impedance modal injection."""

import logging

import h5py
import numpy as np
import pytest

import gprMax
from testing.validation.validate_impedance_modal_injection import (
    ANCHORS,
    DFT_POINTS,
    DL,
    DOMAIN,
    FMAX,
    FMIN,
    GUIDE_LOWER,
    GUIDE_UPPER,
    MAX_ALPHA_RELATIVE_L2_ERROR,
    MAX_SOURCE_REFLECTION_DB,
    MODEL_ID,
    PML_CELLS,
    SOURCE_X,
    NEAR_MONITOR_X,
    FAR_MONITOR_X,
    SURFACE_RESISTANCE,
    TIME_WINDOW,
    _wall_boxes,
    analyse_modal_coefficients,
)


def _swap_x_y(point):
    """Rotate the x-normal reference guide onto the global y axis."""

    return point[1], point[0], point[2]


def _build_y_normal_scene(threads=1):
    """Return the reference guide with left-handed local basis (x, z, y)."""

    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=_swap_x_y(DOMAIN)))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.PMLThickness(thickness=PML_CELLS))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(
        gprMax.SurfaceImpedance(
            id=MODEL_ID,
            resistance=SURFACE_RESISTANCE,
            fit_fmin_hz=FMIN,
            fit_fmax_hz=FMAX,
        )
    )
    for lower, upper in _wall_boxes():
        scene.add(gprMax.ImpedanceBox(_swap_x_y(lower), _swap_x_y(upper), MODEL_ID))

    scene.add(gprMax.EigenmodeBand(id="impedance_te10", fmin=FMIN, fmax=FMAX, points=DFT_POINTS))
    for port, y, direction in (
        (1, SOURCE_X, "+"),
        (2, NEAR_MONITOR_X, "-"),
        (3, FAR_MONITOR_X, "-"),
    ):
        scene.add(
            gprMax.EigenmodePort(
                port=port,
                p1=(GUIDE_LOWER[0], y, GUIDE_LOWER[1]),
                p2=(GUIDE_UPPER[0], y, GUIDE_UPPER[1]),
                direction=direction,
                modes=(1,),
                anchors=ANCHORS,
                plot_fields=False,
            )
        )
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="auto", plot_waveform=False))
    return scene


def _read_port(path, port):
    with h5py.File(path, "r") as data:
        group = data[f"eigenmode_ports/port{port}"]
        frequency = np.asarray(group["frequency"], dtype=np.float64)
        incident = np.asarray(group["incident"])[0]
        outgoing = np.asarray(group["outgoing"])[0]
        valid_name = "power_wave_valid" if "power_wave_valid" in group else "valid"
        valid = np.asarray(group[valid_name], dtype=bool)[0]
    return frequency, incident, outgoing, valid


@pytest.mark.integration
def test_y_normal_impedance_modal_injection_has_low_reflection_and_correct_attenuation(tmp_path):
    """The y-normal port exercises the left-handed local modal basis."""

    stem = tmp_path / "y_normal_impedance_modal_injection"
    try:
        gprMax.run(
            scenes=[_build_y_normal_scene()],
            outputfile=stem,
            cpu_precision="double",
            hide_progress_bars=True,
            log_level=logging.WARNING,
        )
        ports = [_read_port(stem.with_suffix(".h5"), port) for port in (1, 2, 3)]
        frequency = ports[0][0]
        assert all(np.array_equal(item[0], frequency) for item in ports[1:])
        valid = np.logical_and.reduce([item[3] for item in ports])
        valid &= (frequency >= FMIN) & (frequency <= FMAX)
        assert np.all(valid)

        result = analyse_modal_coefficients(
            frequency,
            ports[0][1],
            ports[0][2],
            ports[1][2],
            ports[2][2],
        )
        assert result["maximum_source_reflection_db"] < MAX_SOURCE_REFLECTION_DB
        assert result["alpha_relative_l2_error"] < MAX_ALPHA_RELATIVE_L2_ERROR
    finally:
        logger = logging.getLogger("gprMax")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
