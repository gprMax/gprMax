"""Scientific regression checks for the editable dipole example."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gprMax.toolboxes.Optimisation import BuildContext, PortSpectrum, Scenario
from gprMax.toolboxes.Optimisation.examples import thin_wire_dipole as example
from gprMax.toolboxes.Optimisation.examples.advanced import thin_wire_dipole as benchmark


@pytest.mark.parametrize("arm_cells", [54, 69, 83])
def test_editable_dipole_preserves_the_benchmark_physics(arm_cells):
    import gprMax

    scene = example.build_model({"arm_cells": arm_cells})
    config = benchmark.settings(dz_m=0.001, cycles=100)
    previous = benchmark.build_model(
        {"arm_cells": arm_cells},
        Scenario("free_space", config),
        BuildContext(Path("."), Path("output"), 0, 1),
    ).scene
    assert isinstance(scene, gprMax.Scene)
    for kind in (gprMax.Domain, gprMax.Discretisation, gprMax.TimeWindow):
        current = next(obj for obj in scene.single_use_objects if isinstance(obj, kind))
        original = next(obj for obj in previous.single_use_objects if isinstance(obj, kind))
        assert current.kwargs == original.kwargs
    pml = next(obj for obj in scene.single_use_objects if isinstance(obj, gprMax.PMLThickness))
    assert pml.thickness == (8, 8, 24, 8, 8, 24)
    for current, original in zip(scene.geometry_objects, previous.geometry_objects, strict=True):
        assert isinstance(current, gprMax.ThinWire)
        for key in ("p1", "p2", "radius"):
            np.testing.assert_allclose(
                current.kwargs[key], original.kwargs[key], rtol=0, atol=1e-15
            )
    source = next(obj for obj in scene.grid_objects if isinstance(obj, gprMax.VoltageSource))
    old_source = next(obj for obj in previous.grid_objects if isinstance(obj, gprMax.VoltageSource))
    assert source.kwargs == old_source.kwargs


def dip_spectrum():
    frequency = np.arange(0.5e9, 1.51e9, 10e6)
    power = 0.01 + ((frequency - 1.006e9) / 100e6) ** 2
    valid = np.ones(len(frequency), dtype=bool)
    return PortSpectrum(
        frequency,
        np.sqrt(power).astype(complex),
        valid,
        50 + 1j * (frequency - 1e9) / 1e6,
        valid.copy(),
        50,
        -80,
        Path("synthetic"),
        "/ports/feed",
        10e6,
    )


def test_editable_objective_scores_the_frequency_error_and_saves_the_spectrum():
    spectrum = dip_spectrum()
    saved = {}

    class Output:
        def port(self, name):
            assert name == "feed"
            return spectrum

        def save_npz(self, name, **arrays):
            saved[name] = arrays

    score = example.evaluate({"arm_cells": 69}, Output())
    assert score == pytest.approx(6e6)
    assert saved["s11.npz"]["resonance_hz"] == pytest.approx(1.006e9)
    np.testing.assert_array_equal(saved["s11.npz"]["s11"], spectrum.s11)
    metrics = benchmark.resonance_metrics(spectrum, 1e9, 10e6)
    assert score == metrics["frequency_error_hz"]


@pytest.mark.parametrize(
    "case", ["invalid_band", "coarse_resolution", "boundary", "shallow", "decay"]
)
def test_editable_resonance_helper_rejects_unresolved_spectra(case):
    spectrum = dip_spectrum()
    if case == "invalid_band":
        spectrum.valid[15] = False
    elif case == "coarse_resolution":
        spectrum = replace(spectrum, independent_frequency_resolution_hz=40e6)
    elif case == "boundary":
        spectrum = replace(spectrum, s11=(spectrum.frequency / 1e9).astype(complex))
    elif case == "shallow":
        spectrum = replace(spectrum, s11=np.sqrt(np.abs(spectrum.s11) ** 2 + 0.2).astype(complex))
    else:
        spectrum = replace(spectrum, tail_relative_db=-20)
    with pytest.raises(ValueError):
        example.estimate_resonance(spectrum, 1e9, 10e6)
