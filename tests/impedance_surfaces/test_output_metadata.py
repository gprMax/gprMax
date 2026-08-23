"""Reproducibility metadata for surface-impedance models."""

from types import SimpleNamespace

import h5py
import numpy as np

from gprMax.fields_outputs import _write_surface_impedance_metadata
from gprMax.impedance_surfaces import SurfaceImpedanceModel
from gprMax.surface_impedance_presets import fit_metal_surface_impedance


def test_used_metal_model_writes_continuous_and_discrete_metadata(tmp_path):
    fit = fit_metal_surface_impedance("copper", 1e8, 1e10, 8)
    model = SurfaceImpedanceModel(
        "copper_wall",
        A=fit.A,
        B=fit.B,
        C=fit.C,
        D=fit.D,
        fit_fmin_hz=fit.fmin_hz,
        fit_fmax_hz=fit.fmax_hz,
        preset=fit.preset.key,
        provenance=fit.preset.source,
        fit_max_relative_error=fit.max_relative_error,
    )
    dt = 2e-13
    grid = SimpleNamespace(
        dt=dt,
        surface_impedance_models={model.ID: model},
        impedance_surfaces=SimpleNamespace(model_ids=(model.ID,)),
    )
    path = tmp_path / "metadata.h5"
    with h5py.File(path, "w") as output:
        _write_surface_impedance_metadata(output, grid)

    discrete = model.discretise(dt)
    with h5py.File(path, "r") as output:
        parent = output["surface_impedance_models"]
        saved = parent["model1"]
        assert parent.attrs["TimeConvention"] == "exp(+j*omega*t)"
        assert parent.attrs["SurfaceNormalConvention"] == "metal_to_retained_dielectric"
        assert saved.attrs["ID"] == model.ID
        assert saved.attrs["ModelHashSHA256"] == model.model_hash
        assert saved.attrs["Preset"] == "copper"
        assert saved.attrs["ReferenceTemperatureK"] == 293.0
        assert saved.attrs["UsedByCompiledBoundary"]
        np.testing.assert_array_equal(saved["A"], model.A)
        np.testing.assert_array_equal(saved["B"], model.B)
        np.testing.assert_array_equal(saved["C"], model.C)
        np.testing.assert_array_equal(saved["fdtd_discrete/F"], discrete.F)
        np.testing.assert_array_equal(saved["fdtd_discrete/G"], discrete.G)
        np.testing.assert_array_equal(saved["fdtd_discrete/L"], discrete.L)
        assert saved["fdtd_discrete"].attrs["Z0"] == discrete.Z0
