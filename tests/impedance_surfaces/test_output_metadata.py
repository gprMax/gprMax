# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

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
        conductivity_s_per_m=fit.conductivity_s_per_m,
        fit_requested_order=fit.requested_order,
        fit_pole_count=fit.selected_pole_count,
        fit_tolerance=fit.tolerance,
        fit_max_relative_error=fit.max_relative_error,
        fit_rms_relative_error=fit.rms_relative_error,
        fit_method="passive-foster-bvls-v2",
        plot_fit_in_full_run=True,
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
        assert parent.attrs["SchemaVersion"] == 3
        assert parent.attrs["TimeConvention"] == "exp(+j*omega*t)"
        assert parent.attrs["SurfaceNormalConvention"] == "metal_to_retained_dielectric"
        assert saved.attrs["ID"] == model.ID
        assert saved.attrs["ModelHashSHA256"] == model.model_hash
        assert saved.attrs["Preset"] == "copper"
        assert saved.attrs["SourceKind"] == "preset"
        assert saved.attrs["ReferenceTemperatureK"] == 293.0
        assert saved.attrs["ConductivitySiemensPerMetre"] == fit.conductivity_s_per_m
        assert saved.attrs["FitRequestedOrder"] == "8"
        assert saved.attrs["FitPoleCount"] == 8
        assert saved.attrs["FitTolerance"] == fit.tolerance
        assert saved.attrs["FitMaximumRelativeError"] == fit.max_relative_error
        assert saved.attrs["FitRMSRelativeError"] == fit.rms_relative_error
        assert saved.attrs["FitMethod"] == "passive-foster-bvls-v2"
        assert saved.attrs["FitPlotInFullRun"]
        assert saved.attrs["UsedByCompiledBoundary"]
        np.testing.assert_array_equal(saved["A"], model.A)
        np.testing.assert_array_equal(saved["B"], model.B)
        np.testing.assert_array_equal(saved["C"], model.C)
        runtime = saved["fdtd_discrete"]
        assert set(runtime) == {"f", "q"}
        assert runtime.attrs["StateVariable"] == "y = L * x"
        assert runtime.attrs["Recurrence"] == "y_new = f * y + q * K"
        np.testing.assert_array_equal(runtime["f"], np.diag(discrete.F))
        np.testing.assert_array_equal(runtime["q"], discrete.L * discrete.G)
        assert saved["fdtd_discrete"].attrs["Z0"] == discrete.Z0
