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

import math

import numpy as np
import pytest

from gprMax.user_objects.cmds_singleuse import (
    Discretisation,
    Domain,
    DomainMode,
    MagneticAveraging,
    OMPThreads,
    OutputDir,
    RxSteps,
    SrcSteps,
    TimeStepStabilityFactor,
    TimeWindow,
)


class TestDiscretisationValidation:
    @pytest.mark.parametrize("value", [None, 0.001, (0.001, 0.001), (0.001,) * 4])
    def test_wrong_length_rejected(self, stub_model, value):
        with pytest.raises(ValueError, match="three spatial steps"):
            Discretisation(p1=value).build(stub_model)

    @pytest.mark.parametrize("value", [(0.001, "0.001", 0.001), (0.001, True, 0.001)])
    def test_non_numeric_values_rejected(self, stub_model, value):
        with pytest.raises(ValueError, match="numeric spatial steps"):
            Discretisation(p1=value).build(stub_model)


class TestDomainValidation:
    @pytest.mark.parametrize("value", [None, 0.1, (0.1, 0.1), (0.1,) * 4])
    def test_wrong_length_rejected(self, stub_model, value):
        with pytest.raises(ValueError, match="three domain sizes"):
            Domain(p1=value).build(stub_model)

    @pytest.mark.parametrize("value", [(0.1, "0.1", 0.1), (0.1, True, 0.1)])
    def test_non_numeric_values_rejected(self, stub_model, value):
        with pytest.raises(ValueError, match="numeric domain sizes"):
            Domain(p1=value).build(stub_model)


class TestStepSizeValidation:
    @pytest.mark.parametrize("object_type", [SrcSteps, RxSteps])
    @pytest.mark.parametrize("value", [None, 0.001, (0.001, 0.002), (0.001,) * 4])
    def test_wrong_length_rejected(self, stub_model, object_type, value):
        with pytest.raises(ValueError, match="three step sizes"):
            object_type(p1=value).build(stub_model)

    @pytest.mark.parametrize("object_type", [SrcSteps, RxSteps])
    @pytest.mark.parametrize("value", [(0.001, "0.002", 0.003), (0.001, True, 0.003)])
    def test_non_numeric_values_rejected(self, stub_model, object_type, value):
        with pytest.raises(ValueError, match="numeric step sizes"):
            object_type(p1=value).build(stub_model)

    @pytest.mark.parametrize("object_type", [SrcSteps, RxSteps])
    @pytest.mark.parametrize("value", [(math.nan, 0, 0), (0, math.inf, 0), (0, 0, -math.inf)])
    def test_non_finite_values_rejected(self, stub_model, object_type, value):
        with pytest.raises(ValueError, match="finite step sizes"):
            object_type(p1=value).build(stub_model)

    @pytest.mark.parametrize("object_type,attribute", [(SrcSteps, "srcsteps"), (RxSteps, "rxsteps")])
    @pytest.mark.parametrize(
        "value",
        [
            (0, 0, 0),
            (-0.001, 0.002, -0.003),
            [0.001, 0.0, 0.001],
            np.array([0.001, 0.002, 0.003]),
        ],
    )
    def test_valid_offsets_accepted(self, stub_model, object_type, attribute, value):
        object_type(p1=value).build(stub_model)

        assert getattr(stub_model, attribute).shape == (3,)


class TestTimeStepStabilityFactorValidation:
    @pytest.mark.parametrize("value", [0, -0.5, 1.1, math.nan, math.inf])
    def test_out_of_range_factor_reports_range(self, stub_model, value):
        with pytest.raises(ValueError, match="between zero and one"):
            TimeStepStabilityFactor(f=value).build(stub_model)

    @pytest.mark.parametrize("value", ["0.5", True])
    def test_non_numeric_factor_rejected(self, stub_model, value):
        with pytest.raises(ValueError, match="between zero and one"):
            TimeStepStabilityFactor(f=value).build(stub_model)


class TestTimeWindowValidation:
    @pytest.mark.parametrize("value", [math.nan, math.inf, 0, -1e-9])
    def test_out_of_range_time_reports_range(self, stub_model, value):
        with pytest.raises(ValueError, match="finite and greater than zero"):
            TimeWindow(time=value).build(stub_model)

    @pytest.mark.parametrize("value", ["1e-9", True])
    def test_non_numeric_time_rejected(self, stub_model, value):
        with pytest.raises(ValueError, match="finite and greater than zero"):
            TimeWindow(time=value).build(stub_model)

    def test_string_iteration_count_rejected(self, stub_model):
        with pytest.raises(ValueError, match="positive integer"):
            TimeWindow(iterations="10").build(stub_model)


class TestOMPThreadsValidation:
    def test_negative_thread_count_reports_requirement(self, stub_model):
        with pytest.raises(ValueError, match="integer not less than one"):
            OMPThreads(n=-1).build(stub_model)


class TestModeStringValidation:
    def test_domain_mode_requires_a_string(self):
        with pytest.raises(ValueError, match="mode to be a string"):
            DomainMode(mode=None)

    def test_magnetic_averaging_requires_a_string(self):
        with pytest.raises(ValueError, match="mode to be a string"):
            MagneticAveraging(mode=None)

    def test_domain_mode_is_case_insensitive(self):
        assert DomainMode(mode="tm").requested_mode == "TM"

    def test_magnetic_averaging_is_case_insensitive(self):
        assert MagneticAveraging(mode="ARITHMETIC").mode == "arithmetic"


class TestOutputDirValidation:
    @pytest.mark.parametrize("value", [None, 1, object()])
    def test_non_path_values_rejected(self, value):
        with pytest.raises(ValueError, match="directory path"):
            OutputDir(dir=value)

    def test_pathlib_path_is_accepted(self, tmp_path):
        assert OutputDir(dir=tmp_path).output_dir == tmp_path


pytestmark = pytest.mark.unit
