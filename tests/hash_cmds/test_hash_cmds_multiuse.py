"""Tests for ``gprMax.hash_cmds_multiuse.process_multicmds``.

The dispatcher walks 18 command families. Each family is structured the
same way: iterate the list under ``multicmds[cmdname]``, ``split()`` each
instance into tokens, validate arity (often with multiple valid widths),
then construct the matching user-object and append it.

Tests drive the dispatcher with hand-built dicts so every family is
covered without touching the file system or globals.
"""

import pytest

from gprMax.hash_cmds_multiuse import process_multicmds
from gprMax.user_objects.cmds_multiuse import (
    PMLCFS,
    AddDebyeDispersion,
    AddDrudeDispersion,
    AddLorentzDispersion,
    DiscretePlaneWaveAngles,
    DiscretePlaneWaveAxial,
    DiscretePlaneWaveVector,
    ExcitationFile,
    HertzianDipole,
    MagneticDipole,
    Material,
    MaterialCrim,
    MaterialList,
    MaterialRange,
    Rx,
    RxArray,
    SoilPeplinski,
    TransmissionLine,
    VoltageSource,
    Waveform,
)
from gprMax.user_objects.cmds_output import GeometryObjectsWrite, GeometryView, Snapshot

# ---------------------------------------------------------------------------
# Sanity / shared behaviour
# ---------------------------------------------------------------------------


class TestEmptyDispatch:
    """Empty lists under every key must yield no scene objects."""

    def test_empty_dict_yields_empty_list(self, multicmds_template):
        assert process_multicmds(multicmds_template) == []

    def test_unrelated_family_does_not_pollute(self, multicmds_template):
        multicmds_template["#waveform"] = ["gaussian 1.0 1e9 wf1"]
        objs = process_multicmds(multicmds_template)
        assert len(objs) == 1
        assert isinstance(objs[0], Waveform)


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------


class TestWaveform:
    def test_four_tokens_become_waveform(self, multicmds_template):
        multicmds_template["#waveform"] = ["gaussian 2.0 1e9 wf1"]
        objs = process_multicmds(multicmds_template)
        assert isinstance(objs[0], Waveform)
        assert objs[0].kwargs == {
            "wave_type": "gaussian",
            "amp": 2.0,
            "freq": 1e9,
            "id": "wf1",
        }

    @pytest.mark.parametrize(
        "payload",
        ["gaussian 1.0 1e9", "gaussian 1.0 1e9 wf1 extra"],
    )
    def test_wrong_arity_rejected(self, multicmds_template, payload):
        multicmds_template["#waveform"] = [payload]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)

    def test_multiple_instances_all_dispatched(self, multicmds_template):
        multicmds_template["#waveform"] = [
            "gaussian 1.0 1e9 wf1",
            "ricker 2.0 5e8 wf2",
        ]
        objs = process_multicmds(multicmds_template)
        assert len(objs) == 2
        assert objs[0].kwargs["id"] == "wf1"
        assert objs[1].kwargs["id"] == "wf2"


# ---------------------------------------------------------------------------
# Voltage source
# ---------------------------------------------------------------------------


class TestVoltageSource:
    def test_six_token_short_form(self, multicmds_template):
        multicmds_template["#voltage_source"] = ["x 0.05 0.05 0.05 50 wf1"]
        objs = process_multicmds(multicmds_template)
        vs = objs[0]
        assert isinstance(vs, VoltageSource)
        assert vs.polarisation == "x"
        assert vs.point == (0.05, 0.05, 0.05)
        assert vs.resistance == 50.0
        assert vs.waveform_id == "wf1"
        assert vs.start is None
        assert vs.stop is None

    def test_eight_token_with_window(self, multicmds_template):
        multicmds_template["#voltage_source"] = ["y 0.1 0.2 0.3 75 wfA 1e-9 5e-9"]
        objs = process_multicmds(multicmds_template)
        vs = objs[0]
        assert vs.polarisation == "y"
        assert vs.start == 1e-9
        assert vs.stop == 5e-9

    def test_polarisation_lowercased(self, multicmds_template):
        # Dispatcher applies ``.lower()`` to the polarisation token
        multicmds_template["#voltage_source"] = ["X 0.05 0.05 0.05 50 wf1"]
        objs = process_multicmds(multicmds_template)
        assert objs[0].polarisation == "x"

    @pytest.mark.parametrize(
        "payload",
        ["x 0 0 0 50 wf1 1e-9", "x 0 0 0 50 wf1 1e-9 5e-9 extra"],
    )
    def test_invalid_arity_rejected(self, multicmds_template, payload):
        multicmds_template["#voltage_source"] = [payload]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


# ---------------------------------------------------------------------------
# Hertzian dipole
# ---------------------------------------------------------------------------


class TestHertzianDipole:
    def test_five_token_short_form(self, multicmds_template):
        multicmds_template["#hertzian_dipole"] = ["z 0.01 0.02 0.03 wf1"]
        objs = process_multicmds(multicmds_template)
        hd = objs[0]
        assert isinstance(hd, HertzianDipole)
        assert hd.polarisation == "z"
        assert hd.point == (0.01, 0.02, 0.03)
        assert hd.waveform_id == "wf1"

    def test_seven_token_with_window(self, multicmds_template):
        multicmds_template["#hertzian_dipole"] = ["z 0.0 0.0 0.0 wf1 1e-9 5e-9"]
        objs = process_multicmds(multicmds_template)
        assert objs[0].start == 1e-9
        assert objs[0].stop == 5e-9

    @pytest.mark.parametrize(
        "payload",
        ["z 0 0 0", "z 0 0 0 wf1 1e-9 5e-9 extra"],
    )
    def test_invalid_arity_rejected(self, multicmds_template, payload):
        multicmds_template["#hertzian_dipole"] = [payload]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


# ---------------------------------------------------------------------------
# Magnetic dipole
# ---------------------------------------------------------------------------


class TestMagneticDipole:
    def test_five_token_short_form(self, multicmds_template):
        multicmds_template["#magnetic_dipole"] = ["y 0.0 0.0 0.0 wf1"]
        objs = process_multicmds(multicmds_template)
        assert isinstance(objs[0], MagneticDipole)
        assert objs[0].polarisation == "y"

    def test_seven_token_with_window(self, multicmds_template):
        multicmds_template["#magnetic_dipole"] = ["y 0.0 0.0 0.0 wf1 0 1e-9"]
        objs = process_multicmds(multicmds_template)
        assert objs[0].start == 0.0
        assert objs[0].stop == 1e-9

    @pytest.mark.parametrize(
        "payload",
        ["y 0 0 0", "y 0 0 0 wf1 0 1e-9 extra"],
    )
    def test_invalid_arity_rejected(self, multicmds_template, payload):
        multicmds_template["#magnetic_dipole"] = [payload]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


# ---------------------------------------------------------------------------
# Transmission line
# ---------------------------------------------------------------------------


class TestTransmissionLine:
    def test_six_token_short_form(self, multicmds_template):
        multicmds_template["#transmission_line"] = ["x 0.05 0.05 0.05 50 wf1"]
        objs = process_multicmds(multicmds_template)
        tl = objs[0]
        assert isinstance(tl, TransmissionLine)
        assert tl.resistance == 50.0

    def test_eight_token_with_numeric_window(self, multicmds_template):
        multicmds_template["#transmission_line"] = ["x 0 0 0 50 wf1 1e-9 5e-9"]
        objs = process_multicmds(multicmds_template)
        assert objs[0].start == pytest.approx(1e-9)
        assert objs[0].stop == pytest.approx(5e-9)

    @pytest.mark.parametrize(
        "payload",
        ["x 0 0 0 50", "x 0 0 0 50 wf1 1e-9 5e-9 extra"],
    )
    def test_invalid_arity_rejected(self, multicmds_template, payload):
        multicmds_template["#transmission_line"] = [payload]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


# ---------------------------------------------------------------------------
# Plane wave: angles / axial / vector
# ---------------------------------------------------------------------------


class TestPlaneWaveAngles:
    def test_ten_token_minimum(self, multicmds_template):
        multicmds_template["#plane_wave_angles"] = ["0 0 0 0.1 0.1 0.1 30 60 90 wf1"]
        objs = process_multicmds(multicmds_template)
        assert isinstance(objs[0], DiscretePlaneWaveAngles)
        assert objs[0].kwargs["theta"] == 30.0
        assert objs[0].kwargs["phi"] == 60.0
        assert objs[0].kwargs["psi"] == 90.0
        assert objs[0].kwargs["waveform_id"] == "wf1"

    def test_eleven_token_with_material(self, multicmds_template):
        multicmds_template["#plane_wave_angles"] = ["0 0 0 0.1 0.1 0.1 30 60 90 wf1 air"]
        objs = process_multicmds(multicmds_template)
        assert objs[0].kwargs["material_id"] == "air"

    def test_thirteen_token_with_window(self, multicmds_template):
        multicmds_template["#plane_wave_angles"] = ["0 0 0 0.1 0.1 0.1 30 60 90 wf1 air 1e-9 5e-9"]
        objs = process_multicmds(multicmds_template)
        assert objs[0].kwargs["start"] == 1e-9
        assert objs[0].kwargs["stop"] == 5e-9

    @pytest.mark.parametrize(
        "payload",
        [
            "0 0 0 0.1 0.1 0.1 30 60 90",  # 9 tokens — below minimum
            "0 0 0 0.1 0.1 0.1 30 60 90 wf1 air 1e-9 5e-9 extra",  # 14
        ],
    )
    def test_invalid_arity_rejected(self, multicmds_template, payload):
        multicmds_template["#plane_wave_angles"] = [payload]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


class TestPlaneWaveAxial:
    def test_nine_token_minimum(self, multicmds_template):
        multicmds_template["#plane_wave_axial"] = ["0 0 0 0.1 0.1 0.1 90 X wf1"]
        objs = process_multicmds(multicmds_template)
        assert isinstance(objs[0], DiscretePlaneWaveAxial)
        # axis is lowercased before being passed in
        assert objs[0].kwargs["axis"] == "x"

    def test_eleven_token_with_window(self, multicmds_template):
        multicmds_template["#plane_wave_axial"] = ["0 0 0 0.1 0.1 0.1 90 y wf1 1e-9 5e-9"]
        objs = process_multicmds(multicmds_template)
        assert objs[0].kwargs["start"] == 1e-9
        assert objs[0].kwargs["stop"] == 5e-9

    @pytest.mark.parametrize(
        "payload",
        [
            "0 0 0 0.1 0.1 0.1 90 x",  # 8 tokens — below minimum
            "0 0 0 0.1 0.1 0.1 90 x wf1 1e-9",  # 10 — skipped arity (not in 9/11)
            "0 0 0 0.1 0.1 0.1 90 x wf1 1e-9 5e-9 extra",  # 12
        ],
    )
    def test_invalid_arity_rejected(self, multicmds_template, payload):
        multicmds_template["#plane_wave_axial"] = [payload]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


class TestPlaneWaveVector:
    def test_eleven_token_minimum(self, multicmds_template):
        multicmds_template["#plane_wave_vector"] = ["0 0 0 0.1 0.1 0.1 1 0 0 90 wf1"]
        objs = process_multicmds(multicmds_template)
        pw = objs[0]
        assert isinstance(pw, DiscretePlaneWaveVector)
        assert pw.kwargs["m_vec"] == (1, 0, 0)
        assert pw.kwargs["psi"] == 90.0
        assert pw.kwargs["waveform_id"] == "wf1"

    def test_twelve_token_with_material(self, multicmds_template):
        multicmds_template["#plane_wave_vector"] = ["0 0 0 0.1 0.1 0.1 1 0 0 90 wf1 air"]
        objs = process_multicmds(multicmds_template)
        assert objs[0].kwargs["material_id"] == "air"

    @pytest.mark.parametrize(
        "payload",
        [
            "0 0 0 0.1 0.1 0.1 1 0 0 90",  # 10 — fails both inner branches
            "0 0 0 0.1 0.1 0.1 1 0 0 90 wf1 air 1e-9 extra",  # 14 — "too many"
        ],
    )
    def test_invalid_arity_rejected(self, multicmds_template, payload):
        multicmds_template["#plane_wave_vector"] = [payload]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


class TestPlaneWaveVectorIndexBug:
    """Bug tripwire: ``hash_cmds_multiuse.py:294``.

    The 13-token branch of ``#plane_wave_vector`` reaches ``tmp[13]`` for
    ``stop=float(tmp[13])``. With 13 tokens the valid index range is
    ``0..12``, so this raises ``IndexError``. The branch is unreachable
    through valid input.

    Pin the crash. When the fix lands (e.g. ``stop=float(tmp[12])`` and a
    14-token branch), this test fails — flip it to a passing assertion
    on a populated ``start``/``stop`` pair.
    """

    def test_thirteen_token_branch_now_value_error(self, multicmds_template):
        """Upstream commit d6fc8069 fixed swapped token-count checks.
        The 13-token branch now correctly rejects with ValueError."""
        multicmds_template["#plane_wave_vector"] = ["0 0 0 0.1 0.1 0.1 1 0 0 90 wf1 air 1e-9"]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


# ---------------------------------------------------------------------------
# Excitation file
# ---------------------------------------------------------------------------


class TestExcitationFile:
    def test_single_token_filepath_only(self, multicmds_template):
        multicmds_template["#excitation_file"] = ["my_excite.txt"]
        objs = process_multicmds(multicmds_template)
        assert isinstance(objs[0], ExcitationFile)
        assert objs[0].kwargs["filepath"] == "my_excite.txt"

    def test_three_token_with_kind_and_fill(self, multicmds_template):
        multicmds_template["#excitation_file"] = ["my.txt linear 0.0"]
        objs = process_multicmds(multicmds_template)
        assert objs[0].kwargs["filepath"] == "my.txt"
        assert objs[0].kwargs["kind"] == "linear"
        assert objs[0].kwargs["fill_value"] == 0.0

    @pytest.mark.parametrize("payload", ["a b", "a b c d"])
    def test_invalid_arity_rejected(self, multicmds_template, payload):
        multicmds_template["#excitation_file"] = [payload]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


# ---------------------------------------------------------------------------
# Rx and Rx array
# ---------------------------------------------------------------------------


class TestRx:
    def test_three_token_minimal(self, multicmds_template):
        multicmds_template["#rx"] = ["0.05 0.05 0.05"]
        objs = process_multicmds(multicmds_template)
        rx = objs[0]
        assert isinstance(rx, Rx)
        assert rx.point == (0.05, 0.05, 0.05)
        assert rx.id is None
        assert rx.outputs is None

    def test_five_token_with_id_and_outputs(self, multicmds_template):
        multicmds_template["#rx"] = ["0.05 0.05 0.05 my_rx Ex Ey"]
        objs = process_multicmds(multicmds_template)
        rx = objs[0]
        assert rx.id == "my_rx"
        # outputs collects the tail (everything after id)
        assert rx.outputs == ["Ex", "Ey"]

    @pytest.mark.parametrize("payload", ["0.05 0.05", "0.05 0.05 0.05 my_rx"])
    def test_invalid_arity_rejected(self, multicmds_template, payload):
        # 2 tokens (too few) and 4 tokens (3 < n < 5 is rejected).
        multicmds_template["#rx"] = [payload]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


class TestRxArray:
    def test_nine_token_form(self, multicmds_template):
        multicmds_template["#rx_array"] = ["0 0 0 0.1 0.1 0.1 0.01 0.01 0.01"]
        objs = process_multicmds(multicmds_template)
        rxa = objs[0]
        assert isinstance(rxa, RxArray)
        assert rxa.kwargs["p1"] == (0.0, 0.0, 0.0)
        assert rxa.kwargs["p2"] == (0.1, 0.1, 0.1)
        assert rxa.kwargs["dl"] == (0.01, 0.01, 0.01)

    @pytest.mark.parametrize(
        "payload",
        ["0 0 0 0.1 0.1 0.1 0.01 0.01", "0 0 0 0.1 0.1 0.1 0.01 0.01 0.01 extra"],
    )
    def test_invalid_arity_rejected(self, multicmds_template, payload):
        multicmds_template["#rx_array"] = [payload]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_integer_iterations_branch(self, multicmds_template):
        multicmds_template["#snapshot"] = ["0 0 0 0.1 0.1 0.1 0.01 0.01 0.01 100 snap.vti"]
        objs = process_multicmds(multicmds_template)
        snap = objs[0]
        assert isinstance(snap, Snapshot)
        assert snap.kwargs["iterations"] == 100
        assert "time" not in snap.kwargs or snap.kwargs.get("time") is None
        assert snap.kwargs["filename"] == "snap.vti"
        assert snap.kwargs["fileext"] == ".vti"

    def test_float_time_branch(self, multicmds_template):
        # int("1e-9") raises -> fall through to time branch
        multicmds_template["#snapshot"] = ["0 0 0 0.1 0.1 0.1 0.01 0.01 0.01 1e-9 snap.h5"]
        objs = process_multicmds(multicmds_template)
        snap = objs[0]
        assert snap.kwargs["time"] == 1e-9
        assert snap.kwargs["fileext"] == ".h5"

    def test_filename_without_extension_sets_none(self, multicmds_template):
        multicmds_template["#snapshot"] = ["0 0 0 0.1 0.1 0.1 0.01 0.01 0.01 100 plain_name"]
        objs = process_multicmds(multicmds_template)
        assert objs[0].kwargs["fileext"] is None

    def test_wrong_arity_rejected(self, multicmds_template):
        multicmds_template["#snapshot"] = ["0 0 0 0.1 0.1 0.1 0.01 0.01 0.01 100"]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


# ---------------------------------------------------------------------------
# Material families
# ---------------------------------------------------------------------------


class TestMaterial:
    def test_five_token_form(self, multicmds_template):
        multicmds_template["#material"] = ["4.0 0.01 1.0 0.0 concrete"]
        objs = process_multicmds(multicmds_template)
        mat = objs[0]
        assert isinstance(mat, Material)
        assert mat.kwargs == {
            "er": 4.0,
            "se": 0.01,
            "mr": 1.0,
            "sm": 0.0,
            "id": "concrete",
        }

    @pytest.mark.parametrize("payload", ["4.0 0.01 1.0 0.0", "4.0 0.01 1.0 0.0 concrete extra"])
    def test_invalid_arity_rejected(self, multicmds_template, payload):
        multicmds_template["#material"] = [payload]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


class TestAddDispersionDebye:
    def test_single_pole(self, multicmds_template):
        # poles=1 -> 1 pair (delta_er, tau), then material_ids
        multicmds_template["#add_dispersion_debye"] = ["1 5.0 1e-12 m1"]
        objs = process_multicmds(multicmds_template)
        d = objs[0]
        assert isinstance(d, AddDebyeDispersion)
        assert d.kwargs["poles"] == 1
        assert d.kwargs["er_delta"] == [5.0]
        assert d.kwargs["tau"] == [1e-12]
        assert d.kwargs["material_ids"] == ["m1"]

    def test_two_poles_multiple_materials(self, multicmds_template):
        # poles=2 -> 2 pairs (4 floats), then any number of material ids
        multicmds_template["#add_dispersion_debye"] = ["2 5.0 1e-12 3.0 2e-12 m1 m2"]
        objs = process_multicmds(multicmds_template)
        d = objs[0]
        assert d.kwargs["poles"] == 2
        assert d.kwargs["er_delta"] == [5.0, 3.0]
        assert d.kwargs["tau"] == [1e-12, 2e-12]
        assert d.kwargs["material_ids"] == ["m1", "m2"]

    def test_below_minimum_arity_rejected(self, multicmds_template):
        multicmds_template["#add_dispersion_debye"] = ["1 5.0 1e-12"]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


class TestAddDispersionLorentz:
    def test_single_pole(self, multicmds_template):
        # poles=1 -> 1 triple (delta_er, omega, delta), then material ids
        multicmds_template["#add_dispersion_lorentz"] = ["1 5.0 1e9 0.1 m1"]
        objs = process_multicmds(multicmds_template)
        lz = objs[0]
        assert isinstance(lz, AddLorentzDispersion)
        assert lz.kwargs["poles"] == 1
        assert lz.kwargs["er_delta"] == [5.0]
        assert lz.kwargs["omega"] == [1e9]
        assert lz.kwargs["delta"] == [0.1]
        assert lz.kwargs["material_ids"] == ["m1"]

    def test_below_minimum_arity_rejected(self, multicmds_template):
        multicmds_template["#add_dispersion_lorentz"] = ["1 5.0 1e9 0.1"]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


class TestAddDispersionDrude:
    def test_single_pole(self, multicmds_template):
        # poles=1 -> 1 pair (omega, alpha), then material ids
        multicmds_template["#add_dispersion_drude"] = ["1 1e9 0.05 m1"]
        objs = process_multicmds(multicmds_template)
        dr = objs[0]
        assert isinstance(dr, AddDrudeDispersion)
        assert dr.kwargs["poles"] == 1
        assert dr.kwargs["omega"] == [1e9]
        assert dr.kwargs["alpha"] == [0.05]
        assert dr.kwargs["material_ids"] == ["m1"]

    def test_below_minimum_arity_rejected(self, multicmds_template):
        multicmds_template["#add_dispersion_drude"] = ["1 1e9 0.05"]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


class TestSoilPeplinski:
    def test_seven_token_form(self, multicmds_template):
        multicmds_template["#soil_peplinski"] = ["0.5 0.3 1.5 2.6 0.05 0.2 soil1"]
        objs = process_multicmds(multicmds_template)
        soil = objs[0]
        assert isinstance(soil, SoilPeplinski)
        assert soil.kwargs == {
            "sand_fraction": 0.5,
            "clay_fraction": 0.3,
            "bulk_density": 1.5,
            "sand_density": 2.6,
            "water_fraction_lower": 0.05,
            "water_fraction_upper": 0.2,
            "id": "soil1",
        }

    @pytest.mark.parametrize(
        "payload",
        ["0.5 0.3 1.5 2.6 0.05 0.2", "0.5 0.3 1.5 2.6 0.05 0.2 soil1 extra"],
    )
    def test_invalid_arity_rejected(self, multicmds_template, payload):
        multicmds_template["#soil_peplinski"] = [payload]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


class TestMaterialRange:
    def test_nine_token_form(self, multicmds_template):
        multicmds_template["#material_range"] = ["1.0 5.0 0.0 0.1 1.0 1.5 0.5 2.0 my_range"]
        objs = process_multicmds(multicmds_template)
        mr = objs[0]
        assert isinstance(mr, MaterialRange)
        assert mr.kwargs["er_lower"] == 1.0
        assert mr.kwargs["er_upper"] == 5.0
        assert mr.kwargs["id"] == "my_range"

    def test_wrong_arity_rejected(self, multicmds_template):
        multicmds_template["#material_range"] = ["1.0 5.0 0.0 0.1 1.0 1.5 0.5 2.0"]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


class TestMaterialList:
    def test_variadic_last_token_is_id(self, multicmds_template):
        # all but the last token are existing materials; last token is the id
        multicmds_template["#material_list"] = ["mat1 mat2 mat3 mixed"]
        objs = process_multicmds(multicmds_template)
        ml = objs[0]
        assert isinstance(ml, MaterialList)
        assert ml.kwargs["list_of_materials"] == ["mat1", "mat2", "mat3"]
        assert ml.kwargs["id"] == "mixed"

    def test_two_tokens_is_minimum(self, multicmds_template):
        multicmds_template["#material_list"] = ["mat1 my_id"]
        objs = process_multicmds(multicmds_template)
        ml = objs[0]
        assert ml.kwargs["list_of_materials"] == ["mat1"]
        assert ml.kwargs["id"] == "my_id"

    def test_single_token_rejected(self, multicmds_template):
        multicmds_template["#material_list"] = ["solo"]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


class TestMaterialCrim:
    def test_nine_token_form(self, multicmds_template):
        multicmds_template["#material_crim"] = ["sand 0.6 water 0.02 0.35 1e6 3e9 0.5 wetsand"]
        objs = process_multicmds(multicmds_template)
        crim = objs[0]

        assert isinstance(crim, MaterialCrim)
        assert crim.kwargs == {
            "matrix_id": "sand",
            "matrix_fraction": 0.6,
            "dispersive_id": "water",
            "fraction_lower": 0.02,
            "fraction_upper": 0.35,
            "f_min": 1e6,
            "f_max": 3e9,
            "a": 0.5,
            "id": "wetsand",
        }

    @pytest.mark.parametrize(
        "payload",
        [
            "sand 0.6 water 0.02 0.35 1e6 3e9 0.5",
            "sand 0.6 water 0.02 0.35 1e6 3e9 0.5 wetsand extra",
        ],
    )
    def test_wrong_arity_rejected(self, multicmds_template, payload):
        multicmds_template["#material_crim"] = [payload]
        with pytest.raises(ValueError, match="requires exactly nine parameters"):
            process_multicmds(multicmds_template)


# ---------------------------------------------------------------------------
# Output: geometry view / write
# ---------------------------------------------------------------------------


class TestGeometryView:
    def test_eleven_token_form(self, multicmds_template):
        multicmds_template["#geometry_view"] = ["0 0 0 0.1 0.1 0.1 0.01 0.01 0.01 view1 n"]
        objs = process_multicmds(multicmds_template)
        gv = objs[0]
        assert isinstance(gv, GeometryView)
        assert gv.kwargs["filename"] == "view1"
        assert gv.kwargs["output_type"] == "n"

    def test_wrong_arity_rejected(self, multicmds_template):
        multicmds_template["#geometry_view"] = ["0 0 0 0.1 0.1 0.1 0.01 0.01 0.01 view1"]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


class TestGeometryObjectsWrite:
    def test_seven_token_form(self, multicmds_template):
        multicmds_template["#geometry_objects_write"] = ["0 0 0 0.1 0.1 0.1 outfile"]
        objs = process_multicmds(multicmds_template)
        gow = objs[0]
        assert isinstance(gow, GeometryObjectsWrite)
        assert gow.kwargs["filename"] == "outfile"

    def test_wrong_arity_rejected(self, multicmds_template):
        multicmds_template["#geometry_objects_write"] = ["0 0 0 0.1 0.1 0.1"]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


# ---------------------------------------------------------------------------
# PML CFS
# ---------------------------------------------------------------------------


class TestPMLCFS:
    def test_twelve_token_form(self, multicmds_template):
        multicmds_template["#pml_cfs"] = [
            "constant forward 0 0 quartic forward 1 5 quartic forward 0 1"
        ]
        objs = process_multicmds(multicmds_template)
        cfs = objs[0]
        assert isinstance(cfs, PMLCFS)
        # All 12 tokens are passed through as strings under their named kwargs
        assert cfs.kwargs["alphascalingprofile"] == "constant"
        assert cfs.kwargs["sigmamax"] == "1"

    def test_wrong_arity_rejected(self, multicmds_template):
        multicmds_template["#pml_cfs"] = ["constant forward 0 0"]
        with pytest.raises(ValueError):
            process_multicmds(multicmds_template)


# ---------------------------------------------------------------------------
# Cross-family ordering — single dispatch pass collects all
# ---------------------------------------------------------------------------


class TestMultiFamilyDispatch:
    def test_independent_families_combine_in_source_order(self, multicmds_template):
        multicmds_template["#waveform"] = ["gaussian 1 1e9 wf1"]
        multicmds_template["#material"] = ["4 0.01 1 0 m1"]
        multicmds_template["#rx"] = ["0.05 0.05 0.05"]
        objs = process_multicmds(multicmds_template)
        # Dispatcher walks waveform -> sources -> ... -> rx -> ... -> material
        types = [type(o) for o in objs]
        assert Waveform in types
        assert Rx in types
        assert Material in types
        assert types.index(Waveform) < types.index(Rx)
        assert types.index(Rx) < types.index(Material)


pytestmark = pytest.mark.unit
