"""No public model command may silently swallow an unknown keyword."""

import ast
import inspect
from pathlib import Path

import pytest

import gprMax
from gprMax.user_objects.user_objects import GridUserObject, UserObject
from toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_400, antenna_like_GSSI_1500
from toolboxes.GPRAntennaModels.MALA import antenna_like_MALA_1200

pytestmark = pytest.mark.unit


COMMANDS = [
    cls
    for _, cls in inspect.getmembers(gprMax, inspect.isclass)
    if issubclass(cls, UserObject) and not inspect.isabstract(cls)
]


@pytest.mark.parametrize("command", COMMANDS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("value", [None, False, 0, ""])
def test_every_public_command_rejects_unknown_keywords(command, value):
    # Python's explicit signatures reject the extra keyword before evaluating
    # missing required arguments; legacy **kwargs constructors must do so too.
    with pytest.raises(TypeError, match="unexpected keyword.*misspelled_option"):
        command(misspelled_option=value)


@pytest.mark.parametrize(
    "command", [cls for cls in COMMANDS if cls._allowed_kwargs is not None], ids=lambda cls: cls.__name__
)
def test_declared_keyword_schema_covers_literal_keyword_reads(command):
    """Catch schema drift when a command gains an option in its implementation."""
    tree = ast.parse(inspect.getsource(command))
    used = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and ast.unparse(node.value) in ("self.kwargs", "kwargs")
            and isinstance(node.slice, ast.Constant)
        ):
            used.add(node.slice.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func.value) in ("self.kwargs", "kwargs")
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            used.add(node.args[0].value)
    assert used <= command._allowed_kwargs


def test_optional_typo_reports_command_and_suggests_exact_spelling():
    with pytest.raises(TypeError, match=r"Box \(#box\).*averagin.*did you mean 'averaging'"):
        gprMax.Box(p1=(0, 0, 0), p2=(1, 1, 1), material_id="free_space", averagin=False)
    with pytest.raises(TypeError, match="averagin.*materialid"):
        gprMax.Box(averagin=None, materialid=0)


@pytest.mark.parametrize("key,value", [("material_id", "free_space"), ("material_ids", ["free_space"] * 3)])
def test_geometry_material_forms_and_optional_controls_are_retained(key, value):
    kwargs = dict(p1=(0, 0, 0), p2=(1, 1, 1), averaging=False, tag="target")
    kwargs[key] = value
    assert gprMax.Box(**kwargs).kwargs == kwargs


@pytest.mark.parametrize(
    "command", [gprMax.DiscretePlaneWaveAngles, gprMax.DiscretePlaneWaveVector, gprMax.DiscretePlaneWaveAxial]
)
def test_plane_wave_window_keywords_used_by_shared_helper_are_retained(command):
    obj = command(start=0, stop=1e-9)
    assert obj.kwargs == {"start": 0, "stop": 1e-9}


def test_geometry_import_legacy_keyword_still_gives_migration_error():
    with pytest.raises(ValueError, match="convert-geometry"):
        gprMax.GeometryObjectsRead(p1=(0, 0, 0), geofile="old.h5", matfile="old.txt")


@pytest.mark.parametrize("antenna", [antenna_like_GSSI_1500, antenna_like_GSSI_400, antenna_like_MALA_1200])
@pytest.mark.parametrize("complete", [True, False])
@pytest.mark.parametrize("value", [None, False, 0, ""])
def test_antenna_toolboxes_reject_unknown_optimisation_parameters(antenna, complete, value):
    kwargs = {}
    if complete:
        if antenna is antenna_like_GSSI_1500:
            kwargs = dict(
                absorber1Er=2,
                absorber1sig=0.1,
                absorber2Er=3,
                absorber2sig=0.2,
                pcbEr=4,
                pcbsig=0.01,
                hdpeEr=2.3,
                hdpesig=0,
            )
        else:
            kwargs = dict(excitationfreq=1e9, sourceresistance=100, absorberEr=2, absorbersig=0.1)
    with pytest.raises(TypeError, match="unexpected keyword.*sourceresistnce"):
        antenna(0.5, 0.5, 0.1, **kwargs, sourceresistnce=value)


def test_run_rejects_misspelled_execution_option():
    with pytest.raises(TypeError, match="unexpected keyword.*geometry_fxed"):
        gprMax.run(geometry_fxed=True)


def test_custom_user_objects_can_define_their_own_keyword_contract():
    class CustomObject(GridUserObject):
        order = 1
        hash = "#custom"

        def build(self, grid):
            pass

    assert CustomObject(custom_control=None).kwargs == {"custom_control": None}

    class CustomBox(gprMax.Box):
        _allowed_kwargs = gprMax.Box._allowed_kwargs | {"custom_control"}

    assert CustomBox(custom_control=False).kwargs == {"custom_control": False}
    with pytest.raises(TypeError, match="unexpected keyword.*custom_contol"):
        CustomBox(custom_contol=False)


@pytest.mark.parametrize("value", [None, False, 0, ""])
@pytest.mark.parametrize(
    "study_type",
    [gprMax.GPRStudy, gprMax.PortStudy, gprMax.SourceStudy, gprMax.PlaneWaveStudy, gprMax.EigenmodeStudy],
    ids=lambda cls: cls.__name__,
)
def test_study_overrides_reject_unknown_keywords_before_build(value, study_type):
    scene = gprMax.Scene()
    if study_type is gprMax.GPRStudy:
        obj = gprMax.Rx(p1=(0, 0, 0))
    elif study_type in (gprMax.PortStudy, gprMax.SourceStudy):
        command = gprMax.VoltageSource if study_type is gprMax.PortStudy else gprMax.TransmissionLine
        obj = command(p1=(0, 0, 0), polarisation="z", resistance=50, waveform_id="pulse")
    elif study_type is gprMax.PlaneWaveStudy:
        obj = gprMax.DiscretePlaneWaveAxial(p1=(0, 0, 0), p2=(1, 1, 1), axis="x", psi=0, waveform_id="pulse")
    else:
        obj = gprMax.EigenmodeExcitation(port=1, mode=1)
        scene.add(gprMax.EigenmodePort(port=1, p1=(0, 0, 0), p2=(0, 1, 1), direction="+", modes=1))
    scene.add(obj)
    study = study_type(cases=[gprMax.StudyCase("one", [gprMax.ObjectState(obj, positon=value)])])
    with pytest.raises(ValueError, match="positon"):
        study.bind_scene(scene)


def test_filename_collision_warning_is_shared_by_all_references():
    docs = Path(__file__).resolve().parents[1] / "docs/source"
    include = ".. include:: _includes/geometry_output_filenames.rstinc"
    for name in ("input_api.rst", "input_hash_cmds.rst", "output.rst"):
        assert include in (docs / name).read_text()
    assert "without confirmation" in (docs / "_includes/geometry_output_filenames.rstinc").read_text()
