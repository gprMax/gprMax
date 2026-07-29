"""Hash-command coverage for the positional RxPort interface."""

import pytest

from gprMax.hash_cmds_file import get_user_objects
from gprMax.user_objects.cmds_output import RxPort


def _parse(command):
    return get_user_objects([f"{command}\n"], checkessential=False)


@pytest.mark.parametrize(
    "command, output_id, spectrum_limit",
    [
        ("#rx_port: 0.1 0.2 0.3", None, 10),
        ("#rx_port: 0.1 0.2 0.3 feed", "feed", 10),
        ("#rx_port: 0.1 0.2 0.3 feed 15", "feed", 15),
        ("#rx_port: 0.1 0.2 0.3 feed nyquist", "feed", "nyquist"),
    ],
)
def test_rx_port_positional_forms(command, output_id, spectrum_limit):
    objects = _parse(command)

    assert len(objects) == 1
    assert isinstance(objects[0], RxPort)
    assert objects[0].ID == output_id
    assert objects[0].spectrum_limit == spectrum_limit


@pytest.mark.parametrize(
    "command",
    [
        "#rx_port: 0.1 0.2",
        "#rx_port: 0.1 0.2 0.3 feed 10 extra",
        "#rx_port: 0.1 0.2 0.3 feed full",
        "#rx_port: 0.1 0.2 0.3 feed 2",
        "#rx_port: 0.1 0.2 0.3 feed nan",
    ],
)
def test_rx_port_rejects_malformed_spectrum_limit(command):
    with pytest.raises(ValueError):
        _parse(command)


def test_nondefault_api_limit_requires_id_for_positional_round_trip():
    with pytest.raises(ValueError, match="requires an ID"):
        RxPort((0.1, 0.2, 0.3), spectrum_limit="nyquist")
