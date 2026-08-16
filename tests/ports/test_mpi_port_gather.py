from types import SimpleNamespace

import numpy as np

from gprMax.mpi_model import MPIModel
from gprMax.ports import RationalNetworkPortOutput, VoltageSourcePortMonitor
from gprMax.user_objects.cmds_output import _reserve_mpi_port_output_id


def test_mpi_port_id_reservation_keeps_coordinator_owner():
    grid = SimpleNamespace(mpi_port_output_ids=[], mpi_port_output_owners={})
    first = SimpleNamespace()
    second = SimpleNamespace()

    assert _reserve_mpi_port_output_id(grid, None, first) == "port1"
    assert _reserve_mpi_port_output_id(grid, "feed", second) == "feed"
    assert grid.mpi_port_output_owners == {"port1": first, "feed": second}


def test_voltage_port_rebind_uses_gathered_global_source_and_receiver():
    old_source = SimpleNamespace(polarisation="z", coord=np.asarray((8, 9, 10)))
    old_receiver = SimpleNamespace()
    monitor = VoltageSourcePortMonitor("feed", old_source, old_receiver, 10)
    source = SimpleNamespace(polarisation="z", coord=np.asarray((8, 9, 10)))
    receiver = SimpleNamespace(
        internal=True,
        port_id="feed",
        coord=np.asarray((8, 9, 10)),
    )
    grid = SimpleNamespace(
        rxs=[receiver],
        voltagesources=[source],
        dx=0.1,
        dy=0.2,
        dz=0.3,
    )

    monitor.rebind_after_mpi_gather(grid)

    assert monitor.source is source
    assert monitor.receiver is receiver
    np.testing.assert_allclose(monitor.source_position, (0.8, 1.8, 3.0))


def test_network_port_rebind_uses_gathered_global_terminal():
    monitor = RationalNetworkPortOutput.__new__(RationalNetworkPortOutput)
    monitor.output_id = "load"
    terminal = SimpleNamespace(ID="load", coord=np.asarray((4, 5, 6)), output=None)
    grid = SimpleNamespace(networkterminals=[terminal], dx=0.1, dy=0.2, dz=0.3)

    monitor.rebind_after_mpi_gather(grid)

    assert monitor.terminal is terminal
    assert monitor.source is terminal
    assert terminal.output is monitor
    np.testing.assert_allclose(monitor.source_position, (0.4, 1.0, 1.8))


def test_frill_rx_port_owner_rebinds_to_gathered_source():
    owner = SimpleNamespace(
        _monitor=None,
        _frill_source=None,
        point=(0.008, 0.009, 0.010),
    )
    source = SimpleNamespace(coord=np.asarray((8, 9, 10), dtype=np.int32))
    model = MPIModel.__new__(MPIModel)
    model.G = SimpleNamespace(
        dx=0.001,
        dy=0.001,
        dz=0.001,
        mpi_port_output_owners={"override": owner},
        magneticfrillsources=[source],
    )

    model._rebind_mpi_frill_port_owners()

    assert owner._frill_source is source
