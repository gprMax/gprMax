"""Electrical equivalence of the toolbox receiver and its automatic port.

Small grids exercise the actual gap material/source objects without making
the regular test suite run a full-size, highly conducting antenna model.
"""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_2000
from gprMax.toolboxes.Utilities.trace_time import read_time_history


def _scene(legacy, metal):
    objects = antenna_like_GSSI_2000(0.125, 0.1, 0.05)
    sources = {o.id: o for o in objects if isinstance(o, gprMax.VoltageSource)}
    background = next(o for o in objects if isinstance(o, gprMax.Material) and o.kwargs["id"] == "gssi2000_rxres")
    scene = gprMax.Scene()
    tx, rx = (0.007, 0.009, 0.009), (0.011, 0.009, 0.009)
    for obj in (
        gprMax.Domain(p1=(0.018,) * 3),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.TimeWindow(time=1e-9),
        gprMax.PMLThickness(thickness=2),
        gprMax.OMPThreads(1),
        gprMax.Waveform(wave_type="gaussian", amp=-1, freq=5e9, id="drive"),
        gprMax.Waveform(wave_type="gaussian", amp=0, freq=5e9, id="passive"),
    ):
        scene.add(obj)
    properties = dict(background.kwargs)
    if legacy:
        properties["se"] = 1000 / sources["gssi2000_rx"].resistance
    scene.add(gprMax.Material(**properties))
    scene.add(gprMax.Edge(p1=rx, p2=(rx[0], rx[1] + 0.001, rx[2]), material_id=properties["id"]))
    scene.add(gprMax.VoltageSource(**dict(sources["gssi2000_tx"].kwargs, p1=tx, waveform_id="drive")))
    if not legacy:
        scene.add(gprMax.VoltageSource(**dict(sources["gssi2000_rx"].kwargs, p1=rx, waveform_id="passive")))
    scene.add(gprMax.Rx(p1=rx, id="gssi2000_rxbowtie", outputs=["Ey"]))
    if metal:
        scene.add(gprMax.Plate(p1=(0, 0, 0.006), p2=(0.018, 0.018, 0.006), material_id="pec"))
    elif not legacy:
        scene.add(gprMax.NTFFSurface(p1=(0.004,) * 3, p2=(0.014,) * 3, id="surface"))
        scene.add(gprMax.NTFFFrequencyTransform("surface", "band", frequencies=(5e9,)))
        scene.add(gprMax.NTFFAntennaPorts("band", ("gssi2000_tx", "gssi2000_rx")))
        scene.add(
            gprMax.NTFFFarField(
                theta=90,
                phi=0,
                transform_id="band",
                id="pattern",
                outputs=("radiation_efficiency", "total_efficiency"),
            )
        )
    return scene


@pytest.mark.parametrize("metal", [False, True])
def test_gssi2000_passive_voltage_load_matches_conductive_edge(tmp_path, metal):
    for legacy in (True, False):
        gprMax.run(
            scenes=[_scene(legacy, metal)],
            n=1,
            outputfile=tmp_path / str(legacy),
            cpu_precision="double",
            hide_progress_bars=True,
            log_level=40,
        )
    with h5py.File(tmp_path / "True.h5") as old, h5py.File(tmp_path / "False.h5") as new:
        ey = np.asarray(new["rxs/rx1/Ey"])
        expected = np.asarray(old["rxs/rx1/Ey"])
        assert np.linalg.norm(expected) > 0
        assert np.linalg.norm(ey - expected) / np.linalg.norm(expected) < 1e-12
        assert int(new.attrs["nrx"]) == int(old.attrs["nrx"]) == 1
        port = new["ports/gssi2000_rx"]
        voltage = read_time_history(port["Vtotal"])
        np.testing.assert_allclose(voltage.samples, -0.001 * 0.5 * (ey[:-1] + ey[1:]), rtol=1e-13, atol=0)
        assert voltage.offset == pytest.approx(0.5 * float(new.attrs["dt"]))
        assert port.attrs["GapCapacitance"] == pytest.approx(8.8541878128e-12 * 1.056 * 0.001, rel=1e-8)
        assert port.attrs["BackgroundConductance"] == 0
        assert port.attrs["ReferenceImpedance"] == pytest.approx(200008.0107)
        assert not np.any(port["Vgenerator"][:])
        assert not np.any(port["valid_S11"][:])
        assert not np.any(port["valid_Zin"][:])
        tx = new["ports/gssi2000_tx"]
        valid = tx["valid_S11"][:].astype(bool) & tx["valid_Zin"][:].astype(bool)
        assert np.any(valid)
        assert np.all(np.isfinite(tx["Zin"][:][valid]))
        if not metal:
            ff = new["ntff/surface/frequency/band/far_field/pattern"]
            power = ff["port_power"]
            assert power["port_ids"].asstr()[:].tolist() == ["gssi2000_tx", "gssi2000_rx"]
            assert power["incident_power_per_port"][0, 0] > 0
            assert power["incident_power_per_port"][1, 0] == 0
            assert power["accepted_power_per_port"][1, 0] < 0
            for key in ("radiation_efficiency", "total_efficiency"):
                values = ff[f"fields/{key}"][:]
                assert np.all(np.isfinite(values)) and np.all(values > 0)
