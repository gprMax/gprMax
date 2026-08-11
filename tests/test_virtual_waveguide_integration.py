import h5py
import numpy as np
import pytest

import gprMax


def _uniform_waveguide_scene(normal_axis, direction):
    dl = 1e-3
    cells = [12, 10, 12]
    cells[normal_axis] = 60
    domain = tuple(value * dl for value in cells)

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=domain))
    pml = [0] * 6
    pml[normal_axis + (0 if direction == "-" else 3)] = 5
    scene.add(gprMax.PMLThickness(thickness=tuple(pml)))
    scene.add(gprMax.TimeWindow(time=1e-9))
    scene.add(gprMax.Waveform(wave_type="contsine", amp=1, freq=22e9, id="wave"))

    transverse_axes = [axis for axis in range(3) if axis != normal_axis]
    for axis in transverse_axes:
        lower_stop = list(domain)
        lower_stop[axis] = dl
        scene.add(gprMax.Box(p1=(0, 0, 0), p2=tuple(lower_stop), material_id="pec"))
        upper_start = [0.0, 0.0, 0.0]
        upper_start[axis] = domain[axis] - dl
        scene.add(gprMax.Box(p1=tuple(upper_start), p2=domain, material_id="pec"))

    plane = 0.02 if direction == "+" else 0.04
    p1 = [dl, dl, dl]
    p2 = [domain[axis] - dl for axis in range(3)]
    p1[normal_axis] = plane
    p2[normal_axis] = plane
    scene.add(gprMax.EigenmodeBand(id="band", fmin=22e9, fmax=22e9, points=1))
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=tuple(p1),
            p2=tuple(p2),
            direction=direction,
            modes=(1,),
            anchors=(22e9,),
            plot_fields=False,
        )
    )
    scene.add(
        gprMax.VirtualWaveguide(
            port=1,
            length_cells=18,
            pml_cells=8,
            source_clearance_cells=4,
        )
    )
    scene.add(
        gprMax.EigenmodeExcitation(
            port=1,
            mode=1,
            waveform="wave",
            plot_waveform=False,
        )
    )
    return scene


@pytest.mark.integration
def test_virtual_waveguide_full_solve_is_rotation_and_direction_invariant(tmp_path):
    reflection = {}
    for normal_axis in range(3):
        for direction in ("-", "+"):
            outputfile = tmp_path / f"virtual_{'xyz'[normal_axis]}{direction}"
            gprMax.run(
                scenes=[_uniform_waveguide_scene(normal_axis, direction)],
                n=1,
                outputfile=outputfile,
                hide_progress_bars=True,
                log_level=30,
            )
            with h5py.File(outputfile.with_suffix(".h5"), "r") as output:
                port = output["eigenmode_ports/port1"]
                assert np.all(np.isfinite(port["incident"][...]))
                assert np.all(np.isfinite(port["outgoing"][...]))
                value = float(np.abs(port["S"][0, 0]))
                assert value < 0.02
                reflection[normal_axis, direction] = value

    for normal_axis in range(3):
        assert reflection[normal_axis, "-"] == pytest.approx(
            reflection[normal_axis, "+"], abs=1e-4
        )
