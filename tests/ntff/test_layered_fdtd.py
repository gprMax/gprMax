import h5py
import numpy as np

import gprMax


def _far_vectors(transform_group, output_ids):
    vectors = []
    for output_id in output_ids:
        fields = transform_group[f"far_field/{output_id}/fields"]
        vectors.append(np.asarray([fields[name][0, 0] for name in ("Ex", "Ey", "Ez")]))
    return np.asarray(vectors)


def test_nested_surfaces_are_equivalent_across_a_dielectric_interface(tmp_path):
    """The final layered field must not depend on the chosen Huygens box."""

    inputfile = tmp_path / "nested_layered.in"
    inputfile.write_text(
        "#domain: 0.12 0.12 0.12\n"
        "#dx_dy_dz: 0.004 0.004 0.004\n"
        "#time_window: 1e-9\n"
        "#pml_cells: 3\n"
        "#material: 4 0 1 0 dielectric\n"
        "#box: 0 0 0 0.12 0.12 0.06 dielectric\n"
        "#waveform: ricker 1 3e9 pulse\n"
        "#hertzian_dipole: x 0.06 0.06 0.076 pulse\n"
        "#ntff_surface: 0.040 0.040 0.040 0.080 0.080 0.080 inner\n"
        "#ntff_surface: 0.032 0.032 0.032 0.088 0.088 0.088 outer\n"
        "#ntff_layered_background: halfspace z free_space 0.06 dielectric\n"
        "#ntff_layered_frequency: inner inner_f halfspace 3e9 rectangular\n"
        "#ntff_layered_frequency: outer outer_f halfspace 3e9 rectangular\n"
        "#ntff_far_field: 30 0 inner_f u0 Ex Ey Ez radiation_intensity directivity "
        "exterior_power exterior_maximum\n"
        "#ntff_far_field: 30 90 inner_f u90 Ex Ey Ez\n"
        "#ntff_far_field: 150 0 inner_f l0 Ex Ey Ez\n"
        "#ntff_far_field: 150 90 inner_f l90 Ex Ey Ez\n"
        "#ntff_far_field: 30 0 outer_f u0 Ex Ey Ez\n"
        "#ntff_far_field: 30 90 outer_f u90 Ex Ey Ez\n"
        "#ntff_far_field: 150 0 outer_f l0 Ex Ey Ez\n"
        "#ntff_far_field: 150 90 outer_f l90 Ex Ey Ez\n"
    )
    outputfile = tmp_path / "nested_layered"
    gprMax.run(
        inputfile=str(inputfile),
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    with h5py.File(str(outputfile) + ".h5", "r") as output:
        ids = ("u0", "u90", "l0", "l90")
        inner = _far_vectors(output["ntff/inner/frequency/inner_f"], ids)
        outer = _far_vectors(output["ntff/outer/frequency/outer_f"], ids)
        # One sampled direction is close to a physical pattern null, where a
        # pointwise relative error is ill conditioned. Normalise the complex
        # vector discrepancy by the maximum sampled field, as in the
        # published layered-NTFF RMS comparisons.
        relative = np.linalg.norm(inner - outer, axis=1) / np.max(np.linalg.norm(outer, axis=1))
        assert np.max(relative) < 0.01
        inner_u0 = output["ntff/inner/frequency/inner_f/far_field/u0"]
        assert np.isfinite(inner_u0["fields/radiation_intensity"][...]).all()
        assert np.isfinite(inner_u0["fields/directivity"][...]).all()
        assert np.isfinite(inner_u0["radiated_power"][...]).all()
        exterior = inner_u0["exterior_regions"]
        positive = exterior["positive_axis"]
        negative = exterior["negative_axis"]
        assert exterior.attrs["stack_axis"] == "z"
        assert positive.attrs["material_id"] == "free_space"
        assert negative.attrs["material_id"] == "dielectric"
        np.testing.assert_allclose(
            positive["radiated_power"][...] + negative["radiated_power"][...],
            inner_u0["radiated_power"][...],
            rtol=2e-15,
        )
        np.testing.assert_allclose(
            positive["radiated_fraction"][...] + negative["radiated_fraction"][...],
            1,
            rtol=2e-15,
        )
        for region in (positive, negative):
            expected = (
                4
                * np.pi
                * region["maximum_radiation_intensity"][...]
                / inner_u0["radiated_power"][...]
            )
            np.testing.assert_allclose(region["maximum_directivity"][...], expected)
            assert np.isfinite(region["maximum_theta"][...]).all()
            assert np.isfinite(region["maximum_phi"][...]).all()


def test_layered_exterior_efficiency_uses_antenna_port_power(tmp_path):
    """Regional coupling efficiencies use the existing antenna-port API."""

    inputfile = tmp_path / "layered_antenna_metrics.in"
    inputfile.write_text(
        "#domain: 0.10 0.10 0.10\n"
        "#dx_dy_dz: 0.004 0.004 0.004\n"
        "#time_window: 8e-10\n"
        "#pml_cells: 3\n"
        "#material: 4 0 1 0 dielectric\n"
        "#box: 0 0 0 0.10 0.10 0.05 dielectric\n"
        "#waveform: ricker 1 3e9 pulse\n"
        "#voltage_source: z 0.05 0.05 0.07 50 pulse 0 8e-10 feed 10\n"
        "#ntff_surface: 0.032 0.032 0.036 0.068 0.068 0.080 surface\n"
        "#ntff_layered_background: halfspace z free_space 0.05 dielectric\n"
        "#ntff_layered_frequency: surface band halfspace 3e9 rectangular\n"
        "#ntff_antenna_ports: band feed\n"
        "#ntff_far_field: 30 0 band pattern Etheta Ephi radiation_efficiency "
        "total_efficiency exterior_power exterior_efficiency exterior_maximum\n"
        "#ntff_far_field: 150 0 band efficiency_only exterior_efficiency\n"
    )
    outputfile = tmp_path / "layered_antenna_metrics"
    gprMax.run(
        inputfile=str(inputfile),
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    with h5py.File(str(outputfile) + ".h5", "r") as output:
        group = output["ntff/surface/frequency/band/far_field/pattern"]
        port_power = group["port_power"]
        exterior = group["exterior_regions"]
        accepted_sum = np.zeros(1)
        realized_sum = np.zeros(1)
        for region_name in ("positive_axis", "negative_axis"):
            region = exterior[region_name]
            np.testing.assert_allclose(
                region["accepted_coupling_efficiency"][...],
                region["radiated_power"][...] / port_power["accepted_power"][...],
            )
            np.testing.assert_allclose(
                region["realized_coupling_efficiency"][...],
                region["radiated_power"][...] / port_power["incident_power"][...],
            )
            np.testing.assert_allclose(
                region["maximum_gain"][...],
                4
                * np.pi
                * region["maximum_radiation_intensity"][...]
                / port_power["accepted_power"][...],
            )
            np.testing.assert_allclose(
                region["maximum_realized_gain"][...],
                4
                * np.pi
                * region["maximum_radiation_intensity"][...]
                / port_power["incident_power"][...],
            )
            accepted_sum += region["accepted_coupling_efficiency"][...]
            realized_sum += region["realized_coupling_efficiency"][...]
        np.testing.assert_allclose(
            accepted_sum,
            group["fields/radiation_efficiency"][...],
        )
        np.testing.assert_allclose(
            realized_sum,
            group["fields/total_efficiency"][...],
        )
        efficiency_only = output[
            "ntff/surface/frequency/band/far_field/efficiency_only/exterior_regions"
        ]
        assert "radiated_power" not in efficiency_only["positive_axis"]
        assert "maximum_directivity" not in efficiency_only["positive_axis"]
        assert "accepted_coupling_efficiency" in efficiency_only["positive_axis"]


def test_layered_transform_encloses_a_subgrid_source(tmp_path):
    """A main-grid layered surface includes fields exchanged from an HSG grid."""

    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.09, 0.09, 0.09)))
    scene.add(gprMax.Discretisation(p1=(0.003, 0.003, 0.003)))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(1))
    subgrid = gprMax.SubGridHSG(
        p1=(0.03, 0.03, 0.03),
        p2=(0.06, 0.06, 0.06),
        ratio=3,
        id="fine_grid",
    )
    scene.add(subgrid)
    subgrid.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    subgrid.add(
        gprMax.HertzianDipole(
            p1=(0.045, 0.045, 0.045),
            polarisation="z",
            waveform_id="pulse",
        )
    )
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.015, 0.015, 0.015),
            p2=(0.075, 0.075, 0.075),
            id="surface",
        )
    )
    scene.add(
        gprMax.NTFFLayeredBackground(
            id="homogeneous",
            axis="z",
            materials=("free_space",),
        )
    )
    scene.add(
        gprMax.NTFFLayeredFrequencyTransform(
            surface_id="surface",
            id="band",
            background_id="homogeneous",
            frequencies=(5e9,),
        )
    )
    pattern = gprMax.NTFFFarField(
        theta=(45,),
        phi=(0,),
        transform_id="band",
        id="pattern",
        outputs=("Etheta", "Ephi", "exterior_power"),
    )
    scene.add(pattern)

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=tmp_path / "layered_subgrid",
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    fields = np.stack((pattern.result.fields["Etheta"], pattern.result.fields["Ephi"]))
    assert np.all(np.isfinite(fields))
    assert np.max(np.abs(fields)) > 0
    np.testing.assert_allclose(
        pattern.result.radiation_metrics.exterior.radiated_fraction[:, 0],
        (0.5, 0.5),
        # This deliberately short, coarse HSG smoke model retains about one
        # percent hemispheric imbalance from the grid coupling and truncated
        # pulse, while still resolving the analytical 50/50 symmetry.
        atol=0.006,
    )
