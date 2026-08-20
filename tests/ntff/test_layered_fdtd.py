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
        "#ntff_far_field: 30 0 inner_f u0 Ex Ey Ez radiation_intensity directivity\n"
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
        outputs=("Etheta", "Ephi"),
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
