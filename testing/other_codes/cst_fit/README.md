# SAB-derived gprMax patch model

Here, `cst_fit` denotes CST's time-domain finite-integration-technique (FIT)
solver; it does not mean numerical curve fitting.

`patch_antenna.in` reconstructs the five physical solids in `patch.sab`: the
substrate, ground, matching line, feed line, and rectangular patch. ACIS body
bounding boxes were used as the source dimensions.

The model uses a lossless `er=4.4` substrate and PEC conductors. The numerical
metal is one 0.4 mm cell thick. The uniform microstrip feed, substrate, and
ground continue through the positive-y PML. A 34 mm by 18 mm eigenmode port
captures the quasi-TEM mode and its evanescent field.

Run from the repository root:

```powershell
conda run -n gprMax python -m gprMax testing/other_codes/cst_fit/patch_antenna.in -outputfile testing/other_codes/cst_fit/S-parameter/patch_antenna
conda run -n gprMax python -m gprMax testing/other_codes/cst_fit/patch_antenna_farfield.in -outputfile testing/other_codes/cst_fit/Farfield/patch_antenna_farfield
conda run -n gprMax python testing/other_codes/cst_fit/plot_patch_results.py
```

The first command sweeps S11 from 1.6 to 3.2 GHz in 5 MHz steps. The second
stores a full-sphere far field at 2.45 GHz with 1-degree theta/phi sampling.
The final command exports the far field to CSV, plots S11 and principal
far-field cuts, overlays the gprMax and `patch_cst.s1p` S11 results, and writes
a JSON result summary with both resonance minima and their differences. If
`patch_ff_cst.txt` is present, its 2.45 GHz directivity cuts and peak are also
overlaid and compared with the gprMax far field. The XZ and YZ plots are full
360-degree cuts assembled from phi=0/180 degrees and phi=90/270 degrees.

Retained comparison artifacts are organised under `S-parameter/` and
`Farfield/`. Large field data, geometry views, dense CSV/TXT exports, and
regenerable eigenmode diagnostics are ignored by Git. The comparison PNGs and
the combined `patch_results_summary.json` are retained.
