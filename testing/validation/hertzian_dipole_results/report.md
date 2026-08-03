# Hertzian-dipole analytical validation

For a z-directed Hertzian dipole, the far-zone field is proportional
to `sin(theta)`, the radiation intensity to `sin(theta)^2`, and

`D(theta) = 1.5 sin(theta)^2`.

The analytical peak directivity is 1.5 (1.760913 dBi).
For an ideal lossless dipole, gain equals directivity. gprMax does not
report port-normalised gain for `HertzianDipole` because this impressed
current source has no reference impedance or accepted-port power; the
plot therefore labels the analytical lossless-gain identity explicitly.

Both KSIR and conventional equivalent-current transforms are compared
with the closed-form E- and H-plane patterns. The near-field figure
compares one Ez component from a direct Yee receiver and KSIR with the
complete analytical dipole field, including reactive, induction, and
radiation terms.

The source and receiver command coordinates identify the lower ends
of z-directed Yee edges. Their physical Ez sample centres are each
shifted by dz/2, so the analytical relative vector is unchanged. KSIR
uses absolute Cartesian observation coordinates; its point is therefore
placed at the receiver coordinate plus `(0, 0, dz/2)`.

## Results

Overall validation status: **PASS**.

- ksir: peak directivity 1.499842
  (1.760454 dBi); E-plane
  directivity RMS error 9.826e-05.
- equivalent_current: peak directivity 1.500059
  (1.761082 dBi); E-plane
  directivity RMS error 2.441e-05.
- Direct near-field Ez relative L2 error: 0.0003353.
- KSIR near-field Ez relative L2 error: 0.0001667.

## Outputs

- [Far-field patterns and metrics](hertzian_dipole_far_field.png)
- [Near-field waveform](hertzian_dipole_near_field.png)
- `hertzian_dipole_far_field.csv`
- `hertzian_dipole_near_field.csv`
