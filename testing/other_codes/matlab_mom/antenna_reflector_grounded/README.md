# Reflector-backed strip dipole: gprMax FDTD versus MATLAB MoM

This benchmark compares a balanced 150 x 15 mm PEC strip dipole at two heights
above an analytically infinite PEC ground plane. At 1 GHz the 37.5 and 75 mm
air gaps are approximately one eighth and one quarter of a free-space
wavelength, respectively.

MATLAB Antenna Toolbox represents the ground using its infinite-ground-plane
technique. In gprMax, a five-face layered equivalent-current NTFF surface
omits the face coincident with the PEC termination. These cases satisfy the
terminal-background requirement because the source and strip lie wholly in
air and no finite material object intersects the omitted face.

A finite dielectric block placed directly on the omitted PEC face is not a
valid variant of this benchmark: beneath the block, that face is no longer
locally described by the declared air/PEC terminal background. Such a coating
must either be represented as a layer in the background stack or be enclosed
by a mathematically valid transformation surface.

The MATLAB dipole is a continuous zero-thickness strip with a delta feed.
gprMax uses two zero-thickness PEC plates separated by one 1.5 mm electric
edge, occupied by a 50 Ohm voltage source and its automatic port monitor.
Consequently, normalised radiation patterns and directivity are the primary
comparison. Impedance and S11 additionally include the different feed-gap
discretisation and are retained as a diagnostic rather than asserted as a
like-for-like port validation.

Run the benchmark with:

```console
python testing/other_codes/matlab_mom/antenna_reflector_grounded/reflector_dipole_gprmax.py --gpu 0
matlab -batch "run('testing/other_codes/matlab_mom/antenna_reflector_grounded/reflector_dipole_matlab.m')"
python testing/other_codes/matlab_mom/antenna_reflector_grounded/plot_reflector_dipole_comparison.py
```

The retained CSV and JSON files contain 1 GHz complex principal-plane fields,
absolute directivity and efficiency metrics, and 0.5--1.5 GHz impedance/S11.
Raw HDF5 results are reproducible simulation artefacts and are ignored by the
validation-data policy.
