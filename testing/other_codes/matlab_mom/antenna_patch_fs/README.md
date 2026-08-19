# Rectangular patch: gprMax FDTD/KSIR versus MATLAB MoM

This case compares the gprMax KSIR far-field result with the Method-of-Moments
solution from MATLAB Antenna Toolbox. It reproduces the low-permittivity thin
substrate case in the official MathWorks
[Patch Antenna with Dielectric Substrate](https://www.mathworks.com/help/antenna/ug/patch-antenna-on-dielectric-substrate.html)
example at its reported 2.37 GHz resonance.

The shared physical model is a 40 mm by 30 mm PEC patch over an 80 mm by 60 mm
PEC ground plane, on a 1.57 mm lossless substrate with relative permittivity
2.33. The 1 mm square probe is offset by 5.5 mm in x. MATLAB uses `pcbStack`
and its MoM solver. gprMax uses a 0.5 mm by 0.5 mm in-plane grid, exactly three
cells through the substrate, a 50 Ohm voltage-source gap at the bottom of the
probe, PML boundaries, and a closed KSIR surface in free space. Four gprMax
feed variants are provided: the default drives all nine z-directed Yee edges
with equal 450 Ohm sources in parallel; `--feed single` drives only the central
edge with one 50 Ohm source; and `--feed series` replaces the PEC post with
three stacked sources spanning the full substrate thickness. Each series
element uses one third of the generator voltage and 50/3 Ohms, retaining a
1 V, 50 Ohm modal Thevenin source. `--feed frill` instead removes both the
voltage source and PEC post, joins the ground plane to the patch with a 0.23 mm
radius improved thin wire, and drives it with the Hyun 50 Ohm equivalent
coaxial magnetic frill at the ground-plane aperture.

The MATLAB model uses a 1 mm square via. The improved thin-wire formulation is
a subcell model and requires its radius to be less than half the transverse
cell size; the 0.23 mm frill-probe radius is therefore the largest practical
choice on the 0.5 mm standard mesh. This is a deliberate feed-model comparison,
not an assertion that the local MATLAB delta gap and gprMax coax aperture have
identical geometry. The fixed physical radius also means this frill case is
not available on the fully refined 0.25 mm x-y mesh without selecting a
smaller physical probe.

The zero-thickness PEC `Plate` representation is the default and corresponds
most closely to the surface conductors in the MATLAB MoM model. The diagnostic
`--conductor box` option instead gives the patch and ground one-cell PEC
thickness. The ground extends below its original top surface and the patch
extends above its original bottom surface, so the substrate thickness and feed
length do not change.

The diagnostic `--patch-trim-cells N` option removes `N` x-directed mesh cells
from each end of the patch while retaining its centre and feed offset. Thus,
on the standard 0.5 mm grid, `N=1` and `N=2` produce 39 mm and 38 mm patch
lengths. The default remains the physically specified 40 mm patch.

Similarly, `--board-trim-cells N` removes `N` cells from every lateral edge of
both the dielectric and ground plane while keeping their footprints identical
and centred. This isolates finite-board edge effects without changing the
patch, feed, or substrate thickness.

Two convergence meshes are available. `--mesh fine-z` doubles the vertical
resolution from three to six substrate cells while retaining the 0.5 mm
in-plane grid. `--mesh fine-xyz` also halves the x-y cell size to 0.25 mm.
PML cell counts are increased on each refined axis to preserve their physical
thickness.
The 60 ns time window gives approximately 16.7 MHz independent spectral
resolution and allows the patch response to decay before the frequency
transform is finalised. The S11 post-processor applies eight-times zero
padding, sampling the engineering-convention transform every approximately
2.08 MHz around the narrow resonance. This interpolates the transform without
claiming resolution beyond that provided by the 60 ns record.

All feeds provide a 50 Ohm one-port S11 comparison without using a one-cell
transmission-line source. Named receivers store `Ez` on the driven edge or
edges. For the distributed feed, the post-processor averages their voltages to
obtain the total uniform-port voltage. A resistive gprMax voltage source is a
Thevenin generator:
the waveform is the open-circuit generator voltage, so its launched incident
voltage wave is one half of that waveform. Consecutive electric-field samples
are averaged onto the source's half time step before the reflected wave is
formed as `Vref = Vtotal - Vinc`. The complex spectra use the engineering FFT
convention. Frequencies more than 60 dB below the peak incident spectrum are
excluded. The CSV also reports the RMS variation of the nine complex edge
voltages relative to their common-mode average, making the distributed-port
assumption directly auditable.

The series-source option is a numerical comparison rather than the preferred
physical probe model. A real probe remains conductive above its localised feed
gap, represented by the single- and distributed-feed variants. Replacing that
PEC post with impressed sources distributes both electromotive force and source
resistance through the substrate and therefore describes a different port.

The comparison contains full signed cuts through both principal planes:

- E-plane (x-z): gprMax co-polar component is `Etheta`.
- H-plane (y-z): gprMax co-polar component is `Ephi`.

Each solver is normalised using one global co-polar peak across both cuts. This
tests pattern shape, not absolute realised gain. The methods differ in mesh,
feed implementation, open-boundary treatment, and time/frequency solution, so
small differences are expected.

The feed is deliberately offset in x, as in the MATLAB model, so the raw
E-plane result is not required to be perfectly mirror-symmetric. No numerical
mirror averaging is applied. The upper-hemisphere comparison and the reported
symmetry metrics distinguish this physical effect from the visually amplified
differences in the weak back-radiation/null region.

## Run

From the repository root:

```bash
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --gpu 0
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed single --gpu 0
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed series --gpu 0
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed frill --gpu 0
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed single --conductor box --gpu 0
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed single --patch-trim-cells 1 --gpu 0
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed single --patch-trim-cells 2 --gpu 0
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed single --board-trim-cells 1 --gpu 0
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed single --board-trim-cells 2 --gpu 0
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed single --mesh fine-z --gpu 0
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed single --mesh fine-xyz --gpu 0
MPLCONFIGDIR=/tmp/matplotlib-gprmax conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/check_patch_far_field_formulations.py --gpu 0
MPLCONFIGDIR=/tmp/matplotlib-gprmax conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_3d_gain.py --gpu 0
matlab -batch "run('testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_metrics_matlab.m')"
MPLCONFIGDIR=/tmp/matplotlib-gprmax conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/plot_patch_metric_comparison.py
matlab -batch "run('testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_matlab.m')"
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/plot_patch_pattern_comparison.py
MPLCONFIGDIR=/tmp/matplotlib-gprmax conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/plot_patch_frill_comparison.py
```

Omit `--gpu 0` to use the Cython CPU solver. To generate the geometry without
running FDTD, add `--geometry-only`. To check and post-process an existing
gprMax output without rerunning FDTD, use:

```bash
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --postprocess-only
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed single --postprocess-only
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed series --postprocess-only
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed frill --postprocess-only
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed single --mesh fine-z --postprocess-only
conda run -n gprMax-devel python testing/other_codes/matlab_mom/antenna_patch_fs/patch_antenna_gprmax.py --feed single --mesh fine-xyz --postprocess-only
```

The gprMax script does not use the in-memory KSIR result shortcut. After the
blocking simulation completes, it reopens `patch_antenna_gprmax.h5`, checks the
frequency and both angular grids, checks every required complex field dataset
and its dimensions, and creates the pattern and complex-S11 CSV files from
those persisted data. MATLAB reads the resulting gprMax frequency grid,
calculates the MoM input impedance, and obtains the reference S11 for 50 Ohms
from `(Zin - 50) / (Zin + 50)`. The final plotting command produces both the
principal-plane pattern plots and `results/patch_s11_comparison.png`. Reported
resonance minima use a three-point parabolic estimate around the minimum on the
zero-padded grid; they should be interpreted in the context of the 16.7 MHz
independent resolution of the time record.

For the magnetic frill, post-processing reads the source's persisted `Vinc`
and `Vtotal` histories and verifies their native-bin S11 against the automatic
`/frills/frill1/S11` dataset. It then applies the same eight-times zero padding
as the voltage-source cases. This gives a smooth plotting grid without
claiming additional independent frequency resolution.

## Current S11 comparison

On the standard mesh, the distributed, single-edge, and series-source
resonance estimates are 2.3593, 2.3606, and 2.3628 GHz, respectively. Their S11
depths are -14.96, -15.53, and -18.54 dB, and their -10 dB bandwidths are 25.30,
25.65, and 26.60 MHz. MATLAB MoM gives -14.02 dB at 2.4367 GHz with a 25.73 MHz
bandwidth; the standard fringing-corrected transmission-line estimate is
2.4356 GHz.

Giving the coarse-grid patch and ground one-cell PEC thickness (0.523 mm)
moves the single-feed resonance down by 7.88 MHz, from 2.3606 to 2.3527 GHz.
It deepens S11 from -15.53 to -17.08 dB and increases the -10 dB bandwidth
from 25.65 to 27.24 MHz. This moves the result farther from MATLAB rather than
accounting for the resonance offset, so `Plate` remains the preferred model.

Shortening the coarse-grid plate from 40 mm to 39 mm raises its single-feed
resonance from 2.3606 to 2.4174 GHz, leaving a 19.25 MHz offset below MATLAB.
Shortening it again to 38 mm raises the resonance to 2.4771 GHz, 40.39 MHz
above MATLAB. The 39 mm case is therefore the closer discrete correction, but
it is retained as a diagnostic because its physical dimensions no longer match
the nominal MATLAB model.

Reducing the common ground-plane and dielectric footprint from 80 by 60 mm to
79 by 59 mm changes the resonance by only -14.7 kHz; reducing it to 78 by
58 mm changes it by -37.5 kHz. The S11 depth becomes -15.80 and -16.09 dB,
respectively, while the -10 dB bandwidth increases from 25.65 to 26.07 and
26.50 MHz. The finite board edges therefore influence coupling and bandwidth
slightly but do not explain the resonance-frequency offset.

Doubling only the vertical resolution moves the single-feed resonance upward
by 2.64 MHz to 2.3633 GHz. Refining all axes moves it upward by 10.15 MHz from
the standard result, to 2.3708 GHz, and gives -14.50 dB depth and 25.03 MHz
bandwidth. Full refinement therefore improves both the resonance direction and
bandwidth agreement, but 65.91 MHz remains between the fully refined FDTD and
MoM resonances. The offset is not solely a coarse-grid artefact; differences in
feed representation and the two electromagnetic models remain candidates.

The radiation pattern is already well converged. In the upper hemisphere, the
fully refined pattern differs from the standard single-feed result by at most
0.111 dB in the E-plane and 0.039 dB in the H-plane. The full model contains
46.08 million cells and 122,799 time steps. On an NVIDIA TITAN RTX, its CUDA
solve took 26 minutes 21 seconds and used approximately 2.4 GB host plus 2.8 GB
device memory.

## Independent far-field formulation check

`check_patch_far_field_formulations.py` runs the standard distributed-feed
model once while accumulating KSIR and conventional Love-equivalent-current
transforms at 2.37 GHz on the same `NTFFSurface`. It then reads both results
from HDF5 and compares them with each other and with the existing MATLAB MoM
principal-plane data. Use `--postprocess-only` to regenerate the CSV, JSON,
and plot without repeating the FDTD run.

The two gprMax formulations are effectively indistinguishable. Over the upper
hemisphere, their normalised co-polar patterns differ by 0.000156 dB RMS in
the E-plane and 0.0000624 dB RMS in the H-plane. Combining the two complex
co-polar cuts gives a raw relative L2 difference of 2.09e-5. The best-fit
equivalent-current/KSIR complex scale has magnitude 1.000016 and phase
0.000606 degrees; after removing this single complex scale, the remaining
shape error is 8.29e-6.

The populated H-plane cross-polar component agrees to 1.32e-4 relative L2.
The E-plane cross-polar result is a numerical symmetry null only 2.16e-7 of
the co-polar norm, so its same-component relative error is not meaningful;
the disagreement is just 2.41e-7 when referenced to the co-polar field. These
checks exercise the Love-current signs, E/H Yee collocation, half-step phasor
placement, propagation phase, and common surface origin.

Against MATLAB, the equivalent-current upper-hemisphere RMS differences are
0.4153 dB in the E-plane and 0.04473 dB in the H-plane, compared with 0.4152
dB and 0.04477 dB for KSIR. Thus the remaining MATLAB discrepancy belongs to
the FDTD/MoM antenna-model comparison rather than the NTFF formulation.

## Full-sphere gain

`patch_antenna_3d_gain.py` uses the standard-mesh single 50 Ohm gap feed and
the native voltage-port/`KSIRAntennaPorts` power normalisation. It evaluates a 2
degree full-sphere `KSIRFarFieldArray` at 2.37 GHz and reads the persisted gain
and directivity datasets back from HDF5. Radiation intensity, realized gain,
radiation efficiency, and total efficiency are retained as well. The script
writes the angular data to CSV, a metrics JSON file, and a 3-D PNG whose radial
coordinate and colour are the gain over a 30 dB display range. Add
`--postprocess-only` to recreate the CSV and plot without rerunning FDTD.

`patch_antenna_metrics_matlab.m` independently obtains the same 16,380 angular
samples from Antenna Toolbox using its directivity, gain, realized-gain, and
linear-power pattern types. It also calculates radiation efficiency, input
impedance, mismatch efficiency, and total efficiency. The Python comparison
checks the defining power identities within each solver and creates
`results/patch_antenna_metric_comparison.png`.

## Magnetic-frill comparison

The standard-mesh CUDA frill model contains 5.76 million cells and 61,400 time
steps. Its solve took 1 minute 54 seconds on an NVIDIA TITAN RTX and used about
623 MB host plus 543 MB device memory. The terminal trace decayed to -130.5 dB
relative to its peak, and independently reconstructing S11 from the persisted
voltage histories reproduced the automatic HDF5 port result.

The magnetic frill moves the resonance estimate only slightly, from 2.3606 GHz
for the single-edge voltage feed to 2.3615 GHz. MATLAB remains at 2.4367 GHz,
so the frill reduces the 76.06 MHz offset by only 0.84 MHz. Its S11 minimum is
-16.72 dB and its -10 dB bandwidth is 26.22 MHz, compared with -15.53 dB and
25.65 MHz for the voltage gap and -14.02 dB and 25.73 MHz for MATLAB.

The radiation pattern is effectively feed-independent: in the upper
hemisphere the frill differs from the voltage-gap result by 0.0098 dB RMS in
the E-plane and 0.000016 dB RMS in the H-plane. The result shows that the
equivalent coaxial feed works correctly for the patch, but the previous
gprMax/MATLAB resonance offset is not principally caused by using a simple
voltage-gap excitation.

## ParaView geometry

Open `results/patch_antenna_geometry.vtkhdf` directly in ParaView. It is a
fine `UnstructuredGrid` view of the actual FDTD geometry, with one VTK line
cell for each Yee edge. Colour by the `Material` cell array and use the
metadata material table to identify free space, substrate, PEC, and the
voltage-source material. Increasing ParaView's line width makes the thin PEC
layers easier to inspect. The fully refined equivalent is
`results/patch_antenna_geometry_single_feed_fine_xyz.vtkhdf`.
