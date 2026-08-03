# PEC-plate radar-cross-section comparison

This testing comparison reproduces the square- and circular-PEC-plate cases
from the [MathWorks Radar Cross Section Benchmarking
example][mathworks-benchmark]. It compares three independent calculations:

1. gprMax FDTD with a discrete plane wave and a KSIR frequency-domain
   near-to-far-field transformation;
2. MATLAB Antenna Toolbox physical optics (PO) and Method of Moments (MoM);
   and
3. the closed-form physical-optics plate expressions used by MathWorks.

The benchmark wavelength is 32.5 mm, corresponding to approximately
9.224 GHz. The square is 101.6 mm by 101.6 mm and the circular plate has a
radius of 101.6 mm. Both are zero-thickness PEC plates in free space. The wave
is HH-polarised and its propagation vector lies in the x-z plane.

## Angle and polarisation mapping

MathWorks specifies an azimuth of zero and sweeps elevation from the plane of
the plate towards its normal. The gprMax plate lies in the x-y plane. For an
elevation angle alpha, the incident propagation direction and electric-field
polarisation are

```text
k_inc = (cos(alpha), 0, sin(alpha)),    E_inc || y.
```

The monostatic observation direction is exactly `-k_inc`. The gprMax input
therefore uses spherical output angles `theta = 90 deg + alpha` and
`phi = 180 deg`. A `DiscretePlaneWaveVector` requires an integer direction;
the driver records both the requested angle and the exact angle produced by
the reduced integer vector. MATLAB and the analytical formula are evaluated
at this exact angle, so direction quantisation is not counted as an FDTD
error.

Each incidence angle requires a separate FDTD model. Asking one model for many
KSIR observation angles would produce a *bistatic* pattern, not the monostatic
sweep in the MathWorks example. The driver uses the CPU by default. Add
`--gpu 0` to run the production CUDA plane-wave and NTFF paths on device 0.

## Reference formulae

With wavenumber `k = 2*pi/lambda`, plate area `A`, and elevation alpha measured
from the plate plane, the square-plate PO result is

```text
sigma_square = 4*pi*A^2/lambda^2 * sin(alpha)^2
               * sinc(k*L*cos(alpha))^2,
```

where `sinc(x) = sin(x)/x` and `L` is the dimension in the incidence plane.
For a circular plate of radius `a`,

```text
sigma_circle = 4*pi*A^2/lambda^2 * sin(alpha)^2
               * [2*J1(2*k*a*cos(alpha))/(2*k*a*cos(alpha))]^2.
```

These PO expressions intentionally omit edge and shadow-region currents.
MATLAB MoM is consequently the more appropriate full-wave comparison for
gprMax, while the analytical curves provide useful broadside scaling and null
checks. Neither numerical solver is treated as ground truth.

## Running the comparison

Run a small set of representative angles first:

```bash
conda run -n gprMax-devel python \
  testing/other_codes/matlab_mom/rcs_comparison/plate_rcs_gprmax.py \
  --target square --mesh coarse --gpu 0

conda run -n gprMax-devel python \
  testing/other_codes/matlab_mom/rcs_comparison/plate_rcs_gprmax.py \
  --target circle --mesh coarse --gpu 0
```

The sweeps are resumable: an existing HDF5 file is reopened and checked before
its RCS value is reused. Use `--force` to replace existing runs. To reproduce
the original requested grid of `0.05:1:89.05` degrees, add
`--mathworks-sweep`. This is approximately 180 independent simulations for
the two targets, so it is substantially more expensive than the selected
eight-angle check.

Run the independent MATLAB calculations and create plots with:

```bash
matlab -batch "run('testing/other_codes/matlab_mom/rcs_comparison/plate_rcs_matlab.m')"

conda run -n gprMax-devel python \
  testing/other_codes/matlab_mom/rcs_comparison/plot_plate_rcs_comparison.py
```

The MATLAB script constructs local STL surfaces, calculates PO and MoM values
at the exact gprMax angles, and saves both convenient CSV tables and
`plate_rcs_matlab.mat`. The MAT file contains the complete numerical reference
curves, not only model settings. It is retained so users without MATLAB or the
Antenna Toolbox can run the Python plotting script. The plotter reads the CSV
tables when present and falls back to the MAT file otherwise. It writes the
comparison figures and `plate_rcs_comparison_metrics.json`.

The per-angle gprMax HDF5 files are resumable local caches. They can be
deleted after the compact CSV results have been generated and should not be
committed.

The two supplied spatial resolutions are:

| Mesh | Cell size | Cells per wavelength | Square cells per side |
|---|---:|---:|---:|
| `coarse` | 2.54 mm | 12.8 | 40 |
| `fine` | 1.27 mm | 25.6 | 80 |

RCS minima are intrinsically demanding comparison points: a small geometric,
angular, or phase error can move a null and cause a large dB difference even
when the surrounding full-wave curves agree. Interpret the direct dB metrics
together with the plotted curve and the coarse-to-fine change.

## Current selected-angle result

The included local result set was generated with 20 CPU threads. For the
square plate, refinement reduces the eight-angle gprMax-to-MATLAB-MoM RMS
difference from 1.36 dB to 0.70 dB and the maximum difference from 3.01 dB to
1.23 dB. At 89.10 degrees, gprMax is 0.23 dB above MoM and 0.02 dB below the
analytical PO value.

For the more staircasing-sensitive circular plate, three fine samples were run
at 9.90, 30.19, and 89.10 degrees. Across those same samples, refinement
reduces the RMS difference from 7.32 dB to 1.86 dB and the maximum difference
from 9.64 dB to 2.70 dB. The fine broadside sample is within 0.10 dB of MoM
and 0.02 dB of analytical PO. The large low-angle change is expected because
refining the circular raster reduces edge staircasing and shifts its deep RCS
minima. A publication-quality angular curve should use `--mathworks-sweep` at
the fine resolution rather than interpolating these three convergence samples.

[mathworks-benchmark]: https://www.mathworks.com/help/antenna/ug/radar-cross-section-benchmarking.html
