# Rectangular patch optimisation example

Copy [rectangular_patch.py](rectangular_patch.py) and follow its four numbered
sections: parameters, model, objective and launch. Supporting geometry and
spectrum checks are at the end. The [example guide](../EXAMPLES.rst) contains
commands, the validation comparison and a convergence figure; the
[package guide](../OPTIMISERS.rst) explains how to change optimiser.

## Reference and scope

N. Jin and Y. Rahmat-Samii, “Parallel particle swarm optimization and
finite-difference time-domain (PSO/FDTD) algorithm for multiband and wide-band
patch antenna designs,” *IEEE Transactions on Antennas and Propagation*, 53(11),
3459–3468, 2005. [DOI: 10.1109/TAP.2005.858842](https://doi.org/10.1109/TAP.2005.858842).

This example adapts the single-frequency rectangular patch in Section III and
Table I, Antenna I. The user model uses a magnetic frill as an equivalent coaxial
feed. It is not an exact reproduction of the paper's feed or PSO implementation.
Antenna II is retained as a geometry reference, but its broadband objective is
not used in this example.

## What changes and what stays fixed

| Setting | Value |
| --- | --- |
| Length L | 26–36 mm, step 2 mm |
| Width W | 12–44 mm, step 2 mm |
| Feed offset x | 1–12 mm, step 1 mm, measured from the patch centre |
| Board and PEC ground | 60 × 60 mm |
| Substrate | 3 mm thick, relative permittivity 2.2, lossless |
| Patch and ground metal | Zero-thickness PEC plates |
| Mesh | 1 × 1 × 1 mm |
| Time window | 100 ns, approximately 10 MHz independent frequency resolution |
| Target | S11 at 3.1 GHz |
| Probe | 3 mm vertical thin wire, radius 0.2 mm |
| Equivalent coax | 50 ohms, assumed air-filled |
| Domain | 100 × 100 × 43 mm; 10 mm PML inside the 20 mm padding |

`Integer(..., "mm", step=2)` declares the even physical lengths through the public
API. Every general optimiser adapter uses these allowed values. The patch
centre and feed centreline then lie on the 1 mm design lattice. A one-off
`simulate()` can still check an odd integer dimension; helpers record the
requested and realised feed offset when rounding is necessary.

The paper states L,W in (0,44) mm and x in (0,22) mm with x < L/2. Our bounds
restrict that space and guarantee the feed remains at least 1 mm inside the
patch. Positive offset points toward the model's negative x direction. It is
**not an inset from the left edge**.

The builder places dielectric first, then metal. The thin wire spans from the
ground surface to the patch underside, with no voltage-source gap. The magnetic
frill represents the small insulating coaxial aperture and loading at its bottom
endpoint; the grid PEC ground plate remains uncut. A bare wire without that
source would be a shorting pin. Probe radius and coax filler are assumptions
because the paper does not specify those feed dimensions. The assumed 50-ohm
air coax corresponds to an outer radius of approximately 0.460 mm; no resolved
coaxial cylinder is drawn.

## Criterion and stopping

`evaluate()` reads `output.port("frills/frill1")`, checks the spectrum and returns:

```python
reflection = spectrum.at(TARGET_FREQUENCY_HZ)  # Complex S11, interpolated at 3.1 GHz.
score_db = 20 * np.log10(max(abs(reflection), 1e-12))
```

The optimiser minimises that scalar: -30 dB is better than -20 dB. The paper's
single-frequency fitness adds 50, a constant that does not change the minimiser.
Minimising the dip frequency error or a broadband mismatch would be a different
objective. The numerical floor only avoids log(0).

`run_example()` checks `STARTING_DESIGN` once, then searches. This design is not
seeded into the optimiser. The default is PSO with ten particles, up to 40
proposals, and an acceptable score of -30 dB. Forty proposals is a short trial,
not a convergence claim. Use a new directory on every launch. For a longer run,
pass `evaluations=2000`; pass `target_value=None` to use only the budget. The
current population finishes before target-based stopping.

Each scored candidate saves `s11.npz` with `frequency_hz`, complex `s11`, `valid`,
`target_hz`, complex `s11_at_target`, `score_db`, `requested_feed_offset_mm` and
`realised_feed_offset_mm`. These are example-specific arrays, not mandatory
framework output fields. The [reader guide](../OUTPUT_READERS.rst) explains the
underlying terminal record, masks and units.

## Validation recorded on 9–10 September 2026

Both PSO and DE independently proposed L=30, W=18, x=4 mm, the paper's Antenna I
dimensions. This model gives S11=-31.537468 dB at 3.1 GHz at that geometry; the
paper reports approximately -28 dB. A fresh PSO simulation agreed with the
independent reference; DE later reused the same compatible output after
proposing the geometry. The reference was not seeded.

With seed 7 and ten candidates per population, PSO first reached the reference
at population 31/proposal 304 and DE at population 47/proposal 467. PSO was stopped
by the user after 55 populations (550 proposals, 68 new simulations); DE stopped
after 47 (470 proposals, 100 new simulations). The remainder reused compatible
outputs. PSO used `restart_after=20`, with its first restart after population 51;
DE used `DE/rand/1/bin`. This is one run per method, with different proposal
histories and cache reuse, not a performance ranking.

The historical runs used local stepped wrappers. The public `Integer(step=...)`
interface now expresses the same design lattice. Use fresh campaigns with the
updated source; do not rewrite historical source files to resume an old
checkpoint. The example does not ship the experiments' private caches.

The agreement in dimensions validates the optimisation mechanics for this
adapted model. Spectrum differences, equivalent-feed assumptions and mesh
sensitivity remain: these runs do not establish mesh convergence or exact
reproduction of the paper's electromagnetic solution.
