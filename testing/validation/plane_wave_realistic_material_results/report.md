# Water and Puerto Rico clay DPW half-space validation

The two materials use separate model scales because their Debye relaxation
frequencies differ by more than two orders of magnitude. In both cases,
the free-space reference is subtracted at the same receiver, and the
reflection coefficient is de-embedded with the axial Yee wavenumber to
the half-cell-shifted reflection plane of the non-averagable material.

The engineering-convention material response is

$$\epsilon_r(\omega)=\epsilon_{r,\infty}+\frac{\sigma}{j\omega\epsilon_0}+\sum_p\frac{\Delta\epsilon_{r,p}}{1+j\omega\tau_p}.$$

## Material definitions and frequency bands

### Puerto Rico clay loam, 10% moisture

- $\epsilon_{r,\infty}=5.706$
- $\Delta\epsilon_r=(2.219, 0.958)$
- $\tau=(3.1e-09, 1.1e-10)$ s
- $\sigma=0.003022$ S/m
- relaxation frequencies: 0.05134, 1.4469 GHz
- validation band: 0.02--2.2 GHz
- [parameter source](https://doi.org/10.2528/PIER04061002)

### Fresh water, 25 degC

- $\epsilon_{r,\infty}=4.9$
- $\Delta\epsilon_r=(73.33890625000001,)$
- $\tau=(8.099395053946554e-12,)$ s
- $\sigma=0$ S/m
- relaxation frequencies: 19.65 GHz
- validation band: 1--80 GHz
- [parameter source](https://doi.org/10.1109/JOE.1977.1145319)

## Results

Overall validation status: **PASS**.

| Material | Magnitude RMSE | Maximum magnitude error | Phase RMSE | Maximum phase error |
|---|---:|---:|---:|---:|
| Puerto Rico clay loam, 10% moisture | 6.277e-05 | 0.0001372 | 0.0004768 deg | 0.004621 deg |
| Fresh water, 25 degC | 0.002306 | 0.004447 | 0.1127 deg | 0.2317 deg |

## Outputs

- [Reflection magnitude and phase](reflection_comparison.png)
- [Magnitude and phase residuals](reflection_error.png)
- [Material permittivity](material_permittivity.png)
- [Time-domain fields](time_domain_fields.png)

Each material also has a CSV file containing the complex comparison data.
Simulation HDF5 files are reusable cache data and are not retained as
validation evidence.
