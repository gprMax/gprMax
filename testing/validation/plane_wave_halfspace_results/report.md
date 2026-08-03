# Axial plane-wave dispersive half-space validation

A free-space receiver trace was used as the incident field. For each
half-space, the reflected field was calculated as `total - incident`,
transformed with the engineering FFT convention, and de-embedded to
the appropriate discrete Yee reflection plane. Axial free-space
propagation was removed using the Yee numerical wavenumber rather
than the continuous-space wavenumber.

The analytical comparison is the normal-incidence electric-field Fresnel coefficient

$$\Gamma = \frac{\eta_2-\eta_0}{\eta_2+\eta_0}, \qquad \eta_2=\eta_0/\sqrt{\epsilon_r(\omega)}.$$

With the $\exp(-j\omega t)$ FFT convention, the material models are

$$\epsilon_{r,\mathrm{Debye}}=\epsilon_{r,\infty}+\sum_p\frac{\Delta\epsilon_{r,p}}{1+j\omega\tau_p},$$

$$\epsilon_{r,\mathrm{Lorentz}}=\epsilon_{r,\infty}+\sum_p\frac{\Delta\epsilon_{r,p}\Omega_p^2}{\Omega_p^2-\omega^2+2j\delta_p\omega},$$

$$\epsilon_{r,\mathrm{Drude}}=\epsilon_{r,\infty}-\sum_p\frac{\Omega_p^2}{\omega^2-j\gamma_p\omega}.$$

This follows the inclusive recursive-convolution formulation in [Giannakis and Giannopoulos (2014)](https://doi.org/10.1109/TAP.2014.2308549).

## Results

Overall validation status: **PASS**.

| Material | Magnitude RMSE | Maximum magnitude error | Phase RMSE |
|---|---:|---:|---:|
| Dielectric, $\epsilon_r=4$ (smoothed) | 0.000536 | 0.001177 | 8.63e-06 deg |
| Dielectric, $\epsilon_r=4$ (unsmoothed) | 0.0005357 | 0.001176 | 8.64e-06 deg |
| Debye, 1 pole | 0.0004085 | 0.0008145 | 0.0152 deg |
| Debye, 3 poles | 0.0006136 | 0.001249 | 0.0203 deg |
| Lorentz, 2 poles | 0.0003327 | 0.00075 | 0.00739 deg |
| Drude, 2 poles | 0.001204 | 0.002649 | 0.000437 deg |

Both magnitude and phase are compared throughout 0.25--8 GHz. The
smoothed and unsmoothed dielectric cases explicitly verify the half-cell
change in the discrete reflection plane. The complex Lorentz pole update
uses the full real part `Re(a*T)` required by the recursive-convolution
formulation, rather than the incorrect product `Re(a)*Re(T)`.

## Plots

- [Reflection magnitude](reflection_magnitude.png)
- [Reflection phase](reflection_phase.png)
- [Time-domain fields](time_domain_fields.png)
- [Material permittivity](material_permittivity.png)

Per-frequency complex data are stored in the six `*_reflection.csv` files.
Simulation HDF5 files are reusable cache data and are not retained as
validation evidence.
