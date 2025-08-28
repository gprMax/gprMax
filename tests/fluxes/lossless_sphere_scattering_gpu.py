import PyMieScatt as ps
import numpy as np
import matplotlib.pyplot as plt
import h5py

c = 3e8

sphere_radius = 1e-6
sphere_refractive_index = 2
lambda_min = 2*np.pi*1e-6/10
lambda_max = 2*np.pi*1e-6/2
lambda_cen = (lambda_max + lambda_min)/2
dfrq = c/lambda_min - c/lambda_max
print(lambda_min, lambda_max)

print(c/lambda_cen)
print(1/dfrq)

wavelengths = np.linspace(lambda_min,lambda_max,100)

scatt_eff_theory = [ps.MieQ(sphere_refractive_index,f*1e9,2*sphere_radius * 1e9,asDict=True)['Qsca'] for f in wavelengths]

file = h5py.File('Mie_scattering_gpu_fluxes.out', 'r')

incident_flux = np.array(file["scattering"]["incidents"]["incident1"]["values"][()])/(2*sphere_radius)**2
scatt_flux = np.array(file['boxes']['box1']['values'][()])

scatt_eff= scatt_flux/incident_flux /(np.pi*sphere_radius**2)

dist = np.linalg.norm(scatt_eff - scatt_eff_theory)/np.linalg.norm(scatt_eff_theory)

plt.figure()
fig, ax = plt.subplots()
ax.loglog(2*np.pi*sphere_radius/np.linspace(lambda_min, lambda_max, 100),scatt_eff,'ro-',label='cpu', alpha = 0.3)
ax.loglog(2*np.pi*sphere_radius/np.linspace(lambda_min, lambda_max, 100),scatt_eff_theory,'go-',label='theory', alpha = 0.3)
ax.set_xlabel('(sphere circumference)/wavelength, 2πr/λ')
ax.set_ylabel('scattering efficiency, σ/πr$^{2}$')
ax.set_title('Mie Scattering of a Lossless Dielectric Sphere')
ax.grid(True,which="both",ls="-")
ax.legend(loc = 'upper right')
fig.text(0.5, 0.85, "Distance between fluxes implementation ran with GPU and Theory: {:.2e}".format(dist),
         ha="center", va="top",
         fontsize=8,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
plt.savefig("Mie_scattering_gpu.png", dpi = 300)
plt.close()