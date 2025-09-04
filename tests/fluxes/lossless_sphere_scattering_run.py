import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
from gprMax.gprMax import api as run_sim
import GPUtil
import h5py

deviceIDs = GPUtil.getAvailable()

r = 1.0e-6

wvl_min = 2*np.pi*r/10
wvl_max = 2*np.pi*r/2

frq_min = c/wvl_max
frq_max = c/wvl_min
frq_cen = 0.5*(frq_min+frq_max)
dfrq = frq_max-frq_min
std = wvl_max/c-wvl_min/c
nfrq = 100

## at least 8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
resolution = 25
dl = 1.0e-6/resolution

dpml = 0.5*wvl_max
dair = 0.5*wvl_max

pmlcells = int(np.ceil(dpml/dl))
s = 2*(dpml+dair+r)
domain_size = [s,s,s]
gpu = "gpu"
timewindow = 150e-15
file = open("Theory_comparison.txt", "w")

file.write("#title: Mie_scattering_" + gpu + "\n")
file.write("#domain: {} {} {}\n".format(domain_size[0], domain_size[1], domain_size[2]))
file.write("#dx_dy_dz: {} {} {}\n".format(dl, dl, dl))
file.write("#time_window: {}\n".format(timewindow))
file.write("#pml_cells: {}\n".format(pmlcells))
file.write("#time_step_stability_factor: 0.5\n")

waveform_type = "gaussian"
waveform_freq = frq_cen
waveform_ampl = 1.0
waveform_ID = "my_gaussian"
waveform_std = std

file.write("#waveform: {} {} {} {}\n".format(waveform_type, waveform_ampl, waveform_freq, waveform_ID, waveform_std))
file.write("#plane_voltage_source: z {} {} {} {} {} {} {} {}\n".format(dpml + 2*dl, 0, 0, dpml + 2*dl, s, s, 0, waveform_ID))

file.write("#material: 4 0 1 0 my_material\n")

file.write("#box_flux: {} {} {} {} {} {} {} {} {} {} {} {}\n".format(s/2, s/2, s/2, r, r, r, r, r, r, wvl_min, wvl_max, nfrq))

file.write("#scattering:\n")
file.write("#sphere: {} {} {} {} my_material\n".format(s/2, s/2, s/2, r))
file.write("#scattering_end:\n")

file.write("#geometry_view: 0 0 0 {} {} {} {} {} {} geometry n".format(s,s,s,4*dl,4*dl,4*dl))

file.close()

run_sim(inputfile= 'Meep_comparison.txt', gpu= deviceIDs)

file = h5py.File('Mie_scattering_'+gpu+'_fluxes.out', 'r')

incident_flux = np.array(file["scattering"]["incidents"]["incident1"]["values"][()])/(2*r)**2
scatt_flux = np.array(file['boxes']['box1']['values'][()])

scatt_eff= scatt_flux/incident_flux /(np.pi*r**2)

plt.figure(dpi=150)
plt.loglog(2*np.pi*r/np.linspace(wvl_min, wvl_max, nfrq),scatt_eff,'ro-',label='gpu')
plt.grid()
plt.xlabel('(sphere circumference)/wavelength, 2πr/λ')
plt.ylabel('scattering efficiency, σ/πr$^{2}$')
plt.legend(loc='upper right')
plt.title('Mie Scattering of a Lossless Dielectric Sphere')
plt.tight_layout()
plt.savefig("Mie_scattering.png")
plt.close()
