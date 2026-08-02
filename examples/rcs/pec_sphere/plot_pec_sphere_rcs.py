"""Plot gprMax PEC-sphere RCS and compare it with the Mie series."""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
from scipy.special import spherical_jn, spherical_yn

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "pec_sphere_rcs"
RADIUS = 0.016
GROUP = "ntff/rcs_surface/frequency/rcs_spectrum/far_field/backscatter"


def pec_mie_rcs(frequency, radius, scattering_angle):
    """Return the perpendicular-polarised PEC-sphere RCS."""

    wavenumber = 2 * np.pi * frequency / c
    size_parameter = wavenumber * radius
    maximum_order = max(
        1,
        int(np.ceil(size_parameter + 4 * np.cbrt(size_parameter) + 2)),
    )
    order = np.arange(1, maximum_order + 1)
    jn = spherical_jn(order, size_parameter)
    yn = spherical_yn(order, size_parameter)
    jn_derivative = spherical_jn(order, size_parameter, derivative=True)
    yn_derivative = spherical_yn(order, size_parameter, derivative=True)
    psi = size_parameter * jn
    psi_derivative = jn + size_parameter * jn_derivative
    xi = size_parameter * (jn + 1j * yn)
    xi_derivative = jn + 1j * yn + size_parameter * (jn_derivative + 1j * yn_derivative)
    electric = -psi_derivative / xi_derivative
    magnetic = -psi / xi

    cosine = np.cos(scattering_angle)
    amplitude = np.zeros(cosine.shape, dtype=np.complex128)
    pi_previous = np.zeros(cosine.shape)
    pi_current = np.ones(cosine.shape)
    for n, (electric_n, magnetic_n) in enumerate(zip(electric, magnetic), start=1):
        if n == 1:
            pi_n = pi_current
        else:
            pi_n = ((2 * n - 1) * cosine * pi_current - n * pi_previous) / (n - 1)
            pi_previous, pi_current = pi_current, pi_n
        tau_n = n * cosine * pi_n - (n + 1) * pi_previous
        factor = (2 * n + 1) / (n * (n + 1))
        amplitude += factor * (electric_n * pi_n + magnetic_n * tau_n)
    return 4 * np.pi * np.abs(amplitude) ** 2 / wavenumber**2


def read_gprmax(filename):
    with h5py.File(filename, "r") as output:
        group = output[GROUP]
        transform = output["ntff/rcs_surface/frequency/rcs_spectrum"]
        frequency = np.asarray(transform["frequencies"], dtype=np.float64)
        rcs = np.asarray(group["fields/rcs"][:, 0], dtype=np.float64)
    return frequency, rcs


def plot_comparison(filename, destination):
    frequency, gprmax_rcs = read_gprmax(filename)
    mie_rcs = np.asarray(
        [pec_mie_rcs(value, RADIUS, np.asarray([np.pi]))[0] for value in frequency]
    )
    dense_frequency = np.linspace(frequency[0], frequency[-1], 1200)
    dense_mie_rcs = np.asarray(
        [pec_mie_rcs(value, RADIUS, np.asarray([np.pi]))[0] for value in dense_frequency]
    )
    size_parameter = 2 * np.pi * frequency * RADIUS / c
    dense_size_parameter = 2 * np.pi * dense_frequency * RADIUS / c
    geometric_cross_section = np.pi * RADIUS**2
    tiny = np.finfo(np.float64).tiny
    gprmax_dbsm = 10 * np.log10(np.maximum(gprmax_rcs, tiny))
    mie_dbsm = 10 * np.log10(np.maximum(mie_rcs, tiny))
    error_db = gprmax_dbsm - mie_dbsm

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": (2.4, 1)},
        constrained_layout=True,
    )
    axes[0].semilogy(
        dense_size_parameter,
        dense_mie_rcs / geometric_cross_section,
        color="tab:orange",
        label="PEC Mie series",
        linewidth=2,
    )
    axes[0].semilogy(
        size_parameter,
        gprmax_rcs / geometric_cross_section,
        "o",
        color="tab:blue",
        label="gprMax FDTD",
        markersize=4,
    )
    axes[0].set_ylabel(r"Normalized backscatter RCS, $\sigma/(\pi a^2)$")
    axes[0].set_title(f"PEC sphere monostatic RCS: radius {RADIUS * 1e3:g} mm")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(size_parameter, error_db, "o-", color="tab:red", markersize=4)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel(r"Electrical size, $ka=2\pi f a/c$")
    axes[1].set_ylabel("gprMax - Mie (dB)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(size_parameter[0], size_parameter[-1])

    def ka_to_ghz(value):
        return value * c / (2 * np.pi * RADIUS) / 1e9

    def ghz_to_ka(value):
        return value * 1e9 * 2 * np.pi * RADIUS / c

    frequency_axis = axes[0].secondary_xaxis("top", functions=(ka_to_ghz, ghz_to_ka))
    frequency_axis.set_xlabel("Frequency (GHz)")
    figure.savefig(destination, dpi=180)
    plt.close(figure)

    reference = int(np.argmin(np.abs(size_parameter - 1)))
    print(f"At ka={size_parameter[reference]:.3f}:")
    print(f"  gprMax backscatter: {gprmax_dbsm[reference]:.3f} dBsm")
    print(f"  Mie backscatter: {mie_dbsm[reference]:.3f} dBsm")
    print(f"RMS sweep error: {np.sqrt(np.mean(error_db**2)):.3f} dB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=SCRIPT_DIR / f"{OUTPUT_STEM}.h5",
        help="gprMax HDF5 output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / f"{OUTPUT_STEM}.png",
        help="destination PNG",
    )
    args = parser.parse_args()
    plot_comparison(args.input, args.output)


if __name__ == "__main__":
    main()
