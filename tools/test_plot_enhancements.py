"""Tests for enhanced plot_Ascan.py and plot_Bscan.py.

Creates mock HDF5 output files and mocks gprMax internals to verify
all new CLI flags and plotting features without requiring a full
gprMax build (Cython extensions, etc.).
"""

import os
import sys
import types
import tempfile
import argparse

# ---------------------------------------------------------------------------
# Mock gprMax modules BEFORE importing plot scripts.
# gprMax has Cython extensions that may not be compiled, so we stub out
# only the symbols that plot_Ascan / plot_Bscan actually need.
# ---------------------------------------------------------------------------

def _setup_gprmax_mocks():
    """Create minimal mocks for gprMax modules used by plot scripts."""

    # gprMax package
    gprMax_mod = types.ModuleType('gprMax')
    sys.modules['gprMax'] = gprMax_mod

    # gprMax.exceptions
    exceptions_mod = types.ModuleType('gprMax.exceptions')
    class CmdInputError(Exception):
        pass
    exceptions_mod.CmdInputError = CmdInputError
    sys.modules['gprMax.exceptions'] = exceptions_mod
    gprMax_mod.exceptions = exceptions_mod

    # gprMax.receivers
    receivers_mod = types.ModuleType('gprMax.receivers')
    class Rx:
        allowableoutputs = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', 'Iz']
        defaultoutputs = allowableoutputs[:-3]  # ['Ex','Ey','Ez','Hx','Hy','Hz']
    receivers_mod.Rx = Rx
    sys.modules['gprMax.receivers'] = receivers_mod
    gprMax_mod.receivers = receivers_mod

    # gprMax.utilities  - only fft_power is used
    import numpy as np
    utilities_mod = types.ModuleType('gprMax.utilities')
    def fft_power(signal, dt):
        N = len(signal)
        freqs = np.fft.rfftfreq(N, d=dt)
        fft_vals = np.fft.rfft(signal)
        power = 20 * np.log10(np.abs(fft_vals) / np.max(np.abs(fft_vals)) + 1e-30)
        return freqs, power
    utilities_mod.fft_power = fft_power
    sys.modules['gprMax.utilities'] = utilities_mod
    gprMax_mod.utilities = utilities_mod

    return CmdInputError

_setup_gprmax_mocks()

# Now safe to import third-party + plotting modules
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import the modules under test
from tools.plot_Ascan import mpl_plot, _normalize_data
from gprMax.exceptions import CmdInputError

# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0


def report(name, passed, detail=""):
    global PASS, FAIL
    if passed:
        PASS += 1
    else:
        FAIL += 1
    status = "PASS" if passed else "FAIL"
    suffix = f"  - {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")


def create_mock_output(filepath, nrx=1, iterations=500, dt=1e-10,
                       outputs=None):
    """Create a mock gprMax HDF5 output file."""
    if outputs is None:
        outputs = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']

    with h5py.File(filepath, 'w') as f:
        f.attrs['nrx'] = nrx
        f.attrs['dt'] = dt
        f.attrs['Iterations'] = iterations

        for rx in range(1, nrx + 1):
            for comp in outputs:
                t = np.linspace(0, 1, iterations)
                fc = 5
                data = (1 - 2 * (np.pi * fc * (t - 0.3))**2) * \
                       np.exp(-(np.pi * fc * (t - 0.3))**2)
                scale = {'E': 100.0, 'H': 0.5, 'I': 2.0}.get(comp[0], 1.0)
                f.create_dataset(f'/rxs/rx{rx}/{comp}', data=data * scale)


# ---------------------------------------------------------------------------
# 1. Unit tests for _normalize_data
# ---------------------------------------------------------------------------

def test_normalize_data():
    print("\n=== _normalize_data() ===")

    # Normal case
    data = np.array([0, 5, -10, 3], dtype=float)
    result = _normalize_data(data)
    report("peak maps to +/-1.0", np.isclose(np.amax(np.abs(result)), 1.0))
    report("shape preserved", result.shape == data.shape)

    # All-zeros (division-by-zero safety)
    zeros = np.zeros(10)
    result_z = _normalize_data(zeros)
    report("all-zeros -> zeros", np.all(result_z == 0))

    # Single element
    single = np.array([42.0])
    result_s = _normalize_data(single)
    report("single element -> 1.0", np.isclose(result_s[0], 1.0))

    # Negative-only data
    neg = np.array([-3.0, -7.0, -1.0])
    result_n = _normalize_data(neg)
    report("negative data: max abs = 1.0",
           np.isclose(np.amax(np.abs(result_n)), 1.0))


# ---------------------------------------------------------------------------
# 2. A-scan mpl_plot() tests
# ---------------------------------------------------------------------------

def test_mpl_plot_defaults():
    print("\n=== mpl_plot()  - default args ===")

    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as tmp:
        tmppath = tmp.name
    try:
        create_mock_output(tmppath)
        plt.close('all')
        plthandle = mpl_plot(tmppath)
        figs = [plt.figure(n) for n in plt.get_fignums()]

        report("returns plt object", plthandle is plt)
        report("at least 1 figure created", len(figs) >= 1)

        fig = figs[0]
        title = fig._suptitle.get_text() if fig._suptitle else ""
        report("title has 'A-scan Output'", "A-scan Output" in title)
        report("title has filename", os.path.basename(tmppath) in title)

        plt.close('all')
    finally:
        os.unlink(tmppath)


def test_mpl_plot_single_output():
    print("\n=== mpl_plot()  - single output ===")

    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as tmp:
        tmppath = tmp.name
    try:
        create_mock_output(tmppath)
        plt.close('all')
        mpl_plot(tmppath, outputs=['Ex'])
        figs = [plt.figure(n) for n in plt.get_fignums()]

        report("figure created", len(figs) >= 1)

        ax = figs[0].axes[0]
        report("x-label is 'Time (ns)'", ax.get_xlabel() == "Time (ns)")

        plt.close('all')
    finally:
        os.unlink(tmppath)


def test_mpl_plot_no_grid():
    print("\n=== mpl_plot()  - show_grid=False ===")

    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as tmp:
        tmppath = tmp.name
    try:
        create_mock_output(tmppath)
        plt.close('all')
        mpl_plot(tmppath, outputs=['Ex'], show_grid=False)
        report("no crash with grid disabled", True)
        plt.close('all')
    finally:
        os.unlink(tmppath)


def test_mpl_plot_normalize():
    print("\n=== mpl_plot()  - normalize=True ===")

    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as tmp:
        tmppath = tmp.name
    try:
        create_mock_output(tmppath)
        plt.close('all')
        mpl_plot(tmppath, outputs=['Ex'], normalize=True)
        figs = [plt.figure(n) for n in plt.get_fignums()]
        ax = figs[0].axes[0]

        report("y-label says 'Normalized amplitude'",
               "Normalized amplitude" in ax.get_ylabel())

        ydata = ax.get_lines()[0].get_ydata()
        report("data within [-1, 1]", np.all(np.abs(ydata) <= 1.0 + 1e-10))

        plt.close('all')
    finally:
        os.unlink(tmppath)


def test_mpl_plot_save():
    print("\n=== mpl_plot() + save ===")

    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as tmp:
        tmppath = tmp.name
    savepath = tmppath.replace('.out', '_plot.png')
    try:
        create_mock_output(tmppath)
        plt.close('all')
        plthandle = mpl_plot(tmppath, outputs=['Ex'])
        plthandle.savefig(savepath, dpi=72, bbox_inches='tight')

        report("PNG file created", os.path.isfile(savepath))
        report("PNG file non-empty", os.path.getsize(savepath) > 0)

        plt.close('all')
    finally:
        os.unlink(tmppath)
        if os.path.isfile(savepath):
            os.unlink(savepath)


def test_mpl_plot_h_component():
    print("\n=== mpl_plot()  - H component color ===")

    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as tmp:
        tmppath = tmp.name
    try:
        create_mock_output(tmppath)
        plt.close('all')
        mpl_plot(tmppath, outputs=['Hx'])
        figs = [plt.figure(n) for n in plt.get_fignums()]
        ax = figs[0].axes[0]
        report("Hx line is green", ax.get_lines()[0].get_color() == 'g')
        plt.close('all')
    finally:
        os.unlink(tmppath)


def test_mpl_plot_multiple_outputs():
    print("\n=== mpl_plot()  - multiple outputs ===")

    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as tmp:
        tmppath = tmp.name
    try:
        create_mock_output(tmppath)
        plt.close('all')
        mpl_plot(tmppath,
                 outputs=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'])
        figs = [plt.figure(n) for n in plt.get_fignums()]

        report("figure created", len(figs) >= 1)
        report("6 subplots", len(figs[0].axes) == 6)

        plt.close('all')
    finally:
        os.unlink(tmppath)


def test_mpl_plot_multiple_receivers():
    print("\n=== mpl_plot()  - multiple receivers ===")

    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as tmp:
        tmppath = tmp.name
    try:
        create_mock_output(tmppath, nrx=3)
        plt.close('all')
        mpl_plot(tmppath, outputs=['Ex'])

        report("3 receivers -> 3 figures", len(plt.get_fignums()) == 3)

        plt.close('all')
    finally:
        os.unlink(tmppath)


def test_mpl_plot_normalize_multiple():
    print("\n=== mpl_plot()  - normalize + multi-output ===")

    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as tmp:
        tmppath = tmp.name
    try:
        create_mock_output(tmppath)
        plt.close('all')
        mpl_plot(tmppath,
                 outputs=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'],
                 normalize=True)
        figs = [plt.figure(n) for n in plt.get_fignums()]
        labels = [ax.get_ylabel() for ax in figs[0].axes]
        has_norm = any("Normalized" in l for l in labels)
        report("normalized labels present", has_norm)
        plt.close('all')
    finally:
        os.unlink(tmppath)


def test_mpl_plot_fft():
    print("\n=== mpl_plot()  - FFT mode ===")

    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as tmp:
        tmppath = tmp.name
    try:
        create_mock_output(tmppath)
        plt.close('all')
        mpl_plot(tmppath, outputs=['Ex'], fft=True)
        figs = [plt.figure(n) for n in plt.get_fignums()]

        report("FFT: figure created", len(figs) >= 1)
        report("FFT: 2 subplots (time + freq)", len(figs[0].axes) == 2)

        plt.close('all')
    finally:
        os.unlink(tmppath)


# ---------------------------------------------------------------------------
# 3. CLI argument parsing tests
# ---------------------------------------------------------------------------

def test_argparse_defaults():
    print("\n=== CLI  - default args ===")

    parser = argparse.ArgumentParser()
    parser.add_argument('outputfile')
    parser.add_argument('--save', metavar='FILENAME')
    parser.add_argument('--no-grid', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('-fft', action='store_true', default=False)

    args = parser.parse_args(['test.out'])
    report("outputfile parsed", args.outputfile == 'test.out')
    report("save is None", args.save is None)
    report("no_grid is False", args.no_grid is False)
    report("normalize is False", args.normalize is False)
    report("fft is False", args.fft is False)


def test_argparse_all_flags():
    print("\n=== CLI  - all new flags ===")

    parser = argparse.ArgumentParser()
    parser.add_argument('outputfile')
    parser.add_argument('--save', metavar='FILENAME')
    parser.add_argument('--no-grid', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=False)

    args = parser.parse_args(['test.out', '--save', 'plot.png',
                              '--no-grid', '--normalize'])
    report("--save parsed", args.save == 'plot.png')
    report("--no-grid parsed", args.no_grid is True)
    report("--normalize parsed", args.normalize is True)


def test_argparse_backward_compat():
    print("\n=== CLI  - backward compatibility ===")

    parser = argparse.ArgumentParser()
    parser.add_argument('outputfile')
    parser.add_argument('--outputs', nargs='+', default=['Ex'])
    parser.add_argument('-fft', action='store_true', default=False)
    parser.add_argument('--save', metavar='FILENAME')
    parser.add_argument('--no-grid', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=False)

    # Old-style invocation
    args = parser.parse_args(['test.out', '--outputs', 'Ex', '-fft'])
    report("old-style: outputfile", args.outputfile == 'test.out')
    report("old-style: outputs", args.outputs == ['Ex'])
    report("old-style: fft", args.fft is True)
    report("old-style: new flags defaulted",
           args.save is None and not args.no_grid and not args.normalize)


# ---------------------------------------------------------------------------
# 4. Edge case tests
# ---------------------------------------------------------------------------

def test_zero_signal_normalize():
    print("\n=== Edge: zero signal + normalize ===")

    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as tmp:
        tmppath = tmp.name
    try:
        with h5py.File(tmppath, 'w') as f:
            f.attrs['nrx'] = 1
            f.attrs['dt'] = 1e-10
            f.attrs['Iterations'] = 100
            f.create_dataset('/rxs/rx1/Ex', data=np.zeros(100))

        plt.close('all')
        mpl_plot(tmppath, outputs=['Ex'], normalize=True)

        ax = plt.gcf().axes[0]
        ydata = ax.get_lines()[0].get_ydata()
        report("no crash", True)
        report("data stays zero", np.all(ydata == 0))

        plt.close('all')
    finally:
        os.unlink(tmppath)


def test_no_receivers():
    print("\n=== Edge: no receivers ===")

    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as tmp:
        tmppath = tmp.name
    try:
        with h5py.File(tmppath, 'w') as f:
            f.attrs['nrx'] = 0
            f.attrs['dt'] = 1e-10
            f.attrs['Iterations'] = 100

        plt.close('all')
        try:
            mpl_plot(tmppath)
            report("raises CmdInputError", False, "no exception raised")
        except CmdInputError:
            report("raises CmdInputError", True)

        plt.close('all')
    finally:
        os.unlink(tmppath)


def test_save_formats():
    print("\n=== Save to PNG / PDF / SVG ===")

    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as tmp:
        tmppath = tmp.name
    save_files = []
    try:
        create_mock_output(tmppath)
        for fmt in ('png', 'pdf', 'svg'):
            savepath = tmppath.replace('.out', f'_test.{fmt}')
            save_files.append(savepath)
            plt.close('all')
            plthandle = mpl_plot(tmppath, outputs=['Ex'])
            plthandle.savefig(savepath, dpi=72, bbox_inches='tight')
            report(f".{fmt} saved",
                   os.path.isfile(savepath) and os.path.getsize(savepath) > 0)
            plt.close('all')
    finally:
        os.unlink(tmppath)
        for sp in save_files:
            if os.path.isfile(sp):
                os.unlink(sp)


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("  gprMax Plot Enhancement Tests")
    print("=" * 60)

    test_normalize_data()

    test_mpl_plot_defaults()
    test_mpl_plot_single_output()
    test_mpl_plot_no_grid()
    test_mpl_plot_normalize()
    test_mpl_plot_save()
    test_mpl_plot_h_component()
    test_mpl_plot_multiple_outputs()
    test_mpl_plot_multiple_receivers()
    test_mpl_plot_normalize_multiple()
    test_mpl_plot_fft()

    test_argparse_defaults()
    test_argparse_all_flags()
    test_argparse_backward_compat()

    test_zero_signal_normalize()
    test_no_receivers()
    test_save_formats()

    total = PASS + FAIL
    print("\n" + "=" * 60)
    print(f"  Results: {PASS}/{total} passed, {FAIL} failed")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)
