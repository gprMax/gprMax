# CUDA `RxPort` consistency check

This suite checks invariance to excitation amplitude, reference impedance,
and x/y/z orientation, as well as single/double precision agreement and
passivity. It exercises the production `/ports/feed` HDF5 data path and does
not compare with MATLAB or claim analytical validation.

From the repository root, run:

```bash
FI_PROVIDER=shm python testing/backend_consistency/rx_port/check_rx_port_gpu_invariance.py --gpu 0
```

The HDF5 outputs, plot, and JSON metrics are written beneath `results/`.
