Tracking release verification, 24 October 2026
=============================================

The reviewed selection has 684 distinct passing tests after environment repair, with no remaining failures or skips in that selection. This is a targeted tracking/device/reference selection, not the entire repository test suite. The runtime source hashes match the completed higher-order numerical audit.

The initial review log contains 439 passes, 111 failures from stale compiled PML extensions, and 36 CUDA skips. Rebuilding the extensions and rerunning the affected files produced 152 passes and the same 36 CUDA skips; accounting for repeated cases gives 550 distinct passes. Activating MSVC and using a writable compiler temporary directory then produced 36 CUDA passes. The original device tests used legacy tracking; explicit legacy/auto parameterization adds 72 automatic-tracking cases across CUDA and OpenCL, all passing. The analytical-reference and gradient-investigation suites add 26 passes. The selected test set therefore totals 550 + 36 + 72 + 26 = 684 passes.

Device coverage includes both precisions, all three port normals, both propagation directions, two monochromatic polarizations, broadband mixed excitation, physical and virtual guides, and randomized raw modal bases. Receiver fields are compared with CPU physical-guide references. The preserved JUnit files include individual physical/virtual relative errors; limits are 2e-8 in double precision and 2e-4 in single precision. Compiler-success notices from PyCUDA are warnings, not failed compilations.

For Windows CUDA reproduction, activate the x64 MSVC developer environment (vcvars64.bat), make nvcc and cl available on PATH, set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1, and point TEMP, TMP and PYCUDA_CACHE_DIR at writable directories. Use the gprMax Python environment and run::

    python -m pytest tests/test_eigenmode_degenerate_integration.py -k "test_device_physical_labels_and_virtual_continuation"
    python -m pytest tests/test_analytical_step_index_reference.py tests/test_gradient_first_investigation.py

The device command now runs all 144 legacy/automatic CUDA/OpenCL combinations. The original 36 legacy OpenCL cases are recorded in review_tracking_rebuilt.log; the subsequent XML files record 36 legacy CUDA and 72 automatic device cases. review_old_new.json compares the previous symmetry stage from base commit 8d010441b4a6223dea01e8653a4aa62434eb13a4 with the reviewed stage on identical raw banks. Its timings are incidental single-run measurements, not a performance benchmark.

Gallery and worked-example regeneration
---------------------------------------

All eleven modal galleries were regenerated in fresh output directories, comprising twelve geometry-only modal banks and the 64/96-cell material-crossing sweep. The circular TE11 and square anisotropic-crossing tutorials were rerun for both launched channels, including geometry inspection and result plotting. Forty-two PNGs were published; unchanged deterministic plots can be byte-identical to the previous version. The new circular TE11 scattering figures are now included in the documentation.

All retained anchor numerical/reference/propagation masks pass, modal fields and indices are finite, balanced powers are positive, and there are no unresolved tracking intervals. All four driven runs have valid power-wave/coefficient masks and finite complex scattering data and receiver fields. Copper windows retain artificial-edge confinement flags; these are preserved in each manifest and are not a claim of aperture convergence. regenerated_results.json contains per-bank and per-launch checks, raw-file hashes, published-image hashes, and the executed source hashes. Native HDF5 files remain in the fresh local .codex_tmp/tracking_gallery_refresh_20261024 directory; the committed run manifests record commands and settings.

Representative electric/magnetic galleries, both circular TE11 result figures, the square crossing result, copper real/imaginary dispersion, and the analytical crossing profile comparison were visually inspected. Every published PNG was decoded and verified. Documentation HTML builds with the existing autosummary stubs (autosummary_generate=0), since ReFrame is unavailable in the Windows environment. The full build retains 57 existing ReFrame autodoc/autosummary/toctree warnings and has no warnings in the edited eigenmode pages. The source hashes and build command are recorded in verification.json.

Committed logs have trailing whitespace removed; diagnostic text and outcomes are retained.
