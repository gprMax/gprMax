"""``SimulationConfig._set_precision`` — ``gprMax/config.py:371``.

One string — ``"single"`` or ``"double"`` — chooses six different data types,
and through them the memory footprint of every field array, the Cython fused
type each kernel binds, and the C type name pasted into every generated GPU
kernel.

It also, indirectly, chooses the *name* of the compiled dispersive kernel that
will run: ``CPUUpdates.set_dispersive_updates`` reads the same
``general["precision"]`` string to decide between ``_float_`` and
``_double_``. Those two facts live in different files with nothing tying them
together, so ``TestConsistencyWithTheKernelDispatch`` asserts them side by
side.

The whole thing is one ``if``/``elif`` with **no terminal** ``else``, which is
the most consequential instance of that pattern found in this PR — see
``TestUnknownPrecision``.
"""

import itertools

import cython
import numpy as np
import pytest

# The complete mapping, transcribed from the source. Written out rather than
# computed so that a change to the source shows up as a diff here.
EXPECTED_DTYPES = {
    "single": {
        "float_or_double": np.float32,
        "complex": np.complex64,
        "cython_float_or_double": cython.float,
        "cython_complex": cython.floatcomplex,
        "C_float_or_double": "float",
    },
    "double": {
        "float_or_double": np.float64,
        "complex": np.complex128,
        "cython_float_or_double": cython.double,
        "cython_complex": cython.doublecomplex,
        "C_float_or_double": "double",
    },
}

# ``C_complex`` is the only entry that depends on the solver as well as the
# precision, because each GPU backend spells its complex type differently.
EXPECTED_C_COMPLEX = {
    ("single", "cpu"): None,
    ("single", "cuda"): "pycuda::complex<float>",
    ("single", "opencl"): "cfloat_t",
    ("single", "metal"): "gprMaxComplex",
    ("double", "cpu"): None,
    ("double", "cuda"): "pycuda::complex<double>",
    ("double", "opencl"): "cdouble_t",
    ("double", "metal"): "gprMaxComplex",
}

DTYPE_KEYS = {
    "float_or_double",
    "complex",
    "cython_float_or_double",
    "cython_complex",
    "C_float_or_double",
    "C_complex",
}

# How each precision is reached through the public argument surface. There is
# no ``--precision`` flag: single is the default and double is only reachable
# by asking for sub-grids.
ARGS_FOR_PRECISION = {"single": {}, "double": {"subgrid": True}}


class TestDtypeKeys:
    """The shape of the ``dtypes`` dictionary."""

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_has_exactly_six_keys(self, make_sim_config, precision):
        """Six, at both precisions.

        Note the stand-in configs in every other test directory supply only
        two of these (``float_or_double`` and ``complex``). Anything reading
        a ``cython_*`` or ``C_*`` key would fail against those stubs — which
        is precisely the drift a config-level suite exists to pin.
        """
        sim_config = make_sim_config(**ARGS_FOR_PRECISION[precision])

        assert set(sim_config.dtypes) == DTYPE_KEYS

    def test_dtypes_is_created_during_construction(self, make_sim_config):
        """``_set_precision`` runs unconditionally in ``__init__``."""
        assert hasattr(make_sim_config(), "dtypes")

    def test_each_instance_gets_its_own_dictionary(self, make_sim_config):
        """Unlike ``em_consts``, ``dtypes`` is per instance."""
        first = make_sim_config()
        second = make_sim_config()

        assert first.dtypes is not second.dtypes


class TestSinglePrecision:
    """The default: 32-bit fields."""

    @pytest.mark.parametrize(
        "key,expected", sorted(EXPECTED_DTYPES["single"].items(), key=lambda kv: kv[0])
    )
    def test_dtype_entry(self, make_sim_config, key, expected):
        assert make_sim_config().dtypes[key] == expected

    def test_field_arrays_would_be_float32(self, make_sim_config):
        """``float_or_double`` is what every array allocation uses."""
        dtype = make_sim_config().dtypes["float_or_double"]

        assert np.zeros(1, dtype=dtype).itemsize == 4

    def test_complex_type_matches_the_real_one_in_width(self, make_sim_config):
        """A complex value is two floats of the chosen width."""
        dtypes = make_sim_config().dtypes

        assert (
            np.zeros(1, dtype=dtypes["complex"]).itemsize
            == 2 * np.zeros(1, dtype=dtypes["float_or_double"]).itemsize
        )

    def test_cython_type_is_the_float_shadow(self, make_sim_config):
        """Identity, not equality — the shadow types are singletons.

        ``cython.float`` and ``cython.double`` are distinct objects in the
        pure-Python shadow module, so an identity check is meaningful.
        """
        assert make_sim_config().dtypes["cython_float_or_double"] is cython.float


class TestDoublePrecision:
    """64-bit fields, reached by asking for sub-grids."""

    @pytest.mark.parametrize(
        "key,expected", sorted(EXPECTED_DTYPES["double"].items(), key=lambda kv: kv[0])
    )
    def test_dtype_entry(self, make_sim_config, key, expected):
        assert make_sim_config(subgrid=True).dtypes[key] == expected

    def test_field_arrays_would_be_float64(self, make_sim_config):
        dtype = make_sim_config(subgrid=True).dtypes["float_or_double"]

        assert np.zeros(1, dtype=dtype).itemsize == 8

    def test_cython_type_is_the_double_shadow(self, make_sim_config):
        assert make_sim_config(subgrid=True).dtypes["cython_float_or_double"] is cython.double

    def test_the_two_cython_shadows_are_distinct(self):
        """Guards the identity assertions above from being vacuous."""
        assert cython.float is not cython.double
        assert cython.floatcomplex is not cython.doublecomplex


class TestPrecisionsDiffer:
    """Every entry actually changes between the two precisions."""

    @pytest.mark.parametrize("key", sorted(EXPECTED_DTYPES["single"]))
    def test_entry_differs_between_precisions(self, make_sim_config, key):
        """No key is accidentally shared, which would make it unswitchable."""
        single = make_sim_config().dtypes[key]
        double = make_sim_config(subgrid=True).dtypes[key]

        assert single is not double

    def test_double_precision_doubles_the_field_footprint(self, make_sim_config):
        single = make_sim_config().dtypes["float_or_double"]
        double = make_sim_config(subgrid=True).dtypes["float_or_double"]

        assert np.zeros(1, dtype=double).itemsize == 2 * np.zeros(1, dtype=single).itemsize


class TestCComplexPerSolver:
    """``C_complex`` — the one entry that also depends on the backend.

    Each GPU toolchain spells a complex number differently, and the string
    here is pasted verbatim into generated kernel source. A wrong value is a
    compile error at run time, on hardware CI does not have.
    """

    @pytest.mark.parametrize(
        "precision,solver",
        list(itertools.product(["single", "double"], ["cpu", "cuda", "opencl", "metal"])),
    )
    def test_c_complex_matrix(self, make_sim_config, precision, solver):
        """All eight combinations of precision and backend.

        Half of these cannot be reached through the argument surface — every
        accelerator forces single precision — so the two settings are driven
        directly and ``_set_precision`` re-run. That exercises the real
        mapping rather than skipping the rows, and it is also how a caller
        using the Python API could reach them.
        """
        sim_config = make_sim_config()
        sim_config.general["precision"] = precision
        sim_config.general["solver"] = solver

        sim_config._set_precision()

        assert sim_config.dtypes["C_complex"] == EXPECTED_C_COMPLEX[(precision, solver)]

    @pytest.mark.parametrize(
        "precision,solver",
        list(itertools.product(["single", "double"], ["cpu", "cuda", "opencl", "metal"])),
    )
    def test_real_dtypes_are_unaffected_by_the_solver(self, make_sim_config, precision, solver):
        """Only ``C_complex`` varies with the backend; the other five do not."""
        sim_config = make_sim_config()
        sim_config.general["precision"] = precision
        sim_config.general["solver"] = solver

        sim_config._set_precision()

        for key, expected in EXPECTED_DTYPES[precision].items():
            assert sim_config.dtypes[key] == expected

    def test_cpu_leaves_c_complex_unset(self, make_sim_config):
        """Nothing generates C source on the CPU path, so there is no name."""
        assert make_sim_config().dtypes["C_complex"] is None

    def test_cuda_uses_the_pycuda_complex_template(self, make_sim_config):
        assert make_sim_config(gpu=[0]).dtypes["C_complex"] == "pycuda::complex<float>"

    def test_opencl_uses_the_short_form(self, make_sim_config):
        assert make_sim_config(opencl=[0]).dtypes["C_complex"] == "cfloat_t"

    def test_metal_uses_the_namespaced_form(self, make_sim_config):
        assert make_sim_config(metal=[0]).dtypes["C_complex"] == "gprMaxComplex"

    def test_the_real_c_type_is_independent_of_the_solver(self, make_sim_config):
        """Only the *complex* name varies; ``float`` is ``float`` everywhere."""
        for kwargs in ({}, {"gpu": [0]}, {"opencl": [0]}, {"metal": [0]}):
            assert make_sim_config(**kwargs).dtypes["C_float_or_double"] == "float"


class TestUnknownPrecision:
    """The missing terminal ``else`` — the most consequential one found.

    ``_set_precision`` is ``if precision == "single" ... elif precision ==
    "double"`` with nothing after it. Any other value leaves ``self.dtypes``
    **never assigned at all**, and because the method is called at the very
    end of ``__init__`` the object is returned looking complete.

    The first symptom is an ``AttributeError`` about ``dtypes`` raised from
    whichever module happens to read it first — typically an array allocation
    in ``FDTDGrid``, far from the setting that caused it.

    No test asserts the broken behaviour. These tests establish the boundary:
    the two valid values work, and the attribute's presence is what a caller
    depends on. The defect is written up in
    ``notes/bugs/config-precision-no-terminal-else.md``.
    """

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_a_recognised_precision_produces_a_complete_dtype_table(
        self, make_sim_config, precision
    ):
        sim_config = make_sim_config(**ARGS_FOR_PRECISION[precision])

        assert set(sim_config.dtypes) == DTYPE_KEYS
        assert all(sim_config.dtypes[key] is not None for key in DTYPE_KEYS - {"C_complex"})

    def test_only_two_precision_values_are_reachable_from_the_arguments(self, make_sim_config):
        """There is no ``--precision`` flag.

        Single is the default and double is reached only via ``subgrid``, so
        no user input can select a third value. That is what keeps the
        missing ``else`` latent rather than live — it is reachable only by a
        caller setting ``general["precision"]`` directly, which the API
        permits.
        """
        assert make_sim_config().general["precision"] == "single"
        assert make_sim_config(subgrid=True).general["precision"] == "double"

    def test_calling_set_precision_again_with_a_bad_value_leaves_stale_dtypes(
        self, make_sim_config
    ):
        """Upstream now raises ValueError for unrecognised precision
        instead of silently ignoring it."""
        sim_config = make_sim_config()

        sim_config.general["precision"] = "quadruple"
        with pytest.raises(ValueError):
            sim_config._set_precision()


class TestConsistencyWithTheKernelDispatch:
    """The knot between this file and ``cpu_updates.set_dispersive_updates``.

    Both read ``general["precision"]``. This file turns it into array dtypes;
    the dispatcher turns it into part of a compiled kernel's name. If they
    ever disagreed, the arrays would be allocated at one width and handed to
    a kernel compiled for the other — which Cython rejects at the call
    boundary with a buffer dtype error rather than computing wrongly.

    Nothing in the source ties the two together, so these tests are the tie.
    """

    @pytest.mark.parametrize(
        "precision,marker,expected_dtype",
        [
            ("single", "_float_", np.float32),
            ("double", "_double_", np.float64),
        ],
    )
    def test_array_dtype_and_kernel_name_agree(
        self,
        make_sim_config,
        install_sim_config,
        monkeypatch,
        precision,
        marker,
        expected_dtype,
    ):
        from types import SimpleNamespace

        from gprMax import config
        from gprMax.updates.cpu_updates import CPUUpdates

        sim_config = install_sim_config(**ARGS_FOR_PRECISION[precision])
        model_cfg = SimpleNamespace(
            mode="3D",
            ompthreads=1,
            materials={
                "maxpoles": 1,
                "dispersivedtype": sim_config.dtypes["float_or_double"],
            },
        )
        monkeypatch.setattr(config, "get_model_config", lambda: model_cfg)

        updates = CPUUpdates(SimpleNamespace(ntff_monitors=[]))
        updates.set_dispersive_updates()

        assert sim_config.dtypes["float_or_double"] is expected_dtype
        assert marker in updates.dispersive_update_a.__name__

    def test_the_c_type_name_matches_the_numpy_width(self, make_sim_config):
        """``C_float_or_double`` and ``float_or_double`` describe one type."""
        for kwargs, c_name, itemsize in (
            ({}, "float", 4),
            ({"subgrid": True}, "double", 8),
        ):
            sim_config = make_sim_config(**kwargs)
            assert sim_config.dtypes["C_float_or_double"] == c_name
            assert np.zeros(1, dtype=sim_config.dtypes["float_or_double"]).itemsize == itemsize


pytestmark = pytest.mark.unit
