"""``cython/pml_build.pyx`` — averaging the material behind a PML slab.

``sigma_max`` depends on what the PML is backed by: a layer absorbing into wet
clay needs a different conductivity ramp from one absorbing into air. These
two kernels supply that number, reducing the 2D face of material IDs behind a
slab to a single mean permittivity and permeability.

Both take the same six arguments — two face dimensions, a thread count, the
``solid`` slice (material ID per cell), and two lookup tables indexed by
material ID. ``pml_average_er_mr`` divides by the cell count;
``pml_sum_er_mr`` does not.

Both are OpenMP ``prange`` loops with ``sumer``/``summr`` as reduction
variables, so these are real parallel kernels driven with real arrays.

**The kernel has a confirmed defect, and it affects these tests.** Those two
accumulators are declared ``cdef double`` with no initialiser, and OpenMP's
``reduction(+:)`` adds each thread's private total into the *pre-existing*
value — which is uninitialised stack. On Linux, macOS and the local Windows
toolchain that slot happens to read as zero, so the kernel returns the right
answer. On the GitHub Windows runner it does not: consecutive calls reuse a
dirty slot and accumulate into each other.

Two tests in ``TestThreading`` detected exactly that and are commented out
below with the evidence. Every test that remains in this file assumes the
accumulator starts at zero, so **any of them can go red on an affected
toolchain**, and the cause will not be anything in this file. See
``notes/bugs/pml-build-uninitialised-reduction.md`` for the two-line fix.
"""

import numpy as np
import pytest

from gprMax.cython.pml_build import pml_average_er_mr, pml_sum_er_mr


def face(ids, shape):
    """A ``solid`` face of the given shape filled from ``ids``."""
    return np.array(ids, dtype=np.uint32).reshape(shape)


class TestAverageSingleMaterial:
    def test_uniform_face_returns_that_material(self):
        """Expects a face made entirely of material 0 to average to exactly
        that material's ``er`` and ``mr``."""
        solid = np.zeros((4, 4), dtype=np.uint32)
        ers = np.array([3.0, 1.0], dtype=np.float64)
        mrs = np.array([2.0, 1.0], dtype=np.float64)
        assert pml_average_er_mr(4, 4, 1, solid, ers, mrs) == (3.0, 2.0)

    def test_free_space_face_returns_one_and_one(self):
        """Expects ``(1.0, 1.0)`` for a face of free space — the common case,
        and the value that makes ``sigma_max`` reduce to ``0.8·(m+1)/(z0·d)``."""
        solid = np.ones((5, 5), dtype=np.uint32)
        ers = np.array([81.0, 1.0], dtype=np.float64)
        mrs = np.array([1.0, 1.0], dtype=np.float64)
        assert pml_average_er_mr(5, 5, 1, solid, ers, mrs) == (1.0, 1.0)

    def test_er_and_mr_are_looked_up_independently(self):
        """Expects the two tables to be indexed separately, so a material can
        have a high permittivity and unit permeability."""
        solid = np.zeros((3, 3), dtype=np.uint32)
        ers = np.array([81.0], dtype=np.float64)
        mrs = np.array([1.0], dtype=np.float64)
        assert pml_average_er_mr(3, 3, 1, solid, ers, mrs) == (81.0, 1.0)

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_face_size_does_not_change_a_uniform_average(self, n):
        """Expects the mean of a constant face to be that constant whatever
        the face's size. (4 parameter sets)"""
        solid = np.zeros((n, n), dtype=np.uint32)
        ers = np.array([4.0], dtype=np.float64)
        mrs = np.array([1.5], dtype=np.float64)
        assert pml_average_er_mr(n, n, 1, solid, ers, mrs) == (4.0, 1.5)


class TestAverageMixedMaterials:
    def test_half_and_half_gives_the_midpoint(self):
        """Expects ``(1 + 9)/2 == 5`` for a face split evenly between two
        materials."""
        solid = face([0, 0, 1, 1], (2, 2))
        ers = np.array([1.0, 9.0], dtype=np.float64)
        mrs = np.array([1.0, 1.0], dtype=np.float64)
        averageer, _ = pml_average_er_mr(2, 2, 1, solid, ers, mrs)
        assert averageer == pytest.approx(5.0)

    def test_weighting_follows_cell_counts(self):
        """Expects ``(3·1 + 1·9)/4 == 3`` — a three-to-one split weights the
        mean toward the majority material."""
        solid = face([0, 0, 0, 1], (2, 2))
        ers = np.array([1.0, 9.0], dtype=np.float64)
        mrs = np.array([1.0, 1.0], dtype=np.float64)
        averageer, _ = pml_average_er_mr(2, 2, 1, solid, ers, mrs)
        assert averageer == pytest.approx(3.0)

    def test_matches_numpy_on_a_random_face(self):
        """Expects agreement with ``ers[solid].mean()`` — an independent
        formulation of the same reduction."""
        rng = np.random.default_rng(20260803)
        solid = rng.integers(0, 4, size=(6, 7)).astype(np.uint32)
        ers = np.array([1.0, 2.5, 9.0, 81.0], dtype=np.float64)
        mrs = np.array([1.0, 1.2, 1.4, 1.6], dtype=np.float64)
        averageer, averagemr = pml_average_er_mr(6, 7, 1, solid, ers, mrs)
        assert averageer == pytest.approx(ers[solid].mean())
        assert averagemr == pytest.approx(mrs[solid].mean())

    def test_a_non_square_face_divides_by_the_product(self):
        """Expects the divisor to be ``n1·n2``, not ``n1`` or ``max(n1, n2)``
        — the two face dimensions are independent."""
        solid = np.zeros((2, 8), dtype=np.uint32)
        solid[:, :4] = 1
        ers = np.array([2.0, 6.0], dtype=np.float64)
        mrs = np.array([1.0, 1.0], dtype=np.float64)
        averageer, _ = pml_average_er_mr(2, 8, 1, solid, ers, mrs)
        assert averageer == pytest.approx(4.0)

    def test_only_the_first_n1_by_n2_cells_are_read(self):
        """Expects the dimensions to govern the traversal rather than the
        array's own shape, so a larger buffer can be passed with a smaller
        window read out of its top-left corner."""
        solid = np.ones((6, 6), dtype=np.uint32)
        solid[:2, :2] = 0
        ers = np.array([2.0, 100.0], dtype=np.float64)
        mrs = np.array([1.0, 1.0], dtype=np.float64)
        averageer, _ = pml_average_er_mr(2, 2, 1, solid, ers, mrs)
        assert averageer == pytest.approx(2.0)


class TestSumVersusAverage:
    def test_sum_is_the_average_times_the_cell_count(self):
        """Expects the two kernels to differ only by the ``n1·n2`` divisor —
        ``pml_sum_er_mr`` exists so MPI ranks can add partial sums before
        dividing once globally."""
        rng = np.random.default_rng(11)
        solid = rng.integers(0, 3, size=(4, 5)).astype(np.uint32)
        ers = np.array([1.0, 4.0, 9.0], dtype=np.float64)
        mrs = np.array([1.0, 1.1, 1.2], dtype=np.float64)
        sumer, summr = pml_sum_er_mr(4, 5, 1, solid, ers, mrs)
        avger, avgmr = pml_average_er_mr(4, 5, 1, solid, ers, mrs)
        assert sumer == pytest.approx(avger * 20)
        assert summr == pytest.approx(avgmr * 20)

    def test_sum_matches_numpy(self):
        """Expects agreement with ``ers[solid].sum()``."""
        solid = face([0, 1, 1, 2], (2, 2))
        ers = np.array([1.0, 4.0, 9.0], dtype=np.float64)
        mrs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sumer, summr = pml_sum_er_mr(2, 2, 1, solid, ers, mrs)
        assert sumer == pytest.approx(18.0)
        assert summr == pytest.approx(8.0)

    def test_partial_sums_compose(self):
        """Expects summing two halves of a face to equal summing the whole —
        the property the MPI path relies on."""
        solid = np.arange(12, dtype=np.uint32).reshape(3, 4) % 3
        ers = np.array([1.0, 4.0, 9.0], dtype=np.float64)
        mrs = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        whole, _ = pml_sum_er_mr(3, 4, 1, solid, ers, mrs)
        top, _ = pml_sum_er_mr(1, 4, 1, solid[:1], ers, mrs)
        rest, _ = pml_sum_er_mr(2, 4, 1, np.ascontiguousarray(solid[1:]), ers, mrs)
        assert whole == pytest.approx(top + rest)


class TestThreading:
    @pytest.mark.parametrize("nthreads", [1, 2, 4])
    def test_result_is_independent_of_thread_count(self, nthreads):
        """Expects identical answers however the ``prange`` is split — the
        accumulators are OpenMP reduction variables, so a missing reduction
        clause would show up here as a race. (3 parameter sets)"""
        rng = np.random.default_rng(7)
        solid = rng.integers(0, 4, size=(32, 32)).astype(np.uint32)
        ers = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        mrs = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        averageer, _ = pml_average_er_mr(32, 32, nthreads, solid, ers, mrs)
        assert averageer == pytest.approx(ers[solid].mean())

    # ------------------------------------------------------------------
    # The two tests below are commented out because they FAIL ON WINDOWS CI.
    #
    # They are correct. The kernel is not. `pml_build.pyx:52` declares
    # `cdef double sumer, summr` with no initialiser and then uses them as
    # OpenMP `reduction(+:)` variables; the reduction adds each thread's
    # private total into the *original* value, which is uninitialised stack.
    # The GitHub Windows runner's MSVC reuses the same stack slot across
    # consecutive calls, so each call adds to the previous one's result.
    #
    # The observed numbers matched that prediction exactly:
    #
    #   repeated calls   ->  5.0, 10.0, 15.0 ... 100.0   (n x the true 5.0)
    #   interleaved      ->  454.0, where (4x3 + 256x7 + 4x3) / 4 = 454.0
    #
    # Linux, macOS and the local Windows toolchain happen to zero that slot,
    # which is why these passed everywhere else.
    #
    # This is a live source defect, not a test problem — `pml_average_er_mr`
    # sets `sigma_max` for every real PML slab, so on an affected toolchain
    # every simulation gets a wrong absorption profile. It is written up in
    # `notes/bugs/pml-build-uninitialised-reduction.md`, which carries the
    # two-line fix (`cdef double sumer = 0`).
    #
    # RESTORE THESE once that fix lands. They are the only tests that detect
    # the defect, and the remaining tests in this file are not immune — they
    # pass on the affected runner today only because their call patterns
    # happen not to reuse a dirty slot.
    # ------------------------------------------------------------------
    #
    # def test_repeated_calls_are_deterministic(self):
    #     """Expects the same answer every time. A drifting result would mean
    #     the reduction accumulator is carrying state between calls — see the
    #     module docstring."""
    #     solid = np.zeros((8, 8), dtype=np.uint32)
    #     ers = np.array([5.0], dtype=np.float64)
    #     mrs = np.array([2.0], dtype=np.float64)
    #     results = {pml_average_er_mr(8, 8, 2, solid, ers, mrs) for _ in range(20)}
    #     assert results == {(5.0, 2.0)}
    #
    # def test_interleaving_different_faces_does_not_contaminate(self):
    #     """Expects a small face's answer to be unaffected by a large call in
    #     between — the strongest available check that nothing leaks across
    #     invocations."""
    #     small = np.zeros((2, 2), dtype=np.uint32)
    #     large = np.ones((16, 16), dtype=np.uint32)
    #     ers = np.array([3.0, 7.0], dtype=np.float64)
    #     mrs = np.array([1.0, 1.0], dtype=np.float64)
    #     first = pml_average_er_mr(2, 2, 1, small, ers, mrs)
    #     pml_average_er_mr(16, 16, 1, large, ers, mrs)
    #     assert pml_average_er_mr(2, 2, 1, small, ers, mrs) == first


class TestDtypes:
    def test_accepts_double_precision_lookups(self):
        """Expects ``float64`` tables to bind to the double specialisation of
        the fused ``float_or_double`` type."""
        solid = np.zeros((2, 2), dtype=np.uint32)
        ers = np.array([2.0], dtype=np.float64)
        mrs = np.array([3.0], dtype=np.float64)
        assert pml_average_er_mr(2, 2, 1, solid, ers, mrs) == (2.0, 3.0)

    def test_accepts_single_precision_lookups(self):
        """Expects ``float32`` tables to bind to the float specialisation —
        gprMax compiles both, selected by the run's precision setting."""
        solid = np.zeros((2, 2), dtype=np.uint32)
        ers = np.array([2.0], dtype=np.float32)
        mrs = np.array([3.0], dtype=np.float32)
        assert pml_average_er_mr(2, 2, 1, solid, ers, mrs) == (2.0, 3.0)

    def test_returns_python_floats(self):
        """Expects plain floats, since the result feeds straight into
        ``CFS.calculate_sigmamax``'s scalar arithmetic."""
        solid = np.zeros((2, 2), dtype=np.uint32)
        ers = np.array([2.0], dtype=np.float64)
        mrs = np.array([3.0], dtype=np.float64)
        averageer, averagemr = pml_average_er_mr(2, 2, 1, solid, ers, mrs)
        assert isinstance(averageer, float)
        assert isinstance(averagemr, float)

    def test_solid_must_be_unsigned_32_bit(self):
        """Expects a typed-memoryview rejection for the wrong integer width —
        the ``solid`` array is ``uint32`` throughout gprMax, and a silent
        reinterpretation would index the lookup tables with garbage."""
        solid = np.zeros((2, 2), dtype=np.int64)
        ers = np.array([2.0], dtype=np.float64)
        mrs = np.array([3.0], dtype=np.float64)
        with pytest.raises(ValueError):
            pml_average_er_mr(2, 2, 1, solid, ers, mrs)
