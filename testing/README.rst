============================
Extended testing and studies
============================

The material here complements the focused automated test suite in ``tests``.
The categories identify what each result can demonstrate:

``validation``
    End-to-end gprMax results compared with independent analytical solutions.
    These cases may define quantitative correctness criteria and PASS/FAIL
    thresholds.

``other_codes``
    Numerical inter-code comparisons. Neither solver is ground truth, so
    these report differences rather than claim correctness validation.

``backend_consistency``
    Comparisons across compute backends, precision modes, orientations, or
    equivalent configurations. These detect inconsistencies but cannot prove
    that all compared implementations are physically correct.

``regression``
    Larger behavioural and diagnostic matrices without an independent
    analytical solution.

``benchmarking``
    Performance, memory, and scaling measurements.

``models_basic`` and ``models_pmls``
    Legacy compact model collections used by the historical regression
    workflow. They can be migrated into the categories above as that workflow
    is modernised.

``experimental``
    Experimental-measurement studies and associated models.
