from reframe_tests.tests.base_tests import GprMaxBaseTest
from reframe_tests.tests.mixins import (
    GeometryOnlyMixin,
    GeometryViewMixin,
    ReceiverMixin,
    SnapshotMixin,
)


class GprMaxRegressionTest(ReceiverMixin, GprMaxBaseTest):
    pass


class GprMaxSnapshotTest(SnapshotMixin, GprMaxBaseTest):
    pass


class GprMaxGeometryViewTest(GeometryViewMixin, GeometryOnlyMixin, GprMaxBaseTest):
    pass
