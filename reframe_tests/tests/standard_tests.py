from reframe_tests.tests.base_tests import GprMaxBaseTest
from reframe_tests.tests.mixins import (
    GeometryObjectsReadMixin,
    GeometryObjectsWriteMixin,
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


class GprMaxGeometryObjectsWriteTest(GeometryObjectsWriteMixin, GprMaxBaseTest):
    pass


class GprMaxGeometryObjectsReadTest(GeometryObjectsReadMixin, GprMaxBaseTest):
    pass


class GprMaxGeometryObjectsReadWriteTest(
    GeometryObjectsReadMixin, GeometryObjectsWriteMixin, GprMaxBaseTest
):
    pass


class GprMaxGeometryTest(GeometryObjectMixin, ReceiverMixin, GprMaxBaseTest):
    geometry_objects = ["full_volume"]
