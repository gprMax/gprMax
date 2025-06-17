from pathlib import Path

from reframe.core.builtins import run_before

from reframe_tests.tests.base_tests import GprMaxBaseTest
from reframe_tests.tests.mixins import (
    GeometryObjectsReadMixin,
    GeometryObjectsWriteMixin,
    GeometryOnlyMixin,
    GeometryViewMixin,
    ReceiverMixin,
    SnapshotMixin,
)
from reframe_tests.tests.regression_checks import GeometryObjectMaterialsRegressionCheck


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
    @run_before("sanity")
    def update_material_files(self):
        checks = [
            check
            for check in self.regression_checks
            if isinstance(check, GeometryObjectMaterialsRegressionCheck)
        ]
        for check in checks:
            for geometry_object in self.geometry_objects_read.values():
                material_file = Path(self.stagedir, check.output_file)
                with open(material_file, "r") as f:
                    lines = f.readlines()

                with open(material_file, "w") as f:
                    for line in lines:
                        new_line = line.replace(f"{{{geometry_object}_materials}}", "")
                        f.write(new_line)
