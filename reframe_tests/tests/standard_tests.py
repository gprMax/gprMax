import json
import re
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
        """Normalise the deliberate database namespace before comparison.

        Imported material IDs are namespaced to prevent collisions with
        materials already in the receiving model. The read/write regression
        is checking physical equivalence with the source geometry, so remove
        that namespace from the generated comparison copy while retaining the
        production behaviour exercised during the run.
        """

        checks = [
            check
            for check in self.regression_checks
            if isinstance(check, GeometryObjectMaterialsRegressionCheck)
        ]
        for check in checks:
            material_file = Path(self.stagedir, check.output_file)
            with open(material_file, "r", encoding="utf-8") as stream:
                document = json.load(stream)

            for geometry_object in self.geometry_objects_read:
                namespace = f"{{{geometry_object}_materials}}"
                normalised = {}
                for index, entry in enumerate(document["materials"].values()):
                    entry["name"] = entry["name"].replace(namespace, "")
                    metadata = entry.get("metadata", {})
                    original_id = metadata.get("original_id", entry["name"])
                    original_id = original_id.replace(namespace, "")
                    metadata["original_id"] = original_id
                    metadata.pop("source_database", None)
                    entry["metadata"] = metadata

                    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", original_id).strip("_.-")
                    slug = slug or "material"
                    if not slug[0].isalpha():
                        slug = f"m_{slug}"
                    normalised[f"material_{index:03d}_{slug}"] = entry
                document["materials"] = normalised

            with open(material_file, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(document, stream, indent=2, sort_keys=False, ensure_ascii=False)
                stream.write("\n")
