from os import PathLike
from pathlib import Path
from shutil import copyfile
from typing import Literal, Optional, Union

import reframe.utility.sanity as sn
from reframe.core.runtime import runtime
from reframe.utility import osext


class RegressionCheck:
    """Compare two .h5 files using h5diff"""

    def __init__(
        self, output_file: Union[str, PathLike], reference_file: Union[str, PathLike]
    ) -> None:
        self.output_file = Path(output_file)
        self.reference_file = Path(reference_file)
        self.h5diff_options: list[str] = []

    @property
    def error_msg(self) -> str:
        return "Failed regression check"

    def create_reference_file(self) -> bool:
        if not sn.path_exists(self.reference_file):
            self.reference_file.parent.mkdir(parents=True, exist_ok=True)
            copyfile(self.output_file, self.reference_file)
            return True
        else:
            return False

    def reference_file_exists(self) -> bool:
        return sn.path_isfile(self.reference_file)

    def run(self) -> Literal[True]:
        if runtime().system.name == "archer2":
            h5diff = "/opt/cray/pe/hdf5/default/bin/h5diff"
        else:
            h5diff = "h5diff"

        h5diff_output = osext.run_command(
            [h5diff, *self.h5diff_options, str(self.output_file), str(self.reference_file)]
        )

        return sn.assert_true(
            sn.path_isfile(self.output_file),
            f"Expected output file '{self.output_file}' does not exist",
        ) and sn.assert_false(
            h5diff_output.stdout,
            (
                f"{self.error_msg}\n"
                # f"For more details run: 'h5diff {' '.join(self.h5diff_options)} {self.output_file} {self.reference_file}'\n"
                f"For more details run: '{' '.join(h5diff_output.args)}'\n"
                f"To re-create regression file, delete '{self.reference_file}' and rerun the test."
            ),
        )


class ReceiverRegressionCheck(RegressionCheck):
    def __init__(
        self,
        output_file: Union[str, PathLike],
        reference_file: Union[str, PathLike],
        output_receiver: Optional[str],
        reference_receiver: Optional[str] = None,
    ) -> None:
        super().__init__(output_file, reference_file)

        self.output_receiver = output_receiver
        self.reference_receiver = reference_receiver

        self.h5diff_options.append(f"rxs/{self.output_receiver}")
        if self.reference_receiver is not None:
            self.h5diff_options.append(f"rxs/{self.reference_receiver}")

    @property
    def error_msg(self) -> str:
        return f"Receiver '{self.output_receiver}' failed regression check"


class SnapshotRegressionCheck(RegressionCheck):
    @property
    def error_msg(self) -> str:
        return f"Snapshot '{self.output_file.name}' failed regression check "
