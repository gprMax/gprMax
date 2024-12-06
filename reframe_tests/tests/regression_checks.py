from os import PathLike
from pathlib import Path
from shutil import copyfile
from typing import Literal, Optional, Union

import reframe.utility.sanity as sn
from reframe.core.runtime import runtime
from reframe.utility import osext


class RegressionCheck:
    """Compare two files using diff"""

    def __init__(
        self, output_file: Union[str, PathLike], reference_file: Union[str, PathLike]
    ) -> None:
        """Create a new regression check.

        Args:
            output_file: Path to output file generate by the test.
            reference_file: Path to reference file to run the regression
                check against.
        """
        self.output_file = Path(output_file)
        self.reference_file = Path(reference_file)
        self.cmd = "diff"
        self.options: list[str] = []

    @property
    def error_msg(self) -> str:
        """Message to display if the regression check fails"""
        return "Failed regression check"

    def create_reference_file(self) -> bool:
        """Create reference file if it does not already exist.

        The reference file is created as a copy of the current output
        file.

        Returns:
            file_created: Returns True if a new file was created, False
                if the path already exists.
        """
        if not sn.path_exists(self.reference_file):
            self.reference_file.parent.mkdir(parents=True, exist_ok=True)
            copyfile(self.output_file, self.reference_file)
            return True
        else:
            return False

    def reference_file_exists(self) -> bool:
        """Check if the reference file exists.

        Returns:
            file_exists: Returns true if the reference filepath is a
                regular file, False otherwise.
        """
        return sn.path_isfile(self.reference_file)

    def run(self) -> Literal[True]:
        """Run the regression check.

        Returns:
            check_passed: Returns True if the output file matches the
                reference file (i.e. no output from diff). Otherwise,
                raises a SanityError.

        Raises:
            reframe.core.exceptions.SanityError: If the output file does
                not exist, or the regression check fails.
        """

        completed_process = osext.run_command(
            [
                self.cmd,
                *self.options,
                str(self.output_file.absolute()),
                str(self.reference_file),
            ]
        )

        return sn.assert_true(
            sn.path_isfile(self.output_file),
            f"Expected output file '{self.output_file}' does not exist",
        ) and sn.assert_false(
            completed_process.stdout,
            (
                f"{self.error_msg}\n"
                f"For more details run: '{' '.join(completed_process.args)}'\n"
                f"To re-create regression file, delete '{self.reference_file}' and rerun the test."
            ),
        )


class H5RegressionCheck(RegressionCheck):
    """Compare two hdf5 files using h5diff"""

    def __init__(
        self, output_file: Union[str, PathLike], reference_file: Union[str, PathLike]
    ) -> None:
        super().__init__(output_file, reference_file)
        if runtime().system.name == "archer2":
            self.cmd = "/opt/cray/pe/hdf5/default/bin/h5diff"
        else:
            self.cmd = "h5diff"


class ReceiverRegressionCheck(H5RegressionCheck):
    """Run regression check on individual reveivers in output files.

    This can include arbitrary receivers in each file, or two receivers
    in the same file.
    """

    def __init__(
        self,
        output_file: Union[str, PathLike],
        reference_file: Union[str, PathLike],
        output_receiver: str,
        reference_receiver: Optional[str] = None,
    ) -> None:
        """Create a new receiver regression check.

        Args:
            output_file: Path to output file generate by the test.
            reference_file: Path to reference file to run the regression
                check against.
            output_receiver: Output receiver to check.
            reference_receiver: Optional receiver to check against in
                the reference file. If None, this will be the same as
                the output receiver.
        """
        super().__init__(output_file, reference_file)

        self.output_receiver = output_receiver
        self.reference_receiver = reference_receiver

        self.options.append(f"rxs/{self.output_receiver}")
        if self.reference_receiver is not None:
            self.options.append(f"rxs/{self.reference_receiver}")

    @property
    def error_msg(self) -> str:
        return f"Receiver '{self.output_receiver}' failed regression check"


class SnapshotRegressionCheck(H5RegressionCheck):
    """Run regression check on a gprMax Snapshot."""

    @property
    def error_msg(self) -> str:
        return f"Snapshot '{self.output_file.name}' failed regression check"


class GeometryObjectRegressionCheck(H5RegressionCheck):
    """Run regression check on a GprMax GeometryObject."""

    @property
    def error_msg(self) -> str:
        return f"GeometryObject '{self.output_file.name}' failed regression check"


class GeometryObjectMaterialsRegressionCheck(RegressionCheck):
    """Run regression check on materials output by a GeometryObject."""

    @property
    def error_msg(self) -> str:
        return f"GeometryObject materials file '{self.output_file}' failed regression check"


class GeometryViewRegressionCheck(H5RegressionCheck):
    """Run regression check on a GprMax GeometryView."""

    @property
    def error_msg(self) -> str:
        return f"GeometryView '{self.output_file.name}' failed regression check"
