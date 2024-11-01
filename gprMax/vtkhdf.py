import logging
from abc import abstractmethod
from contextlib import AbstractContextManager
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Literal, Optional, Union

import h5py
import numpy as np
import numpy.typing as npt
from mpi4py import MPI

logger = logging.getLogger(__name__)


class VtkHdfFile(AbstractContextManager):
    VERSION = [2, 2]
    FILE_EXTENSION = ".vtkhdf"
    ROOT_GROUP = "VTKHDF"

    @property
    @abstractmethod
    def TYPE(self) -> str:
        pass

    def __init__(self, filename: Union[str, PathLike], comm: Optional[MPI.Comm] = None) -> None:
        """Create a new VtkHdfFile.

        If the file already exists, it will be overriden. Required
        attributes (Type and Version) will be written to the file.

        The file will be opened using the 'mpio' h5py driver if an MPI
        communicator is provided.

        Args:
            filename: Name of the file (can be a file path). The file
                extension will be set to '.vtkhdf'.
            comm (optional): MPI communicator containing all ranks that
                want to write to the file.

        """
        # Ensure the filename uses the correct extension
        self.filename = Path(filename)
        if self.filename.suffix != "" and self.filename.suffix != self.FILE_EXTENSION:
            logger.warning(
                f"Invalid file extension '{self.filename.suffix}' for VTKHDF file. Changing to '{self.FILE_EXTENSION}'."
            )

        self.filename = self.filename.with_suffix(self.FILE_EXTENSION)

        self.comm = comm

        # Check if the filehandler should use an MPI driver
        if self.comm is None:
            self.file_handler = h5py.File(self.filename, "w")
        else:
            self.file_handler = h5py.File(self.filename, "w", driver="mpio", comm=self.comm)

        self.root_group = self.file_handler.create_group(self.ROOT_GROUP)

        # Set required Version and Type root attributes
        self._set_root_attribute("Version", self.VERSION)

        type_as_ascii = self.TYPE.encode("ascii")
        self._set_root_attribute(
            "Type", type_as_ascii, h5py.string_dtype("ascii", len(type_as_ascii))
        )

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        """Close the file when the context is exited.

        The parameters describe the exception that caused the context to
        be exited. If the context was exited without an exception, all
        three arguments will be None. Any exception will be
        processed normally upon exit from this method.

        Returns:
            suppress_exception (optional): Returns True if the exception
                should be suppressed (i.e. not propagated). Otherwise,
                the exception will be processed normally upon exit from
                this method.
        """
        self.close()

    def close(self) -> None:
        """Close the file handler"""
        self.file_handler.close()

    def _get_root_attribute(self, attribute: str) -> npt.NDArray:
        """Get attribute from the root VTKHDF group if it exists.

        Args:
            attribute: Name of the attribute.

        Raises:
            KeyError: Raised if the attribute is not present as a key.
        """
        value = self.root_group.attrs[attribute]
        if isinstance(value, h5py.Empty):
            raise KeyError(f"Attribute '{attribute}' not present in /{self.ROOT_GROUP} group")
        return value

    def _set_root_attribute(
        self, attribute: str, value: npt.ArrayLike, dtype: npt.DTypeLike = None
    ):
        """Set attribute in the root VTKHDF group.

        Args:
            attribute: Name of the new attribute.
            value: An array to initialize the attribute.
            dtype (optional): Data type of the attribute. Overrides
                value.dtype if both are given.
        """
        self.root_group.attrs.create(attribute, value, dtype=dtype)

    def _build_dataset_path(self, *path: str) -> str:
        """Build an HDF5 dataset path attached to the root VTKHDF group.

        Args:
            *path: Components of the required path.
        """
        return "/".join([self.ROOT_GROUP, *path])

    def _write_dataset(
        self,
        path: str,
        data: npt.NDArray,
        shape: Optional[npt.NDArray[np.intc]] = None,
        offset: Optional[npt.NDArray[np.intc]] = None,
    ):
        """Write the specified dataset to the file.

        Args:
            path: Absolute path to the dataset.
            data: Data to initialize the dataset.
            shape (optional): Size of the full dataset being created.
                Can be omitted if data provides the full dataset.
            offset (optional): Offset to store the provided data at. Can
                be omitted if data provides the full dataset.
        """

        # VTKHDF stores datasets using ZYX ordering rather than XYZ
        data = data.transpose()

        if shape is not None:
            shape = np.flip(shape)

        if offset is not None:
            offset = np.flip(offset)

        if shape is None or all(shape == data.shape):
            self.file_handler.create_dataset(path, data=data)
        else:
            dimensions = len(data.shape)
            if offset is None:
                offset = np.zeros(dimensions, dtype=np.intc)

            if dimensions != len(shape):
                raise ValueError(
                    "The data and specified shape must have the same number of dimensions."
                    f" {dimensions} != {len(shape)}"
                )

            if dimensions != len(offset):
                raise ValueError(
                    "The data and specified offset must have the same number of dimensions."
                    f" {dimensions} != {len(offset)}"
                )

            if any(offset + data.shape > shape):
                raise ValueError(
                    "The provided offset and data does not fit within the bounds of the dataset."
                    f" {offset} + {data.shape} = {offset + data.shape} > {shape}"
                )

            dataset = self.file_handler.create_dataset(path, shape, data.dtype)

            start = offset
            stop = offset + data.shape

            dataset_slice = (slice(start[i], stop[i]) for i in range(dimensions))

            dataset[dataset_slice] = data

    def add_point_data(
        self,
        name: str,
        data: npt.NDArray,
        shape: Optional[npt.NDArray[np.intc]] = None,
        offset: Optional[npt.NDArray[np.intc]] = None,
    ):
        dataset_path = self._build_dataset_path("PointData", name)
        self._write_dataset(dataset_path, data, shape, offset)

    def add_field_data(
        self,
        name: str,
        data: npt.NDArray,
        shape: Optional[npt.NDArray[np.intc]] = None,
        offset: Optional[npt.NDArray[np.intc]] = None,
    ):
        dataset_path = self._build_dataset_path("FieldData", name)
        self._write_dataset(dataset_path, data, shape, offset)

    def add_cell_data(
        self,
        name: str,
        data: npt.NDArray,
        shape: Optional[npt.NDArray[np.intc]] = None,
        offset: Optional[npt.NDArray[np.intc]] = None,
    ):
        dataset_path = self._build_dataset_path("CellData", name)
        self._write_dataset(dataset_path, data, shape, offset)


class VtkImageData(VtkHdfFile):
    DIRECTION_ATTR = "Direction"
    ORIGIN_ATTR = "Origin"
    SPACING_ATTR = "Spacing"
    WHOLE_EXTENT_ATTR = "WholeExtent"

    @property
    def TYPE(self) -> Literal["ImageData"]:
        return "ImageData"

    def __init__(
        self,
        filename: Union[str, PathLike],
        shape: npt.NDArray[np.intc],
        origin: Optional[npt.NDArray[np.single]] = None,
        spacing: Optional[npt.NDArray[np.single]] = None,
        direction: Optional[npt.NDArray[np.single]] = None,
        comm: Optional[MPI.Cartcomm] = None,
    ) -> None:
        super().__init__(filename, comm)

        self.shape = shape
        self.points_shape = shape + 1
        whole_extent = np.zeros(2 * len(self.shape), dtype=np.intc)
        whole_extent[1::2] = self.shape
        self._set_root_attribute(self.WHOLE_EXTENT_ATTR, whole_extent)

        if origin is None:
            origin = np.zeros(len(self.shape), dtype=np.single)
        self.set_origin(origin)

        if spacing is None:
            spacing = np.ones(len(self.shape), dtype=np.single)
        self.set_spacing(spacing)

        if direction is None:
            direction = np.diag(np.ones(len(self.shape), dtype=np.single)).flatten()
        self.set_direction(direction)

    @property
    def whole_extent(self) -> npt.NDArray[np.intc]:
        return self._get_root_attribute(self.WHOLE_EXTENT_ATTR)

    @property
    def origin(self) -> npt.NDArray[np.single]:
        return self._get_root_attribute(self.ORIGIN_ATTR)

    @property
    def spacing(self) -> npt.NDArray[np.single]:
        return self._get_root_attribute(self.SPACING_ATTR)

    @property
    def direction(self) -> npt.NDArray[np.single]:
        return self._get_root_attribute(self.DIRECTION_ATTR)

    def set_origin(self, origin: npt.NDArray[np.single]):
        self._set_root_attribute(self.ORIGIN_ATTR, origin)

    def set_spacing(self, spacing: npt.NDArray[np.single]):
        self._set_root_attribute(self.SPACING_ATTR, spacing)

    def set_direction(self, direction: npt.NDArray[np.single]):
        self._set_root_attribute(self.DIRECTION_ATTR, direction)

    def add_point_data(
        self, name: str, data: npt.NDArray, offset: Optional[npt.NDArray[np.intc]] = None
    ):
        if offset is None and any(data.shape != self.points_shape):  # type: ignore
            raise ValueError(
                f"If no offset is specified, data.shape {data.shape} must match the shape of the"
                f" VtkImageData point datasets {self.points_shape}"
            )
        return super().add_point_data(name, data, self.points_shape, offset)

    def add_cell_data(
        self, name: str, data: npt.NDArray, offset: Optional[npt.NDArray[np.intc]] = None
    ):
        if offset is None and any(data.shape != self.shape):  # type: ignore
            raise ValueError(
                f"If no offset is specified, data.shape {data.shape} must match the shape of the"
                f" VtkImageData {self.shape}"
            )
        return super().add_cell_data(name, data, self.shape, offset)
