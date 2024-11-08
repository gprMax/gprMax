import logging
from abc import abstractmethod
from contextlib import AbstractContextManager
from enum import Enum
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

    # TODO: Can this be moved to using an Enum like root datasets?
    # Main barrier: Can't subclass an enum with members and any base
    # Enum class would need VERSION and TYPE as members.
    VERSION_ATTR = "Version"
    TYPE_ATTR = "Type"

    class Dataset(str, Enum):
        pass

    @property
    @abstractmethod
    def TYPE(self) -> str:
        pass

    def __enter__(self):
        return self

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
        self._set_root_attribute(self.VERSION_ATTR, self.VERSION)

        type_as_ascii = self.TYPE.encode("ascii")
        self._set_root_attribute(
            self.TYPE_ATTR, type_as_ascii, h5py.string_dtype("ascii", len(type_as_ascii))
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

        Returns:
            value: Current value of the attribute if it exists.

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

        Returns:
            path: Path to the dataset.
        """
        return "/".join([self.ROOT_GROUP, *path])

    def _get_root_dataset(self, name: "VtkHdfFile.Dataset") -> h5py.Dataset:
        """Get specified dataset from the root group of the VTKHDF file.

        Args:
            path: Name of the dataset.

        Returns:
            dataset: Returns specified h5py dataset.
        """
        path = self._build_dataset_path(name)
        return self._get_dataset(path)

    def _get_dataset(self, path: str) -> h5py.Dataset:
        """Get specified dataset.

        Args:
            path: Absolute path to the dataset.

        Returns:
            dataset: Returns specified h5py dataset.

        Raises:
            KeyError: Raised if the dataset does not exist, or the path
                points to some other object, e.g. a Group not a Dataset.
        """
        cls = self.file_handler.get(path, getclass=True)
        if cls == "default":
            raise KeyError("Path does not exist")
        elif cls != h5py.Dataset:
            raise KeyError(f"Dataset not found. Found '{cls}' instead")

        dataset = self.file_handler.get(path)
        assert isinstance(dataset, h5py.Dataset)
        return dataset

    def _write_root_dataset(
        self,
        name: "VtkHdfFile.Dataset",
        data: npt.ArrayLike,
        shape: Optional[npt.NDArray[np.intc]] = None,
        offset: Optional[npt.NDArray[np.intc]] = None,
        xyz_data_ordering=True,
    ):
        """Write specified dataset to the root group of the VTKHDF file.

        Args:
            name: Name of the dataset.
            data: Data to initialize the dataset.
            shape (optional): Size of the full dataset being created.
                Can be omitted if data provides the full dataset.
            offset (optional): Offset to store the provided data at. Can
                be omitted if data provides the full dataset.
            xyz_data_ordering (optional): If True, the data will be
                transposed as VTKHDF stores datasets using ZYX ordering.
                Default True.
        """
        path = self._build_dataset_path(name)
        self._write_dataset(path, data, shape, offset, xyz_data_ordering)

    def _write_dataset(
        self,
        path: str,
        data: npt.ArrayLike,
        shape: Optional[npt.NDArray[np.intc]] = None,
        offset: Optional[npt.NDArray[np.intc]] = None,
        xyz_data_ordering=True,
    ):
        """Write specified dataset to the VTKHDF file.

        If data has shape (d1, d2, ..., dn), i.e. n dimensions, then, if
        specified, shape and offset must be of length n.

        Args:
            path: Absolute path to the dataset.
            data: Data to initialize the dataset.
            shape (optional): Size of the full dataset being created.
                Can be omitted if data provides the full dataset.
            offset (optional): Offset to store the provided data at. Can
                be omitted if data provides the full dataset.
            xyz_data_ordering (optional): If True, the data will be
                transposed as VTKHDF stores datasets using ZYX ordering.
                Default True.

        Raises:
            ValueError: Raised if the combination of data.shape, shape,
                and offset are invalid.
        """

        if not isinstance(data, np.ndarray):
            data = np.array([data])

        # VTKHDF stores datasets using ZYX ordering rather than XYZ
        if xyz_data_ordering:
            data = data.transpose()

        if shape is not None:
            shape = np.flip(shape)

        if offset is not None:
            offset = np.flip(offset)

        if shape is None or all(shape == data.shape):
            self.file_handler.create_dataset(path, data=data)
        elif offset is None:
            raise ValueError(
                "Offset must not be None as the full dataset has not been provided."
                " I.e. data.shape != shape"
            )
        else:
            dimensions = len(data.shape)

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
        """Add point data to the VTKHDF file.

        Args:
            name: Name of the dataset.
            data: Data to be saved.
            shape (optional): Size of the full dataset being created.
                Can be omitted if data provides the full dataset.
            offset (optional): Offset to store the provided data at. Can
                be omitted if data provides the full dataset.
        """
        dataset_path = self._build_dataset_path("PointData", name)
        self._write_dataset(dataset_path, data, shape, offset)

    def add_cell_data(
        self,
        name: str,
        data: npt.NDArray,
        shape: Optional[npt.NDArray[np.intc]] = None,
        offset: Optional[npt.NDArray[np.intc]] = None,
    ):
        """Add cell data to the VTKHDF file.

        Args:
            name: Name of the dataset.
            data: Data to be saved.
            shape (optional): Size of the full dataset being created.
                Can be omitted if data provides the full dataset.
            offset (optional): Offset to store the provided data at. Can
                be omitted if data provides the full dataset.
        """
        dataset_path = self._build_dataset_path("CellData", name)
        self._write_dataset(dataset_path, data, shape, offset)

    def add_field_data(
        self,
        name: str,
        data: npt.NDArray,
        shape: Optional[npt.NDArray[np.intc]] = None,
        offset: Optional[npt.NDArray[np.intc]] = None,
    ):
        """Add field data to the VTKHDF file.

        Args:
            name: Name of the dataset.
            data: Data to be saved.
            shape (optional): Size of the full dataset being created.
                Can be omitted if data provides the full dataset.
            offset (optional): Offset to store the provided data at. Can
                be omitted if data provides the full dataset.
        """
        dataset_path = self._build_dataset_path("FieldData", name)
        self._write_dataset(dataset_path, data, shape, offset)


class VtkImageData(VtkHdfFile):
    """File handler for creating a VTKHDF Image Data file.

    File format information is available here:
    https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#image-data
    """

    DIRECTION_ATTR = "Direction"
    ORIGIN_ATTR = "Origin"
    SPACING_ATTR = "Spacing"
    WHOLE_EXTENT_ATTR = "WholeExtent"

    DIMENSIONS = 3

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
        comm: Optional[MPI.Comm] = None,
    ):
        """Create a new VtkImageData file.

        If the file already exists, it will be overriden. Required
        attributes (Type and Version) will be written to the file.

        The file will be opened using the 'mpio' h5py driver if an MPI
        communicator is provided.

        Args:
            filename: Name of the file (can be a file path). The file
                extension will be set to '.vtkhdf'.
            shape: Shape of the image data to be stored in the file.
                This specifies the number of cells. Image data can be
                1D, 2D, or 3D.
            origin (optional): Origin of the image data. Default
                [0, 0, 0].
            spacing (optional): Discritisation of the image data.
                Default [1, 1, 1].
            direction (optional): Array of direction vectors for each
                dimension of the image data. Can be a flattened array.
                I.e. [[1, 0, 0], [0, 1, 0], [0, 0, 1]] and
                [1, 0, 0, 0, 1, 0, 0, 0, 1] are equivalent. Default
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]].
            comm (optional): MPI communicator containing all ranks that
                want to write to the file.
        """
        super().__init__(filename, comm)

        if len(shape) == 0:
            raise ValueError(f"Shape must not be empty.")
        if len(shape) > self.DIMENSIONS:
            raise ValueError(f"Shape must not have more than {self.DIMENSIONS} dimensions.")
        elif len(shape) < self.DIMENSIONS:
            shape = np.concatenate((shape, np.ones(self.DIMENSIONS - len(shape), dtype=np.intc)))

        self.shape = shape

        whole_extent = np.zeros(2 * self.DIMENSIONS, dtype=np.intc)
        whole_extent[1::2] = self.shape
        self._set_root_attribute(self.WHOLE_EXTENT_ATTR, whole_extent)

        if origin is None:
            origin = np.zeros(self.DIMENSIONS, dtype=np.single)
        self.set_origin(origin)

        if spacing is None:
            spacing = np.ones(self.DIMENSIONS, dtype=np.single)
        self.set_spacing(spacing)

        if direction is None:
            direction = np.diag(np.ones(self.DIMENSIONS, dtype=np.single))
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
        """Set the origin coordinate of the image data.

        Args:
            origin: x, y, z coordinates to set as the origin.
        """
        if len(origin) != self.DIMENSIONS:
            raise ValueError(f"Origin attribute must have {self.DIMENSIONS} dimensions.")
        self._set_root_attribute(self.ORIGIN_ATTR, origin)

    def set_spacing(self, spacing: npt.NDArray[np.single]):
        """Set the discritisation of the image data.

        Args:
            spacing: Discritisation of the x, y, and z dimensions.
        """
        if len(spacing) != self.DIMENSIONS:
            raise ValueError(f"Spacing attribute must have {self.DIMENSIONS} dimensions.")
        self._set_root_attribute(self.SPACING_ATTR, spacing)

    def set_direction(self, direction: npt.NDArray[np.single]):
        """Set the coordinate system of the image data.

        Args:
            direction: Array of direction vectors for each dimension of
                the image data. Can be a flattened array. I.e.
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]] and
                [1, 0, 0, 0, 1, 0, 0, 0, 1] are equivalent.
        """
        direction = direction.flatten()
        if len(direction) != self.DIMENSIONS * self.DIMENSIONS:
            raise ValueError(
                f"Direction array must contain {self.DIMENSIONS * self.DIMENSIONS} elements."
            )
        self._set_root_attribute(self.DIRECTION_ATTR, direction)

    def add_point_data(
        self, name: str, data: npt.NDArray, offset: Optional[npt.NDArray[np.intc]] = None
    ):
        """Add point data to the VTKHDF file.

        Args:
            name: Name of the dataset.
            data: Data to be saved.
            offset (optional): Offset to store the provided data at. Can
                be omitted if data provides the full dataset.

        Raises:
            ValueError: Raised if data has invalid dimensions.
        """
        points_shape = self.shape + 1
        if offset is None and any(data.shape != points_shape):  # type: ignore
            raise ValueError(
                "If no offset is specified, data.shape must be one larger in each dimension than"
                f" this vtkImageData object. {data.shape} != {points_shape}"
            )
        return super().add_point_data(name, data, points_shape, offset)

    def add_cell_data(
        self, name: str, data: npt.NDArray, offset: Optional[npt.NDArray[np.intc]] = None
    ):
        """Add cell data to the VTKHDF file.

        Args:
            name: Name of the dataset.
            data: Data to be saved.
            offset (optional): Offset to store the provided data at. Can
                be omitted if data provides the full dataset.

        Raises:
            ValueError: Raised if data has invalid dimensions.
        """
        if offset is None and any(data.shape != self.shape):  # type: ignore
            raise ValueError(
                "If no offset is specified, data.shape must match the dimensions of this"
                f" VtkImageData object. {data.shape} != {self.shape}"
            )
        return super().add_cell_data(name, data, self.shape, offset)


class VtkCellType(np.uint8, Enum):
    """VTK cell types as defined here:
    https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html#l00019
    """

    # Linear cells
    EMPTY_CELL = 0
    VERTEX = 1
    POLY_VERTEX = 2
    LINE = 3
    POLY_LINE = 4
    TRIANGLE = 5
    TRIANGLE_STRIP = 6
    POLYGON = 7
    PIXEL = 8
    QUAD = 9
    TETRA = 10
    VOXEL = 11
    HEXAHEDRON = 12
    WEDGE = 13
    PYRAMID = 14
    PENTAGONAL_PRISM = 15
    HEXAGONAL_PRISM = 16


class VtkUnstructuredGrid(VtkHdfFile):
    """File handler for creating a VTKHDF Unstructured Grid file.

    File format information is available here:
    https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#unstructured-grid
    """

    class Dataset(VtkHdfFile.Dataset):
        CONNECTIVITY = "Connectivity"
        NUMBER_OF_CELLS = "NumberOfCells"
        NUMBER_OF_CONNECTIVITY_IDS = "NumberOfConnectivityIds"
        NUMBER_OF_POINTS = "NumberOfPoints"
        OFFSETS = "Offsets"
        POINTS = "Points"
        TYPES = "Types"

    @property
    def TYPE(self) -> Literal["UnstructuredGrid"]:
        return "UnstructuredGrid"

    def __init__(
        self,
        filename: Union[str, PathLike],
        points: npt.NDArray,
        cell_types: npt.NDArray[VtkCellType],
        connectivity: npt.NDArray,
        cell_offsets: npt.NDArray,
        comm: Optional[MPI.Comm] = None,
    ) -> None:
        """Create a new VtkUnstructuredGrid file.

        An unstructured grid has N points and C cells. A cell is defined
        as a collection of points which is specified by the connectivity
        and cell_offsets arguments along with the list of cell_types.

        If the file already exists, it will be overriden. Required
        attributes (Type and Version) will be written to the file.

        The file will be opened using the 'mpio' h5py driver if an MPI
        communicator is provided.

        Args:
            filename: Name of the file (can be a file path). The file
                extension will be set to '.vtkhdf'.
            points: Array of point coordinates of shape (N, 3).
            cell_types: Array of VTK cell types of shape (C,).
            connectivity: Array of point IDs that together with
                cell_offsets, defines the points that make up each cell.
                Each point ID has a value between 0 and (N-1) inclusive
                and corresponds to a point in the points array.
            cell_offsets: Array listing where each cell starts and ends
                in the connectivity array. It has shape (C + 1,).
            comm (optional): MPI communicator containing all ranks that
                want to write to the file.

        Raises:
            Value Error: Raised if argument dimensions are invalid.
        """
        super().__init__(filename, comm)

        if len(cell_offsets) != len(cell_types) + 1:
            raise ValueError(
                "cell_offsets should be one longer than cell_types."
                " I.e. one longer than the number of cells"
            )

        is_sorted = lambda a: np.all(a[:-1] <= a[1:])
        if not is_sorted(cell_offsets):
            raise ValueError("cell_offsets should be sorted in ascending order")

        if len(connectivity) < cell_offsets[-1]:
            raise ValueError("Connectivity array is shorter than final cell_offsets value")

        elif len(connectivity) > cell_offsets[-1]:
            raise logger.warning(
                "Connectivity array longer than final cell_offsets value."
                " Some connectivity data will be ignored"
            )

        self._write_root_dataset(self.Dataset.CONNECTIVITY, connectivity)
        self._write_root_dataset(self.Dataset.NUMBER_OF_CELLS, len(cell_types))
        self._write_root_dataset(self.Dataset.NUMBER_OF_CONNECTIVITY_IDS, len(connectivity))
        self._write_root_dataset(self.Dataset.NUMBER_OF_POINTS, len(points))
        self._write_root_dataset(self.Dataset.OFFSETS, cell_offsets)
        self._write_root_dataset(self.Dataset.POINTS, points, xyz_data_ordering=False)
        self._write_root_dataset(self.Dataset.TYPES, cell_types)

    @property
    def number_of_cells(self) -> int:
        number_of_cells = self._get_root_dataset(self.Dataset.NUMBER_OF_CELLS)
        return np.sum(number_of_cells, dtype=np.intc)

    @property
    def number_of_connectivity_ids(self) -> int:
        number_of_connectivity_ids = self._get_root_dataset(self.Dataset.NUMBER_OF_CONNECTIVITY_IDS)
        return np.sum(number_of_connectivity_ids, dtype=np.intc)

    @property
    def number_of_points(self) -> int:
        number_of_points = self._get_root_dataset(self.Dataset.NUMBER_OF_POINTS)
        return np.sum(number_of_points, dtype=np.intc)

    def add_point_data(
        self, name: str, data: npt.NDArray, offset: Optional[npt.NDArray[np.intc]] = None
    ):
        """Add point data to the VTKHDF file.

        Args:
            name: Name of the dataset.
            data: Data to be saved.
            offset (optional): Offset to store the provided data at. Can
                be omitted if data provides the full dataset.

        Raises:
            ValueError: Raised if data has invalid dimensions.
        """
        shape = np.array(data.shape)
        number_of_dimensions = len(shape)

        if number_of_dimensions < 1 or number_of_dimensions > 2:
            raise ValueError(f"Data must have 1 or 2 dimensions, not {number_of_dimensions}")
        elif len(data) != self.number_of_points:
            raise ValueError(
                "Length of data must match the number of points in the vtkUnstructuredGrid."
                f" {len(data)} != {self.number_of_points}"
            )
        elif number_of_dimensions == 2 and shape[1] != 1 and shape[1] != 3:
            raise ValueError(f"The second dimension should have shape 1 or 3, not {shape[1]}")

        return super().add_point_data(name, data, shape, offset)

    def add_cell_data(
        self, name: str, data: npt.NDArray, offset: Optional[npt.NDArray[np.intc]] = None
    ):
        """Add cell data to the VTKHDF file.

        Args:
            name: Name of the dataset.
            data: Data to be saved.
            offset (optional): Offset to store the provided data at. Can
                be omitted if data provides the full dataset.

        Raises:
            ValueError: Raised if data has invalid dimensions.
        """
        shape = np.array(data.shape)
        number_of_dimensions = len(shape)

        if number_of_dimensions < 1 or number_of_dimensions > 2:
            raise ValueError(f"Data must have 1 or 2 dimensions, not {number_of_dimensions}.")
        elif len(data) != self.number_of_cells:
            raise ValueError(
                "Length of data must match the number of cells in the vtkUnstructuredGrid."
                f" {len(data)} != {self.number_of_cells}"
            )
        elif number_of_dimensions == 2 and shape[1] != 1 and shape[1] != 3:
            raise ValueError(f"The second dimension should have shape 1 or 3, not {shape[1]}")

        return super().add_cell_data(name, data, shape, offset)
