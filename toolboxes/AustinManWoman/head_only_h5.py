import argparse
import logging
from pathlib import Path

import h5py

logger = logging.getLogger(__name__)

_SPACING_ATTRIBUTE = "dx_dy_dz"
_LEGACY_SPACING_ATTRIBUTE = "dx, dy, dz"


def _geometry_spacing(source, filename):
    """Read and validate current or legacy gprMax geometry spacing."""

    current = source.attrs.get(_SPACING_ATTRIBUTE)
    legacy = source.attrs.get(_LEGACY_SPACING_ATTRIBUTE)
    if current is None and legacy is None:
        raise ValueError(
            f"{filename} is not a gprMax geometry file with data and "
            f"{_SPACING_ATTRIBUTE} (or legacy '{_LEGACY_SPACING_ATTRIBUTE}')"
        )
    if current is not None and legacy is not None:
        if len(current) != len(legacy) or any(current != legacy):
            raise ValueError(f"{filename} contains inconsistent current and legacy grid spacings")
    return current if current is not None else legacy


def _dataset_creation_options(dataset, output_shape):
    """Return safe creation options for a sliced copy of an HDF5 dataset."""

    options = {}
    if dataset.chunks is not None:
        options["chunks"] = tuple(
            min(chunk, dimension) for chunk, dimension in zip(dataset.chunks, output_shape)
        )
    for name in ("compression", "compression_opts", "shuffle", "fletcher32", "scaleoffset"):
        value = getattr(dataset, name, None)
        if value is not None and value is not False:
            options[name] = value
    return options


def extract_head(filename, outputfile=None, first_head_plane=None):
    """Extract the upper part of an AustinMan/AustinWoman geometry file.

    By default the historical AustinMan/Woman convention is retained and the
    upper eighth is selected. ``first_head_plane`` can be supplied explicitly
    for another model version or anatomical extent. All root attributes and
    auxiliary datasets are preserved, including material-database keys.
    """
    filename = Path(filename)
    outputfile = (
        Path(outputfile)
        if outputfile is not None
        else filename.with_name(f"{filename.stem}_head.h5")
    )

    with h5py.File(filename, "r") as source:
        if "data" not in source:
            raise ValueError(f"{filename} is not a gprMax geometry file with a data dataset")
        spacing = _geometry_spacing(source, filename)
        data = source["data"]
        if data.ndim != 3:
            raise ValueError(f"Expected three-dimensional geometry data in {filename}")

        if first_head_plane is None:
            first_head_plane = 7 * (data.shape[2] // 8)
        if not isinstance(first_head_plane, int):
            raise TypeError("first_head_plane must be an integer")
        if first_head_plane < 0 or first_head_plane >= data.shape[2]:
            raise ValueError(
                f"first_head_plane must be between 0 and {data.shape[2] - 1}, "
                f"not {first_head_plane}"
            )
        output_shape = (*data.shape[:2], data.shape[2] - first_head_plane)
        logger.info(
            "Dimensions of head model: %d x %d x %d cells",
            *output_shape,
        )

        with h5py.File(outputfile, "w") as destination:
            for name, value in source.attrs.items():
                destination.attrs[name] = value
            # Austin v2.3 files use the historical ``dx, dy, dz`` name.
            # Always add the current spelling so the result is directly
            # readable by modern gprMax while retaining source metadata.
            destination.attrs[_SPACING_ATTRIBUTE] = spacing
            destination.attrs["HeadExtractionFirstPlane"] = first_head_plane
            destination.attrs["HeadExtractionOriginalDimensions"] = data.shape
            destination.create_dataset(
                "data",
                data=data[:, :, first_head_plane:],
                dtype=data.dtype,
                **_dataset_creation_options(data, output_shape),
            )
            for name in source:
                if name != "data":
                    source.copy(name, destination, name=name)

    return outputfile


def main():
    parser = argparse.ArgumentParser(
        description="Writes a HDF5 file of an AustinMan or AustinWoman head.",
        usage="python -m toolboxes.AustinManWoman.head_only_h5 filename",
    )
    parser.add_argument(
        "filename",
        help="HDF5 file containing the full AustinMan or AustinWoman model",
    )
    parser.add_argument("-o", "--output", help="optional output HDF5 filename")
    parser.add_argument(
        "--first-plane",
        type=int,
        help="optional zero-based first z plane to retain (default: upper eighth)",
    )
    args = parser.parse_args()

    outputfile = extract_head(args.filename, args.output, args.first_plane)
    logger.info("Written head geometry file: %s", outputfile)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main()
