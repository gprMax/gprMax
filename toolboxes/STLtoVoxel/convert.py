import numpy as np
from stl import mesh

from . import slice


def convert_meshes(meshes, discretization, parallel=False):
    if not meshes:
        raise ValueError("at least one mesh is required")

    scale, shift, shape = slice.calculate_scale_shift(meshes, discretization)
    vol = np.zeros(shape[::-1], dtype=np.int16)
    vol.fill(-1)  # Fill array with -1 to indicate background in gprMax

    for mesh_ind, org_mesh in enumerate(meshes):
        # Work on a copy: callers may reuse their original meshes or compare
        # serial and parallel conversion results.
        scaled_mesh = np.array(org_mesh, dtype=float, copy=True)
        slice.scale_and_shift_mesh(scaled_mesh, scale, shift)
        cur_vol = slice.mesh_to_plane(scaled_mesh, shape, parallel)
        vol[cur_vol] = mesh_ind  # Removed plus 1 to work with gprMax material indexing

    return vol, scale, shift


def convert_file(input_file_path, discretization, pad=1, parallel=False):
    return convert_files([input_file_path], discretization, pad=pad, parallel=parallel)


def convert_files(input_file_paths, discretization, colors=None, pad=0, parallel=False):
    """Convert one or more STL files to a gprMax material-index array.

    ``colors`` is retained for compatibility with the upstream API but is not
    used by gprMax. ``pad`` adds background cells around the converted object.
    """
    if pad < 0:
        raise ValueError("pad must be non-negative")

    meshes = []

    for input_file_path in input_file_paths:
        mesh_obj = mesh.Mesh.from_file(input_file_path)
        org_mesh = np.hstack(
            (
                mesh_obj.v0[:, np.newaxis],
                mesh_obj.v1[:, np.newaxis],
                mesh_obj.v2[:, np.newaxis],
            )
        )
        meshes.append(org_mesh)
    vol, scale, shift = convert_meshes(meshes, discretization, parallel)
    vol = np.transpose(vol)
    if pad:
        vol = np.pad(vol, pad, mode="constant", constant_values=-1)

    return vol
