import numpy as np
from stl import mesh

from . import slice


def convert_meshes(meshes, discretization, parallel=True):
    scale, shift, shape = slice.calculate_scale_shift(meshes, discretization)
    vol = np.zeros(shape[::-1], dtype=np.int8)
    vol.fill(-1)  # Fill array with -1 to indicate background in gprMax

    for mesh_ind, org_mesh in enumerate(meshes):
        slice.scale_and_shift_mesh(org_mesh, scale, shift)
        cur_vol = slice.mesh_to_plane(org_mesh, shape, parallel)
        vol[cur_vol] = mesh_ind  # Removed plus 1 to work with gprMax material indexing

    return vol, scale, shift


def convert_file(input_file_path, discretization, pad=1, parallel=False):
    return convert_files([input_file_path], discretization, pad=pad, parallel=parallel)


def convert_files(
    input_file_paths, discretization, colors=[(0, 0, 0)], pad=1, parallel=False
):
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

    return vol
