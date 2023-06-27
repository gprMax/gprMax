import itertools
import multiprocessing as mp
import sys

import numpy as np
from gprMax.utilities.utilities import get_terminal_width
from tqdm import tqdm

from . import perimeter


def mesh_to_plane(mesh, bounding_box, parallel):
    if parallel:
        pool = mp.Pool(mp.cpu_count())
        result_ids = []

    vol = np.zeros(bounding_box[::-1], dtype=bool)

    current_mesh_indices = set()
    z = 0
    with tqdm(total=bounding_box[2], desc="Processing Layers", ncols=get_terminal_width() - 1, file=sys.stdout) as pbar:
        for event_z, status, tri_ind in generate_tri_events(mesh):
            while event_z - z >= 0:
                mesh_subset = [mesh[ind] for ind in current_mesh_indices]

                if parallel:
                    pass
                else:
                    pbar.update(1)
                    _, pixels = paint_z_plane(mesh_subset, z, bounding_box[1::-1])
                    vol[z] = pixels
                z += 1

            if status == "start":
                assert tri_ind not in current_mesh_indices
                current_mesh_indices.add(tri_ind)
            elif status == "end":
                assert tri_ind in current_mesh_indices
                current_mesh_indices.remove(tri_ind)

    if parallel:
        results = [r.get() for r in result_ids]

        for z, pixels in results:
            vol[z] = pixels

        pool.close()
        pool.join()

    return vol


def paint_z_plane(mesh, height, plane_shape):
    pixels = np.zeros(plane_shape, dtype=bool)

    lines = []
    for triangle in mesh:
        triangle_to_intersecting_lines(triangle, height, pixels, lines)
    perimeter.lines_to_voxels(lines, pixels)

    return height, pixels


def linear_interpolation(p1, p2, distance):
    """
    :param p1: Point 1
    :param p2: Point 2
    :param distance: Between 0 and 1, Lower numbers return points closer to p1.
    :return: A point on the line between p1 and p2
    """
    return p1 * (1 - distance) + p2 * distance


def triangle_to_intersecting_lines(triangle, height, pixels, lines):
    assert len(triangle) == 3
    above = list(filter(lambda pt: pt[2] > height, triangle))
    below = list(filter(lambda pt: pt[2] < height, triangle))
    same = list(filter(lambda pt: pt[2] == height, triangle))
    if len(same) == 3:
        for i in range(0, len(same) - 1):
            for j in range(i + 1, len(same)):
                lines.append((same[i], same[j]))
    elif len(same) == 2:
        lines.append((same[0], same[1]))
    elif len(same) == 1:
        if above and below:
            side1 = where_line_crosses_z(above[0], below[0], height)
            lines.append((side1, same[0]))
        else:
            x = int(same[0][0])
            y = int(same[0][1])
            pixels[y][x] = True
    else:
        cross_lines = [(b, a) for a, b in itertools.product(above, below)]
        side1 = where_line_crosses_z(cross_lines[0][0], cross_lines[0][1], height)
        side2 = where_line_crosses_z(cross_lines[1][0], cross_lines[1][1], height)
        lines.append((side1, side2))


def where_line_crosses_z(p1, p2, z):
    if p1[2] > p2[2]:
        p1, p2 = p2, p1
    # now p1 is below p2 in z
    if p2[2] == p1[2]:
        distance = 0
    else:
        distance = (z - p1[2]) / (p2[2] - p1[2])

    return linear_interpolation(p1, p2, distance)


def calculate_scale_shift(meshes, discretization):
    mesh_min = meshes[0].min(axis=(0, 1))
    mesh_max = meshes[0].max(axis=(0, 1))

    for mesh in meshes[1:]:
        mesh_min = np.minimum(mesh_min, mesh.min(axis=(0, 1)))
        mesh_max = np.maximum(mesh_max, mesh.max(axis=(0, 1)))
    amplitude = mesh_max - mesh_min
    # Standard Unit of STL is mm
    vx = discretization[0] * 1000
    vy = discretization[1] * 1000
    vz = discretization[2] * 1000

    bx = int(amplitude[0] / vx)
    by = int(amplitude[1] / vy)
    bz = int(amplitude[2] / vz)
    bounding_box = [bx + 1, by + 1, bz + 1]

    return max(1 / vx, 1 / vy, 1 / vz), mesh_min, bounding_box


def scale_and_shift_mesh(mesh, scale, shift):
    for i, dim_shift in enumerate(shift):
        mesh[..., i] = (mesh[..., i] - dim_shift) * scale


def generate_tri_events(mesh):
    # Create data structure for plane sweep
    events = []
    for i, tri in enumerate(mesh):
        bottom, middle, top = sorted(tri, key=lambda pt: pt[2])
        events.append((bottom[2], "start", i))
        events.append((top[2], "end", i))

    return sorted(events, key=lambda tup: tup[0])
