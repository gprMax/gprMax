import copy

import h5py
from lxml import etree
import numpy as np

from gprMax.grid import Grid



class Edges:

    def __init__(self, grid):

        """
            Class to define some connectivity of for an n x l x m
            grid
        """
        self.total_edges = grid.n_edges()
        self.edges = np.zeros((self.total_edges, 2), np.float32)
        self.edge_count = 0
        self.grid = grid
    """
        Adds the the edge specified by in_node and the i,j,k position of the outnode
    """
    def add_edge(self, in_label, i, j, k):

        out_label = self.grid.get(i, j, k)
        self.edges[self.edge_count] = np.array([in_label, out_label])
        self.edge_count += 1


class Coordinates:

    def __init__(self, grid):
        self.total_coordinates = grid.nx * grid.ny * grid.nz
        self.coordinate_count = 0
        self.coordinates = np.zeros((self.total_coordinates, 3), np.float32)

    def add_coordinate(self, x, y, z):
        self.coordinates[self.coordinate_count] = np.array([x, y, z])
        self.coordinate_count += 1


def hexCellPicker(grid, i, j, k):
    """
        This is the ordering of nodes in the hexahedron cell.

                7 --------- 6
               /           /|
              4 --------- 5 2
              |  3        | /
              | /         |/
              0 --------- 1

        0 1 2 3 4 5 6 7
    """

    cell = [
        grid[i][j][k],
        # 1
        grid[i + 1][j][k],
        # 2
        grid[i + 1][j + 1][k],
        # 3
        grid[i][j + 1][k],
        # 4
        grid[i][j][k + 1],
        # 5
        grid[i + 1][j][k + 1],
        # 6
        grid[i + 1][j + 1][k + 1],
        # 7
        grid[i][j + 1][k + 1]
    ]

    return cell


class Solids:

    def __init__(self, fdtd_grid):
        self.count = 0
        self.fdtd_grid = fdtd_grid
        self.total_solids = fdtd_grid.n_cells()
        self.solids = np.zeros((self.total_solids), np.float32)

    def add_solid(self, i, j, k):

        self.solids[self.count] = self.fdtd_grid.solid[i][j][k]
        self.count += 1


class SolidLabels():

    def __init__(self, label_grid):
        self.count = 0
        self.label_grid = label_grid
        self.total_solids = label_grid.n_cells()
        self.solid_labels = np.zeros((self.total_solids, 8), np.float32)

    def add(self, i, j, k):

        solid_labels = hexCellPicker(self.label_grid.grid, i, j, k)
        self.solid_labels[self.count] = solid_labels
        self.count += 1


class Materials:

    def __init__(self, fdtd_grid):
        self.fdtd_grid = fdtd_grid
        self.n_edges = fdtd_grid.n_edges()
        self.materials = np.zeros((self.n_edges), np.float32)
        self.material_count = 0

    # direction x->0 y->1 z->2
    def add_material(self, i, j, k, direction):

        material = self.fdtd_grid.ID[direction, i, j, k]
        self.materials[self.material_count] = material

        self.material_count += 1


def process_grid(fdtd_grid):

    # Dimensions of the problem domain.
    nx = fdtd_grid.nx
    ny = fdtd_grid.ny
    nz = fdtd_grid.nz

    # label each node in the space
    labels = np.arange(nx * ny * nz).reshape(nx, ny, nz)

    label_grid = Grid(labels)

    # Edges define the connectivity of the grid.
    edges = Edges(label_grid)

    # Material for each edge
    edge_materials = Materials(fdtd_grid)

    # Define coordinates for each node
    coordinates = Coordinates(fdtd_grid)

    # Material for each solid
    solids = Solids(fdtd_grid)

    # Connectivity for hexhahedron grid
    solid_labels = SolidLabels(label_grid)

    i_max = nx - 1
    j_max = ny - 1
    k_max = nz - 1

    for i, ix in enumerate(labels):
        for j, jx in enumerate(ix):
            for k, kx in enumerate(jx):

                label = labels[i][j][k]

                # Each vertex can have varying numbers of edges

                # Type 1 vertex
                if i < i_max and j < j_max and k < k_max:
                    edges.add_edge(label, i + 1, j, k)
                    edges.add_edge(label, i, j + 1, k)
                    edges.add_edge(label, i, j, k + 1)

                    edge_materials.add_material(i, j, k, 0)
                    edge_materials.add_material(i, j, k, 1)
                    edge_materials.add_material(i, j, k, 2)

                    # Only this node can support a cell
                    solids.add_solid(i, j, k)
                    solid_labels.add(i, j, k)

                # Type 2 vertex
                elif i < i_max and j == j_max and k == k_max:
                    edges.add_edge(label, i + 1, j, k)
                    edge_materials.add_material(i, j, k, 0)

                # Type 7 vertex
                elif i < i_max and j == j_max and k < k_max:
                    edges.add_edge(label, i + 1, j, k)
                    edges.add_edge(label, i, j, k + 1)
                    edge_materials.add_material(i, j, k, 0)
                    edge_materials.add_material(i, j, k, 2)

                # Type 6 vertex
                elif i == i_max and j == j_max and k < k_max:
                    edges.add_edge(label, i, j, k + 1)
                    edge_materials.add_material(i, j, k, 2)

                # Type 5 vertex
                elif i == i_max and j < j_max and k < k_max:
                    edges.add_edge(label, i, j, k + 1)
                    edges.add_edge(label, i, j + 1, k)
                    edge_materials.add_material(i, j, k, 2)
                    edge_materials.add_material(i, j, k, 1)

                # Type 4 vertex
                elif i == i_max and j < j_max and k == k_max:
                    edges.add_edge(label, i, j + 1, k)
                    edge_materials.add_material(i, j, k, 1)

                # Type 8 vertex
                elif i < i_max and j < j_max and k == k_max:
                    edges.add_edge(label, i, j + 1, k)
                    edges.add_edge(label, i + 1, j, k)
                    edge_materials.add_material(i, j, k, 1)
                    edge_materials.add_material(i, j, k, 0)

                # Type 3 vertex
                # Has no new connectivity
                elif i == i_max and j == j_max and k == k_max:
                    pass
                else:
                    print('oh no')

                # Add the coordinates
                coordinates.add_coordinate(i, j, k)

    return {
        'coordinates': coordinates,
        'solids': solids,
        'solid_labels': solid_labels,
        'edges': edges,
        'edge_materials': edge_materials,
    }


def write_output_file(filename, grid):

    data = process_grid(grid)
    data['filename'] = filename
    data['xml_doc'] = create_xdmf_markup(data)

    write_H5file(data)
    write_xml_doc(data)


def write_xml_doc(options):
    #write xml to file
    with open(options['filename'] + '.xdmf', 'wb') as xdmf_f:
        xdmf_f.write(options['xml_doc'])


def write_H5file(options):

        f = h5py.File(options['filename'] + '.h5', "w")
        coords = f.create_group("mesh")
        coords.create_dataset('coordinates', data=options['coordinates'].coordinates)
        coords.create_dataset('connectivity', data=options['edges'].edges)
        coords.create_dataset('solid_connectivity', data=options['solid_labels'].solid_labels)
        data = f.create_group("data")
        data.create_dataset('materials', data=options['edge_materials'].materials)
        data.create_dataset('solids', data=options['solids'].solids)


def create_xdmf_markup(options):

    # Write the XDMF markup for edge style grid
    xdmf_el = etree.Element("Xdmf", Version="2.0")

    domain_el = etree.Element("Domain")
    xdmf_el.append(domain_el)

    grid_el = etree.Element("Grid", Name="Edges", GridType="Uniform")
    domain_el.append(grid_el)

    # Create the grid node
    topology_el = etree.Element("Topology", TopologyType="Polyline", NumberOfElements=str(options['edges'].total_edges))
    grid_el.append(topology_el)

    topology_dimensions = "{} 2".format(options['edges'].total_edges)
    top_data_el = etree.Element("DataItem", Dimensions=topology_dimensions, NumberType="Float", Precision="8", Format="HDF")
    top_data_el.text = "{}:/mesh/connectivity".format(options['filename'] + '.h5')
    topology_el.append(top_data_el)

    # Create the Geometry node
    geometry_el = etree.Element("Geometry", GeometryType="XYZ")
    grid_el.append(geometry_el)

    # Create the origin coordinates
    coordinates_dimensions = "{} 3".format(options['coordinates'].total_coordinates)
    origin_el = etree.Element("DataItem", Dimensions=coordinates_dimensions, NumberType="Float", Precision="8", Format="HDF")
    origin_el.text = "{}:/mesh/coordinates".format(options['filename'] + '.h5')
    geometry_el.append(origin_el)

    # Create the materials attribute
    attr_el = etree.Element("Attribute", Center="Cell", Name="Edge_Materials")
    grid_el.append(attr_el)

    materials_dimensions = "{} 1".format(options['edge_materials'].material_count)
    materials_el = etree.Element("DataItem", Dimensions=materials_dimensions, NumberType="Float", Precision="8", Format="HDF")
    materials_el.text = "{}:/data/materials".format(options['filename'] + '.h5')
    attr_el.append(materials_el)

    v_grid_el = etree.Element("Grid", Name="Voxel", GridType="Uniform")
    domain_el.append(v_grid_el)

    n_solids = str(options['solids'].solids.size)
    v_topology_el = etree.Element("Topology", TopologyType="Hexahedron", NumberOfElements=str(options['solids'].solids.size))
    v_grid_el.append(v_topology_el)

    solid_label_d = "{} {}".format(n_solids, 8)
    solid_labels_el = etree.Element("DataItem", Dimensions=solid_label_d, Format="HDF")
    solid_labels_el.text = "{}:/mesh/solid_connectivity".format(options['filename'] + '.h5')
    v_topology_el.append(solid_labels_el)

    # Same geometry as edges
    v_grid_el.append(copy.deepcopy(geometry_el))

    v_attr = etree.Element("Attribute", Name="Voxel_Materials", Center="Cell")
    v_grid_el.append(v_attr)

    d4 = etree.Element("DataItem", Format="HDF", NumberType="Float", Precision="4", Dimensions=str(options['solids'].solids.size))
    d4.text = "{}:/data/solids".format(options['filename'] + '.h5')
    v_attr.append(d4)

    # Define a doctype - useful for parsers
    doc_type = '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>'

    # Serialize elements
    xml_doc = etree.tostring(xdmf_el, xml_declaration=True,
        encoding="utf-8",  doctype=doc_type, pretty_print=True)

    return xml_doc
