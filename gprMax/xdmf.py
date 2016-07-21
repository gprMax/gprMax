import copy

import h5py
from lxml import etree
import numpy as np

from gprMax.grid import Grid


class ListCounter():

    def __init__(self, npArray):
        self.array = npArray
        self.count = 0

    def add(self, item):
        self.array[self.count] = item
        self.count += 1


class EdgeLabels:

    def __init__(self, grid):
        """
            Class to define some connectivity of for an n x l x m
            grid
        """
        self.total_edges = grid.n_edges()
        self.grid = grid
        self.edges = np.zeros((self.total_edges, 2), np.int8)
        self.edge_counter = ListCounter(self.edges)

    def add_edge(self, in_label, i, j, k):
        """
        Adds the the edge specified by in_node and the i,j,k position of the outnode
        """
        out_label = self.grid.get(i, j, k)
        edge = np.array([in_label, out_label])
        self.edge_counter.add(edge)


class EdgeMaterials:

    def __init__(self, fdtd_grid):
        self.fdtd_grid = fdtd_grid
        self.n_edges = fdtd_grid.n_edges()
        self.materials = np.zeros((self.n_edges), np.int8)
        self.materialCounter = ListCounter(self.materials)

    # direction x->0 y->1 z->2
    def add_material(self, i, j, k, direction):

        material = self.fdtd_grid.ID[direction, i, j, k]
        self.materialCounter.add(material)


class Coordinates:

    def __init__(self, grid):
        self.total_coordinates = grid.n_nodes()
        self.coordinates = np.zeros((self.total_coordinates, 3), np.int8)
        self.coord_counter = ListCounter(self.coordinates)

    def add_coordinate(self, x, y, z):
        self.coord_counter.add(np.array([x, y, z]))


class Solids:

    def __init__(self, fdtd_grid):
        self.fdtd_grid = fdtd_grid
        self.total_solids = fdtd_grid.n_cells()
        self.solids = np.zeros((self.total_solids), np.int8)
        self.solid_counter = ListCounter(self.solids)

    def add_solid(self, i, j, k):

        self.solid_counter.add(self.fdtd_grid.solid[i][j][k])


class SolidLabels():

    def __init__(self, label_grid):
        self.label_grid = label_grid
        self.total_solids = label_grid.n_cells()
        self.solid_labels = np.zeros((self.total_solids, 8), np.int8)
        self.label_counter = ListCounter(self.solid_labels)

    def hexCellPicker(self, grid, i, j, k):
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

    def add(self, i, j, k):

        solid_labels = self.hexCellPicker(self.label_grid.grid, i, j, k)
        self.label_counter.add(solid_labels)


class SolidManager(Grid):

    def __init__(self, label_grid, fdtd_grid):

        super().__init__(label_grid.grid)
        self.solids = Solids(fdtd_grid)
        self.solid_labels = SolidLabels(label_grid)

    def createSolid(self, i, j, k):
        if i < self.i_max and j < self.j_max and k < self.k_max:
            self.solids.add_solid(i, j, k)
            self.solid_labels.add(i, j, k)


class EdgeManager(Grid):
    """
    Class to manage the creation of edges and matching edge materials.
    """

    def __init__(self, label_grid, fdtd_grid):
        super().__init__(label_grid.grid)
        self.edges = EdgeLabels(label_grid)
        self.edge_materials = EdgeMaterials(fdtd_grid)

    def createEdges(self, i, j, k):
        """
            Create the relevant edges and corresponding edge materials.
            Args:
                i (int): i index of label in labels_grid
                j (int): j index of label in labels_grid
                k (int): k index of label in labels_grid

        """
        edges = self.edges
        edge_materials = self.edge_materials
        i_max = self.i_max
        j_max = self.j_max
        k_max = self.k_max
        label = self.edges.grid.get(i, j, k)

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


def process_grid(fdtd_grid, res):

    # Create a grid of labels with equal dimension to fdtd grid
    labels = np.arange(fdtd_grid.n_nodes()).reshape(fdtd_grid.nx, fdtd_grid.ny, fdtd_grid.nz)

    label_grid = Grid(labels)

    # Define coordinates for each node
    coordinates = Coordinates(fdtd_grid)

    solid_manager = SolidManager(label_grid, fdtd_grid)

    if res == 'f':
        edge_manager = EdgeManager(label_grid, fdtd_grid)

    # Iterate through the label and create relevant edges and solids.

    for i, ix in enumerate(labels):
        for j, jx in enumerate(ix):
            for k, kx in enumerate(jx):

                if res == 'f':
                    edge_manager.createEdges(i, j, k)

                solid_manager.createSolid(i, j, k)

                # Add the coordinates
                coordinates.add_coordinate(i, j, k)

    data = {
        'coordinates': coordinates,
        'solids': solid_manager.solids,
        'solid_labels': solid_manager.solid_labels,
    }

    if res == 'f':
        data['edges'] = edge_manager.edges
        data['edge_materials'] = edge_manager.edge_materials

    dir(edge_manager.edge_materials)

    return data


def write_output_file(filename, grid, res):

    data = process_grid(grid, res)
    data['filename'] = filename
    data['xml_doc'] = create_xdmf_markup(data)

    write_H5file(data)
    write_xml_doc(data)


def write_xml_doc(options):
    # write xml to file
    with open(options['filename'] + '.xdmf', 'wb') as xdmf_f:
        xdmf_f.write(options['xml_doc'])


def write_H5file(options):

    f = h5py.File(options['filename'] + '.h5', "w")
    coords = f.create_group("mesh")
    data = f.create_group("data")

    coords.create_dataset('coordinates', data=options['coordinates'].coordinates)
    coords.create_dataset('solid_connectivity', data=options['solid_labels'].solid_labels)
    data.create_dataset('solids', data=options['solids'].solids)

    if 'edges' in options:
        data.create_dataset('materials', data=options['edge_materials'].materials)
        coords.create_dataset('connectivity', data=options['edges'].edges)


def create_xdmf_markup(options):

    # Write the XDMF markup for edge style grid
    xdmf_el = etree.Element("Xdmf", Version="2.0")

    domain_el = etree.Element("Domain")
    xdmf_el.append(domain_el)

    geometry_el = etree.Element("Geometry", GeometryType="XYZ")
    coordinates_dimensions = "{} 3".format(options['coordinates'].total_coordinates)
    origin_el = etree.Element("DataItem", Dimensions=coordinates_dimensions, NumberType="Float", Precision="8", Format="HDF")
    origin_el.text = "{}:/mesh/coordinates".format(options['filename'] + '.h5')
    geometry_el.append(origin_el)

    # Check if there are edges to write
    if 'edges' in options:

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
        grid_el.append(copy.deepcopy(geometry_el))

        # Create the origin coordinates

        # Create the materials attribute
        attr_el = etree.Element("Attribute", Center="Cell", Name="Edge_Materials")
        grid_el.append(attr_el)

        materials_dimensions = "{} 1".format(options['edge_materials'].materials.size)
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
                             encoding="utf-8", doctype=doc_type, pretty_print=True)

    return xml_doc
