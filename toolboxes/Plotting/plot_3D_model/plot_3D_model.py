import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path


def visualize_vti_output(filename,orientation):
    
    #pv.global_theme.jupyter_backend = 'static'
    pv.global_theme.notebook = True
    pv.global_theme.axes.show = True
    
    vti_path = Path(filename)

    # Initialize arrays

    objects = []
    materials = []
    srcs_pml = []
    rxs = []


    # Find materials-sources-receiver IDs in files

    with open(vti_path, 'rb') as f:       
        for line in f:
            if line.startswith(b'<Material'):
                line.rstrip(b'\n')
                tmp = (int(ET.fromstring(line).text), ET.fromstring(line).attrib.get('name'))
                materials.append(tmp)
            elif line.startswith(b'<Sources') or line.startswith(b'<PML'):
                line.rstrip(b'\n')
                tmp = (int(ET.fromstring(line).text), ET.fromstring(line).attrib.get('name'))
                srcs_pml.append(tmp)
            elif line.startswith(b'<Receivers'):
                line.rstrip(b'\n')
                tmp = (int(ET.fromstring(line).text), ET.fromstring(line).attrib.get('name'))
                rxs.append(tmp)
            

    # Initialize vtk reader
          
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    vti = reader.GetOutput()
    grid1 = pv.get_reader(filename)

    # Read vti data

    grid = grid1.read()

    #grid.cell_data['Material']=np.arange(grid.n_cells)

    select = grid.extract_points(grid.points==grid.points)

    # Prints info for data
    #print(grid)

    # Get array names in file

    #print(grid.array_names)

    # Get grid dimensions

    dim = grid.dimensions

    # Get spacing

    spacing = grid.spacing

    # Get bounds

    bounds = grid.bounds

    # Initialise rxs-srcs list

    rs_list = []

    # Find material IDs in grid

    material_ints = np.unique(grid['Material']).astype(int)

    for i in range(0,len(material_ints)):
        grid['Material'][grid['Material']==material_ints[i]] = i

    all_labels = [materials[i] for i in material_ints]

    # Find receivers and set their position in the grid

    for i in range(len(rxs)):
        current_rxs = rxs[i][1]
        if current_rxs.find("(")==-1 & current_rxs.find(")")==-1:
            all_labels.append(rxs[i])
            Rx = vtk_to_numpy(vti.GetCellData().GetArray('Receivers'))
            rx_res = Rx.flatten().reshape(dim[2]-1, dim[1]-1, dim[0]-1)
            pos_rxs = np.array(np.where(rx_res==1)[::-1]).reshape(1,3)[0]*0.001
            box = pv.Box(bounds=(pos_rxs[0],     
                         pos_rxs[0]+spacing[0], 
                         pos_rxs[1], 
                         pos_rxs[1]+spacing[1],
                         pos_rxs[2], 
                         pos_rxs[2]+spacing[2]), level=0)
            rs_list.append(box)
        
        else:
            all_labels.append(rxs[i])
            current_rxs = current_rxs[current_rxs.find("(")+1:current_rxs.find(")")]
            pos_rxs = np.array(current_rxs.split(",")).astype(np.float64)*0.001
            box = pv.Box(bounds=(pos_rxs[0],     
                         pos_rxs[0]+spacing[0], 
                         pos_rxs[1], 
                         pos_rxs[1]+spacing[1],
                         pos_rxs[2], 
                         pos_rxs[2]+spacing[2]), level=0)
            rs_list.append(box)
    
    
    # Add sources to the labels list

    all_labels = all_labels + srcs_pml[1:]   

    # Convert labels to integers

    labels_ints = np.zeros(len(all_labels))
    labels_ints[0:material_ints.shape[0]] = material_ints
    for i in range(material_ints.shape[0], len(all_labels)):
        labels_ints[i] = labels_ints[i-1]+1

    # Find sources and set the geometry

    for i in range(len(srcs_pml)):
        if srcs_pml[i][1]!='PML boundary region':
            current_src = srcs_pml[i][1]
            current_src = current_src[current_src.find("(")+1:current_src.find(")")]
            pos_src = np.array(current_src.split(",")).astype(np.float64)*0.001
            box = pv.Box(bounds=(pos_src[0],
                             pos_src[0]+spacing[0], 
                             pos_src[1], 
                             pos_src[1]+spacing[1],
                             pos_src[2], 
                             pos_src[2]+spacing[2]), level=0)
            rs_list.append(box)


    # Set colormap and normalize colors

    cmap = plt.cm.get_cmap("viridis", len(labels_ints))
    norm = matplotlib.colors.Normalize(vmin=labels_ints[0], vmax=labels_ints[-1]+1)
    colours = [cmap(norm(label)) for label in labels_ints]

    # Set legend entries

    legend_entries = []
    for i in range(len(labels_ints)):
        legend_entries.append([all_labels[i][1],colours[i][0:3]])


    # Initialize plot

    p = pv.Plotter()

    col_ind = material_ints.shape[0]

    # Add sources - receivers to plot

    for cell in rs_list:
        p.add_mesh(cell, color=colours[col_ind])
        col_ind+=1

    # Add materials to plot - opacity parameter used to specify transparency

    for i in range(len(material_ints)):
        material_grid = select.threshold((i,i))
        p.add_mesh(material_grid, color = colours[i], show_scalar_bar=False, opacity=0.2)


    # Plot modelled geometry

    p.add_axes()
    p.show_bounds()
    p.add_legend(legend_entries, face='r')
    p.show(cpos=orientation)


#visualize_vti_output('simplesand2.vti','xy')