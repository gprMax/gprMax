def visualize_vti_output(filename, orientation = 'xy', opacity = 0.4, backend = 'static'):
    
    """Plots a 3D visualization of the geometry view file. All the 
    objects like 'pec', 'free space', 'half space' are shown using different
    colours as given on the legend on the top right of the visualization.
    
    Args:
        filename: string of filename (including path) of the geometry view file.
                    eg: cylinder_half_space.vti
        orientation: string in which plane the 3D output is to be shown.
                    eg: xy, yz, zx
        opacity: float to determine the transparency of the background.
        backend: string for the 3D plot to be static or dynamic.
    
    Returns:
        p.show: PyVista plot object.
    """
                    
    import pyvista as pv
    import vtk
    import numpy as np
    from vtk.util.numpy_support import vtk_to_numpy
    from xml.etree import ElementTree as ET
    import matplotlib.pyplot as plt
    import matplotlib
    import json
    from pathlib import Path
    
    pv.global_theme.jupyter_backend = backend
    pv.global_theme.notebook = True
    pv.global_theme.axes.show = True
    
    vti_path = Path(filename)

    # Initialize arrays
    
    materials = []
    srcs_pml = []
    rxs = []
    all_labels = []
    labels_ints = []

    # Read data from vtk file 
    
    # Initialize vtk reader
              
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    vti = reader.GetOutput()
    grid1 = pv.get_reader(filename)
    
    # Read vti data
    
    grid = grid1.read()
    
    select = grid.extract_points(grid.points==grid.points)
    
    select.set_active_scalars('Material')
    
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
    
    # Read Materials, Sources, Receivers
    
    with open(filename, 'rb') as f:
        
        f.readline()
        f.readline()
        c = f.readline().decode()
        c = c[5:-5]
        c = json.loads(c)
    
        # Print gprMax version
        
        print("\ngprMax version: " + c["gprMax_version"])
    
    # Find materials
    
    for i, mat in enumerate(c["Materials"]):
        if 'rxres' not in mat and 'Source' not in mat: 
            all_labels.append(mat)
            materials.append(mat)
            labels_ints.append(i)

    # Find receivers
    
    for item in c["Receivers"]:
        pos_rxs = item["position"]
        name = item["name"]
        box = pv.Box(bounds=(pos_rxs[0],     
                             pos_rxs[0]+spacing[0], 
                             pos_rxs[1], 
                             pos_rxs[1]+spacing[1],
                             pos_rxs[2], 
                             pos_rxs[2]+spacing[2]), level=0)
        rs_list.append(box)
        all_labels.append(name)
        labels_ints.append(labels_ints[-1]+1)
    
    # Find sources
    
    for item in c["Sources"]:
        pos_src = item["position"]
        name = item["name"]
        box = pv.Box(bounds=(pos_src[0],
                                 pos_src[0]+spacing[0], 
                                 pos_src[1], 
                                 pos_src[1]+spacing[1],
                                 pos_src[2], 
                                 pos_src[2]+spacing[2]), level=0)
        rs_list.append(box)
        all_labels.append(name)
        labels_ints.append(labels_ints[-1]+1)
    
    
    # Set colormap and normalize colors
    
    cmap = plt.cm.get_cmap("viridis", len(labels_ints))
    norm = matplotlib.colors.Normalize(vmin=labels_ints[0], vmax=labels_ints[-1]+1)
    colours = [cmap(norm(label)) for label in labels_ints]
    
    # Set legend entries
    
    legend_entries = []
    for i in range(len(labels_ints)):
        legend_entries.append([all_labels[i],colours[i][0:3]])
    
    # Initialize plot
    
    p = pv.Plotter()
    
    col_ind = len(materials)
    
    # Add sources - receivers to plot
    
    for cell in rs_list:
        p.add_mesh(cell, color=colours[col_ind])
        col_ind+=1
    
    # Add materials to plot - opacity parameter used to specify transparency
    
    for i in range(len(materials)):
        material_grid = select.threshold((i,i))
        p.add_mesh(material_grid, color = colours[i], show_scalar_bar=False, opacity=opacity)
    
    
    # Plot modelled geometry
    
    ax = p.add_axes()
    
    if backend =='static':
        if 'x' in orientation:
            p.add_ruler(
            pointa=[grid.bounds[0], 0, 0],
            pointb=[grid.bounds[1], 0, 0],
            title="X Axis")
        if 'y' in orientation:
            p.add_ruler(
            pointa=[0, grid.bounds[3], 0],
            pointb=[0, grid.bounds[2], 0],
            flip_range = True,
            title="Y Axis")
        if 'z' in orientation:
            p.add_ruler(
            pointa=[0, 0, grid.bounds[4]],
            pointb=[0, 0, grid.bounds[5]],
            title="Z Axis")
        p.enable_parallel_projection()
    else:
        p.add_ruler(
            pointa=[grid.bounds[0], 0, 0],
            pointb=[grid.bounds[1], 0, 0],
            title="X Axis")
        p.add_ruler(
            pointa=[0, grid.bounds[3], 0],
            pointb=[0, grid.bounds[2], 0],
            flip_range = True,
            title="Y Axis")
        p.add_ruler(
            pointa=[0, 0, grid.bounds[4]],
            pointb=[0, 0, grid.bounds[5]],
            title="Z Axis")
    
    p.add_legend(legend_entries, face='r')
    p.show(cpos=orientation)




