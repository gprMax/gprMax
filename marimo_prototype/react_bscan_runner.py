#react_bscan_runner.py
import numpy as np
import copy
import h5py
from gprMax.gprMax import run
from toolboxes.Utilities.outputfiles_merge import merge_files

def get_positions(model):
    return np.arange(model.start, model.end + model.step, model.step)

def update_position(model, pos):

    axis = model.scan_axis
    target = model.scan_target

    if target == "source":
        setattr(model, f"src_{axis}", pos)

    elif target == "receiver":
        setattr(model, f"rx_{axis}", pos)

    elif target == "both":
        setattr(model, f"src_{axis}", pos)
        setattr(model, f"rx_{axis}", pos)


def run_simulation(input_file,output_name):
    # run(
    #     inputfile=input_file,
    #     n=1
    # )
    run(
        inputfile=input_file,
        outputfile=output_name
    )

"""If mentors are interested instead of calling merg file function we can directly merg files here"""

def run_bscan(model):

    positions = get_positions(model)
    outfiles = []
    for i, pos in enumerate(positions):

        print(f"Running step {i}, position = {pos}")

        m = copy.deepcopy(model)  # copying model so that the actual model doesnt get updated. 

        update_position(m, pos) 

        content = m.build_input()

        filename = f"temp_bscan{i+1}.in"

        with open(filename, "w") as f:
            f.write(content)
        
        outfiles.append(f"temp_bscan{i+1}.h5")

        run_simulation(filename,f"temp_bscan{i+1}")
    from toolboxes.Utilities.outputfiles_merge import merge_files

    merge_files(outputfiles=outfiles, removefiles=True)

    #merge_files("temp_bscan.h5", removefiles=True)

    merged_file = "temp_bscan_merged.h5"
    
    return merged_file

def extract_bscan_data(merged_file, field):
    import h5py
    import numpy as np

    with h5py.File(merged_file, "r") as f:

        #nrx = f.attrs["nrx"]  
        nrx = len(f["rxs"])       
        dt = f.attrs["dt"]

        data = []

        for i in range(1, nrx + 1):
            data.append(f[f"rxs/rx{i}/{field}"][:])

    data = np.array(data).T  

    return data, dt, nrx, field





