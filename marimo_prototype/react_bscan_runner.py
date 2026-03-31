#react_bscan_runner.py
import numpy as np
import copy
import h5py
from gprMax.gprMax import run
from toolboxes.Utilities.outputfiles_merge import merge_files
import marimo as mo
import io
import contextlib

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


def run_simulation(input_file,output_name,stream):
    with contextlib.redirect_stdout(stream):
        run(
            inputfile=input_file,
            outputfile=output_name
        )

"""If mentors are interested instead of calling merg file function we can directly merg files here"""

def run_bscan(model):

    positions = get_positions(model)
    outfiles = []
    stream = LiveLogStream()
    for i, pos in enumerate(positions):

        print(f"Running step {i}, position = {pos}")

        m = copy.deepcopy(model)  # copying model so that the actual model doesnt get updated. 

        update_position(m, pos) 

        content = m.build_input()

        filename = f"temp_bscan{i+1}.in"

        with open(filename, "w") as f:
            f.write(content)
        
        outfiles.append(f"temp_bscan{i+1}.h5")

        run_simulation(filename,f"temp_bscan{i+1}", stream)
    from toolboxes.Utilities.outputfiles_merge import merge_files

    merge_files(outputfiles=outfiles, removefiles=True)


    merged_file = "temp_bscan_merged.h5"
    
    return merged_file, stream.getvalue()

def extract_bscan_data(merged_file, field):
    import h5py
    import numpy as np

    with h5py.File(merged_file, "r") as f:

        nrx = len(f["rxs"])       
        dt = f.attrs["dt"]

        data = []

        for i in range(1, nrx + 1):
            data.append(f[f"rxs/rx{i}/{field}"][:])

    data = np.array(data).T  

    return data, dt, nrx, field


#Class to display logs and overwrite progress to create a progess bar effect. 

class LiveLogStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self._log = ""
        
    def write(self, text):
        #Over writing line to simulate progress bar
        if '\r' in text:
            parts = text.split('\r')
            self._log = self._log.rsplit('\n', 1)[0] + '\n' + parts[-1]
        else:
            self._log += text
        
        #Showing output before all the processing is done. 
        mo.output.replace(render_log(self._log))
        return super().write(text)

def render_log(text: str):
    log_pannel = mo.md(f"""
<div style="
  background:#0f172a; color:#e5e7eb;
  font-family:monospace; font-size:12px;
  padding:12px; border-radius:8px;
  height:400px; overflow-y:auto;
">
  <b style="color:#60a5fa">gprMax Simulation Logs</b>
  <hr style="border-color:#374151; margin:6px 0;">
  <pre style="margin:0; white-space:pre-wrap;">{text}</pre>
</div>
""")
    return log_pannel



