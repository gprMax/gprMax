#react
import gprMax.gprMax
import sys
import io
import contextlib
import marimo as mo

#We are running our model here, creating a temp file and returning an outfile with new values. 
#We also save logs over here and then display them 

def run_model(model):
    temp_file = "temp_model.in"

    with open(temp_file, "w") as f:
        f.writelines(model.to_in_file())

    stream = LiveLogStream()

    with contextlib.redirect_stdout(stream):
        #gprMax.gprMax.run([temp_file])
        gprMax.gprMax.run(
            inputfile=temp_file,
            outputfile="temp_model"
        )

    return "temp_model.h5", stream.getvalue()

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

#rendering the logs with desired design 

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
