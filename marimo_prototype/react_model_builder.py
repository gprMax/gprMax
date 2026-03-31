class GPRMaxModel:

    def __init__(self):

        self.dx = 0.02
        self.dy = 0.02
        self.dz = 0.02
        self.pml_cells = 0

        self.domain = (0.4,0.4,0.2) 

        self.time_window = 5e-9   

        self.material = {
            "eps":4,
            "sigma":0.01,
            "mur":1,
            "sigma_m":0,
            "name":"half_space"
        }

        self.waveform = {
            "type":"gaussian",
            "amplitude":1,
            "frequency":1e8,
            "name":"pulse"
        }

        self.source = {
            "dir":"z",
            "x":0.1,
            "y":0.1,
            "z":0.05
        }

        self.receiver = {
            "x":0.15,
            "y":0.1,
            "z":0.05
        }

    def to_in_file(self):

        lines = []

        lines.append(f"#dx_dy_dz: {self.dx} {self.dy} {self.dz}\n")

        x,y,z = self.domain
        lines.append(f"#domain: {x} {y} {z}\n")

        lines.append(f"#pml_cells: {self.pml_cells}\n")

        lines.append(f"#time_window: {self.time_window}\n")

        m = self.material
        lines.append(f"#material: {m['eps']} {m['sigma']} {m['mur']} {m['sigma_m']} {m['name']}\n")

        w = self.waveform
        lines.append(f"#waveform: gaussian {w['amplitude']} {w['frequency']} {w['name']}\n")

        s = self.source
        lines.append(f"#hertzian_dipole: {s['dir']} {s['x']} {s['y']} {s['z']} {w['name']}\n")

        r = self.receiver
        lines.append(f"#rx: {r['x']} {r['y']} {r['z']}\n")

        return lines