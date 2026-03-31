# react_bscan_model.py
# model class to build the basic model. 
class GPRMaxBscanModel:
    def __init__(self):

        self.dx = 0.01
        self.dy = 0.01
        self.dz = 0.01

        self.domain_x = 0.1
        self.domain_y = 0.1
        self.domain_z = 0.1

        self.eps = 4
        self.sigma = 0.0
        self.mur = 1
        self.sigma_m = 0

        self.material_name = "soil"

        self.amplitude = 1
        self.frequency = 1e8
        self.waveform_name = "pulse"

        self.direction = "z"

        self.src_x = 0.05
        self.src_y = 0.05
        self.src_z = 0.05

        self.rx_x = 0.06
        self.rx_y = 0.05
        self.rx_z = 0.05

        self.start = 0.0
        self.end = 0.06
        self.step = 0.02

        self.scan_axis = "x"
        self.scan_target = "source"

        self.field = "Ez"

    def build_input(self):
        lines = []

        # title
        lines.append("#title: B-scan simulation")

        # domain
        lines.append(f"#domain: {self.domain_x} {self.domain_y} {self.domain_z}")
        lines.append(f"#dx_dy_dz: {self.dx} {self.dy} {self.dz}")
        lines.append("#time_window: 5e-9")
        lines.append("#pml_cells: 2")

        # material
        lines.append(
            f"#material: {self.eps} {self.sigma} {self.mur} {self.sigma_m} {self.material_name}"
        )

        # waveform
        lines.append(
            f"#waveform: ricker {self.amplitude} {self.frequency} {self.waveform_name}"
        )

        # source
        lines.append(
            f"#hertzian_dipole: {self.direction} {self.src_x} {self.src_y} {self.src_z} {self.waveform_name}"
        )

        # receiver
        lines.append(
            f"#rx: {self.rx_x} {self.rx_y} {self.rx_z}"
        )

        # box
        lines.append("#box: 0 0 0 0.1 0.1 0.03 soil")

        return "\n".join(lines)