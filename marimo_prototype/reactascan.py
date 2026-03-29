import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import sys
    import os

    sys.path.append(os.path.abspath(".."))

    return (mo,)


@app.cell
def _():
    from toolboxes.Plotting.plot_Ascan import mpl_plot

    return (mpl_plot,)


@app.cell
def _(mo):
    # Here we are creating the title symbol (gprMax)

    title = mo.md("""
    <div style="font-size:40px;font-weight:800;text-align:center;margin:0;padding:0;">
    <span style="color:#2563eb">gpr</span><span style="color:#1f2937">MAX</span>
    </div>

    <hr style="border-color:#d1d5db;margin:6px 0 12px 0;">
    """)

    # We are defining numbers to enter here with a given value. 
    # I am not specifically using slider as getting an exact value in slider is a bit rigerous
    # where as entering a value is more easy with no uper limit
    # but they can be changed to slider by replacing "mo.ui.number()" with "mo.ui.slider()"

    dx = mo.ui.number(value=0.02, step=0.001, label="dx (m)")
    dy = mo.ui.number(value=0.02, step=0.001, label="dy (m)")
    dz = mo.ui.number(value=0.02, step=0.001, label="dz (m)")

    domain_x = mo.ui.number(value=0.2, step=0.1, label="domain_x (m)")
    domain_y = mo.ui.number(value=0.2, step=0.1, label="domain_y (m)")
    domain_z = mo.ui.number(value=0.1, step=0.05, label="domain_z (m)")

    eps = mo.ui.number(value=4, step=0.1, label="Permittivity (εr)")
    sigma = mo.ui.number(value=0.01, step=0.01, label="Conductivity (σ)")
    mur = mo.ui.number(value=1, step=0.1, label="Magnetic permeability (μr)")
    sigma_m = mo.ui.number(value=0, step=0.01, label="Magnetic conductivity (σm)")

    material_name = mo.ui.dropdown(options=["half_space","soil","concrete","custom_material"],value="half_space",label="Material name")

    frequency = mo.ui.number(value=1e8, step=1e8, label="Frequency (Hz)")
    amplitude = mo.ui.number(value=1, step=0.1, label="Amplitude")
    waveform_name = mo.ui.text(value="pulse", label="Waveform name")

    src_x = mo.ui.number(value=0.05, step=0.01, label="Source x (m)")
    src_y = mo.ui.number(value=0.05, step=0.01, label="Source y (m)")
    src_z = mo.ui.number(value=0.04, step=0.01, label="Source z (m)")

    direction = mo.ui.dropdown(options=["x","y","z"],value="z",label="Dipole direction")

    rx_x = mo.ui.number(value=0.05, step=0.01, label="Receiver x (m)")
    rx_y = mo.ui.number(value=0.05, step=0.01, label="Receiver y (m)")
    rx_z = mo.ui.number(value=0.04, step=0.01, label="Receiver z (m)")

    #I tried using mo.ui.button but due to some issues it wasnt working properly
    #I upgraded it to this temperory solution for now where it breaks like a MCB (Miniuatre Circuit Breaker)
    #When value is non zero it computes simulation but when it is 0 you can enter multiple parameters without simulation running.
    #Thus it saves computational power and time. I WILL BE WORKING ON A MORE ROBUST SOLUTION

    run_button = mo.ui.number(
    value=1, step = 1, label="Run Simulation"
    )

    domain_section = mo.accordion({"Domain":mo.vstack([dx,dy,dz,domain_x,domain_y,domain_z])})
    material_section = mo.accordion({"Material":mo.vstack([eps,sigma,mur,sigma_m,material_name])})
    waveform_section = mo.accordion({"Waveform":mo.vstack([amplitude,frequency,waveform_name])})
    source_section = mo.accordion({"Source":mo.vstack([direction,src_x,src_y,src_z])})
    receiver_section = mo.accordion({"Receiver":mo.vstack([rx_x,rx_y,rx_z])})

    sidebar = mo.vstack([title,domain_section,material_section,waveform_section,source_section,receiver_section,run_button])
    return (
        domain_x,
        domain_y,
        domain_z,
        dx,
        dy,
        dz,
        eps,
        frequency,
        rx_x,
        rx_y,
        rx_z,
        sidebar,
        sigma,
        src_x,
        src_y,
        src_z,
        run_button
    )


@app.cell
def _(mo, sidebar):
    mo.sidebar(sidebar)
    return


@app.cell
def _(
    domain_x,
    domain_y,
    domain_z,
    dx,
    dy,
    dz,
    eps,
    frequency,
    mo,
    rx_x,
    rx_y,
    rx_z,
    sigma,
    src_x,
    src_y,
    src_z,
    run_button
):

    from react_model_builder import GPRMaxModel
    from react_run_simulation import run_model

    # Here we are copying the predefined model
    model = GPRMaxModel()

    # We are updating the value of the parameters of those models

    model.dx = dx.value
    model.dy = dy.value
    model.dz = dz.value

    model.domain = (domain_x.value, domain_y.value, domain_z.value)

    model.material["eps"] = eps.value
    model.material["sigma"] = sigma.value

    model.waveform["frequency"] = frequency.value

    model.source["x"] = src_x.value
    model.source["y"] = src_y.value
    model.source["z"] = src_z.value

    model.receiver["x"] = rx_x.value
    model.receiver["y"] = rx_y.value
    model.receiver["z"] = rx_z.value

    if run_button.value != 0:
        simulation_output, _ = run_model(model)
    else:
        simulation_output = None

    return (simulation_output,)


@app.cell
def _(mpl_plot, simulation_output):

    import matplotlib.pyplot as plt

    fig = None

    if simulation_output is not None:
        mpl_plot(simulation_output)
        fig = plt.gcf()

    fig
    return


if __name__ == "__main__":
    app.run()
