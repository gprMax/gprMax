import marimo

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    return mo,

@app.cell
def __(mo):
    mo.md(
        """
        # gprMax Material Parameter Generator POC
        Adjust the physical parameters below to dynamically generate a valid gprMax `#material` command. 
        This demonstrates how `marimo` can act as a reactive, interactive dashboard for gprMax models.
        """
    )
    return

@app.cell
def __(mo):
    # Sliders for key physical properties in gprMax
    permittivity_slider = mo.ui.slider(start=1.0, stop=80.0, step=0.1, value=1.0, label="Relative Permittivity ($\epsilon_r$)")
    conductivity_slider = mo.ui.slider(start=0.0, stop=10.0, step=0.01, value=0.0, label="Conductivity ($\sigma$) [S/m]")
    permeability_slider = mo.ui.slider(start=1.0, stop=5.0, step=0.1, value=1.0, label="Relative Permeability ($\mu_r$)")
    mag_loss_slider = mo.ui.slider(start=0.0, stop=10.0, step=0.1, value=0.0, label="Magnetic Loss ($\sigma_m$) [Ohms/m]")
    
    # Text input for the material name
    name_input = mo.ui.text(value="my_new_material", label="Material Identifier")
    
    # Display the interactive UI elements
    mo.vstack([
        permittivity_slider,
        conductivity_slider,
        permeability_slider,
        mag_loss_slider,
        name_input
    ])
    return (
        conductivity_slider,
        mag_loss_slider,
        name_input,
        permeability_slider,
        permittivity_slider,
    )

@app.cell
def __(
    conductivity_slider,
    mag_loss_slider,
    name_input,
    permeability_slider,
    permittivity_slider,
    mo,
):
    # The reactive hook: This cell automatically updates whenever any of the sliders change
    generated_command = f"#material: {permittivity_slider.value:.2f} {conductivity_slider.value:.2f} {permeability_slider.value:.2f} {mag_loss_slider.value:.2f} {name_input.value}"
    
    # Visualizing as markdown
    mo.md(
        f"""
        ### Live Generated Command:
        
        Copy and paste this directly into your gprMax `.in` file:
        ```text
        {generated_command}
        ```
        """
    )
    return generated_command,

if __name__ == "__main__":
    app.run()
