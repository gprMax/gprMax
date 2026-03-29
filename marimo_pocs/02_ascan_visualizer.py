import marimo

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt

@app.cell
def __(mo):
    mo.md(
        """
        # gprMax Reactive A-Scan Visualizer POC
        This interactive notebook demonstrates how we can parse output `.out` (HDF5) files and visualize A-Scans directly inside a `marimo` notebook.
        
        *Note: For this POC, if no file is provided, it generates a mock GPR waveform to demonstrate the reactivity of the plotting tools.*
        """
    )
    return

@app.cell
def __(mo):
    # Interactive UI for data analysis
    noise_slider = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.2, label="Simulated Noise Level")
    frequency_slider = mo.ui.slider(start=0.5, stop=5.0, step=0.1, value=1.5, label="Center Frequency (GHz)")
    
    ui_elements = mo.vstack([
        mo.md("### Adjust Parameters to see the Plot React:"),
        frequency_slider,
        noise_slider
    ])
    ui_elements
    return frequency_slider, noise_slider, ui_elements

@app.cell
def __(frequency_slider, mo, noise_slider, np, plt):
    # In a real GSoC implementation, this would be:
    # import h5py
    # f = h5py.File('my_model.out', 'r')
    # time_array = np.array(f['time'][:])
    # ez_field = np.array(f['rxs']['rx1']['Ez'][:])
    
    # We generate a mock A-Scan (Ricker wavelet + noise) for demonstration
    time = np.linspace(0, 10, 500)
    freq = frequency_slider.value
    t0 = 2.0
    tau = time - t0
    
    # Ricker Wavelet Formula
    wavelet = (1 - 2 * (np.pi * freq * tau)**2) * np.exp(-(np.pi * freq * tau)**2)
    noise = np.random.normal(0, noise_slider.value, len(time))
    
    ez_field = wavelet + noise 
    
    # Plotting using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, ez_field, color='blue', linewidth=1.5, label="Ez Field Data")
    ax.set_title("Interactive A-Scan Visualization")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Electric Field Strength (V/m)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    
    # Marimo automatically displays the matplotlib figure!
    return_fig = mo.as_html(fig) if hasattr(mo, 'as_html') else fig
    return_fig
    
    return ax, ez_field, fig, freq, noise, t0, tau, time, wavelet

if __name__ == "__main__":
    app.run()
