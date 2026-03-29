import numpy as np
import plotly.graph_objects as go


def analyze_water_content():
    print("Running water content sensitivity analysis...")

    # Mock data for parameter sweep
    water_contents = np.linspace(0.01, 0.4, 40)
    depths = np.linspace(0.1, 5.0, 50)

    W, D = np.meshgrid(water_contents, depths)
    # Mock amplitude decreasing with depth and water content (attenuation)
    amplitude = np.exp(-D * W * 10) * np.sin(2 * np.pi * D / (W + 0.1))

    # Mock time of flight increasing with water content (slower velocity)
    time_of_flight = D / (0.1 / np.sqrt(1 + W * 80))

    # 2D Heatmap: Time-of-flight vs Water Content and Depth
    fig1 = go.Figure(
        data=go.Heatmap(
            z=time_of_flight, x=water_contents, y=depths, colorscale="Viridis"
        )
    )
    fig1.update_layout(
        title="Time-of-Flight vs Water Content & Depth",
        xaxis_title="Water Content",
        yaxis_title="Depth (m)",
    )

    # 3D Surface Plot: Amplitude
    fig2 = go.Figure(data=[go.Surface(z=amplitude, x=water_contents, y=depths)])
    fig2.update_layout(
        title="GPR Response Amplitude",
        scene=dict(
            xaxis_title="Water Content",
            yaxis_title="Depth (m)",
            zaxis_title="Amplitude",
        ),
    )

    fig1.write_html("water_content_heatmap.html")
    fig2.write_html("water_content_surface.html")
    print("Saved water_content_heatmap.html and water_content_surface.html")


if __name__ == "__main__":
    analyze_water_content()
