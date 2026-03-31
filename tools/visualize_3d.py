import numpy as np
import plotly.graph_objects as go
import argparse
import os

def generate_sample_grid(size=(20, 20, 20)):
    """Generate a sample permittivity grid for demonstration."""
    grid = np.ones(size)
    # Add a sphere in the middle
    z, y, x = np.ogrid[:size[0], :size[1], :size[2]]
    center = (size[0]//2, size[1]//2, size[2]//2)
    dist_from_center = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    grid[dist_from_center <= 5] = 5.0
    return grid

def plot_3d_grid(grid, title="gprMax 3D Geometry Visualization"):
    """Plot a 3D voxel grid using Plotly."""
    z, y, x = np.where(grid > 1.0)
    values = grid[grid > 1.0]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=values,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Permittivity")
        )
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    output_path = "geometry_3d_view.html"
    fig.write_html(output_path)
    print(f"Generated 3D visualization at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize gprMax geometry in 3D.")
    parser.add_argument("--input", type=str, help="Path to gprMax geometry file (not implemented for raw HDF5 yet, using demo mode)")
    args = parser.parse_args()

    print("Initializing 3D Visualizer...")
    # For now, generate a demo grid to show functionality
    grid = generate_sample_grid()
    plot_3d_grid(grid)

if __name__ == "__main__":
    main()
