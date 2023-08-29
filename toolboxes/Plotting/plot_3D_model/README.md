## 3D Visualization Toolbox

This folder contains the `plot_3D_model.py` file that uses the PyVista library to show the 3D geometry of the model.

To use this in a Jupyter cell, follow these steps:

1. Install the PyVista library by running the following command in a code cell:
   ```python
   !pip install pyvista -qq
   ```
2. Keep the plot_3D_model.py file in the same directory as your Jupyter Notebook.

3. In your Jupyter Notebook, import the function from the plot_3D_model.py file and use it to visualize the 3D geometry of the      model.

## Example Code 

```python

    from plot_3D_model import visualize_3D_model

    # Your other code

    visualize_3D_model(arguments)  # Call the function to visualize the 3D model


```
