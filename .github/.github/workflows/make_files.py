import numpy as np
data = np.sin(np.linspace(0, 10, 100)) 
np.save('golden_reference.npy', data)
np.save('current_simulation.npy', data)
print("Files Created Successfully!")