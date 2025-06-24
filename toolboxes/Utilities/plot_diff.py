import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_h5_files(file1, file2, tolerance=1e-3):
    def visit_items(name, obj, file_dict):
        """Access the HDF5 file and collect datasets recursively."""
        if isinstance(obj, h5py.Dataset):
            file_dict[name] = obj[()]
    
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        # Collect datasets from both files
        datasets1 = {}
        f1.visititems(lambda name, obj: visit_items(name, obj, datasets1))
        
        datasets2 = {}
        f2.visititems(lambda name, obj: visit_items(name, obj, datasets2))
        # print("datasets1:", datasets1)
        # print("datasets2:", datasets2)
        # Check the number of datasets
        if len(datasets1) != len(datasets2):
            raise ValueError(f"Number of datasets doesn't match: {len(datasets1)} vs {len(datasets2)}")

        # Go through each dataset in the first file and compare with the second
        for name in datasets1:
            if name not in datasets2:
                raise ValueError(f"Dataset {name} does not exist in the second file")

        for name in datasets2:
            if name not in datasets1:
                raise ValueError(f"Dataset {name} does not exist in the first file")
        
        for name in datasets1:
            d1 = datasets1[name]
            d2 = datasets2[name]        
            # Compare the datasets
            if isinstance(d1, np.ndarray):
                if d1.shape != d2.shape:
                    raise ValueError(f"Dataset {name} doesn't match the shape: {d1.shape} vs {d2.shape}")
                diff_values = d1 - d2
                # Create a figure with three subplots
                plt.figure(figsize=(15, 5))

                # Plot d1
                # plt.subplot(131)
                plt.plot(d1)
                plt.title(f'{name} - First Dataset')

                # Plot d2
                # plt.subplot(132)
                plt.plot(d2)
                plt.title(f'{name} - Second Dataset')

                # Plot difference
                # plt.subplot(133)
                plt.plot(diff_values)
                plt.title(f'{name} - Difference')

                plt.tight_layout()
                plt_name = f"{name.replace('/', '_')}_diff.png"
                plt.savefig(plt_name)
        return True

# get the command line input file names
import sys
if len(sys.argv) != 3:
    print("Usage: python plot_diff.py <file1.h5> <file2.h5>")
    sys.exit(1)
file1 = sys.argv[1]
file2 = sys.argv[2]
plot_h5_files(file1, file2)