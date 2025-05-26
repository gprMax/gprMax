import h5py
import numpy as np
def compare_h5_files(file1, file2, tolerance=1e-6):
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
        
        # Check the number of datasets
        if len(datasets1) != len(datasets2):
            print(f"Number of datasets doesn't match: {len(datasets1)} vs {len(datasets2)}")
            return False
        
        # Go through each dataset in the first file and compare with the second
        for name in datasets1:
            if name not in datasets2:
                print(f"Dataset {name} does not exist in the second file")
                return False
            
            d1 = datasets1[name]
            d2 = datasets2[name]
            
            if isinstance(d1, np.ndarray):
                if d1.shape != d2.shape:
                    print(f"Dataset {name} doesn't match the shape: {d1.shape} vs {d2.shape}")
                    return False
                
                if not np.allclose(d1, d2, atol=tolerance):
                    print(f"Dataset {name} doesn't match")
                    print("Max diff:", np.max(np.abs(d1 - d2)))
                    return False
            else:
                if not np.allclose(np.array(d1), np.array(d2), atol=tolerance):
                    print(f"Scaler dataset {name} doesn't match")
                    return False
        
        # 检查是否有额外数据集在第二个文件中
        for name in datasets2:
            if name not in datasets1:
                print(f"Dataset {name} does not exist in the first file")
                return False
        
        return True

# get the command line input file names
import sys
if len(sys.argv) != 3:
    print("Usage: python check_hdf5.py <file1.h5> <file2.h5>")
    sys.exit(1)
file1 = sys.argv[1]
file2 = sys.argv[2]
print(f"comparing {file1} and {file2}")
if compare_h5_files(file1, file2):
    print("Files are the same, correctness check PASSED!")
else:
    print("Files diff, correctness check FAILED!")