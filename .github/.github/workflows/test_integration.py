import numpy as np
import sys

def run_integration_check(reference_file, current_file, threshold=0.01):
    """
    Automated Integration Test for GPR Physics Validation.
    Calculates NRMSE between simulation output and theoretical reference.
    """
    try:
        # Data loading logic
        ref_data = np.load(reference_file)
        cur_data = np.load(current_file)

        # NRMSE Calculation logic
        rmse = np.sqrt(np.mean((ref_data - cur_data)**2))
        data_range = np.max(ref_data) - np.min(ref_data)
        
        if data_range == 0:
            nrmse = 0.0
        else:
            nrmse = rmse / data_range

        print(f"--- GPR Integration Report ---")
        print(f"NRMSE Calculated: {nrmse:.6f}")
        print(f"Error Threshold: {threshold}")

        if nrmse <= threshold:
            print("Status: ✅ PASS - Physics Integrity Maintained")
            return True
        else:
            print("Status: ❌ FAIL - Regression Detected in Simulation")
            return False

    except FileNotFoundError as e:
        print(f"Error: Required simulation files not found. {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    REFERENCE = "golden_reference.npy"
    CURRENT = "current_simulation.npy"
    success = run_integration_check(REFERENCE, CURRENT)
    
    if not success:
        sys.exit(1)
    else:
        sys.exit(0)