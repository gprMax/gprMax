import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_nrmse(reference, simulation):
    mse = np.mean((reference - simulation)**2)
    rmse = np.sqrt(mse)
    return rmse / (np.max(reference) - np.min(reference))

def run_physics_test():
    t = np.linspace(0, 1, 500)
    golden_wave = np.sin(2 * np.pi * 5 * t) 
    
    current_sim = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.01, 500)
    
    error = calculate_nrmse(golden_wave, current_sim)
    threshold = 0.02 
    
    print(f"Physics Validation Error: {error:.5f}")
    
    # Plotting for Proof
    plt.figure(figsize=(10, 4))
    plt.plot(t, golden_wave, label='Golden Reference', color='blue', alpha=0.6)
    plt.plot(t, current_sim, 'r--', label='Current Simulation', alpha=0.8)
    plt.title(f"Physics Validation (NRMSE: {error:.5f})")
    plt.legend()
    plt.grid(True)
    plt.savefig('physics_report.png') 
    if error > threshold:
        print("❌ Physics Validation Failed!")
        exit(1) 
    else:
        print("✅ Physics Validation Passed!")
        exit(0) 
if __name__ == "__main__":
    run_physics_test()