#!/usr/bin/env python3
"""
Apple Metal GPU Benchmarking Script for gprMax
Measures performance across different domain sizes and compares with CPU.
"""

import os
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt


def create_benchmark_input(size, name):
    """Create a benchmark input file with specified domain size."""
    content = f"""#title: Metal GPU Benchmark - {size}x{size}x{size} domain
#domain: {size*0.001:.3f} {size*0.001:.3f} {size*0.001:.3f}
#dx_dy_dz: 0.001 0.001 0.001
#time_window: 3e-09
#pml_cells: 0

#material: 6 0 1 0 half_space

#waveform: ricker 1 1.5e9 my_ricker
#hertzian_dipole: x {size*0.001/2:.3f} {size*0.001/2:.3f} {size*0.001/2:.3f} my_ricker
#rx: {size*0.001/2:.3f} {size*0.001/2:.3f} {size*0.001/2:.3f}

#box: 0 0 0 {size*0.001:.3f} {size*0.001/2:.3f} {size*0.001:.3f} half_space
"""
    
    with open(f"testing/benchmarking/{name}.in", "w") as f:
        f.write(content)
    return f"testing/benchmarking/{name}.in"


def run_benchmark(input_file, backend_flag="", output_suffix=""):
    """Run a single benchmark and extract performance metrics."""
    cmd = ["python", "-m", "gprMax", input_file]
    if backend_flag:
        cmd.append(backend_flag)
    if output_suffix:
        cmd.extend(["-o", f"benchmark_{output_suffix}"])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract timing information from output
        lines = result.stdout.split('\n')
        
        # Find execution time
        exec_time = None
        iterations = None
        
        for line in lines:
            if "iterations)" in line and "secs" in line:
                # Extract iterations from pattern like "1559 iterations)"
                match = re.search(r'(\d+) iterations\)', line)
                if match:
                    iterations = int(match.group(1))
            
            if "=== Simulation completed in" in line:
                # Extract time from pattern like "=== Simulation completed in 4.6083 seconds ==="
                match = re.search(r'completed in ([\d\.]+) seconds', line)
                if match:
                    exec_time = float(match.group(1))
        
        # Extract domain information from the input file
        with open(input_file, 'r') as f:
            content = f.read()
            
        # Extract domain size from #domain line
        domain_match = re.search(r'#domain: ([\d\.]+) ([\d\.]+) ([\d\.]+)', content)
        if domain_match:
            dx_size = float(domain_match.group(1))
            dy_size = float(domain_match.group(2)) 
            dz_size = float(domain_match.group(3))
            
        # Extract cell size
        cell_match = re.search(r'#dx_dy_dz: ([\d\.]+) ([\d\.]+) ([\d\.]+)', content)
        if cell_match:
            dx = float(cell_match.group(1))
            dy = float(cell_match.group(2))
            dz = float(cell_match.group(3))
            
            # Calculate number of cells
            nx = int(dx_size / dx)
            ny = int(dy_size / dy) 
            nz = int(dz_size / dz)
        
        if exec_time and iterations and nx and ny and nz:
            # Calculate performance using the formula: P = (NX × NY × NZ × NT) / (T × 1×10⁶)
            total_cells = nx * ny * nz * iterations
            performance = total_cells / (exec_time * 1e6)
            
            return {
                'nx': nx, 'ny': ny, 'nz': nz, 'nt': iterations,
                'exec_time': exec_time, 'performance': performance,
                'total_cells': total_cells
            }
        else:
            print(f"Failed to extract metrics from output")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed: {e}")
        print(f"STDERR: {e.stderr}")
        return None


def main():
    """Main benchmarking function."""
    print("🚀 Apple Metal GPU Benchmarking Suite for gprMax")
    print("=" * 60)
    
    # Change to gprMax directory
    os.chdir("/Users/cwarren/Desktop/gprMax")
    
    # Define domain sizes to test (cells per side for cubic domains)
    domain_sizes = [50, 75, 100, 125, 150, 175, 200, 250, 300, 400]  # Start smaller for Apple Silicon
    
    # Storage for results
    metal_results = []
    cpu_results = []
    
    print(f"Testing domain sizes: {domain_sizes} cells per side")
    
    for size in domain_sizes:
        print(f"\n📊 Testing {size}×{size}×{size} domain...")
        
        # Create benchmark input file
        input_file = create_benchmark_input(size, f"benchmark_{size}")
        
        # Test Metal backend
        print(f"🔥 Testing Metal backend...")
        metal_result = run_benchmark(input_file, "-metal", f"metal_{size}")
        if metal_result:
            metal_result['size'] = size
            metal_results.append(metal_result)
            print(f"   Metal: {metal_result['performance']:.1f} Mcells/s")
        
        # Test CPU backend
        print(f"🖥️ Testing CPU backend...")
        cpu_result = run_benchmark(input_file, "", f"cpu_{size}")
        if cpu_result:
            cpu_result['size'] = size
            cpu_results.append(cpu_result)
            print(f"   CPU:   {cpu_result['performance']:.1f} Mcells/s")
            
        # Calculate speedup if both succeeded
        if metal_result and cpu_result:
            speedup = metal_result['performance'] / cpu_result['performance']
            print(f"   Speedup: {speedup:.2f}×")
    
    # Create performance comparison plot
    if metal_results and cpu_results:
        create_performance_plot(metal_results, cpu_results)
        
    # Print summary
    print_summary(metal_results, cpu_results)


def create_performance_plot(metal_results, cpu_results):
    """Create a performance comparison plot."""
    metal_sizes = [r['size'] for r in metal_results]
    metal_perf = [r['performance'] for r in metal_results]
    cpu_sizes = [r['size'] for r in cpu_results]
    cpu_perf = [r['performance'] for r in cpu_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance comparison
    ax1.plot(metal_sizes, metal_perf, 'ro-', label='Apple Metal', linewidth=2, markersize=8)
    ax1.plot(cpu_sizes, cpu_perf, 'bo-', label='CPU (OpenMP)', linewidth=2, markersize=8)
    ax1.set_xlabel('Domain Size [cells per side]')
    ax1.set_ylabel('Performance [Mcells/s]')
    ax1.set_title('gprMax Performance: Metal vs CPU')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Speedup plot
    common_sizes = sorted(set(metal_sizes) & set(cpu_sizes))
    speedups = []
    for size in common_sizes:
        metal_perf_val = next(r['performance'] for r in metal_results if r['size'] == size)
        cpu_perf_val = next(r['performance'] for r in cpu_results if r['size'] == size)
        speedups.append(metal_perf_val / cpu_perf_val)
    
    ax2.plot(common_sizes, speedups, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='No speedup')
    ax2.set_xlabel('Domain Size [cells per side]')
    ax2.set_ylabel('Speedup (Metal/CPU)')
    ax2.set_title('Metal GPU Speedup over CPU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metal_benchmark_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 Performance plot saved as: metal_benchmark_results.png")
    plt.show()


def print_summary(metal_results, cpu_results):
    """Print benchmark summary."""
    print(f"\n{'='*80}")
    print("🎯 BENCHMARK SUMMARY")
    print('='*80)
    print(f"{'Size':<8} {'Metal [Mcells/s]':<15} {'CPU [Mcells/s]':<15} {'Speedup':<10} {'Exec Time (M)':<12} {'Exec Time (C)':<12}")
    print('-'*80)
    
    for metal_r in metal_results:
        size = metal_r['size']
        cpu_r = next((r for r in cpu_results if r['size'] == size), None)
        
        if cpu_r:
            speedup = metal_r['performance'] / cpu_r['performance']
            print(f"{size:<8} {metal_r['performance']:<15.1f} {cpu_r['performance']:<15.1f} "
                  f"{speedup:<10.2f} {metal_r['exec_time']:<12.2f} {cpu_r['exec_time']:<12.2f}")
        else:
            print(f"{size:<8} {metal_r['performance']:<15.1f} {'N/A':<15} {'N/A':<10} "
                  f"{metal_r['exec_time']:<12.2f} {'N/A':<12}")
    
    if metal_results and cpu_results:
        avg_speedup = np.mean([metal_r['performance'] / next(cpu_r['performance'] for cpu_r in cpu_results if cpu_r['size'] == metal_r['size']) 
                              for metal_r in metal_results if any(cpu_r['size'] == metal_r['size'] for cpu_r in cpu_results)])
        print(f"\n🚀 Average Speedup: {avg_speedup:.2f}×")
        
        max_metal_perf = max(r['performance'] for r in metal_results)
        max_cpu_perf = max(r['performance'] for r in cpu_results)
        print(f"🔥 Peak Metal Performance: {max_metal_perf:.1f} Mcells/s")
        print(f"🖥️  Peak CPU Performance: {max_cpu_perf:.1f} Mcells/s")


if __name__ == "__main__":
    main()
