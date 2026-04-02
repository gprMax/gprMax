import os
import time
import psutil
import subprocess


def run_simulation(model_file):
    start_time = time.time()
    # Execute the gprMax module with CPU backend
    process = subprocess.Popen(
        ["python", "-m", "gprMax", model_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    max_memory = 0
    while process.poll() is None:
        try:
            mem = psutil.Process(process.pid).memory_info().rss / (1024 * 1024)
            if mem > max_memory:
                max_memory = mem
        except psutil.NoSuchProcess:
            break
        time.sleep(0.5)

    end_time = time.time()
    runtime = end_time - start_time

    # Check if simulation succeeded
    _, stderr = process.communicate()
    success = process.returncode == 0
    return runtime, max_memory, success


if __name__ == "__main__":
    models = [
        "examples/environmental/soil_moisture.in",
        "examples/environmental/contamination_plume.in",
    ]
    results = []

    # Create mock models if they don't exist yet so it runs for CI benchmarking without error
    os.makedirs("examples/environmental", exist_ok=True)
    for m in models:
        if not os.path.exists(m):
            with open(m, "w") as f:
                f.write(
                    "#domain: 1.0 1.0 1.0\n#dx_dy_dz: 0.1 0.1 0.1\n#time_window: 10e-9\n"
                )

    for model in models:
        print(f"Running benchmark: {model}")
        rt, mem, success = run_simulation(model)
        grid_size = 1000  # Sample parameter representation
        water_content = 0.1  # Sample parameter representation
        results.append(
            {
                "model": model,
                "runtime_s": rt,
                "memory_mb": mem,
                "success": success,
                "grid_size": grid_size,
                "water_content": water_content,
            }
        )

    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("Benchmarks completed and saved to benchmark_results.csv")
