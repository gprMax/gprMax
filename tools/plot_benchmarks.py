import pandas as pd
import plotly.express as px
import os

if __name__ == "__main__":
    if not os.path.exists("benchmark_results.csv"):
        print("No benchmark results found.")
        exit(1)

    df = pd.DataFrame(pd.read_csv("benchmark_results.csv"))

    # 1. Bar Chart: Runtime by Model
    fig1 = px.bar(
        df,
        x="model",
        y="runtime_s",
        title="Simulation Runtime by Model",
        color="success",
    )

    # 2. Scatter Plot: Memory vs Runtime
    fig2 = px.scatter(
        df,
        x="runtime_s",
        y="memory_mb",
        size="grid_size",
        color="model",
        title="Memory vs Runtime",
    )

    with open("benchmark_plots.html", "w") as f:
        f.write("<html><head><title>Benchmark Results</title></head><body>")
        f.write("<h1>gprMax Environmental Benchmarks</h1>")
        f.write(fig1.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write(fig2.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write("</body></html>")

    # Write summary for PR comment
    with open("summary.md", "w") as f:
        f.write("## 🚀 Benchmark Results\n")
        f.write(df.to_markdown(index=False))
        f.write(
            "\n\n*Interactive plots have been generated and uploaded as artifacts!*"
        )

    print("Interactive plots and markdown summary saved.")
